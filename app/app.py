# -*- coding: utf-8 -*-
"""
IQC 智能来料检测系统 - Streamlit 界面
两个端：
  检验员端：拍照/上传 → 自动检测 → 判定入库 → 人工复核
  主管端：供应商/批次/缺陷统计报表，追溯查询，报告导出

运行：cd iqc_vision && streamlit run app/app.py
"""
import sys
from datetime import date
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import quality_db as db
from src import report as rp
from src.detection import pipeline
from src.config import SPEC_CLASSES

st.set_page_config(page_title="IQC 智能来料检测系统", page_icon="🔍", layout="wide")


# ================= 初始化 =================

def init():
    db.init_db()
    if "batch_info" not in st.session_state:
        st.session_state.batch_info = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def load_models_ui():
    if "models_loaded" not in st.session_state:
        with st.spinner("正在加载 AI 模型（首次约需数十秒）..."):
            pipeline.load_models()
            st.session_state.models_loaded = True


# ================= 检验员端 =================

def inspector_page():
    st.header("检验员端：智能来料检测")
    load_models_ui()

    c1, c2 = st.columns([1, 1.4])
    with c1:
        # ---- 来料信息 ----
        st.subheader("① 来料信息")
        with st.form("batch_form"):
            batch_id = st.text_input("批次号 *", placeholder="如 IQC20260807-01")
            material_no = st.text_input("料号 *", placeholder="如 M4-HEX-SCREW")
            material_name = st.text_input("名称", placeholder="内六角螺钉")
            spec_expected = st.selectbox("期望规格（图纸/料单）", ["", *SPEC_CLASSES],
                                         help="用于规格防错比对")
            supplier = st.text_input("供应商")
            quantity = st.number_input("批次数量", min_value=0, value=0)
            inspector = st.text_input("检验员")
            submitted = st.form_submit_button("登记批次")

        if submitted and batch_id and material_no:
            if not db.batch_exists(batch_id):
                db.create_batch(batch_id, material_no, material_name, spec_expected or None,
                                supplier=supplier, quantity=int(quantity))
            st.session_state.batch_info = {
                "batch_id": batch_id, "material_no": material_no,
                "spec_expected": spec_expected, "inspector": inspector,
            }
            st.success(f"批次 {batch_id} 已登记")

        # ---- 图像采集 ----
        st.subheader("② 图像采集")
        src_mode = st.radio("图像来源", ["上传图片", "本地摄像头"], horizontal=True)
        uploaded = None
        if src_mode == "上传图片":
            uploaded = st.file_uploader("选择零件照片", type=["jpg", "png", "jpeg", "bmp"])
        else:
            cam = st.camera_input("对准零件拍照", key="cam1")

        img_bgr = None
        if uploaded is not None:
            img_bgr = cv2.imdecode(np_from(uploaded.read()), cv2.IMREAD_COLOR)
        elif "cam1" in st.session_state and st.session_state.cam1 is not None:
            img_bgr = cv2.imdecode(np_from(st.session_state.cam1.getvalue()), cv2.IMREAD_COLOR)

        if img_bgr is not None:
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="待检图像", width=320)
            if st.button("🔍 开始检测", type="primary", use_container_width=True):
                if not st.session_state.get("batch_info"):
                    st.warning("请先登记批次")
                else:
                    with st.spinner("AI 检测中..."):
                        res = pipeline.run_detection(
                            img_bgr, inspector=inspector,
                            expected_spec=st.session_state.batch_info.get("spec_expected") or None)
                        st.session_state.last_result = res
                        st.session_state.last_batch = st.session_state.batch_info["batch_id"]

    with c2:
        # ---- 检测结果 ----
        st.subheader("③ 检测结果与判定")
        res = st.session_state.get("last_result")
        if res is None:
            st.info("完成图像采集并点击「开始检测」后，结果将显示在这里。")
            return

        spec = res["spec_result"]
        conf = res["spec_confidence"]
        m1, m2, m3 = st.columns(3)
        method = "尺寸反推" if res.get("spec_method") == "dimension" else "CNN" if res.get("spec_method") == "cnn" else ""
        m1.metric("AI 规格识别", spec or "未识别",
                  f"{method} · 置信度 {conf:.2f}" if conf else method or "")
        m2.metric("缺陷检出", _fmt_defect(res["defect_summary"]) or "无")
        vc = "✅ 通过" if res["ai_verdict"] == "OK" else "🚨 异常"
        m3.metric("AI 初判", vc)

        dim = res.get("dimension")
        if dim:
            if dim.get("outer_diam_mm") is not None:
                st.markdown(f"**尺寸测量**：{dim.get('part_type', '?')} "
                            f"外径 {dim['outer_diam_mm']:.2f}mm"
                            + (f"，内径 {dim['inner_diam_mm']:.2f}mm" if dim.get("inner_diam_mm") else ""))
            if dim.get("reason"):
                if dim.get("status") == "NG":
                    st.error(dim["reason"])
                else:
                    st.caption(dim["reason"])

        if res["reasons"]:
            st.warning("；".join(res["reasons"]))
        if res.get("annotated") is not None:
            st.image(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB),
                     caption="缺陷标注结果", width=480)

        # ---- 入库 + 复核 ----
        st.markdown("---")
        if st.button("📥 记录入库", type="primary"):
            bi = st.session_state.batch_info
            rec_id, verdict = pipeline.save_record(
                bi["batch_id"], res, inspector=bi["inspector"],
                expected_spec=bi["spec_expected"] or None)
            st.success(f"已入库：记录#{rec_id}，AI判定 {verdict}")

        with st.form("review_form"):
            st.markdown("**人工复核**（AI 初筛 + 工程师决策）")
            rev_rec = st.number_input("记录ID", min_value=0, step=1)
            verdict_opts = ["", "PASS", "REJECT"]
            rev_v = st.selectbox("复核结论", verdict_opts)
            rev_note = st.text_area("备注", placeholder="如：确认NG，划伤超标退回供应商")
            rev_by = st.text_input("复核人")
            if st.form_submit_button("提交复核") and rev_rec and rev_v:
                db.review_record(int(rev_rec), rev_v, rev_note, rev_by)
                st.success(f"记录#{rev_rec} 已复核为 {rev_v}")


def np_from(data):
    import numpy as np
    return np.frombuffer(data, dtype=np.uint8)


def _fmt_defect(summary):
    if not summary:
        return ""
    return ", ".join(f"{k}×{v}" for k, v in summary.items())


# ================= 主管端 =================

def supervisor_page():
    st.header("主管端：质量统计与追溯")
    tab1, tab2, tab3 = st.tabs(["质量总览", "批次/供应商报表", "追溯查询"])

    # ---- 总览 ----
    with tab1:
        c1, c2, c3 = st.columns(3)
        vstats = db.verdict_stats()
        c1.metric("检验记录总数", sum(vstats.values()))
        c2.metric("AI 判定 OK", vstats.get("OK", 0))
        c3.metric("AI 判定 NG", vstats.get("NG", 0))
        st.markdown("**AI 判定分布**")
        st.bar_chart(pd.Series(vstats))

        st.markdown("**缺陷类型分布**")
        ddf = rp.defect_report()
        if not ddf.empty:
            st.bar_chart(ddf.set_index("缺陷类型"))
            st.dataframe(ddf, use_container_width=True)
        else:
            st.info("暂无缺陷记录")

    # ---- 批次/供应商报表 ----
    with tab2:
        st.subheader("供应商质量统计")
        sdf = rp.supplier_report()
        if not sdf.empty:
            st.dataframe(sdf, use_container_width=True)
            st.bar_chart(sdf.set_index("supplier")[["total", "ng_count"]])
        else:
            st.info("暂无供应商数据")

        st.subheader("批次报表")
        bdf = rp.batch_report()
        if not bdf.empty:
            st.dataframe(bdf, use_container_width=True)
            fmt = st.radio("导出格式", ["csv", "xlsx"], horizontal=True)
            if st.button("导出报表"):
                path = rp.export_report(bdf, fmt=fmt)
                st.success(f"已导出: {path}")

    # ---- 追溯查询 ----
    with tab3:
        st.subheader("追溯查询")
        q = st.text_input("按批次号/料号/供应商模糊搜索")
        verdict = st.selectbox("AI判定", ["", "OK", "NG", "UNSURE"])
        date_from = st.date_input("开始日期", value=None)
        date_to = st.date_input("结束日期", value=None)
        # 单一关键词同时匹配批次号/料号/供应商，客户端侧过滤
        if st.button("查询", type="primary"):
            tdf = rp.trace_query(
                batch_id=q or None, verdict=verdict or None,
                date_from=str(date_from) if date_from else None,
                date_to=str(date_to) if date_to else None)
            if q and not tdf.empty:
                tdf = tdf[
                    tdf["批次号"].astype(str).str.contains(q, case=False, na=False)
                    | tdf["料号"].astype(str).str.contains(q, case=False, na=False)
                    | tdf["供应商"].astype(str).str.contains(q, case=False, na=False)
                ].reset_index(drop=True)
            if tdf.empty:
                st.info("未找到匹配记录")
            else:
                st.dataframe(tdf, use_container_width=True)
                if st.button("导出查询结果"):
                    path = rp.export_report(tdf, fmt="csv")
                    st.success(f"已导出: {path}")


# ================= 主入口 =================

def main():
    init()
    st.sidebar.title("🔍 IQC 智能来料检测")
    st.sidebar.markdown("光学成像 × AI × 质量工程")
    page = st.sidebar.radio("功能端", ["检验员端", "主管端"])
    st.sidebar.markdown("---")
    if st.sidebar.button("初始化数据库"):
        db.init_db()
        st.sidebar.success("数据库已初始化")
    if st.sidebar.button("生成模拟规格数据"):
        from src.data import make_standard_parts as msp
        msp.generate(msp.STD_PARTS_DIR)
        st.sidebar.success("模拟数据已生成")

    if page == "检验员端":
        inspector_page()
    else:
        supervisor_page()


if __name__ == "__main__":
    main()
