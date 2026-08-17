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
from src import inspection_checklist as cl
from src.detection import vision_detector
from src.detection import engine
from src.config import SPEC_CLASSES

st.set_page_config(page_title="IQC 智能来料检测系统", page_icon="🔍", layout="wide")

# ================= 全局样式 =================
GLOBAL_CSS = """
<style>
/* 卡片容器 */
.sec-card {
    background: #ffffff;
    border: 1px solid #e6e9ef;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.sec-card h3 {
    margin: 0 0 4px 0;
    font-size: 15px;
    font-weight: 600;
    color: #1f2329;
}
.sec-card .sec-desc {
    margin: 0 0 12px 0;
    font-size: 12px;
    color: #8a919f;
}
.sec-card .sec-body {
    margin-top: 8px;
}
/* 步骤编号 */
.step-badge {
    display: inline-block;
    width: 22px; height: 22px; line-height: 22px;
    background: #2563eb; color: #fff;
    border-radius: 50%; text-align: center;
    font-size: 12px; font-weight: 700;
    margin-right: 8px; vertical-align: middle;
}
/* 判定结果横幅 */
.verdict-banner {
    padding: 14px 18px;
    border-radius: 10px;
    font-size: 16px; font-weight: 600;
    margin-bottom: 12px;
}
.verdict-OK  { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
.verdict-NG  { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.verdict-UNSURE, .verdict-SKIP { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
/* 指标卡片更紧凑 */
div[data-testid="stMetric"] {
    background: #f8fafc;
    border: 1px solid #eef2f6;
    border-radius: 10px;
    padding: 10px 12px;
}
/* 表格样式 */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
/* 标签页更清爽 */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 16px;
}
/* 按钮统一圆角 */
.stButton button, .stFormSubmitButton button {
    border-radius: 8px;
}
</style>
"""


def sec_card(title, desc=None):
    """卡片容器：返回 with 块用的 markdown 头 + 说明。"""
    desc_html = f'<p class="sec-desc">{desc}</p>' if desc else ""
    st.markdown(
        f'<div class="sec-card"><h3>{title}</h3>{desc_html}<div class="sec-body">',
        unsafe_allow_html=True)
    return st.container()


def end_card():
    st.markdown("</div></div>", unsafe_allow_html=True)


def step_badge(n):
    return f'<span class="step-badge">{n}</span>'


def verdict_banner(verdict, text=""):
    v = str(verdict).upper()
    cls = "verdict-OK" if v == "OK" else "verdict-NG" if v == "NG" else "verdict-UNSURE"
    label = {"OK": "合格通过", "NG": "异常待复核", "UNSURE": "无法判定", "SKIP": "已跳过"}.get(v, v)
    icon = {"OK": "✅", "NG": "🚨", "UNSURE": "⚠️", "SKIP": "⏭️"}.get(v, "ℹ️")
    if text:
        st.markdown(
            f'<div class="verdict-banner {cls}">{icon} <b>{label}</b>：{text}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-banner {cls}">{icon} 判定：{label}</div>',
                    unsafe_allow_html=True)


# ================= 初始化 =================

def init():
    db.init_db()
    if "batch_info" not in st.session_state:
        st.session_state.batch_info = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def load_models_ui():
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = True
        st.caption("✅ 检测引擎就绪：规格识别 + 缺陷检测 + 尺寸判定（标准件默认本地免费）")


# ================= 检验员端 =================

def inspector_page():
    st.header("检验员端：智能来料检测")
    load_models_ui()

    c1, c2 = st.columns([1, 1.4])
    with c1:
        # ---- ① 来料信息 ----
        with sec_card(f"{step_badge(1)} 来料信息", "登记批次与物料，AI 将按料号自动匹配检验配置"):
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
        end_card()

        if submitted and batch_id and material_no:
            if not db.batch_exists(batch_id):
                db.create_batch(batch_id, material_no, material_name, spec_expected or None,
                                supplier=supplier, quantity=int(quantity))
            # 读取该料号的物料白名单配置（无则 None = 未启用 AI）
            mat_cfg = db.get_material(material_no)
            st.session_state.batch_info = {
                "batch_id": batch_id, "material_no": material_no,
                "spec_expected": spec_expected, "inspector": inspector,
                "material_config": mat_cfg,
            }
            if mat_cfg:
                if not mat_cfg["ai_enabled"]:
                    st.info(f"物料 {material_no} 未启用 AI 检测，将按普通人工记录处理")
                else:
                    items = "、".join(n for on, n in
                                      [(mat_cfg["check_spec"], "规格"), (mat_cfg["check_dim"], "尺寸"),
                                       (mat_cfg["check_defect"], "缺陷")] if on)
                    st.success(f"批次 {batch_id} 已登记，该物料已启用 AI 检测项：{items}")
            else:
                st.warning(f"批次 {batch_id} 已登记，但料号 {material_no} 不在 AI 检测白名单中，"
                           "将跳过 AI 判定（主管端「物料管理」可添加）")
            st.success(f"批次 {batch_id} 已登记")

        # ---- ② 图像采集 ----
        with sec_card(f"{step_badge(2)} 图像采集", "选择检测模式，上传或拍摄零件照片"):
            detect_mode = st.radio("检测模式",
                                   ["标准件（卡尺读数定规格）", "加工件（多角度表面缺陷）",
                                    "螺纹（侧视图）"],
                                   horizontal=True, key="detect_mode")
            is_thread = detect_mode.startswith("螺纹")
        # 标准件/螺纹模式：检测引擎选择（默认混合引擎推荐）
        detect_engine = "混合引擎"
        if not detect_mode.startswith("加工件"):
            detect_engine = st.radio("检测引擎",
                                     ["混合引擎（推荐）", "本地免费", "视觉大模型（付费）"],
                                     horizontal=True, key="detect_engine")
            if detect_engine == "混合引擎":
                st.caption("✅ 混合：规格/尺寸/螺纹本地免费，表面缺陷用视觉大模型（约0.02元/件，准）")
            elif detect_engine == "本地免费":
                st.caption("✅ 本地引擎：RTX 4060 全部本地，零成本（缺陷检测误检偏高，需复核）")
            else:
                st.caption("⚠️ 视觉大模型：Qwen-VL-plus，约0.02元/件，读卡尺LCD更稳")

        # 加工件：关联图纸清单（优先按料号自动匹配检验标准，也可手动选）
        drawing_no = None
        auto_cl = None
        if detect_mode.startswith("加工件"):
            bi = st.session_state.get("batch_info")
            # 按已登记料号自动匹配
            if bi and bi.get("material_no"):
                auto_cl = cl.find_checklist(material_no=bi["material_no"])
            cl_lists = cl.list_checklists()
            if cl_lists:
                opts = {f"{c['drawing_no']} · {c['part_name'] or ''}": c["drawing_no"]
                        for c in cl_lists}
                default_sel = "不关联图纸"
                if auto_cl:
                    default_sel = f"{auto_cl['drawing_no']} · {auto_cl.get('part_name') or ''}"
                sel = st.selectbox("关联图纸（按料号自动匹配，可手动改）",
                                   ["不关联图纸", *opts.keys()],
                                   index=["不关联图纸", *opts.keys()].index(default_sel)
                                   if default_sel in ["不关联图纸", *opts.keys()] else 0)
                drawing_no = opts.get(sel) if sel != "不关联图纸" else None
                if auto_cl and sel == default_sel:
                    st.caption(f"✅ 已按料号 {bi['material_no']} 自动匹配检验标准 "
                               f"{auto_cl['drawing_no']}（{auto_cl.get('part_name') or ''}）")

        src_mode = st.radio("图像来源", ["上传图片", "本地摄像头"], horizontal=True)
        uploaded = None
        if src_mode == "上传图片":
            if detect_mode.startswith("加工件"):
                uploaded = st.file_uploader("选择零件照片（可多选：正面/侧面/孔位）",
                                            type=["jpg", "png", "jpeg", "bmp"],
                                            accept_multiple_files=True)
            else:
                uploaded = st.file_uploader("选择零件照片", type=["jpg", "png", "jpeg", "bmp"])
        else:
            cam = st.camera_input("对准零件拍照", key="cam1")

        # 收集待检图（内存 BGR）
        if src_mode == "上传图片" and uploaded:
            if isinstance(uploaded, list):
                imgs_bgr = [cv2.imdecode(np_from(u.read()), cv2.IMREAD_COLOR) for u in uploaded]
                imgs_bgr = [i for i in imgs_bgr if i is not None]
            else:
                imgs_bgr = [cv2.imdecode(np_from(uploaded.read()), cv2.IMREAD_COLOR)]
        elif src_mode == "本地摄像头" and "cam1" in st.session_state and st.session_state.cam1 is not None:
            imgs_bgr = [cv2.imdecode(np_from(st.session_state.cam1.getvalue()), cv2.IMREAD_COLOR)]
        else:
            imgs_bgr = []

        if imgs_bgr:
            for k, im in enumerate(imgs_bgr):
                st.image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                         caption=f"待检图像 {k+1}" if len(imgs_bgr) > 1 else "待检图像", width=320)

            # 拍照规范提示 + 图像质检
            from src.img_quality import check_image_quality, check_part_size_ratio
            with st.expander("📷 拍照规范（建议）", expanded=False):
                st.markdown("**拍摄提示**：零件水平摆放占画面 60%~80%；卡尺与零件同框、读数清晰；"
                            "光照均匀避免强反光；对焦清晰后稳定拍摄。")
            for k, im in enumerate(imgs_bgr):
                issues, _tips = check_image_quality(im)
                if issues:
                    st.warning(f"图{k+1} 质量问题：{'；'.join(issues)}")
                ok, ratio, size_issue = check_part_size_ratio(im)
                if not ok and size_issue:
                    st.warning(f"图{k+1}：{size_issue}")

            # 加工件+清单模式：尺寸项实测值录入（机器量化判定用）
            measured_vals = {}
            if detect_mode.startswith("加工件") and drawing_no:
                cl_current = cl.load_checklist(drawing_no)
                dim_items = [it for it in (cl_current.get("items") or [])
                             if it.get("type") in ("dimension", "geometric")]
                if dim_items:
                    st.markdown("**尺寸项实测值录入**（卡尺读数，机器按公差自动判定）")
                    for it in dim_items:
                        val = st.number_input(
                            f"{it['label']}（{it['location'] or '位置'}"
                            + (f"，名义 {it['tolerance']['nominal_mm']}mm" if it.get("tolerance", {}).get("nominal_mm") else "")
                            + "）mm", min_value=0.0, value=0.0, step=0.01,
                            key=f"measured_{it['id']}")
                        if val > 0:
                            measured_vals[it["id"]] = float(val)

            if st.button("🔍 开始检测", type="primary", use_container_width=True):
                if not st.session_state.get("batch_info"):
                    st.warning("请先登记批次")
                else:
                    bi = st.session_state.batch_info
                    mat_cfg = bi.get("material_config")
                    if mat_cfg and not mat_cfg["ai_enabled"]:
                        st.warning(f"物料 {bi['material_no']} 未启用 AI 检测，跳过 AI 判定")
                    else:
                        with st.spinner("AI 检测中..."):
                            if detect_mode.startswith("加工件"):
                                if drawing_no:
                                    res = vision_detector.run_checklist_detection(
                                        [(im, None) for im in imgs_bgr], drawing_no=drawing_no,
                                        measured=measured_vals or None)
                                else:
                                    res = vision_detector.run_machined_detection(
                                        [(im, None) for im in imgs_bgr])
                            elif is_thread:
                                # 螺纹：侧视图走本地螺纹分类模型（免费）
                                res = engine.run_detection_local(
                                    imgs_bgr[0],
                                    expected_spec=bi.get("spec_expected") or None)
                            else:
                                # 标准件：默认混合引擎，可选本地免费 / 视觉大模型
                                if detect_engine == "本地免费":
                                    res = engine.run_detection_local(
                                        imgs_bgr[0],
                                        expected_spec=bi.get("spec_expected") or None)
                                elif detect_engine == "视觉大模型（付费）":
                                    res = engine.run_detection_vision(
                                        imgs_bgr[0],
                                        expected_spec=bi.get("spec_expected") or None)
                                else:  # 混合引擎（默认）
                                    res = engine.run_detection_hybrid(
                                        imgs_bgr[0],
                                        expected_spec=bi.get("spec_expected") or None)
                            st.session_state.last_result = res
                            st.session_state.last_batch = bi["batch_id"]
                            st.session_state.last_mode = detect_mode
        end_card()

    with c2:
        # ---- ③ 检测结果与判定 ----
        res = st.session_state.get("last_result")
        if res is None:
            with sec_card(f"{step_badge(3)} 检测结果与判定", "完成图像采集并点击「开始检测」后显示"):
                st.info("检测结果将显示在这里：AI 规格识别、缺陷检出、综合判定与复核。")
            end_card()
            return

        with sec_card(f"{step_badge(3)} 检测结果与判定", f"检验批次：{st.session_state.get('last_batch', '-')}"):
            verdict_banner(res["ai_verdict"], "；".join(res["reasons"]) or res["ai_verdict"])

            is_machined = res.get("spec_method") == "vision_machined"
            is_checklist = res.get("spec_method") == "vision_checklist"
            spec = res["spec_result"]
            conf = res["spec_confidence"]
            m1, m2, m3 = st.columns(3)
            if is_checklist:
                m1.metric("校验对象", res.get("checklist_name") or "加工件",
                          f"图纸 {res.get('checklist_no', '')}")
            elif is_machined:
                m1.metric("检测对象", "加工件", "多角度视觉")
            else:
                method = ("卡尺读数" if res.get("spec_method") == "vision_reading"
                          else "尺寸反推" if res.get("spec_method") == "dimension"
                          else "CNN" if res.get("spec_method") == "cnn" else "")
                m1.metric("规格识别", spec or "未识别",
                          f"{method}·{conf:.0%}" if conf else method)
            m2.metric("缺陷检出", _fmt_defect(res["defect_summary"]) or "无",
                      "异常" if res["defect_summary"] else "干净")
            m3.metric("AI 初判",
                      {"OK": "通过", "NG": "异常", "UNSURE": "不确定", "SKIP": "跳过"}.get(
                          res["ai_verdict"], res["ai_verdict"]),
                      res.get("engine") or "")
        end_card()

        # 加工件：逐张角度展示
        if is_machined and res.get("per_image"):
            st.markdown("**各角度检测明细**")
            rows = []
            for p in res["per_image"]:
                tag = p.get("image") or "内存图"
                if p.get("defects"):
                    rows.append({"角度": tag, "结果": "异常", "说明": _fmt_defect(p["defects"])})
                elif p.get("warn"):
                    rows.append({"角度": tag, "结果": "不确定", "说明": p["warn"]})
                else:
                    rows.append({"角度": tag, "结果": "通过", "说明": ""})
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # 图纸清单：逐项校验明细
        if is_checklist and res.get("per_item"):
            st.markdown(f"**图纸清单逐项校验**（{res.get('checklist_no', '')}）")
            rows = []
            for it in res["per_item"]:
                loc = it.get("location") or ""
                status_txt = {"OK": "✅ 合格", "NG": "🚨 不合格", "UNSURE": "⚠️ 未判定"}.get(
                    it["status"], it["status"])
                if it["status"] == "UNSURE" and not it.get("visual"):
                    status_txt = "🔬 需仪器"
                rows.append({
                    "检查项": it["label"],
                    "检验位置": loc,
                    "结果": status_txt,
                    "说明": it["reason"] or "",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        dim = res.get("dimension")
        if dim:
            if dim.get("reading_mm") is not None:
                status = dim.get("status", "")
                icon = {"OK": "✅", "NG": "🚨", "UNSURE": "⚠️"}.get(status, "")
                st.markdown(f"**卡尺读数** {icon} **{dim['reading_mm']:.2f}mm**"
                            f"（名义 {dim.get('nominal_mm')}mm ±{dim.get('tolerance_mm')}mm"
                            + (f"，偏差 {dim['dist_mm']:.2f}mm" if dim.get("dist_mm") is not None else "")
                            + "）")
            elif dim.get("outer_diam_mm") is not None:
                st.markdown(f"**尺寸测量**：{dim.get('part_type', '?')} "
                            f"外径 **{dim['outer_diam_mm']:.2f}mm**"
                            + (f"，内径 {dim['inner_diam_mm']:.2f}mm" if dim.get("inner_diam_mm") else ""))
            if dim.get("reason"):
                if dim.get("status") == "NG":
                    st.error(dim["reason"])
                else:
                    st.caption(dim["reason"])
            # 读数自动入库提示：攒成训练数据
            if dim.get("reading_mm") is not None:
                st.caption("ℹ️ 卡尺读数已自动记录，入库后累计为训练数据（见主管端「数据积累」），"
                           "攒够 300 对可训练本地读数模型 → 免视觉大模型")

        # 螺纹状态检测结果
        thread_state = res.get("thread_result")
        if thread_state:
            state_txt = {"good": "正常", "missing": "缺牙", "broken": "烂牙/断裂"}.get(
                thread_state, thread_state)
            icon = "✅" if thread_state == "good" else "🚨"
            st.markdown(f"**螺纹状态** {icon} **{state_txt}**"
                        + (f"（置信度 {res['thread_confidence']:.0%}）"
                           if res.get("thread_confidence") else ""))

        if res["reasons"]:
            st.warning("；".join(res["reasons"]))
        if res.get("annotated") is not None:
            st.image(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB),
                     caption="缺陷标注结果", width=480)

        # ---- 入库 + 复核 ----
        st.markdown("---")
        st.markdown("**确认与入库**")
        st.caption("AI 初筛结果需工程师复核确认后入库，形成可追溯记录。")
        cli_a, cli_b = st.columns([1, 1])
        with cli_a:
            if st.button("📥 记录入库", type="primary", use_container_width=True):
                bi = st.session_state.batch_info
                exp = None if (is_machined or is_checklist) else (bi["spec_expected"] or None)
                rec_id, verdict = vision_detector.save_record(
                    bi["batch_id"], res, inspector=bi["inspector"], expected_spec=exp)
                st.session_state.last_record_id = rec_id
                st.session_state.last_verdict = verdict
                st.success(f"已入库：记录#{rec_id}，AI判定 {verdict}")

            # 打印检测单（入库后可打印）
            if st.session_state.get("last_record_id") and st.button(
                    "🖨️ 打印检测单", use_container_width=True):
                from src.print_slip import build_slip_html
                bi = st.session_state.batch_info
                slip = build_slip_html(bi, res, st.session_state.last_record_id,
                                       st.session_state.last_verdict)
                import base64
                st.markdown(
                    f'<a href="data:text/html;base64,{base64.b64encode(slip.encode("utf-8")).decode()}" '
                    f'target="_blank" style="display:none">打开</a>', unsafe_allow_html=True)
                # 新标签页打开 HTML，浏览器打印
                import webbrowser
                tmp = Path(__file__).resolve().parent / "_slip_preview.html"
                tmp.write_text(slip, encoding="utf-8")
                webbrowser.open(f"file:///{tmp}")
                st.info("检测单已生成（新标签页），浏览器 Ctrl+P 打印即可。免手写。")
        with cli_b:
            st.caption("入库后，可在右侧「人工复核」对记录ID做 PASS/REJECT 终审。")

        with st.expander("人工复核（AI 初筛 + 工程师决策）", expanded=False):
            rev_rec = st.number_input("记录ID", min_value=0, step=1)
            verdict_opts = ["", "PASS", "REJECT"]
            rev_v = st.selectbox("复核结论", verdict_opts)
            rev_note = st.text_area("备注", placeholder="如：确认NG，划伤超标退回供应商")
            rev_by = st.text_input("复核人")
            if st.button("提交复核", type="secondary") and rev_rec and rev_v:
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["质量总览", "批次/供应商报表", "追溯查询", "物料管理", "检验标准", "数据积累"])

    # ---- 总览 ----
    with tab1:
        # 顶部 KPI 卡片
        c1, c2, c3 = st.columns(3)
        vstats = db.verdict_stats()
        c1.metric("检验记录总数", sum(vstats.values()))
        c2.metric("AI 判定 OK", vstats.get("OK", 0))
        c3.metric("AI 判定 NG", vstats.get("NG", 0))

        # 左：判定分布 + 缺陷分布；右：物料良率趋势
        rowL, rowR = st.columns([1, 1])
        with rowL:
            with sec_card("AI 判定分布"):
                st.bar_chart(pd.Series(vstats))
            end_card()
            with sec_card("缺陷类型分布"):
                ddf = rp.defect_report()
                if not ddf.empty:
                    st.bar_chart(ddf.set_index("缺陷类型"))
                    st.dataframe(ddf, use_container_width=True)
                else:
                    st.info("暂无缺陷记录")
            end_card()
        with rowR:
            with sec_card("物料良率趋势", "按检验顺序累计 OK 占比，观察质量波动"):
                mtr = rp.material_trend()
                if not mtr.empty:
                    st.line_chart(mtr.set_index("累计件数")[["良率"]])
                    st.dataframe(mtr[["检验时间", "料号", "AI判定", "良率"]].tail(10),
                                 use_container_width=True)
                else:
                    st.info("暂无检验数据")
            end_card()
            with sec_card("复核一致性", "AI 初判 vs 人工复核冲突，衡量 AI 可靠性"):
                rcf = rp.review_consistency()
                if not rcf.empty:
                    conflict = (rcf["冲突"] == "是").sum()
                    st.metric("已复核记录", len(rcf), f"冲突 {conflict} 条")
                    st.dataframe(rcf[["记录ID", "料号", "AI判定", "人工复核", "冲突"]],
                                 use_container_width=True)
                else:
                    st.info("暂无复核记录（检验员端复核后显示）")
            end_card()

        # 数据回灌：导出复核数据为训练集
        st.markdown("---")
        with sec_card("🔄 数据回灌", "把人工复核/AI判定OK的照片导出为训练数据，本地免费引擎越用越准"):
            cexp1, cexp2 = st.columns([1, 1])
            if cexp1.button("导出训练数据（含AI-OK种子）", type="secondary"):
                from src import export_training as et
                stats = et.export_training_data()
                st.success(f"已导出：规格 {sum(stats['spec'].values())} 张 {dict(stats['spec'])}，"
                           f"缺陷 OK {stats['defect_ok']} / NG {stats['defect_ng']}")
                st.info(f"训练数据位置：data/training/\n"
                        f"重训规格模型：python src/detection/spec_model.py --epochs 25 --backbone resnet18\n"
                        f"重训缺陷模型：python src/detection/defect_model.py --train --epochs 40")
            if cexp2.button("仅导出有人工复核的记录", type="secondary"):
                from src import export_training as et
                stats = et.export_training_data(only_reviewed=True)
                st.success(f"已导出（仅复核）: 规格 {sum(stats['spec'].values())} 张，"
                           f"缺陷 OK {stats['defect_ok']} / NG {stats['defect_ng']}")
        end_card()

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

    # ---- 物料管理（白名单）----
    with tab4:
        st.subheader("物料白名单管理")
        st.caption("对稳定供应的重要来料启用 AI 检测；其余物料仍走普通人工记录，不做 AI 判定。"
                   "检测项可单独开关。")
        c1, c2 = st.columns([1.2, 1])

        # 左：新增/编辑物料配置
        with c1:
            with st.form("material_form"):
                m_no = st.text_input("料号 *", placeholder="如 M4-HEX-SCREW")
                m_name = st.text_input("名称", placeholder="内六角螺钉")
                m_spec = st.selectbox("规格", ["", *SPEC_CLASSES])
                m_cat = st.selectbox("类别", ["standard", "parts"])
                ai_on = st.checkbox("启用 AI 检测", value=True)
                ck_spec = st.checkbox("检测项：规格识别", value=True)
                ck_dim = st.checkbox("检测项：尺寸测量", value=True)
                ck_defect = st.checkbox("检测项：表面缺陷", value=True)
                m_pri = st.selectbox("重要性", ["", "high", "medium", "low"])
                m_sup = st.text_input("常用供应商")
                m_note = st.text_area("备注", placeholder="如：每周稳定到货，重点监控表面划伤")
                saved = st.form_submit_button("保存物料配置")
            if saved and m_no:
                db.upsert_material(m_no, m_name or None, m_spec or None, m_cat,
                                   ai_on, ck_spec, ck_dim, ck_defect,
                                   m_pri or None, m_sup or None, m_note or None)
                st.success(f"物料 {m_no} 配置已保存")
                st.rerun()

        # 右：已有物料列表 + 操作
        with c2:
            flt_enabled = st.radio("筛选", ["全部", "已启用", "已停用"], horizontal=True, key="mat_flt")
            f = None
            if flt_enabled == "已启用":
                f = {"ai_enabled": 1}
            elif flt_enabled == "已停用":
                f = {"ai_enabled": 0}
            mats = db.get_materials(f)
            if not mats:
                st.info("暂无物料配置。左边填写料号保存，或用下方按钮从历史批次导入。")
            else:
                for m in mats:
                    st.markdown(f"**{m['material_no']}**"
                                + (f" · {m['material_name']}" if m['material_name'] else "")
                                + (f" · {m['spec']}" if m['spec'] else "")
                                + (" · ⚠️停用" if not m['ai_enabled'] else " · ✅启用"))
                    st.caption("检测项：" +
                               "、".join(n for on, n in
                                         [(m['check_spec'], "规格"), (m['check_dim'], "尺寸"),
                                          (m['check_defect'], "缺陷")] if on)
                               + (f" · 重要性:{m['priority']}" if m['priority'] else "")
                               + (f" · {m['note']}" if m['note'] else ""))
                    if st.button("停用", key=f"off_{m['material_no']}"):
                        m["ai_enabled"] = 0
                        db.upsert_material(**m)
                        st.rerun()
                    if not m["ai_enabled"] and st.button("启用", key=f"on_{m['material_no']}"):
                        m["ai_enabled"] = 1
                        db.upsert_material(**m)
                        st.rerun()

            st.markdown("---")
            st.caption("从历史批次导入（把检过的料号登记为白名单物料）：")
            batches = db.get_batches(limit=200)
            seen = set()
            candidates = []
            for b in batches:
                if b["material_no"] and b["material_no"] not in seen:
                    seen.add(b["material_no"])
                    candidates.append(f"{b['material_no']} · {b['material_name'] or ''}")
            if candidates:
                sel = st.selectbox("选择料号导入", candidates, key="import_sel")
                if st.button("导入为白名单物料"):
                    for b in batches:
                        if f"{b['material_no']} · {b['material_name'] or ''}" == sel:
                            db.upsert_material(b["material_no"], b.get("material_name"),
                                               b.get("spec"), b.get("category", "standard"))
                            break
                    st.success("已导入，默认启用全部检测项")
                    st.rerun()
            else:
                st.info("历史批次暂无数据")

    # ---- 检验标准（图纸技术要求 → 逐项检验规范）----
    with tab5:
        st.subheader("检验标准编辑器")
        st.caption("把图纸技术要求/检验部位转化为逐项检验规范：部位（检验位置）+ 项目 + 图纸阈值。"
                   "检验员端按料号自动匹配执行。")

        # 选择清单：已有或新建
        cl_lists = cl.list_checklists()
        cl_opts = ["➕ 新建标准"] + [f"{c['drawing_no']} · {c['part_name'] or ''}" for c in cl_lists]
        sel_cl = st.selectbox("选择/新建检验标准", cl_opts, key="cl_sel")

        # 当前编辑的清单（session 缓存）
        if "editing_checklist" not in st.session_state or \
                st.session_state.get("cl_key") != sel_cl:
            if sel_cl == "➕ 新建标准":
                st.session_state.editing_checklist = {
                    "drawing_no": "", "part_name": "", "material_no": "",
                    "material": "", "heat_treatment": "", "hardness": "",
                    "dimensions_mm": {}, "surface_roughness": {},
                    "items": [],
                }
            else:
                dn = sel_cl.split(" ·")[0]
                st.session_state.editing_checklist = cl.load_checklist(dn)
            st.session_state.cl_key = sel_cl

        cl_data = st.session_state.editing_checklist
        if cl_data is None:
            st.info("未找到清单")
        else:
            cA, cB = st.columns([1.1, 1])
            with cA:
                # 清单头信息
                st.markdown("**清单头信息**")
                c1, c2 = st.columns(2)
                cl_data["drawing_no"] = c1.text_input("图号 *", value=cl_data.get("drawing_no", ""),
                                                      key="cl_dn")
                cl_data["part_name"] = c2.text_input("零件名称", value=cl_data.get("part_name", ""),
                                                     key="cl_pn")
                cl_data["material_no"] = c1.text_input("料号（匹配白名单）",
                                                       value=cl_data.get("material_no", ""),
                                                       key="cl_mn")
                cl_data["material"] = c2.text_input("材料", value=cl_data.get("material", ""),
                                                    key="cl_mat")
                cl_data["heat_treatment"] = c1.text_input("热处理", value=cl_data.get("heat_treatment", ""),
                                                          key="cl_ht")
                cl_data["hardness"] = c2.text_input("硬度", value=cl_data.get("hardness", ""),
                                                    key="cl_hr")

                # 新增检查项表单
                st.markdown("**新增检查项**")
                with st.form("checkitem_form", clear_on_submit=True):
                    it_label = st.text_input("项目名称 *", placeholder="如 头部端面划伤")
                    it_loc = st.text_input("检验位置", placeholder="如 头部端面 / 螺杆外表面 / 孔内",
                                           help="引导视觉模型到指定部位检查；Phase1 预留像素ROI")
                    it_type = st.selectbox("类型（视觉可验证性）",
                                           ["surface", "dimension", "hardness", "roughness",
                                            "geometric", "deburr"])
                    it_req = st.text_input("技术要求（从图纸）", placeholder="如 无长度>2mm划伤，最多1处")
                    it_visual = it_type in cl.VISUAL_TYPES
                    if not it_visual:
                        st.caption("⚠️ 非表面类：照片无法验证，检测时记 UNSURE，需仪器检测")
                    # 量化阈值（机器判定用）
                    st.caption("**量化阈值（可选，机器自动判定）**")
                    it_tol = {}
                    if it_type in ("dimension", "geometric"):
                        cN, cT = st.columns(2)
                        nom = cN.number_input("名义值 mm", min_value=0.0, value=0.0, step=0.1)
                        tolm = cT.number_input("公差 ±mm", min_value=0.0, value=0.0, step=0.01)
                        if nom > 0 or tolm > 0:
                            it_tol = {"nominal_mm": float(nom), "tol_mm": float(tolm)}
                    elif it_type in ("surface", "deburr"):
                        mx = st.number_input("缺陷数量上限", min_value=0, value=0, step=1)
                        if mx > 0:
                            it_tol = {"max_count": int(mx)}
                    add_item = st.form_submit_button("添加检查项")
                if add_item and it_label:
                    cl_data["items"].append({
                        "id": f"{len(cl_data['items'])+1:02d}",
                        "type": it_type, "label": it_label,
                        "location": it_loc or None,
                        "requirement": it_req or "",
                        "tolerance": it_tol or None,
                    })
                    st.success(f"已添加: {it_label}")
                    st.rerun()

                # 保存
                if st.button("💾 保存检验标准", type="primary"):
                    if not cl_data.get("drawing_no"):
                        st.error("图号必填")
                    else:
                        path = cl.save_checklist(cl_data)
                        st.success(f"已保存: {path.name}")
                        st.rerun()

            with cB:
                # 现有检查项列表 + 删除
                st.markdown(f"**检查项列表（{len(cl_data.get('items', []))} 项）**")
                if not cl_data.get("items"):
                    st.info("暂无检查项，左边添加")
                for it in cl_data.get("items", []):
                    loc = f" · 📍{it['location']}" if it.get("location") else ""
                    vis = "" if it.get("type") in cl.VISUAL_TYPES else " · 🔬需仪器"
                    tol = it.get("tolerance")
                    tol_txt = ""
                    if tol:
                        if "nominal_mm" in tol:
                            tol_txt = f" · 阈 {tol['nominal_mm']}±{tol['tol_mm']}mm"
                        elif "max_count" in tol:
                            tol_txt = f" · 阈 ≤{tol['max_count']}处"
                    st.markdown(f"- **{it['label']}**{loc}{vis}{tol_txt}：{it.get('requirement')}")
                    if st.button("删除", key=f"del_{it['id']}"):
                        cl_data["items"] = [x for x in cl_data["items"] if x["id"] != it["id"]]
                        st.rerun()

    # ---- 数据积累（攒训练数据 → 本地模型变准）----
    with tab6:
        st.subheader("训练数据积累进度")
        st.caption("系统通过复核回灌自动积累真实数据。数据攒够后，可重训本地模型，"
                   "逐步替代视觉大模型，实现完全本地免费检测。")
        from src import export_training as et
        prog = et.training_progress()

        # 三个进度条
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**规格识别数据**")
            st.progress(min(1.0, prog["spec_count"] / max(1, prog["spec_target"])),
                        text=f"{prog['spec_count']} / {prog['spec_target']} 张")
            st.caption(f"目标：5 类规格各 {et.TARGET_SPEC_PER_CLASS} 张")
        with c2:
            st.markdown("**卡尺读数配对**")
            st.progress(min(1.0, prog["reading_count"] / max(1, prog["reading_target"])),
                        text=f"{prog['reading_count']} / {prog['reading_target']} 对")
            st.caption("千问读数自动入库 → 攒成图+读数配对，训练本地读数模型")
        with c3:
            st.markdown("**缺陷样本（NG）**")
            st.progress(min(1.0, prog["defect_ng"] / max(1, prog["defect_target"])),
                        text=f"{prog['defect_ng']} / {prog['defect_target']} 张")
            st.caption(f"当前 OK {prog['defect_ok']} 张 / NG {prog['defect_ng']} 张")

        st.markdown("---")
        st.info("**攒够目标后**：\n"
                f"- 规格识别 → `python src/detection/spec_model.py --epochs 25 --backbone resnet18`\n"
                f"- 缺陷检测 → `python src/detection/fastener_defect_model.py --train --epochs 40`\n"
                f"- 重训后本地模型准确率提升，可逐步关闭视觉大模型 → 检测零成本")

        if st.button("🔄 立即导出当前训练数据"):
            stats = et.export_training_data()
            st.success(f"已导出：规格 {sum(stats['spec'].values())} 张，"
                       f"缺陷 OK {stats['defect_ok']} / NG {stats['defect_ng']}")


# ================= 主入口 =================

def main():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
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
