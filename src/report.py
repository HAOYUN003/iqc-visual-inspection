# -*- coding: utf-8 -*-
"""
报表与追溯模块：供应商/批次质量统计、缺陷分布、检测报告导出
供 Streamlit 主管端调用，也可独立使用。
"""
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import quality_db as db
from config import DEFECT_CLASSES


def batch_report(filters=None):
    """按批次维度汇总：批次数、检验件数、OK/NG、一次合格率、抽检覆盖率"""
    batches = db.get_batches(filters)
    rows = []
    for b in batches:
        records = db.get_records({"batch_id": b["batch_id"]})
        total = len(records)
        ok = sum(1 for r in records if r["ai_verdict"] == "OK")
        ng = total - ok
        reviewed = [r for r in records if r.get("review_verdict")]
        pass_after_review = sum(1 for r in reviewed if r["review_verdict"] == "PASS")
        qty = b["quantity"] or 0
        cover = (total / qty) if qty else None
        rows.append({
            "批次号": b["batch_id"],
            "料号": b["material_no"],
            "名称": b["material_name"],
            "规格": b["spec"],
            "供应商": b["supplier"],
            "数量": qty,
            "到货日期": b["arrival_date"],
            "检验件数": total,
            "抽检覆盖率": round(cover, 3) if cover is not None else None,
            "OK数": ok,
            "NG数": ng,
            "AI合格率": round(ok / total, 4) if total else None,
            "复检通过数": pass_after_review,
        })
    return pd.DataFrame(rows)


def supplier_report():
    """供应商质量排行"""
    return pd.DataFrame(db.supplier_stats())


def defect_report():
    """缺陷类型分布（转为 DataFrame）"""
    d = db.defect_type_stats()
    df = pd.DataFrame({"缺陷类型": list(d.keys()), "出现次数": list(d.values())})
    df = df.sort_values("出现次数", ascending=False).reset_index(drop=True)
    return df


def trace_query(batch_id=None, material_no=None, supplier=None, verdict=None,
                date_from=None, date_to=None, limit=500):
    """追溯查询：返回检验记录明细 DataFrame"""
    filters = {k: v for k, v in {
        "batch_id": batch_id, "material_no": material_no, "supplier": supplier,
        "ai_verdict": verdict,
        "date_from": date_from, "date_to": date_to,
    }.items() if v}
    records = db.get_records(filters, limit=limit)
    rows = []
    for r in records:
        r = db.parse_defect_result(r)
        dim = r.get("dimension")
        if dim and dim.get("reading_mm") is not None:
            dim_str = f"读数 {dim['reading_mm']:.2f}mm {dim.get('status', '')}"
        elif dim and dim.get("outer_diam_mm") is not None:
            dim_str = f"{dim['outer_diam_mm']:.2f}mm {dim.get('status', '')}"
        else:
            dim_str = ""
        rows.append({
            "记录ID": r["record_id"],
            "批次号": r["batch_id"],
            "料号": r["material_no"],
            "规格": r["spec"],
            "供应商": r["supplier"],
            "检验员": r["inspector"],
            "检验时间": r["inspected_at"],
            "AI规格": r["spec_result"],
            "尺寸判定": dim_str,
            "缺陷": _fmt_defect(r["defect_result"]),
            "清单校验": _fmt_checklist(r.get("checklist")),
            "AI判定": r["ai_verdict"],
            "人工复核": r["review_verdict"],
            "备注": r["review_note"],
        })
    return pd.DataFrame(rows)


def _fmt_defect(defect_json):
    if not defect_json:
        return ""
    import json
    try:
        d = json.loads(defect_json)
        if isinstance(d, dict):
            return ", ".join(f"{k}×{v}" for k, v in d.items())
    except Exception:
        pass
    return str(defect_json)


def _fmt_checklist(checklist):
    """清单逐项结果 → 摘要，如「表面状态OK/去毛刺NG/硬度需仪器」"""
    if not checklist:
        return ""
    out = []
    for it in checklist:
        if isinstance(it, dict):
            status = {"OK": "OK", "NG": "NG", "UNSURE": "需仪器"}.get(it.get("status"), "?")
            out.append(f"{it.get('label')}{status}")
    return "；".join(out)


def export_report(df, filepath=None, fmt="csv"):
    """导出检测报告。fmt: csv | xlsx"""
    if filepath is None:
        fname = datetime.now().strftime("iqc_report_%Y%m%d_%H%M%S")
        filepath = Path(__file__).resolve().parents[1] / "reports" / f"{fname}.{fmt}"
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "xlsx":
        df.to_excel(filepath, index=False)
    else:
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
    return filepath


# ================= 报表增强分析 =================

def material_trend(material_no=None, limit=500):
    """按物料的良率趋势（按时间），用于观察重点物料质量波动。"""
    records = db.get_records({"material_no": material_no} if material_no else None,
                             limit=limit)
    rows = []
    for r in records:
        rows.append({
            "检验时间": r["inspected_at"],
            "料号": r["material_no"],
            "AI判定": r["ai_verdict"],
            "复核": r["review_verdict"],
            "缺陷": _fmt_defect(r["defect_result"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("检验时间").reset_index(drop=True)
    # 良率 = OK 占比（按记录序号滚动）
    df["累计OK数"] = (df["AI判定"] == "OK").cumsum()
    df["累计件数"] = range(1, len(df) + 1)
    df["良率"] = df["累计OK数"] / df["累计件数"]
    return df


def review_consistency(limit=500):
    """复核一致性：AI 初判 vs 人工复核的冲突率。
    冲突 = AI判定 OK 但复核 REJECT，或 AI判定 NG 但复核 PASS。"""
    records = db.get_records(limit=limit)
    rows = []
    for r in records:
        if not r.get("review_verdict"):
            continue
        ai = r.get("ai_verdict")
        rv = r["review_verdict"]
        conflict = (ai == "OK" and rv == "REJECT") or (ai == "NG" and rv == "PASS")
        rows.append({
            "记录ID": r["record_id"],
            "料号": r["material_no"],
            "AI判定": ai,
            "人工复核": rv,
            "冲突": "是" if conflict else "否",
            "复核备注": r.get("review_note", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["冲突率"] = df["冲突"].apply(lambda x: 1 if x == "是" else 0)
    return df


def defect_breakdown(limit=500):
    """缺陷类型分布明细（含每张图的缺陷计数），用于看主要缺陷。"""
    records = db.get_records(limit=limit)
    from collections import Counter
    counter = Counter()
    for r in records:
        r = db.parse_defect_result(r)
        d = r.get("defect_result")
        if isinstance(d, dict):
            counter.update(d)
        # 清单校验里的 NG 表面项也算
        cl = r.get("checklist")
        if isinstance(cl, list):
            for it in cl:
                if isinstance(it, dict) and it.get("status") == "NG" and it.get("visual"):
                    counter[it.get("label")] += 1
    if not counter:
        return pd.DataFrame()
    df = pd.DataFrame({"缺陷类型": list(counter.keys()),
                       "出现次数": list(counter.values())})
    return df.sort_values("出现次数", ascending=False).reset_index(drop=True)


def batch_traceback(batch_id):
    """上机不良反查：输入批次号，拉出该批次全部检验记录明细。
    用于"上机发现不良 → 反查该批次抽检记录"，快速定位是否漏检。"""
    records = db.get_records({"batch_id": batch_id}, limit=500)
    rows = []
    for r in records:
        r = db.parse_defect_result(r)
        img = r.get("image_path") or ""
        rows.append({
            "记录ID": r["record_id"],
            "检验时间": r["inspected_at"],
            "检验员": r["inspector"],
            "AI判定": r["ai_verdict"],
            "人工复核": r["review_verdict"],
            "规格": r["spec_result"],
            "缺陷": _fmt_defect(r.get("defect_result")),
            "照片": img,
        })
    df = pd.DataFrame(rows)
    return df


def batch_traceback_summary(batch_id):
    """反查摘要：该批次检验件数、OK/NG 分布、是否有复核、照片是否留存。
    帮助判断"抽检记录是否足以定位问题"。"""
    records = db.get_records({"batch_id": batch_id}, limit=500)
    n = len(records)
    ok = sum(1 for r in records if r["ai_verdict"] == "OK")
    ng = n - ok
    reviewed = sum(1 for r in records if r.get("review_verdict"))
    with_photo = sum(1 for r in records if r.get("image_path"))
    return {
        "batch_id": batch_id,
        "检验件数": n,
        "OK数": ok,
        "NG数": ng,
        "已复核": reviewed,
        "留照片": with_photo,
    }


if __name__ == "__main__":
    print("批次报表:")
    print(batch_report())
    print("\n供应商统计:")
    print(supplier_report())
    print("\n缺陷分布:")
    print(defect_report())
