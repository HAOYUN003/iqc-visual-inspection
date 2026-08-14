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
    """按批次维度汇总：批次数、检验件数、OK/NG、一次合格率"""
    batches = db.get_batches(filters)
    rows = []
    for b in batches:
        records = db.get_records({"batch_id": b["batch_id"]})
        total = len(records)
        ok = sum(1 for r in records if r["ai_verdict"] == "OK")
        ng = total - ok
        reviewed = [r for r in records if r.get("review_verdict")]
        pass_after_review = sum(1 for r in reviewed if r["review_verdict"] == "PASS")
        rows.append({
            "批次号": b["batch_id"],
            "料号": b["material_no"],
            "名称": b["material_name"],
            "规格": b["spec"],
            "供应商": b["supplier"],
            "数量": b["quantity"],
            "到货日期": b["arrival_date"],
            "检验件数": total,
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


if __name__ == "__main__":
    print("批次报表:")
    print(batch_report())
    print("\n供应商统计:")
    print(supplier_report())
    print("\n缺陷分布:")
    print(defect_report())
