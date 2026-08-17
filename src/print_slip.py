# -*- coding: utf-8 -*-
"""
检测单生成：检验合格/不合格后生成可打印的 HTML 单据，浏览器打印即可。
免手写，提高效率。含批次/料号/规格/数量/判定/缺陷/检验员/时间/供应商。
"""
import html
from datetime import datetime


def _h(s):
    return html.escape(str(s if s is not None else ""))


def _fmt_defect(summary):
    if not summary:
        return "无"
    return "、".join(f"{k}×{v}" for k, v in summary.items())


def _verdict_label(verdict):
    return {"OK": "合格", "NG": "不合格", "UNSURE": "需复检", "SKIP": "跳过"}.get(
        str(verdict).upper(), str(verdict))


def build_slip_html(batch_info, result, record_id, verdict, reviewer_note=""):
    """
    batch_info: dict(batch_id/material_no/material_name/spec_expected/supplier/inspector/quantity)
    result: 检测结果 dict（spec_result/dimension/defect_summary/thread_result/reasons）
    返回 HTML 字符串（A4 单张，可直接浏览器打印）。
    """
    bi = batch_info or {}
    dim = result.get("dimension") or {}
    reading = dim.get("reading_mm")
    outer = dim.get("outer_diam_mm")
    if reading is not None:
        dim_txt = f"{reading:.2f}mm"
        if dim.get("nominal_mm"):
            dim_txt += f"（名义 {dim['nominal_mm']}±{dim.get('tolerance_mm')}）"
    elif outer is not None:
        dim_txt = f"外径 {outer:.2f}mm"
    else:
        dim_txt = "—"

    thread = result.get("thread_result")
    thread_txt = {"good": "正常", "missing": "缺牙", "broken": "烂牙"}.get(thread, thread or "—")

    reasons = "；".join(result.get("reasons") or []) or "—"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    vlabel = _verdict_label(verdict)
    color = "#047857" if verdict == "OK" else "#b91c1c" if verdict == "NG" else "#b45309"

    rows = [
        ("批次号", bi.get("batch_id")),
        ("料号", bi.get("material_no")),
        ("物料名称", bi.get("material_name")),
        ("规格", bi.get("spec_expected") or result.get("spec_result") or "—"),
        ("供应商", bi.get("supplier")),
        ("批次数量", bi.get("quantity")),
        ("检验员", bi.get("inspector")),
        ("检验时间", now),
    ]

    body = "".join(
        f"<tr><td class='lbl'>{_h(k)}</td><td class='val'>{_h(v)}</td></tr>"
        for k, v in rows)

    return f"""<!DOCTYPE html>
<html lang="zh">
<head><meta charset="utf-8"><title>IQC 检测单</title>
<style>
  body {{ font-family: 'Microsoft YaHei', sans-serif; padding: 24px; width: 190mm; }}
  h1 {{ text-align: center; font-size: 22px; margin: 0 0 4px 0; }}
  .sub {{ text-align: center; color: #666; font-size: 13px; margin-bottom: 16px; }}
  .verdict {{ text-align: center; font-size: 40px; font-weight: 800; color: {color};
             border: 3px solid {color}; border-radius: 8px; padding: 12px;
             margin: 12px 0 20px 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 15px; }}
  td {{ border: 1px solid #999; padding: 8px 12px; }}
  .lbl {{ background: #f3f4f6; width: 30%; font-weight: 600; }}
  .val {{ width: 70%; }}
  .detail {{ margin-top: 16px; font-size: 14px; line-height: 1.8; }}
  .foot {{ margin-top: 24px; font-size: 13px; color: #666;
          display: flex; justify-content: space-between; }}
  @media print {{ body {{ padding: 0; }} }}
</style></head>
<body>
  <h1>IQC 来料检验单</h1>
  <div class="sub">智能来料检测系统 · 记录 #{_h(record_id)}</div>
  <div class="verdict">{_h(vlabel)}</div>
  <table>{body}</table>
  <div class="detail">
    <b>检测结果：</b><br>
    · 规格识别：{_h(result.get('spec_result') or '—')}<br>
    · 尺寸判定：{_h(dim_txt)}<br>
    · 螺纹状态：{_h(thread_txt)}<br>
    · 表面缺陷：{_h(_fmt_defect(result.get('defect_summary')))}<br>
    · 判定原因：{_h(reasons)}
  </div>
  <div class="foot">
    <span>检验员：{_h(bi.get('inspector'))}</span>
    <span>打印时间：{now}</span>
    <span>复核备注：{_h(reviewer_note)}</span>
  </div>
</body></html>"""


def slip_html_to_str(h):
    """占位：如需把 HTML 转文本/PDF 可扩展。"""
    return h
