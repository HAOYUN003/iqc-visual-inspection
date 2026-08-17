# -*- coding: utf-8 -*-
"""
图纸检测清单：加载 / 查找 / 构造视觉判定 prompt / 解析逐项结果。

清单 JSON 存于 data/inspection_checklists/<drawing_no>.json，结构见
R05-03130412-E01.json（items 数组：id / type / label / requirement）。

诚实性原则：照片只能验证 surface 类项（划伤/碰伤/毛刺/倒钝），
hardness / 精确尺寸 / 粗糙度 无法从照片验证 → 判 UNSURE 并标注"需仪器检测"，
不交给视觉模型瞎猜。
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR

CHECKLIST_DIR = DATA_DIR / "inspection_checklists"

# 视觉可验证的项类型（照片能实际看到的）
VISUAL_TYPES = {"surface", "deburr"}


def list_checklists():
    """返回所有清单的摘要列表（drawing_no / part_name / material_no）。"""
    if not CHECKLIST_DIR.exists():
        return []
    out = []
    for f in sorted(CHECKLIST_DIR.glob("*.json")):
        try:
            cl = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "drawing_no": cl.get("drawing_no", f.stem),
            "part_name": cl.get("part_name"),
            "material_no": cl.get("material_no"),
        })
    return out


def load_checklist(drawing_no=None, path=None):
    """按图号加载清单 dict。path 直接指定时优先。"""
    if path:
        p = Path(path)
    elif drawing_no:
        p = CHECKLIST_DIR / f"{drawing_no}.json"
    else:
        raise ValueError("load_checklist 需要 drawing_no 或 path")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def find_checklist(material_no=None, drawing_no=None):
    """按料号或图号查找清单。料号命中优先，其次图号。"""
    for cl in list_checklists():
        if material_no and cl["material_no"] == material_no:
            return load_checklist(cl["drawing_no"])
    for cl in list_checklists():
        if drawing_no and cl["drawing_no"] == drawing_no:
            return load_checklist(cl["drawing_no"])
    return None


def save_checklist(checklist):
    """保存/覆盖清单 JSON。返回保存路径。"""
    drawing_no = checklist.get("drawing_no")
    if not drawing_no:
        raise ValueError("清单必须包含 drawing_no")
    CHECKLIST_DIR.mkdir(parents=True, exist_ok=True)
    path = CHECKLIST_DIR / f"{drawing_no}.json"
    path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def split_visual_nonvisual(checklist):
    """把 items 分成 [可视觉验证项, 需仪器项]。"""
    vis, nonvis = [], []
    for it in checklist.get("items", []):
        (vis if it.get("type") in VISUAL_TYPES else nonvis).append(it)
    return vis, nonvis


def build_checklist_prompt(checklist, images_info=None):
    """
    构造清单逐项判定 prompt。
    images_info: 可选 str，描述待检角度（如"正面、侧面、孔位"）。
    检验项支持 location（检验部位）字段：有则注入 prompt，引导视觉模型到指定部位查。
    """
    header = (
        "这是机加工零件（{part}，{drawing_no}）的照片，按图纸技术要求逐项校验。\n"
        "只检查零件本身，忽略背景、桌面、反光、手指和测量工具。\n"
        "下列各检查项中：\n"
        "  表面/倒钝类：根据照片如实判定，看到缺陷输出 NG 并说明缺陷；\n"
        "  硬度/尺寸/粗糙度/形位类：照片无法验证，一律输出 UNSURE 且 reason 注明'需仪器检测'，不要猜测。\n"
    ).format(part=checklist.get("part_name", ""),
             drawing_no=checklist.get("drawing_no", ""))
    if images_info:
        header += f"照片角度：{images_info}\n"

    items = checklist.get("items", [])
    lines = []
    for i, it in enumerate(items, 1):
        label = it.get("label")
        loc = it.get("location")
        req = it.get("requirement")
        loc_txt = f"（检验位置：{loc}）" if loc else ""
        lines.append(f"{i}. {label}{loc_txt}：{req}")
    header += "检查项：\n" + "\n".join(lines)

    header += (
        "\n\n只输出一个 JSON 数组，每个检查项一个对象，不要任何解释和代码块标记：\n"
        '[{"item": 1, "status": "OK|NG|UNSURE", "reason": "简短中文说明"}]'
    )
    return header


def parse_item_results(text, checklist):
    """
    解析视觉返回的逐项 JSON，映射回清单 items。
    返回 [ {id,label,requirement,type,status,reason,visual} ]。
    解析失败或缺项时：该项记 UNSURE（未获得视觉结果）。
    未返回的项：UNSURE + "视觉未给出该项判定"。
    """
    items = checklist.get("items", [])
    by_id = {it.get("id"): it for it in items}
    by_idx = {str(i): it for i, it in enumerate(items, 1)}

    parsed = []
    if text:
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    parsed = data
            except json.JSONDecodeError:
                parsed = []

    # 归一化 status 大小写 / 中文
    def norm_status(s):
        if not s:
            return "UNSURE"
        s = str(s).strip().upper()
        if s in ("OK", "NG", "UNSURE"):
            return s
        if s in ("通过", "合格", "符合"):
            return "OK"
        if s in ("不合格", "不通过", "不符合", "超差"):
            return "NG"
        return "UNSURE"

    results = []
    seen = set()
    for row in parsed:
        if not isinstance(row, dict):
            continue
        it = None
        if row.get("item") is not None:
            it = by_idx.get(str(row.get("item"))) or by_id.get(str(row.get("item")))
        elif row.get("id"):
            it = by_id.get(str(row["id"]))
        if it is None:
            continue
        iid = it["id"]
        if iid in seen:
            continue
        seen.add(iid)
        results.append({
            "id": iid,
            "label": it.get("label"),
            "location": it.get("location"),
            "requirement": it.get("requirement"),
            "type": it.get("type"),
            "visual": it.get("type") in VISUAL_TYPES,
            "tolerance": it.get("tolerance"),
            "roi": it.get("roi"),
            "status": norm_status(row.get("status")),
            "reason": row.get("reason") or "",
        })

    # 补漏：清单里但视觉没返回的项
    for it in items:
        if it["id"] not in seen:
            results.append({
                "id": it["id"],
                "label": it.get("label"),
                "location": it.get("location"),
                "requirement": it.get("requirement"),
                "type": it.get("type"),
                "visual": it.get("type") in VISUAL_TYPES,
                "tolerance": it.get("tolerance"),
                "roi": it.get("roi"),
                "status": "UNSURE",
                "reason": "视觉未给出该项判定",
            })
    return results


def judge_item_quantitative(item, measured_mm=None, defect_counts=None):
    """
    按 tolerance 做机器量化判定，覆盖视觉模型的 UNSURE。
    - dimension/几何类：measured_mm（实测值） vs nominal_mm ± tol_mm → OK/NG/UNSURE
    - surface 类：defect_counts（缺陷计数 dict） vs max_count/max_len → OK/NG
    返回 (status, reason)。无 tolerance 或数据不足时返回视觉原判。
    """
    tol = item.get("tolerance")
    if not tol:
        return None, None
    itype = item.get("type")

    # ---- 尺寸类：实测值 vs 公差 ----
    if itype in ("dimension", "geometric") and measured_mm is not None:
        nominal = tol.get("nominal_mm")
        tol_mm = tol.get("tol_mm")
        if nominal is None or tol_mm is None:
            return None, None
        dist = abs(measured_mm - nominal)
        if dist <= tol_mm:
            return "OK", f"实测 {measured_mm:.2f}mm，名义 {nominal:.2f}mm±{tol_mm:.2f}，偏差 {dist:.2f}mm（合格）"
        return "NG", f"实测 {measured_mm:.2f}mm，名义 {nominal:.2f}mm±{tol_mm:.2f}，偏差 {dist:.2f}mm（超差）"

    # ---- 表面类：缺陷计数 vs 上限 ----
    if itype in ("surface", "deburr") and defect_counts:
        max_count = tol.get("max_count")
        # 尺寸超差仍无、缺计数上限时退回视觉原判
        if max_count is None:
            return None, None
        total = sum(defect_counts.values())
        if total == 0:
            return "OK", "无可见缺陷"
        if total <= max_count:
            return "OK", f"缺陷 {total} 处（上限 {max_count}），在允许范围"
        return "NG", f"缺陷 {total} 处（上限 {max_count}），超标"
    return None, None


def resolve_checklist_verdict(item_results):
    """逐项判定 → 综合判定：任一 NG → NG；全 OK → OK；否则 UNSURE。
    返回 (verdict, reasons)。"""
    ngs = [r for r in item_results if r["status"] == "NG"]
    unsure = [r for r in item_results if r["status"] == "UNSURE"]
    if ngs:
        reasons = [f"NG: {r['label']}（{r['reason']}）" for r in ngs]
        return "NG", reasons
    if unsure:
        reasons = [f"无法判定: {r['label']}（{r['reason'] or '需进一步检测'}）" for r in unsure]
        return "UNSURE", reasons
    reasons = [f"OK: {r['label']}" for r in item_results]
    return "OK", reasons


if __name__ == "__main__":
    cl = load_checklist("R05-03130412-E01")
    print("清单:", cl["drawing_no"], cl["part_name"], cl["material"], cl["hardness"])
    print("项数:", len(cl["items"]))
    vis, nonvis = split_visual_nonvisual(cl)
    print(f"可视觉验证 {len(vis)} 项: {[i['label'] for i in vis]}")
    print(f"需仪器 {len(nonvis)} 项: {[i['label'] for i in nonvis]}")
