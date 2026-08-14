# -*- coding: utf-8 -*-
"""
视觉检测通道（卡尺照片）：Qwen-VL 读卡尺读数 → 型号识别 → 尺寸判定 → 视觉缺陷检测。

替代原 pipeline.run_detection 的 CV/YOLO 通道——真实卡尺特写照片上 CV 无法分割、
YOLO 缺陷模型误检严重，统一改用视觉模型。

返回 dict 与 run_detection 字段对齐：spec_result / spec_confidence / spec_method /
dimension / defect_summary / defect_boxes / annotated / ai_verdict / reasons / image_path。
"""
import base64
import datetime
import io
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (RAW_DIR, SPEC_NOMINAL_HEAD_MM)
from ask_vision import ask, ask_b64
from vision_pipeline import sample_reading, infer_spec_by_reading, READING_RANGE_MM

# 缺陷类别（视觉判定）
DEFECT_CLASSES = ["scratch", "crack", "rust", "pit"]
DEFECT_PROMPT = (
    "这是一张用数显卡尺测量螺钉零件的照片。请只检查零件金属表面（螺钉头部、可见螺杆部分）是否有缺陷，"
    "忽略卡尺本身的划痕、反光、阴影、卡尺爪和桌面背景。逐项判定并给出数量（没有就为0）：\n"
    "scratch 划伤/刮痕；crack 裂纹（贯穿性细线）；rust 锈蚀/锈斑；pit 麻点/点蚀。\n"
    "只输出一个 JSON 对象，不要任何解释和代码块标记，格式：\n"
    '{"scratch": 0, "crack": 0, "rust": 0, "pit": 0}\n'
    "若画面中无法看清零件表面或无法判断，输出 {\"unclear\": true}。"
)

# 中英文缺陷键归一化
DEFECT_KEY_MAP = {
    "scratch": "scratch", "划伤": "scratch", "刮痕": "scratch", "划痕": "scratch",
    "crack": "crack", "裂纹": "crack", "裂缝": "crack",
    "rust": "rust", "锈蚀": "rust", "锈斑": "rust", "生锈": "rust",
    "pit": "pit", "麻点": "pit", "点蚀": "pit", "凹坑": "pit",
    "burr": "burr", "毛刺": "burr", "飞边": "burr",
    "dent": "dent", "碰伤": "dent", "凹痕": "dent", "压伤": "dent", "磕碰": "dent",
    "deform": "deform", "变形": "deform", "翘曲": "deform", "弯曲": "deform",
    "blocked": "blocked", "孔堵": "blocked", "堵孔": "blocked", "毛刺堵孔": "blocked",
}

# 加工件缺陷类别（多角度表面检测）
MACHINED_DEFECT_CLASSES = ["scratch", "burr", "dent", "crack", "rust", "pit"]
MACHINED_DEFECT_PROMPT = (
    "这是机加工件的实物照片（可能包含零件的一个或多个面）。请只检查零件表面是否有以下缺陷，"
    "忽略背景、桌面、反光、手指和测量工具。逐项判定并给出数量（没有就为0）：\n"
    "scratch 划伤/刮痕；burr 毛刺/飞边；dent 碰伤/凹痕/压伤；crack 裂纹（贯穿性细线）；"
    "rust 锈蚀/锈斑；pit 麻点/点蚀。\n"
    "只输出一个 JSON 对象，不要任何解释和代码块标记，格式：\n"
    '{"scratch": 0, "burr": 0, "dent": 0, "crack": 0, "rust": 0, "pit": 0}\n'
    "若照片中看不清零件或无法判断，输出 {\"unclear\": true}。"
)

MAX_IMG_DIM = 1536          # 降采样上限：12MP 全图过大，压缩到 ~1.5MP 省流量降时延
READING_CALLS = 3           # 每种提示的采样次数
# 视觉读 LCD 固有误差约 ±0.3mm，判定公差放宽到 ±0.4mm（避免误杀合格品）
READING_TOLERANCE_MM = 0.4


# ================= 图片编码 =================

def _bgr_to_b64(img_bgr, max_dim=MAX_IMG_DIM, quality=90):
    """BGR numpy 数组 → 降采样 → JPEG base64。"""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def _ask_img(img_bgr, image_path, prompt, max_dim=MAX_IMG_DIM):
    """有路径走原 ask（精度一致），否则内存图转 base64。"""
    if image_path and os.path.exists(image_path):
        return ask(image_path, prompt)
    return ask_b64(_bgr_to_b64(img_bgr, max_dim), "jpeg", prompt)


# ================= 读数 + 型号 =================

def _sample_reading_any(img_bgr, image_path):
    """读卡尺读数。reader 用内存图或路径。"""
    reader = lambda p: _ask_img(img_bgr, image_path, p)
    return sample_reading(image_path, reader=reader)


def _make_confidence(reading, samples, dist_mm):
    """合成置信度：一致率 + 规格偏差。"""
    total = sum(c for _, c in samples)
    agree = samples[0][1] / total if total else 0
    dist_conf = max(0.0, 1.0 - (dist_mm or 0) / 1.5)
    return round(0.5 * agree + 0.5 * dist_conf, 3)


# ================= 尺寸判定 =================

def judge_dimension(reading_mm, spec_inferred, expected_spec=None):
    """尺寸判定。返回 dimension dict。
    名义值优先用 expected_spec（规格防错权威），未给时用最近匹配规格。
    公差用视觉读数判定阈值 READING_TOLERANCE_MM（视觉读 LCD 固有误差约 ±0.3mm）。"""
    if reading_mm is None:
        return {"method": "qwen_reading", "status": "UNKNOWN",
                "reason": "卡尺读数识别失败"}
    judge_spec = expected_spec if expected_spec and expected_spec in SPEC_NOMINAL_HEAD_MM \
        else spec_inferred
    if judge_spec is None:
        return {"method": "qwen_reading", "reading_mm": reading_mm,
                "status": "UNKNOWN", "reason": "读数无法匹配到已知规格"}
    nominal = SPEC_NOMINAL_HEAD_MM[judge_spec]
    tol = READING_TOLERANCE_MM
    dist = abs(reading_mm - nominal)
    status = "OK" if dist <= tol else "NG"
    reason = (f"读数 {reading_mm:.2f}mm，名义值 {judge_spec}={nominal}mm±{tol}mm，"
              f"偏差 {dist:.2f}mm" + ("在公差内" if status == "OK" else "超出公差"))
    return {
        "method": "qwen_reading",
        "reading_mm": reading_mm,
        "spec_inferred": spec_inferred,
        "judge_spec": judge_spec,
        "nominal_mm": nominal,
        "tolerance_mm": tol,
        "dist_mm": round(dist, 3),
        "status": status,
        "reason": reason,
    }


# ================= 缺陷视觉判定 =================

def _parse_defect_json(text):
    """容错解析缺陷 JSON，返回 {class: count} 或 {"unclear": True}。"""
    if not text:
        return {"unclear": True}
    # 提取第一个 {...}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"unclear": True}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"unclear": True}
    if not isinstance(data, dict):
        return {"unclear": True}
    if data.get("unclear") or data.get("无法判断"):
        return {"unclear": True}
    result = {}
    for k, v in data.items():
        key = DEFECT_KEY_MAP.get(str(k).strip().lower())
        if key is None:
            key = DEFECT_KEY_MAP.get(str(k).strip())
        if key is None:
            continue
        try:
            n = int(v)
        except (ValueError, TypeError):
            continue
        if n > 0:
            result[key] = n
    return result


def inspect_defects(img_bgr, image_path):
    """视觉判断零件表面缺陷。返回 summary dict（仅 count>0），无法判断返回 None。"""
    try:
        text = _ask_img(img_bgr, image_path, DEFECT_PROMPT)
        parsed = _parse_defect_json(text)
    except Exception:
        return None, "缺陷视觉判定异常"
    if "unclear" in parsed:
        return None, "缺陷视觉判定不确定（无法看清零件表面）"
    return (parsed if parsed else None), None


# ================= 加工件多角度缺陷检测 =================

def inspect_machined_defects(img_bgr, image_path):
    """单张加工件照片表面缺陷判定。返回 (summary, warn)。"""
    try:
        text = _ask_img(img_bgr, image_path, MACHINED_DEFECT_PROMPT)
        parsed = _parse_defect_json(text)
    except Exception as e:
        return None, f"缺陷视觉判定异常: {e}"
    if "unclear" in parsed:
        return None, "缺陷视觉判定不确定（无法看清零件表面）"
    return (parsed if parsed else None), None


def inspect_machined_multi(images):
    """
    多角度加工件照片聚合检测。
    images: list of (img_bgr, image_path) 或 dict {name: path}
    返回 dict：per_image（每张的缺陷与警告）、summary（聚合计数，仅>0）、warnings。
    """
    per_image = []
    aggregated = {}
    warnings = []
    for item in images:
        if isinstance(item, tuple):
            img_bgr, image_path = item
        else:
            img_bgr, image_path = None, item
        label = image_path if image_path else "内存图"
        summary, warn = inspect_machined_defects(img_bgr, image_path)
        per_image.append({"image": label, "defects": summary, "warn": warn})
        if warn:
            warnings.append(f"{label}: {warn}")
        if summary:
            for cls, cnt in summary.items():
                aggregated[cls] = aggregated.get(cls, 0) + cnt
    return {"per_image": per_image, "summary": aggregated, "warnings": warnings}


def run_machined_detection(images, save_image=True, material_no=None):
    """
    加工件多角度缺陷检测主入口。
    images: list of (img_bgr, path) 或 list of path。至少一张。
    返回 dict（对齐 run_detection）：image_path / spec_result=None / defect_summary /
    ai_verdict / reasons / per_image / material_no。
    """
    # 统一 images 为 (bgr, path) 列表
    norm = []
    for item in images:
        if isinstance(item, tuple):
            norm.append(item)
        else:
            norm.append((None, item))

    # 保存原始图（多图）
    saved = []
    if save_image:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for _, path in norm:
            if path and os.path.exists(path):
                fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_machined.jpg"
                dest = str(RAW_DIR / fname)
                import shutil
                shutil.copy(path, dest)
                saved.append(dest)
            elif path is None:
                pass  # 内存图暂不落盘

    result = inspect_machined_multi(norm)

    summary = result["summary"]
    warnings = result["warnings"]
    reasons = list(warnings)
    if summary:
        for cls, cnt in summary.items():
            reasons.append(f"{cls}×{cnt}")
        verdict = "NG"
    elif warnings:
        # 有图无法判断但无明确缺陷：不判 NG，提示不确定
        verdict = "UNSURE" if all("不确定" in w or "异常" in w for w in warnings) else "OK"
    else:
        verdict = "OK"
        reasons.append("所有角度检测通过，无表面缺陷")

    return {
        # 多图：image_path 存 JSON 数组字符串（入库追溯可见全部角度）
        "image_path": json.dumps(saved, ensure_ascii=False) if saved else None,
        "image_paths": saved,
        "spec_result": None,
        "spec_confidence": None,
        "spec_probs": None,
        "spec_method": "vision_machined",
        "dimension": None,
        "defect_summary": summary if summary else None,
        "defect_boxes": None,
        "annotated": None,
        "ai_verdict": verdict,
        "reasons": reasons,
        "per_image": result["per_image"],
        "material_no": material_no,
    }


# ================= 综合判定 =================

def resolve_verdict(dim, spec_inferred, expected_spec, defect_summary,
                    reading_failed, defect_warn):
    """优先级：读数失败 > 规格不符 > 尺寸NG > 缺陷>0 > OK。返回 (verdict, reasons)"""
    reasons = []
    if reading_failed:
        return "UNSURE", ["卡尺读数识别失败"]
    if expected_spec and spec_inferred and expected_spec != spec_inferred:
        return "NG", [f"规格不符: 期望{expected_spec}, 识别{spec_inferred}"]
    if dim and dim.get("status") == "NG":
        return "NG", [dim.get("reason", "尺寸超差")]
    if defect_summary:
        for cls, cnt in defect_summary.items():
            reasons.append(f"{cls}×{cnt}")
        reasons.append("表面缺陷")
        return "NG", reasons
    if dim and dim.get("status") == "OK":
        reasons.append(dim.get("reason", "尺寸合格"))
    if defect_warn:
        reasons.append(defect_warn)
    if not reasons:
        reasons.append("检测通过")
    return "OK", reasons


# ================= 主入口 =================

def run_vision_detection(img_bgr=None, image_path=None, expected_spec=None,
                         inspector="", save_image=True):
    """
    执行一次卡尺照片视觉检测。
    返回 dict（对齐 run_detection）：image_path / spec_result / spec_confidence /
    spec_method / dimension / defect_summary / defect_boxes / annotated /
    ai_verdict / reasons。
    """
    if img_bgr is None:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

    result = {
        "image_path": None, "spec_result": None, "spec_confidence": None,
        "spec_probs": None, "spec_method": "vision_reading",
        "dimension": None, "defect_summary": None, "defect_boxes": None,
        "annotated": None, "ai_verdict": "UNSURE", "reasons": [],
    }

    # 保存原始图（追溯）
    if save_image:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        result["image_path"] = str(RAW_DIR / fname)
        cv2.imwrite(result["image_path"], img_bgr)

    # 1) 读数 + 型号
    reading, samples = _sample_reading_any(img_bgr, image_path)
    reading_failed = reading is None
    spec_inferred = None
    dist_mm = None
    if not reading_failed:
        spec_inferred, dist_mm = infer_spec_by_reading(reading)
        result["spec_result"] = spec_inferred
        result["spec_confidence"] = _make_confidence(reading, samples, dist_mm)

    # 2) 尺寸判定
    dim = judge_dimension(reading, spec_inferred, expected_spec)
    result["dimension"] = dim
    if not reading_failed:
        dim["samples"] = dict(samples)
        dim["range_mm"] = list(READING_RANGE_MM)

    # 3) 缺陷视觉判定
    defect_summary, defect_warn = inspect_defects(img_bgr, image_path)
    result["defect_summary"] = defect_summary

    # 4) 综合判定
    verdict, reasons = resolve_verdict(dim, spec_inferred, expected_spec,
                                       defect_summary, reading_failed, defect_warn)
    result["ai_verdict"] = verdict
    result["reasons"] = reasons
    return result


# ================= 图纸清单逐项校验 =================

def run_checklist_detection(images, drawing_no=None, checklist=None,
                            save_image=True, material_no=None):
    """
    按图纸检测清单逐项视觉校验（加工件模式，多角度照片）。
    images: list of (img_bgr, path) 或 list of path。至少一张。
    drawing_no / checklist: 二选一，指定清单。
    返回 dict（对齐 run_detection 基础上加 checklist 字段）：
        image_path / spec_result=None / spec_method="vision_checklist" /
        defect_summary / ai_verdict / reasons / per_item / checklist_no / material_no。
    """
    from inspection_checklist import (load_checklist, find_checklist,
                                      build_checklist_prompt, parse_item_results,
                                      resolve_checklist_verdict)

    if checklist is None:
        checklist = load_checklist(drawing_no) if drawing_no else find_checklist(material_no=material_no)
    if checklist is None:
        raise FileNotFoundError(f"未找到图纸清单: drawing_no={drawing_no} material_no={material_no}")

    # 统一 images 为 (bgr, path) 列表
    norm = []
    for item in images:
        if isinstance(item, tuple):
            norm.append(item)
        else:
            norm.append((None, item))

    # 保存原始图（多图）
    saved = []
    if save_image:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for _, path in norm:
            if path and os.path.exists(path):
                fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_checklist.jpg"
                dest = str(RAW_DIR / fname)
                import shutil
                shutil.copy(path, dest)
                saved.append(dest)

    # 角度描述（文件名或"第N张"）
    parts = []
    for k, (_, path) in enumerate(norm, 1):
        name = Path(path).stem if path else f"图{k}"
        parts.append(f"{name}")
    images_info = "、".join(parts)

    # 构造清单 prompt，注入全部项（视觉仅对 surface 类如实判定，其余要求 UNSURE）
    prompt = build_checklist_prompt(checklist, images_info=images_info)

    # 用第一张图做逐项判定（多角度聚合时仍以单图为主，缺陷类已有多角度通道）
    img_bgr, path = norm[0]
    try:
        text = _ask_img(img_bgr, path, prompt)
    except Exception as e:
        text = None
        reasons = [f"清单视觉判定异常: {e}"]
        per_item = []
        verdict = "UNSURE"
    else:
        per_item = parse_item_results(text, checklist)
        verdict, reasons = resolve_checklist_verdict(per_item)

    # defect_summary：仅汇总 surface 类缺陷（NG 项）
    defect_summary = {}
    for r in per_item:
        if r.get("visual") and r["status"] == "NG":
            defect_summary[r["label"]] = defect_summary.get(r["label"], 0) + 1

    return {
        "image_path": json.dumps(saved, ensure_ascii=False) if saved else None,
        "image_paths": saved,
        "spec_result": None,
        "spec_confidence": None,
        "spec_probs": None,
        "spec_method": "vision_checklist",
        "dimension": None,
        "defect_summary": defect_summary or None,
        "defect_boxes": None,
        "annotated": None,
        "ai_verdict": verdict,
        "reasons": reasons,
        "per_item": per_item,
        "checklist_no": checklist.get("drawing_no"),
        "checklist_name": checklist.get("part_name"),
        "material_no": material_no,
    }


def save_record(batch_id, result, inspector="", part_index=0, expected_spec=None):
    """复用 pipeline.save_record 入库（字段兼容）。checklist 结果一并写入。"""
    from detection import pipeline
    return pipeline.save_record(batch_id, result, inspector=inspector,
                                part_index=part_index, expected_spec=expected_spec,
                                checklist=result.get("per_item"))


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/real/03619a85ba6c261517c193a1d0c635b.jpg"
    expected = sys.argv[2] if len(sys.argv) > 2 else None
    r = run_vision_detection(image_path=img_path, expected_spec=expected, save_image=False)
    print("规格:", r["spec_result"], r["spec_confidence"], r["spec_method"])
    print("尺寸:", json.dumps({k: v for k, v in r["dimension"].items() if k != "samples"},
                              ensure_ascii=False))
    print("缺陷:", r["defect_summary"])
    print("判定:", r["ai_verdict"], r["reasons"])
