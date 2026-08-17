# -*- coding: utf-8 -*-
"""
检测流水线：统一的"拍照/上传 → 检测 → 判定 → 入库"入口
被 Streamlit 界面调用，也可命令行/脚本直接调用。

流程：
    input(BGR or 路径) → 规格识别(标准件防错) → 缺陷检测(加工件)
                      → 判定 OK/NG/UNSURE → 生成结果 → 写入质检数据库
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (RAW_DIR, SPEC_CONF_THRESH, DEFECT_CONF_THRESH,
                    FASTENER_DEFECT_CONF_THRESH)
from data import quality_db as db
from detection import spec_model, fastener_defect_model, thread_model
from config import patch_cv_io; patch_cv_io()


# 模型懒加载：界面启动时避免重复加载
_SPEC = {"model": None, "classes": None}
_FASTENER_DEFECT = {"model": None}
_THREAD = {"model": None, "classes": None}


def load_models():
    """加载规格识别 + 紧固件缺陷 + 螺纹检测模型（懒加载缓存）"""
    device = "cuda" if _has_gpu() else "cpu"
    if _SPEC["model"] is None:
        try:
            _SPEC["model"], _SPEC["classes"], _ = spec_model.load_spec_model(device)
        except FileNotFoundError:
            print("[pipeline] 规格模型未训练，规格防错跳过")
    if _FASTENER_DEFECT["model"] is None:
        try:
            _FASTENER_DEFECT["model"] = fastener_defect_model.load_fastener_defect_model(device)
        except FileNotFoundError as e:
            print(f"[pipeline] 紧固件缺陷模型未训练，表面缺陷检测跳过: {e}")
    if _THREAD["model"] is None:
        try:
            _THREAD["model"], _THREAD["classes"] = thread_model.load_thread_model(device)
        except FileNotFoundError as e:
            print(f"[pipeline] 螺纹模型未训练，螺纹检测跳过: {e}")
    return device


def _has_gpu():
    import torch
    return torch.cuda.is_available()


def run_detection(img_bgr, material_no=None, inspector="", save_image=True,
                  expected_spec=None, material_config=None, use_local_defect=True):
    """
    执行一次完整检测。
    material_config: dict, 物料白名单配置（来自 quality_db.get_material）。
        控制检测项启停：ai_enabled / check_spec / check_dim / check_defect。
        未传或 ai_enabled=0 时，只保存图片，不做 AI 判定。
    use_local_defect: 是否启用本地 YOLO 缺陷检测。
        本地缺陷模型在真实照片上误检偏高，默认关闭时由视觉大模型接管缺陷检测。
    返回 dict：
        image_path, spec_result, spec_confidence, spec_probs,
        dimension, defect_summary, defect_boxes, annotated,
        ai_verdict, reasons, checks, thread_result
    """
    import cv2
    load_models()

    # 白名单开关：默认全部启用；未启用 AI 的物料直接跳过 AI 检测
    cfg = material_config or {}
    ai_enabled = bool(cfg.get("ai_enabled", 1))
    check_spec = bool(cfg.get("check_spec", 1))
    check_dim = bool(cfg.get("check_dim", 1))
    check_defect = bool(cfg.get("check_defect", 1))

    # 保存原始图（用于追溯）
    image_path = None
    if save_image:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        import datetime
        fname = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        image_path = str(RAW_DIR / fname)
        cv2.imwrite(image_path, img_bgr)

    result = {
        "image_path": image_path,
        "spec_result": None, "spec_confidence": None, "spec_probs": None,
        "dimension": None,
        "defect_summary": None, "defect_boxes": None, "annotated": None,
        "ai_verdict": "OK", "reasons": [],
        "checks": {"spec": check_spec, "dim": check_dim, "defect": check_defect,
                   "ai_enabled": ai_enabled},
    }

    if not ai_enabled:
        result["ai_verdict"] = "SKIP"
        result["reasons"].append("物料未启用 AI 检测（白名单外/停用）")
        return result

    # 1) 尺寸测量 + 规格识别（仅当配置启用）
    # 主通道：传统 CV 判型 + 测外径 → 反推规格（确定性、可解释、无需训练）
    # 辅助：CNN 分类（仅在尺寸反推失败时兜底）
    if check_dim or check_spec:
        from detection import dimension_model
        dim = dimension_model.measure_dimensions(img_bgr)
        result["dimension"] = dim
        if dim.get("outer_diam_mm") is not None:
            inferred, dist = dimension_model.infer_spec(
                dim.get("part_type"), dim["outer_diam_mm"])
            if check_spec and inferred is not None:
                result["spec_result"] = inferred
                result["spec_confidence"] = max(0.0, 1.0 - dist / 1.5)   # 偏差越大置信度越低
                result["spec_method"] = "dimension"
            # 尺寸超差判定：以反推规格为名义值对照（仅当配置启用尺寸检测）
            if check_dim:
                judge_spec = expected_spec if expected_spec else inferred
                dim_judge = dimension_model._judge_dimension(
                    judge_spec, dim.get("part_type"), dim["outer_diam_mm"])
                dim.update(judge_spec=judge_spec)
                if dim_judge.get("status") == "NG":
                    result["ai_verdict"] = "NG"
                    result["reasons"].append(dim_judge.get("reason", "尺寸超差"))
        else:
            result["reasons"].append(dim.get("reason", "尺寸测量失败"))

    # 辅助通道：CNN 分类兜底（仅当规格检测启用且尺寸反推失败时）
    if check_spec and result["spec_result"] is None and _SPEC["model"] is not None:
        spec, conf, probs = spec_model.predict_spec(
            _SPEC["model"], _SPEC["classes"], img_bgr,
            device=_SPEC["model"].device if hasattr(_SPEC["model"], "device") else "cpu",
            conf_thresh=SPEC_CONF_THRESH)
        result["spec_result"] = spec
        result["spec_confidence"] = conf
        result["spec_probs"] = probs
        result["spec_method"] = "cnn"
        if spec == "UNKNOWN":
            result["reasons"].append("规格识别置信度不足")

    # 规格防错：期望规格已知且与识别结果不一致 → NG
    if expected_spec and result["spec_result"] and result["spec_result"] != expected_spec:
        result["ai_verdict"] = "NG"
        result["reasons"].append(f"规格不符: 期望{expected_spec}, 识别{result['spec_result']}")

    # 3) 表面缺陷检测（紧固件域 YOLO，仅当配置启用且未由视觉大模型接管时）
    if check_defect and use_local_defect and _FASTENER_DEFECT["model"] is not None:
        summary, boxes, annotated = fastener_defect_model.predict_fastener_defects(
            _FASTENER_DEFECT["model"], img_bgr,
            conf_thresh=FASTENER_DEFECT_CONF_THRESH)
        result["defect_summary"] = summary
        result["defect_boxes"] = boxes
        result["annotated"] = annotated
        if summary:
            result["ai_verdict"] = "NG"
            for cls, cnt in summary.items():
                result["reasons"].append(f"{cls}×{cnt}")
        else:
            result["reasons"].append("无可见缺陷（本地模型）")

    # 4) 螺纹状态检测（侧视图，可选）
    if check_defect and _THREAD["model"] is not None:
        state, tconf, tprobs = thread_model.predict_thread(
            _THREAD["model"], _THREAD["classes"], img_bgr,
            device=next(_THREAD["model"].parameters()).device)
        result["thread_result"] = state
        result["thread_confidence"] = tconf
        if state in ("missing", "broken"):
            result["ai_verdict"] = "NG"
            result["reasons"].append(f"螺纹{state}（置信度 {tconf:.2f}）")

    return result


def save_record(batch_id, result, inspector="", part_index=0, expected_spec=None,
                checklist=None):
    """
    把检测结果写入质检数据库。
    expected_spec: 料单/图纸期望规格，用于"规格防错"比对（不一致 → NG）
    checklist: 图纸清单逐项校验结果（JSON 数组），随记录入库
    返回 (record_id, ai_verdict)
    """
    verdict = result["ai_verdict"]
    # 规格防错：期望规格已知且与识别结果不一致 → NG（run_detection 已加入原因）
    if expected_spec and result["spec_result"] and result["spec_result"] != expected_spec \
            and not any("规格不符" in r for r in result["reasons"]):
        verdict = "NG"
        result["reasons"].append(f"规格不符: 期望{expected_spec}, 识别{result['spec_result']}")

    record_id = db.add_record(
        batch_id=batch_id,
        image_path=result.get("image_path"),
        spec_result=result.get("spec_result"),
        spec_confidence=result.get("spec_confidence"),
        defect_result=result.get("defect_summary"),
        defect_boxes=result.get("defect_boxes"),
        dimension=result.get("dimension"),
        thread_result=result.get("thread_result"),
        ai_verdict=verdict,
        inspector=inspector,
        part_index=part_index,
        checklist=checklist,
    )
    return record_id, verdict


if __name__ == "__main__":
    import cv2
    img = cv2.imread(sys.argv[1])
    res = run_detection(img)
    print("规格:", res["spec_result"], round(res["spec_confidence"], 3) if res["spec_confidence"] else "")
    print("缺陷:", res["defect_summary"])
    print("判定:", res["ai_verdict"], res["reasons"])
