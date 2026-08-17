# -*- coding: utf-8 -*-
"""
检测引擎统一入口：
    local  本地免费（规格+尺寸+螺纹+本地YOLO缺陷，YOLO真实照片误检偏高）
    hybrid 混合引擎【默认推荐】：本地规格/尺寸/螺纹（免费） + 视觉大模型缺陷（0.02元/件，准）
    vision 视觉大模型（全部走 Qwen-VL）

返回格式对齐 vision_detector 的 run_vision_detection，网页可无缝切换。
字段：spec_result / spec_confidence / spec_method / dimension / defect_summary /
      defect_boxes / annotated / ai_verdict / reasons / image_path / engine。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import patch_cv_io; patch_cv_io()


def run_detection_local(img_bgr, expected_spec=None, inspector="", save_image=True,
                        material_config=None):
    """本地免费引擎：尺寸反推/CNN 规格 + 紧固件 YOLO 缺陷 + CV 尺寸 + 螺纹。
    返回 dict，engine 标记为 local。"""
    from detection import pipeline
    res = pipeline.run_detection(
        img_bgr, expected_spec=expected_spec, inspector=inspector,
        save_image=save_image, material_config=material_config,
        use_local_defect=True)
    res["engine"] = "local"
    return res


def run_detection_hybrid(img_bgr, expected_spec=None, inspector="", save_image=True,
                         material_config=None):
    """混合引擎【默认推荐】：本地规格/尺寸/螺纹（免费） + 视觉大模型缺陷（准）。
    本地 YOLO 缺陷在真实照片上误检偏高，由视觉大模型接管缺陷判定。"""
    from detection import pipeline
    from detection import vision_detector

    # 本地部分：关闭本地 YOLO 缺陷，保留规格/尺寸/螺纹
    res = pipeline.run_detection(
        img_bgr, expected_spec=expected_spec, inspector=inspector,
        save_image=save_image, material_config=material_config,
        use_local_defect=False)
    res["engine"] = "hybrid"

    # 缺陷部分：视觉大模型判定（覆盖本地空缺陷）
    defect_summary, defect_warn = vision_detector.inspect_defects(
        img_bgr, None)
    res["defect_summary"] = defect_summary
    res["defect_method"] = "vision"
    if defect_summary:
        res["ai_verdict"] = "NG"
        for cls, cnt in defect_summary.items():
            res["reasons"].append(f"{cls}×{cnt}")
    elif defect_warn:
        res["reasons"].append(defect_warn)
    else:
        if res["ai_verdict"] == "OK":
            res["reasons"].append("表面缺陷：无（视觉大模型）")
    return res


def run_detection_vision(img_bgr, expected_spec=None, save_image=True):
    """视觉大模型引擎（Qwen-VL，付费）。返回 dict，engine 标记为 vision。"""
    from detection import vision_detector
    res = vision_detector.run_vision_detection(
        img_bgr=img_bgr, expected_spec=expected_spec, save_image=save_image)
    res["engine"] = "vision"
    return res


def run(img_bgr, engine="hybrid", expected_spec=None, inspector="", save_image=True,
        material_config=None):
    """统一入口。engine: "hybrid"（默认，推荐） | "local"（全免费） | "vision"（全付费）"""
    if engine == "vision":
        return run_detection_vision(img_bgr, expected_spec=expected_spec,
                                    save_image=save_image)
    if engine == "local":
        return run_detection_local(img_bgr, expected_spec=expected_spec,
                                   inspector=inspector, save_image=save_image,
                                   material_config=material_config)
    return run_detection_hybrid(img_bgr, expected_spec=expected_spec,
                                inspector=inspector, save_image=save_image,
                                material_config=material_config)


if __name__ == "__main__":
    import cv2
    import sys as _sys
    img = cv2.imread(_sys.argv[1])
    eng = _sys.argv[2] if len(_sys.argv) > 2 else "hybrid"
    r = run(img, engine=eng)
    print(f"引擎: {r['engine']}")
    print(f"规格: {r['spec_result']} ({r.get('spec_method')}, conf={r['spec_confidence']})")
    print(f"缺陷: {r['defect_summary']} (方法: {r.get('defect_method') or r.get('spec_method')})")
    print(f"螺纹: {r.get('thread_result')}")
    print(f"判定: {r['ai_verdict']} | {r['reasons']}")
