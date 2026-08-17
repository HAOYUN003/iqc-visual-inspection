# -*- coding: utf-8 -*-
"""
精度对比：本地免费引擎 vs 视觉大模型 API（真实照片批量跑）
用法：python src/bench_engine.py [--limit N] [--vision]

输出：每张照片的规格识别/缺陷/判定 + 汇总准确率。
- 本地引擎（默认，免费）：尺寸反推 / CNN / YOLO 缺陷
- --vision：额外调 Qwen-VL（付费）对比，需 API key
"""
import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, SPEC_NOMINAL_HEAD_MM
from config import patch_cv_io; patch_cv_io()



def collect_photos(limit=None):
    """收集真实照片，排除中间产物（_开头/lcd/head）。"""
    real_dir = DATA_DIR / "real"
    photos = sorted(p for p in real_dir.glob("*.jpg")
                    if not p.name.startswith("_")
                    and "lcd" not in p.name and "head" not in p.name)
    return photos[:limit] if limit else photos


def run_local(img):
    """本地免费引擎。返回 (spec, method, conf, defects, verdict, reasons)。"""
    from detection import engine
    r = engine.run_detection_local(img, save_image=False)
    return (r["spec_result"], r.get("spec_method"), r["spec_confidence"],
            r["defect_summary"], r["ai_verdict"], r["reasons"])


def run_vision(img):
    """视觉大模型（付费）。返回 (spec, method, conf, defects, verdict, reasons)。"""
    from detection import engine
    r = engine.run_detection_vision(img, save_image=False)
    return (r["spec_result"], r.get("spec_method"), r["spec_confidence"],
            r["defect_summary"], r["ai_verdict"], r["reasons"])


def summarize(rows):
    """汇总：规格可识别率、缺陷检出数、判定分布。"""
    import cv2
    from collections import Counter
    specs = [r["spec"] for r in rows if r["spec"] and r["spec"] not in ("UNKNOWN",)]
    n_total = len(rows)
    n_spec = len(specs)
    verdicts = Counter(r["verdict"] for r in rows)
    n_defect = sum(1 for r in rows if r["defects"])
    return {
        "total": n_total,
        "spec_recognized": n_spec,
        "spec_rate": n_spec / n_total if n_total else 0,
        "spec_dist": Counter(r["spec"] for r in rows if r["spec"]),
        "verdicts": dict(verdicts),
        "defect_count": n_defect,
        "defect_dist": Counter(
            cls for r in rows if r["defects"] for cls in r["defects"]),
    }


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--vision", action="store_true", help="同时跑视觉API对比（付费）")
    args = ap.parse_args()

    photos = collect_photos(args.limit)
    print(f"共 {len(photos)} 张真实照片\n")

    rows = []
    for p in photos:
        img = cv2.imread(str(p))
        if img is None:
            continue
        spec, method, conf, defects, verdict, reasons = run_local(img)
        rows.append({"file": p.name, "spec": spec, "method": method, "conf": conf,
                     "defects": defects, "verdict": verdict, "reasons": reasons})
        print(f"{p.name[:20]:22} 规格={spec} ({method}/{round(conf,2) if conf else '-'}) "
              f"缺陷={defects} 判定={verdict}")

    s = summarize(rows)
    print("\n=== 本地引擎汇总 ===")
    print(f"规格可识别率: {s['spec_rate']:.0%} ({s['spec_recognized']}/{s['total']})")
    print(f"规格分布: {dict(s['spec_dist'])}")
    print(f"判定分布: {s['verdicts']}")
    print(f"缺陷检出: {s['defect_count']} 张 | 分布: {dict(s['defect_dist'])}")

    if args.vision:
        print("\n=== 视觉API对比（付费）===")
        vrows = []
        for p in photos:
            img = cv2.imread(str(p))
            if img is None:
                continue
            spec, method, conf, defects, verdict, reasons = run_vision(img)
            vrows.append({"file": p.name, "spec": spec, "method": method,
                          "defects": defects, "verdict": verdict})
            print(f"{p.name[:20]:22} 规格={spec} 缺陷={defects} 判定={verdict}")
        vs = summarize(vrows)
        print(f"规格可识别率: {vs['spec_rate']:.0%} ({vs['spec_recognized']}/{vs['total']})")
        print(f"判定分布: {vs['verdicts']}")
        print(f"缺陷检出: {vs['defect_count']} 张 | 分布: {dict(vs['defect_dist'])}")


if __name__ == "__main__":
    main()
