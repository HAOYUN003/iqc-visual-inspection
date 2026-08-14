# -*- coding: utf-8 -*-
"""批量验证 run_vision_detection 在 8 张真实照片上的表现"""
import glob
import json
import os
import sys

sys.path.insert(0, ".")
from src.detection.vision_detector import run_vision_detection

real_dir = "data/real"
photos = [p for p in sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
          if not os.path.basename(p).startswith("_")
          and "lcd" not in os.path.basename(p)
          and "head" not in os.path.basename(p)]

print(f"共 {len(photos)} 张\n")
results = []
for p in photos:
    name = os.path.basename(p)[:16]
    print(f"=== {name} ===")
    r = run_vision_detection(image_path=p, save_image=False)
    dim = r["dimension"]
    print(f"  读数={dim.get('reading_mm')}  规格={r['spec_result']}  置信度={r['spec_confidence']}")
    print(f"  尺寸={dim.get('status')}  缺陷={r['defect_summary']}  判定={r['ai_verdict']}")
    print(f"  原因={r['reasons']}")
    results.append({
        "file": name, "reading_mm": dim.get("reading_mm"),
        "spec": r["spec_result"], "conf": r["spec_confidence"],
        "dim_status": dim.get("status"), "defects": r["defect_summary"],
        "verdict": r["ai_verdict"], "reasons": r["reasons"],
    })
    print()

out = os.path.join(real_dir, "vision_detector_batch.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已保存: {out}")
