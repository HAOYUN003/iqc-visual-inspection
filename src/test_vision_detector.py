# -*- coding: utf-8 -*-
"""冒烟测试 run_vision_detection（写文件方式避免内联 key 审查）"""
import json
import sys

sys.path.insert(0, ".")
from src.detection.vision_detector import run_vision_detection

img_path = sys.argv[1] if len(sys.argv) > 1 else "data/real/03619a85ba6c261517c193a1d0c635b.jpg"
expected = sys.argv[2] if len(sys.argv) > 2 else None

r = run_vision_detection(image_path=img_path, expected_spec=expected, save_image=False)
print("规格:", r["spec_result"], r["spec_confidence"], r["spec_method"])
dim = {k: v for k, v in r["dimension"].items() if k != "samples"}
print("尺寸:", json.dumps(dim, ensure_ascii=False))
print("缺陷:", r["defect_summary"])
print("判定:", r["ai_verdict"], r["reasons"])
