# -*- coding: utf-8 -*-
"""
批量视觉分析真实照片：读出卡尺读数 + 描述测量部位/被测物。
用法: python src/vision_analyze.py [照片序号, 默认全部]
"""
import base64
import json
import os
import sys
import glob

import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask_vision import KEY

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-vl-max"


def ask(image_path, prompt, timeout=90):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = "jpeg" if image_path.lower().endswith(".jpg") else "png"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    }
    req = urllib.request.Request(
        BASE, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    data = json.loads(resp.read().decode())
    return data["choices"][0]["message"]["content"]


def main():
    real_dir = "data/real"
    photos = [p for p in sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
              if not os.path.basename(p).startswith("_")]
    idx = sys.argv[1] if len(sys.argv) > 1 else None
    if idx is not None:
        photos = [photos[int(idx) - 1]]
    prompt = (
        "这是一张用数显卡尺测量零件的照片。请精确回答：\n"
        "1) LCD 显示屏上的读数是多少 mm？（务必看清小数点位置，例如 9.82 还是 98.2）\n"
        "2) 卡尺测量的是什么部位？是外径/直径、对边宽度、还是长度？\n"
        "3) 被测物整体是什么？形状、颜色、大概类型（螺钉/螺栓/螺母/垫片/其它）\n"
        "4) 画面里除了卡尺和被测物还有什么？\n"
    )
    for p in photos:
        name = os.path.basename(p)[:16]
        print(f"===== {name} =====")
        try:
            print(ask(p, prompt))
        except Exception as e:
            print(f"ERROR: {e}")
        print()


if __name__ == "__main__":
    main()
