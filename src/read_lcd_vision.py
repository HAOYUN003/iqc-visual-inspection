# -*- coding: utf-8 -*-
"""
对裁剪放大的 LCD 图用 Qwen-VL 高精度读数。
"""
import base64
import glob
import json
import os
import sys

import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ask_vision import KEY

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL = "qwen-vl-plus"


def ask(image_path, prompt, timeout=60):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = "png" if image_path.lower().endswith(".png") else "jpeg"
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
    lcd_dir = os.path.join(real_dir, "lcd_crop")
    # 只读原色放大版（非二值）
    files = sorted(glob.glob(os.path.join(lcd_dir, "lcd_*_th.png")))
    prompt = (
        "这是数显卡尺显示屏的放大图。请精确读出屏幕上显示的数字（多少mm）。\n"
        "只回答数字本身，例如：9.82。如果看不清就回答 UNREADABLE。"
    )
    results = {}
    for f in files:
        name = os.path.basename(f)
        key = name.split("_")[1]
        try:
            txt = ask(f, prompt)
            results[key] = txt.strip()
            print(f"{key}: {txt.strip()}")
        except Exception as e:
            print(f"{key}: ERROR {e}")
    with open(os.path.join(real_dir, "lcd_readings.json"), "w", encoding="utf-8") as fo:
        json.dump(results, fo, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
