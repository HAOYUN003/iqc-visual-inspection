# -*- coding: utf-8 -*-
"""Qwen-VL 视觉问答封装：支持文件路径 / 字节 / base64 三种输入。"""
import base64
import json
import os
import sys

import urllib.request

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


def _load_key():
    """API key 优先级：环境变量 VISION_API_KEY > 本地 .mcp.json（不入库）"""
    key = os.environ.get("VISION_API_KEY")
    if key:
        return key
    mcp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".mcp.json")
    try:
        with open(mcp, "r", encoding="utf-8") as f:
            env = json.load(f)["mcpServers"]["qwen-vision"]["env"]
            return env.get("VISION_API_KEY")
    except Exception:
        return None


KEY = _load_key()


def ask_b64(b64, mime, prompt, timeout=90):
    """用 base64 图片数据问 Qwen-VL，返回回答文本。"""
    payload = {"model": "qwen-vl-max", "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}},
        {"type": "text", "text": prompt},
    ]}]}
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    return json.loads(resp.read().decode())["choices"][0]["message"]["content"]


def ask_bytes(data, name, prompt, timeout=90):
    """用图片字节问 Qwen-VL。name 用于推断图片格式。"""
    b64 = base64.b64encode(data).decode()
    mime = "png" if str(name).lower().endswith(".png") else "jpeg"
    return ask_b64(b64, mime, prompt, timeout=timeout)


def ask(img_path, prompt, timeout=90):
    """用图片文件路径问 Qwen-VL，返回回答文本。"""
    with open(img_path, "rb") as f:
        return ask_bytes(f.read(), img_path, prompt, timeout=timeout)


if __name__ == "__main__":
    print(ask(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "请描述这张图"))
