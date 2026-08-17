# -*- coding: utf-8 -*-
"""
OpenCV 中文路径兼容层。
cv2.imread / imwrite 在 Windows 下无法处理含中文/非 ASCII 的文件路径，
用 np.fromfile + cv2.imdecode / imencode 替代，保证真实照片（如"图纸1.jpg"）可读写。
"""
import numpy as np
import cv2


def imread(path, flags=cv2.IMREAD_COLOR):
    """读图，兼容中文路径。path 为 str 或 Path。"""
    import os
    p = os.fspath(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, img, params=None):
    """写图，兼容中文路径。"""
    import os
    from pathlib import Path
    p = os.fspath(path)
    ext = Path(p).suffix or ".png"
    ok, buf = cv2.imencode(ext, img, params if params is not None else [])
    if not ok:
        return False
    buf.tofile(p)
    return True
