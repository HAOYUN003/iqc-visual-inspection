# -*- coding: utf-8 -*-
"""
卡尺 LCD 显示屏精确定位 + 裁剪放大。
数显卡尺 LCD 是卡尺主体上的一块暗色矩形屏（白色/浅色七段数字）。
本脚本用形态学定位 LCD，裁剪 + 放大 + 增强，输出供视觉模型高精度读数。
"""
import glob
import os
from config import patch_cv_io; patch_cv_io()


import cv2
import numpy as np


def locate_lcd(gray, name):
    """定位 LCD 显示屏矩形区域，返回 (x, y, w, h) 或 None"""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 数显卡尺 LCD：通常是画面中对比度最高的矩形暗块
    # 策略1：Otsu 二值找暗块
    candidates = []

    # 自适应阈值找暗块（LCD 外壳通常比周围金属暗）
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 51, -8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # LCD 比例：宽约30-150px(全图), 高约15-80px，矩形
        if not (30 <= bw <= 500 and 15 <= bh <= 250):
            continue
        if bw < 2 * bh or bh > 2 * bw:  # 不要太极端
            continue
        if area < bw * bh * 0.25:  # 需要有一定填充
            continue
        candidates.append((area, x, y, bw, bh))

    if not candidates:
        return None
    # 取最大的候选（LCD 是卡尺上最大的暗矩形块）
    candidates.sort(reverse=True)
    return candidates[0][1:]


def main():
    real_dir = "data/real"
    out_dir = os.path.join(real_dir, "lcd_crop")
    os.makedirs(out_dir, exist_ok=True)
    photos = [p for p in sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
              if not os.path.basename(p).startswith("_")]
    for p in photos:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        name = os.path.basename(p)[:16]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        loc = locate_lcd(gray, name)
        if loc:
            x, y, bw, bh = loc
            # 扩展一点包围屏幕
            pad_x, pad_y = int(bw * 0.15), int(bh * 0.25)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            crop = img[y0:y1, x0:x1]
            # 放大 4x 供精确读数
            big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            # 锐化增强数字
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharp = cv2.filter2D(big, -1, kernel)
            fname = f"lcd_{name}_{x0}_{y0}.png"
            cv2.imwrite(os.path.join(out_dir, fname), sharp,
                        [cv2.IMWRITE_PNG_COMPRESSION, 6])
            # 二值化版本
            g = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
            th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cv2.imwrite(os.path.join(out_dir, f"lcd_{name}_th.png"), th)
            print(f"{name}: LCD=({x},{y},{bw}x{bh}) 裁剪→{fname}")
        else:
            print(f"{name}: 未定位到 LCD")


if __name__ == "__main__":
    main()
