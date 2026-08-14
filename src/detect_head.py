# -*- coding: utf-8 -*-
"""
真实照片螺钉头像素直径检测。
俯视图螺钉头应为圆形，用 Canny+Hough 在中央 ROI 检测，
并叠加形态学+圆度验证。输出每张照片的候选及直径。
"""
import glob
import os

import cv2
import numpy as np


def detect_head(gray):
    """检测中央区域螺钉头圆形，返回 (直径px, 中心x, 中心y, 圆度, 边缘密度) 或 None"""
    h, w = gray.shape
    y0, y1 = int(h * 0.22), int(h * 0.82)
    x0, x1 = int(w * 0.10), int(w * 0.92)
    roi = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi, (5, 5), 0)

    best = None
    for (t1, t2) in [(30, 80), (50, 120), (60, 150), (80, 200)]:
        edges = cv2.Canny(blur, t1, t2)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=400,
                                   param1=200, param2=60,
                                   minRadius=int(min(roi.shape) * 0.06),
                                   maxRadius=int(min(roi.shape) * 0.45))
        if circles is None:
            continue
        for c in circles[0]:
            cx, cy, r = c
            if not (0.3 < cy / roi.shape[0] < 0.85):
                continue
            # 边缘密度验证：圆环附近边缘像素占比
            x0i, y0i = max(0, int(cx - r - 8)), max(0, int(cy - r - 8))
            x1i, y1i = min(roi.shape[1], int(cx + r + 8)), min(roi.shape[0], int(cy + r + 8))
            if x1i - x0i < 20 or y1i - y0i < 20:
                continue
            ring = edges[y0i:y1i, x0i:x1i]
            dens = (ring > 0).mean()
            # 圆内部暗（螺钉头通常比背景暗）
            inner = blur[y0i:y1i, x0i:x1i]
            inner_mean = inner.mean()
            # 评分：边缘密度高 + 内部偏暗
            score = dens + (0 if inner_mean < np.percentile(roi, 50) else -0.3)
            if best is None or score > best[0]:
                best = (score, 2 * r, int(x0 + cx), int(y0 + cy), r, dens, inner_mean)
    return best


def main():
    real_dir = "data/real"
    out_dir = os.path.join(real_dir, "roi")
    os.makedirs(out_dir, exist_ok=True)
    photos = [p for p in sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
              if not os.path.basename(p).startswith("_")]
    results = {}
    for p in photos:
        img = cv2.imread(p)
        name = os.path.basename(p)[:16]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best = detect_head(gray)
        if best:
            score, diam, cx, cy, r, dens, inner = best
            print(f"{name}: 螺钉头 直径={diam:.0f}px 中心=({cx},{cy}) 圆度分={score:.3f} 边缘密度={dens:.3f} 内部亮度={inner:.0f}")
            cv2.circle(img, (cx, cy), int(r), (0, 0, 255), 4)
            cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(img, f"d={diam:.0f}px", (cx - int(r), cy - int(r) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
            results[name] = {"diam_px": round(diam), "cx": cx, "cy": cy}
        else:
            print(f"{name}: 未检测到螺钉头")
        cv2.imwrite(os.path.join(out_dir, f"head_{name}.jpg"), img)

    import json
    with open(os.path.join(real_dir, "head_diam_px.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n直径已保存到 data/real/head_diam_px.json")


if __name__ == "__main__":
    main()
