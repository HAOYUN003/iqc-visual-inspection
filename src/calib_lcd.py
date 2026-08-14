# -*- coding: utf-8 -*-
"""
真实照片标定工具：定位数显卡尺 LCD 显示屏。
输出每张照片的 LCD 候选裁剪图（供人工核对读数），并尝试 OCR。
"""
import glob
import os

import cv2
import numpy as np

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ASUS\anaconda3\envs\iqc\Library\bin\tesseract.EXE"
    OCR_OK = True
except ImportError:
    OCR_OK = False


def find_bright_strip(gray):
    """全图找白色长条（尺身候选）：亮度>200 的连通域，宽高比>2.5"""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    bright = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 60000:
            continue
        ar = bw / bh if bh > 0 else 0
        ar2 = bh / bw if bw > 0 else 0
        if ar > 2.5 or ar2 > 2.5:
            if best is None or area > best[5]:
                best = (x, y, bw, bh, ar, area)
    return best


def find_lcd_in_strip(roi_gray):
    """在尺身 ROI 内找 LCD：暗色密集笔画小矩形"""
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    base = np.percentile(blur, 60)
    dark = cv2.threshold(blur, base - 50, 255, cv2.THRESH_BINARY_INV)[1]
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    cands = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 25 or bh < 18 or bw > 500 or bh > 200:
            continue
        if area < bw * bh * 0.08:
            continue
        cands.append((area, x, y, bw, bh))
    cands.sort(reverse=True)
    return cands


def ocr_roi(gray):
    """多种预处理 OCR 数字"""
    if not OCR_OK:
        return []
    res = []
    up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    up = cv2.bilateralFilter(up, 9, 75, 75)
    th1 = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th2 = cv2.adaptiveThreshold(up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    for img in [up, th1, th2, 255 - th1]:
        for psm in ["7", "8", "6"]:
            try:
                txt = pytesseract.image_to_string(
                    img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.mm")
                txt = txt.strip()
                if txt and txt not in res:
                    res.append(f"psm{psm}:{txt}")
            except Exception:
                continue
    return res


def main():
    real_dir = "data/real"
    out_dir = os.path.join(real_dir, "lcd")
    os.makedirs(out_dir, exist_ok=True)
    photos = [p for p in sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
              if not os.path.basename(p).startswith("_")]
    for p in photos:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        name = os.path.basename(p)[:16]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(f"=== {name} ({w}x{h}) ===")

        strip = find_bright_strip(gray)
        found_lcd = False
        if strip:
            x, y, bw, bh, ar, area = strip
            print(f"  尺身: ({x},{y}) {bw}x{bh} AR={ar:.1f}")
            # 尺身周围扩展区域
            x0, y0 = max(0, x - 60), max(0, y - 60)
            x1, y1 = min(w, x + bw + 60), min(h, y + bh + 60)
            roi_gray = gray[y0:y1, x0:x1]
            for cand_area, cx, cy, cbw, cbh in find_lcd_in_strip(roi_gray)[:5]:
                pad = max(15, int(cbw * 0.2))
                rx0, ry0 = max(0, cx - pad), max(0, cy - pad)
                rx1, ry1 = min(roi_gray.shape[1], cx + cbw + pad), min(roi_gray.shape[0], cy + cbh + pad)
                lroi = roi_gray[ry0:ry1, rx0:rx1]
                if lroi.size == 0:
                    continue
                txts = ocr_roi(lroi)
                fname = f"lcd_{name}_{x0 + rx0}_{y0 + ry0}.jpg"
                crop = img[y0 + ry0:y0 + ry1, x0 + rx0:x0 + rx1]
                cv2.imwrite(os.path.join(out_dir, fname), crop)
                found_lcd = True
                print(f"  LCD({x0 + rx0},{y0 + ry0},{rx1 - rx0}x{ry1 - ry0}) OCR={txts} -> {fname}")
        if not found_lcd:
            print("  未定位到 LCD")


if __name__ == "__main__":
    main()
