# -*- coding: utf-8 -*-
"""
真实照片全自动分析：
1. 每张照片自动定位尺子（长条状物体）
2. 检测尺子上的刻度线 → 求 px_per_mm（相邻刻度 = 1mm）
3. 检测照片中的圆形零件（螺钉）→ 测像素直径
4. 换算 mm → 反推规格 + 跑缺陷检测 → 输出报告

用法：python src/real_analysis.py
"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detection import dimension_model as dm
from src.detection import pipeline


def find_ruler(img_gray):
    """定位画面中的尺子：长宽比>3 的最大前景连通域。
    返回 (中心点, 长边px, 短边px, 旋转角) 或 None"""
    h, w = img_gray.shape[:2]
    blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = otsu_inv if otsu_inv.mean() < otsu.mean() else otsu
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < 5000:
            continue
        rect = cv2.minAreaRect(c)
        bw, bh = rect[1]
        if min(bw, bh) < 1:
            continue
        ar = max(bw, bh) / min(bw, bh)
        if ar > 3 and (best is None or area > best[0]):
            best = (area, rect, bw, bh)
    if not best:
        return None
    _, rect, bw, bh = best
    (cx, cy), (rw, rh), ang = rect
    long_len = max(bw, bh)
    short_len = min(bw, bh)
    return (cx, cy, long_len, short_len, ang)


def detect_ruler_ticks(img_gray, ruler_info):
    """在尺子区域内检测刻度线，返回相邻刻度像素间距"""
    h, w = img_gray.shape[:2]
    cx, cy, long_len, short_len, ang = ruler_info
    # 旋转使尺子水平
    M = cv2.getRotationMatrix2D((cx, cy), ang if ang < 45 else ang - 90, 1.0)
    rot = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR)
    # 沿尺子长轴扫描行的明暗变化（刻度线 = 与长轴垂直的暗线）
    # 尺子在旋转后中心 (cx,cy) 不变，取中心行区域
    y0 = int(max(0, cy - short_len / 2))
    y1 = int(min(h, cy + short_len / 2))
    x0 = int(max(0, cx - long_len / 2))
    x1 = int(min(w, cx + long_len / 2))
    if y1 - y0 < 10 or x1 - x0 < 50:
        return None
    roi = rot[y0:y1, x0:x1]
    # 每列平均亮度（垂直投影）
    col_mean = roi.mean(axis=0)
    # 检测暗刻线（比平均暗 20 以上）
    base = np.median(col_mean)
    dark = col_mean < (base - 20)
    # 找连续的暗段中心（刻线位置）
    ticks = []
    in_tick = False
    for i, d in enumerate(dark):
        if d and not in_tick:
            tick_start = i
            in_tick = True
        elif not d and in_tick:
            ticks.append((tick_start + i) // 2)
            in_tick = False
    if in_tick:
        ticks.append(tick_start)
    if len(ticks) < 3:
        return None
    # 相邻刻度间距的众数
    gaps = np.diff(ticks)
    # 用中位数间距作为刻度间距
    med = np.median(gaps)
    # 1mm 刻度（最短刻线）间距
    return med


def find_screw_circles(img_gray, min_d=80, max_d=2500):
    """找照片中的圆形零件，返回 [(直径px, 中心x, 中心y, 圆度)]"""
    h, w = img_gray.shape[:2]
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    mask = otsu_inv if otsu_inv.mean() < otsu.mean() else otsu
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 3000:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        d = 2 * r
        if not (min_d <= d <= max_d):
            continue
        circ = area / (np.pi * r * r) if r > 0 else 0
        # 圆度 0.55~1.3 视为圆形零件
        if 0.55 < circ < 1.3:
            objs.append((d, int(x), int(y), circ))
    # 去重（重叠的圆保留大的）
    objs.sort(reverse=True)
    uniq = []
    for d, x, y, circ in objs:
        if all(abs(x - u[1]) > 30 or abs(y - u[2]) > 30 for u in uniq):
            uniq.append((d, x, y, circ))
    return uniq


def main():
    real_dir = Path("data/real")
    photos = sorted(glob.glob(str(real_dir / "*.jpg")))
    photos = [p for p in photos if "_" not in os.path.basename(p)]
    if not photos:
        print(f"未在 {real_dir} 找到照片")
        return

    report = []
    print(f"共 {len(photos)} 张照片，开始全自动分析\n")
    for p in photos:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        name = os.path.basename(p)[:16]
        print(f"=== {name} ===")

        # 1. 找尺子 + 刻度
        ruler = find_ruler(gray)
        px_per_mm = None
        if ruler:
            tick_gap = detect_ruler_ticks(gray, ruler)
            if tick_gap and tick_gap > 5:
                # 尺子刻度最小间距 = 0.5mm 或 1mm，取 1mm 假设（常用）
                px_per_mm = tick_gap
                print(f"  尺子: 长{ruler[2]:.0f}px, 刻度间距={tick_gap:.1f}px → px_per_mm≈{px_per_mm:.1f}")
        if px_per_mm is None:
            print(f"  [警告] 未检测到尺子刻度，跳过 mm 换算")

        # 2. 找螺钉圆
        circles = find_screw_circles(gray)
        if not circles:
            print("  [提示] 未检测到圆形零件")
        for d, x, y, circ in circles[:4]:
            mm = d / px_per_mm if px_per_mm else None
            print(f"  圆: 直径{d:.0f}px" + (f" = {mm:.2f}mm" if mm else ""))

        # 3. 跑完整 pipeline（用默认标定）
        res = pipeline.run_detection(img, save_image=False)
        dim = res.get("dimension")
        spec = res.get("spec_result")
        print(f"  pipeline: 规格={spec} 类型={dim.get('part_type') if dim else '?'} 缺陷={res.get('defect_summary')} 判定={res.get('ai_verdict')}")
        report.append({"file": name, "spec": spec, "defects": res.get("defect_summary"),
                       "verdict": res.get("ai_verdict"), "px_per_mm": px_per_mm,
                       "circles_px": [round(d) for d, *_ in circles[:4]]})
        print()

    # 写报告
    out = real_dir / "analysis_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main()
