# -*- coding: utf-8 -*-
"""
尺寸测量模块（传统 CV，不训练模型）
从标准件俯视图测量关键直径，判定是否超差（尺寸不良 ③）。

流程：
    Otsu 分割 → 最大轮廓 → 判型（螺钉/垫片/螺母）→ 亚像素径向边缘圆拟合
    → 像素×标定系数=mm → 对照名义值±公差 → OK/NG

遵循光学方案设计口径：尺寸走光学测量（传统CV），AI 不做像素级猜测。
Phase 0 用模拟生成器的 px_per_mm；Phase 1 用标定板实测值。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (SPEC_NOMINAL_HEAD_MM, SPEC_HEAD_TOLERANCE_MM,
                    CALIB_PX_PER_MM)


# ================= 判型 =================

def _get_part_components(component_mask):
    """返回 (外轮廓, 孔轮廓列表)。用层级找外边界+内孔"""
    contours, hier = cv2.findContours(component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer = max(contours, key=cv2.contourArea) if contours else None
    holes = []
    if outer is not None and hier is not None:
        # hier: [next, prev, first_child, parent]
        for i, h in enumerate(hier[0]):
            if h[3] != -1:   # 有父轮廓 → 是孔
                area = cv2.contourArea(contours[i])
                if area > 100:
                    holes.append(contours[i])
    return outer, holes


def _detect_part_type(mask, outer_contour, holes):
    """根据外轮廓形状 + 孔特征判型：screw / washer / nut / unknown"""
    if outer_contour is None:
        return "unknown"
    area = cv2.contourArea(outer_contour)
    if area <= 0:
        return "unknown"
    # 圆度：外接圆面积 vs 实际面积（圆→接近1；六角→略低）
    (x, y), r = cv2.minEnclosingCircle(outer_contour)
    circularity = area / (np.pi * r * r) if r > 0 else 0
    # 顶点数：轮廓近似后的角点
    peri = cv2.arcLength(outer_contour, True)
    approx = cv2.approxPolyDP(outer_contour, 0.02 * peri, True)
    n_verts = len(approx)
    # 孔：用最大内孔面积占比（washer 孔≈外径×0.42，占面积≈17%；screw 六角孔≈外径×0.45）
    hole_area = max((cv2.contourArea(h) for h in holes), default=0)
    hole_ratio = hole_area / area if area > 0 else 0

    if n_verts == 6 and 0.55 < circularity < 0.98:
        return "nut"               # 六角轮廓 → 螺母（可带小孔）
    if hole_ratio > 0.06 and circularity > 0.85:
        return "washer"            # 高圆度 + 明显中心通孔 → 垫片
    if circularity > 0.85:
        return "screw"             # 高圆度 + 无大孔 → 螺钉头
    return "unknown"


# ================= 亚像素边缘 =================

def _subpixel_edge_along_ray(gray, cx, cy, theta, search_r):
    """沿角度 theta 的射线，在外圆附近找亚像素边缘点（灰度梯度峰值）"""
    # 采样点（覆盖边缘附近）
    px = cx + search_r * np.cos(theta)
    py = cy + search_r * np.sin(theta)
    px2 = cx + (search_r + 15) * np.cos(theta)
    py2 = cy + (search_r + 15) * np.sin(theta)
    # 沿射线在两点间插值采样
    x = np.linspace(px, px2, 40)
    y = np.linspace(py, py2, 40)
    vals = np.array([_bilinear(gray, x[i], y[i]) for i in range(40)])
    # 灰度梯度峰值（边缘处梯度最大）
    grad = np.abs(np.gradient(vals))
    k = int(np.argmax(grad))
    return x[k], y[k]


def _bilinear(gray, x, y):
    x0, y0 = int(x), int(y)
    if x0 < 0 or y0 < 0 or x0 >= gray.shape[1] - 1 or y0 >= gray.shape[0] - 1:
        return 0
    fx, fy = x - x0, y - y0
    p00 = gray[y0, x0]; p10 = gray[y0, x0 + 1]
    p01 = gray[y0 + 1, x0]; p11 = gray[y0 + 1, x0 + 1]
    return (p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy)
            + p01 * (1 - fx) * fy + p11 * fx * fy)


def _fit_circle_radius(gray, cx, cy, approx_r, n_rays=64):
    """亚像素径向圆拟合：沿 n_rays 条法线找边缘点，代数最小二乘拟合圆半径"""
    pts = []
    for i in range(n_rays):
        theta = 2 * np.pi * i / n_rays
        x, y = _subpixel_edge_along_ray(gray, cx, cy, theta, approx_r - 5)
        pts.append([x, y])
    pts = np.array(pts)
    # 去中心化提高数值稳定性，再代数拟合圆：u²+v² = 2uc·u + 2vc·v + c
    xm, ym = pts.mean(axis=0)
    u, v = pts[:, 0] - xm, pts[:, 1] - ym
    A = np.column_stack([2 * u, 2 * v, np.ones(len(u))])
    b = u ** 2 + v ** 2
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    uc, vc, _ = sol
    # 从拟合中心直接算到各边缘点距离的均值 = 半径（避免代数拟合的负值根号问题）
    dist = np.sqrt((u - uc) ** 2 + (v - vc) ** 2)
    rc = float(dist.mean())
    return rc, float(dist.std())


# ================= 主测量 =================

def measure_dimensions(img_bgr, px_per_mm=None, expected_spec=None):
    """
    测量标准件俯视图的直径。
    返回 dict：
        part_type, outer_diam_px, outer_diam_mm, inner_diam_mm (垫片孔径/螺母内径),
        status (OK/NG), nominal_mm, tolerance_mm, overlay (标注图)
    """
    px_per_mm = px_per_mm or CALIB_PX_PER_MM
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 固定阈值分割：模拟图零件灰度 100+，黑背景（<30 轻微噪声抬亮）
    # 用 Otsu 在"背景占绝大比例"时阈值会漂高导致二值破碎，固定阈值更稳
    _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    # 取最大连通域（零件主体）
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return {"status": "UNKNOWN", "reason": "未检测到零件", "part_type": "unknown",
                "overlay": img_bgr.copy()}
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) )
    big_area = stats[big, cv2.CC_STAT_AREA]
    if big_area < 500:
        return {"status": "UNKNOWN", "reason": "零件区域过小", "part_type": "unknown",
                "overlay": img_bgr.copy()}
    component_mask = (labels == big).astype(np.uint8) * 255
    outer, holes = _get_part_components(component_mask)
    if outer is None:
        return {"status": "UNKNOWN", "reason": "未检测到零件轮廓", "part_type": "unknown",
                "overlay": img_bgr.copy()}
    part_type = _detect_part_type(component_mask, outer, holes)

    # 中心
    M = cv2.moments(outer)
    cx = M["m10"] / M["m00"] if M["m00"] else mask.shape[1] / 2
    cy = M["m01"] / M["m00"] if M["m00"] else mask.shape[0] / 2
    approx_r = np.sqrt(cv2.contourArea(outer) / np.pi)

    overlay = img_bgr.copy()
    result = {"part_type": part_type, "overlay": overlay}

    if part_type == "nut":
        # 外六角外接圆直径（外径）+ 内孔
        result.update(_measure_hex(img_bgr, gray, outer, holes, cx, cy, px_per_mm,
                                   expected_spec))
    elif part_type in ("screw", "washer"):
        # 外圆亚像素拟合
        rr, unc = _fit_circle_radius(gray, cx, cy, approx_r)
        outer_diam_px = 2 * rr
        outer_diam_mm = outer_diam_px / px_per_mm
        # 内孔（垫片孔径 / 螺钉内六角近似）：用层级孔轮廓
        inner_mm = None
        if holes:
            big_in = max(holes, key=cv2.contourArea)
            if cv2.contourArea(big_in) > 200:
                (x0, y0), ri = cv2.minEnclosingCircle(big_in)
                inner_mm = 2 * ri / px_per_mm
        result.update({
            "outer_diam_px": outer_diam_px, "outer_diam_mm": outer_diam_mm,
            "measure_uncertainty_mm": unc / px_per_mm,
            "inner_diam_mm": inner_mm,
        })
        # overlay：画外圆
        cv2.circle(overlay, (int(cx), int(cy)), int(rr), (0, 0, 255), 2)
        # 判定
        result.update(_judge_dimension(expected_spec, part_type, outer_diam_mm))
    else:
        result["status"] = "UNKNOWN"
        result["reason"] = "无法判型"
    return result


def _measure_hex(img_bgr, gray, outer, holes, cx, cy, px_per_mm, expected_spec=None):
    """螺母：测量外接圆直径（外径）+ 中心孔径。
    真值口径 = 外接圆直径（= HEAD_DIAM），对边距不是目标，直接测外接圆。"""
    # 外接圆直径 = 2 × max(轮廓点到中心距离)
    pts = outer.reshape(-1, 2).astype(float)
    dists = np.sqrt(((pts - [cx, cy]) ** 2).sum(axis=1))
    outer_diam_px = 2 * float(dists.max())
    outer_diam_mm = outer_diam_px / px_per_mm
    # 中心孔
    inner_mm = None
    if holes:
        big_in = max(holes, key=cv2.contourArea)
        if cv2.contourArea(big_in) > 100:
            (x0, y0), ri = cv2.minEnclosingCircle(big_in)
            inner_mm = 2 * ri / px_per_mm
    overlay = img_bgr.copy()
    cv2.circle(overlay, (int(cx), int(cy)), int(dists.max()), (0, 0, 255), 2)
    result = {
        "outer_diam_px": outer_diam_px, "outer_diam_mm": outer_diam_mm,
        "inner_diam_mm": inner_mm, "measure_uncertainty_mm": float(dists.std()),
    }
    result.update(_judge_dimension(expected_spec, "nut", outer_diam_mm))
    return result


def _judge_dimension(expected_spec, part_type, measured_mm):
    """对照名义值±公差判定 OK/NG。无期望规格时只返回测量值。"""
    if expected_spec and expected_spec in SPEC_NOMINAL_HEAD_MM:
        # 名义外径：washer 外径 = 头径×1.25，screw/nut 外径 = 头径
        nominal = SPEC_NOMINAL_HEAD_MM[expected_spec] * (1.25 if part_type == "washer" else 1)
        tol = SPEC_HEAD_TOLERANCE_MM
        if abs(measured_mm - nominal) <= tol:
            status = "OK"
            reason = f"外径 {measured_mm:.3f}mm 在 {nominal}±{tol} 内"
        else:
            status = "NG"
            reason = (f"尺寸超差: 实测 {measured_mm:.3f}mm, 标准 {nominal}±{tol}mm")
        return {"status": status, "reason": reason,
                "nominal_mm": nominal, "tolerance_mm": tol}
    return {"status": "N/A", "reason": "未提供期望规格，仅测量",
            "nominal_mm": None, "tolerance_mm": None}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("img")
    ap.add_argument("--spec", default=None, help="期望规格 M3~M8，用于公差判定")
    ap.add_argument("--px_per_mm", type=float, default=None)
    args = ap.parse_args()
    import json
    img = cv2.imread(args.img)
    res = measure_dimensions(img, px_per_mm=args.px_per_mm, expected_spec=args.spec)
    # 移除 overlay 避免打印太长
    out = {k: v for k, v in res.items() if k != "overlay"}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
