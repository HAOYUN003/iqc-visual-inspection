# -*- coding: utf-8 -*-
"""
国标公差知识库：按图纸"未注公差"自动查表，实现复杂读图的第一步。
数据来源：
  - GB/T 1804-m（一般公差 线性尺寸 m 级）：未注线性尺寸极限偏差（mm）
  - GB/T 1804-m（倒圆半径/倒角高度）
  - GB/T 1184-K（一般几何公差 K 级）：直线度/平面度/垂直度/对称度/圆跳动

用法：
    from gb_tolerance import linear_tolerance, radius_tolerance, geometric_tolerance
    tol = linear_tolerance(15.0)      # → ±0.2（未注公差，m级）
    tol = linear_tolerance(15.0, "f") # → ±0.1（精级）
"""
from bisect import bisect_left

# ---------- GB/T 1804-m 线性尺寸未注公差（±mm） ----------
# 尺寸段边界（mm）：0~3, 3~6, 6~30, 30~120, 120~400, 400~1000, 1000~2000
_LINEAR_LIMITS = [0, 3, 6, 30, 120, 400, 1000, 2000]
# 公差等级 → 各尺寸段极限偏差（mm）
_LINEAR_TOL = {
    "f": [0.05, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],    # 精密级
    "m": [0.1, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2],       # 中等级（最常用）
    "c": [0.2, 0.2, 0.5, 0.8, 1.2, 2.0, 3.0],       # 粗糙级
    "v": [0.5, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0],       # 最粗级
}

# ---------- GB/T 1804-m 倒圆半径与倒角高度（未注公差） ----------
_RADIUS_LIMITS = [0, 0.5, 3, 6, 30, 120]
_RADIUS_TOL = {
    "f": [0.2, 0.2, 0.5, 1.0, 2.0],
    "m": [0.2, 0.5, 1.0, 2.0, 4.0],
    "c": [0.4, 1.0, 2.0, 4.0, 8.0],
    "v": [0.4, 1.0, 2.0, 4.0, 8.0],
}

# ---------- GB/T 1184-K 一般几何公差（未注） ----------
# 公差等级 K → 各尺寸段最大允许值（mm）
# 尺寸段：~10, 10~30, 30~100, 100~300, 300~1000
_GEOMETRIC_LIMITS = [0, 10, 30, 100, 300, 1000]
_GEOMETRIC_STRAIGHT_FLAT = {   # 直线度 / 平面度（K级）
    "H": [0.02, 0.05, 0.1, 0.2, 0.4],
    "K": [0.05, 0.1, 0.2, 0.4, 0.8],
    "L": [0.1, 0.2, 0.4, 0.8, 1.6],
}
_GEOMETRIC_PERP_SYMM = {       # 垂直度 / 对称度（K级，相对基准长度）
    "H": [0.2, 0.4, 0.6, 1.0, 1.6],
    "K": [0.4, 0.8, 1.2, 2.0, 3.0],
    "L": [0.6, 1.2, 2.0, 3.0, 5.0],
}
_GEOMETRIC_RUNOUT = {          # 圆跳动（K级）
    "H": [0.1, 0.2, 0.3, 0.5, 0.8],
    "K": [0.2, 0.4, 0.6, 1.0, 1.6],
    "L": [0.3, 0.6, 1.0, 1.6, 2.5],
}


def _index(limits, value):
    """返回 value 落在 limits 中的尺寸段索引（二分）。"""
    i = bisect_left(limits, value) - 1
    return max(0, min(i, len(limits) - 2))


def linear_tolerance(nominal_mm, grade="m"):
    """GB/T 1804 未注线性尺寸公差（±mm）。
    nominal_mm: 名义尺寸；grade: f/m/c/v（精密/中等/粗糙/最粗）。"""
    grade = grade.lower()
    if grade not in _LINEAR_TOL:
        grade = "m"
    idx = _index(_LINEAR_LIMITS, nominal_mm)
    return _LINEAR_TOL[grade][idx]


def radius_tolerance(nominal_mm, grade="m"):
    """GB/T 1804 未注倒圆半径/倒角高度公差（±mm）。"""
    grade = grade.lower()
    if grade not in _RADIUS_TOL:
        grade = "m"
    idx = _index(_RADIUS_LIMITS, nominal_mm)
    return _RADIUS_TOL[grade][idx]


def geometric_tolerance(nominal_mm, feature="straight", grade="K"):
    """GB/T 1184 未注几何公差（mm）。
    feature: straight(直线度/平面度) | perpendicular(垂直度/对称度) | runout(圆跳动)
    grade: H/K/L。"""
    grade = grade.upper()
    if grade not in ("H", "K", "L"):
        grade = "K"
    idx = _index(_GEOMETRIC_LIMITS, nominal_mm)
    if feature in ("straight", "flat", "直线度", "平面度"):
        return _GEOMETRIC_STRAIGHT_FLAT[grade][idx]
    if feature in ("perpendicular", "symmetry", "垂直度", "对称度"):
        return _GEOMETRIC_PERP_SYMM[grade][idx]
    if feature in ("runout", "圆跳动"):
        return _GEOMETRIC_RUNOUT[grade][idx]
    return None


def spec_label(nominal_mm, grade="m"):
    """生成判定说明文本，如 "Φ15 未注公差(GB/T1804-m) ±0.2mm"。 """
    tol = linear_tolerance(nominal_mm, grade)
    return f"名义 {nominal_mm:g}mm，未注公差(GB/T 1804-{grade}) ±{tol:g}mm"


if __name__ == "__main__":
    # 自测
    print("=== 线性尺寸公差 GB/T1804-m ===")
    for v in [2, 5, 15, 50, 200, 1500]:
        print(f"  {v}mm → ±{linear_tolerance(v):g}mm")
    print("=== 各公差等级 (15mm) ===")
    for g in "fmcv":
        print(f"  {g}级 → ±{linear_tolerance(15, g):g}mm")
    print("=== 几何公差 GB/T1184-K ===")
    for v in [5, 15, 50, 200]:
        print(f"  {v}mm → 直线度 {geometric_tolerance(v,'straight'):g} / "
              f"垂直度 {geometric_tolerance(v,'perpendicular'):g} / "
              f"圆跳动 {geometric_tolerance(v,'runout'):g}")
