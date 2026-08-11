# -*- coding: utf-8 -*-
"""
螺纹状态侧视图模拟数据集生成器
画螺钉侧视图（头部 + 杆部 + 三角牙螺纹），生成三类：
    good    : 正常牙
    missing : 缺牙（1~2 个牙位缺牙，该处光轴裸露）
    broken  : 烂牙（连续 3~6 牙挤压变形：牙高压缩、牙尖噪声、个别歪斜）

设计说明：
- 俯视图（圆形法兰+内六角孔）里螺纹不可见（螺钉螺纹在杆部、螺母在孔内），
  因此螺纹检测必须用侧视图。这是本模块存在的根本原因。
- 螺纹尺寸参考 ISO 公制螺纹（M3~M8）：牙距 pitch 与牙高按公称直径缩放。
- 不做随机旋转（螺纹有方向性），增广仅亮度/对比度/轻噪声。
- 目录按类组织（复用 SpecDataset 的按类子目录约定）：
  {train,val}/{good,missing,broken}/*.png
  → Phase 1 换真实侧视图，仅替换目录内容即可。
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, THREAD_CLASSES

SIZE = 224
N_TRAIN = 300
N_VAL = 80
# 各规格公称直径 mm 与牙距 mm（ISO 粗牙近似）
SPEC_PITCH_MM = {"M3": 0.5, "M4": 0.7, "M5": 0.8, "M6": 1.0, "M8": 1.25}
SPEC_D_NOM = {"M3": 3.0, "M4": 4.0, "M5": 5.0, "M6": 6.0, "M8": 8.0}
PX_PER_MM = 14.0   # 比俯视图(8.5)更高，让螺纹/缺牙特征在224px内更明显


# ================= 基础绘制 =================

def _draw_screw_side(spec, px_per_mm=PX_PER_MM, size=SIZE, state="good", rng=None):
    """画一颗螺钉侧视图，state: good/missing/broken。返回 RGB 图"""
    rng = rng or random
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx = size // 2
    d_nom = SPEC_D_NOM[spec]
    pitch = SPEC_PITCH_MM[spec]

    head_d = d_nom * 1.6          # 头部直径 ≈ 1.6×d
    head_h = d_nom * 1.0          # 头部高度
    root_d = d_nom * 0.75         # 螺纹底径（杆部直径）
    thread_h = d_nom * 0.14       # 牙高

    top_y = 30
    # 头部（矩形 + 顶部圆角 + 倒角）
    head_w_px = int(head_d * px_per_mm)
    head_h_px = int(head_h * px_per_mm)
    x0, x1 = cx - head_w_px // 2, cx + head_w_px // 2
    y_top = top_y
    cv2.rectangle(img, (x0, y_top), (x1, y_top + head_h_px), (150, 150, 155), -1)
    cv2.circle(img, (cx, y_top + int(head_h_px * 0.3)), head_w_px // 2, (150, 150, 155), -1)
    cv2.line(img, (x0, y_top + head_h_px), (x0 + head_w_px // 6, y_top + head_h_px + 6), (90, 90, 95), 2)
    cv2.line(img, (x1, y_top + head_h_px), (x1 - head_w_px // 6, y_top + head_h_px + 6), (90, 90, 95), 2)

    # 杆部 + 螺纹
    shaft_top = y_top + head_h_px + 8
    shaft_bottom = size - 30
    root_w_px = int(root_d * px_per_mm)
    pitch_px = int(pitch * px_per_mm)
    thread_h_px = int(thread_h * px_per_mm)
    n_teeth = int((shaft_bottom - shaft_top) / pitch_px)

    # 逐牙绘制
    teeth = []  # 每牙的 [center_y, x_left, x_right, tooth_h_px]
    for i in range(n_teeth):
        cy = shaft_top + i * pitch_px + pitch_px // 2
        if cy > shaft_bottom:
            break
        teeth.append([cy, thread_h_px, 0])

    if state == "good":
        _draw_teeth_good(img, cx, root_w_px, teeth, shaft_top, shaft_bottom)
    elif state == "missing":
        # 缺 1~2 牙：随机选牙位，该处不画牙（杆部裸露）
        miss_idx = rng.sample(range(len(teeth)), rng.randint(1, 2))
        for i, t in enumerate(teeth):
            if i in miss_idx:
                continue
            _draw_one_tooth(img, cx, root_w_px, t[0], t[1], t[2])
        _draw_shaft(img, cx, root_w_px, shaft_top, shaft_bottom)
    else:  # broken
        # 烂牙：连续 3~6 牙挤压变形
        start = rng.randint(0, max(0, len(teeth) - 6))
        broken_len = rng.randint(3, 6)
        for i, t in enumerate(teeth):
            if start <= i < start + broken_len:
                # 变形牙：牙高压缩 + 牙尖噪声 + 歪斜
                compressed_h = max(2, int(t[1] * rng.uniform(0.3, 0.6)))
                _draw_one_tooth(img, cx, root_w_px, t[0], compressed_h,
                                rng.uniform(-0.15, 0.15) * root_w_px)
            else:
                _draw_one_tooth(img, cx, root_w_px, t[0], t[1], t[2])
        _draw_shaft(img, cx, root_w_px, shaft_top, shaft_bottom)

    # 头部阴影 + 轻微增广
    img = _augment_thread(img, rng)
    return img


def _draw_shaft(img, cx, root_w_px, top, bottom):
    cv2.rectangle(img, (cx - root_w_px // 2, top), (cx + root_w_px // 2, bottom),
                  (120, 120, 125), -1)


def _draw_teeth_good(img, cx, root_w_px, teeth, top, bottom):
    _draw_shaft(img, cx, root_w_px, top, bottom)
    for t in teeth:
        _draw_one_tooth(img, cx, root_w_px, t[0], t[1], t[2])


def _draw_one_tooth(img, cx, root_w_px, cy, tooth_h, skew):
    """画单个三角牙：从杆边缘向外突的三角形"""
    half = root_w_px // 2
    x_inner_l, x_inner_r = cx - half, cx + half
    # 左侧牙：顶点在杆左外侧
    tip_l_x = x_inner_l - tooth_h + skew
    cv2.fillPoly(img, [np.array([[x_inner_l, cy - 2], [tip_l_x, cy], [x_inner_l, cy + 2]], np.int32)],
                 (105, 105, 110))
    # 右侧牙
    tip_r_x = x_inner_r + tooth_h + skew
    cv2.fillPoly(img, [np.array([[x_inner_r, cy - 2], [tip_r_x, cy], [x_inner_r, cy + 2]], np.int32)],
                 (105, 105, 110))


def _augment_thread(img, rng):
    """增广：亮度/对比度/轻噪声（不旋转，螺纹有方向性）"""
    out = img.copy()
    alpha = rng.uniform(0.9, 1.1)
    beta = rng.uniform(-10, 10)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.4:
        # 保留带符号噪声再 clip，避免负噪声截断导致背景抬亮
        noise = np.random.normal(0, rng.uniform(2, 4), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# ================= 主流程 =================

def make_one(spec, state, rng):
    return _draw_screw_side(spec, state=state, rng=rng)


def generate(root=None, n_train=N_TRAIN, n_val=N_VAL):
    root = Path(root or DATA_DIR / "thread_side")
    rng = random.Random(42)
    for split, n in (("train", n_train), ("val", n_val)):
        for cls in THREAD_CLASSES:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                spec = rng.choice(list(SPEC_PITCH_MM.keys()))
                img = make_one(spec, cls, rng)
                # cv2.imwrite 需 BGR；此处生成 RGB，转换保存
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(d / f"{spec}_{i:04d}.png"), img_bgr)
    print(f"[thread_side] 生成完成: {root}")
    for split in ("train", "val"):
        for cls in THREAD_CLASSES:
            n = len(list((root / split / cls).glob("*.png")))
            print(f"  {split}/{cls}: {n}")


if __name__ == "__main__":
    generate()
