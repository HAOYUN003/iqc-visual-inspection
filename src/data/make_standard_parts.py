# -*- coding: utf-8 -*-
"""
标准件规格模拟数据集生成器
用 OpenCV 程序化绘制 M3/M4/M5/M6/M8 内六角螺钉、垫片、螺母的俯视图像，
叠加光照变化、噪声、旋转、模糊等增广，模拟不同拍照环境。

用途：在没有真实硬件/照片的前提下，为"规格防错"分类模型提供训练数据，
打通"采集→训练→检测"的完整流程。Phase 1 拿到真实零件/工业相机后，
用同架构重新采集训练即可无缝替换。

规格对应真实尺寸（GB/T 70.1 内六角螺钉头部直径，单位 mm）：
    M3: 5.5   M4: 7.0   M5: 8.5   M6: 10.0   M8: 13.0
"""
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# 参考真实尺寸（头部直径 mm），用于成比例缩放
HEAD_DIAM_MM = {"M3": 5.5, "M4": 7.0, "M5": 8.5, "M6": 10.0, "M8": 13.0}

IMAGE_SIZE = 256
N_TRAIN = 200       # 每类训练图
N_VAL = 50          # 每类验证图


# ================= 基础绘制 =================

def draw_hex_screw_head(spec, scale, size=IMAGE_SIZE):
    """内六角螺钉头部俯视图：圆形法兰 + 内六角沉孔"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    # 头部外圆半径(px) = 该规格头部直径(mm) × 每毫米像素数
    r_outer = int(HEAD_DIAM_MM[spec] * scale)
    cv2.circle(img, (c, c), r_outer, (160, 160, 160), -1)
    # 内六角沉孔（六边形）
    r_hex = int(r_outer * 0.45)
    _draw_hexagon(img, c, c, r_hex, angle=0, color=(80, 80, 90), thickness=-1)
    # 沉孔倒角阴影
    cv2.circle(img, (c, c), r_hex, (40, 40, 50), 1)
    return img


def draw_washer(spec, scale, is_flat=True, size=IMAGE_SIZE):
    """垫片俯视图：圆环"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    # 垫片外圆半径(px) = 头部直径(mm) × 1.25 放大 × 每毫米像素数
    base_r = int(HEAD_DIAM_MM[spec] * scale * 1.25)
    hole_r = int(base_r * 0.42)
    cv2.circle(img, (c, c), base_r, (200, 200, 205), -1)
    cv2.circle(img, (c, c), hole_r, (0, 0, 0), -1)
    if not is_flat:
        # 非平垫：画出凸缘轮廓线
        cv2.circle(img, (c, c), base_r, (120, 120, 125), 1)
    return img


def draw_nut(spec, scale, size=IMAGE_SIZE):
    """螺母俯视图：外六角 + 中心螺纹孔"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    # 螺母外接圆半径(px) = 头部直径(mm) × 每毫米像素数
    hex_r = int(HEAD_DIAM_MM[spec] * scale)
    _draw_hexagon(img, c, c, hex_r, angle=30, color=(170, 170, 175), thickness=-1)
    hole_r = int(hex_r * 0.42)
    cv2.circle(img, (c, c), hole_r, (30, 30, 35), -1)
    return img


def _draw_hexagon(img, cx, cy, r, angle, color, thickness):
    pts = []
    for i in range(6):
        a = math.radians(angle + i * 60)
        pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
    cv2.fillPoly(img, [np.array(pts, np.int32)], color)
    if thickness > 0:
        cv2.polylines(img, [np.array(pts, np.int32)], True, (30, 30, 35), thickness)


# ================= 增广（模拟拍照环境） =================

def augment(img):
    """随机增广：旋转、轻微亮度/对比度、轻噪声、平移（强度适中，保留尺寸特征）"""
    h, w = img.shape[:2]
    out = img.copy()
    # 随机旋转（背景填黑，保留完整零件）
    angle = random.uniform(-180, 180)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    out = cv2.warpAffine(out, M, (w, h), borderValue=(0, 0, 0))
    # 轻微平移（模拟摆放偏移，幅度小不裁掉边缘信息）
    tx = random.uniform(-4, 4); ty = random.uniform(-4, 4)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    out = cv2.warpAffine(out, M, (w, h), borderValue=(0, 0, 0))
    # 亮度/对比度（模拟光照变化，范围收窄）
    alpha = random.uniform(0.85, 1.15)
    beta = random.uniform(-15, 15)
    out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    # 轻高斯噪声（保留带符号噪声再 clip，避免 astype(uint8) 截断负值导致背景整体抬亮）
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(2, 5), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # 轻微模糊
    if random.random() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


# ================= 主入口 =================

def generate(root: Path, n_train=N_TRAIN, n_val=N_VAL, size=IMAGE_SIZE, px_per_mm=None):
    root = Path(root)
    for split in ("train", "val"):
        (root / split).mkdir(parents=True, exist_ok=True)
        (root / split).parent.mkdir(parents=True, exist_ok=True)

    for spec in HEAD_DIAM_MM:
        for split, n in (("train", n_train), ("val", n_val)):
            split_dir = root / split / spec
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                # 每毫米像素数：默认微幅抖动（模拟轻微对焦/距离差异），
                # 可传固定值 px_per_mm（尺寸验证集用），拉开规格尺寸差且 M8 垫片不溢出
                scale = px_per_mm if px_per_mm else random.uniform(8.4, 8.6)
                kind = random.choice(["screw", "washer", "nut"])
                if kind == "screw":
                    img = draw_hex_screw_head(spec, scale)
                elif kind == "washer":
                    img = draw_washer(spec, scale)
                else:
                    img = draw_nut(spec, scale)
                img = augment(img)
                cv2.imwrite(str(split_dir / f"{kind}_{i:04d}.png"), img)
    print(f"[generate_standard] 生成完成: {root}")
    print(f"  类别: {list(HEAD_DIAM_MM.keys())}")
    print(f"  train/val = {n_train}/{n_val} 每类，图像尺寸 {size}x{size}")


def verify(root: Path):
    """快速校验数据集结构"""
    root = Path(root)
    total = 0
    for split in ("train", "val"):
        for spec_dir in sorted((root / split).iterdir()):
            if spec_dir.is_dir():
                n = len(list(spec_dir.glob("*.png")))
                total += n
                print(f"  {split}/{spec_dir.name}: {n}")
    print(f"  总计 {total} 张")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config import STD_PARTS_DIR
    generate(STD_PARTS_DIR)
    verify(STD_PARTS_DIR)
