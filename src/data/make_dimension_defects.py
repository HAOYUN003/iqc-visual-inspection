# -*- coding: utf-8 -*-
"""
尺寸验证集生成器（不训练，仅用于验证"尺寸测量"模块的精度）
生成固定 px_per_mm 的标准件图，其中被测量的最外圈直径人为偏置
±(0.15~0.8)mm（含合格件与超差件），真值写入 manifest.csv，供测量模块比对。

三种零件统一语义：**真值 = 最外层轮廓直径**（外径）
    screw : 头部外径 = HEAD_DIAM[spec] × scale
    washer: 外径     = HEAD_DIAM[spec] × scale × 1.25
    nut   : 六角外接圆直径 = HEAD_DIAM[spec] × scale
生成时按 kind 反推 scale，使实际外径 = head_mm（期望值）。

标签语义：
    ok        : 直径在标称 ±0.2mm 内（合格）
    undersize : 偏小超差
    oversize  : 偏大超差
"""
import csv
import random
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import make_standard_parts as msp
from config import DIMENSION_TEST_DIR, CALIB_PX_PER_MM

N_OK = 40
N_BAD = 60   # 超差件（含 undersize/oversize 各半）
PX_PER_MM = CALIB_PX_PER_MM
SIZE = 256


def nominal_outer_mm(spec, kind):
    """每种零件的名义最外层直径 mm（尺寸测量真值围绕它偏置）"""
    if kind == "washer":
        return msp.HEAD_DIAM_MM[spec] * 1.25   # 垫片外径 = 头径×1.25
    return msp.HEAD_DIAM_MM[spec]              # 螺钉头径 / 螺母外接圆直径


def draw_with_outer_diam(spec, kind, outer_mm, px_per_mm=PX_PER_MM, size=SIZE):
    """按指定外径 mm 绘制三种零件（统一测量真值 = 最外层轮廓直径）。
    msp 绘制函数的 scale 使"半径px = HEAD_DIAM×scale"，故外径px = 2×HEAD_DIAM×scale。
    要外径px = outer_mm×px_per_mm：scale = outer_mm×px_per_mm/(2×HEAD_DIAM)。
    washer 内部再乘 1.25，故除以 1.25 抵消。
    """
    if kind == "screw":
        scale = outer_mm * px_per_mm / (2 * msp.HEAD_DIAM_MM[spec])
        return msp.draw_hex_screw_head(spec, scale, size=size)
    if kind == "washer":
        scale = outer_mm * px_per_mm / (2 * msp.HEAD_DIAM_MM[spec] * 1.25)
        return msp.draw_washer(spec, scale, size=size)
    scale = outer_mm * px_per_mm / (2 * msp.HEAD_DIAM_MM[spec])
    return msp.draw_nut(spec, scale, size=size)


def generate(root=None, n_ok=N_OK, n_bad=N_BAD, px_per_mm=PX_PER_MM):
    root = Path(root or DIMENSION_TEST_DIR)
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(123)
    rows = []
    specs = list(msp.HEAD_DIAM_MM.keys())
    kinds = ["screw", "washer", "nut"]
    nominal = msp.HEAD_DIAM_MM

    idx = 0
    # 合格件
    for _ in range(n_ok):
        spec = rng.choice(specs)
        kind = rng.choice(kinds)
        nominal = nominal_outer_mm(spec, kind)
        outer_mm = nominal + rng.uniform(-0.15, 0.15)
        img = draw_with_outer_diam(spec, kind, outer_mm, px_per_mm)
        img = msp.augment(img)
        cv2.imwrite(str(img_dir / f"img_{idx:04d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append([f"img_{idx:04d}.jpg", spec, kind, f"{outer_mm:.4f}",
                     f"{nominal:.4f}", "ok", str(px_per_mm)])
        idx += 1
    # 超差件（half undersize, half oversize）
    for i in range(n_bad):
        spec = rng.choice(specs)
        kind = rng.choice(kinds)
        nominal = nominal_outer_mm(spec, kind)
        sign = -1 if i % 2 == 0 else 1
        delta = rng.uniform(0.25, 0.8)
        outer_mm = nominal + sign * delta
        img = draw_with_outer_diam(spec, kind, outer_mm, px_per_mm)
        img = msp.augment(img)
        cv2.imwrite(str(img_dir / f"img_{idx:04d}.jpg"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        label = "undersize" if sign < 0 else "oversize"
        rows.append([f"img_{idx:04d}.jpg", spec, kind, f"{outer_mm:.4f}",
                     f"{nominal:.4f}", label, str(px_per_mm)])
        idx += 1

    # manifest.csv
    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "spec", "kind", "true_outer_mm", "nominal_mm",
                    "label", "px_per_mm"])
        w.writerows(rows)
    print(f"[dimension_test] 生成完成: {root}，共 {idx} 张")
    print(f"  ok: {n_ok}，undersize: {n_bad//2}，oversize: {n_bad - n_bad//2}")


if __name__ == "__main__":
    generate()
