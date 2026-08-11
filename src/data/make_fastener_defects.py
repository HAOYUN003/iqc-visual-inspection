# -*- coding: utf-8 -*-
"""
紧固件表面缺陷模拟数据集生成器
在标准件俯视图（螺钉/垫片/螺母）上程序化叠加表面缺陷：
    scratch : 划伤（随机折线）
    pitted  : 麻面（随机暗圆点）
    crack   : 裂纹（细锯齿线）

关键设计：
- 缺陷像素只落在零件 mask 内（不出现"背景上漂缺陷"的假样本）
- 每个缺陷记录实际包围盒，自动生成 YOLO 标注 txt（无需人工标注）
- 不做随机旋转（旋转会破坏 label 坐标系），旋转增广交给训练时 degrees=45
- 目录结构与 neu_yolo 一致：{train,val}/images/*.jpg + labels/*.txt + data.yaml
  → Phase 1 换真实数据时，labelme 标注转 YOLO 后直接替换即可复用代码
"""
import random
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import make_standard_parts as msp
from config import DATA_DIR, FASTENER_DEFECT_CLASSES

N_TRAIN = 300   # 每张图 0~2 缺陷，总体缺陷图约 train 图数的一半以上
N_VAL = 80
SIZE = 256
# 缺陷类别索引
CLS = {c: i for i, c in enumerate(FASTENER_DEFECT_CLASSES)}


# ================= 缺陷画法（都在 mask 内） =================

def _rand_point_in_mask(mask, rng, max_try=200):
    """在 mask 内随机取一个像素点（远离边界，保证缺陷完整落在零件上）"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    for _ in range(max_try):
        i = rng.randint(0, len(xs) - 1)
        x, y = int(xs[i]), int(ys[i])
        if (x > 4 and x < mask.shape[1] - 4 and y > 4 and y < mask.shape[0] - 4
                and rng.random() < 0.5):
            return (x, y)
    return (int(xs[len(xs)//2]), int(ys[len(ys)//2]))


def _rect_from_points(pts, pad=3):
    """由点集得包围盒 (x1,y1,x2,y2) 带外扩"""
    pts = np.array(pts)
    x1, y1 = int(pts[:, 0].min()) - pad, int(pts[:, 1].min()) - pad
    x2, y2 = int(pts[:, 0].max()) + pad, int(pts[:, 1].max()) + pad
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, msp.IMAGE_SIZE - 1), min(y2, msp.IMAGE_SIZE - 1)
    return [x1, y1, x2, y2]


def _clamp_pt(x, y, size):
    return max(0, min(x, size - 1)), max(0, min(y, size - 1))


def _safe_line(img, p1, p2, color, thickness, rng):
    """画线并做边界裁剪，越界目标点回退到图像边缘（并插入夹住后的端点）"""
    h, w = img.shape[:2]
    x1, y1 = _clamp_pt(int(p1[0]), int(p1[1]), w)
    x2, y2 = p2
    # 目标点越界时，先夹到边缘（保证颜色采样不越界）
    x2, y2 = _clamp_pt(int(x2), int(y2), w)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return [x2, y2]


def draw_scratch(img, mask, rng):
    """随机折线划痕，返回 (img, box_xyxy)"""
    p = _rand_point_in_mask(mask, rng)
    if p is None:
        return img, None
    pts = [list(p)]
    x, y = p
    angle = rng.uniform(0, 180)
    for _ in range(rng.randint(2, 4)):
        seg = rng.randint(20, 40)
        x2 = int(x + seg * np.cos(np.radians(angle)))
        y2 = int(y + seg * np.sin(np.radians(angle)))
        color = _defect_color(img, (x, y), rng)
        x2, y2 = _safe_line(img, (x, y), (x2, y2), color, rng.randint(1, 2), rng)
        pts.append([x2, y2])
        x, y = x2, y2
        angle += rng.uniform(-50, 50)
    return img, _rect_from_points(pts)


def draw_pitted(img, mask, rng):
    """麻面：mask 内撒暗圆点，返回 (img, box_xyxy)"""
    n = rng.randint(8, 25)
    pts = []
    for _ in range(n):
        p = _rand_point_in_mask(mask, rng)
        if p is None:
            continue
        x, y = p
        r = rng.randint(1, 3)
        cv2.circle(img, (int(x), int(y)), r, (35, 35, 45), -1)
        pts.append([x, y])
    if not pts:
        return img, None
    return img, _rect_from_points(pts)


def draw_crack(img, mask, rng):
    """细锯齿裂纹（可带短分支），返回 (img, box_xyxy)"""
    p = _rand_point_in_mask(mask, rng)
    if p is None:
        return img, None
    x, y = p
    pts = [[x, y]]
    for _ in range(rng.randint(4, 9)):
        seg = rng.randint(3, 7)
        angle = rng.uniform(0, 360)
        x2 = int(x + seg * np.cos(np.radians(angle)))
        y2 = int(y + seg * np.sin(np.radians(angle)))
        x2, y2 = _safe_line(img, (x, y), (x2, y2), (30, 28, 35), 1, rng)
        pts.append([x2, y2])
        # 短分支
        if rng.random() < 0.3:
            bx = int(x2 + (seg // 2) * np.cos(np.radians(angle + 90)))
            by = int(y2 + (seg // 2) * np.sin(np.radians(angle + 90)))
            bx, by = _safe_line(img, (x2, y2), (bx, by), (30, 28, 35), 1, rng)
            pts.append([bx, by])
        x, y = x2, y2
    return img, _rect_from_points(pts)


def _defect_color(img, p, rng):
    """缺陷颜色：比该处零件色亮或暗（保证可见）"""
    x, y = p
    base = int(img[y, x].mean())
    if rng.random() < 0.5:
        return (min(base + rng.randint(30, 60), 255),) * 3
    return (max(base - rng.randint(30, 60), 0),) * 3


# ================= 主流程 =================

def make_one(rng):
    """生成一张带 0~2 缺陷的标准件图，返回 (img, yolo_lines)"""
    # 复用 msp 生成一张干净标准件（固定 scale，避免抖动干扰缺陷检测）
    spec = rng.choice(list(msp.HEAD_DIAM_MM.keys()))
    kind = rng.choice(["screw", "washer", "nut"])
    scale = rng.uniform(8.4, 8.6)
    if kind == "screw":
        img = msp.draw_hex_screw_head(spec, scale, size=SIZE)
    elif kind == "washer":
        img = msp.draw_washer(spec, scale, size=SIZE)
    else:
        img = msp.draw_nut(spec, scale, size=SIZE)
    img = msp.augment(img)  # 增广（含旋转，但这里旋转发生在缺陷之前，不影响标注）

    mask = (img.sum(2) > 0).astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))  # 收缩，保证缺陷完整在零件内

    yolo_lines = []
    n_def = rng.randint(0, 2)
    for _ in range(n_def):
        draw_fn = rng.choice([draw_scratch, draw_pitted, draw_crack])
        img, box = draw_fn(img, mask, rng)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if w < 5 or h < 5:
            continue
        cx = (x1 + x2) / 2 / SIZE
        cy = (y1 + y2) / 2 / SIZE
        cls_name = draw_fn.__name__.replace("draw_", "")
        yolo_lines.append(f"{CLS[cls_name]} {cx:.5f} {cy:.5f} {w/SIZE:.5f} {h/SIZE:.5f}")
    return img, yolo_lines


def generate(root=None, n_train=N_TRAIN, n_val=N_VAL):
    root = Path(root or DATA_DIR / "fastener_defects")
    for split, n in (("train", n_train), ("val", n_val)):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(42 if split == "train" else 7)
        for i in range(n):
            img, lines = make_one(rng)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # cv2 保存 BGR
            cv2.imwrite(str(img_dir / f"img_{i:05d}.jpg"), img)
            if lines:
                (lbl_dir / f"img_{i:05d}.txt").write_text("\n".join(lines) + "\n")

    # data.yaml
    import yaml
    data = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "names": {i: c for i, c in enumerate(FASTENER_DEFECT_CLASSES)},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    print(f"[fastener_defects] 生成完成: {root}")
    for split in ("train", "val"):
        imgs = len(list((root / split / "images").glob("*.jpg")))
        lbls = len(list((root / split / "labels").glob("*.txt")))
        print(f"  {split}: {imgs} 图, {lbls} 带缺陷标注")


if __name__ == "__main__":
    generate()
