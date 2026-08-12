# -*- coding: utf-8 -*-
"""
螺纹状态分类模型（侧视图 3 类：good/missing/broken）
输入螺钉侧视图，判断螺纹是否正常 / 缺牙 / 烂牙。

复用 spec_model.SpecCNN 轻量网络，按类子目录组织数据：
    data/thread_side/{train,val}/{good,missing,broken}/*.png
用法：
    python thread_model.py --train --epochs 30
    python thread_model.py --predict <img_path>
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (THREAD_SIDE_DIR, THREAD_CLASSES, THREAD_IMG_SIZE,
                    THREAD_MODEL_PATH)
from detection import spec_model


class ThreadDataset(Dataset):
    _IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

    def __init__(self, root: Path, split: str, size=THREAD_IMG_SIZE):
        self.samples = []
        self.labels = []
        root = Path(root) / split
        for label, cls in enumerate(THREAD_CLASSES):
            cls_dir = root / cls
            if not cls_dir.exists():
                continue
            for img_path in sorted(_iter_images(cls_dir)):
                self.samples.append(str(img_path))
                self.labels.append(label)
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img = Image.open(self.samples[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def _iter_images(directory: Path):
    """按扩展名收集图像文件（png/jpg/jpeg/bmp），支持真实照片放 jpg"""
    import itertools
    return itertools.chain(*(sorted(directory.glob(pat)) for pat in ThreadDataset._IMG_EXTS))


def build_model(n_classes=len(THREAD_CLASSES)):
    return spec_model.SpecCNN(n_classes)


def train(epochs=30, batch_size=32, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[thread] device={device} classes={THREAD_CLASSES}")

    train_ds = ThreadDataset(THREAD_SIDE_DIR, "train")
    val_ds = ThreadDataset(THREAD_SIDE_DIR, "val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            THREAD_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "classes": THREAD_CLASSES},
                       THREAD_MODEL_PATH)
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch}/{epochs}  loss={running_loss/len(train_ds):.4f}  "
                  f"val_acc={acc:.4f}  best={best_acc:.4f}")

    print(f"[thread] 完成, best_val_acc={best_acc:.4f}, 用时 {time.time()-t0:.0f}s")
    print(f"[thread] 权重已保存: {THREAD_MODEL_PATH}")
    return best_acc


def load_thread_model(device="cpu"):
    if not THREAD_MODEL_PATH.exists():
        raise FileNotFoundError(f"螺纹模型不存在: {THREAD_MODEL_PATH}")
    ckpt = torch.load(THREAD_MODEL_PATH, map_location=device)
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt["classes"]


def predict_thread(model, classes, img_bgr, device="cpu", conf_thresh=0.35):
    """输入侧视图 BGR 图，返回 (状态, 置信度, 概率dict)。
    阈值默认 0.35：模拟数据下验证为总准确率 85%、无拒识（0.55 过严丢弃大量正确结果）。"""
    import cv2
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    transform = transforms.Compose([
        transforms.Resize(THREAD_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model_device = next(model.parameters()).device
    x = transform(pil).unsqueeze(0).to(model_device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    prob = float(probs[idx])
    if prob < conf_thresh:
        return "unknown", prob, {c: float(p) for c, p in zip(classes, probs)}
    return classes[idx], prob, {c: float(p) for c, p in zip(classes, probs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--predict", type=str, default=None)
    args = ap.parse_args()
    if args.train:
        train(epochs=args.epochs)
    if args.predict:
        import cv2
        m, cls = load_thread_model()
        img = cv2.imread(args.predict)
        state, conf, probs = predict_thread(m, cls, img)
        print("螺纹状态:", state, "| 置信度:", round(conf, 3))
        print("各类概率:", {k: round(v, 3) for k, v in probs.items()})
