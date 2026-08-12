# -*- coding: utf-8 -*-
"""
规格识别模型（标准件 M3/M4/M5/M6/M8 分类）
用于"规格防错"：检验员拍照 → AI 判断这是什么规格的零件 → 与图纸/料单比对，防止拿错混料。

架构：
    spec_cnn  : 自研小型 CNN（轻量、易训练、无预训练权重依赖）
    resnet18  : 预训练 ResNet18（效果好，但需要联网下载权重）
训练好的权重保存到 models/spec_classifier.pth，由 pipeline 加载使用。
"""
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
from config import STD_PARTS_DIR, SPEC_CLASSES, SPEC_IMG_SIZE, SPEC_MODEL_PATH


# ================= 小型 CNN =================

class SpecCNN(nn.Module):
    """轻量卷积网络：3 conv blocks + 2 fc"""
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                      # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                      # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                      # 28
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ================= 数据集 =================

class SpecDataset(Dataset):
    _IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

    def __init__(self, root: Path, split: str, size=(224, 224)):
        self.samples = []
        self.labels = []
        root = Path(root) / split
        for label, cls in enumerate(SPEC_CLASSES):
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
    return itertools.chain(*(sorted(directory.glob(pat)) for pat in SpecDataset._IMG_EXTS))


# ================= 训练 =================

def build_model(backbone="spec_cnn"):
    if backbone == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, len(SPEC_CLASSES))
    else:
        model = SpecCNN(len(SPEC_CLASSES))
    return model


def train(epochs=30, batch_size=32, lr=1e-3, backbone="resnet18"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_spec] device={device} backbone={backbone} classes={SPEC_CLASSES}")

    train_ds = SpecDataset(STD_PARTS_DIR, "train")
    val_ds = SpecDataset(STD_PARTS_DIR, "val")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(backbone).to(device)
    # label smoothing 缓解过拟合，AdamW + weight decay 提升泛化
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    no_improve = 0
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

        # 验证
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
            no_improve = 0
            SPEC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "classes": SPEC_CLASSES,
                        "backbone": backbone},
                       SPEC_MODEL_PATH)
        else:
            no_improve += 1
            if no_improve >= 10:   # 早停：10 epoch 无提升
                print(f"  epoch {epoch}: 早停（10 epoch 无提升）")
                break
        if epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch}/{epochs}  loss={running_loss/len(train_ds):.4f}  val_acc={acc:.4f}  best={best_acc:.4f}")

    print(f"[train_spec] 完成, best_val_acc={best_acc:.4f}, 用时 {time.time()-t0:.0f}s")
    print(f"[train_spec] 权重已保存: {SPEC_MODEL_PATH}")
    return best_acc


# ================= 推理 =================

def load_spec_model(device="cpu"):
    """加载规格识别模型，返回 (model, classes, device)"""
    if not SPEC_MODEL_PATH.exists():
        raise FileNotFoundError(f"规格模型不存在: {SPEC_MODEL_PATH}，请先运行训练")
    ckpt = torch.load(SPEC_MODEL_PATH, map_location=device)
    model = build_model(ckpt.get("backbone", "spec_cnn"))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt["classes"], device


def predict_spec(model, classes, img_bgr, device="cpu", conf_thresh=0.5, margin_thresh=None):
    """输入 BGR 图（H,W,3），返回 (预测规格, 置信度, 全类别概率 dict)。
    拒识策略（规格防错宁可"无法识别"也不误报）：
      - top1 置信度低于 conf_thresh → UNKNOWN
      - top1 与 top2 差距小于 margin_thresh → UNKNOWN（边界样本交给复检）
    """
    from config import SPEC_MARGIN_THRESH
    margin_thresh = SPEC_MARGIN_THRESH if margin_thresh is None else margin_thresh
    rgb = cv2_bgr_to_rgb(img_bgr)
    pil = Image.fromarray(rgb)
    transform = transforms.Compose([
        transforms.Resize(SPEC_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # tensor 必须与模型同 device，直接跟随模型，避免调用方传错
    model_device = next(model.parameters()).device
    x = transform(pil).unsqueeze(0).to(model_device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    top2 = np.argsort(probs)[::-1][:2]
    idx = int(top2[0])
    prob = float(probs[idx])
    margin = float(probs[top2[0]] - probs[top2[1]])
    if prob < conf_thresh or margin < margin_thresh:
        return "UNKNOWN", prob, {c: float(p) for c, p in zip(classes, probs)}
    return classes[idx], prob, {c: float(p) for c, p in zip(classes, probs)}


def cv2_bgr_to_rgb(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--backbone", default="spec_cnn", choices=["spec_cnn", "resnet18"])
    args = ap.parse_args()
    train(epochs=args.epochs, backbone=args.backbone)
