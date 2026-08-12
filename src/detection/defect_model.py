# -*- coding: utf-8 -*-
"""
表面缺陷检测模型（YOLOv8 + NEU-DET）
检测机加工表面常见缺陷：crazing(裂纹), inclusion(夹杂), patches(斑块),
pitted_surface(麻面), rolled_in_scale(氧化皮), scratches(划伤)

NEU-DET 数据集：东北大学发布的热轧带钢表面缺陷公开数据集，6 类共 1800 张。
Phase 1 拿到真实零件图像后，用 labelme 标注 + 本模块同架构再训练即可替换。

用法：
    python defect_model.py --download   # 下载/准备 NEU-DET 并转 YOLO 格式
    python defect_model.py --train      # 训练 YOLOv8n
    python defect_model.py --predict <img_path>  # 单图预测
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, DEFECT_MODEL_PATH, DEFECT_CLASSES, DEFECT_CONF_THRESH

# 已验证可用的 NEU-DET 镜像仓库（内含 IMAGES/*.jpg + ANNOTATIONS/*.xml，共1800张）
NEU_URL = "https://github.com/SprAJR/NEU-DET-Steel-Surface-Defect-Detection/archive/refs/heads/master.zip"
NEU_DIR = DATA_DIR / "neu_det"
YOLO_DATA_YAML = DATA_DIR / "neu_yolo" / "data.yaml"

# NEU-DET 6 类（索引与 XML 中类别名对应）
NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
               "rolled-in_scale", "scratches"]


# ================= 数据准备 =================

def _xml_to_yolo(xml_path, img_w, img_h):
    """将 NEU-DET 的 VOC-XML 标注转换为 YOLO 格式行 (class cx cy w h 归一化)"""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in NEU_CLASSES:
            continue
        cls_idx = NEU_CLASSES.index(name)
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)
        cx = (xmin + xmax) / 2 / img_w
        cy = (ymin + ymax) / 2 / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def download_neu():
    """下载 NEU-DET（SprAJR 镜像）并转为 YOLO 标注格式"""
    import zipfile
    import urllib.request
    import shutil

    NEU_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "neu_master.zip"
    if not zip_path.exists():
        print("[defect] 下载 NEU-DET 数据集（1800张，约30MB）...")
        req = urllib.request.Request(NEU_URL, headers={"User-Agent": "Mozilla/5.0"})
        urllib.request.urlretrieve(NEU_URL, zip_path)
    print("[defect] 解压中...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    src = None
    for p in DATA_DIR.iterdir():
        if p.is_dir() and "NEU-DET-Steel-Surface-Defect-Detection" in p.name:
            src = p
    if src is None:
        raise RuntimeError("解压失败，未找到数据集目录")

    neu_imgs = src / "IMAGES"
    neu_anns = src / "ANNOTATIONS"
    if not neu_imgs.exists() or not neu_anns.exists():
        raise RuntimeError(f"未找到 IMAGES/ANNOTATIONS: {src}")

    out_dir = DATA_DIR / "neu_yolo"
    for split in ("train", "val"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    jpgs = sorted(neu_imgs.glob("*.jpg"))
    n = len(jpgs)
    val_n = int(n * 0.2)
    import random
    random.seed(42)
    random.shuffle(jpgs)
    for i, img in enumerate(jpgs):
        xml = neu_anns / (img.stem + ".xml")
        if not xml.exists():
            continue
        split = "val" if i < val_n else "train"
        shutil.copy(img, out_dir / split / "images" / img.name)
        # XML -> YOLO txt
        lines = _xml_to_yolo(xml, 200, 200)
        if lines:
            (out_dir / split / "labels" / (img.stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

    # 写 data.yaml
    import yaml
    data = {
        "path": str(out_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {i: c for i, c in enumerate(DEFECT_CLASSES)},
    }
    with open(YOLO_DATA_YAML, "w") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    print(f"[defect] 数据准备完成: train/val 已写入 {out_dir}, 共 {n} 张")
    print(f"[defect] train images: {len(list((out_dir/'train'/'images').glob('*.jpg')))}  "
          f"val images: {len(list((out_dir/'val'/'images').glob('*.jpg')))}")


def ensure_neu_data():
    """确保 NEU 数据存在，不存在则自动下载"""
    if (DATA_DIR / "neu_yolo" / "train" / "images").exists() and \
       (DATA_DIR / "neu_yolo" / "val" / "images").exists():
        return
    download_neu()


# ================= 训练 =================

def train(epochs=50, imgsz=640, device=None, workers=0):
    from ultralytics import YOLO
    ensure_neu_data()
    model = YOLO("yolov8n.pt")   # 预训练权重（首次联网下载）
    results = model.train(
        data=str(YOLO_DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=device or (0 if _has_gpu() else "cpu"),
        project=str(DATA_DIR / "runs"),
        name="defect_yolo",
        patience=10,
        workers=workers,   # Windows 下多进程 DataLoader 易崩溃，默认用 0
    )
    best = DATA_DIR / "runs" / "defect_yolo" / "weights" / "best.pt"
    if best.exists():
        DEFECT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best, DEFECT_MODEL_PATH)
        print(f"[defect] 最佳权重已保存到 {DEFECT_MODEL_PATH}")
    return results


def _has_gpu():
    import torch
    return torch.cuda.is_available()


# ================= 推理 =================

def load_defect_model(device="cpu"):
    from ultralytics import YOLO
    if not DEFECT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"缺陷模型不存在: {DEFECT_MODEL_PATH}，请先运行训练"
            "(python defect_model.py --train)")
    return YOLO(DEFECT_MODEL_PATH, verbose=False)


def predict_defects(model, img_bgr, conf_thresh=DEFECT_CONF_THRESH):
    """输入 BGR 图，返回 (defect_summary, defect_boxes, annotated_bgr)
    defect_summary: {缺陷类型: 数量}
    defect_boxes:   [{class, conf, xyxy}]
    """
    import cv2
    results = model.predict(source=img_bgr, conf=conf_thresh, verbose=False)
    r = results[0]
    summary = {}
    boxes = []
    if r.boxes is not None:
        names = r.names
        for b in r.boxes:
            cls = names[int(b.cls)]
            conf = float(b.conf)
            xyxy = [float(v) for v in b.xyxy[0]]
            summary[cls] = summary.get(cls, 0) + 1
            boxes.append({"class": cls, "conf": conf, "xyxy": xyxy})
    annotated = r.plot() if r.boxes is not None else img_bgr
    return summary, boxes, annotated


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--predict", type=str, default=None)
    args = ap.parse_args()
    if args.download:
        download_neu()
    if args.train:
        train(epochs=args.epochs)
    if args.predict:
        import cv2
        m = load_defect_model()
        img = cv2.imread(args.predict)
        summary, boxes, ann = predict_defects(m, img)
        print("缺陷:", summary)
        for b in boxes:
            print(f"  {b['class']}: conf={b['conf']:.3f} box={[int(v) for v in b['xyxy']]}")
        cv2.imwrite("predict_out.jpg", ann)
        print("已保存标注结果: predict_out.jpg")
