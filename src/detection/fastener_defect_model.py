# -*- coding: utf-8 -*-
"""
紧固件表面缺陷检测（YOLOv8）
检测标准件俯视图上的程序生成缺陷：
    scratch 划伤, pitted 麻面, crack 裂纹

数据：data/fastener_defects（由 make_fastener_defects.py 生成，含自动 YOLO 标注）
与 NEU-DET 模型的区别：本模型专门针对"紧固件表面"，域匹配度更高。
用法：
    python fastener_defect_model.py --train --epochs 40
    python fastener_defect_model.py --predict <img_path>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATA_DIR, FASTENER_DEFECT_MODEL_PATH,
                    FASTENER_DEFECT_DATA, FASTENER_DEFECT_CONF_THRESH,
                    FASTENER_DEFECT_CLASSES, FASTENER_DEFECT_IMG_SIZE)

DATA_YAML = FASTENER_DEFECT_DATA / "data.yaml"


def ensure_data():
    """确保紧固件缺陷数据存在，不存在则自动生成"""
    if (FASTENER_DEFECT_DATA / "train" / "images").exists() and \
       (FASTENER_DEFECT_DATA / "train" / "labels").exists():
        return
    from data import make_fastener_defects as mfd
    mfd.generate()


def train(epochs=50, imgsz=FASTENER_DEFECT_IMG_SIZE, device=None, workers=0):
    from ultralytics import YOLO
    ensure_data()
    model = YOLO("yolov8n.pt")   # 预训练权重（已缓存，无需联网）
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=device or (0 if _has_gpu() else "cpu"),
        project=str(DATA_DIR / "runs"),
        name="fastener_defect",
        patience=10,
        workers=workers,          # Windows 多进程 DataLoader 易崩
        degrees=45,               # 旋转增广（数据生成时不旋转，交由训练时补）
        scale=0.4,
    )
    best = DATA_DIR / "runs" / "fastener_defect" / "weights" / "best.pt"
    if best.exists():
        FASTENER_DEFECT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best, FASTENER_DEFECT_MODEL_PATH)
        print(f"[fastener_defect] 最佳权重已保存到 {FASTENER_DEFECT_MODEL_PATH}")
    return results


def _has_gpu():
    import torch
    return torch.cuda.is_available()


# ================= 推理 =================

def load_fastener_defect_model(device="cpu"):
    from ultralytics import YOLO
    path = FASTENER_DEFECT_MODEL_PATH if FASTENER_DEFECT_MODEL_PATH.exists() else "yolov8n.pt"
    return YOLO(path, verbose=False)


def predict_fastener_defects(model, img_bgr, conf_thresh=FASTENER_DEFECT_CONF_THRESH):
    """输入 BGR 图，返回 (defect_summary, defect_boxes, annotated_bgr)"""
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
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--predict", type=str, default=None)
    args = ap.parse_args()
    if args.train:
        train(epochs=args.epochs)
    if args.predict:
        import cv2
        m = load_fastener_defect_model()
        img = cv2.imread(args.predict)
        summary, boxes, ann = predict_fastener_defects(m, img)
        print("缺陷:", summary)
        for b in boxes:
            print(f"  {b['class']}: conf={b['conf']:.3f} box={[int(v) for v in b['xyxy']]}")
        cv2.imwrite("fastener_pred_out.jpg", ann)
        print("已保存标注结果: fastener_pred_out.jpg")
