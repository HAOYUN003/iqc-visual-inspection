# -*- coding: utf-8 -*-
"""
把真实 OK 照片混入紧固件缺陷训练集（作为干净件负样本）。
目的：降低本地缺陷模型在真实零件上的误检——模型没见过真实纹理，
把正常表面当缺陷。混入真实 OK 样本后，模型学会"真实干净件不出框"。

用法：python src/add_real_ok_samples.py
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, FASTENER_DEFECT_DATA


def main():
    ok_dir = DATA_DIR / "training" / "defect" / "ok"
    if not ok_dir.exists():
        print("[add_ok] 无真实 OK 样本目录，先运行 export_training.py 导出")
        return 0

    # 目标：训练集 + 验证集各混入部分真实 OK 图
    ok_imgs = sorted(ok_dir.glob("*.jpg")) + sorted(ok_dir.glob("*.png"))
    print(f"[add_ok] 真实 OK 样本: {len(ok_imgs)} 张")

    # 前 70% 进训练集，后 30% 进验证集
    n_train = int(len(ok_imgs) * 0.7)
    added = 0
    for i, img in enumerate(ok_imgs):
        split = "train" if i < n_train else "val"
        img_dir = FASTENER_DEFECT_DATA / split / "images"
        lab_dir = FASTENER_DEFECT_DATA / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        dest = img_dir / f"real_ok_{i:04d}{img.suffix.lower()}"
        # 统一转 jpg（YOLO 训练最稳）
        import cv2
        from config import patch_cv_io; patch_cv_io()
        cv2.imwrite(str(dest.with_suffix(".jpg")), cv2.imread(str(img)))
        # 空标注文件（真实 OK = 无缺陷）
        (lab_dir / f"real_ok_{i:04d}.txt").write_text("", encoding="utf-8")
        added += 1

    print(f"[add_ok] 已混入 {added} 张真实 OK 样本到训练/验证集")
    print(f"        训练集图片: {len(list((FASTENER_DEFECT_DATA/'train'/'images').glob('*.jpg')))}")
    print(f"        验证集图片: {len(list((FASTENER_DEFECT_DATA/'val'/'images').glob('*.jpg')))}")
    return added


if __name__ == "__main__":
    main()
