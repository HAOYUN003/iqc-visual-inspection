# -*- coding: utf-8 -*-
"""
人工复核数据导出训练集：把检验记录里的照片按判定结果组织成可训练格式。

思路：
- 规格识别训练集：data/training/spec/<label>/img.jpg
    label = AI规格识别结果（M3~M8），取复核 PASS 或 AI判定 OK 的干净样本
- 缺陷检测训练集：data/training/defect/ok|ng/img.jpg（二分类）
    ok = 复核 PASS / AI判定 OK 且无缺陷
    ng = 复核 REJECT / AI判定 NG（含缺陷）
- 真实缺陷照片积累到一定量后，用 spec_model/defect_model 重训本地模型

注意：这是"正样本积累"通道，不是精确标注。真实缺陷坐标标注仍需 labelme，
但至少先把"这张图合格/这张图有缺陷"的判别积累起来。
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR
from data import quality_db as db

TRAIN_DIR = DATA_DIR / "training"
SPEC_DIR = TRAIN_DIR / "spec"       # spec/<label>/img.jpg
DEFECT_DIR = TRAIN_DIR / "defect"   # defect/ok|ng/img.jpg


def export_training_data(only_reviewed=False, dry_run=False):
    """
    扫描检验记录，导出训练数据。
    only_reviewed=True: 只导出有人工复核结论的记录（PASS/REJECT），最可信
    only_reviewed=False(默认): 复核 PASS 或 AI 判定 OK 都算可信正样本
    （种子数据即"AI OK 未复核"，应纳入补充正样本）
    返回统计 dict。
    """
    records = db.get_records(limit=2000)
    stats = {"spec": {}, "defect_ok": 0, "defect_ng": 0, "skipped": 0}

    for r in records:
        img_path = r.get("image_path")
        if not img_path or not Path(img_path).exists():
            stats["skipped"] += 1
            continue
        review = r.get("review_verdict")        # PASS/REJECT/None
        ai = r.get("ai_verdict")                 # OK/NG/UNSURE
        spec = r.get("spec_result")              # M3~M8/UNKNOWN/None

        # 判断是否可信（有人工复核结论，或允许未复核的AI判定OK）
        trusted = review in ("PASS", "REJECT") or (not only_reviewed and ai == "OK")
        if not trusted:
            stats["skipped"] += 1
            continue

        # ---- 规格训练集：干净样本（复核PASS 或 AI判定OK）----
        if spec and spec not in ("UNKNOWN",) and (review in ("PASS", None)):
            label_dir = SPEC_DIR / spec
            stats["spec"][spec] = stats["spec"].get(spec, 0) + 1
            if not dry_run:
                label_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, label_dir / f"{r['record_id']}_{Path(img_path).name}")

        # ---- 缺陷二分类：ok / ng ----
        if review == "PASS" or (review is None and ai == "OK"):
            stats["defect_ok"] += 1
            if not dry_run:
                (DEFECT_DIR / "ok").mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, DEFECT_DIR / "ok" / f"{r['record_id']}_{Path(img_path).name}")
        elif review == "REJECT" or ai == "NG":
            stats["defect_ng"] += 1
            if not dry_run:
                (DEFECT_DIR / "ng").mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, DEFECT_DIR / "ng" / f"{r['record_id']}_{Path(img_path).name}")

    return stats


def print_stats(stats):
    spec = stats["spec"]
    total_spec = sum(spec.values())
    print(f"规格训练集: {total_spec} 张 {dict(spec)}")
    print(f"缺陷训练集: OK {stats['defect_ok']} 张 / NG {stats['defect_ng']} 张")
    print(f"跳过（无照片/不可信）: {stats['skipped']} 张")


# 训练目标阈值（攒够即可重训本地模型）
TARGET_SPEC_PER_CLASS = 100     # 每规格目标样本数
TARGET_READING_PAIRS = 300      # 卡尺图+读数 配对目标（用于本地读数模型）
TARGET_DEFECT_NG = 100          # 缺陷 NG 样本目标（用于缺陷模型）


def training_progress():
    """
    统计当前训练数据积累进度，返回 dict：
      spec_count / spec_target / spec_pct
      reading_count / reading_target / reading_pct
      defect_ok / defect_ng / defect_target / defect_pct
    """
    # 规格训练集：已导出目录里的样本数
    spec_count = 0
    if SPEC_DIR.exists():
        for d in SPEC_DIR.iterdir():
            if d.is_dir():
                spec_count += len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))

    # 卡尺读数配对：数据库里有 reading_mm 的记录数
    reading_count = 0
    try:
        for r in db.get_records(limit=2000):
            dim = r.get("dimension")
            if dim and dim.get("reading_mm") is not None:
                reading_count += 1
    except Exception:
        pass

    # 缺陷 NG 样本：已导出的 ng 目录
    defect_ng = len(list((DEFECT_DIR / "ng").glob("*.jpg"))) if (DEFECT_DIR / "ng").exists() else 0
    defect_ok = len(list((DEFECT_DIR / "ok").glob("*.jpg"))) if (DEFECT_DIR / "ok").exists() else 0

    return {
        "spec_count": spec_count,
        "spec_target": TARGET_SPEC_PER_CLASS * 5,  # 5 类规格
        "reading_count": reading_count,
        "reading_target": TARGET_READING_PAIRS,
        "defect_ok": defect_ok,
        "defect_ng": defect_ng,
        "defect_target": TARGET_DEFECT_NG,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只统计不复制")
    ap.add_argument("--reviewed-only", action="store_true",
                    help="只导出有人工复核的记录（默认含AI判定OK的种子）")
    args = ap.parse_args()
    stats = export_training_data(only_reviewed=args.reviewed_only, dry_run=args.dry_run)
    print_stats(stats)
    if not args.dry_run:
        print(f"\n训练数据已导出到: {TRAIN_DIR}")
