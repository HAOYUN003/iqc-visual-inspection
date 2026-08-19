# -*- coding: utf-8 -*-
"""
IQC 智能来料检测系统 - 全局配置
集中管理路径、参数，方便后续迁移/企业落地时调整。
"""
from pathlib import Path

# ============ 项目根路径 ============
BASE_DIR = Path(__file__).resolve().parent.parent

# ============ 数据目录 ============
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # 原始来料照片
STD_PARTS_DIR = DATA_DIR / "standard_parts"  # 标准件规格模拟图
DRAWINGS_DIR = DATA_DIR / "drawings"    # 图纸文件（PDF/图片）
DB_DIR = DATA_DIR / "db"
DB_PATH = DB_DIR / "iqc_records.db"

# ============ 模型目录 ============
MODEL_DIR = BASE_DIR / "models"
SPEC_MODEL_PATH = MODEL_DIR / "spec_classifier.pth"   # 规格识别 CNN
DEFECT_MODEL_PATH = MODEL_DIR / "defect_yolov8n.pt"   # 缺陷检测 YOLO
FASTENER_DEFECT_MODEL_PATH = MODEL_DIR / "fastener_defect_yolov8n.pt"  # 紧固件表面缺陷 YOLO
THREAD_MODEL_PATH = MODEL_DIR / "thread_classifier.pth"                # 螺纹状态 CNN

# ============ 规格识别（标准件分类）============
SPEC_CLASSES = ["M3", "M4", "M5", "M6", "M8"]   # 标准件规格类别（可扩展）
SPEC_IMG_SIZE = (256, 256)
SPEC_MODEL_TRAIN = "resnet18"                     # 'spec_cnn' | 'resnet18' 等

# ============ 紧固件表面缺陷（YOLO）============
# 在标准件俯视图上程序生成的缺陷，类别名对应画法
FASTENER_DEFECT_CLASSES = ["scratch", "pitted", "crack"]
FASTENER_DEFECT_DATA = DATA_DIR / "fastener_defects"      # YOLO 格式数据集（images+labels+data.yaml）
FASTENER_DEFECT_CONF_THRESH = 0.25
FASTENER_DEFECT_IMG_SIZE = 640

# ============ 螺纹状态（侧视图分类）============
THREAD_SIDE_DIR = DATA_DIR / "thread_side"        # 侧视图数据集 {train,val}/{good,missing,broken}
THREAD_CLASSES = ["good", "missing", "broken"]    # 正常 / 缺牙 / 烂牙
THREAD_IMG_SIZE = (224, 224)

# ============ 尺寸测量（传统 CV）============
# 标定系数：像素 ↔ 毫米（Phase 0 用模拟生成器已知值，Phase 1 用标定板实测）
# 该值同时决定模拟数据中零件在画布上的像素尺寸，须兼顾"规格区分度"与"不溢出"：
#   直径档位 M3~M8 = 82/105/127/150/195px（间隔22-25px，分类可分）
#   M8 垫片外径 = 13mm×1.25×15 = 244px < 256 画布（不溢出）
CALIB_PX_PER_MM = 15.0
# 标准件名义直径（头部直径 mm，GB/T 70.1 内六角螺钉），公差默认 ±0.2mm
SPEC_NOMINAL_HEAD_MM = {"M3": 5.5, "M4": 7.0, "M5": 8.5, "M6": 10.0, "M8": 13.0}
SPEC_HEAD_TOLERANCE_MM = 0.2
# 尺寸验证集目录（生成带真值 manifest 的图，用于验证测量精度）
DIMENSION_TEST_DIR = DATA_DIR / "dimension_test"

# ============ 缺陷检测（YOLO / NEU-DET）============
# NEU-DET 6 类缺陷
DEFECT_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface",
                  "rolled_in_scale", "scratches"]
DEFECT_CONF_THRESH = 0.25   # 缺陷置信度阈值
DEFECT_IMG_SIZE = 640

# ============ 检测判定阈值 ============
SPEC_CONF_THRESH = 0.60     # 规格识别置信度，低于此值判定"无法识别"
SPEC_MARGIN_THRESH = 0.10   # top1-top2 概率差低于此值判定"无法识别"（间隔拒识）
DEFECT_AREA_THRESH = 0.0    # 缺陷面积阈值（保留，可扩展）

# ============ 相机/硬件（Phase 1 预留）============
CAMERA_INDEX = 0            # 本地摄像头索引（电脑镜头/手机 RTSP 预留）
CAMERA_API = "cv2"          # cv2 | basler | hikvision

# ============ 中文路径兼容：全局替换 cv2.imread/imwrite ============
# Windows 下 cv2.imread 无法读中文文件名，用 imdecode/imencode 替代。
# 用法：在模块顶部 from config import patch_cv_io; patch_cv_io() 后，
#       该模块的 cv2.imread/cv2.imwrite 自动兼容中文路径。
import cv2 as _cv2
import numpy as _np
from pathlib import Path as _Path


def _safe_imread(path, flags=_cv2.IMREAD_COLOR):
    p = _Path(path)
    try:
        data = _np.fromfile(p, dtype=_np.uint8)
    except Exception:
        return None
    if data.size == 0:
        return None
    return _cv2.imdecode(data, flags)


def _safe_imwrite(path, img, params=None):
    p = _Path(path)
    ext = p.suffix or ".png"
    ok, buf = _cv2.imencode(ext, img, params if params is not None else [])
    if not ok:
        return False
    buf.tofile(p)
    return True


def patch_cv_io():
    """把 cv2.imread / cv2.imwrite 换成中文路径安全版本（幂等）。"""
    if getattr(_cv2, "imread", None) is not _safe_imread:
        _cv2.imread = _safe_imread
    if getattr(_cv2, "imwrite", None) is not _safe_imwrite:
        _cv2.imwrite = _safe_imwrite
    return _cv2
