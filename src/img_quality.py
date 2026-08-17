# -*- coding: utf-8 -*-
"""
图像质量预检：拍照是否规范（太暗/太糊/零件太小）。
用于检验员端上传后提醒重拍，减少因拍照不规范导致的误判。
"""
import cv2
import numpy as np


def check_image_quality(img_bgr):
    """
    检查图像质量，返回 (issues, tips)。
    issues: list[str]，每个是一句提醒；空列表=通过。
    检查项：
      - 过暗 / 过亮
      - 模糊（拉普拉斯方差）
      - 对比度太低（无法区分零件/背景）
    """
    issues = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1) 亮度
    mean = gray.mean()
    if mean < 50:
        issues.append("照片过暗，请增加照明")
    elif mean > 220:
        issues.append("照片过亮（过曝），请减少强光")

    # 2) 清晰度（拉普拉斯方差，越大越清晰）
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap < 30:
        issues.append("照片模糊，请对准焦距、稳持拍摄")

    # 3) 对比度（标准差衡量，太小说明零件/背景难分）
    std = gray.std()
    if std < 25:
        issues.append("对比度低，零件与背景难区分，请调整摆放与光照")

    tips = [
        "拍摄提示：零件水平摆放，占画面 60%~80%",
        "卡尺与零件同框，读数清晰可辨",
        "光照均匀，避免强反光与阴影",
        "对焦清晰后稳定拍摄",
    ]
    return issues, tips


def check_part_size_ratio(img_bgr, min_ratio=0.3):
    """
    检查零件是否太小（占画面比例过低，导致尺寸计算失真）。
    用简单分割估算前景占比。min_ratio: 零件至少占画面比例。
    返回 (ok: bool, ratio: float, issue: str)。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Otsu 分割出前景（零件）
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 取较大的一侧作为前景（零件可能是亮或暗）
    white = (th == 255).sum()
    total = th.size
    ratio = max(white, total - white) / total
    if ratio < min_ratio:
        return False, ratio, f"零件在画面中占比偏小（约{ratio:.0%}），请靠近拍摄，避免尺寸失真"
    return True, ratio, ""
