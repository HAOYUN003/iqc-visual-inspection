# -*- coding: utf-8 -*-
"""
IQC 真实照片完整检测链路：
  照片 → Qwen-VL 读卡尺读数（多次采样取一致值）
       → 对照规格表最近匹配 → 型号
       → 输出检测报告

读数口径：卡尺外接圆直径(mm)，规格 = 最近规格表匹配。
"""
import glob
import json
import os
import sys
import re

sys.path.insert(0, str(os.path.dirname(os.path.abspath(__file__))))
from config import SPEC_NOMINAL_HEAD_MM
from ask_vision import ask

READINGS_SAMPLES = 3
PROMPT = (
    "数显卡尺显示屏上显示的测量值是多少mm？"
    "注意：这是一个小数字（例如 9.67），一位或两位整数加小数点后两位。"
    "务必看清小数点位置。只回答数字本身，例如 9.67。不要任何说明。"
)
# 针对"个位数读数被误读成两位数"的补充提示（如 5.37 被读成 53.7）
PROMPT_SINGLE = (
    "数显卡尺LCD上显示的数值是多少mm？"
    "数值可能是个位数（例如 5.37），也可能是十位数的两位数（例如 12.70）。"
    "请先确定整数部分是几位数，再精确读出整个数字。只回答数字本身。"
)


def parse_reading(text):
    """从视觉返回文本提取数字，处理可能的小数点误读"""
    m = re.search(r"(\d+)[.,](\d+)", text.replace(" ", ""))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"(\d+\.\d+)", text)
    if m:
        return float(m.group(1))
    return None


# 规格域：M3~M8 头径 5.5~13mm，留缓冲过滤小数点错位误读
READING_RANGE_MM = (4.5, 14.5)


def sample_reading(image_path, reader=None, range_mm=READING_RANGE_MM):
    """混合采样读数：通用提示 + 个位数提示，综合投票。
    reader: 可调用对象 reader(prompt)->text，用于自定义取图方式（如内存图），
            为 None 时用 ask(image_path, prompt)。
    规格域约束：过滤明显超出范围的误读（如 5.37 被读成 53.7/15.97）。"""
    if reader is None:
        reader = lambda p: ask(image_path, p)
    readings = []
    prompts = [PROMPT] * READINGS_SAMPLES + [PROMPT_SINGLE] * READINGS_SAMPLES
    for p in prompts:
        try:
            txt = reader(p)
            v = parse_reading(txt)
            if v is not None:
                v = round(v, 2)
                if range_mm[0] <= v <= range_mm[1]:
                    readings.append(v)
        except Exception:
            pass
    if not readings:
        return None, []
    from collections import Counter
    cnt = Counter(readings)
    # 取最高票，若平票取最接近规格域的
    most_common = cnt.most_common(1)[0][0]
    return most_common, sorted(cnt.items(), key=lambda x: -x[1])


def infer_spec_by_reading(mm):
    """最近规格匹配"""
    if mm is None:
        return None, None
    best = min(SPEC_NOMINAL_HEAD_MM, key=lambda s: abs(SPEC_NOMINAL_HEAD_MM[s] - mm))
    dist = abs(SPEC_NOMINAL_HEAD_MM[best] - mm)
    return best, dist


def main():
    real_dir = "data/real"
    # 只保留原始照片：hash 命名的 jpg，排除下划线/裁剪/标注等中间产物
    photos = [p for p in sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
              if not os.path.basename(p).startswith("_")
              and "lcd" not in os.path.basename(p)
              and "head" not in os.path.basename(p)]
    report = []
    print(f"共 {len(photos)} 张照片，每张采样 {READINGS_SAMPLES} 次\n")
    for p in photos:
        name = os.path.basename(p)[:16]
        print(f"=== {name} ===")
        reading, hist = sample_reading(p)
        if reading is None:
            print("  读数失败（视觉无法识别）")
            report.append({"file": name, "reading_mm": None, "spec": None, "error": "read_failed"})
            continue
        spec, dist = infer_spec_by_reading(reading)
        # 判定：读数与名义值偏差
        verdict = "OK"
        reason = f"读数 {reading}mm → 最近规格 {spec}（名义 {SPEC_NOMINAL_HEAD_MM.get(spec)}mm，偏差 {dist:.2f}mm）"
        if dist > 1.5:
            verdict = "UNSURE"
        print(f"  读数={reading}mm (采样历史: {hist})")
        print(f"  规格={spec}  偏差={dist:.2f}mm  判定={verdict}")
        print(f"  → {reason}")
        report.append({
            "file": name, "reading_mm": reading, "samples": dict(hist),
            "spec": spec, "dist_mm": round(dist, 2),
            "nominal_mm": SPEC_NOMINAL_HEAD_MM.get(spec), "verdict": verdict,
        })
        print()

    out = os.path.join(real_dir, "vision_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main()
