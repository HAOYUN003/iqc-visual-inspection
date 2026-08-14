# -*- coding: utf-8 -*-
"""验证 run_machined_detection 的多图聚合逻辑（mock 视觉，不触发真实 API）"""
import sys

sys.path.insert(0, ".")
import src.detection.vision_detector as vd


def main():
    # mock 视觉调用：返回预置缺陷 JSON
    calls = []
    def fake_ask(img_bgr, image_path, prompt):
        calls.append(image_path)
        n = len(calls)
        if n == 1:
            return '{"scratch": 0, "burr": 2, "dent": 0, "crack": 0, "rust": 0, "pit": 1}'
        elif n == 2:
            return '{"scratch": 1, "burr": 0, "dent": 0, "crack": 0, "rust": 0, "pit": 0}'
        else:
            return '{"scratch": 0, "burr": 0, "dent": 0, "crack": 0, "rust": 0, "pit": 0}'
    vd._ask_img = fake_ask

    # 三张角度照片
    images = [("内存图1", None), ("内存图2", None), ("内存图3", None)]
    res = vd.run_machined_detection(images, save_image=False)
    print("判定:", res["ai_verdict"])
    print("汇总:", res["defect_summary"])
    print("原因:", res["reasons"])
    print("逐张:", res["per_image"])
    print("image_path:", res["image_path"])

    # 断言：burr×2 + scratch×1 + pit×1 → NG
    assert res["ai_verdict"] == "NG", "应判 NG"
    assert res["defect_summary"] == {"burr": 2, "scratch": 1, "pit": 1}, res["defect_summary"]
    assert "burr×2" in res["reasons"], "应含 burr×2"
    print("\n[OK] 加工件多图聚合逻辑正确")


if __name__ == "__main__":
    main()
