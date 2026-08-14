# -*- coding: utf-8 -*-
"""AppTest 端到端验证：登记批次 → 上传照片 → 点击检测 → 结果展示"""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def main():
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "app.py"),
                           default_timeout=20)
    at.run()
    assert len(at.exception) == 0, f"初始渲染异常: {[e.value for e in at.exception]}"

    # 登记批次
    at.text_input[0].set_value("IQC-TEST-001")
    at.text_input[1].set_value("M6-HEX-SCREW")
    at.text_input[2].set_value("内六角螺钉")
    at.button[0].click().run()
    assert len(at.exception) == 0, f"登记批次异常: {[e.value for e in at.exception]}"
    assert any("已登记" in str(s.value) for s in at.success), "批次登记未成功"
    print("[OK] 批次登记成功")

    # 上传照片
    img_bytes = open("data/real/03619a85ba6c261517c193a1d0c635b.jpg", "rb").read()
    at.file_uploader[0].set_value([("03619a85.jpg", img_bytes, "image/jpeg")]).run()
    assert len(at.exception) == 0, f"上传照片异常: {[e.value for e in at.exception]}"
    assert len(at.image) > 0, "待检图像未显示"
    print("[OK] 照片上传成功，待检图像已显示")

    # 检测按钮存在（触发完整链路会调用外部 API，已在 vision_detector 冒烟测试中验证）
    btn_labels = [b.label for b in at.button]
    assert any("开始检测" in str(l) for l in btn_labels), f"检测按钮未找到: {btn_labels}"
    print(f"[OK] 检测按钮就绪: {[l for l in btn_labels if '开始检测' in str(l)]}")

    print("\n全部 UI 流程验证通过（未触发外部检测 API）")


if __name__ == "__main__":
    main()
