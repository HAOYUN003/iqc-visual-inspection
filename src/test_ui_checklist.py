# -*- coding: utf-8 -*-
"""AppTest 验证加工件模式关联图纸清单：选择图纸 → 上传 → 检测按钮就绪"""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")


def main():
    at = AppTest.from_file(APP, default_timeout=20)
    at.run()
    assert len(at.exception) == 0, f"初始异常: {[e.value for e in at.exception]}"

    # 登记批次
    at.text_input[0].set_value("IQC-CKL-001")
    at.text_input[1].set_value("DANG-LIAO")
    at.text_input[2].set_value("挡料")
    at.button[0].click().run()
    assert any("已登记" in str(s.value) for s in at.success), "批次登记未成功"
    print("[OK] 批次登记成功")

    # 切到加工件模式
    at.radio[0].set_value("加工件（多角度表面缺陷）").run()
    assert len(at.exception) == 0, f"切模式异常: {[e.value for e in at.exception]}"
    print("[OK] 加工件模式可用")

    # 选择关联图纸清单（精确定位"关联图纸"下拉框）
    sel = None
    for s in at.selectbox:
        if "关联图纸" in str(s.label):
            sel = s
            break
    assert sel is not None, f"关联图纸下拉框缺失: {[s.label for s in at.selectbox]}"
    assert len(sel.options) >= 2, f"应至少有关联图纸选项: {sel.options}"
    sel.set_value(sel.options[1])  # 第一个图纸（不关联图纸之外）
    at.run()
    assert len(at.exception) == 0, f"选图纸异常: {[e.value for e in at.exception]}"
    print("[OK] 可关联图纸清单:", sel.options[1])

    # 多图上传（2张）
    img_bytes = open("data/real/图纸1.jpg", "rb").read()
    at.file_uploader[0].set_value([
        ("dang1.jpg", img_bytes, "image/jpeg"),
        ("dang2.jpg", img_bytes, "image/jpeg"),
    ]).run()
    assert len(at.exception) == 0, f"多图上传异常: {[e.value for e in at.exception]}"
    n_img = len(at.image)
    assert n_img >= 2, f"应显示至少2张待检图, 实际 {n_img}"
    print(f"[OK] 多图上传成功，显示 {n_img} 张待检图像")

    # 检测按钮存在（触发将调用外部 API，跳过）
    btn_labels = [b.label for b in at.button]
    assert any("开始检测" in str(l) for l in btn_labels), f"检测按钮缺失: {btn_labels}"
    print("[OK] 检测按钮就绪（触发将调用外部 API，跳过）")

    print("\n加工件+图纸清单 UI 流程验证通过（未触发外部 API/数据库写入）")


if __name__ == "__main__":
    main()
