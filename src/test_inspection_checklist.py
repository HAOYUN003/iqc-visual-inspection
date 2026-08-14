# -*- coding: utf-8 -*-
"""验证图纸清单逐项校验逻辑（mock 视觉，不触发真实 API）"""
import sys

sys.path.insert(0, ".")
import src.detection.vision_detector as vd


def mock_ask(img_bgr, image_path, prompt):
    """模拟 Qwen-VL 返回：表面 OK、倒钝 OK，其余 UNSURE。"""
    return (
        '[{"item":1,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":2,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":3,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":4,"status":"OK","reason":"表面无划伤碰伤毛刺"},'
        '{"item":5,"status":"OK","reason":"棱边无毛刺"},'
        '{"item":6,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":7,"status":"UNSURE","reason":"需仪器检测"}]'
    )


def mock_ask_ng(img_bgr, image_path, prompt):
    """模拟视觉发现划伤 → NG。"""
    return (
        '[{"item":1,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":2,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":3,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":4,"status":"NG","reason":"发现划伤"},'
        '{"item":5,"status":"OK","reason":"棱边无毛刺"},'
        '{"item":6,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":7,"status":"UNSURE","reason":"需仪器检测"}]'
    )


def test_pass():
    vd._ask_img = mock_ask
    res = vd.run_checklist_detection([("内存图", None)], drawing_no="R05-03130412-E01",
                                     save_image=False)
    print("判定:", res["ai_verdict"])
    print("原因:", res["reasons"])
    print("清单:", res["checklist_no"], res["checklist_name"])
    assert res["ai_verdict"] == "UNSURE", "全 OK+需仪器项 → UNSURE"
    assert res["spec_method"] == "vision_checklist"
    # surface/deburr 可视觉验证，应 OK
    surf = [r for r in res["per_item"] if r["id"] == "surface"][0]
    deb = [r for r in res["per_item"] if r["id"] == "deburr"][0]
    assert surf["status"] == "OK" and surf["visual"] is True
    assert deb["status"] == "OK" and deb["visual"] is True
    # 需仪器项应 UNSURE
    hard = [r for r in res["per_item"] if r["id"] == "hardness"][0]
    assert hard["status"] == "UNSURE" and hard["visual"] is False
    assert "需仪器检测" in hard["reason"]
    assert res["defect_summary"] is None, "无缺陷时 summary 应为 None"
    print("[OK] 全部合格 + 需仪器项 → UNSURE")


def test_ng():
    vd._ask_img = mock_ask_ng
    res = vd.run_checklist_detection([("内存图", None)], drawing_no="R05-03130412-E01",
                                     save_image=False)
    print("\n判定:", res["ai_verdict"])
    print("原因:", res["reasons"])
    assert res["ai_verdict"] == "NG", "表面划伤 → NG"
    assert res["defect_summary"] == {"表面状态": 1}, res["defect_summary"]
    ng = [r for r in res["per_item"] if r["id"] == "surface"][0]
    assert ng["status"] == "NG"
    print("[OK] 表面划伤 → NG")


def test_missing_listing():
    vd._ask_img = mock_ask
    try:
        vd.run_checklist_detection([("内存图", None)], drawing_no="NONEXIST", save_image=False)
        assert False, "应抛异常"
    except FileNotFoundError as e:
        print(f"\n[OK] 未找到清单正确抛错: {e}")


if __name__ == "__main__":
    test_pass()
    test_ng()
    test_missing_listing()
    print("\n全部清单校验逻辑测试通过")
