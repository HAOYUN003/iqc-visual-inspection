# -*- coding: utf-8 -*-
"""验证图纸清单校验结果入库（checklist_json 追溯字段）"""
import sys

sys.path.insert(0, ".")
import src.detection.vision_detector as vd
from src.data import quality_db as db


def main():
    # mock 视觉：全部合格 + 需仪器项
    vd._ask_img = lambda *a, **k: (
        '[{"item":1,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":2,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":3,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":4,"status":"OK","reason":"表面无划伤碰伤毛刺"},'
        '{"item":5,"status":"OK","reason":"棱边无毛刺"},'
        '{"item":6,"status":"UNSURE","reason":"需仪器检测"},'
        '{"item":7,"status":"UNSURE","reason":"需仪器检测"}]')

    res = vd.run_checklist_detection(["data/real/图纸1.jpg"],
                                     drawing_no="R05-03130412-E01", save_image=False)
    assert res["ai_verdict"] == "UNSURE", res["ai_verdict"]
    assert len(res["per_item"]) == 7, f"应有7项, 实际 {len(res['per_item'])}"

    # 入库
    batch = "TEST-CKL-DB-001"
    db.init_db()
    if not db.batch_exists(batch):
        db.create_batch(batch, "DANG-LIAO", "挡料")
    rec_id, verdict = vd.save_record(batch, res, inspector="tester", expected_spec=None)
    print("入库:", rec_id, verdict)

    # 读回验证 checklist 字段
    rec = db.parse_defect_result(db.get_record(rec_id))
    ckl = rec.get("checklist")
    assert isinstance(ckl, list) and len(ckl) == 7, f"checklist 应7项, 实际 {ckl}"
    print("checklist[0]:", ckl[0]["label"], ckl[0]["status"])
    print("checklist[3]:", ckl[3]["label"], ckl[3]["status"], ckl[3]["visual"])
    assert ckl[3]["label"] == "表面状态" and ckl[3]["status"] == "OK"
    assert ckl[0]["status"] == "UNSURE" and ckl[0]["visual"] is False

    # 清理
    with db.get_conn() as conn:
        conn.execute("DELETE FROM inspection_records WHERE batch_id=?", (batch,))
    print("\n[OK] 清单校验入库 + 追溯字段正确")


if __name__ == "__main__":
    main()
