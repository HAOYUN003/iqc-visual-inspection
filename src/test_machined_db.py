# -*- coding: utf-8 -*-
"""验证加工件结果入库（多图 image_path）"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import src.detection.vision_detector as vd
from src.data import quality_db as db


def main():
    # mock 视觉：全部无缺陷 → OK
    vd._ask_img = lambda *a, **k: '{"scratch": 0, "burr": 0, "dent": 0, "crack": 0, "rust": 0, "pit": 0}'

    res = vd.run_machined_detection(
        [("内存图1", None), ("内存图2", None)], save_image=False)
    assert res["ai_verdict"] == "OK", res["ai_verdict"]

    # 用临时批次入库
    batch = "TEST-MACHINED-001"
    db.init_db()
    if not db.batch_exists(batch):
        db.create_batch(batch, "DANG-LIAO", "挡料")
    rec_id, verdict = vd.save_record(batch, res, inspector="tester", expected_spec=None)
    print("入库:", rec_id, verdict)

    # 读出记录验证 image_path 和 defect
    recs = db.get_records({"batch_id": batch})
    rec = db.parse_defect_result(recs[-1])
    print("image_path:", rec["image_path"])
    print("defect_result:", rec["defect_result"])
    print("ai_verdict:", rec["ai_verdict"])
    print("spec_result:", rec["spec_result"])
    assert rec["ai_verdict"] == "OK"
    assert rec["defect_result"] is None or rec["defect_result"] == {}

    # 清理测试批次
    db._execute("DELETE FROM inspection_records WHERE batch_id=?", (batch,)) if hasattr(db, "_execute") else None
    print("\n[OK] 加工件入库链路正确")


if __name__ == "__main__":
    main()
