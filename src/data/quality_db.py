# -*- coding: utf-8 -*-
"""
质检数据库层（SQLite）
表结构：
    incoming_batches  来料批次（供应商、料号、批次、数量…）
    inspection_records 检验记录（批次、检验员、时间、图片、AI结果、人工复核…）
    defect_findings   缺陷明细（缺陷类型、框坐标、置信度…）
    ai_training_log   模型再训练记录（回灌数据追踪）

设计说明：
- 用外键关联批次与检验记录，支持"按批次/供应商/料号/时间"追溯。
- 检验记录包含 AI 初判 + 人工复核两套结果，支撑"AI初筛+工程师决策"闭环。
- 图片保存为文件路径（存磁盘），库内只存元数据，避免库膨胀。
"""
import sqlite3
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from config import DB_PATH


@contextmanager
def get_conn():
    """数据库连接上下文管理器"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """初始化所有表"""
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS incoming_batches (
            batch_id      TEXT PRIMARY KEY,          -- 批次号
            material_no   TEXT NOT NULL,             -- 料号
            material_name TEXT,                      -- 名称
            spec          TEXT,                      -- 规格（如 M4）
            category      TEXT DEFAULT 'standard',   -- 类别 standard/parts
            supplier      TEXT,                      -- 供应商
            quantity      INTEGER DEFAULT 0,         -- 数量
            arrival_date  TEXT,                      -- 到货日期
            created_at    TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS inspection_records (
            record_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id     TEXT NOT NULL,
            part_index   INTEGER DEFAULT 0,          -- 批次内第几个件
            inspector    TEXT,                       -- 检验员
            inspected_at TEXT DEFAULT (datetime('now','localtime')),
            image_path   TEXT,                       -- 检测图片路径
            spec_result  TEXT,                       -- AI 规格识别结果（M4/NG...）
            spec_confidence REAL,                    -- 规格置信度
            defect_result TEXT,                      -- 缺陷检测汇总（JSON: {class:count}）
            defect_boxes TEXT,                       -- 缺陷框（JSON 数组）
            ai_verdict   TEXT,                       -- AI 初判 OK/NG/UNSURE
            review_verdict TEXT,                     -- 人工复核 PASS/REJECT/空
            review_note   TEXT,                      -- 复核备注
            review_by     TEXT,                      -- 复核人
            reviewed_at   TEXT,
            FOREIGN KEY (batch_id) REFERENCES incoming_batches(batch_id)
        );

        CREATE TABLE IF NOT EXISTS ai_training_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at   TEXT DEFAULT (datetime('now','localtime')),
            model_type  TEXT,                        -- spec | defect
            samples     INTEGER,                     -- 回灌样本数
            version     TEXT,                        -- 模型版本
            note        TEXT
        );
        """)


# ================= 批次操作 =================

def create_batch(batch_id, material_no, material_name=None, spec=None,
                 category="standard", supplier=None, quantity=0,
                 arrival_date=None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO incoming_batches "
            "(batch_id, material_no, material_name, spec, category, supplier, quantity, arrival_date) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (batch_id, material_no, material_name, spec, category,
             supplier, quantity, arrival_date))


def batch_exists(batch_id):
    with get_conn() as conn:
        row = conn.execute("SELECT batch_id FROM incoming_batches WHERE batch_id=?", (batch_id,)).fetchone()
        return row is not None


def get_batches(filters=None, limit=200):
    """查询批次，filters: dict，支持 supplier/material_no/batch_id/日期范围"""
    sql = "SELECT * FROM incoming_batches WHERE 1=1"
    args = []
    if filters:
        if filters.get("batch_id"):
            sql += " AND batch_id LIKE ?"; args.append(f"%{filters['batch_id']}%")
        if filters.get("material_no"):
            sql += " AND material_no LIKE ?"; args.append(f"%{filters['material_no']}%")
        if filters.get("supplier"):
            sql += " AND supplier LIKE ?"; args.append(f"%{filters['supplier']}%")
        if filters.get("date_from"):
            sql += " AND arrival_date >= ?"; args.append(filters["date_from"])
        if filters.get("date_to"):
            sql += " AND arrival_date <= ?"; args.append(filters["date_to"])
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(limit)
    with get_conn() as conn:
        rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]


# ================= 检验记录操作 =================

def add_record(batch_id, image_path=None, spec_result=None, spec_confidence=None,
               defect_result=None, defect_boxes=None, ai_verdict=None,
               inspector=None, part_index=0):
    """写入一条检验记录。defect_result/defect_boxes 传 dict/list，内部序列化为 JSON。"""
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO inspection_records "
            "(batch_id, part_index, inspector, image_path, spec_result, spec_confidence, "
            " defect_result, defect_boxes, ai_verdict) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (batch_id, part_index, inspector, image_path, spec_result,
             spec_confidence,
             json.dumps(defect_result, ensure_ascii=False) if defect_result else None,
             json.dumps(defect_boxes, ensure_ascii=False) if defect_boxes else None,
             ai_verdict))
        return cur.lastrowid


def review_record(record_id, verdict, note="", reviewer=""):
    """人工复核：verdict = PASS / REJECT"""
    with get_conn() as conn:
        conn.execute(
            "UPDATE inspection_records SET review_verdict=?, review_note=?, review_by=?, "
            "reviewed_at=datetime('now','localtime') WHERE record_id=?",
            (verdict, note, reviewer, record_id))


def get_records(filters=None, limit=500):
    """查询检验记录，filters 支持 batch_id/verdict/date 范围"""
    sql = ("SELECT r.*, b.material_no, b.material_name, b.spec, b.supplier "
           "FROM inspection_records r "
           "LEFT JOIN incoming_batches b ON r.batch_id=b.batch_id WHERE 1=1")
    args = []
    if filters:
        if filters.get("batch_id"):
            sql += " AND r.batch_id = ?"; args.append(filters["batch_id"])
        if filters.get("ai_verdict"):
            sql += " AND r.ai_verdict = ?"; args.append(filters["ai_verdict"])
        if filters.get("review_verdict"):
            sql += " AND r.review_verdict = ?"; args.append(filters["review_verdict"])
        if filters.get("date_from"):
            sql += " AND date(r.inspected_at) >= ?"; args.append(filters["date_from"])
        if filters.get("date_to"):
            sql += " AND date(r.inspected_at) <= ?"; args.append(filters["date_to"])
    sql += " ORDER BY r.record_id DESC LIMIT ?"
    args.append(limit)
    with get_conn() as conn:
        rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]


def get_record(record_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT r.*, b.material_no, b.material_name, b.spec, b.supplier "
            "FROM inspection_records r "
            "LEFT JOIN incoming_batches b ON r.batch_id=b.batch_id "
            "WHERE r.record_id=?", (record_id,)).fetchone()
        return dict(row) if row else None


def parse_defect_result(record):
    """把库里的 JSON 缺陷结果解析成 dict/list"""
    r = dict(record)
    if r.get("defect_result"):
        r["defect_result"] = json.loads(r["defect_result"])
    if r.get("defect_boxes"):
        r["defect_boxes"] = json.loads(r["defect_boxes"])
    return r


# ================= 统计分析 =================

def supplier_stats():
    """按供应商统计：来料批次数、检验件数、NG 数、复检合格率"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT b.supplier,
                   COUNT(DISTINCT b.batch_id) AS batches,
                   COUNT(r.record_id)         AS total,
                   SUM(CASE WHEN r.ai_verdict='NG' THEN 1 ELSE 0 END) AS ng_count
            FROM incoming_batches b
            LEFT JOIN inspection_records r ON r.batch_id=b.batch_id
            WHERE b.supplier IS NOT NULL AND b.supplier != ''
            GROUP BY b.supplier
            ORDER BY total DESC
        """).fetchall()
        return [dict(r) for r in rows]


def verdict_stats():
    """AI 判定分布统计"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT ai_verdict, COUNT(*) AS cnt
            FROM inspection_records GROUP BY ai_verdict
        """).fetchall()
        return {r["ai_verdict"]: r["cnt"] for r in rows}


def defect_type_stats():
    """缺陷类型分布（按批次统计）"""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT defect_result, COUNT(*) AS cnt
            FROM inspection_records WHERE defect_result IS NOT NULL
            GROUP BY defect_result
        """).fetchall()
    # 汇总各缺陷类型出现次数
    from collections import Counter
    counter = Counter()
    for r in rows:
        try:
            d = json.loads(r["defect_result"])
            if isinstance(d, dict):
                counter.update(d)
        except Exception:
            pass
    return dict(counter)


def log_training(model_type, samples, version, note=""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO ai_training_log (model_type, samples, version, note) VALUES (?,?,?,?)",
            (model_type, samples, version, note))
