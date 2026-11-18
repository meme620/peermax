#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 Phase B′ 契约测试：老师 MD&A 语料面板（teacher_mda_panel.csv）。

运行：python tests/test_s3_teacher_corpus_contracts.py
对应执行契约：methodology_execution_plan.md §3 S3「语料来源与版本」块。

8 项（与自建语料的 test_s3_mda_contracts.py 对称，但判据针对"外部导出"这一事实调整：
无原文 PDF 层、无 CID/锚点概念，改测 hash 来源、universe 映射与行数对账）：
  1. 面板存在且列齐全
  2. 唯一键 (firm_id, year) 无重复
  3. 与原始 CSV 逐行对账（保留行数 = 读入 − 北交所 − 未匹配）
  4. 年份范围 ⊂ [2014, 2022]
  5. 无北交所代码（src_code 不以 8/4 开头）
  6. firm_id 全部落在 firm_id_map（universe）
  7. 长度门：length == len(text) 且 length >= 500
  8. status 全为 ok 且 text 非空
"""
from __future__ import annotations

import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/teacher_mda_panel.csv"
RAW = ROOT / "data/raw/teacher/mda_data_2023.csv"
IDMAP = ROOT / "data/processed/firm_id_map.csv"
INGEST_REP = ROOT / "reports/data_gates/G0/s3_bp0_corpus_ingest.json"

COLS = ["firm_id", "year", "text", "src_code", "src_year", "length", "status", "reason"]
YEAR_LO, YEAR_HI = 2014, 2022
MIN_LEN = 500
BSE_PREFIX = ("8", "4")

csv.field_size_limit(10**9)


def _rows():
    with open(PANEL, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_panel_exists_and_columns():
    assert PANEL.exists(), f"缺少 {PANEL}（先运行 scripts/data_audit/s3_teacher_corpus_ingest.py）"
    with open(PANEL, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == COLS, f"列漂移: {header}"


def test_unique_key_and_size():
    rows = _rows()
    assert rows, "面板为空"
    keys = [(r["firm_id"], r["year"]) for r in rows]
    dup = {k for k in keys if keys.count(k) > 1}
    assert not dup, f"唯一键冲突: {list(dup)[:5]}"
    assert len(rows) > 30_000, f"行数异常偏低: {len(rows)}"


def test_raw_reconciliation():
    """面板保留行数 = 原始 CSV 读入 − 北交所 − 未匹配（与 B′0 报告一致）。"""
    idmap = set()
    with open(IDMAP, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            idmap.add(str(r["code6"]).zfill(6))
    n_read = n_bse = n_unmatched = 0
    with open(RAW, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            n_read += 1
            code = (r.get("f_stock_id") or "").strip()
            if code.startswith(BSE_PREFIX):
                n_bse += 1
            elif code.zfill(6) not in idmap:
                n_unmatched += 1
    expected = n_read - n_bse - n_unmatched
    rows = _rows()
    assert len(rows) == expected, f"行数对账失败: 面板 {len(rows)} vs 期望 {expected}"


def test_year_range():
    bad = sorted({r["year"] for r in _rows() if not (YEAR_LO <= int(r["year"]) <= YEAR_HI)})
    assert not bad, f"越界年份: {bad}"


def test_no_bse_codes():
    bad = [r["src_code"] for r in _rows() if r["src_code"].startswith(BSE_PREFIX)]
    assert not bad, f"含北交所代码: {bad[:5]}"


def test_universe_mapping():
    idmap = set()
    with open(IDMAP, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            idmap.add(r["firm_id"])
    miss = sorted({r["firm_id"] for r in _rows() if r["firm_id"] not in idmap})
    assert not miss, f"firm_id 不在 universe: {miss[:5]}"


def test_length_gate():
    bad = [r["src_code"] for r in _rows()
           if int(r["length"]) < MIN_LEN or int(r["length"]) != len(r["text"])]
    assert not bad, f"长度门失败（length 与 text 不符或过短）: {bad[:5]}"


def test_status_and_text():
    bad = [r["src_code"] for r in _rows() if r["status"] != "ok" or not (r["text"] or "").strip()]
    assert not bad, f"status/文本异常: {bad[:5]}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            ok += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
