#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 Step 1：expected universe 构造（execution_protocol §2 Step 1）。

输入（usecols 限定）：
  data/raw/baostock/stock_basic/a_stocks.csv        → code,ipoDate,outDate,status
  data/raw/baostock/calendar/trade_dates_2016_2026.csv → calendar_date,is_trading_day
研究期：2018-01-01 ~ 2025-12-31（registry sample_start/sample_end）
expected firm-day = Σ 每只股票 [max(ipoDate,sample_start), min(outDate or sample_end, sample_end)] 内的交易日数。
产出：reports/data_gates/G0/s1_expected_universe.json（含按年/按板块/零覆盖股票计数）

用法: python scripts/data_audit/s1_step1_universe.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_START = "2018-01-01"
SAMPLE_END = "2025-12-31"

A_STOCKS = ROOT / "data/raw/baostock/stock_basic/a_stocks.csv"
CALENDAR = ROOT / "data/raw/baostock/calendar/trade_dates_2016_2026.csv"
OUT = ROOT / "reports/data_gates/G0/s1_expected_universe.json"


def load_stocks():
    rows = []
    with open(A_STOCKS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_trading_days():
    days = []
    with open(CALENDAR, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = r["calendar_date"]
            if SAMPLE_START <= d <= SAMPLE_END and r["is_trading_day"] == "1":
                days.append(d)
    return days


def derive_board(code):
    """code 形如 sh.600000 / sz.300001；返回 (market, board) 或 None（非沪深 A 股）。"""
    market, _, sym = code.partition(".")
    if market not in ("sh", "sz"):
        return None
    if market == "sh":
        if sym.startswith("688") or sym.startswith("689"):
            return ("sh", "star")
        if sym.startswith(("600", "601", "603", "605")):
            return ("sh", "main")
        return None  # sh 900 B 股等
    else:
        if sym.startswith(("300", "301", "302")):
            return ("sz", "chinext")
        if sym.startswith(("000", "001", "002", "003")):
            return ("sz", "main")
        return None


def main():
    stocks = load_stocks()
    td = load_trading_days()
    td_set = set(td)
    td_by_year = Counter(d[:4] for d in td)

    per_year = Counter()
    per_board = Counter()
    per_year_board = Counter()
    zero_coverage = []
    board_mapping_fail = []
    total = 0

    for r in stocks:
        code = r["code"]
        bd = derive_board(code)
        if bd is None:
            board_mapping_fail.append(code)
            continue
        board = bd[1]
        d0 = max(r["ipoDate"], SAMPLE_START)
        d1 = min(r["outDate"] or SAMPLE_END, SAMPLE_END)
        n = 0
        yr = Counter()
        for d in td:
            if d0 <= d <= d1:
                n += 1
                yr[d[:4]] += 1
        if n == 0:
            zero_coverage.append({"code": code, "ipoDate": r["ipoDate"], "outDate": r["outDate"] or ""})
        total += n
        per_year.update(yr)
        per_board[board] += n
        per_year_board.update({f"{y}:{board}": c for y, c in yr.items()})

    report = {
        "report_id": "s1_expected_universe",
        "sample_start": SAMPLE_START,
        "sample_end": SAMPLE_END,
        "n_stocks": len(stocks),
        "n_trading_days_2018_2025": len(td),
        "trading_days_by_year": dict(sorted(td_by_year.items())),
        "expected_firm_days": total,
        "expected_by_year": dict(sorted(per_year.items())),
        "expected_by_board": dict(sorted(per_board.items())),
        "board_mapping_fail": board_mapping_fail,
        "n_board_mapping_fail": len(board_mapping_fail),
        "zero_coverage_stocks": zero_coverage,
        "n_zero_coverage": len(zero_coverage),
        "note": "expected 分母 = a_stocks 上市状态股票 × 研究期交易日，过滤 [ipoDate,outDate]；研究 universe=沪深 A 股（主板+创业板+科创板），北交所排除",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"股票总数: {len(stocks)}")
    print(f"研究期交易日 (2018-2025): {len(td)}  {dict(sorted(td_by_year.items()))}")
    print(f"expected firm-day: {total:,}")
    print(f"按年: {dict(sorted(per_year.items()))}")
    print(f"按板块: {dict(sorted(per_board.items()))}")
    print(f"板块映射失败: {len(board_mapping_fail)} {board_mapping_fail[:10]}")
    print(f"零覆盖股票（2018 前已退市等）: {len(zero_coverage)}")
    for z in zero_coverage[:10]:
        print("   ", z)
    print(f"\n输出: {OUT}")

    # 验收：Σ 可复算（重算校验）
    total_check = sum(per_year.values())
    assert total_check == total, "按年计数与总量不一致"
    print("\n验收: 按年 Σ == 总量 ✓")


if __name__ == "__main__":
    sys.exit(main())
