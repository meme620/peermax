#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 Step 1：ID 标准化与交易日连接。

输出:
  data/processed/firm_id_map.csv          (firm_id, code6, name, board, ipo_date, out_date)
  data/processed/trade_day_coverage.csv   (date, n_stocks_with_rows)
  data/processed/daily_returns_clean/{code6}.csv 追加 board 列（重写，幂等）

board: star=688/689, chinext=300/301, main=其余
用法: python scripts/build_id_map.py
"""
import csv
import glob
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
A_STOCKS = os.path.join(BASE, "baostock", "stock_basic", "a_stocks.csv")
RETURNS = os.path.join(PROC, "daily_returns_clean")
os.makedirs(PROC, exist_ok=True)


def board_of(code6: str) -> str:
    if code6.startswith(("688", "689")):
        return "star"
    if code6.startswith(("300", "301")):
        return "chinext"
    return "main"


def main():
    # 1. firm_id_map
    firms = []
    with open(A_STOCKS, encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and len(r) >= 4 and "." in r[0]:
                prefix, code6 = r[0].split(".")
                firms.append([f"{prefix}{code6}", code6, r[1] if len(r) > 1 else "",
                              board_of(code6), r[2] or "", r[3] or ""])
    with open(os.path.join(PROC, "firm_id_map.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm_id", "code6", "name", "board", "ipo_date", "out_date"])
        w.writerows(firms)
    ids = [x[0] for x in firms]
    assert len(ids) == len(set(ids)), "firm_id 重复"
    print(f"firm_id_map: {len(firms)} 只, 无重复", file=sys.stderr)

    # 2. 日收益表加 board 列（重写）
    board_by_code = {x[1]: x[3] for x in firms}
    n_files = 0
    for p in glob.glob(os.path.join(RETURNS, "*.csv")):
        code6 = os.path.basename(p)[:6]
        if code6 not in board_by_code:
            continue
        with open(p, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue
        hdr = rows[0]
        if "board" in hdr:
            continue  # 幂等
        hdr.append("board")
        for r in rows[1:]:
            r.append(board_by_code[code6])
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows)
        n_files += 1
    print(f"日收益表加 board 列: {n_files} 文件", file=sys.stderr)

    # 3. 交易日覆盖
    coverage = {}
    for p in glob.glob(os.path.join(RETURNS, "*.csv")):
        with open(p, encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and len(r) > 1 and r[1][:4].isdigit():
                    coverage[r[1]] = coverage.get(r[1], 0) + 1
    with open(os.path.join(PROC, "trade_day_coverage.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "n_stocks_with_rows"])
        for d in sorted(coverage):
            w.writerow([d, coverage[d]])
    print(f"trade_day_coverage: {len(coverage)} 交易日, 总行 {sum(coverage.values())}", file=sys.stderr)
    print(f"日收益总行数应 = {sum(coverage.values())}（对照 universe audit observed 8,546,009）", file=sys.stderr)


if __name__ == "__main__":
    main()
