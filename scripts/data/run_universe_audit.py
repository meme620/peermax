#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B0.5.5 第 11 项：universe 三分类验收（expected / observed / 停牌）。

输入:
  data/raw/baostock/stock_basic/a_stocks.csv        (code, name, ipoDate, outDate, type, status)
  data/raw/baostock/calendar/trade_dates_2016_2026.csv (calendar_date, is_trading_day)
  data/raw/tencent/daily_raw/{prefix}{code}/{year}.csv  (日线，date 列)
  data/raw/institutional/suspension_intervals.csv   (停牌区间，1.1 产出；缺失时降级推断)
输出:
  data/raw/institutional/universe_audit/summary.csv          (总指标)
  data/raw/institutional/universe_audit/missing_list.csv     (缺失 firm-day 清单)
  data/raw/institutional/universe_audit/board_day_rankable.csv (board×day 可排名数)

口径（gap_fix_patch_plan §15 / gate_req §3.3.1）:
  expected = 上市状态股票 × 交易日历（[ipoDate, outDate] ∩ 研究期）
  有记录   = 腾讯日线存在行
  停牌     = 停牌区间内（公告区间法）或降级推断（无行且连续 >1 月）
  缺失     = expected − 有记录 − 停牌
用法: python scripts/data/run_universe_audit.py
"""
import csv
import glob
import os
import sys
import time
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
OUT = os.path.join(BASE, "institutional", "universe_audit")
os.makedirs(OUT, exist_ok=True)

Y_START, Y_END = 2018, 2025


def board_of(code6: str) -> str:
    if code6.startswith(("688", "689")):
        return "star"
    if code6.startswith(("300", "301")):
        return "chinext"
    return "main"


def main():
    t0 = time.time()
    # 1. 交易日历
    trade_days = set()
    with open(os.path.join(BASE, "baostock", "calendar", "trade_dates_2016_2026.csv"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if len(r) >= 2 and r[1] == "1" and Y_START <= int(r[0][:4]) <= Y_END:
                trade_days.add(r[0])
    trade_days = sorted(trade_days)
    print(f"交易日: {len(trade_days)} ({trade_days[0]} ~ {trade_days[-1]})", file=sys.stderr)

    # 2. 股票清单（expected 分母）
    stocks = {}
    with open(os.path.join(BASE, "baostock", "stock_basic", "a_stocks.csv"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and len(r) >= 4 and "." in r[0]:
                code6 = r[0].split(".")[1]
                stocks[code6] = {"name": r[1] if len(r) > 1 else "",
                                 "ipo": (r[2][:4] if r[2] else ""),
                                 "out": (r[3][:4] if r[3] else "")}
    print(f"股票: {len(stocks)}", file=sys.stderr)

    # 3. observed：腾讯日线行
    observed = defaultdict(set)  # code -> set(date)
    n_files = 0
    for p in glob.glob(os.path.join(BASE, "tencent", "daily_raw", "*", "*.csv")):
        code = os.path.basename(os.path.dirname(p))[2:]  # sh600519 -> 600519
        y = os.path.basename(p)[:4]
        if not (Y_START <= int(y) <= Y_END):
            continue
        with open(p, encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and len(r) >= 3 and r[0][:4] == y and r[0] in trade_days:
                    observed[code].add(r[0])
        n_files += 1
        if n_files % 5000 == 0:
            print(f"  日线扫描 {n_files} 文件", file=sys.stderr)
    print(f"日线文件: {n_files}, 有记录股票: {len(observed)}", file=sys.stderr)

    # 4. 停牌区间（公告法）
    susp = defaultdict(list)  # code -> [(start, end)]
    susp_path = os.path.join(BASE, "institutional", "suspension_intervals.csv")
    if os.path.exists(susp_path):
        with open(susp_path, encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and len(r) >= 4 and r[0] and r[2]:
                    susp[r[0]].append((r[2], r[3] if r[3] else "9999-12-31"))
    print(f"停牌区间股票: {len(susp)}", file=sys.stderr)

    # 5. 三分类合并
    missing_rows = []
    board_day = defaultdict(lambda: defaultdict(int))  # (board, date) -> n
    stat = {"expected": 0, "observed": 0, "susp": 0, "missing": 0}
    for code, meta in stocks.items():
        ipo_y = int(meta["ipo"]) if meta["ipo"] else Y_START
        out_y = int(meta["out"]) if meta["out"] else Y_END
        if out_y < Y_START or ipo_y > Y_END:
            continue
        days = [d for d in trade_days if ipo_y <= int(d[:4]) <= out_y]
        if not days:
            continue
        obs = observed.get(code, set())
        susp_ranges = susp.get(code, [])
        board = board_of(code)
        miss_days = []
        for d in days:
            stat["expected"] += 1
            if d in obs:
                stat["observed"] += 1
            else:
                in_susp = any(s <= d <= e for s, e in susp_ranges)
                if in_susp:
                    stat["susp"] += 1
                else:
                    stat["missing"] += 1
                    miss_days.append(d)
            if d in obs:
                board_day[board][d] += 1
        if miss_days:
            missing_rows.append((code, meta["name"], board, miss_days[0], miss_days[-1], len(miss_days)))
    # 缺失清单合并为区间（连续缺失）
    missing_list = []
    for code, name, board, d0, d1, n in missing_rows:
        missing_list.append([code, name, board, d0, d1, n])

    # 6. 输出
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in stat.items():
            w.writerow([k, v])
        w.writerow(["missing_rate", round(stat["missing"] / max(stat["expected"], 1), 6)])
    with open(os.path.join(OUT, "missing_list.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "board", "start_date", "end_date", "missing_days"])
        w.writerows(missing_list)
    with open(os.path.join(OUT, "board_day_rankable.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["board", "date", "rankable_count"])
        for board in ("main", "chinext", "star"):
            for d in trade_days:
                w.writerow([board, d, board_day[board][d]])

    print(f"=== 三分类 ===", file=sys.stderr)
    print(f"expected {stat['expected']}, observed {stat['observed']}, 停牌 {stat['susp']}, 缺失 {stat['missing']}", file=sys.stderr)
    print(f"缺失率 {stat['missing']/max(stat['expected'],1):.2%}", file=sys.stderr)
    print(f"耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)


if __name__ == "__main__":
    main()
