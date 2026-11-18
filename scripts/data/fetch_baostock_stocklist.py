#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baostock 每日全市场证券列表下载（A 类：universe 三分类中停牌核验源）。

- 输入: data/raw/baostock/calendar/trade_dates_2016_2026.csv（is_open=1 的日期）
- 输出: data/raw/baostock/stock_list/YYYYMMDD.csv（code, tradeStatus, code_name）
- 断点续传: 已存在文件跳过
- 注意: baostock 0.9.3 的 get_data() 与 pandas>=2 不兼容，必须用 next() 循环

用法: python scripts/data/fetch_baostock_stocklist.py
"""
import csv
import os
import sys
import time

import baostock as bs

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "baostock")
CAL = os.path.join(BASE, "calendar", "trade_dates_2016_2026.csv")
OUT = os.path.join(BASE, "stock_list")


def main():
    with open(CAL, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    trade_days = [r[0] for r in rows[1:] if len(r) >= 2 and r[1] == "1"
                  and "2018-01-01" <= r[0] <= "2025-12-31"]
    todo = [d.replace("-", "") for d in trade_days if not os.path.exists(os.path.join(OUT, d.replace("-", "") + ".csv"))]
    print(f"交易日 {len(trade_days)}, 待下载 {len(todo)}", file=sys.stderr)

    lg = bs.login()
    print(f"login: {lg.error_code} {lg.error_msg}", file=sys.stderr)
    if lg.error_code != "0":
        sys.exit(1)

    failed = []
    t0 = time.time()
    for i, d in enumerate(todo):
        rs = bs.query_all_stock(day=f"{d[:4]}-{d[4:6]}-{d[6:]}")
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            failed.append(d)
            continue
        with open(os.path.join(OUT, f"{d}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(rs.fields)
            w.writerows(rows)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"进度 {i+1}/{len(todo)} ({el:.0f}s, {(i+1)/el:.1f} 日/s)", file=sys.stderr)
        time.sleep(0.05)

    bs.logout()
    print(f"完成: {len(todo)-len(failed)}/{len(todo)}, 失败 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    if failed:
        with open(os.path.join(OUT, "failed_days.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date"])
            w.writerows([[d] for d in failed])


if __name__ == "__main__":
    main()
