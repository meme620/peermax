#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2.3 月收益：R_{i,t} = Π(1+rraw) - 1（valid_trade=1 连乘）。

输出: data/processed/monthly_returns_clean.csv
口径: execution_plan §5.4 Step 3；min_valid_trade_days 候选 15（登记不冻结）
"""
import csv
import glob
import os
import sys
from collections import defaultdict

RET = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "daily_returns_clean")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MIN_DAYS = 15  # min_valid_trade_days 候选

os.makedirs(OUT, exist_ok=True)


def main():
    out_rows = []
    stat = {"PASS": 0, "CONDITIONAL": 0, "BLOCKED": 0}
    n_files = 0
    for p in glob.glob(os.path.join(RET, "*.csv")):
        code = os.path.basename(p)[:6]
        # month -> [rraw, ...]（valid_trade=1）
        months = defaultdict(list)
        with open(p, encoding="utf-8") as f:
            for r in csv.reader(f):
                if not r or len(r) < 3 or not r[1][:4].isdigit():
                    continue
                if r[5] == "1" and r[2]:
                    months[r[1][:7]].append(float(r[2]))
        for m in sorted(months):
            rrs = months[m]
            n = len(rrs)
            prod = 1.0
            for x in rrs:
                prod *= (1 + x)
            R = prod - 1
            if n >= MIN_DAYS:
                status = "PASS"
            elif n >= 5:
                status = "CONDITIONAL"
            else:
                status = "BLOCKED"
            stat[status] += 1
            out_rows.append([f"sh{code}" if code.startswith(("6", "9")) else f"sz{code}", m, R, n, status, ""])
        n_files += 1
        if n_files % 1000 == 0:
            print(f"进度 {n_files} 文件", file=sys.stderr)

    with open(os.path.join(OUT, "monthly_returns_clean.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm_id", "month", "R", "n_trading_days", "monthly_return_status", "missing_reason"])
        w.writerows(out_rows)
    print(f"月收益: {len(out_rows)} 月, {stat}", file=sys.stderr)


if __name__ == "__main__":
    main()
