#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""baostock 财务数据下载（B 类：BM/控制变量输入候选）。

- 接口: query_profit_data / query_balance_data / query_growth_data / query_operation_data
  / query_cash_flow_data / query_dupont_data（季频，含 pubDate）
- 存储: data/raw/baostock/financial/{interface}/{code}_{year}q{quarter}.csv
- 断点续传: 已存在文件跳过; 每请求超时重试 3 次（退避）; 失败清单记录
- 并发: 默认 4（baostock 服务不稳定，保守）
- 注意: baostock 0.9.3 get_data() 与 pandas>=2 不兼容，用 next() 循环

用法: python scripts/fetch_baostock_financial.py --interfaces profit,balance
      --years 2018-2025 --quarters 1-4 --workers 4
"""
import argparse
import csv
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import baostock as bs

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "baostock", "financial")
CODES = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "baostock", "stock_basic", "a_stocks.csv")
INTERFACES = {
    "profit": bs.query_profit_data,
    "balance": bs.query_balance_data,
    "growth": bs.query_growth_data,
    "operation": bs.query_operation_data,
    "cashflow": bs.query_cash_flow_data,
    "dupont": bs.query_dupont_data,
}
_lock = threading.Lock()


def fetch_one(interface, code, year, quarter, retries=4):
    fn = INTERFACES[interface]
    for i in range(retries):
        try:
            rs = fn(code=code, year=year, quarter=quarter)
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            # 空 rows 视为失败（2026-08-09 实测服务降级时 error_code=0 但返回空，
            # 曾导致 17 万个空壳文件）；error_code 非 0 同样重试
            if not rows or (rs.error_code and rs.error_code != "0"):
                if i < retries - 1:
                    time.sleep(2 * (i + 1))
                continue
            return rs.fields, rows
        except Exception:
            if i < retries - 1:
                time.sleep(2 * (i + 1))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interfaces", default="profit,balance", help="逗号分隔")
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--quarters", default="1,2,3,4")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-codes", type=int, default=0, help="0=全部")
    args = ap.parse_args()

    interfaces = args.interfaces.split(",")
    ys = args.years.split("-")
    years = list(range(int(ys[0]), int(ys[1]) + 1))
    quarters = [int(q) for q in args.quarters.split(",")]

    with open(CODES, encoding="utf-8") as f:
        codes = [r[0] for r in csv.reader(f)][1:]
    if args.max_codes:
        codes = codes[: args.max_codes]

    plan = []
    for c in codes:
        for iface in interfaces:
            for y in years:
                for q in quarters:
                    fp = os.path.join(BASE, iface, f"{c}_{y}q{q}.csv")
                    if not os.path.exists(fp):
                        plan.append((iface, c, y, q, fp))

    print(f"计划: {len(codes)} 只 × {interfaces} × {len(years)} 年 × {len(quarters)} 季 = {len(plan)} 请求", file=sys.stderr)
    lg = bs.login()
    print(f"login: {lg.error_code} {lg.error_msg}", file=sys.stderr)
    if lg.error_code != "0":
        sys.exit(1)

    failed = []
    done = 0
    t0 = time.time()

    def work(item):
        iface, c, y, q, fp = item
        fields, rows = fetch_one(iface, c, y, q)
        if fields is None:
            return item, False
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(fields)
            w.writerows(rows)
        return item, True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in plan}
        for fut in as_completed(futs):
            item, ok = fut.result()
            done += 1
            if not ok:
                failed.append(item)
            if done % 200 == 0:
                el = time.time() - t0
                print(f"进度 {done}/{len(plan)} ({el:.0f}s, {done/el:.1f}/s)", file=sys.stderr)

    bs.logout()
    print(f"完成: {done - len(failed)}/{done}, 失败 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    if failed:
        with open(os.path.join(BASE, "failed_requests.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["interface", "code", "year", "quarter"])
            w.writerows(failed)
        print(f"失败清单: {os.path.join(BASE, 'failed_requests.csv')}", file=sys.stderr)


if __name__ == "__main__":
    main()
