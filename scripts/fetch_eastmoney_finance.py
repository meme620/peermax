#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""东财数据中心 F10 主要财务指标下载（B/C 类：净资产/总资产/利润/资产增速，BM/因子输入）。

- 接口: datacenter-web.eastmoney.com/api/data/v1/get
  reportName=RPT_F10_FINANCE_MAINFINADATA（165 字段，含 TOTAL_EQUITY_PK/TOTAL_ASSETS_PK/
  PARENTNETPROFIT/TA_YOYRATIO_PK/BPS/ROE/NOTICE_DATE，逐期 REPORT_DATE 倒序）
- 存储: data/raw/eastmoney/finance/{code}.csv（每只股票 1 文件，全期）
- 断点续传: 已存在跳过; 失败重试 3 次; 失败清单
- 并发: 默认 8

用法: python scripts/fetch_eastmoney_finance.py [--codes path] [--workers 8]
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "eastmoney", "finance")
CODES = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "baostock", "stock_basic", "a_stocks.csv")
H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0",
     "Referer": "https://emweb.securities.eastmoney.com/"}
URL = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
       "?reportName=RPT_F10_FINANCE_MAINFINADATA&columns=ALL"
       "&filter=(SECURITY_CODE%3D%22{c}%22)&pageSize=60&sortColumns=REPORT_DATE&sortTypes=-1")


def fetch(code6: str, retries=3):
    url = URL.format(c=code6)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=H)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8"))
            rows = (d.get("result") or {}).get("data") or []
            return rows
        except Exception:
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=CODES)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-codes", type=int, default=0)
    args = ap.parse_args()

    with open(args.codes, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    codes = []
    for r in rows[1:]:
        if r and r[0]:
            # baostock sh.600000 -> 600000
            codes.append(r[0].split(".")[1])
    if args.max_codes:
        codes = codes[: args.max_codes]

    todo = [c for c in codes if not os.path.exists(os.path.join(BASE, f"{c}.csv"))]
    print(f"计划: {len(todo)}/{len(codes)} 只（跳过已存在）", file=sys.stderr)

    failed = []
    done = 0
    t0 = time.time()

    def work(code):
        data = fetch(code)
        if data is None:
            return code, False
        if not data:
            return code, "empty"
        os.makedirs(BASE, exist_ok=True)
        with open(os.path.join(BASE, f"{code}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(list(data[0].keys()))
            for row in data:
                w.writerow([row.get(k) for k in data[0].keys()])
        return code, True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, c): c for c in todo}
        for fut in as_completed(futs):
            code, ok = fut.result()
            done += 1
            if ok is not True:
                failed.append((code, str(ok)))
            if done % 500 == 0:
                el = time.time() - t0
                print(f"进度 {done}/{len(todo)} ({el:.0f}s, {done/el:.1f}/s)", file=sys.stderr)

    print(f"完成: {done - len(failed)}/{done}, 异常 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    if failed:
        with open(os.path.join(BASE, "failed.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code", "reason"])
            w.writerows(failed)


if __name__ == "__main__":
    main()
