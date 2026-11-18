#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""东财龙虎榜每日明细下载（G4 assignment/listing audit 层）。

- 接口: datacenter-web.eastmoney.com/api/data/v1/get
  reportName=RPT_DAILYBILLBOARD_DETAILSNEW（TRADE_DATE 过滤）
- 存储: data/raw/eastmoney/dragon_tiger/{YYYYMMDD}.csv
- 断点续传: 已存在跳过; 重试 3 次
用法: python scripts/fetch_eastmoney_dragon_tiger.py [--workers 6]
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "eastmoney", "dragon_tiger")
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"}
URL = ("https://datacenter-web.eastmoney.com/api/data/v1/get"
       "?reportName=RPT_DAILYBILLBOARD_DETAILSNEW&columns=ALL"
       "&filter=(TRADE_DATE%3D%27{d}%27)&pageSize=500&pageNumber=1")


def fetch_day(date_str: str, retries=3):
    """date_str: 2024-01-02。返回 (fields, rows) 或 (None, None)。"""
    url = URL.format(d=date_str)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=H)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8", "ignore"))
            rows = (d.get("result") or {}).get("data") or []
            if rows:
                return list(rows[0].keys()), rows
            if d.get("code") == 0:
                return [], []  # 当日无上榜（合法空，写表头文件）
        except Exception:
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    # 交易日历（baostock 已下载；只取 is_open=1）
    cal = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "baostock", "calendar", "trade_dates_2016_2026.csv")
    with open(cal, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    days = [r[0] for r in rows[1:] if len(r) >= 2 and r[1] == "1"]
    todo = [d for d in days if args.start <= d <= args.end and not os.path.exists(
        os.path.join(BASE, d.replace("-", "") + ".csv"))]
    print(f"待下载: {len(todo)} 交易日", file=sys.stderr)

    failed = []
    done = 0
    t0 = time.time()

    def work(d):
        fields, rows = fetch_day(d)
        if fields is None:
            return d, False
        os.makedirs(BASE, exist_ok=True)
        with open(os.path.join(BASE, d.replace("-", "") + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(fields)
            # rows 是 dict 列表：writerows 直接写 dict 会输出键（2026-08-09 bug，曾致全表头）
            w.writerows([[row.get(k) for k in fields] for row in rows])
        return d, True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, d): d for d in todo}
        for fut in as_completed(futs):
            d, ok = fut.result()
            done += 1
            if not ok:
                failed.append(d)
            if done % 200 == 0:
                el = time.time() - t0
                print(f"进度 {done}/{len(todo)} ({el:.0f}s)", file=sys.stderr)

    print(f"完成: {done - len(failed)}/{done}, 失败 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    if failed:
        with open(os.path.join(BASE, "failed_days.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date"])
            w.writerows([[d] for d in failed])


if __name__ == "__main__":
    main()
