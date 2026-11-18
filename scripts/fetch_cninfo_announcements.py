#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CNINFO 公告索引下载 v2（按股票×年，searchkey 模式；2026-08-09 实测翻页只在 searchkey 搜索模式有效）。

- 接口: www.cninfo.com.cn/new/hisAnnouncement/query
  （searchkey=股票代码, seDate=年度, category 空取全部公告, pageNum 翻页）
- 存储: data/raw/cninfo/announcements/{code}_{year}.csv
- 断点续传: 已存在且行数>50 跳过; 重试 3 次
- 注意: 按日全市场模式 pageNum 失效（每页返回同一页），已弃用
用法: python scripts/fetch_cninfo_announcements.py [--workers 5] [--years 2018-2025]
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "cninfo", "announcements_v2")
CODES = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "baostock", "stock_basic", "a_stocks.csv")
H = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded",
     "Referer": "http://www.cninfo.com.cn/new/index"}


def fetch_stock_year(code: str, year: int, retries=3):
    """拉某股票某年全部公告。返回 (fields, rows)。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    all_rows = []
    fields = None
    for page in range(1, 300):
        body = (f"pageNum={page}&pageSize=50&column=szse&tabName=fulltext&plate=&stock="
                f"&searchkey={code}&secid=&category=&trade=&seDate={year}-01-01~{year}-12-31"
                f"&sortName=&sortType=&isHLtitle=true")
        req = urllib.request.Request("http://www.cninfo.com.cn/new/hisAnnouncement/query", headers=H, data=body.encode())
        try:
            with opener.open(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8", "ignore"))
        except Exception:
            if retries > 0:
                retries -= 1
                time.sleep(2)
                continue
            return None, None
        anns = d.get("announcements") or []
        if not anns:
            break
        if fields is None and anns:
            fields = list(anns[0].keys())
        all_rows.extend([list(a.values()) for a in anns])
        total = d.get("totalRecordNum") or 0
        if len(all_rows) >= total:
            break
    return fields, all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=CODES)
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-codes", type=int, default=0)
    args = ap.parse_args()
    ys = args.years.split("-")
    years = list(range(int(ys[0]), int(ys[1]) + 1))

    with open(args.codes, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    codes = [r[0].split(".")[1] for r in rows[1:] if r and r[0] and "." in r[0]]
    if args.max_codes:
        codes = codes[: args.max_codes]

    todo = []
    for c in codes:
        for y in years:
            fp = os.path.join(BASE, f"{c}_{y}.csv")
            if os.path.exists(fp) and os.path.getsize(fp) > 3000:
                continue
            todo.append((c, y))
    print(f"待下载: {len(todo)} (股票 {len(codes)} × 年 {years})", file=sys.stderr)

    failed = []
    done = 0
    t0 = time.time()

    def work(item):
        c, y = item
        fields, rows = fetch_stock_year(c, y)
        if fields is None:
            return item, False
        fp = os.path.join(BASE, f"{c}_{y}.csv")
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(fields)
            w.writerows(rows)
        return item, True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for fut in as_completed(futs):
            it, ok = fut.result()
            done += 1
            if not ok:
                failed.append(it)
            if done % 300 == 0:
                el = time.time() - t0
                print(f"进度 {done}/{len(todo)} ({el:.0f}s)", file=sys.stderr)

    print(f"完成: {done - len(failed)}/{done}, 失败 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    if failed:
        with open(os.path.join(BASE, "failed_stock_year.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code", "year"])
            w.writerows(failed)


if __name__ == "__main__":
    main()
