#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""腾讯日 K 线下载器（A 类：日行情 source_returns_daily 候选主路径）。

- 接口: web.ifzq.gtimg.cn/appstock/app/newfqkline/get
- 分段: 按自然年拉取（接口单次行数上限 ~800，8 年需分段）
- 两版: qfq（前复权, adj_return 候选）与不复权（raw_return 候选）
- 断点续传: 已存在文件跳过; 失败重试 3 次后记录失败清单
- 并发: 默认 8 线程
- 取证: 每文件含代码+区间; 日志写 stderr

用法: python scripts/data/fetch_tencent_daily.py --codes <csv> --year 2018
      --start 2018 --end 2025 --workers 8
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "tencent")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"
REF = "https://gu.qq.com/"
URL = ("https://web.ifzq.gtimg.cn/appstock/app/newfqkline/get"
       "?param={code},day,{start},{end},{count},{fq}")


def fetch(code: str, start: str, end: str, fq: str, count: int = 320, retries: int = 3):
    """返回 klines 列表（每行 [date,open,close,high,low,volume,_,pct_chg,amount,_]）。

    接口语义（实测 2026-08）：返回"最新 count 行"——count < 区间交易日数会丢最早行；
    count > 交易日数会向前回溯填充。count=320（> 年交易日 242-244）保证回溯且区间内
    完整，下载后按 [start, end] 过滤去除回溯冗余。
    """
    url = URL.format(code=code, start=start, end=end, count=count, fq=fq)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": REF})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8"))
            data = d.get("data")
            if not isinstance(data, dict):
                return None
            k = data.get(code, {})
            kl = k.get("qfqday", k.get("day", []))
            # 按区间过滤（去除回溯冗余）
            return [row for row in kl if start <= row[0] <= end]
        except Exception:
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="CSV with baostock a_stocks.csv 格式（含 code 列）")
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--versions", default="qfq,raw", help="逗号分隔: qfq / raw")
    ap.add_argument("--dry", action="store_true", help="只打印计划不下载")
    args = ap.parse_args()

    with open(args.codes, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    codes = []
    for r in rows[1:]:
        if not r or not r[0]:
            continue
        c = r[0].strip()
        # baostock sh.600000 -> 腾讯 sh600000
        codes.append(c.replace(".", ""))

    versions = args.versions.split(",")
    plan = []
    for code in codes:
        for ver in versions:
            fq = "qfq" if ver == "qfq" else ""
            outdir = os.path.join(BASE, "daily_qfq" if ver == "qfq" else "daily_raw", code)
            for y in range(args.start, args.end + 1):
                fp = os.path.join(outdir, f"{y}.csv")
                if os.path.exists(fp):
                    continue
                plan.append((code, f"{y}-01-01", f"{y}-12-31", fq, fp))

    print(f"计划下载: {len(codes)} 只 × {len(versions)} 版本, 待拉 {len(plan)} 段", file=sys.stderr)
    if args.dry:
        print(f"dry: 前 5 段 {plan[:5]}")
        return

    failed = []
    t0 = time.time()
    done = 0

    def work(item):
        code, s, e, fq, fp = item
        kl = fetch(code, s, e, fq)
        if kl is None:
            return code, s, False
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["date", "open", "close", "high", "low", "volume", "extra", "pct_chg", "amount", "extra2"])
            w.writerows(kl)
        return code, s, True

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in plan}
        for fut in as_completed(futs):
            code, seg, ok = fut.result()
            done += 1
            if not ok:
                failed.append((code, seg))
            if done % 500 == 0:
                el = time.time() - t0
                print(f"进度 {done}/{len(plan)} ({el:.0f}s, {done/el:.1f} 段/s)", file=sys.stderr)

    print(f"完成: {done - len(failed)}/{done}, 失败 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    if failed:
        with open(os.path.join(BASE, "failed_segments.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code", "segment"])
            w.writerows(failed)
        print(f"失败清单: {os.path.join(BASE, 'failed_segments.csv')}", file=sys.stderr)


if __name__ == "__main__":
    main()
