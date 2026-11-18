#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""东财公告补漏（D 类：CNINFO 代码搜索关联缺陷的补充源）。

- 输入: failed_annual.csv（code, year, reason=no_annual）
- 查询: np-anotice-stock.eastmoney.com/api/security/ann
  （stock_list=CODE, begin_time=y-01-01, end_time=y+1-06-30, 翻页找标题含 "{y}年年度报告"）
- 下载: pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf
- 提取: pymupdf（与 CNINFO 脚本同）
- 存储: 同 data/raw/cninfo/annual_pdf|annual_txt（合并）

用法: python scripts/data/fetch_eastmoney_annual.py [--failed path] [--workers 6]
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pymupdf

BASE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "cninfo")
FAILED = os.path.join(BASE, "combined_failed.csv")
H = {"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/notices/"}
ANN_URL = ("https://np-anotice-stock.eastmoney.com/api/security/ann"
           "?sr=-1&page_size=50&page_index={p}&ann_type=A&client_source=web"
           "&stock_list={code}&begin_time={b}&end_time={e}")


def find_annual(code: str, year: int, max_pages=20):
    """在东财公告里找 {year} 年报，返回 art_code 或 None。"""
    b = f"{year}-01-01"
    e = f"{year + 1}-06-30"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for p in range(1, max_pages + 1):
        url = ANN_URL.format(p=p, code=code, b=b, e=e)
        req = urllib.request.Request(url, headers=H)
        try:
            with opener.open(req, timeout=20) as r:
                d = json.loads(r.read().decode("utf-8", "ignore"))
        except Exception:
            time.sleep(1)
            continue
        lst = d.get("data", {}).get("list") or []
        if not lst:
            return None
        for a in lst:
            t = a.get("title", "")
            if (f"{year}年年度报告" in t and "摘要" not in t and "英文" not in t
                    and "提示性公告" not in t and "披露" not in t
                    and "更正" not in t and "补充" not in t and "之" not in t):
                return a.get("art_code")
    return None


def download_extract(code: str, year: int, art_code: str) -> bool:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    pdf_url = f"https://pdf.dfcfw.com/pdf/H2_{art_code}_1.pdf"
    req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
    with opener.open(req, timeout=120) as r:
        pdf = r.read()
    if pdf[:4] != b"%PDF":
        return False
    pdir = os.path.join(BASE, "annual_pdf", code)
    tdir = os.path.join(BASE, "annual_txt", code)
    os.makedirs(pdir, exist_ok=True)
    os.makedirs(tdir, exist_ok=True)
    pdfp = os.path.join(pdir, f"{year}.pdf")
    with open(pdfp, "wb") as f:
        f.write(pdf)
    txtp = os.path.join(tdir, f"{year}.txt")
    doc = pymupdf.open(pdfp)
    full = "".join(page.get_text() for page in doc)
    doc.close()
    if not full.strip():
        return False
    with open(txtp, "w", encoding="utf-8") as f:
        f.write(full)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed", default=FAILED)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-pages", type=int, default=20, help="东财翻页上限（公告多的股票年报在深页）")
    args = ap.parse_args()

    with open(args.failed, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    todo = []
    for r in rows[1:]:
        if len(r) >= 3 and r[2].startswith("no_annual"):
            txt = os.path.join(BASE, "annual_txt", r[0], f"{r[1]}.txt")
            if not os.path.exists(txt):
                todo.append((r[0], int(r[1])))
    print(f"待补: {len(todo)} 项", file=sys.stderr)

    failed = []
    done = 0
    t0 = time.time()

    def work(item):
        code, y = item
        try:
            art = find_annual(code, y, max_pages=args.max_pages)
            if art is None:
                return code, y, "not_found"
            if download_extract(code, y, art):
                return code, y, True
            return code, y, "download_extract_fail"
        except Exception as e:
            return code, y, f"err:{str(e)[:50]}"

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, it): it for it in todo}
        for fut in as_completed(futs):
            code, y, ok = fut.result()
            done += 1
            if ok is not True:
                failed.append((code, y, ok))
            if done % 200 == 0:
                el = time.time() - t0
                print(f"进度 {done}/{len(todo)} ({el:.0f}s)", file=sys.stderr)

    print(f"完成: {done - len(failed)}/{done}, 未补 {len(failed)}, 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)
    if failed:
        with open(os.path.join(BASE, "failed_annual_em.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["code", "year", "reason"])
            w.writerows(failed)


if __name__ == "__main__":
    main()
