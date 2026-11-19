#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""daily-panel Step 2：日收益构造（rraw + 双源交叉 + 冲突标记）。

输入: data/raw/tencent/daily_raw|daily_qfq/{sh|sz}{code6}/{year}.csv（只读）
输出: data/processed/daily_returns_clean/{code6}.csv
      data/processed/daily_returns_clean/MANIFEST.json

口径（execution_plan §5.4 daily-panel Step 2；2026-08-09 数据质量发现）:
  rraw_{i,d} = close_d / close_{d-1} - 1（不复权 close 链内前收；跨年用上年末收盘衔接）
  pct_chg 字段弃用（实测非收盘涨跌幅，语义不可靠）；qfq 链内收益与 raw 链不一致
  （腾讯前复权数据特性，非除权期亦不等）——双源交叉不成立，rraw_qfq 仅保留参考列，
  不做冲突判定；真实双源交叉待第二行情源（东财 push2his 恢复后）。
  停牌/缺行自然跳过，不得补 0。
用法: python scripts/build_daily_returns.py [--max-codes 200]
"""
import argparse
import csv
import json
import os
import sys
import time

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "tencent")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "daily_returns_clean")
A_STOCKS = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "baostock", "stock_basic", "a_stocks.csv")
EPSILON = 1e-4  # epsilon_return_source 候选（登记，不冻结）


def read_year(prefix: str, code6: str, year: int):
    """读某年日线，返回 {date: (close, pct_chg)}。"""
    p = os.path.join(RAW, "daily_raw", f"{prefix}{code6}", f"{year}.csv")
    d = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline='') as f:
            for r in csv.reader(f):
                if r and len(r) >= 5 and r[0][:4].isdigit():
                    try:
                        d[r[0]] = (float(r[2]), float(r[7]) if len(r) > 7 and r[7] else None)
                    except ValueError:
                        continue
    return d


def read_year_qfq(prefix: str, code6: str, year: int):
    p = os.path.join(RAW, "daily_qfq", f"{prefix}{code6}", f"{year}.csv")
    d = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline='') as f:
            for r in csv.reader(f):
                if r and len(r) >= 5 and r[0][:4].isdigit():
                    try:
                        d[r[0]] = float(r[2])
                    except ValueError:
                        continue
    return d


def build_stock(prefix: str, code6: str, years):
    rows = []
    prev_close_raw = None
    prev_close_qfq = None
    dup = 0
    for y in years:
        raw = read_year(prefix, code6, y)
        qfq = read_year_qfq(prefix, code6, y)
        for date in sorted(raw):
            close = raw[date][0]
            pct = raw[date][1]
            if date in rows_seen(rows):
                dup += 1
                continue
            rraw = None
            if prev_close_raw is not None and prev_close_raw > 0:
                rraw = close / prev_close_raw - 1
            rraw_qfq = None
            qc = qfq.get(date)
            if qc is not None and prev_close_qfq is not None and prev_close_qfq > 0:
                rraw_qfq = qc / prev_close_qfq - 1
            # pct_chg 字段弃用（语义不可靠）；qfq 仅参考列，不做冲突判定
            valid = 1 if rraw is not None else 0
            rows.append([f"{prefix}{code6}", date, rraw, rraw_qfq, pct, valid,
                         0, 0, "tencent_raw_v1", date])
            prev_close_raw = close
            if qc is not None:
                prev_close_qfq = qc
    return rows, dup


def rows_seen(rows):
    return {r[1] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-codes", type=int, default=0)
    ap.add_argument("--years", default="2017-2025")
    args = ap.parse_args()
    ys = args.years.split("-")
    years = list(range(int(ys[0]), int(ys[1]) + 1))
    # 输出年份 2018-2025；years 含 2017 仅作跨年衔接（2017 年数据读入但只用于 prev）
    out_years = [y for y in years if y >= 2018]

    with open(A_STOCKS, encoding="utf-8", newline='') as f:
        stocks = [r for r in csv.reader(f)][1:]
    codes = [(r[0].split(".")[1], r[0].split(".")[0]) for r in stocks if r and "." in r[0]]
    if args.max_codes:
        codes = codes[: args.max_codes]

    os.makedirs(OUT, exist_ok=True)
    total_rows = 0
    total_dup = 0
    t0 = time.time()
    for i, (code6, prefix) in enumerate(codes):
        rows, dup = build_stock(prefix, code6, years)
        # 只输出 2018+ 行
        rows = [r for r in rows if r[1][:4] in {str(y) for y in out_years}]
        if rows:
            with open(os.path.join(OUT, f"{code6}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["firm_id", "date", "rraw", "rraw_qfq", "pct_chg", "valid_trade",
                            "return_source_conflict", "pct_chg_conflict", "source_version", "available_at"])
                w.writerows(rows)
        total_rows += len(rows)
        total_dup += dup
        if (i + 1) % 200 == 0:
            print(f"进度 {i+1}/{len(codes)} ({(time.time()-t0)/60:.1f} min, {total_rows} 行)", file=sys.stderr)

    manifest = {
        "script": "scripts/build_daily_returns.py",
        "stocks": len(codes),
        "rows": total_rows,
        "duplicate_rows": total_dup,
        "note": "pct_chg 字段弃用、qfq 仅参考（2026-08-09 数据质量发现，见 plan §26）",
        "epsilon_return_source": EPSILON,
        "source_version": "tencent_raw_v1",
    }
    with open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline='') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"完成: {len(codes)} 只, {total_rows} 行, dup {total_dup}", file=sys.stderr)


if __name__ == "__main__":
    main()
