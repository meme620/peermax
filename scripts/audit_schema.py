#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inventory Step 3：schema 审计（5 表逐项检查）。

输出: data/processed/schema_audit_report.csv（table, check, result, detail）
检查项:
  1. 腾讯日线: 唯一键 (code6,date) / date 格式 / close>0 / 日期⊆交易日历 / 年度行数≤交易日
  2. 东财财务: 唯一键 (code6,REPORT_DATE) / NOTICE_DATE>=REPORT_DATE / TOTAL_EQUITY_PK>0 比例
  3. 年报文本: 文件存在性 vs a_stocks×[ipo,out]∩2018-2025 / 大小>1KB
  4. 龙虎榜: 唯一键近似 / FREE_MARKET_CAP>0
  5. 送转表: 唯一键 / BONUS_IT_RATIO>0
用法: python scripts/audit_schema.py
"""
import csv
import glob
import os
import sys

BASE = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)

REPORT = []


def rec(table, check, result, detail):
    REPORT.append([table, check, result, detail])
    print(f"{table} | {check} | {result} | {detail[:80]}", file=sys.stderr)


def load_calendar():
    days = {}
    with open(os.path.join(BASE, "baostock", "calendar", "trade_dates_2016_2026.csv"), encoding="utf-8", newline='') as f:
        for r in csv.reader(f):
            if len(r) >= 2 and r[0][:4].isdigit():
                days[r[0]] = r[1]
    return days


def main():
    cal = load_calendar()
    trade = {d for d, v in cal.items() if v == "1"}

    # 1. 腾讯日线（抽样 500 文件全查唯一键）
    files = [p for p in glob.glob(os.path.join(BASE, "tencent", "daily_raw", "*", "*.csv"))
             if 2018 <= int(os.path.basename(p)[:4]) <= 2025]
    dup_total = 0
    bad_date = 0
    bad_close = 0
    not_in_cal = 0
    over_rows = 0
    n_rows = 0
    n_files = 0
    for p in files:
        code = os.path.basename(os.path.dirname(p))[2:]
        y = os.path.basename(p)[:4]
        seen = set()
        row_count = 0
        with open(p, encoding="utf-8", newline='') as f:
            for r in csv.reader(f):
                if not r or not r[0][:4].isdigit():
                    continue
                n_rows += 1
                row_count += 1
                d = r[0]
                if d in seen:
                    dup_total += 1
                seen.add(d)
                if len(d) != 10 or d[4] != "-":
                    bad_date += 1
                try:
                    if float(r[2]) <= 0:
                        bad_close += 1
                except ValueError:
                    bad_close += 1
                if d not in trade:
                    not_in_cal += 1
        n_files += 1
        days_in_year = sum(1 for d in trade if d[:4] == y)
        if row_count > days_in_year:
            over_rows += 1
    rec("tencent_daily", "唯一键(code,date)重复", "PASS" if dup_total == 0 else "FAIL", f"重复 {dup_total}/{n_rows}")
    rec("tencent_daily", "date 格式 YYYY-MM-DD", "PASS" if bad_date == 0 else "FAIL", f"异常 {bad_date}")
    rec("tencent_daily", "close>0", "PASS" if bad_close == 0 else "FAIL", f"异常 {bad_close}")
    rec("tencent_daily", "日期⊆交易日历", "PASS" if not_in_cal == 0 else "FAIL", f"不在日历 {not_in_cal}")
    rec("tencent_daily", "年度行数≤交易日", "PASS" if over_rows == 0 else "FAIL", f"超行文件 {over_rows}/{n_files}")

    # 2. 东财财务（抽样 300）
    fin_files = glob.glob(os.path.join(BASE, "eastmoney", "finance", "*.csv"))
    import random
    random.seed(1)
    sample = random.sample(fin_files, min(300, len(fin_files)))
    fin_dup = 0
    fwd = 0
    eq_neg = 0
    fin_rows = 0
    for p in sample:
        code = os.path.basename(p)[:6]
        seen = set()
        with open(p, encoding="utf-8", newline='') as f:
            rows = list(csv.reader(f))
        hdr = rows[0]
        try:
            ir = hdr.index("REPORT_DATE")
            in_ = hdr.index("NOTICE_DATE")
            ie = hdr.index("TOTAL_EQUITY_PK")
        except ValueError:
            continue
        for r in rows[1:]:
            if len(r) <= max(ir, in_, ie):
                continue
            fin_rows += 1
            key = (code, r[ir][:10])
            if key in seen:
                fin_dup += 1
            seen.add(key)
            if r[in_] and r[ir] and r[in_][:10] < r[ir][:10]:
                fwd += 1
            try:
                if r[ie] and float(r[ie]) <= 0:
                    eq_neg += 1
            except ValueError:
                pass
    rec("em_finance", "唯一键(code,REPORT_DATE)", "PASS" if fin_dup == 0 else "FAIL", f"重复 {fin_dup}/{fin_rows}")
    rec("em_finance", "NOTICE_DATE>=REPORT_DATE", "PASS" if fwd == 0 else "FAIL", f"前视 {fwd}")
    rec("em_finance", "TOTAL_EQUITY_PK>0 比例", "PASS" if eq_neg == 0 else "CONDITIONAL", f"非正 {eq_neg}/{fin_rows}")

    # 3. 年报文本
    a = {}
    with open(os.path.join(BASE, "baostock", "stock_basic", "a_stocks.csv"), encoding="utf-8", newline='') as f:
        for r in csv.reader(f):
            if r and len(r) >= 4 and "." in r[0]:
                a[r[0].split(".")[1]] = r
    expected = 0
    have = 0
    small = 0
    for code, r in a.items():
        ipo = int(r[2][:4]) if r[2] else 2018
        out = int(r[3][:4]) if r[3] else 2025
        for y in range(max(2018, ipo), min(2025, out) + 1):
            expected += 1
            p = os.path.join(BASE, "cninfo", "annual_txt", code, f"{y}.txt")
            if os.path.exists(p):
                have += 1
                if os.path.getsize(p) < 1000:
                    small += 1
    rec("annual_txt", "文件存在性", "PASS" if have / max(expected, 1) >= 0.97 else "FAIL", f"{have}/{expected} ({have/max(expected,1):.1%})")
    rec("annual_txt", "大小>1KB", "PASS" if small == 0 else "CONDITIONAL", f"小文件 {small}")

    # 4. 龙虎榜（抽样 100 文件）
    dt_files = glob.glob(os.path.join(BASE, "eastmoney", "dragon_tiger", "*.csv"))
    dt_files = [p for p in dt_files if os.path.basename(p) != "failed_days.csv"]
    dt_dup = 0
    dt_mcap = 0
    dt_rows = 0
    for p in random.sample(dt_files, min(100, len(dt_files))):
        with open(p, encoding="utf-8", newline='') as f:
            rows = list(csv.reader(f))
        hdr = rows[0]
        try:
            ic = hdr.index("SECURITY_CODE")
            im = hdr.index("FREE_MARKET_CAP")
            it = hdr.index("TRADE_DATE")
            ie = hdr.index("EXPLAIN")
        except ValueError:
            continue
        seen = set()
        for r in rows[1:]:
            if len(r) <= max(ic, im, it, ie):
                continue
            dt_rows += 1
            key = (r[it][:10], r[ic], r[ie])
            if key in seen:
                dt_dup += 1
            seen.add(key)
            try:
                if float(r[im]) > 0:
                    dt_mcap += 1
            except ValueError:
                pass
    rec("dragon_tiger", "唯一键近似(TRADE_DATE,code,EXPLAIN)", "PASS" if dt_dup == 0 else "CONDITIONAL", f"重复 {dt_dup}/{dt_rows}")
    rec("dragon_tiger", "FREE_MARKET_CAP>0", "PASS" if dt_mcap / max(dt_rows, 1) > 0.99 else "FAIL", f"{dt_mcap}/{dt_rows}")

    # 5. 送转表（抽样 200）
    sb_files = glob.glob(os.path.join(BASE, "eastmoney", "sharebonus", "*.csv"))
    sb_dup = 0
    sb_ratio = 0
    sb_rows = 0
    for p in random.sample(sb_files, min(200, len(sb_files))):
        with open(p, encoding="utf-8", newline='') as f:
            rows = list(csv.reader(f))
        hdr = rows[0]
        try:
            ic = hdr.index("SECURITY_CODE")
            ir = hdr.index("REPORT_DATE")
            ip = hdr.index("PLAN_NOTICE_DATE")
            ib = hdr.index("BONUS_IT_RATIO")
        except ValueError:
            continue
        seen = set()
        for r in rows[1:]:
            if len(r) <= max(ic, ir, ip, ib):
                continue
            sb_rows += 1
            key = (r[ic], r[ir][:10], r[ip][:10])
            if key in seen:
                sb_dup += 1
            seen.add(key)
            try:
                if float(r[ib]) > 0:
                    sb_ratio += 1
            except ValueError:
                pass
    rec("sharebonus", "唯一键(code,REPORT_DATE,PLAN_NOTICE_DATE)", "PASS" if sb_dup == 0 else "CONDITIONAL", f"重复 {sb_dup}/{sb_rows}")
    rec("sharebonus", "存在 BONUS_IT_RATIO>0 记录", "PASS" if sb_ratio > 0 else "FAIL", f"正 ratio {sb_ratio}/{sb_rows}（0=纯分红，正常）")

    with open(os.path.join(OUT, "schema_audit_report.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["table", "check", "result", "detail"])
        w.writerows(REPORT)
    fails = sum(1 for r in REPORT if r[2] == "FAIL")
    print(f"审计完成: {len(REPORT)} 项, FAIL {fails}", file=sys.stderr)


if __name__ == "__main__":
    main()
