#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2.4 控制变量（Size/BM/Mom/Rev/Liquidity/Turnover；AnalystCov 由 S3A 填充）。

依赖顺序：本脚本输出的 analyst_cov 为空列；重跑本脚本后必须再执行
scripts/data_audit/s3a_step11_merge_chars.py 从
data/processed/analyst/firm_analyst_derived.csv 填充 analyst_cov（S3A v3 全期口径）。

口径: execution_plan §5.4 Step 4；全部 available_at ≤ characteristic_cutoff_t（无前视）
输入: market_equity_panel / monthly_returns_clean / Drive turnover(官方,全期 2000 起) / 东财财务 TOTAL_EQUITY_PK(NOTICE_DATE)
2026-08-29 全期化：S1 面板已替换全期版（2006-2025）；turnover 改用 Drive 官方（batch2 决定替换）；
       liquidity 保留腾讯日收益口径（2018 起，早期标 missing 如实）。
输出: data/processed/firm_chars_clean.csv
"""
import csv
import glob
import pickle
import os
import sys
from collections import defaultdict

PROC = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
RAW_TENCENT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "tencent", "daily_raw")
FIN = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "eastmoney", "finance")
OUT = PROC
MIN_MOM = 8    # min_valid_months_momentum 候选
MIN_LIQ = 15   # min_valid_days_liquidity 候选
MIN_TO = 15    # min_valid_days_turnover 候选


def main():
    # 1. 月末市值 (month -> code -> me)
    me = {}
    with open(os.path.join(PROC, "market_equity_panel.csv"), encoding="utf-8") as f:
        for r in csv.reader(f):
            # 全期版（2026-08-29）：[firm_id, date, market_equity] 3 列；旧版 [firm_id, date, close, end_share, market_equity]
            if r and r[0] != "firm_id":
                try:
                    v = float(r[4]) if len(r) > 4 and r[4] else (float(r[2]) if len(r) > 2 and r[2] else None)
                    if v and v > 0:
                        me.setdefault(r[1][:7], {})[r[0]] = v
                except (ValueError, IndexError):
                    continue

    # 1.5 Drive 官方换手率（全期 2000 起；batch2 决定替换自算口径）
    drive_to = {}
    _dt = pickle.load(open(os.path.join(PROC, "..", "raw", "drive", "daily_turnover_rate_20260630.pkl"), "rb"))
    _dtm = _dt.resample("ME").mean()
    for _d, _row in _dtm.iterrows():
        _m = _d.strftime("%Y-%m")
        for _c, _v in _row.items():
            if _v == _v and _v > 0:  # 非 NaN
                drive_to.setdefault(_m, {})[_c.split(".")[0]] = float(_v)

    # 2. 月收益
    mr = defaultdict(dict)
    with open(os.path.join(PROC, "monthly_returns_clean.csv"), encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and len(r) > 2 and r[0] != "firm_id" and r[2]:
                fid = r[0][2:] if r[0].startswith(("sh", "sz")) else r[0]  # 统一 6 位
                mr[fid][r[1]] = float(r[2])

    # 3. 东财净资产（code -> [(notice_date, TOTAL_EQUITY_PK)]）
    eq = {}
    for p in glob.glob(os.path.join(FIN, "*.csv")):
        code = os.path.basename(p)[:6]
        with open(p, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            continue
        hdr = rows[0]
        try:
            ir = hdr.index("REPORT_DATE")
            in_ = hdr.index("NOTICE_DATE")
            ie = hdr.index("TOTAL_EQUITY_PK")
        except ValueError:
            continue
        lst = []
        for r in rows[1:]:
            if len(r) > max(ir, in_, ie) and r[ie]:
                lst.append((r[in_][:10], float(r[ie])))
        if lst:
            lst.sort()
            eq[code] = lst

    # 4. 输出：每股每月
    out_rows = []
    months = sorted(me)
    for month in months:
        y = int(month[:4])
        for code, me_val in me[month].items():
            code6 = code[2:] if code.startswith(("sh", "sz")) else code
            # Size
            size = None
            if me_val > 0:
                size = __import__("math").log(me_val)
            # BM：NOTICE_DATE ≤ 月末 的最近净资产
            bm = None
            eq_list = eq.get(code6, [])
            best = None
            for nd, v in eq_list:
                if nd <= month + "-31" and v > 0:
                    best = v
            if best is not None and me_val > 0:
                bm = best / me_val
            # Mom / Rev（月收益）
            r_series = mr.get(code, {})
            mom = None
            rev = None
            if month in r_series:
                rev = r_series[month]
            ys = sorted(m for m in r_series if m < month)
            need = ys[-12:-1]  # t-12 ~ t-2
            if len(need) >= MIN_MOM:
                prod = 1.0
                for m in need:
                    prod *= (1 + r_series[m])
                mom = prod - 1
            # Liquidity / Turnover：配对日收益表(valid) 与腾讯原始(amount/volume)
            prefix = "sh" if code6.startswith(("6", "9")) else "sz"
            p_raw = os.path.join(RAW_TENCENT, f"{prefix}{code6}", f"{y}.csv")
            p_ret = os.path.join(PROC, "daily_returns_clean", f"{code6}.csv")
            liq = None
            to = None
            if os.path.exists(p_raw) and os.path.exists(p_ret):
                amt_map = {}
                with open(p_raw, encoding="utf-8") as f:
                    for r in csv.reader(f):
                        if r and r[0][:7] == month and len(r) > 8:
                            try:
                                amt_map[r[0]] = (float(r[8]) * 10000, float(r[5]) * 100)
                            except (ValueError, IndexError):
                                continue
                es = me.get(month, {}).get(code, 0)
                s_liq = 0.0
                n_liq = 0
                s_to = 0.0
                n_to = 0
                with open(p_ret, encoding="utf-8") as f:
                    for r in csv.reader(f):
                        if r and len(r) > 5 and r[1][:7] == month and r[5] == "1" and r[2]:
                            amt_vol = amt_map.get(r[1])
                            if amt_vol:
                                amt, vol = amt_vol
                                if amt > 0:
                                    s_liq += abs(float(r[2])) / amt
                                    n_liq += 1
                # 2026-08-29：turnover 改用 Drive 官方月度换手率（全期；自算 vol/es 仅 2018 起）
                to = drive_to.get(month, {}).get(code6)
            missing = ""
            missing = ""
            if size is None: missing += "size;"
            if bm is None: missing += "bm;"
            if mom is None: missing += "mom;"
            if rev is None: missing += "rev;"
            if liq is None: missing += "liq;"
            if to is None: missing += "to;"
            out_rows.append([code, month, size, bm, mom, rev, liq, to, "", missing])

    with open(os.path.join(OUT, "firm_chars_clean.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm_id", "month", "size", "bm", "mom", "rev", "liquidity", "turnover", "analyst_cov", "missing_reason"])
        w.writerows(out_rows)
    print(f"控制变量: {len(out_rows)} 行", file=sys.stderr)
    # 茅台抽样
    for r in out_rows:
        if r[0] == "600519" and r[1] in ("2023-12", "2024-12"):
            print(f"茅台 {r[1]}: size={r[2] and round(r[2],3)} bm={r[3] and round(r[3],3)} mom={r[4] and round(r[4],4)} rev={r[5] and round(r[5],4)}", file=sys.stderr)


if __name__ == "__main__":
    main()
