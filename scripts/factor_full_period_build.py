# -*- coding: utf-8 -*-
"""S2.5 因子全期化（2026-08-29）：
1. 用 Drive daily_close_none（2000 起）构造 daily_panel 2006-2013（close_raw + rraw 日收益），补全 2014 前的日面板
2. 重跑 factor_build（FF4 全期 2006-2025）+ build_rev_factor（REV 全期）
用法: python scripts/factor_full_period_build.py
"""
import io, sys
from fileio import read_pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/daily_panel"
CLOSE = ROOT / "data/raw/drive/daily_close_none_20260630.pkl"
DRIVE_TO = ROOT / "data/raw/drive/daily_turnover_rate_20260630.pkl"

# ---- 1) 构造 2006-2013 日面板（close_raw + rraw；volume/amount 用 Drive 换手率*流通盘不可得，填空）----
close = read_pickle(CLOSE)  # index=date, columns=order_book_id '000001.SZ'
rraw = close.pct_change(fill_method=None)
start, end = "2006-01-01", "2013-12-31"
mask = (close.index >= start) & (close.index <= end)
n_missing = 0
for year in range(2006, 2014):
    ym = (close.index >= f"{year}-01-01") & (close.index <= f"{year}-12-31")
    c_sub = close[ym]
    r_sub = rraw[ym]
    st = c_sub.stack().rename('close_raw').reset_index()
    st.columns = ['date', 'firm', 'close_raw']
    rr = r_sub.stack().rename('rraw').reset_index()
    rr.columns = ['date', 'firm', 'rraw']
    d = st.merge(rr, on=['date', 'firm'])
    d['code'] = d['firm'].astype(str).str.split('.').str.get(0)
    d = d[d['close_raw'].notna()]
    out = pd.DataFrame({
        'code': d['code'], 'date': d['date'].astype(str),
        'close_raw': d['close_raw'], 'close_qfq': '',
        'volume': '', 'amount': '', 'rraw': d['rraw'], 'ca_flag': '',
    })
    # 与已有 daily_panel 年度文件对比（2014+ 已存在，不覆盖）
    f = PANEL / f"{year}.csv"
    if f.exists():
        print(f"  {year}.csv 已存在，跳过（{len(out)} 派生行）", flush=True)
        continue
    out.to_csv(f, index=False, encoding='utf-8')
    n_missing += 1
    print(f"  写入 {year}.csv: {len(out)} 行", flush=True)
print(f"daily_panel 补齐 {n_missing} 个年度文件", flush=True)

# ---- 2) 重跑 FF4 + REV ----
import subprocess
import os
env = dict(os.environ)
r1 = subprocess.run([sys.executable, "scripts/factor_build.py"], capture_output=True, text=True)
print("factor_build:", r1.returncode, r1.stdout[-300:] if r1.stdout else "", r1.stderr[-300:] if r1.stderr else "", flush=True)
r2 = subprocess.run([sys.executable, "scripts/build_rev_factor.py"], capture_output=True, text=True)
print("build_rev_factor:", r2.returncode, r2.stdout[-300:] if r2.stdout else "", flush=True)
print("DONE 因子全期化")
