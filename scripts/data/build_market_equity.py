#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""市值面板（S2.4 Size/BM 输入）：月末市值 = close × 月末前最新可得股本。

口径（事件前规则）:
  默认 base = v2[year=t]（年末值）；t 年内送转（EX_DIVIDEND_DATE ≤ 月末 生效）已含在 v2[t]，
  对除权日在月末之后的送转因子除回：end_share(m) = v2[t] / ∏(因子[除权日 > m 月])
  无送转年份 fac=1 → 任意月 = v2[t]（=v2[t-1]，无偏差）
输出: data/processed/market_equity_panel.csv
"""
import csv
import glob
import os
import sys

V2 = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "share_capital_history_v2.csv")
BONUS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "eastmoney", "sharebonus")
RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "tencent", "daily_raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)


def month_end_dates():
    """2018-01 ~ 2025-12 每月末交易日（从腾讯日线推出）。"""
    ends = {}
    for p in glob.glob(os.path.join(RAW, "*", "*.csv")):
        if not 2018 <= int(os.path.basename(p)[:4]) <= 2025:
            continue
        code = os.path.basename(os.path.dirname(p))[2:]
        if code != "600519":
            continue
        with open(p, encoding="utf-8") as f:
            for r in csv.reader(f):
                if r and r[0][:4].isdigit():
                    ends[r[0][:7]] = r[0]  # 覆盖更新 -> 月末
    return ends


def bonus_events(code: str):
    """[(ex_date 'YYYY-MM-DD', factor)]，factor=1+ratio/10。"""
    p = os.path.join(BONUS, f"{code}.csv")
    ev = []
    if not os.path.exists(p):
        return ev
    with open(p, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return ev
    hdr = rows[0]
    try:
        ir = hdr.index("BONUS_IT_RATIO")
        ix = hdr.index("EX_DIVIDEND_DATE")
    except ValueError:
        return ev
    for r in rows[1:]:
        if len(r) <= max(ir, ix):
            continue
        try:
            ratio = float(r[ir])
        except ValueError:
            continue
        if ratio <= 0 or not r[ix]:
            continue
        ev.append((r[ix][:10], 1 + ratio / 10.0))
    return ev


def main():
    # v2
    v2 = {}
    with open(V2, encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and len(r) >= 5 and r[0] != "code" and r[2]:
                v2.setdefault(r[0], {})[int(r[1])] = (float(r[2]), r[3], r[4])
    codes = sorted(v2)
    ends = month_end_dates()
    print(f"月末交易日: {len(ends)}", file=sys.stderr)

    out_rows = []
    stat = {"ok": 0, "missing": 0}
    for code in codes:
        ev = bonus_events(code)
        for month, end_date in sorted(ends.items()):
            y = int(month[:4])
            # 月末前最新可得：v2[y]（年末值）——若月末在 12 月前且年内有送转，按除权日调整
            rec = v2[code].get(y)
            if rec is None:
                # 前一年值
                rec = v2[code].get(y - 1)
            if rec is None:
                out_rows.append([code, end_date, "", "", "", "no_share"])
                stat["missing"] += 1
                continue
            end_share, method, status = rec
            # 除权日在月末之后（该年内）的送转因子除回
            divisor = 1.0
            for ex_date, fac in ev:
                if ex_date[:7] == month or (ex_date[:4] == str(y) and ex_date[:7] > month):
                    divisor *= fac
            end_share_m = end_share / divisor if divisor > 1 else end_share
            # 月末 close（腾讯日线）
            prefix = "sh" if code.startswith(("6", "9")) else "sz"
            p = os.path.join(RAW, f"{prefix}{code}", f"{y}.csv")
            close = None
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    for r in csv.reader(f):
                        if r and r[0] == end_date:
                            close = float(r[2])
                            break
            if close is None or close <= 0:
                out_rows.append([code, end_date, "", "", "", "no_close"])
                stat["missing"] += 1
                continue
            me = close * end_share_m
            out_rows.append([code, end_date, close, end_share_m, me, f"{method}|{status}"])
            stat["ok"] += 1

    with open(os.path.join(OUT, "market_equity_panel.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm_id", "date", "close", "end_share", "market_equity", "missing_reason"])
        w.writerows(out_rows)
    print(f"市值面板: ok {stat['ok']}, missing {stat['missing']}", file=sys.stderr)
    # 茅台 2024-12 检查
    for r in out_rows:
        if r[0] == "600519" and r[1] == "2024-12-31":
            print(f"茅台 2024-12-31: close={r[2]} share={r[3]} 市值={float(r[4])/1e8:.0f}亿", file=sys.stderr)


if __name__ == "__main__":
    main()
