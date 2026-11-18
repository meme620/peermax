#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股本 v2（修正版）：全覆盖 + 双向送转展开 + 三级锚点（A > C 交叉 > 东财当前值）。

策略:
  锚点: 最新 A（可信）> 最新 C（交叉后可信）> 东财 TOTAL_SHARE（2025 末，needs_review）
  双向展开: 向后 share(y)=share(y-1)×(1+Σr_y)；向前 share(y-1)=share(y)/(1+Σr_y)
    （r_y = 除权日在 y 年的送转比例和/10；无送转事件年份不变）
  增发校验: 展开值与 v1 A 值差 >5% → 用 A 值 + needs_review
输出: data/processed/share_capital_history_v2.csv
"""
import csv
import glob
import os
import sys

V1 = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "institutional", "share_capital_history.csv")
BONUS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "eastmoney", "sharebonus")
FIN = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "eastmoney", "finance")
A_STOCKS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "baostock", "stock_basic", "a_stocks.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
os.makedirs(OUT, exist_ok=True)

YEARS = list(range(2018, 2026))


def load_bonus(code: str):
    p = os.path.join(BONUS, f"{code}.csv")
    evts = []
    if not os.path.exists(p):
        return evts
    with open(p, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return evts
    hdr = rows[0]
    try:
        ir = hdr.index("BONUS_IT_RATIO")
        ix = hdr.index("EX_DIVIDEND_DATE")
        ip = hdr.index("PLAN_NOTICE_DATE")
    except ValueError:
        return evts
    for r in rows[1:]:
        if len(r) <= max(ir, ix, ip):
            continue
        try:
            ratio = float(r[ir])
        except ValueError:
            continue
        if ratio <= 0:
            continue
        d = (r[ix] or r[ip])[:4]
        if d.isdigit() and 2015 <= int(d) <= 2025:
            evts.append((int(d), ratio))
    return evts


def bonus_factor_by_year(evts):
    """year -> 送转乘数（1 + Σratio/10）。"""
    f = {}
    for y, r in evts:
        f[y] = f.get(y, 1.0) * (1 + r / 10.0)
    return f


def main():
    # v1 提取值
    v1 = {}
    with open(V1, encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and len(r) >= 5 and r[0] != "code" and r[2]:
                v1.setdefault(r[0], {})[int(r[1])] = (float(r[2]), r[3])
    # 全部股票
    codes = []
    with open(A_STOCKS, encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and "." in r[0]:
                codes.append(r[0].split(".")[1])
    codes = sorted(set(codes))

    rows = []
    stat = {"A": 0, "C": 0, "bonus_back": 0, "em_anchor": 0, "no_anchor": 0}
    n_review = 0
    for code in codes:
        yrs = v1.get(code, {})
        a_pts = {y: v for y, (v, m) in yrs.items() if m == "A"}
        c_pts = {y: v for y, (v, m) in yrs.items() if m == "C"}
        b_pts = {y: v for y, (v, m) in yrs.items() if m == "B_carry"}
        evts = load_bonus(code)
        fac = bonus_factor_by_year(evts)

        # 三级锚点
        anchor, anchor_year, anchor_src, anchor_status = None, None, None, "ok"
        if a_pts:
            anchor_year = max(a_pts)
            anchor = a_pts[anchor_year]
            anchor_src = "A"
        elif c_pts:
            anchor_year = max(c_pts)
            anchor = c_pts[anchor_year]
            anchor_src = "C"
            anchor_status = "c_anchor"
        elif b_pts:
            # B_carry 链值（继承 A）可信度次于直接 A
            anchor_year = max(b_pts)
            anchor = b_pts[anchor_year]
            anchor_src = "B_carry"
            anchor_status = "ok"
        else:
            # 东财 TOTAL_SHARE（当前值回填）作 2025 锚（needs_review）
            fp = os.path.join(FIN, f"{code}.csv")
            if os.path.exists(fp):
                with open(fp, encoding="utf-8") as f:
                    fr = list(csv.reader(f))
                if len(fr) > 1:
                    hdr = fr[0]
                    try:
                        it = hdr.index("TOTAL_SHARE")
                        if fr[1][it]:
                            anchor = float(fr[1][it])
                            anchor_year = 2025
                            anchor_src = "EM"
                            anchor_status = "em_anchor"
                    except (ValueError, IndexError):
                        pass
        if anchor is None:
            for y in YEARS:
                rows.append([code, y, "", "no_anchor", "no_anchor", ""])
                stat["no_anchor"] += 1
            continue

        # 双向展开
        val = {anchor_year: anchor}
        for y in range(anchor_year + 1, YEARS[-1] + 1):
            val[y] = val[y - 1] * fac.get(y, 1.0)
        for y in range(anchor_year - 1, YEARS[0] - 1, -1):
            val[y] = val[y + 1] / fac.get(y + 1, 1.0)

        for y in YEARS:
            back = val.get(y)
            method = "bonus_back"
            value = back
            status = anchor_status if anchor_status != "ok" else "ok"
            if y in a_pts:
                value = a_pts[y]
                method = "A"
                if back and back > 0 and abs(back - value) / value > 0.05:
                    status = "needs_review"
            elif y in c_pts:
                cval = c_pts[y]
                if back and back > 0 and abs(cval - back) / back <= 0.05:
                    value = cval
                    method = "C"
                else:
                    value = back if back else cval
                    method = "bonus_back" if back else "C"
                    if back and cval > 0 and abs(cval - back) / back > 0.05:
                        status = "needs_review"
            if anchor_src == "EM" and status not in ("needs_review",):
                status = "em_anchor"
            if anchor_src == "C" and y not in a_pts and status not in ("needs_review",):
                status = "c_anchor"
            if status == "needs_review":
                n_review += 1
            if method in stat:
                stat[method] += 1
            rows.append([code, y, value, method, status, f"anchor_{anchor_src}@{anchor_year}"])

    with open(os.path.join(OUT, "share_capital_history_v2.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "end_share", "method", "status", "source"])
        w.writerows(rows)

    total = len(rows)
    covered = sum(1 for r in rows if r[2])
    print(f"v2: {total} 股票-年份（{len(codes)} 只 × {len(YEARS)} 年）, 覆盖 {covered} ({covered/max(total,1):.0%})", file=sys.stderr)
    print(f"方法: {stat}", file=sys.stderr)
    print(f"needs_review 行: {n_review}", file=sys.stderr)


if __name__ == "__main__":
    main()
