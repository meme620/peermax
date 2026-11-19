#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factor mv 交叉校验（mv_crosscheck_tolerance 补救链）。

输入：
  data/processed/daily_panel/{year}.csv        （面板，close_raw/rraw）
  data/processed/sharebonus_events.csv         （RPT_SHAREBONUS_DET 全量）
  data/raw/eastmoney/finance/{code}.csv        （TOTAL_SHARE/A_FREE_SHARE/NOTICE_DATE/REPORT_DATE）
输出：
  data/processed/corporate_action_events.csv   （SHAREBONUS 驱动的除权日表，替代面板 ca_flag）
  reports/data_gates/G0/mv_crosscheck_report.json
校验逻辑：
  1. 除权日表 = SHAREBONUS EX_DIVIDEND_DATE ∈ [2018-01-01, 2025-12-31]，按 (SECURITY_CODE, EX_DIVIDEND_DATE)
  2. 股本重构：以 SHAREBONUS.TOTAL_SHARES 为事件点（事件日及其后生效），构造逐日总股本序列
  3. 东财 ffill：TOTAL_SHARE 按 NOTICE_DATE 前向填充（禁 REPORT_DATE）
  4. 交叉：两序列逐日相对差 > mv_crosscheck_tolerance(0.02) 的 firm-day 计数；超限日应集中在事件边界
  5. 辅助：qfq/raw 比值跳变清单（面板 ca_flag，已知过度标记）仅作事件边界吻合度参考
用法: python scripts/daily_panel_market_value_check.py [--limit N]
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/daily_panel"
SHAREBONUS = ROOT / "data/processed/sharebonus_events.csv"
FINANCE = ROOT / "data/raw/eastmoney/finance"
OUT_EVENTS = ROOT / "data/processed/corporate_action_events.csv"
OUT_JSON = ROOT / "reports/data_gates/G0/mv_crosscheck_report.json"
TOL = 0.02  # mv_crosscheck_tolerance 草案


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只校验前 N 只股票")
    args = ap.parse_args()

    # ---------- 1. 除权日表（SHAREBONUS 驱动） ----------
    events = defaultdict(list)  # code -> [(ex_date, total_shares, bonus_ratio)]
    n_events = 0
    with open(SHAREBONUS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r["EX_DIVIDEND_DATE"] or "")[:10]
            if not ("2018-01-01" <= d <= "2025-12-31"):
                continue
            code = r["SECURITY_CODE"]
            ts = r["TOTAL_SHARES"]
            events[code].append((d, ts, (r["BONUS_RATIO"] or ""), (r["IT_RATIO"] or "")))
            n_events += 1
    with open(OUT_EVENTS, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "ex_date", "total_shares", "bonus_ratio", "it_ratio"])
        for code, evs in events.items():
            for ev in evs:
                w.writerow([code] + list(ev))
    print(f"除权日表: {n_events} 事件 / {len(events)} 只股票 -> {OUT_EVENTS}")

    # ---------- 2. 抽检股票做 mv 交叉（分层抽样 200 只） ----------
    import random
    rng = random.Random(20260809)
    codes = sorted(events.keys())
    if args.limit:
        sample = codes[: args.limit]
    else:
        sample = rng.sample(codes, min(200, len(codes)))

    over_tol = 0
    checked = 0
    over_by_year = Counter()
    event_boundary_hits = 0
    no_finance = 0

    for code in sample:
        fin = FINANCE / f"{code}.csv"
        if not fin.exists():
            no_finance += 1
            continue
        # 东财 TOTAL_SHARE 序列（NOTICE_DATE 对齐）
        fin_rows = []
        with open(fin, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                nd = (r.get("NOTICE_DATE") or "")[:10]
                ts = r.get("TOTAL_SHARE") or ""
                if nd and ts:
                    fin_rows.append((nd, float(ts)))
        if not fin_rows:
            continue
        fin_rows.sort()
        # 事件重构序列：事件日生效的 TOTAL_SHARES
        evs = sorted(events.get(code, []))
        ev_days = {e[0] for e in evs}
        # 面板 close 序列（该股票全部年份）
        panel_days = []
        for yf in sorted(PANEL.glob("*.csv")):
            with open(yf, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["code"] == code:
                        panel_days.append(r["date"])
        panel_days.sort()
        if not panel_days:
            continue
        # ffill 股本（东财）
        fin_idx = 0
        fin_dates = [r[0] for r in fin_rows]
        fin_shares = [r[1] for r in fin_rows]
        for d in panel_days:
            while fin_idx + 1 < len(fin_dates) and fin_dates[fin_idx + 1] <= d:
                fin_idx += 1
            ffill_ts = fin_shares[fin_idx] if fin_dates[fin_idx] <= d else None
            if ffill_ts is None:
                continue
            # 事件重构：最后生效事件
            ref = None
            for ev in evs:
                if ev[0] <= d:
                    ref = ev[1]
                else:
                    break
            if ref:
                try:
                    ref_ts = float(ref)
                except ValueError:
                    continue
                checked += 1
                if ref_ts > 0 and abs(ffill_ts - ref_ts) / ref_ts > TOL:
                    over_tol += 1
                    over_by_year[d[:4]] += 1
                    # 超限日是否在事件边界（±5 交易日）？
                    near = any(abs((ev[0] > d) - (ev[0] < d)) == 0 and (ev[0] == d) for ev in evs)
                    if ev_days and d in ev_days:
                        event_boundary_hits += 1

    report = {
        "report_id": "mv_crosscheck",
        "tolerance": TOL,
        "sample_firms": len(sample),
        "no_finance_firms": no_finance,
        "checked_firm_days": checked,
        "over_tolerance_firm_days": over_tol,
        "over_rate": round(over_tol / checked, 6) if checked else None,
        "over_by_year": dict(sorted(over_by_year.items())),
        "note": "东财 TOTAL_SHARE 按 NOTICE_DATE ffill vs SHAREBONUS TOTAL_SHARES 事件重构；超限应集中于事件边界（股本变动窗口）；qfq/raw 比值跳变已确认不可作识别源（浦发 2018 标记 27 天 vs 真实 1 天）",
        "verdict_draft": "PASS（over_rate <= 容差观察阈值）" if (over_tol / checked if checked else 1) <= 0.05 else "CHECK",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mv 交叉: 抽样 {len(sample)} 只，无财务表 {no_finance}，firm-day {checked:,}，超限 {over_tol:,} "
          f"({report['over_rate']}) 按年 {dict(sorted(over_by_year.items()))}")
    print(f"输出: {OUT_JSON}")


if __name__ == "__main__":
    main()
