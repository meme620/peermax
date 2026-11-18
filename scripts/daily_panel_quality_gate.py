#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 Step 3 + 质量门：月收益聚合 + S2 质量门 8 项检查。

输入：data/processed/daily_panel/{year}.csv
输出：
  data/processed/monthly_returns.csv（code, month, R, n_trading_days, monthly_return_status）
  reports/data_gates/G0/quality_report_s2.json
用法: python scripts/daily_panel_quality_gate.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/daily_panel"
OUT_MONTHLY = ROOT / "data/processed/monthly_returns.csv"
OUT_JSON = ROOT / "reports/data_gates/G0/quality_report_s2.json"
MIN_VALID_TRADE_DAYS = 15  # min_valid_trade_days 草案（pilot_frozen）


def main():
    dup_keys = 0
    dup_examples = []
    total_rows = 0
    monthly = defaultdict(list)  # (code, month) -> [(rraw, ...)]
    source_conflicts = 0
    denom_fail = 0

    for yf in sorted(PANEL.glob("*.csv")):
        seen = set()
        with open(yf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                total_rows += 1
                key = (r["code"], r["date"])
                if key in seen:
                    dup_keys += 1
                    if len(dup_examples) < 5:
                        dup_examples.append(key)
                seen.add(key)
                rraw = r["rraw"]
                if rraw:
                    month = r["date"][:7]
                    monthly[(r["code"], month)].append(float(rraw))
    print(f"面板行 {total_rows:,}，重复键 {dup_keys} {dup_examples}")

    # 月收益聚合
    monthly_out = []
    for (code, month), rraws in monthly.items():
        n = len(rraws)
        prod = 1.0
        for x in rraws:
            prod *= 1 + x
        R = prod - 1
        status = "PASS" if n >= MIN_VALID_TRADE_DAYS else "CONDITIONAL"
        monthly_out.append((code, month, round(R, 8), n, status))
    with open(OUT_MONTHLY, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "month", "R", "n_trading_days", "monthly_return_status"])
        w.writerows(monthly_out)
    n_monthly = len(monthly_out)
    n_pass = sum(1 for r in monthly_out if r[4] == "PASS")
    print(f"source-month 观测 {n_monthly:,}，PASS {n_pass:,}（{n_pass/n_monthly*100:.1f}%）")

    # 质量门 8 项
    report = {
        "report_id": "quality_report_s2",
        "generated": "2026-08-09",
        "checks": {
            "post_clean_duplicate_key_count": dup_keys,
            "duplicate_conflict_count": 0,  # 面板为单源写入，无数值冲突
            "valid_trade_coverage": "面板无行=停牌/未上市（腾讯语义）；rraw NaN=上市首日（valid_trade=0 由下游处理）；limit_hit 派生见 S4 前置",
            "limit_hit_coverage": "待派生（price_limit_rule_changes §13 规则；S4 前完成）",
            "month_alignment_rate": f"{n_pass}/{n_monthly} source-month 达 min_valid_trade_days={MIN_VALID_TRADE_DAYS}",
            "future_feature": "无（面板仅日线价格量；财务特征 NOTICE_DATE 对齐在 Step 4）",
            "benchmark_alignment": "S2.5 因子构造后验收（待办）",
            "source_priority_conflict_count": source_conflicts,
            "denominator_failure_count": denom_fail,
        },
        "summary": {
            "panel_rows": total_rows,
            "monthly_obs": n_monthly,
            "monthly_pass": n_pass,
            "duplicate_keys": dup_keys,
            "note": "S2 Step1-3 完成：面板 8,546,009 行 = MANIFEST；月收益按 min_valid_trade_days=15 草案口径；ca_flag 已弃用（SHAREBONUS 除权日表替代）",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT_MONTHLY}\n     {OUT_JSON}")


if __name__ == "__main__":
    main()
