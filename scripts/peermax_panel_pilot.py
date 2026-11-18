#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SUPERSEDED（2026-09-15 审计）：pilot S4 面板构建（读旧 network/、缺失源收益填 0）。

**请勿使用**——正式实现与产物：
  - M-B 正式构网 → scripts/network_mda_edges_build.py（--tau 冻结值）→ data/processed/network_mb/
  - M-A 正式构网 → scripts/network_business_edges_build.py → data/processed/network_ma/
  - S4 面板（正式） → scripts/peermax_panel_build.py → data/processed/monthly_pricing_panel_{ma,mb}.csv
本文件仅作历史留档（pilot 口径），不得作为任何分析输入的来源。

S4 Step 2：PeerMAXraw/OwnMAXraw/PeerReturn（execution_plan §5.6 Step 2）。

输入：
  data/processed/source_month_argmax_panel.csv（MAXraw/R/argmax）
  data/processed/network/{year}_similarity.csv（网络版本=年报年；研究期月 t 用 y-1 版本）
构造：
  PeerMAXraw_{i,t} = Σ_{j≠i} w_{ij,y-1} × MAXraw_{j,t}
  OwnMAXraw_{i,t} = MAXraw_{i,t}
  PeerReturn_{i,t} = Σ_{j≠i} w_{ij,y-1} × R_{j,t}
  support/covered/missing mass 报告
输出：data/processed/monthly_pricing_panel_partial.csv（PeerMAXraw/OwnMAXraw/PeerReturn 部分）
      reports/data_gates/G0/s4_step2_report.json
用法: python scripts/peermax_panel_pilot.py [--max-year 2020]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARGMAX = ROOT / "data/processed/source_month_argmax_panel.csv"
NET = ROOT / "data/processed/network"
OUT = ROOT / "data/processed/monthly_pricing_panel_partial.csv"
OUT_JSON = ROOT / "reports/data_gates/G0/s4_step2_report.json"
SAMPLE_START = 2018
SAMPLE_END = 2025


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-year", type=int, default=0, help="调试：只处理到某年")
    args = ap.parse_args()

    # 1. source-month MAXraw/R
    maxraw = defaultdict(dict)  # (month) -> {source: MAXraw}
    Rmap = defaultdict(dict)
    with open(ARGMAX, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r["month"]
            if not (SAMPLE_START <= int(m[:4]) <= SAMPLE_END):
                continue
            maxraw[m][r["source_id"]] = float(r["MAXraw"])
            Rmap[m][r["source_id"]] = float(r["R"])

    months = sorted(maxraw)
    if args.max_year:
        months = [m for m in months if int(m[:4]) <= args.max_year]
    print(f"研究期月份: {len(months)} ({months[0]}~{months[-1]})")

    # 2. 网络按版本加载（惰性）
    net_cache = {}

    def net_for(year):
        """研究期年份 Y 用年报年 Y-1 的网络版本。"""
        if year not in net_cache:
            ver = str(year - 1)
            p = NET / f"{ver}_similarity.csv"
            edges = defaultdict(list)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        edges[r["focal_id"]].append((r["source_id"], float(r["weight"])))
            net_cache[year] = edges
        return net_cache[year]

    # 3. 逐月构造
    out_rows = []
    stats = {"covered_mass": 0.0, "support_mass": 0.0, "no_support_focals": 0}
    t0_month = None
    for mi, month in enumerate(months):
        yr = int(month[:4])
        edges = net_for(yr)
        m_max = maxraw[month]
        m_R = Rmap[month]
        for focal, lst in edges.items():
            pm = 0.0
            pr = 0.0
            covered = 0.0
            support = 0.0
            for src, w in lst:
                support += w
                if src in m_max:
                    pm += w * m_max[src]
                    pr += w * m_R.get(src, 0.0)
                    covered += w
            if support <= 0:
                stats["no_support_focals"] += 1
                continue
            own = m_max.get(focal)
            out_rows.append({
                "focal_id": focal, "month": month,
                "PeerMAXraw": round(pm, 8), "OwnMAXraw": round(own, 8) if own is not None else "",
                "PeerReturn": round(pr, 8),
                "support_mass": round(support, 6), "covered_mass": round(covered, 6),
                "coverage_ratio": round(covered / support, 6) if support > 0 else 0,
            })
            stats["support_mass"] += support
            stats["covered_mass"] += covered
        if mi % 12 == 0:
            print(f"  {month}: {len(out_rows):,} focal-month 累计")

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    report = {
        "report_id": "peermax_panel_pilot",
        "months": len(months),
        "focal_month_rows": len(out_rows),
        "support_mass": round(stats["support_mass"], 4),
        "covered_mass": round(stats["covered_mass"], 4),
        "overall_coverage_ratio": round(stats["covered_mass"] / stats["support_mass"], 6) if stats["support_mass"] else None,
        "no_support_focals": stats["no_support_focals"],
        "note": "网络版本：研究期年份 Y 用年报年 Y-1（event-prior）；IVOL/Skew/ID 另步；limit_hit 分层待 S4 收口时并入",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"focal-month {len(out_rows):,}，覆盖 {report['overall_coverage_ratio']}")
    print(f"输出: {OUT}\n     {OUT_JSON}")


if __name__ == "__main__":
    main()
