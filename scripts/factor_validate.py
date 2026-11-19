#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""factor 验收：factor_returns_daily 7 项检查 + MKT vs 沪深300 交叉（sensitivity）。

检查：
  ① 设计矩阵 [1,MKT,SMB,HML,MOM] 满秩 5（高斯消元）
  ② 与 trading_calendar 对齐（因子日 ⊆ 面板交易日；研究期 2018-2025 全覆盖）
  ③ 覆盖 ≥2016 年底（IVOL 前置 252 日）
  ④ decimal 单位（|日收益| < 1）
  ⑤ factor_version 标识
  ⑥ 无前视（构造逻辑：月末排序次月组合——代码审计）
  ⑦ 缺失日不插值（因子日无缺失计数）
交叉：MKT_d vs 沪深300 日收益 Pearson 相关（新浪指数日线）
产出：reports/data_gates/G0/factor_validation_report.json
用法: python scripts/factor_validate.py [--skip-net]
"""
import argparse
import csv
import http.client
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTOR = ROOT / "data/processed/factor_returns_daily.csv"
PANEL = ROOT / "data/processed/daily_panel"
OUT = ROOT / "reports/data_gates/G0/factor_validation_report.json"


def rank_5(matrix_cols):
    """高斯消元求秩（列向量列表，浮点）。"""
    m = [list(col) for col in zip(*matrix_cols)]
    rows, cols = len(m), len(m[0])
    r = 0
    eps = 1e-9
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if abs(m[i][c]) > eps:
                pivot = i
                break
        if pivot is None:
            continue
        m[r], m[pivot] = m[pivot], m[r]
        pv = m[r][c]
        m[r] = [x / pv for x in m[r]]
        for i in range(rows):
            if i != r and abs(m[i][c]) > eps:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        r += 1
        if r == rows:
            break
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-net", action="store_true")
    args = ap.parse_args()

    rows = []
    with open(FACTOR, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    dates = [r["date"] for r in rows]
    print(f"因子日度: {len(rows)} 天, {dates[0]} ~ {dates[-1]}")

    # ① 满秩
    sample = rows[500:1000]  # 取一段检查
    cols = [[1.0] * len(sample)]
    for k in ["MKT_d", "SMB_d", "HML_d", "MOM_d"]:
        cols.append([float(r[k]) for r in sample])
    rk = rank_5(cols)
    print(f"① 设计矩阵秩: {rk}（需 5）")

    # ② 日历对齐：因子日 vs 面板交易日
    panel_days = set()
    for yf in sorted(PANEL.glob("*.csv")):
        with open(yf, encoding="utf-8") as f:
            for line in f:
                if line.startswith("code"):
                    continue
                panel_days.add(line.split(",")[1])
    missing = [d for d in dates if d not in panel_days]
    # 研究期 2018-2025 全覆盖？
    study_days = sorted(d for d in panel_days if "2018-01-01" <= d <= "2025-12-31")
    factor_set = set(dates)
    study_missing = [d for d in study_days if d not in factor_set]
    print(f"② 因子日不在面板的: {len(missing)}；研究期交易日缺因子: {len(study_missing)}")

    # ③ 覆盖
    cov_ok = dates[0] <= "2016-12-31" and dates[-1] >= "2025-12-31"
    print(f"③ 覆盖 2016 底~2025 末: {cov_ok}")

    # ④ decimal 单位
    unit_bad = sum(1 for r in rows if r["MKT_d"] and abs(float(r["MKT_d"])) >= 1)
    print(f"④ decimal 单位异常（|r|>=1）: {unit_bad}")

    # ⑤ factor_version
    versions = {r["factor_version"] for r in rows}
    print(f"⑤ factor_version: {versions}")

    # ⑦ 缺失
    miss_count = {k: sum(1 for r in rows if not r[k]) for k in ["MKT_d", "SMB_d", "HML_d", "MOM_d"]}
    print(f"⑦ 缺失: {miss_count}")

    # 交叉：MKT vs 沪深300（新浪）
    corr = None
    if not args.skip_net:
        try:
            path = "/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol=sh000300&scale=240&ma=no&datalen=4000"
            conn = http.client.HTTPSConnection("quotes.sina.cn", timeout=25)
            conn.request("GET", path, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            conn.close()
            idx = {d["day"]: float(d["close"]) for d in json.loads(body)}
            # 指数日收益
            idx_r = {}
            prev = None
            for d in sorted(idx):
                if prev:
                    idx_r[d] = idx[d] / prev - 1
                prev = idx[d]
            f_map = {r["date"]: float(r["MKT_d"]) for r in rows}
            common = [(f_map[d], idx_r[d]) for d in f_map if d in idx_r]
            if len(common) > 100:
                n = len(common)
                mx = sum(a for a, _ in common) / n
                my = sum(b for _, b in common) / n
                num = sum((a - mx) * (b - my) for a, b in common)
                dx = math.sqrt(sum((a - mx) ** 2 for a, _ in common))
                dy = math.sqrt(sum((b - my) ** 2 for _, b in common))
                corr = num / (dx * dy) if dx * dy else None
                print(f"交叉: MKT vs 沪深300 相关 = {corr:.4f}（{n} 日）")
        except Exception as e:
            print(f"交叉: 沪深300 拉取失败 {type(e).__name__}: {str(e)[:60]}")

    report = {
        "report_id": "factor_validation", "factor_version": "FF4_SELF_BUILT_v1",
        "checks": {
            "design_matrix_rank": rk, "rank_ok": rk == 5,
            "calendar_alignment": {"factor_days_not_in_panel": len(missing),
                                   "study_missing_factor_days": len(study_missing), "ok": len(missing) == 0 and len(study_missing) == 0},
            "coverage": {"first": dates[0], "last": dates[-1], "ok": cov_ok},
            "decimal_unit": {"bad_count": unit_bad, "ok": unit_bad == 0},
            "factor_version": list(versions),
            "no_lookahead": "构造逻辑月末排序/次月组合（代码审计）；BM 用 NOTICE_DATE≤排序日；股本用事件日≤排序日",
            "no_interpolation": {"missing": miss_count, "ok": all(v == 0 for v in miss_count.values())},
        },
        "mkt_vs_csi300_corr": corr,
        "expected_corr_note": "MKT 为全市场市值加权（含小盘），沪深300 为大市值加权——预期相关 0.85-0.97；若显著偏低检查排序/加权逻辑",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
