#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inventory Step 4：universe 三分类 + 停牌核验 + 双源交叉抽检（execution_protocol §2 Step 4）。

4a universe 三分类：expected（inventory_expected_universe.json 8,612,917）− observed（腾讯数据行 8,546,009）
   − 停牌（4b）→ 缺失；停牌核验覆盖期精确拆分，断档期（2020-10-30~2025-07-15）合并"待核验"。
4b 停牌独立核验：baostock stock_list 967 文件 × 2 列（code,tradeStatus），filter 5543 只股票。
4c 双源交叉抽检：新浪 quotes.sina.cn 30 只 × 2020 全年，腾讯 raw close 派生 vs 新浪 close 派生。
4d 时序/泄漏：登记确认（历史快照无前视），无计算。
产出：reports/data_gates/G0/inventory_coverage_report.json

用法: python scripts/inventory_coverage.py [--skip-net]
"""
import argparse
import csv
import http.client
import json
import random
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOCK_LIST = ROOT / "data/raw/baostock/stock_list"
A_STOCKS = ROOT / "data/raw/baostock/stock_basic/a_stocks.csv"
TENCENT = ROOT / "data/raw/tencent"
EXPECTED = ROOT / "reports/data_gates/G0/inventory_expected_universe.json"
OUT = ROOT / "reports/data_gates/G0/inventory_coverage_report.json"

# 草案阈值（用户确认后冻结）
DRAFT_UNIVERSE_MISSING_FLOOR = 0.01
DRAFT_EPSILON_RETURN_SOURCE = 0.005
DRAFT_SOURCE_CONFLICT_RATE = 0.001
OBSERVED_ROWS = 8546009  # Step3 3d 实测 = MANIFEST


def load_stock_codes():
    with open(A_STOCKS, encoding="utf-8") as f:
        return {r["code"] for r in csv.DictReader(f)}


def read_stock_list(path, valid_codes):
    susp = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["code"] in valid_codes and r["tradeStatus"] == "0":
                susp.append(r["code"])
    return susp


def sina_returns(code, datalen=1900):
    """新浪日线 close：{date: close}。返回 None 失败。"""
    path = f"/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol={code}&scale=240&ma=no&datalen={datalen}"
    try:
        conn = http.client.HTTPSConnection("quotes.sina.cn", timeout=20)
        conn.request("GET", path, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
        r = conn.getresponse()
        body = r.read().decode("utf-8")
        conn.close()
        return {d["day"]: float(d["close"]) for d in json.loads(body)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-net", action="store_true")
    args = ap.parse_args()

    exp = json.loads(EXPECTED.read_text(encoding="utf-8"))
    expected_total = exp["expected_firm_days"]

    # ---------- 4b 停牌独立核验（967 文件 × 2 列） ----------
    valid = load_stock_codes()
    day_files = sorted(f for f in STOCK_LIST.iterdir() if f.name.endswith(".csv") and f.name != "failed_days.csv")
    failed_days = [l.strip() for l in (STOCK_LIST / "failed_days.csv").read_text(encoding="utf-8").splitlines() if l.strip() and l.strip() != "date"]
    susp_counts = Counter()
    susp_firm_days_total = 0
    for f in day_files:
        d = f.stem
        n = len(read_stock_list(f, valid))
        susp_counts[d] = n
        susp_firm_days_total += n
    n_covered_days = len(day_files)
    print(f"4b 停牌核验: {n_covered_days} 天（+failed {len(failed_days)}），停牌 firm-day 合计 {susp_firm_days_total:,}")

    # ---------- 4a universe 三分类 ----------
    # 停牌核验覆盖期（967 天）内：缺失 = expected(同期) − observed(同期) − 停牌(同期)
    # 简化精确口径：对覆盖期日期，expected 同期按比例估算（按日粒度需重建 expected 日期集，
    # 这里用覆盖期比例近似并注明；daily-panel 生成 firm-day 面板后精确复算）
    covered_ratio = n_covered_days / 1942
    expected_covered = int(expected_total * covered_ratio)
    observed_covered = int(OBSERVED_ROWS * covered_ratio)
    missing_covered = expected_covered - observed_covered - susp_firm_days_total
    missing_rate_covered = missing_covered / expected_covered if expected_covered else None
    gap_period_note = ("断档期 2020-10-30~2025-07-15（975 天）停牌/缺失合并'待核验'，"
                       "不强行拆分；该段无行 firm-day = expected(段内) − observed(段内)，待停牌列表续传后复算")
    # 全期口径：expected − observed = 停牌+缺失 合计（断档段无法拆分）
    unobserved_total = expected_total - OBSERVED_ROWS

    # ---------- 4c 双源交叉抽检（新浪 30 只 × 2020 全年） ----------
    if args.skip_net:
        dual = {"skipped": True}
    else:
        rng = random.Random(20260809)
        codes = sorted(d.name for d in (TENCENT / "daily_raw").iterdir() if d.is_dir())
        sample = rng.sample(codes, 30)
        checked = 0
        mismatch = 0
        conflicts = []
        for code in sample:
            p = TENCENT / "daily_raw" / code / "2020.csv"
            if not p.exists():
                continue
            rows = {}
            with open(p, encoding="utf-8") as f:
                for i, row in enumerate(csv.reader(f)):
                    if i == 0:
                        continue
                    rows[row[0]] = float(row[2])
            sina = sina_returns(code)
            time.sleep(0.3)
            if not sina:
                continue
            dates = sorted(set(rows) & set(sina))
            for i in range(1, len(dates)):
                d0, d1 = dates[i - 1], dates[i]
                rt = rows[d1] / rows[d0] - 1
                rs = sina[d1] / sina[d0] - 1
                checked += 1
                if abs(rt - rs) > DRAFT_EPSILON_RETURN_SOURCE:
                    mismatch += 1
                    conflicts.append({"code": code, "date": d1, "tencent": round(rt, 5), "sina": round(rs, 5)})
        conflict_rate = mismatch / checked if checked else None
        dual = {
            "source": "sina getKLineData",
            "sample": "30 只 × 2020 全年",
            "checked_days": checked,
            "mismatch_days": mismatch,
            "conflict_rate": conflict_rate,
            "draft_threshold": DRAFT_SOURCE_CONFLICT_RATE,
            "pass_draft": conflict_rate is not None and conflict_rate <= DRAFT_SOURCE_CONFLICT_RATE,
            "conflict_sample": conflicts[:10],
        }
        print(f"4c 双源抽检: checked={checked} mismatch={mismatch} rate={conflict_rate}")

    report = {
        "report_id": "inventory_coverage_report",
        "4a_universe_three_class": {
            "expected_firm_days": expected_total,
            "observed_rows_tencent": OBSERVED_ROWS,
            "unobserved_total_expected_minus_observed": unobserved_total,
            "unobserved_rate_total": round(unobserved_total / expected_total, 6),
            "suspension_covered_days": n_covered_days,
            "suspension_firm_days_covered_period": susp_firm_days_total,
            "missing_firm_days_covered_period": missing_covered,
            "missing_rate_covered_period": round(missing_rate_covered, 6) if missing_rate_covered else None,
            "draft_universe_missing_floor": DRAFT_UNIVERSE_MISSING_FLOOR,
            "gap_period_note": gap_period_note,
            "note": "覆盖期缺失率按 967/1942 比例近似（daily-panel firm-day 面板后精确复算）；断档期 975 天不拆分",
        },
        "4b_suspension": {
            "day_files": n_covered_days,
            "failed_days": len(failed_days),
            "failed_day_range": [failed_days[0], failed_days[-1]] if failed_days else [],
            "suspension_firm_days_total": susp_firm_days_total,
            "avg_suspension_per_day": round(susp_firm_days_total / n_covered_days, 1) if n_covered_days else None,
        },
        "4c_dual_source": dual,
        "4d_timing_leakage": {
            "status": "confirmed_no_leakage",
            "note": "全部数据为历史快照下载（2026-08-09 前），available_at 早于一切事件构造点；qfq 前复权基准=下载时点已登记",
        },
        "draft_parameters": {
            "universe_missing_floor": DRAFT_UNIVERSE_MISSING_FLOOR,
            "epsilon_return_source": DRAFT_EPSILON_RETURN_SOURCE,
            "epsilon_source_conflict_rate": DRAFT_SOURCE_CONFLICT_RATE,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"4a: expected {expected_total:,} − observed {OBSERVED_ROWS:,} = 未观测 {unobserved_total:,} "
          f"({unobserved_total/expected_total*100:.2f}%)")
    print(f"4a: 覆盖期缺失 {missing_covered:,}（率 {report['4a_universe_three_class']['missing_rate_covered_period']}）")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
