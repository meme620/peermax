#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S1 Step 3：schema/主键/代码映射抽检（execution_protocol §2 Step 3）。

3a 代码映射：腾讯目录名 vs a_stocks code 差集（零内容读取）
3d 文件级行数统计：全量 88688 文件只数行数（不解析），Σ vs MANIFEST 8,546,009
3c 主键/列数抽检：分层抽样 100 只 × 2 年文件，查 (code,date) 重复/列数/日期递增
3f qfq/raw close 比值排查：抽样 50 只 × 2020 年，找比值跳变（除权事件）与非跳变区间 CV
3e pct_chg 抽验：抽样 10 只 × 连续 3 天 vs eastmoney push2his K 线（urllib，标准库）
产出：reports/data_gates/G0/s1_schema_check.json + field_mapping_manifest.json

用法: python scripts/inventory_schema_check.py [--skip-net]
"""
import argparse
import csv
import json
import random
import statistics
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TENCENT = ROOT / "data/raw/tencent"
A_STOCKS = ROOT / "data/raw/baostock/stock_basic/a_stocks.csv"
OUT_SCHEMA = ROOT / "reports/data_gates/G0/s1_schema_check.json"
OUT_MAPPING = ROOT / "reports/data_gates/G0/field_mapping_manifest.json"
EXPECTED_TOTAL_ROWS = 8546009  # MANIFEST.json rows（raw/qfq 各）
SAMPLE_STOCKS = 100
SAMPLE_YEARS = ["2018", "2023"]
RATIO_SAMPLE = 50
RATIO_YEAR = "2020"


def count_lines(path):
    """只数换行符，不解析内容；表头文件返回 1。"""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def load_stock_codes():
    with open(A_STOCKS, encoding="utf-8") as f:
        return [r["code"].replace(".", "") for r in csv.DictReader(f)]


def derive_board(code):
    sym = code[2:] if code.startswith(("sh", "sz")) else code
    if code.startswith("sh"):
        if sym.startswith(("688", "689")):
            return "star"
        if sym.startswith(("600", "601", "603", "605")):
            return "main"
        return None
    if sym.startswith(("300", "301", "302")):
        return "chinext"
    if sym.startswith(("000", "001", "002", "003")):
        return "main"
    return None


def eastmoney_returns(code, begin, end):
    """东财 push2his 不复权日线：返回 {date: pct_chg}。code 形如 sh600000。"""
    market = "1" if code.startswith("sh") else "0"
    url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get?secid={market}.{code[2:]}"
           f"&fields1=f1,f2,f3&fields2=f51,f53,f59&klt=101&fqt=0&beg={begin}&end={end}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 禁用失效系统代理（gap_fix §18 已知坑）
    with opener.open(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = {}
    for k in (data.get("data") or {}).get("klines", []):
        parts = k.split(",")
        out[parts[0]] = float(parts[2])  # f59 = 涨跌幅%
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-net", action="store_true", help="跳过 3e 东财 API 抽验")
    args = ap.parse_args()

    result = {}

    # ---------- 3a 代码映射（零内容读取） ----------
    tencent_codes = sorted(d for d in (TENCENT / "daily_raw").iterdir() if d.is_dir())
    tc_set = {d.name for d in tencent_codes}
    stock_codes = load_stock_codes()
    sc_set = set(stock_codes)
    diff = tc_set ^ sc_set
    result["3a_code_mapping"] = {
        "n_tencent_dirs": len(tc_set),
        "n_stock_basic": len(sc_set),
        "diff": sorted(diff),
        "pass": len(diff) == 0,
    }

    # ---------- 3d 文件级行数统计（全量，不解析内容） ----------
    per_version = {}
    for ver in ["daily_raw", "daily_qfq"]:
        total = 0
        nonempty = 0
        empty = 0
        n_files = 0
        for code_dir in (TENCENT / ver).iterdir():
            if not code_dir.is_dir():
                continue
            for f in code_dir.iterdir():
                n_files += 1
                n = count_lines(f)
                if n > 1:
                    nonempty += 1
                    total += n - 1
                else:
                    empty += 1
        per_version[ver] = {"n_files": n_files, "nonempty": nonempty, "empty": empty, "data_rows": total}
    rows_raw = per_version["daily_raw"]["data_rows"]
    rows_qfq = per_version["daily_qfq"]["data_rows"]
    result["3d_file_rows"] = {
        "per_version": per_version,
        "expected_rows_manifest": EXPECTED_TOTAL_ROWS,
        "diff_raw": rows_raw - EXPECTED_TOTAL_ROWS,
        "diff_qfq": rows_qfq - EXPECTED_TOTAL_ROWS,
        "pass": rows_raw == EXPECTED_TOTAL_ROWS and rows_qfq == EXPECTED_TOTAL_ROWS,
    }

    # ---------- 3c 主键/列数抽检（分层抽样 100 只 × 2 年） ----------
    rng = random.Random(20260809)
    by_board = {"main": [], "chinext": [], "star": []}
    for c in tc_set:
        b = derive_board(c)
        if b:
            by_board[b].append(c)
    sample = []
    for b, codes in by_board.items():
        n = max(1, round(SAMPLE_STOCKS * len(codes) / len(tc_set)))
        sample.extend(rng.sample(codes, min(n, len(codes))))
    dup_issues, col_issues, mono_issues, rows_checked = [], [], [], 0
    for code in sample:
        for yr in SAMPLE_YEARS:
            p = TENCENT / "daily_raw" / code / f"{yr}.csv"
            if not p.exists():
                continue
            dates = []
            cols = None
            with open(p, encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        cols = len(row)
                        continue
                    dates.append(row[0])
            rows_checked += len(dates)
            if cols != 10:
                col_issues.append(f"{code}/{yr}: {cols} cols")
            if len(set(dates)) != len(dates):
                dup_issues.append(f"{code}/{yr}: {len(dates)-len(set(dates))} dups")
            if dates != sorted(dates):
                mono_issues.append(f"{code}/{yr}: date 非递增")
    result["3c_key_check"] = {
        "sample_n": len(sample),
        "years": SAMPLE_YEARS,
        "rows_checked": rows_checked,
        "duplicate_issues": dup_issues,
        "column_issues": col_issues,
        "monotonic_issues": mono_issues,
        "pass": not (dup_issues or col_issues or mono_issues),
    }

    # ---------- 3f qfq/raw 比值排查（50 只 × 2020） ----------
    ratio_codes = rng.sample(sorted(tc_set), min(RATIO_SAMPLE, len(tc_set)))
    jumps = []
    cv_list = []
    for code in ratio_codes:
        p_raw = TENCENT / "daily_raw" / code / f"{RATIO_YEAR}.csv"
        p_qfq = TENCENT / "daily_qfq" / code / f"{RATIO_YEAR}.csv"
        if not (p_raw.exists() and p_qfq.exists()):
            continue
        rows = {}
        for p, tag in ((p_raw, "raw"), (p_qfq, "qfq")):
            with open(p, encoding="utf-8") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i == 0:
                        continue
                    rows.setdefault(row[0], {})[tag] = float(row[2])  # close
        dates = sorted(rows)
        ratios = []
        for d in dates:
            if "raw" in rows[d] and "qfq" in rows[d] and rows[d]["raw"] > 0:
                ratios.append((d, rows[d]["qfq"] / rows[d]["raw"]))
        if len(ratios) < 20:
            continue
        vals = [r for _, r in ratios]
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals)
        cv = sd / mean if mean else float("nan")
        cv_list.append(cv)
        # 跳变检测：相邻比值差 > 0.1%
        for i in range(1, len(ratios)):
            if abs(ratios[i][1] - ratios[i - 1][1]) / ratios[i - 1][1] > 0.001:
                jumps.append({"code": code, "date": ratios[i][0],
                              "ratio_before": round(ratios[i - 1][1], 6),
                              "ratio_after": round(ratios[i][1], 6)})
    n_jump_codes = len(set(j["code"] for j in jumps))
    result["3f_qfq_ratio"] = {
        "sample_n": len(ratio_codes),
        "year": RATIO_YEAR,
        "jumps_total": len(jumps),
        "codes_with_jumps": n_jump_codes,
        "jump_sample": jumps[:15],
        "cv_min": round(min(cv_list), 6) if cv_list else None,
        "cv_max": round(max(cv_list), 6) if cv_list else None,
        "pass": n_jump_codes == 0,
        "note": "跳变应出现在除权除息日（公司行为）；无除权区间比值应恒定；>0.1% 视为跳变",
    }

    # ---------- 3e pct_chg 抽验（10 只 × 连续 3 天 vs 东财） ----------
    if args.skip_net:
        result["3e_pct_chg"] = {"skipped": True}
    else:
        pct_codes = rng.sample(sorted(tc_set), 10)
        mismatches = []
        checked = 0
        for code in pct_codes:
            p = TENCENT / "daily_raw" / code / "2020.csv"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = [(r[0], float(r[7])) for r in reader if r[0] != "date"]
            if len(rows) < 4:
                continue
            sample_days = rows[2:5]
            begin, end = sample_days[0][0].replace("-", ""), sample_days[-1][0].replace("-", "")
            try:
                em = eastmoney_returns(code, begin, end)
            except Exception:
                continue
            for d, pct_tencent in sample_days:
                if d in em:
                    checked += 1
                    if abs(pct_tencent - em[d]) > 0.05:  # 0.05pp 容差
                        mismatches.append({"code": code, "date": d,
                                           "tencent_pct_chg": pct_tencent,
                                           "eastmoney_pct_chg": em[d]})
        result["3e_pct_chg"] = {
            "checked_days": checked,
            "mismatch_days": len(mismatches),
            "mismatch_sample": mismatches[:15],
            "pass": len(mismatches) == 0,
            "note": "腾讯 pct_chg 与东财涨跌幅可比对；已确认浦发 2018-01-03 不符（0.13 vs -0.47）",
        }

    OUT_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    OUT_SCHEMA.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # field_mapping_manifest（含 pct_chg 废弃记录）
    mapping = {
        "manifest_id": "field_mapping_manifest",
        "created": "2026-08-09",
        "source": "tencent newfqkline（raw/qfq 两版同 schema）",
        "fields": [
            {"source_field": "date", "canonical": "date", "type": "str YYYY-MM-DD", "note": "交易日（无行=停牌/未上市/退市）"},
            {"source_field": "open", "canonical": "open_price", "type": "float 元"},
            {"source_field": "close", "canonical": "close_price", "type": "float 元", "note": "rraw 主路径：qfq close 派生收益"},
            {"source_field": "high", "canonical": "high_price", "type": "float 元"},
            {"source_field": "low", "canonical": "low_price", "type": "float 元"},
            {"source_field": "volume", "canonical": "volume", "type": "float 手", "note": "Turnover 分子候选"},
            {"source_field": "extra", "canonical": "info_unused", "type": "str", "note": "接口占位列，弃用"},
            {"source_field": "pct_chg", "canonical": "DEPRECATED", "type": "float", "note": "实测与涨跌幅不符（浦发 2018-01-03：0.13 vs 东财 -0.47），禁止用作 rraw；由 close 派生替代"},
            {"source_field": "amount", "canonical": "dollar_volume", "type": "float 万元", "note": "Liquidity 分母候选"},
            {"source_field": "extra2", "canonical": "unused", "type": "str", "note": "raw 版 0.00,0.00 / qfq 版空，弃用"},
        ],
        "code_mapping": "sh600000 -> firm_id（目录名）；board 派生：600/601/603/605=main(sh)，688/689=star，000/001/002/003=main(sz)，300/301/302=chinext",
        "notes": ["北交所不在数据源（排除登记）", "qfq 前复权基准=2026-08-09 下载时点", "schema 检查详见 s1_schema_check.json"],
    }
    OUT_MAPPING.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAPPING.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要
    print("=== 3a 代码映射 ===")
    print(f"  腾讯目录 {result['3a_code_mapping']['n_tencent_dirs']} vs stock_basic {result['3a_code_mapping']['n_stock_basic']}，差集 {len(diff)} -> {'PASS' if not diff else 'FAIL'}")
    print("=== 3d 文件行数（全量）===")
    for ver, v in per_version.items():
        print(f"  {ver}: {v['n_files']} 文件，非空 {v['nonempty']}，空 {v['empty']}，数据行 {v['data_rows']:,}")
    print(f"  期望 {EXPECTED_TOTAL_ROWS:,} -> {'PASS' if result['3d_file_rows']['pass'] else 'FAIL'}")
    print("=== 3c 主键/列数抽检 ===")
    c = result["3c_key_check"]
    print(f"  抽样 {c['sample_n']} 只 × {c['years']}，行 {c['rows_checked']:,}；重复 {len(c['duplicate_issues'])}，列错 {len(c['column_issues'])}，非递增 {len(c['monotonic_issues'])} -> {'PASS' if c['pass'] else 'FAIL'}")
    print("=== 3f qfq/raw 比值 ===")
    r = result["3f_qfq_ratio"]
    print(f"  抽样 {r['sample_n']} 只×{r['year']}；跳变 {r['jumps_total']} 次 / {r['codes_with_jumps']} 只；CV [{r['cv_min']}, {r['cv_max']}]")
    print("=== 3e pct_chg 抽验 ===")
    if result["3e_pct_chg"].get("skipped"):
        print("  已跳过（--skip-net）")
    else:
        e = result["3e_pct_chg"]
        print(f"  比对 {e['checked_days']} 天，不符 {e['mismatch_days']} 天 -> {'PASS(需结合已知浦发不符结论)' if e['pass'] else 'FAIL'}")
    print(f"\n输出: {OUT_SCHEMA}\n     {OUT_MAPPING}")


if __name__ == "__main__":
    main()
