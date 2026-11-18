#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""name_history 候选提取（st_namechange_crosscheck 补救链，并行任务）。

源 1：CNINFO 年报 txt（D 类数据，2018-2025）——三路提取：
  A. "公司信息"表：变更后的股票简称（如有）字段 → 变更后简称候选（年报年=变更发生年附近）
  B. 正文"（曾用名"XX"）"引用 → 曾用名候选
  C. "变更证券简称"叙述段（含公告日期模式 YYYY 年 M 月 D 日）→ 变更事件候选
源 2：RPT_F10_BASIC_ORGINFO FORMERNAME（可选，--fetch-f10）
产出：data/processed/name_history_candidates.csv + reports/data_gates/G0/name_history_extract_report.json
注意：本步是候选提取（coverage 级），定版需人工抽检 + 与 ST 状态交叉（st_namechange_crosscheck_coverage）。
用法: python scripts/data_audit/name_history_extract.py [--limit N] [--fetch-f10]
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data/raw/cninfo/annual_txt"
OUT_CSV = ROOT / "data/processed/name_history_candidates.csv"
OUT_JSON = ROOT / "reports/data_gates/G0/name_history_extract_report.json"

# A: 公司信息表的"变更后的股票简称"字段（表格行模式：字段名 后跟 值）
PAT_CHANGED_ABBR = re.compile(r"变更后的股票简称\s*[（(]如有[)）]\s*([^\s\n]{2,12})")
# B: 曾用名引用
PAT_FORMER = re.compile(r"曾用名[（(]?\s*[“\"']?([\u4e00-\u9fa5A-Z0-9*ST（）()]{2,20})[”\"']?[)）]?")
# C: 变更叙述（"关于变更证券简称的公告" 或 "证券简称变更为" 带日期）
PAT_CHANGE_NARR = re.compile(r"([0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日)[^。；]{0,60}?证券简称变更为\s*[“\"']?([\u4e00-\u9fa5A-Z0-9*ST（）()]{2,12})")
PAT_CHANGE_NARR2 = re.compile(r"证券简称变更为\s*[“\"']?([\u4e00-\u9fa5A-Z0-9*ST（）()]{2,12})[”\"']?[^。；]{0,60}?([0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日)")
PAT_ANNOUNCE = re.compile(r"关于\s*变更证券简称\s*的公告[^。；]{0,40}([0-9]{4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日)?")


def norm_date(s):
    """'2020年12月1日' -> '2020-12-01'；失败返回原串。"""
    m = re.match(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", s or "")
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 code 目录（调试）")
    ap.add_argument("--fetch-f10", action="store_true", help="额外拉取 RPT_F10_BASIC_ORGINFO")
    args = ap.parse_args()

    out_rows = []
    stats = Counter()
    n_firms = 0
    n_files = 0

    for code_dir in sorted(TXT.iterdir()):
        if not code_dir.is_dir():
            continue
        if args.limit and n_firms >= args.limit:
            break
        code = code_dir.name
        firm_rows = []
        for f in sorted(code_dir.iterdir()):
            if f.suffix != ".txt":
                continue
            n_files += 1
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(t) < 500:
                continue
            year = f.stem
            # A: 变更后简称字段
            for m in PAT_CHANGED_ABBR.finditer(t):
                name = m.group(1).strip("，。；")
                if name and name not in ("无", "—", "-", "不适用"):
                    firm_rows.append({"code": code, "year": year, "action": "changed_abbr",
                                      "name": name, "date": "", "evidence": f"A|{f.name}"})
                    stats["A_changed_abbr"] += 1
            # B: 曾用名
            for m in PAT_FORMER.finditer(t):
                firm_rows.append({"code": code, "year": year, "action": "former_name",
                                  "name": m.group(1), "date": "", "evidence": f"B|{f.name}"})
                stats["B_former"] += 1
                break  # 每年报取 1 条曾用名（正文首处）
            # C: 变更叙述
            for m in PAT_CHANGE_NARR.finditer(t):
                firm_rows.append({"code": code, "year": year, "action": "change_narr",
                                  "name": m.group(2), "date": norm_date(m.group(1)), "evidence": f"C|{f.name}"})
                stats["C_narr"] += 1
            for m in PAT_CHANGE_NARR2.finditer(t):
                firm_rows.append({"code": code, "year": year, "action": "change_narr",
                                  "name": m.group(1), "date": norm_date(m.group(2)), "evidence": f"C2|{f.name}"})
                stats["C_narr2"] += 1
            for m in PAT_ANNOUNCE.finditer(t):
                firm_rows.append({"code": code, "year": year, "action": "announce",
                                  "name": "", "date": norm_date(m.group(1)) if m.group(1) else "",
                                  "evidence": f"C3|{f.name}"})
                stats["C_announce"] += 1
        # 去重（同 code 同年同 action 同 name）
        seen = set()
        for r in firm_rows:
            key = (r["year"], r["action"], r["name"], r["date"])
            if key in seen:
                continue
            seen.add(key)
            out_rows.append(r)
        n_firms += 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "year", "action", "name", "date", "evidence"])
        w.writeheader()
        w.writerows(out_rows)

    report = {
        "report_id": "name_history_extract",
        "firms_scanned": n_firms,
        "files_scanned": n_files,
        "candidates": len(out_rows),
        "by_action": {k: v for k, v in stats.items()},
        "note": "候选提取层：A=变更后简称字段；B=曾用名引用；C=变更叙述+公告日期。定版需人工抽检+与 ST 交叉",
        "sample": out_rows[:20],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"扫描 {n_firms} 只 / {n_files} 文件，候选 {len(out_rows)} 条")
    print("by_action:", dict(stats))
    print(f"输出: {OUT_CSV}")
    for r in out_rows[:15]:
        print("  ", r)


if __name__ == "__main__":
    main()
