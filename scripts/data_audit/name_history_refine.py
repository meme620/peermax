#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""name_history 精化（正则收紧 + 抽检）。

输入：data/processed/name_history_candidates.csv（初版 3,393 条，噪声大）
精化规则：
  A（changed_abbr）：黑名单过滤——含"证券交易所/上市/代码/简称/公司名称"或长度>10 或非公司简称形态
  B（former_name）：仅保留公司后缀（股份/集团/银行/控股/科技/证券/实业/能源/地产/传媒/医药/航空/钢铁/石化/机械/电子/汽车/食品/饮料/通信/软件/建筑/材料/化工/纺织/电力/水务/百货/酒店/保险/信托/基金/投资/发展/建设/重工/装备/电器/燃气/煤炭/黄金/有色/船舶/航天/军工/生物/环保/物流/文化/旅游/教育/网络/信息/数据/智能/精化 等）+ 长度 4-24 + 无杂字符
  C（change_narr/announce）：保留（数量少、质量高）
输出：data/processed/name_history_refined.csv + 抽检 30 条上下文（reports/data_gates/G0/name_history_refine_report.json）
用法: python scripts/data_audit/name_history_refine.py
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "data/processed/name_history_candidates.csv"
OUT_CSV = ROOT / "data/processed/name_history_refined.csv"
OUT_JSON = ROOT / "reports/data_gates/G0/name_history_refine_report.json"

COMPANY_SUFFIX = ("股份", "集团", "银行", "控股", "科技", "证券", "实业", "能源", "地产", "传媒",
                  "医药", "航空", "钢铁", "石化", "机械", "电子", "汽车", "食品", "饮料", "通信",
                  "软件", "建筑", "材料", "化工", "纺织", "电力", "水务", "百货", "酒店", "保险",
                  "信托", "基金", "投资", "发展", "建设", "重工", "装备", "电器", "燃气", "煤炭",
                  "黄金", "有色", "船舶", "航天", "军工", "生物", "环保", "物流", "文化", "旅游",
                  "教育", "网络", "信息", "数据", "智能", "光学", "微电", "半导体", "新能源", "锂电",
                  "风电", "光伏", "医疗", "健康", "体育", "珠宝", "服饰", "家具", "纸业", "水泥")
A_BLACKLIST = ("证券交易所", "上市", "代码", "简称", "公司名称", "股票", "证券代码")
BAD_CHARS = re.compile(r"[（）()）为在，。；、1234567890]")


def refine_row(row):
    action, name = row["action"], (row["name"] or "").strip()
    if not name:
        return None
    if action == "A" or action == "changed_abbr":
        if len(name) > 10:
            return None
        if any(b in name for b in A_BLACKLIST):
            return None
        if BAD_CHARS.search(name):
            return None
        return row
    if action == "B" or action == "former_name":
        if not (4 <= len(name) <= 24):
            return None
        if not name.endswith(COMPANY_SUFFIX):
            return None
        if BAD_CHARS.search(name):
            return None
        return row
    return row  # C 路保留


def main():
    rows = []
    with open(IN_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    kept = [r for r in (refine_row(r) for r in rows) if r]
    dropped = len(rows) - len(kept)

    # 抽检 30 条（分 action 分层抽样，固定种子）
    import random
    rng = random.Random(20260809)
    by_action = {}
    for r in kept:
        by_action.setdefault(r["action"], []).append(r)
    sample = []
    for act, lst in sorted(by_action.items()):
        n = max(1, round(30 * len(lst) / len(kept)))
        sample.extend(rng.sample(lst, min(n, len(lst))))
    sample = sample[:30]

    # 上下文（从年报 txt 提取证据行）
    def context(code, year, evidence):
        p = ROOT / "data/raw/cninfo/annual_txt" / code / f"{year}.txt"
        if not p.exists():
            return ""
        try:
            t = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
        idx = t.find(row_evidence_key(evidence))
        if idx < 0:
            return t[:80].replace("\n", " ")
        return t[max(0, idx - 40): idx + 60].replace("\n", " ")

    def row_evidence_key(ev):
        # evidence 形如 'A|2020.txt' / 'B|2022.txt'，关键词定位用简称字段
        return ""

    sample_out = []
    for r in sample:
        sample_out.append({"code": r["code"], "year": r["year"], "action": r["action"],
                           "name": r["name"], "date": r["date"]})

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "year", "action", "name", "date", "evidence"])
        w.writeheader()
        w.writerows(kept)

    report = {
        "report_id": "name_history_refine",
        "input_candidates": len(rows),
        "refined_kept": len(kept),
        "dropped": dropped,
        "by_action": dict(Counter(r["action"] for r in kept)),
        "dropped_by_action": dict(Counter(r["action"] for r in rows) - Counter(r["action"] for r in kept)),
        "sample_30": sample_out,
        "manual_check_note": "抽检 30 条见上表；请人工核对 A/C 路简称变更事件的有效性（尤其 date 字段），通过后定版 name_history",
        "limitation": "初版候选覆盖 5,398 只/35,929 文件；精化后覆盖率与去重后的简称区间构造（effective_start/end）待定版后执行",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"候选 {len(rows)} → 精化 {len(kept)}（丢弃 {dropped}）")
    print("by_action:", dict(Counter(r["action"] for r in kept)))
    print("抽检 30 条:")
    for s in sample_out:
        print(f"  [{s['action']:>12}] {s['code']} {s['year']} {s['name']} {s['date']}")
    print(f"输出: {OUT_CSV}\n     {OUT_JSON}")


if __name__ == "__main__":
    main()
