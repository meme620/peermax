#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3-2：年报业务章节提取（公司业务概要/主营业务）。

锚点优先序：
  1. "报告期内公司从事的主要业务"（深交所格式，最精确）
  2. "主要业务"（正文首次出现，跳过目录"...."行）
  3. "公司业务概要" / "业务概要"（2020 前格式）
截取至下一节标题（"经营情况讨论与分析"/"主营业务分析"/"报告期内公司经营"）或固定 900 字符。
清洗：去目录点线、去空白、去表格符。
输出：data/processed/business_texts.csv（code, year, text, anchor, length）
用法: python scripts/data_audit/s3_extract_business.py [--limit N]
"""
import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data/raw/cninfo/annual_txt"
OUT = ROOT / "data/processed/business_texts.csv"
MAX_LEN = 900

ANCHORS = ["报告期内公司从事的主要业务", "主要业务", "公司业务概要", "业务概要"]
STOP_TITLES = ["经营情况讨论与分析", "主营业务分析", "报告期内公司经营", "公司未来发展的展望",
               "报告期内主要经营情况", "公司经营情况"]


def to_panel_code(num_code):
    """纯数字 code → sh/sz 前缀（面板 code 体系）。"""
    if num_code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + num_code
    if num_code.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "sz" + num_code
    return num_code


def is_table_text(text):
    """（v5：仅保留极端表格特征，防误杀；文档级去重处理模板文本）"""
    return text.count("子公司") >= 8 or text.count("有限公司") >= 10


def extract_business(t):
    """返回 (text, anchor) 或 (None, None)。"""
    for anchor in ANCHORS:
        idx = t.find(anchor)
        if idx < 0:
            continue
        # 跳过目录行（含点线）
        seg_start = idx
        for _ in range(3):  # 最多跳 3 次目录误匹配
            line_end = t.find("\n", seg_start)
            line = t[seg_start:line_end if line_end > 0 else seg_start + 200]
            if "...." in line or "…" in line or len(line.strip()) < 10:
                nxt = t.find("\n", line_end + 1) if line_end > 0 else -1
                if nxt < 0:
                    break
                seg_start = nxt + 1
            else:
                break
        # 截取到下一个停止标题
        end_candidates = [t.find(st, seg_start) for st in STOP_TITLES if t.find(st, seg_start) > 0]
        end = min(end_candidates) if end_candidates else seg_start + MAX_LEN
        text = t[seg_start:end]
        # 清洗
        text = re.sub(r"[ \t\u3000]+", "", text)
        text = re.sub(r"\.{4,}", " ", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"[，。；、：（）()\-—]{2,}", " ", text)
        if len(text) < 30:
            continue  # 提取太短，试下一锚点
        text = text[:MAX_LEN]  # 先截断再检查表格特征（防全文误杀）
        if is_table_text(text):
            continue  # 极端表格误命中，试下一锚点
        return text, anchor
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    n_firms = 0
    n_fail = 0
    anchors = {}
    for code_dir in sorted(TXT.iterdir()):
        if not code_dir.is_dir():
            continue
        if args.limit and n_firms >= args.limit:
            break
        for f in sorted(code_dir.iterdir()):
            if f.suffix != ".txt":
                continue
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if len(t) < 500:
                continue
            text, anchor = extract_business(t)
            if text is None:
                n_fail += 1
                continue
            anchors[anchor] = anchors.get(anchor, 0) + 1
            rows.append({"code": to_panel_code(code_dir.name), "year": f.stem, "text": text, "anchor": anchor,
                         "length": len(text)})
        n_firms += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # 文档级去重：完全相同的 text 只保留 1 份（模板/误提取文本，其余剔除登记）
    seen_text = {}
    dup_removed = 0
    dedup_rows = []
    for r in rows:
        key = (r["year"], r["text"])
        if key in seen_text:
            dup_removed += 1
            continue
        seen_text[key] = True
        dedup_rows.append(r)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["code", "year", "text", "anchor", "length"])
        w.writeheader()
        w.writerows(dedup_rows)
    total = len(dedup_rows)
    print(f"提取成功 {total:,} 份（原始 {len(rows):,}，去重剔除 {dup_removed:,}），{n_firms} 只股票")
    print("锚点分布:", anchors)
    print(f"输出: {OUT}")
    # 样例
    for r in rows[:3]:
        print(f"  [{r['code']}/{r['year']}/{r['anchor']}] {r['text'][:80]}...")


if __name__ == "__main__":
    main()
