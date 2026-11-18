#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 老师语料章节身份扫描（只读，2026-09-23 / v1.2：+锚点质量分类 +--lexicon 对照）。

问：老师导出的 mda_data_2023.csv 到底导了年报的哪一章？前几年（2014-2020）里
    "公司业务概要 / 报告期内公司从事的主要业务" 正文是否真的一个字都没有？

口径（全部行首/行内正则，逐行流式，不改动任何数据）：
  1. 章节身份   每行首行的 "第X节 <章名>"（归一化空白）→ 判定导的是哪一章
  2. 交叉引用   含 "公司业务概要" 且匹配 (详见|参见|请见|见本报告|内容见|具体见|详情见|参照) 前缀
  3. 内嵌章节   行首出现 "第X节 公司业务概要"（A+H 等异版式公司，导出内真的含该章）
  4. 业务子节   行首编号小标题（一/二/三、（一）…）匹配"（所）从事的(主要)业务"，
                尾部 ≤45 字（容纳"…、经营模式及行业情况说明"长标题）；剔除
                主营业务分析类（分行业/分产品/分地区/毛利率/收入构成）误报

产出：reports/data_gates/G0/s3_teacher_corpus_chapter_scan.json
用法：python scripts/corpus_teacher_chapter_scan.py
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/teacher/mda_data_2023.csv"
OUT = ROOT / "reports/data_gates/G0/s3_teacher_corpus_chapter_scan.json"

CHAP = re.compile(r"^\d*\s*(第[一二三四五六七八九十]+节\s*[^\n]{2,20})")
XREF = re.compile(r"(详见|参见|请见|见本报告|内容见|具体见|详情见|参照)[^。\n]{0,25}公司业务概要")
EMBED_CHAP = re.compile(r"(?m)^\s*第[一二三四五六七八九十]+节\s*公司业务概要\s*$")
SUB_HEAD = re.compile(
    r"(?m)^\s*[（(]?[一二三四五六七八九十]{1,2}[、.）)]\s*[^\n]{0,20}?"
    r"(?:报告期内|本年度内|本报告期内)?\s*(?:公司|本集团|本公司)?\s*(?:所)?\s*从事的?(?:主要)?业务[^\n]{0,45}$"
)
SUB_BAD = re.compile(r"分行业|分产品|分地区|毛利率|营业收入构成|收入构成")
LITERAL_A = re.compile(r"报告期内公司从事的主要业务")
IND_INFO = re.compile(r"行业经营性信息分析")  # 深市行业指引驱动的业务散文节（旧年 MD&A 内的替代性业务材料）
CORE_HEAD = re.compile(r"(?m)^\s*[（(]?[一二三四五六七八九十]{1,2}[、.）)]\s*核心竞争力分析")
CHAP_ANY = re.compile(r"^\s*\d*\s*第[一二三四五六七八九十]+(节|章)")
XREF_FIRST = re.compile(r"(详见|参见|请见|请阅读|见本报告|具体见|内容见|详情见|阅见)")
SUBTITLE_FIRST = re.compile(r"^\s*\d*\s*[（(]?[一二三四五六七八九十]{1,2}[、.）)]")
PAGENO_FIRST = re.compile(r"^\s*[\d\s/－\-\u2014]*\d+\s*[\d\s/－\-\u2014]*$")
CLASSIC_CASES = {("601975", "2019"), ("600519", "2019")}
# 业务描述节内容：FY2014 仍内嵌于 董事会报告；2015-2020 外移至第三节 公司业务概要；2021+ 并入 MD&A


def main() -> None:
    csv.field_size_limit(10**9)
    per_year: dict[str, Counter] = defaultdict(Counter)
    chap_titles: dict[str, Counter] = defaultdict(Counter)
    samples: dict[str, list] = defaultdict(list)
    aq_samples: dict[str, list] = defaultdict(list)
    classic: dict[tuple, dict] = {}
    lens: dict[str, list] = defaultdict(list)

    with open(SRC, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            yr = (r.get("f_year") or "").strip()
            txt = r.get("f_chapter_content") or ""
            c = per_year[yr]
            c["rows"] += 1

            m = CHAP.match(txt.strip())
            chap_titles[yr][re.sub(r"\s+", " ", m.group(1)).strip() if m else "<无标题开头>"] += 1

            if "公司业务概要" in txt:
                c["mentions_business_overview"] += 1
            if XREF.search(txt):
                c["xref_business_overview"] += 1
            if EMBED_CHAP.search(txt):
                c["embedded_old_chapter"] += 1
            if LITERAL_A.search(txt):
                c["literal_new_heading"] += 1
            if IND_INFO.search(txt):
                c["industry_info_section"] += 1
            if "核心竞争力" in txt:
                c["mentions_core_competence"] += 1
            if CORE_HEAD.search(txt):
                c["core_competence_heading"] += 1
            if "主营业务构成" in txt:
                c["mentions_main_business_composition"] += 1
            lens[yr].append(len(txt))

            # --- v1.2 锚点质量：不以章节标题开头的行分类 ---
            if not CHAP_ANY.match(txt.strip()):
                c["not_at_chapter_title"] += 1
                first = re.split(r"[\r\n]", txt.strip())[0][:80] if txt.strip() else ""
                if XREF_FIRST.search(first) and ("节" in first or "章" in first):
                    cls = "A_xref"
                elif PAGENO_FIRST.match(first) or len(first) <= 6:
                    cls = "C_pageno"
                elif SUBTITLE_FIRST.match(first):
                    cls = "B_mid_subsection"
                else:
                    cls = "D_other"
                c[f"aq_{cls}"] += 1
                if cls == "A_xref" and "核心竞争力" in txt:
                    c["aq_A_with_core_competence"] += 1
                if len(aq_samples[yr]) < 12:
                    aq_samples[yr].append({"cls": cls, "stock": r.get("f_stock_id"), "first": first})

            key = (str(r.get("f_stock_id") or "").zfill(6), yr)
            if key in CLASSIC_CASES:
                classic[key] = {
                    "len": len(txt),
                    "first": re.split(r"[\r\n]", txt.strip())[0][:60],
                    "has_core_competence": "核心竞争力" in txt,
                    "has_business_overview": "公司业务概要" in txt,
                }

            hit = next((mm for mm in SUB_HEAD.finditer(txt) if not SUB_BAD.search(mm.group(0))), None)
            if hit:
                c["business_subsection"] += 1
                if yr <= "2020" and len(samples[yr]) < 3:
                    samples[yr].append({
                        "stock": r.get("f_stock_id"),
                        "ctx": txt[max(0, hit.start() - 60):hit.start() + 120].replace("\n", " | "),
                    })

    out = {
        "step": "S3 老师语料章节身份扫描",
        "source": "data/raw/teacher/mda_data_2023.csv",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "detectors": {
            "chapter": "首行 第X节 <章名>",
            "business_subsection": "行首编号小标题匹配 (所)从事的(主要)业务[情况]（尾 ≤45 字，剔主营业务分析类）",
            "literal_new_heading": "字面串 报告期内公司从事的主要业务",
            "xref": "(详见|参见|…)[^。]{0,25}公司业务概要",
        },
        "per_year": {
            y: {
                **{k: per_year[y][k] for k in (
                    "rows", "mentions_business_overview", "xref_business_overview",
                    "embedded_old_chapter", "literal_new_heading", "business_subsection",
                    "industry_info_section", "mentions_core_competence", "core_competence_heading",
                    "mentions_main_business_composition")},
                "length_median": int(statistics.median(lens[y])),
                "anchor_quality": {k: per_year[y][k] for k in (
                    "not_at_chapter_title", "aq_A_xref", "aq_A_with_core_competence",
                    "aq_B_mid_subsection", "aq_C_pageno", "aq_D_other")},
                "chapter_title_top": chap_titles[y].most_common(5),
                "pct_business_subsection": round(per_year[y]["business_subsection"] / per_year[y]["rows"] * 100, 1),
                "pct_literal_new_heading": round(per_year[y]["literal_new_heading"] / per_year[y]["rows"] * 100, 1),
            }
            for y in sorted(per_year)
        },
        "samples_old_years_subsection_hits": dict(samples),
        "anchor_quality_samples": dict(aq_samples),
        "classic_cases": {f"{k[0]}_{k[1]}": v for k, v in classic.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'year':<6}{'rows':>7}{'业务子节':>9}{'%':>7}{'字面A':>8}{'%':>7}"
          f"{'业务概要串':>10}{'交叉引用':>9}{'行业经营性':>10}{'核心竞争力':>8}{'核/标题':>8}{'主营业务构成':>9}{'长度中位':>9}")
    for y, d in out["per_year"].items():
        print(f"{y:<6}{d['rows']:>7}{d['business_subsection']:>9}{d['pct_business_subsection']:>6.1f}%"
              f"{d['literal_new_heading']:>8}{d['pct_literal_new_heading']:>6.1f}%"
              f"{d['mentions_business_overview']:>10}{d['xref_business_overview']:>9}"
              f"{d['industry_info_section']:>10}{d['mentions_core_competence']:>8}"
              f"{d['core_competence_heading']:>8}{d['mentions_main_business_composition']:>9}{d['length_median']:>9}")
    print(f"\n{'财年':>5}{'非标题开头':>10}{'A交叉引用':>9}{'A含核心':>8}{'B丢章首':>8}{'C页码页眉':>9}{'D其他':>7}")
    for y, d in out["per_year"].items():
        aq = d["anchor_quality"]
        print(f"{y:>5}{aq['not_at_chapter_title']:>10}{aq['aq_A_xref']:>9}{aq['aq_A_with_core_competence']:>8}"
              f"{aq['aq_B_mid_subsection']:>8}{aq['aq_C_pageno']:>9}{aq['aq_D_other']:>7}")
    print(f"\nJSON → {OUT.relative_to(ROOT)}")


def lexicon_compare(years=("2017", "2018", "2019", "2020")) -> dict:
    """MD&A 章 vs 自建业务描述章：M-A 冻结词库命中数对照（--lexicon，2026-09-23）。

    用途：回答"MD&A 里到底有没有足够的产品词"。匹配器与词库沿用 M-A 冻结实现
    （network_business_lexicon：product_lexicon_v2.csv keep_df25 + Aho-Corasick 全子串、集合语义）。
    """
    import statistics as _st
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from network_business_lexicon import load_lexicon, build_automaton, match_terms_normalized, normalize

    terms, meta = load_lexicon()
    A = build_automaton(terms)
    mda: dict[str, dict[str, str]] = defaultdict(dict)
    with open(SRC, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            y = (r.get("f_year") or "").strip()
            if y in years:
                mda[y][str(r.get("f_stock_id") or "").zfill(6)] = r.get("f_chapter_content") or ""
    bus: dict[str, dict[str, str]] = defaultdict(dict)
    with open(ROOT / "data/processed/business_texts_v7.csv", encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok" and (r.get("year") or "") in years:
                code = "".join(ch for ch in str(r.get("code") or "") if ch.isdigit())[-6:].zfill(6)
                bus[r["year"]][code] = r.get("text") or ""
    out = {"lexicon_terms": meta["terms_kept"], "lexicon_file": meta["path"],
           "business_corpus": "data/processed/business_texts_v7.csv",
           "note": ("原始命中词数受篇幅混淆，不可直接比较；可比口径=密度(每千字命中词)+覆盖率"
                    "(|MD&A∩业务章|/|业务章|)+等长前缀对照。terms 为去重词条数（集合语义，即 M-A 0/1 向量维度）。"),
           "per_year": {}}
    for y in years:
        common = sorted(set(mda[y]) & set(bus[y]))
        rows = []  # (set_mda, set_bus, len_mda, len_bus, set_mda_prefix)
        for code in common:
            tm = mda[y][code]
            tb = bus[y][code]
            sm = match_terms_normalized(normalize(tm), A)
            sb = match_terms_normalized(normalize(tb), A)
            sp = match_terms_normalized(normalize(tm[:len(tb)]), A) if tb else set()
            rows.append((sm, sb, len(tm), len(tb), sp))

        def med(vals):
            vals = [v for v in vals if v is not None]
            return round(float(_st.median(vals)), 4) if vals else None

        out["per_year"][y] = {
            "n_pairs": len(common),
            "mda_median_terms": med([len(r[0]) for r in rows]),
            "business_median_terms": med([len(r[1]) for r in rows]),
            "mda_median_chars": med([r[2] for r in rows]),
            "business_median_chars": med([r[3] for r in rows]),
            "mda_median_terms_per_1k": med([len(r[0]) / r[2] * 1000 if r[2] else None for r in rows]),
            "business_median_terms_per_1k": med([len(r[1]) / r[3] * 1000 if r[3] else None for r in rows]),
            "length_matched_mda_prefix_median_terms": med([len(r[4]) for r in rows]),
            "coverage_median": med([len(r[0] & r[1]) / len(r[1]) if r[1] else None for r in rows]),
            "reverse_coverage_median": med([len(r[0] & r[1]) / len(r[0]) if r[0] else None for r in rows]),
            "jaccard_median": med([len(r[0] & r[1]) / len(r[0] | r[1]) if (r[0] | r[1]) else None for r in rows]),
        }
    return out


if __name__ == "__main__":
    import sys
    if "--lexicon" in sys.argv:
        _lc = lexicon_compare()
        _out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        _out["lexicon_comparison"] = _lc
        OUT.write_text(json.dumps(_out, ensure_ascii=False, indent=1), encoding="utf-8")
        for _y, _d in _lc["per_year"].items():
            print(f"{_y} 对数{_d['n_pairs']:>5} | MD&A 词{_d['mda_median_terms']:>7} 字{_d['mda_median_chars']:>7} "
                  f"密度/千字{_d['mda_median_terms_per_1k']:>7} | 业务章 词{_d['business_median_terms']:>7} "
                  f"字{_d['business_median_chars']:>7} 密度{_d['business_median_terms_per_1k']:>7} | "
                  f"等长前缀{_d['length_matched_mda_prefix_median_terms']:>7} | 覆盖{_d['coverage_median']:>6} "
                  f"反向{_d['reverse_coverage_median']:>6} Jaccard{_d['jaccard_median']:>6}")
    else:
        main()
