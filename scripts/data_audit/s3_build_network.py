#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SUPERSEDED（2026-09-15 审计）：pilot 构网（bigram+Top-200、按整年取 Y-1 版本、权重分母截断）。

**请勿使用**——正式实现与产物：
  - M-B 正式构网 → scripts/data_audit/s3_mb_build_edges.py（--tau 冻结值）→ data/processed/network_mb/
  - M-A 正式构网 → scripts/data_audit/s3_ma_build_edges.py → data/processed/network_ma/
  - S4 面板（正式） → scripts/data_audit/s4_step2_peermax_formal.py → data/processed/monthly_pricing_panel_{ma,mb}.csv
本文件仅作历史留档（pilot 口径），不得作为任何分析输入的来源。

S3-3/4/5：TF-IDF（字符 bigram）+ 余弦相似度 → 按年产品市场网络。

方法（已冻结：A 方案，TF-IDF 余弦；jieba 不可装 → 字符 bigram）：
  1. 业务文本 → 字符 bigram（清洗后相邻字符对）
  2. 文档频率过滤：df ∈ [min_df, max_df_frac×N]
  3. TF-IDF：w = (1+log tf) × log(N/df)
  4. 余弦（倒排索引加速：仅共享 bigram 的文档对）
  5. 网络版本 = 年报年（y），研究期使用 y-1（event-prior）
输出（按年）：
  data/processed/network/{year}_similarity.csv（focal_id, source_id, similarity, network_version）
  reports/data_gates/G0/network_construction_report.json（稠密度分布、权重审计、诊断）
用法: python scripts/data_audit/s3_build_network.py [--year 2017] [--topk 200]
"""
import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUSINESS = ROOT / "data/processed/business_texts.csv"
OUT_DIR = ROOT / "data/processed/network"
OUT_JSON = ROOT / "reports/data_gates/G0/network_construction_report.json"
MIN_DF = 5
MAX_DF_FRAC = 0.60
BIGRAM_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z0-9]")


def bigrams(text):
    """字符 bigram（相邻两个可表意字符）。"""
    chars = [c for c in text if BIGRAM_RE.match(c)]
    return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=0, help="只处理指定年报年（0=全部 2017-2024）")
    ap.add_argument("--topk", type=int, default=200, help="每只股票保留 Top-k 相似边")
    args = ap.parse_args()

    # 1. 读业务文本，按年分组
    docs = defaultdict(dict)  # year -> {code: text}
    with open(BUSINESS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            docs[r["year"]].setdefault(r["code"], r["text"])
    years = [str(args.year)] if args.year else sorted(docs)
    print(f"年报年: {years}")

    summary = {}
    for yr in years:
        t0 = time.time()
        if yr not in docs:
            print(f"{yr}: 无数据"); continue
        year_docs = docs[yr]
        codes = sorted(year_docs)
        N = len(codes)
        if N < 50:
            continue
        # 2. bigram + df
        grams = {}  # code -> Counter(bigram)
        df = Counter()
        for c in codes:
            g = Counter(bigrams(year_docs[c]))
            grams[c] = g
            df.update(g.keys())
        vocab = {g for g, d in df.items() if MIN_DF <= d <= int(N * MAX_DF_FRAC)}
        # 3. TF-IDF 稀疏向量
        vecs = {}
        norms = {}
        inverted = defaultdict(list)  # gram -> [codes]
        for c in codes:
            v = {}
            for g, tf in grams[c].items():
                if g in vocab:
                    w = (1 + math.log(tf)) * math.log(N / df[g])
                    v[g] = w
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            vecs[c] = v
            norms[c] = norm
            for g in v:
                inverted[g].append(c)
        # 4. 余弦（倒排累积）
        dot = defaultdict(float)
        for g, clist in inverted.items():
            if len(clist) < 2:
                continue
            for i in range(len(clist)):
                wi = vecs[clist[i]].get(g, 0.0)
                for j in range(i + 1, len(clist)):
                    dot[(clist[i], clist[j])] += wi * vecs[clist[j]].get(g, 0.0)
        # 5. Top-k 边 + 相似度
        edges = []  # (focal, source, sim)
        for (a, b), d in dot.items():
            sim = d / (norms[a] * norms[b])
            edges.append((a, b, sim))
        # 每只 focal 保留 Top-k
        by_focal = defaultdict(list)
        for a, b, s in edges:
            by_focal[a].append((b, s))
            by_focal[b].append((a, s))
        kept = []
        for c, lst in by_focal.items():
            lst.sort(key=lambda x: -x[1])
            for b, s in lst[: args.topk]:
                kept.append((c, b, s))
        kept.sort()
        # 6. 权重归一化 + 审计
        wsum_err = 0.0
        no_support = 0
        per_focal = defaultdict(list)
        for a, b, s in kept:
            per_focal[a].append((b, s))
        w_rows = []
        for a, lst in per_focal.items():
            total = sum(s for _, s in lst) or 1.0
            if total <= 0 or not lst:
                no_support += 1
                continue
            for b, s in lst:
                w_rows.append((a, b, s, s / total))
        for a, lst in per_focal.items():
            ssum = sum(s / (sum(x for _, x in lst) or 1.0) for _, s in lst) if lst else 0
            wsum_err = max(wsum_err, abs(ssum - 1.0))
        # 写文件
        out = OUT_DIR / f"{yr}_similarity.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["focal_id", "source_id", "similarity", "weight", "network_version"])
            for a, b, s, wgt in w_rows:
                w.writerow([a, b, round(s, 6), round(wgt, 6), yr])
        # 诊断
        sims = [s for _, _, s, _ in w_rows]
        n_edges = len(w_rows)
        summary[yr] = {
            "firms": N, "edges": n_edges, "edges_per_firm": round(n_edges / N, 1),
            "sim_mean": round(sum(sims) / len(sims), 5) if sims else None,
            "sim_p50": round(sorted(sims)[len(sims) // 2], 5) if sims else None,
            "sim_p95": round(sorted(sims)[int(len(sims) * 0.95)], 5) if sims else None,
            "weight_sum_error_max": round(wsum_err, 8),
            "no_support_firms": no_support,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{yr}: N={N} 边={n_edges:,}（{n_edges/N:.1f}/只）sim_p50={summary[yr]['sim_p50']} "
              f"耗时 {summary[yr]['seconds']}s → {out}")

    report = {
        "report_id": "network_construction",
        "method": "A: TF-IDF 字符 bigram 余弦（jieba 不可装→bigram）；df∈[5, 0.6N]；Top-200/只",
        "network_version_semantics": "网络版本=年报年 y；研究期事件使用 y-1 版本（event-prior）",
        "by_year": summary,
        "notes": ["similarity_cutoff 未冻结：pilot 看分布后定（outcome-blind）",
                  "权重 = similarity 归一化（denom=Σsimilarity）；denom=0 → no-network-support 不赋 0"],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT_JSON}")


if __name__ == "__main__":
    main()
