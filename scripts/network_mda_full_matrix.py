"""network-M-B 全期相似度矩阵（2026-09-13）。

在已冻结参数下（df_threshold=0.25、mda_text_extraction_rule=v2）对文本年 2017-2025
（对应网络年 2018-2025）逐年计算 TF-IDF 余弦相似度分布，**不校准 τ**
（τ 校准依赖申万二级行业数据，属待办项）。

产出（每年一行统计）：
  reports/data_gates/G0/network_mda_full_matrix.json

用法: python scripts/network_mda_full_matrix.py [--years 2017-2025] [--sample 600]
"""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fileio import file_size

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data/processed/mda_texts.csv"
OUT = ROOT / "reports/data_gates/G0/network_mda_full_matrix.json"
SPEC = ROOT / "configs/analyst_spec.yaml"
DF_CUTOFF = 0.25          # registry: df_threshold（2026-09-13 冻结）
PEER_TARGET = 10          # 占位目标（正式 τ 校准待申万二级）

STOPWORDS = set("""的 了 和 与 及 或 在 是 为 对 到 由 于 以 并 等 中 上 下 内 外 该 本 各 其 之
有 无 不 也 都 会 将 已 未 可 能 应 需 要 且 而 但 则 若 如 因 此 等 上述 其中 报告期内 报告期 公司 本公司年度 年度
期间 情况 影响 主要 相关 同比 上年 同期 增加 减少 增长 下降 实现 完成 达到 保持 进一步 持续 不断 继续 推动
加强 提升 优化 完善 推进 落实 开展 进行 通过 按照 根据 由于 以及 同时 此外 另外 目前 未来 下一步 后续 总体
亿元 万元 元 个百分点 家 户 人 项 个 名 只 台 辆 吨 平方米 万千瓦 万 亿 千 百 十""".split())
TOKEN_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{0,}$")


def load_texts(text_year):
    rows = []
    csv.field_size_limit(10 ** 9)
    with open(V2, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and r["year"] == str(text_year):
                rows.append((r["code"], r["text"]))
    return rows


def tokenize(text, jieba):
    toks = []
    for w in jieba.cut(text):
        w = w.strip()
        if len(w) < 2 and not re.match(r"^[A-Za-z]{2,}$", w):
            continue
        if not TOKEN_RE.match(w):
            continue
        if w in STOPWORDS:
            continue
        toks.append(w)
    return toks


def year_metrics(text_year, sample):
    import jieba
    jieba.setLogLevel(60)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    rows = load_texts(text_year)
    if not rows:
        return {"text_year": text_year, "error": "no_ok_rows"}
    codes = [c for c, _ in rows]
    toks = [tokenize(t, jieba) for _, t in rows]
    lens = [len(x) for x in toks]

    df = Counter()
    for ts in toks:
        df.update(set(ts))
    n_doc = len(toks)
    kept = {w for w, c in df.items() if c / n_doc <= DF_CUTOFF}
    toks2 = [[w for w in ts if w in kept] for ts in toks]

    vec = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None, min_df=1)
    X = vec.fit_transform(toks2)
    Xn = normalize(X)

    idx = np.arange(len(codes))
    if sample and sample < len(codes):
        rng = np.random.default_rng(20260913)
        focal = rng.choice(idx, sample, replace=False)
    else:
        focal = idx
    sims, peers_at_tau = [], []
    B = 128
    rows_all = []
    for s in range(0, len(focal), B):
        f = focal[s:s + B]
        S = (Xn[f] @ Xn.T).toarray()
        for k, i in enumerate(f):
            row = S[k].copy()
            row[idx == i] = -1.0
            rows_all.append(row)
    sim = np.concatenate(rows_all)
    pos = sim[sim > 0]
    tau = np.quantile(pos, 1 - min(1.0, PEER_TARGET / max(1, len(codes) - 1))) if len(pos) else 1.0
    peers = [int((r > tau).sum()) for r in rows_all]

    return {
        "text_year": text_year,
        "network_year": text_year + 1,      # 文本年 y-1 -> 网络年 y
        "n_firms": len(codes),
        "tokens_per_doc_median": int(np.median(lens)),
        "tokens_per_doc_p10": int(np.percentile(lens, 10)),
        "empty_token_docs_raw": int(sum(1 for n in lens if n == 0)),
        "vocab_after_df_filter": len(kept),
        "empty_docs_after_df_filter": int(sum(1 for ts in toks2 if not ts)),
        "sim_nonzero_share": float((sim > 0).mean()),
        "sim_median_of_positive": float(np.median(pos)) if len(pos) else 0.0,
        "sim_p90_of_positive": float(np.percentile(pos, 90)) if len(pos) else 0.0,
        "sim_p99_of_positive": float(np.percentile(pos, 99)) if len(pos) else 0.0,
        "tau_placeholder": float(tau),
        "peers_median_at_tau": int(np.median(peers)),
        "peers_p90_at_tau": int(np.percentile(peers, 90)),
        "isolated_firms_at_tau": int(sum(1 for p in peers if p == 0)),
        "sampled_focals": len(focal),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2017-2025")
    ap.add_argument("--sample", type=int, default=600)
    args = ap.parse_args()
    y0, y1 = [int(x) for x in args.years.split("-")]

    spec_hash = file_size(SPEC) if SPEC.exists() else None
    out = {"spec_id": "analyst_spec_20260909", "spec_hash": spec_hash,
           "run_at": datetime.now(timezone.utc).isoformat(),
           "df_threshold": DF_CUTOFF, "peer_target_placeholder": PEER_TARGET,
           "note": ("M-B 全期相似度矩阵；τ 为占位（中位 peer≈10），正式校准待申万二级数据到位；"
                    "文本年 y-1 → 网络年 y，故 2017-2025 文本年对应 2018-2026 网络年（2026 年无收益数据，仅作分布）"),
           "years": []}
    for y in range(y0, y1 + 1):
        m = year_metrics(y, args.sample)
        out["years"].append(m)
        print(json.dumps(m, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
