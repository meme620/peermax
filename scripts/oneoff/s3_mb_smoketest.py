"""S3-M-B 可用性冒烟验证（2026-09-12）。

目的：用 mda_texts_v2.csv 实跑执行计划 §S3 M-B 链（第 214/231 行）：
  MD&A → 分词 + 停用词 + df>25% 通用词删除 → TF-IDF → 余弦相似度
  → 固定阈值 τ（本脚本用"中位 peer 数=10"作占位目标；正式校准需申万二级行业数据，
     属 M-B 待冻结参数 sw2_median_peer_target）→ PeerCount / TotalSimilarity

回答的问题：数据能否支撑网络构造？有没有公司因词表为空/被过滤成为孤立节点？
相似度分布是否合理？同行业公司是否互相识别？
输出：reports/data_gates/G0/mb_smoketest.json（含统计与样例）
用法: python scripts/data_audit/s3_mb_smoketest.py [--text-year 2021] [--sample 400]
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/processed/mda_texts_v2.csv"
OUT = ROOT / "reports/data_gates/G0/mb_smoketest.json"
DF_CUTOFF = 0.25
PEER_TARGET = 10          # 占位目标（正式：申万二级中位 peer 数）

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-year", type=int, default=2021)
    ap.add_argument("--sample", type=int, default=400, help="相似度抽样的 focal 数（0=全量）")
    args = ap.parse_args()
    import jieba
    jieba.setLogLevel(60)

    rows = load_texts(args.text_year)
    print(f"文本年 {args.text_year}: {len(rows)} 篇 ok", flush=True)
    codes = [c for c, _ in rows]
    toks = [tokenize(t, jieba) for _, t in rows]
    lens = [len(x) for x in toks]
    empty_raw = sum(1 for n in lens if n == 0)

    # df 过滤（> 25% 文档出现 = boilerplate）
    df = Counter()
    for ts in toks:
        df.update(set(ts))
    n_doc = len(toks)
    kept = {w for w, c in df.items() if c / n_doc <= DF_CUTOFF}
    toks2 = [[w for w in ts if w in kept] for ts in toks]
    empty_after = sum(1 for ts in toks2 if not ts)
    vocab = len(kept)

    # TF-IDF + 余弦（sklearn）
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer=lambda x: x, token_pattern=None, min_df=1)
    X = vec.fit_transform(toks2)
    from sklearn.preprocessing import normalize
    Xn = normalize(X)
    # 分块求相似度统计（避免 N^2 内存）
    idx = np.arange(len(codes))
    if args.sample and args.sample < len(codes):
        rng = np.random.default_rng(20260912)
        focal = rng.choice(idx, args.sample, replace=False)
    else:
        focal = idx
    sims_all = []
    sim_by_focal = {}
    B = 128
    for s in range(0, len(focal), B):
        f = focal[s:s + B]
        S = (Xn[f] @ Xn.T).toarray()
        for k, i in enumerate(f):
            row = S[k].copy()
            row[idx == i] = -1.0     # 排除自身
            sims_all.append(row)
        sim_by_focal.update({int(i): S[k] for k, i in enumerate(f)})

    sim = np.concatenate(sims_all)
    pos = sim[sim > 0]
    stats = {
        "text_year": args.text_year,
        "n_firms": len(codes),
        "tokens_per_doc_median": int(np.median(lens)),
        "tokens_per_doc_p10": int(np.percentile(lens, 10)),
        "empty_token_docs_raw": empty_raw,
        "vocab_after_df_filter": vocab,
        "empty_docs_after_df_filter": empty_after,
        "sim_nonzero_share": float((sim > 0).mean()),
        "sim_median_of_positive": float(np.median(pos)) if len(pos) else 0.0,
        "sim_p90_of_positive": float(np.percentile(pos, 90)) if len(pos) else 0.0,
        "sim_p99_of_positive": float(np.percentile(pos, 99)) if len(pos) else 0.0,
    }
    # τ 校准：使中位 peer 数 ≈ PEER_TARGET
    taus = np.quantile(pos, 1 - min(1.0, PEER_TARGET / max(1, len(codes) - 1)), ) if len(pos) else 1.0
    peers = []
    for i, row in zip(focal, sims_all):
        peers.append(int((row > taus).sum()))
    stats["tau_placeholder"] = float(taus)
    stats["peers_median_at_tau"] = int(np.median(peers))
    stats["peers_p90_at_tau"] = int(np.percentile(peers, 90))
    stats["isolated_firms_at_tau"] = int(sum(1 for p in peers if p == 0))

    # 样例：3 家知名公司的 top-5 相似 peer
    samples = {}
    code2idx = {c: i for i, c in enumerate(codes)}
    for c in ["sh600519", "sz000858", "sh600036", "sz000651", "sz300750"]:
        if c not in code2idx:
            continue
        i = code2idx[c]
        if i not in sim_by_focal:
            S = (Xn[i] @ Xn.T).toarray().ravel()
            S[idx == i] = -1
        else:
            S = sim_by_focal[i]
        top = np.argsort(-S)[:5]
        samples[c] = [[codes[j], round(float(S[j]), 4)] for j in top]
    stats["samples_top5"] = samples

    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
