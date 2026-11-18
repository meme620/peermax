#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F2：新语料（年报全文）上重算词库 df（与 Phase B″1 同口径）。

口径（与 `network_business_lexicon_df_recompute.py` 逐条一致，**不新设阈值**）：
  - 匹配：Aho-Corasick **全子串**、词条集合语义（0/1；重复/嵌套命中只计一次），
    文本与词条同用 `network_business_lexicon.normalize`（NFKC + 去空白 + ASCII 小写）；
    命中结果由 F1 的命中缓存提供（`annual_fulltext_hits/{year}_hits.npz`，词表 = product_lexicon_v2.csv
    全表 1,355,237 条，行号即词条序号）——等价于逐篇现场匹配，见 F3 前的缓存核验证据；
  - 逐年（各文本年**独立**）统计 `doc_count_t = 命中该词的 ok 文档数`，`df_ratio_t = doc_count_t / N_docs_t`；
  - 阈值：`ma_lexicon_df_max = 0.25`（沿用主口径冻结值；**不重标定**）；
  - 单列聚合：`df_ratio = max_t df_ratio_t`（年度上确界，doc_count 取 argmax 年计数），
    对照列 `df_ratio_pooled = Σdoc_count / ΣN_docs`（与主口径同式）。

产物：
  data/processed/product_lexicon_v2_fulltext.csv            （原列 + 新 df/keep 列）
  data/processed/product_lexicon_v2_fulltext_df_by_year.csv （逐年 doc_count，命中词条）
  reports/data_gates/G0/product_lexicon_df_report_fulltext.json

用法：
  python scripts/network_business_fulltext_lexicon_df.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'data/processed/annual_fulltext_panel.parquet'
HITS_DIR = ROOT / 'data/processed/annual_fulltext_hits'
LEXICON = ROOT / 'data/processed/product_lexicon_v2.csv'
OUT = ROOT / 'data/processed/product_lexicon_v2_fulltext.csv'
OUT_BY_YEAR = ROOT / 'data/processed/product_lexicon_v2_fulltext_df_by_year.csv'
REPORT = ROOT / 'reports/data_gates/G0/product_lexicon_df_report_fulltext.json'
PROGRESS = ROOT / 'artifacts/tmp/fulltext_corr_progress.md'
TEACHER_LEX = ROOT / 'data/processed/product_lexicon_v2_teacher.csv'
DF_MAX = 0.25


def psay(msg: str) -> None:
    line = f'[FT-2] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')


def sha256(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', default=str(PANEL))
    ap.add_argument('--hits-dir', default=str(HITS_DIR))
    ap.add_argument('--lexicon', default=str(LEXICON))
    ap.add_argument('--df-max', type=float, default=DF_MAX)
    ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--out-by-year', default=str(OUT_BY_YEAR))
    ap.add_argument('--out-report', default=str(REPORT))
    args = ap.parse_args()

    t_all = time.time()
    lex = pd.read_csv(args.lexicon, dtype=str)
    terms = [str(t) for t in lex['term']]
    assert len(terms) == len(set(terms)), '词表有重复词条'
    n_terms = len(terms)
    hitsd = Path(args.hits_dir)
    pan = pd.read_parquet(args.panel, columns=['firm_id', 'year', 'status', 'n_hits'])
    lex_sha = sha256(Path(args.lexicon))
    z0 = np.load(hitsd / f'{int(pan["year"].min())}_hits.npz')
    assert str(z0['lex_sha256']) == lex_sha, '命中缓存词表 sha 与本次词表不符'
    assert int(z0['lex_rows']) == n_terms
    psay(f'状态 = 开始；语料 = {Path(args.panel).name}（{len(pan):,} 行 / {pan["year"].nunique()} 年）；'
         f'词表 = {n_terms:,} 条（sha256 {lex_sha[:12]}…）；df 阈值 = {args.df_max}（沿用，不重标定）')

    years = sorted(int(y) for y in pan['year'].unique())
    counts = np.zeros((n_terms, len(years)), dtype=np.int64)
    n_docs = np.zeros(len(years), dtype=np.int64)
    zero_hit = np.zeros(len(years), dtype=np.int64)
    for i, y in enumerate(years):
        sub = pan[pan['year'] == y]
        z = np.load(hitsd / f'{y}_hits.npz')
        assert len(z['firm_id']) == len(sub) and bool((z['firm_id'] == sub['firm_id'].to_numpy()).all()), \
            f'{y}: 命中缓存与面板行序不一致'
        ok = (sub['status'] == 'ok').to_numpy()
        n_docs[i] = int(ok.sum())
        idx = z['hits_idx']
        off = z['offsets']
        # 仅 ok 文档计入（非 ok 行文本已置空 → 其命中数组为空，此处显式按行过滤）
        keep = np.repeat(ok, np.diff(off))
        counts[:, i] = np.bincount(idx[keep], minlength=n_terms)
        zero_hit[i] = int((sub['n_hits'].to_numpy()[ok] == 0).sum())
        psay(f'[{y}] df 计数状态 = ok：ok 文档 {int(n_docs[i]):,} / 命中词条 {int((counts[:, i] > 0).sum()):,} / '
             f'零命中文档 {int(zero_hit[i])}')
    assert (n_docs > 0).all(), '存在 ok 文档数为 0 的年份'

    df_t = counts / n_docs[None, :]
    argmax = df_t.argmax(axis=1)
    df_max_col = df_t.max(axis=1)
    doc_count_max = counts[np.arange(n_terms), argmax]
    pooled = counts.sum(axis=1)
    df_pooled = pooled / n_docs.sum()
    keep = df_max_col <= args.df_max

    out = lex.copy()
    out = out.rename(columns={'doc_count': 'doc_count_business', 'df_ratio': 'df_ratio_business',
                              'keep_df25': 'keep_df25_business'})
    out['doc_count'] = doc_count_max
    out['df_ratio'] = np.round(df_max_col, 6)
    out['keep_df25'] = keep.astype(int)
    out['df_year'] = [years[i] for i in argmax]
    out['doc_count_pooled'] = pooled
    out['df_ratio_pooled'] = np.round(df_pooled, 6)
    outp = Path(args.out)
    out.to_csv(outp, index=False, encoding='utf-8')
    psay(f'词表落盘状态 = ok（{outp.name}，{outp.stat().st_size / 2**20:.1f} MiB）；'
         f'保留 {int(keep.sum()):,} / 剔除 {int((~keep).sum()):,}（df > {args.df_max}）')

    fired = pooled > 0
    by_year = pd.DataFrame({'term': np.array(terms)[fired]})
    for i, y in enumerate(years):
        by_year[f'doc_count_{y}'] = counts[fired, i]
        by_year[f'n_docs_{y}'] = int(n_docs[i])
    byp = Path(args.out_by_year)
    by_year.to_csv(byp, index=False, encoding='utf-8')

    # ---- 与既有两语料口径对照（描述性；阈值来自 Phase B″1，本轮不作停机线）----
    def cmp_block(other_path, tag):
        if not Path(other_path).exists():
            return {'available': False, 'path': str(other_path)}
        o = pd.read_csv(other_path, dtype={'term': str})
        o_drop = set(o.loc[o['keep_df25'].astype(str) != '1', 'term'])
        n_drop = set(np.array(terms)[~keep])
        inter = o_drop & n_drop
        oh = pd.to_numeric(o['doc_count'], errors='coerce').fillna(0)
        mass_o = float(oh[o['term'].isin(o_drop)].sum() / max(1.0, oh.sum()))
        mass_n = float(pooled[np.isin(np.array(terms), list(n_drop))].sum() / max(1.0, pooled.sum()))
        return {'available': True, 'path': str(Path(other_path).name),
                'n_dropped_other': len(o_drop), 'n_dropped_fulltext': len(n_drop),
                'n_intersection': len(inter),
                'jaccard': round(len(inter) / max(1, len(o_drop | n_drop)), 4),
                'hit_mass_share_other': round(mass_o, 6), 'hit_mass_share_fulltext': round(mass_n, 6),
                'abs_diff_pp': round(abs(mass_n - mass_o) * 100, 4),
                'threshold_role': 'descriptive（Phase B″1 的 5pp/0.90 阈值口径仅作对照，本轮不作停机线）'}

    top_dropped_idx = np.argsort(-df_max_col)[:50]
    rep = {
        'step': 'F2 全文语料（annual_fulltext）词库 df 重算',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'corpus': 'ma_fulltext（自建年报全文面板）',
        'corpus_path': str(Path(args.panel).relative_to(ROOT)).replace('\\', '/'),
        'hits_dir': str(hitsd.relative_to(ROOT)).replace('\\', '/'),
        'lexicon_path': str(Path(args.lexicon).relative_to(ROOT)).replace('\\', '/'),
        'df_max': args.df_max,
        'same_rule_as': 'network_business_lexicon_df_recompute.py（Phase B″1，语料 = 老师 MD&A）',
        'aggregation_rule': ('df_ratio = max_t df_ratio_t（逐年独立上确界；doc_count 取 argmax 年计数）；'
                             '对照列 df_ratio_pooled = Σdoc_count / ΣN_docs'),
        'match_rule': 'full_substring (Aho-Corasick, set semantics), NFKC+strip_ws+lower；'
                      '命中由 F1 缓存提供（缓存核验证据见 s3_ma_fulltext_cache_verification.json）',
        'per_year': [{'text_year': y, 'n_docs': int(n_docs[i]), 'zero_hit_docs': int(zero_hit[i]),
                      'zero_hit_ratio': round(float(zero_hit[i] / max(1, n_docs[i])), 6),
                      'n_terms_df_gt_max': int((df_t[:, i] > args.df_max).sum()),
                      'n_terms_fired': int((counts[:, i] > 0).sum())}
                     for i, y in enumerate(years)],
        'n_docs_total': int(n_docs.sum()),
        'terms_total_in_file': n_terms,
        'terms_kept': int(keep.sum()), 'terms_dropped': int((~keep).sum()),
        'terms_fired_at_least_once': int(fired.sum()),
        'zero_hit_docs': int(zero_hit.sum()),
        'zero_hit_ratio': round(float(zero_hit.sum() / max(1, n_docs.sum())), 6),
        'stop_condition': {'rule': '零命中文档占比 > 0.1% 停机（沿用 Phase B″1 停机线）',
                           'value': round(float(zero_hit.sum() / max(1, n_docs.sum())), 6),
                           'pass': bool(zero_hit.sum() / max(1, n_docs.sum()) <= 0.001)},
        'top_dropped_by_fulltext_df': [
            {'term': terms[i], 'df_ratio': round(float(df_max_col[i]), 6), 'doc_count': int(doc_count_max[i]),
             'df_year': int(years[argmax[i]]),
             'df_ratio_business': round(float(pd.to_numeric(lex['df_ratio'], errors='coerce').fillna(0).iloc[i] or 0), 6)}
            for i in top_dropped_idx if not keep[i]],
        'comparison_vs_business_corpus': cmp_block(args.lexicon, 'business'),
        'comparison_vs_teacher_corpus': cmp_block(TEACHER_LEX, 'teacher'),
        'outputs': {'lexicon': str(outp.relative_to(ROOT)).replace('\\', '/'), 'lexicon_sha256': sha256(outp),
                    'by_year': str(byp.relative_to(ROOT)).replace('\\', '/'), 'by_year_sha256': sha256(byp)},
        'elapsed_sec': round(time.time() - t_all, 1),
    }
    rep['status'] = 'PASS' if rep['stop_condition']['pass'] else 'FAIL'
    rp = Path(args.out_report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    psay(f'状态 = {rep["status"]}：词条 {n_terms:,} → 保留 {rep["terms_kept"]:,} / 剔除 {rep["terms_dropped"]:,}；'
         f'零命中占比 {rep["zero_hit_ratio"]}（停机线 0.1%）；报告 {rp.name}')
    print(json.dumps({k: rep[k] for k in ('terms_total_in_file', 'terms_kept', 'terms_dropped',
                                          'zero_hit_ratio', 'status')}, ensure_ascii=False))
    return 0 if rep['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
