"""Phase B″1：词库 df 过滤在新语料上重算（全子串口径，与匹配器一致）。

背景（execution_plan §8 Phase B″1）：
  `product_lexicon_v2.csv` 的 `doc_count/df_ratio/keep_df25` 是在**业务描述**语料
  （`business_texts.csv`，38,958 篇）上按全子串匹配口径算的。主口径改用老师 MD&A
  导出后必须在新语料上重算，否则 `ma_lexicon_df_max=0.25` 的通用词过滤形同虚设。

口径（冻结，不新设阈值）：
  - 匹配：Aho-Corasick **全子串**，词条集合语义（0/1，重复/嵌套命中只计一次），
    文本与词条同用 `network_business_lexicon.normalize`（NFKC + 去空白 + ASCII 小写）；
  - 逐年（9 个文本年**各自独立**）统计 `doc_count_t`，`df_ratio_t = doc_count_t / N_docs_t`；
  - 阈值：`ma_lexicon_df_max = 0.25`（冻结值，取自 registry）。

单列聚合（本次显式决策，`df_ratio` 列的语义）：
  `load_lexicon(path, df_max)` 在存在 `df_ratio` 列时按**单列**过滤，故须给单列值；
  本脚本取 **`df_ratio = max_t df_ratio_t`（年度上确界，argmax 年的 doc_count 配对）**：
  ① 与"逐年独立"一致且是唯一不需要改匹配器/构网算法的聚合；
  ② 保守方向（只有全年都不超标的词条才保留），不会用某年的低 df 把通用词留在词表里；
  ③ 逐词可复算（`df_ratio * N_docs(argmax 年) = doc_count`）。
  同时落盘 `df_ratio_pooled`（九年池化，对照原业务描述口径的算法）供审计比对。

产物：
  data/processed/product_lexicon_v2_teacher.csv          （原列 + 新 df/keep 列）
  data/processed/product_lexicon_v2_teacher_df_by_year.csv（逐年 doc_count，命中词条）
  reports/data_gates/G0/product_lexicon_df_report_teacher.json

用法：
  python scripts/network_business_lexicon_df_recompute.py \
      --corpus teacher --out data/processed/product_lexicon_v2_teacher.csv \
      --out-report reports/data_gates/G0/product_lexicon_df_report_teacher.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from network_business_lexicon import build_automaton, normalize  # noqa: E402

CORPORA = {
    'teacher': {'path': ROOT / 'data/processed/teacher_mda_panel.csv', 'id_col': 'firm_id'},
    'self': {'path': ROOT / 'data/processed/business_texts.csv', 'id_col': 'code'},
}
LEXICON = ROOT / 'data/processed/product_lexicon_v2.csv'
ORIG_DF_REPORT = ROOT / 'reports/data_gates/G0/product_lexicon_df_report.json'
DF_MAX = 0.25


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def count_docs(corpus: str, terms: list[str], automaton):
    """逐年统计每个词条的命中文档数；返回 (counts[int32 n×ny], years, n_docs, zero_hit)。"""
    spec = CORPORA[corpus]
    years = []
    with open(spec['path'], encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            if r.get('status') == 'ok':
                y = int(r['year'])
                if y not in years:
                    years.append(y)
    years = sorted(years)
    yidx = {y: i for i, y in enumerate(years)}
    counts = np.zeros((len(terms), len(years)), dtype=np.int32)
    tidx = {t: i for i, t in enumerate(terms)}
    n_docs = np.zeros(len(years), dtype=np.int64)
    zero_hit = np.zeros(len(years), dtype=np.int64)
    csv.field_size_limit(10 ** 9)
    with open(spec['path'], encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            if r.get('status') != 'ok':
                continue
            yi = yidx[int(r['year'])]
            n_docs[yi] += 1
            hits = {v for _, v in automaton.iter(normalize(r['text']))}
            if not hits:
                zero_hit[yi] += 1
                continue
            for t in hits:
                counts[tidx[t], yi] += 1
    return counts, years, n_docs, zero_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=['teacher', 'self'], default='teacher')
    ap.add_argument('--lexicon', default=str(LEXICON))
    ap.add_argument('--df-max', type=float, default=DF_MAX)
    ap.add_argument('--out', default=str(ROOT / 'data/processed/product_lexicon_v2_teacher.csv'))
    ap.add_argument('--out-by-year', default=str(ROOT / 'data/processed/product_lexicon_v2_teacher_df_by_year.csv'))
    ap.add_argument('--out-report', default=str(ROOT / 'reports/data_gates/G0/product_lexicon_df_report_teacher.json'))
    args = ap.parse_args()

    t_all = time.time()
    lex = pd.read_csv(args.lexicon, dtype=str)
    lex['term'] = lex['term'].astype(str)
    terms = lex['term'].tolist()
    assert len(terms) == len(set(terms)), '词表有重复词条（会破坏 doc_count 与词表的一一对应）'
    A = build_automaton(terms)
    print(f'词库 {len(terms):,} 条，自动机就绪 [{round(time.time() - t_all, 1)}s]', flush=True)

    counts, years, n_docs, zero_hit = count_docs(args.corpus, terms, A)
    print(f'匹配完成：{int(n_docs.sum()):,} 篇 / {len(years)} 年（{int(zero_hit.sum())} 篇零命中）'
          f' [{round(time.time() - t_all, 1)}s]', flush=True)

    df_t = counts / n_docs[None, :]                     # df_ratio_t
    argmax = df_t.argmax(axis=1)
    df_max_col = df_t.max(axis=1)
    doc_count_max = counts[np.arange(len(terms)), argmax]
    pooled = counts.sum(axis=1)
    df_pooled = pooled / n_docs.sum()
    keep = df_max_col <= args.df_max

    out = lex.copy()
    out = out.rename(columns={'doc_count': 'doc_count_business',
                              'df_ratio': 'df_ratio_business',
                              'keep_df25': 'keep_df25_business'})
    out['doc_count'] = doc_count_max
    out['df_ratio'] = np.round(df_max_col, 6)
    out['keep_df25'] = keep.astype(int)
    out['df_year'] = [years[i] for i in argmax]
    out['doc_count_pooled'] = pooled
    out['df_ratio_pooled'] = np.round(df_pooled, 6)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(outp, index=False, encoding='utf-8')
    print(f'-> {outp} [{round(time.time() - t_all, 1)}s]', flush=True)

    # 逐年 doc_count（命中词条）落盘：供复算/审计
    fired = pooled > 0
    by_year = pd.DataFrame({'term': np.array(terms)[fired]})
    for i, y in enumerate(years):
        by_year[f'doc_count_{y}'] = counts[fired, i]
        by_year[f'n_docs_{y}'] = int(n_docs[i])
    byp = Path(args.out_by_year)
    by_year.to_csv(byp, index=False, encoding='utf-8')

    # ---- 与原口径比对 ----
    orig = pd.read_csv(args.lexicon, dtype={'term': str})
    orig_drop = set(orig.loc[orig['keep_df25'].astype(str) != '1', 'term'])
    new_drop = set(out.loc[out['keep_df25'] == 0, 'term'])
    inter = orig_drop & new_drop
    thr = args.df_max
    # 命中量占比（口径：Σ doc_count 中，被剔词贡献的比例）
    orig_hits = orig['doc_count'].astype(float)
    mass_orig_drop = float(orig_hits[orig['term'].isin(orig_drop)].sum() / max(1.0, orig_hits.sum()))
    new_hits = out['doc_count_pooled'].astype(float)
    mass_new_drop = float(new_hits[out['term'].isin(new_drop)].sum() / max(1.0, new_hits.sum()))
    nzero_ratio = float(zero_hit.sum() / max(1, n_docs.sum()))
    cmp_block = {
        'criterion_2_hit_mass_share': {
            'rule': '被剔词贡献的词命中量占比（Σdoc_count 口径）',
            'original_business_corpus_dropped_share': round(mass_orig_drop, 6),
            'teacher_corpus_dropped_share': round(mass_new_drop, 6),
            'abs_diff_pp': round(abs(mass_new_drop - mass_orig_drop) * 100, 4),
            'threshold_pp': 5.0,
            'pass': bool(abs(mass_new_drop - mass_orig_drop) * 100 <= 5.0),
        },
        'criterion_3_dropped_list_overlap': {
            'rule': '被剔词清单重叠（teacher vs 原业务描述口径）',
            'n_orig_dropped': len(orig_drop), 'n_teacher_dropped': len(new_drop),
            'n_intersection': len(inter),
            'overlap_teacher_in_orig': round(len(inter) / max(1, len(new_drop)), 4),
            'overlap_orig_in_teacher': round(len(inter) / max(1, len(orig_drop)), 4),
            'jaccard': round(len(inter) / max(1, len(orig_drop | new_drop)), 4),
            'threshold': 0.90,
            'pass': bool(len(inter) / max(1, len(orig_drop | new_drop)) >= 0.90),
        },
    }
    if not (cmp_block['criterion_3_dropped_list_overlap']['pass']):
        cmp_block['criterion_3_dropped_list_overlap']['manual_signoff_required'] = True

    # ---- 被剔词逐条记录（向量化预取：原实现把 pd.to_numeric 放进列表推导里，
    # 每个被剔词重算一次全表 1.35M 行，1,758 词 ⇒ ~1.5 h 静默计算；此处预取一次，
    # 数值与排序完全不变）----
    biz_df = pd.to_numeric(lex['df_ratio'], errors='coerce').to_numpy(dtype=float)
    biz_df = np.where(np.isnan(biz_df), 0.0, biz_df)
    sogou_col = out['in_sogou'].to_numpy(dtype=object) if 'in_sogou' in out.columns else None
    em_col = out['in_em'].to_numpy(dtype=object) if 'in_em' in out.columns else None
    dropped_idx = np.flatnonzero(~keep)
    drec = {int(i): {'term': terms[i], 'df_ratio': round(float(df_max_col[i]), 6),
                     'doc_count': int(doc_count_max[i]), 'df_year': int(years[argmax[i]]),
                     'in_sogou': str(sogou_col[i]) if sogou_col is not None else '',
                     'in_em': str(em_col[i]) if em_col is not None else '',
                     'df_ratio_business': round(float(biz_df[i]), 6),
                     'per_year_df': {str(y): round(float(df_t[i, j]), 6) for j, y in enumerate(years)}}
             for i in dropped_idx}
    top_dropped = [drec[i] for i in sorted(dropped_idx.tolist(), key=lambda i: -df_max_col[i])[:50]]

    rep = {
        'step': 'Phase B″1 词库 df 重算',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'corpus': args.corpus,
        'corpus_path': str(CORPORA[args.corpus]['path'].relative_to(ROOT)).replace('\\', '/'),
        'lexicon_path': str(Path(args.lexicon).relative_to(ROOT)).replace('\\', '/'),
        'df_max': args.df_max,
        'aggregation_rule': ('df_ratio = max_t df_ratio_t（逐年独立的上确界；doc_count 取 argmax 年计数，'
                             '故 df_ratio = doc_count / N_docs(df_year)）；对照列 df_ratio_pooled = Σdoc_count / ΣN_docs'),
        'match_rule': 'full_substring (Aho-Corasick, set semantics), NFKC+strip_ws+lower',
        'per_year': [{'text_year': y, 'n_docs': int(n_docs[i]), 'zero_hit_docs': int(zero_hit[i]),
                      'zero_hit_ratio': round(float(zero_hit[i] / max(1, n_docs[i])), 6),
                      'n_terms_df_gt_max': int((df_t[:, i] > args.df_max).sum()),
                      'n_terms_fired': int((counts[:, i] > 0).sum())}
                     for i, y in enumerate(years)],
        'n_docs_total': int(n_docs.sum()),
        'terms_total_in_file': len(terms),
        'terms_kept': int(keep.sum()),
        'terms_dropped': int((~keep).sum()),
        'terms_fired_at_least_once': int((pooled > 0).sum()),
        'docs_with_any_hit': int(n_docs.sum() - zero_hit.sum()),
        'docs_with_any_hit_ratio': round(1 - nzero_ratio, 6),
        'zero_hit_docs': int(zero_hit.sum()),
        'zero_hit_ratio': round(nzero_ratio, 6),
        'stop_condition': {'rule': '零命中文档占比 > 0.1% 停机',
                           'value': round(nzero_ratio, 6), 'pass': bool(nzero_ratio <= 0.001)},
        'top_dropped_by_df': [
            {'term': t, 'df_ratio': round(float(r), 6), 'doc_count': int(c), 'df_year': int(yy),
             'in_sogou': str(sg), 'in_em': str(em), 'df_ratio_business': round(float(dbf), 6)}
            for t, r, c, yy, sg, em, dbf in sorted(
                [(t, df_max_col[i], doc_count_max[i], years[argmax[i]],
                  out['in_sogou'].iloc[i] if 'in_sogou' in out.columns else '',
                  out['in_em'].iloc[i] if 'in_em' in out.columns else '',
                  float(pd.to_numeric(lex['df_ratio'], errors='coerce').iloc[i] or 0))
                 for i, t in enumerate(terms) if not keep[i]],
                key=lambda x: -x[1])[:50]],
        'dropped_terms_detail': [
            {'term': t, 'df_ratio': round(float(df_max_col[i]), 6), 'df_year': int(years[argmax[i]]),
             'df_ratio_business': round(float(pd.to_numeric(lex['df_ratio'], errors='coerce').iloc[i] or 0), 6),
             'per_year_df': {str(y): round(float(df_t[i, j]), 6) for j, y in enumerate(years)}}
            for i, t in enumerate(terms) if not keep[i]],
        'dropped_only_in_original': sorted(orig_drop - new_drop),
        'dropped_only_in_teacher': sorted(new_drop - orig_drop),
        'comparison_with_original': cmp_block,
        'outputs': {
            'lexicon': str(outp.relative_to(ROOT)).replace('\\', '/'),
            'lexicon_sha256': sha256(outp),
            'by_year': str(byp.relative_to(ROOT)).replace('\\', '/'),
            'by_year_sha256': sha256(byp),
            'original_df_report': str(ORIG_DF_REPORT.relative_to(ROOT)).replace('\\', '/'),
        },
        'elapsed_sec': round(time.time() - t_all, 1),
    }
    rep['status'] = ('PASS' if (rep['stop_condition']['pass']
                                and cmp_block['criterion_2_hit_mass_share']['pass']) else 'FAIL')
    rp = Path(args.out_report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: rep[k] for k in ('terms_total_in_file', 'terms_kept', 'terms_dropped',
                                          'docs_with_any_hit_ratio', 'zero_hit_ratio')}, ensure_ascii=False))
    print(json.dumps(cmp_block, ensure_ascii=False, indent=1))
    print('status =', rep['status'], '->', rp)


if __name__ == '__main__':
    main()
