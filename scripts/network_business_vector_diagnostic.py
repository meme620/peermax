"""M-A 向量信息量诊断：同申万二级 vs 跨行业相似度（2026-09-14）。

目的：在冻结聚类参数之前，先确认 0/1 产品词向量是否携带行业信息
（若同业对的相似度显著高于跨行业对，则向量有效，聚类与阈值才有意义）。

产物：reports/data_gates/G0/network_business_vector_diag.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
from network_business_lexicon import build_automaton, load_lexicon, normalize as norm_text  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
CORPUS = ROOT / 'data/processed/business_texts.csv'
IDMAP = ROOT / 'data/processed/firm_id_map.csv'
IND = ROOT / 'data/raw/baostock/industry/all_current.csv'
SW2 = ROOT / 'data/processed/sw2_industry_latest.csv'


def main(year=2021, n_sample=1500, seed=7):
    terms, _ = load_lexicon()
    A = build_automaton(terms)
    d = pd.read_csv(CORPUS, usecols=['code', 'year', 'text', 'status'])
    d = d[(d['status'] == 'ok') & (d['year'] == year)]
    rng = np.random.default_rng(seed)
    d = d.iloc[rng.permutation(len(d))[:n_sample]]

    firms, rows, cols, vocab = [], [], [], {}
    for code, text in zip(d['code'], d['text']):
        hits = {v for _, v in A.iter(norm_text(text))}
        if not hits:
            continue
        ri = len(firms)
        firms.append(code)
        for t in hits:
            ci = vocab.setdefault(t, len(vocab))
            rows.append(ri); cols.append(ci)
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(firms), len(vocab)), dtype=np.float32)
    Xn = normalize(X)
    C = (Xn @ Xn.T).toarray().astype(np.float32)
    np.fill_diagonal(C, np.nan)

    # 行业标签：申万二级
    lat = pd.read_csv(SW2, dtype={'code6': str, 'l2_code': str})
    c2l2 = dict(zip(lat['code6'], lat['l2_code']))
    l2 = np.array([c2l2.get(c[2:], 'NA') for c in firms])
    ind = pd.read_csv(IND, dtype=str)
    ind['code6'] = ind['code'].str.replace('.', '', regex=False).str[2:]
    c2m = dict(zip(ind['code6'], ind['industry'].astype(str).str[:1]))
    menlei = np.array([c2m.get(c[2:], 'NA') for c in firms])

    iu = np.triu_indices(len(firms), k=1)
    sims = C[iu]
    same_sw2 = l2[iu[0]] == l2[iu[1]]
    same_menlei = menlei[iu[0]] == menlei[iu[1]]

    def stat(mask, name):
        s = sims[mask]
        return {'group': name, 'n_pairs': int(mask.sum()),
                'mean': round(float(s.mean()), 5), 'median': round(float(np.median(s)), 5),
                'p90': round(float(np.quantile(s, .9)), 5), 'p99': round(float(np.quantile(s, .99)), 5),
                'share_gt_0.1': round(float((s > 0.1).mean()), 4),
                'share_gt_0.2': round(float((s > 0.2).mean()), 4)}

    out = {
        'run_at': datetime.now(timezone.utc).isoformat(), 'year': year,
        'firms': len(firms), 'active_vocab': len(vocab), 'mean_terms_per_firm': round(X.getnnz(axis=1).mean(), 1),
        'overall': stat(np.ones_like(same_sw2), 'all_pairs'),
        'same_sw2': stat(same_sw2, 'same_sw2_l2'),
        'diff_sw2': stat(~same_sw2, 'diff_sw2_l2'),
        'same_menlei': stat(same_menlei, 'same_csrc_menlei'),
        'diff_menlei': stat(~same_menlei, 'diff_csrc_menlei'),
    }
    out['lift_sw2_mean'] = round(out['same_sw2']['mean'] / out['diff_sw2']['mean'], 3)
    out['lift_sw2_p99'] = round(out['same_sw2']['p99'] / out['diff_sw2']['p99'], 3)
    out['lift_menlei_mean'] = round(out['same_menlei']['mean'] / out['diff_menlei']['mean'], 3)

    # Top 相似对抽样（看是否同业）
    order = np.argsort(-sims)[:20]
    pairs = []
    for k in order:
        i, j = iu[0][k], iu[1][k]
        pairs.append({'a': firms[i], 'b': firms[j], 'sim': round(float(sims[k]), 4),
                      'same_sw2': bool(same_sw2[k]), 'l2_a': l2[i], 'l2_b': l2[j]})
    out['top20_pairs'] = pairs
    out['share_top20_same_sw2'] = round(float(np.mean([p['same_sw2'] for p in pairs])), 3)

    (REPD / 'network_business_vector_diag.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k not in ('top20_pairs',)},
                     ensure_ascii=False, indent=1)[:2000])
    print('\ntop10 pairs:')
    for p in pairs[:10]:
        print('  ', p)


if __name__ == '__main__':
    main(year=int(sys.argv[1]) if len(sys.argv) > 1 else 2021)
