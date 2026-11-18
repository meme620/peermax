"""M-A 第二种解读探索：对手集合 = 同团成员（cluster-only）（2026-09-14）。

背景：按 plan 字面"团内最小相似度作阈值"，在该数据上阈值为~0 → 对手集合=全市场
（K≤400 时中位 peer 3,500/3,544），网络退化。第二种解读（更可能是原文设计）：
**同团成员即对手集合**，K 直接控制粒度。本脚本评估该解读下的网络指标，
并找出使公司层面中位 peer 数≈SW2 目标(56) 的 K，供参数冻结决策。

产物：reports/data_gates/G0/s3_ma_K_explore2.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
from network_business_cluster_grid import build_vectors, load_refs  # noqa: E402
from network_business_lexicon import build_automaton, load_lexicon  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
KS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]


def eval_cluster_only(labels, l2, base_rate):
    n = len(labels)
    ks, inv = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inv)
    peers = sizes[inv] - 1
    M = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), inv)), shape=(n, len(ks)))
    same = (l2[:, None] == l2[None, :])
    # 同团且同 SW2 的对数
    same_l2_int = np.zeros(n, dtype=np.float32)
    for k in ks:  # 每个团内同行业对数
        pass
    # 向量化：按团统计同行业对数
    df = pd.DataFrame({'c': inv, 's': np.arange(n)})
    pair_same = 0
    tot_pairs = 0
    from collections import defaultdict
    byc = defaultdict(list)
    for c, s in zip(inv, l2):
        byc[c].append(s)
    for c, arr in byc.items():
        m = len(arr)
        tot_pairs += m * (m - 1) // 2
        if m > 1:
            vc = pd.Series(arr).value_counts()
            pair_same += int((vc * (vc - 1) // 2).sum())
    return {
        'n_clusters': int(len(ks)),
        'size_median': float(np.median(sizes)), 'size_max': int(sizes.max()),
        'cluster_size_p90': float(np.quantile(sizes, .9)),
        'peer_median': float(np.median(peers)), 'peer_mean': round(float(peers.mean()), 1),
        'peer_p90': float(np.quantile(peers, .9)),
        'isolated_share': round(float((peers == 0).mean()), 4),
        'edges_per_firm_mean': round(float(peers.mean()), 2),
        'same_sw2_edge_share': round(pair_same / max(1, tot_pairs), 4),
        'sw2_lift': round((pair_same / max(1, tot_pairs)) / max(1e-9, base_rate), 2),
    }


def main():
    year = 2021
    terms, _ = load_lexicon()
    A = build_automaton(terms)
    uni = load_refs()
    firms, X, stats = build_vectors(year, A, uni)
    Xn = normalize(X)
    C = (Xn @ Xn.T).toarray().astype(np.float32)
    np.fill_diagonal(C, 1.0)
    D = np.clip(1.0 - C, 0, 2); np.fill_diagonal(D, 0.0)
    l2 = np.array([uni.loc[f, 'l2_code'] if (f in uni.index and isinstance(uni.loc[f, 'l2_code'], str)) else 'NA'
                   for f in firms])
    iu = np.triu_indices(len(firms), k=1)
    base = float((l2[iu[0]] == l2[iu[1]]).mean())
    cond = squareform(D, checks=False)
    out = {'year': year, 'n_firms': len(firms), 'base_same_sw2_rate': round(base, 5),
           'filter_stats': stats, 'interpretation': 'competitors = same-cluster members', 'results': {}}
    for lk in ['average', 'complete']:
        Z = linkage(cond, method=lk)
        out['results'][lk] = {}
        for K in KS:
            lab = fcluster(Z, K, criterion='maxclust')
            r = eval_cluster_only(lab, l2, base)
            out['results'][lk][str(K)] = r
            print(f"  [{lk}] K={K:>5} size_med={r['size_median']:.0f} peer_med={r['peer_median']:.0f} "
                  f"peer_p90={r['peer_p90']:.0f} iso={r['isolated_share']:.3f} lift={r['sw2_lift']}", flush=True)
    (REPD / 's3_ma_K_explore2.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('->', REPD / 's3_ma_K_explore2.json')


if __name__ == '__main__':
    main()
