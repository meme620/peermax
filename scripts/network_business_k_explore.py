"""M-A 阈值有效性探索：K 与对手集合规模的关系（2026-09-14）。

问题：K 较小时团内最小相似度≈0 → 对手集合退化为全市场（实测 K<=30 时中位 peer=3,503/3,544）。
本脚本在指定 K 网格上直接评估，找出使对手集合规模进入合理区间（数十个）的 K 范围。

产物：reports/data_gates/G0/network_business_k_explore.json
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
from network_business_cluster_grid import build_vectors, load_refs  # noqa: E402
from network_business_lexicon import build_automaton, load_lexicon  # noqa: E402

REPD = ROOT / 'reports/data_gates/G0'
KS = [30, 60, 100, 150, 200, 300, 400, 600, 800, 1200]


def eval_at(C, labels, l2, base_rate):
    n = C.shape[0]
    thr = np.empty(n, dtype=np.float32)
    sizes = []
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        sizes.append(len(idx))
        if len(idx) > 1:
            sub = C[np.ix_(idx, idx)].copy()
            np.fill_diagonal(sub, 2.0)
            thr[idx] = float(sub.min())
        else:
            thr[idx] = 1.0
    mask = C > thr[:, None]
    np.fill_diagonal(mask, False)
    peers = mask.sum(axis=1)
    same = (l2[:, None] == l2[None, :])
    tot = int(mask.sum()); sm = int((mask & same).sum())
    sizes = np.array(sizes)
    return {'n_clusters': int(len(sizes)), 'size_median': float(np.median(sizes)),
            'size_min': int(sizes.min()), 'size_max': int(sizes.max()),
            'thr_median': round(float(np.median(thr)), 5), 'thr_p90': round(float(np.quantile(thr, .9)), 5),
            'peer_median': float(np.median(peers)), 'peer_mean': round(float(peers.mean()), 1),
            'peer_p90': float(np.quantile(peers, .9)), 'isolated_share': round(float((peers == 0).mean()), 4),
            'same_sw2_edge_share': round(sm / max(1, tot), 4),
            'sw2_lift': round((sm / max(1, tot)) / max(1e-9, base_rate), 2)}


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
    out = {'year': year,
           # Phase 0.3（2026-10-27）：N-9 口径注记（append-only；不改任何数值/算法）
           'metric_scope': {
               'peer_rule': 'min_threshold_within_cluster（原文第四步阈值法，eval_at）：peer_i = '
                            '#{j: sim(i,j) > min_{k,l in same cluster} sim(k,l)}',
               'sw2_lift_rule': '同申万二级有向边占比 / 同样本边际基线',
               'not_comparable_to': 'network_business_param_freeze_teacher.json 的同名字段'
                                    '（同团成员法：peer = 团规模−1）',
               'official_evidence_note': '2021 年 K=400 的 peer_median=3,347、sw2_lift=1.19 是本报告读数；'
                                         'freeze 2021 K=434 的 peer_median=45、sw2_lift=21.48 是另一轮'
                                         '（n_firms 3,544 vs 3,570，K 400 vs 434），'
                                         '不能读作同一网络的 74×/18× 差异',
               'phase0_3_note': '2026-10-27 Phase 0.3 口径注记；未改任何数值、未重跑网络',
           },
           'n_firms': len(firms), 'base_same_sw2_rate': round(base, 5),
           'filter_stats': stats, 'results': {}}
    for lk in ['average', 'complete', 'single']:
        t0 = time.time()
        Z = linkage(cond, method=lk)
        out['results'][lk] = {}
        for K in KS:
            lab = fcluster(Z, K, criterion='maxclust')
            r = eval_at(C, lab, l2, base)
            out['results'][lk][str(K)] = r
            print(f"  [{lk}] K={K:>5} size_med={r['size_median']:.0f} thr_med={r['thr_median']:.3f} "
                  f"peer_med={r['peer_median']:.0f} peer_p90={r['peer_p90']:.0f} lift={r['sw2_lift']}", flush=True)
        print(f'  {lk} done {time.time()-t0:.0f}s', flush=True)
    (REPD / 'network_business_k_explore.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('->', REPD / 'network_business_k_explore.json')


if __name__ == '__main__':
    main()
