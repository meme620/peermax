"""network-B4.1：双网交叉审计（M-A headline vs M-B robustness，2026-09-14）。

检查项（plan §8 B4.1）：
  1. 分布对比：逐年 peer 数、权重集中度（HHI/Top1）、孤立率 —— 同节点集下并排
  2. 行业对齐：边中同申万二级占比 + 相对基准 lift
  3. 重合度：Jaccard |E_A∩E_B|/|E_A∪E_B|（(focal,source) 对，限共同节点集）
  4. 聚合一致性（构造效度）：同月 PeerReturn 在两网下的相关性与分位一致性
  5. 差异归因：分歧最大的公司按行业/规模画像

产物：reports/data_gates/G0/network_cross_audit.json
用法：python scripts/network_cross_audit.py [--years 2017 ... 2025]
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'data/gates' if False else ROOT / 'reports/data_gates/G0'
# 主口径 = 老师 MD&A 语料（teacher）两列；自建语料的 network_ma/network_mb 不在库，
# 指向它们会让本审计直接 FileNotFoundError。文本年 2014–2022 对应网络年 2015–2023。
MA = ROOT / 'data/processed/network_ma_teacher'
MB = ROOT / 'data/processed/network_mb_teacher'
SW2L = ROOT / 'data/processed/sw2_industry_latest.csv'
RET = ROOT / 'data/processed/monthly_returns_clean_v2.csv'
YEARS = list(range(2014, 2023))


def net_stats(edges, l2, base):
    peer = edges.groupby('focal_id').size()
    w = edges.groupby('focal_id')['weight']
    hhi = (edges['weight'] ** 2).groupby(edges['focal_id']).sum()
    top1 = w.max()
    tot = len(edges)
    ek = edges[['focal_id', 'source_id']]
    a = ek['focal_id'].map(l2); b = ek['source_id'].map(l2)
    same = int((a == b).sum())
    return {
        'n_edges': int(tot), 'n_nodes_with_edges': int(len(peer)),
        'peer_median': float(peer.median()), 'peer_p90': float(peer.quantile(.9)),
        'peer_max': int(peer.max()),
        'weight_hhi_median': float(hhi.median()), 'top1_weight_median': float(top1.median()),
        'same_sw2_edge_share': round(same / max(1, tot), 4),
        'sw2_lift': round((same / max(1, tot)) / max(1e-9, base), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, nargs='*', default=YEARS)
    args = ap.parse_args()
    lat = pd.read_csv(SW2L, dtype={'code6': str, 'l2_code': str})
    l2 = dict(zip(lat['firm_id'], lat['l2_code']))
    ret = pd.read_csv(RET, dtype={'firm_id': str}, usecols=['firm_id', 'month', 'R'])
    rep = {'run_at': datetime.now(timezone.utc).isoformat(), 'years': {}}
    for y in args.years:
        ea = pd.read_csv(MA / f'{y}_similarity.csv', dtype={'focal_id': str, 'source_id': str})
        eb = pd.read_csv(MB / f'{y}_similarity.csv', dtype={'focal_id': str, 'source_id': str})
        na, nb = set(ea['focal_id']), set(eb['focal_id'])
        common = na & nb
        ea_c = ea[ea['focal_id'].isin(common) & ea['source_id'].isin(common)]
        eb_c = eb[eb['focal_id'].isin(common) & eb['source_id'].isin(common)]
        sa = set(zip(ea_c['focal_id'], ea_c['source_id']))
        sb = set(zip(eb_c['focal_id'], eb_c['source_id']))
        jac = len(sa & sb) / max(1, len(sa | sb))
        # 基准：共同节点内的 SW2 同行业对占比
        cl = pd.Series(list(common))
        vc = cl.map(l2).value_counts()
        tot_pairs = len(common) * (len(common) - 1) / 2
        base = float((vc * (vc - 1) / 2).sum() / tot_pairs)
        st_a = net_stats(ea_c, l2, base)
        st_b = net_stats(eb_c, l2, base)
        # 聚合一致性：取该网络版本可用月起的一个月，算 PeerReturn 相关
        # 文本年 y → 网络年 y+1，available_at = (y+1)-04-30 → 选 (y+1)-06 月
        month = f'{y + 1}-06'
        rm = ret[ret['month'] == month].set_index('firm_id')['R'].to_dict()

        def peer_return(edges):
            e = edges[edges['source_id'].isin(rm.keys())].copy()
            e['r'] = e['source_id'].map(rm)
            e['wr'] = e['weight'] * e['r']
            g = e.groupby('focal_id').agg(wr=('wr', 'sum'), w=('weight', 'sum'),
                                          n=('source_id', 'size'), nret=('r', 'count'))
            g = g[g['nret'] >= 5]
            return (g['wr'] / g['w']).rename('pr'), g
        pa, ga = peer_return(ea_c)
        pb, gb = peer_return(eb_c)
        both = pa.to_frame('A').join(pb.to_frame('B'), how='inner')
        corr = float(both['A'].corr(both['B'])) if len(both) > 10 else None
        spear = float(both['A'].corr(both['B'], method='spearman')) if len(both) > 10 else None
        rep['years'][str(y)] = {
            'text_year': y, 'network_year': y + 1, 'month_for_aggregation': month,
            'nodes': {'ma': len(na), 'mb': len(nb), 'common': len(common),
                      'ma_only': len(na - nb), 'mb_only': len(nb - na)},
            'edge_jaccard_on_common': round(jac, 4),
            'base_same_sw2_rate_common': round(base, 5),
            'ma': st_a, 'mb': st_b,
            'peer_return': {'n_firms': int(len(both)), 'pearson': round(corr, 4) if corr else None,
                            'spearman': round(spear, 4) if spear else None,
                            'ma_std': round(float(both['A'].std()), 5) if len(both) else None,
                            'mb_std': round(float(both['B'].std()), 5) if len(both) else None},
        }
        print(f"[{y}] common={len(common)} jaccard={jac:.3f} | MA edges={st_a['n_edges']} peer_med={st_a['peer_median']:.0f} "
              f"lift={st_a['sw2_lift']} | MB edges={st_b['n_edges']} peer_med={st_b['peer_median']:.0f} lift={st_b['sw2_lift']} "
              f"| PeerReturn corr={corr:.3f} (n={len(both)})", flush=True)
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'network_cross_audit.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print('->', REPD / 'network_cross_audit.json')


if __name__ == '__main__':
    main()
