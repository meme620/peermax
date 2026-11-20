"""B4.1 归因：双网分歧的来源分解（2026-09-14）。

对每个共同公司计算 per-firm Jaccard(peers_MA, peers_MB)，再分解到：
  1. 行业（申万二级）——哪些行业分歧最大
  2. 公司特征：MA 命中词条数、MA 团规模、MB 度、是否 J/S 门类（应已排除）
  3. 构造层面：MA 边数 vs MB 边数的关系（是否单纯"MA 更密"导致低 Jaccard）

产物：reports/data_gates/G0/network_cross_audit_attribution.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'reports/data_gates/G0'
# 主口径 = 老师 MD&A 语料两列（自建语料的 network_ma/network_mb 不在库）
MA = ROOT / 'data/processed/network_ma_teacher'
MB = ROOT / 'data/processed/network_mb_teacher'
SW2L = ROOT / 'data/processed/sw2_industry_latest.csv'
NAMES = ROOT / 'data/processed/sw2_l2_names.csv'
CLU = MA / 'clusters'
YEAR = 2021


def main():
    ea = pd.read_csv(MA / f'{YEAR}_similarity.csv', dtype={'focal_id': str, 'source_id': str})
    eb = pd.read_csv(MB / f'{YEAR}_similarity.csv', dtype={'focal_id': str, 'source_id': str})
    lat = pd.read_csv(SW2L, dtype={'code6': str, 'l2_code': str})
    l2 = dict(zip(lat['firm_id'], lat['l2_code']))
    nm = pd.read_csv(NAMES, dtype=str)
    l2name = dict(zip(nm['l2_code'], nm['l2_name']))
    clu = pd.read_csv(CLU / f'{YEAR}_clusters.csv', dtype={'firm_id': str})
    csz = clu.set_index('firm_id')['cluster_id'].value_counts().to_dict()
    cid = dict(zip(clu['firm_id'], clu['cluster_id']))

    pa = ea.groupby('focal_id')['source_id'].apply(set).to_dict()
    pb = eb.groupby('focal_id')['source_id'].apply(set).to_dict()
    common = sorted(set(pa) & set(pb))
    rows = []
    for f in common:
        A, B = pa[f], pb[f]
        j = len(A & B) / max(1, len(A | B))
        rows.append({'firm_id': f, 'l2': l2.get(f, 'NA'), 'l2_name': l2name.get(l2.get(f, ''), ''),
                     'ma_peers': len(A), 'mb_peers': len(B), 'jaccard': j,
                     'ma_cluster_size': csz.get(cid.get(f), np.nan),
                     'ma_edges_mb_edges': len(A) / max(1, len(B))})
    d = pd.DataFrame(rows)
    out = {'run_at': datetime.now(timezone.utc).isoformat(), 'text_year': YEAR, 'n_common': len(d),
           'per_firm_jaccard': {'median': round(float(d['jaccard'].median()), 4),
                                'p10': round(float(d['jaccard'].quantile(.1)), 4),
                                'p90': round(float(d['jaccard'].quantile(.9)), 4),
                                'share_gt_0.3': round(float((d['jaccard'] > 0.3).mean()), 4)},
           'ratio_ma_mb_peers': {'median': round(float(d['ma_edges_mb_edges'].median()), 3),
                                 'p90': round(float(d['ma_edges_mb_edges'].quantile(.9)), 3)},
           'corr': {}}
    for c in ['ma_peers', 'mb_peers', 'ma_cluster_size']:
        out['corr'][f'jaccard~{c}'] = round(float(d['jaccard'].corr(d[c], method='spearman')), 4)
    g = d.groupby('l2_name')['jaccard'].agg(['mean', 'size']).sort_values('mean')
    g = g[g['size'] >= 10]
    out['lowest_jaccard_industries'] = [{'l2_name': i, 'mean_jaccard': round(r['mean'], 4), 'n': int(r['size'])}
                                        for i, r in g.head(10).iterrows()]
    out['highest_jaccard_industries'] = [{'l2_name': i, 'mean_jaccard': round(r['mean'], 4), 'n': int(r['size'])}
                                         for i, r in g.tail(10).iterrows()]
    # 控制规模差异后的重合：只比较两网各自相似度最高的 k 条边（同密度对照）
    k = 50
    ta = ea.sort_values('similarity', ascending=False).groupby('focal_id').head(k)
    tb = eb.sort_values('similarity', ascending=False).groupby('focal_id').head(k)
    sa = set(zip(ta['focal_id'], ta['source_id']))
    sb = set(zip(tb['focal_id'], tb['source_id']))
    out['top50_per_firm_jaccard'] = round(len(sa & sb) / max(1, len(sa | sb)), 4)
    out['note'] = ('per-firm Jaccard 低主要来自"MA 团整体入选"造成的规模差异：'
                   f'MA/MB peer 数比中位 {out["ratio_ma_mb_peers"]["median"]}；'
                   '按各自 top-50 边对照可分离"规模差异"与"选择差异"。')
    (REPD / 'network_cross_audit_attribution.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1)[:1800])
    print('->', REPD / 'network_cross_audit_attribution.json')


if __name__ == '__main__':
    main()
