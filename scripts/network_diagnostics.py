"""S3 Step 3：网络诊断（设计规格 required output，2026-09-14）。

设计真源（execution_plan §5.6 Step 3）：
    NetworkMass_{i,d}       = sum_j Similarity_{ij,d-}
    EffectivePeerCount_{i,d}= 1 / sum_j w_{ij,d-}^2
    Top1Weight_{i,d}        = max_j w_{ij,d-}
  口径要求：仅当存在 ≥1 条有效边且 sum_j w^2 > epsilon_weight(=1e-4) 时计算 EPC；
  无有效边 / 分母过小 / Top1 集合为空 → 写 missing 并沿用 no-network-support，**不得写 0 或无穷**。
  报告：边数、NetworkMass、EffectivePeerCount、Top1Weight、no-network-support 占比、高度集中网络占比。

产物：
  data/processed/network_{ma,mb}/{text_year}_diagnostics.csv
  reports/data_gates/G0/s3_diagnostics_{ma,mb}_report.json
用法：python scripts/network_diagnostics.py [--use-case ma|mb|both]
"""
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'reports/data_gates/G0'
NETS = {'ma': ROOT / 'data/processed/network_ma', 'mb': ROOT / 'data/processed/network_mb'}
EPSILON_WEIGHT = 1e-4
YEARS = list(range(2017, 2026))
HIGH_CONC_TOP1 = 0.5   # 高度集中定义：单条边权重 > 0.5
COLS = ['network_use_case', 'network_version', 'focal_id', 'available_at',
        'n_edges', 'network_mass', 'effective_peer_count', 'top1_weight', 'top1_source_id',
        'network_support_status', 'source_version', 'coverage_flag', 'missing_reason']


def run(uc, year):
    d = NETS[uc]
    ver = json.loads((d / f'{year}_build_report.json').read_text(encoding='utf-8'))
    avail, srcver = ver['available_at'], ver.get('network_version') or f'{uc}_{year}'
    agg = defaultdict(lambda: {'mass': 0.0, 'w2': 0.0, 'top1': 0.0, 'top1src': '', 'n': 0})
    with open(d / f'{year}_similarity.csv', encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            a = agg[r['focal_id']]
            s, w = float(r['similarity']), float(r['weight'])
            a['mass'] += s
            a['w2'] += w * w
            a['n'] += 1
            if w > a['top1']:
                a['top1'], a['top1src'] = w, r['source_id']
    sup = pd.read_csv(d / f'{year}_support.csv', dtype={'focal_id': str})
    rows, stats = [], {'n_nodes': 0, 'n_no_support': 0, 'n_epc_missing': 0, 'n_high_conc': 0,
                       'mass': [], 'epc': [], 'top1': [], 'edges': []}
    for focal in sup['focal_id']:
        st = sup.loc[sup['focal_id'] == focal, 'network_support_status'].iloc[0]
        a = agg.get(focal)
        stats['n_nodes'] += 1
        if a is None or a['n'] == 0:
            stats['n_no_support'] += 1
            rows.append([f'product_market_{uc}', srcver, focal, avail, 0, '', '', '', '',
                         'no-network-support', 'business_texts', 'not_covered', 'no_network_support'])
            continue
        if a['w2'] <= EPSILON_WEIGHT:
            stats['n_epc_missing'] += 1
            epc, epc_reason = '', 'denominator_below_epsilon'
        else:
            epc, epc_reason = round(1.0 / a['w2'], 6), ''
        if a['top1'] > HIGH_CONC_TOP1:
            stats['n_high_conc'] += 1
        stats['mass'].append(a['mass']); stats['top1'].append(a['top1']); stats['edges'].append(a['n'])
        if epc != '':
            stats['epc'].append(epc)
        rows.append(['product_market_' + uc, srcver, focal, avail, a['n'],
                     round(a['mass'], 8), epc, round(a['top1'], 8), a['top1src'],
                     'support', 'business_texts', 'covered', epc_reason])
    with open(d / f'{year}_diagnostics.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows(rows)
    n = stats['n_nodes']

    def q(x, p):
        return round(float(np.quantile(x, p)), 6) if x else None
    return {
        'text_year': year, 'network_year': year + 1, 'network_version': srcver, 'available_at': avail,
        'n_nodes': n, 'n_no_support': stats['n_no_support'],
        'no_network_support_share': round(stats['n_no_support'] / n, 4),
        'n_effective_peer_count_missing': stats['n_epc_missing'],
        'high_concentration_share': round(stats['n_high_conc'] / n, 4),
        'edges': {'median': q(stats['edges'], .5), 'p90': q(stats['edges'], .9)},
        'network_mass': {'median': q(stats['mass'], .5), 'p90': q(stats['mass'], .9)},
        'effective_peer_count': {'median': q(stats['epc'], .5), 'p10': q(stats['epc'], .1)},
        'top1_weight': {'median': q(stats['top1'], .5), 'p90': q(stats['top1'], .9)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--use-case', choices=['ma', 'mb', 'both'], default='both')
    args = ap.parse_args()
    ucs = ['ma', 'mb'] if args.use_case == 'both' else [args.use_case]
    rep = {'run_at': datetime.now(timezone.utc).isoformat(), 'epsilon_weight': EPSILON_WEIGHT,
           'high_concentration_definition': f'top1_weight > {HIGH_CONC_TOP1}', 'by_use_case': {}}
    for uc in ucs:
        rep['by_use_case'][uc] = [run(uc, y) for y in YEARS]
        m = [r['effective_peer_count']['median'] for r in rep['by_use_case'][uc]]
        ns = [r['no_network_support_share'] for r in rep['by_use_case'][uc]]
        print(f'{uc}: EPC 中位 {m} | no-support 占比 {ns}', flush=True)
    (REPD / 's3_diagnostics_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print('->', REPD / 's3_diagnostics_report.json')


if __name__ == '__main__':
    main()
