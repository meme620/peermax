"""S3A Gate 2.2：B 线独立网络重实现。

规则来源：`configs/s3a_final_spec.yaml`（spec_hash 见输出），据此从冻结预测输入
重建 241 个月末网络，**不读生产 edges、不调用生产核心函数**。

与生产实现（A 线 run_analyst_pipeline 的 step5+step67）的有意差异：
  - 窗口取数用布尔掩码，而非 searchsorted + 排序切片；
  - 二部图投影用 pandas merge（analyst 作 join key），而非 defaultdict 循环；
  - 行归一化分母用 groupby.transform('sum')，而非手工 defaultdict 累加。

输出（每个指标带三线对账所需的逐月粒度）：
  artifacts/s3a_crosscheck_B.json          241 月逐月指标
  artifacts/s3a_crosscheck_B_detail.json   3 个抽查月明细（top edges）

用法：
  python scripts/analyst_crosscheck_reimplementation.py [--months N] [--limit-firms K]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/s3a_final_spec.yaml'
_SPEC_BYTES = SPEC.read_bytes()
SPEC_DATA = yaml.safe_load(_SPEC_BYTES.decode('utf-8'))
SPEC_HASH = sha256(_SPEC_BYTES).hexdigest()

SRC = ROOT / SPEC_DATA['inputs']['analyst_forecast']
VERSION = SPEC_DATA['network']['primary_version']
WINDOW_MONTHS = SPEC_DATA['network']['coverage_window_months']
CODE_RE = SPEC_DATA['network']['analyst_code_regex']
CUTOFFS = pd.date_range(
    SPEC_DATA['network']['cutoff_range'].split('..')[0],
    SPEC_DATA['network']['cutoff_range'].split('..')[1],
    freq='ME')
SPOT_MONTHS = ['2006-06-30', '2015-06-30', '2024-06-30']


def _load_records(limit_firms=None):
    """独立读原始清洗表：只保留 spec 规定的列，自助施加 mapping/A_code/FY 规则。"""
    usecols = ['firm_id', 'Anndats', 'Fpenddt', 'analyst_id', 'broker_code', 'mapping_status']
    rec = pd.read_csv(SRC, dtype=str, keep_default_na=False, usecols=usecols)
    rec = rec[rec['mapping_status'] == 'ok'].copy()
    rec = rec[rec['analyst_id'].str.fullmatch(CODE_RE, na=False)].copy()
    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    rec['fpend'] = pd.to_datetime(rec['Fpenddt'], errors='coerce')
    gap = rec['fpend'].dt.year - rec['report_date'].dt.year
    rec = rec[gap.between(0, 1)].copy()
    if limit_firms:
        keep = sorted(rec['firm_id'].unique())[:limit_firms]
        rec = rec[rec['firm_id'].isin(keep)].copy()
    return rec


def _month_metrics(cov_month, prev_firms=None):
    """给定单月覆盖表 (firm_id, analyst_id)，独立算边与权重。"""
    firm_of_analyst = cov_month.groupby('analyst_id')['firm_id'].apply(list)
    rows = []
    for aid, firms in firm_of_analyst.items():
        fs = sorted(set(firms))
        if len(fs) < 2:
            continue
        # 用 numpy 广播生成无序对
        arr = np.array(fs, dtype=object)
        i_idx, j_idx = np.triu_indices(len(arr), k=1)
        rows.append(pd.DataFrame({'focal_id': arr[i_idx], 'source_id': arr[j_idx]}))
    if not rows:
        return None, None
    pairs = pd.concat(rows, ignore_index=True)
    pairs['common_analyst_count'] = 1
    strong = pairs.groupby(['focal_id', 'source_id'], as_index=False)['common_analyst_count'].sum()

    # 对称展开：无向边 (i,j) 须在 i 与 j 各产生一行 focal 记录，
    # 否则 focal 的行归一化分母只累计「较小 id 那端」的边，权重被系统性放大。
    mirror = strong.rename(columns={'focal_id': 'source_id', 'source_id': 'focal_id'})
    strong = pd.concat([strong, mirror], ignore_index=True)

    analyst_count = cov_month.groupby('firm_id')['analyst_id'].nunique().rename('analyst_count')
    strong = strong.merge(analyst_count.rename_axis('focal_id').reset_index(), on='focal_id', how='left')
    strong = strong.merge(analyst_count.rename_axis('source_id').reset_index().rename(
        columns={'analyst_count': 'analyst_count_source'}), on='source_id', how='left')
    denom = strong.groupby('focal_id')['common_analyst_count'].transform('sum')
    strong['weight'] = strong['common_analyst_count'] / denom

    # 无向指标：按 (min,max) 折叠
    lo = np.minimum(strong['focal_id'], strong['source_id'])
    hi = np.maximum(strong['focal_id'], strong['source_id'])
    undirected = pd.DataFrame({'lo': lo, 'hi': hi,
                               'common': strong['common_analyst_count'].values})
    undirected = undirected.drop_duplicates(['lo', 'hi'])
    return strong, undirected


def _deg_quantiles(undirected):
    deg = pd.concat([undirected['lo'], undirected['hi']]).value_counts()
    d = np.sort(deg.values)
    return (float(np.percentile(d, 50)), float(np.percentile(d, 90))) if len(d) else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=0, help='只跑前 N 个月（冒烟用）')
    ap.add_argument('--limit-firms', type=int, default=0, help='只保留前 K 家公司（冒烟用）')
    args = ap.parse_args()

    rec = _load_records(args.limit_firms or None)
    print(f'[B] records: {len(rec)}  firms={rec["firm_id"].nunique()}', flush=True)

    cutoffs = CUTOFFS[:args.months] if args.months else CUTOFFS
    monthly, details = [], {}
    for k, cutoff in enumerate(cutoffs, 1):
        ws = cutoff - pd.DateOffset(months=WINDOW_MONTHS)
        m = rec[(rec['report_date'] > ws) & (rec['report_date'] <= cutoff)]
        if m.empty:
            continue
        cov = m[['firm_id', 'analyst_id']].drop_duplicates()
        strong, undirected = _month_metrics(cov)
        if strong is None:
            continue
        key = cutoff.strftime('%Y-%m-%d')
        p50, p90 = _deg_quantiles(undirected)
        cc = np.sort(undirected['common'].values)
        top1 = strong.groupby('focal_id')['weight'].max()
        wsum = strong.groupby('focal_id')['weight'].sum()
        hhi = strong.assign(w2=strong['weight'] ** 2).groupby('focal_id')['w2'].sum()
        rec_row = {
            'cutoff': key,
            'covered_firms': int(cov['firm_id'].nunique()),
            'covered_analysts': int(cov['analyst_id'].nunique()),
            'covered_records': int(len(cov)),
            'undirected_edges': int(len(undirected)),
            'degree_p50': p50, 'degree_p90': p90,
            'common_count_p50': float(np.percentile(cc, 50)),
            'common_count_p90': float(np.percentile(cc, 90)),
            'common_count_max': int(cc.max()),
            'weak_edge_ratio': float((undirected['common'] == 1).mean()),
            'top1_weight_p50': float(top1.median()),
            'weight_hhi_p50': float(hhi.median()),
            'weight_rowsum_max_err': float((wsum - 1).abs().max()),
            'duplicate_keys': int(strong.duplicated(['focal_id', 'source_id']).sum()),
            'self_loops': int((strong['focal_id'] == strong['source_id']).sum()),
        }
        monthly.append(rec_row)
        if key in SPOT_MONTHS:
            top = (strong.sort_values('common_analyst_count', ascending=False)
                   .head(10)[['focal_id', 'source_id', 'common_analyst_count']]
                   .to_dict(orient='records'))
            details[key] = {'top_edges_by_common': top}
        if k % 40 == 0:
            print(f'[B] {k}/{len(cutoffs)} months done', flush=True)

    out = {
        'line': 'B_independent_reimplementation',
        'spec_id': SPEC_DATA['spec_id'], 'spec_hash': SPEC_HASH,
        'input_hashes': SPEC_DATA['provenance']['input_hashes'],
        'network_version': VERSION,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'script': Path(__file__).name, 'status': 'COMPUTED',
        'smoke': {'months': args.months or len(cutoffs), 'limit_firms': args.limit_firms or None},
        'months': monthly,
        'totals': {
            'n_months': len(monthly),
            'edges_total_undirected': int(sum(r['undirected_edges'] for r in monthly)),
            'covered_records_total': int(sum(r['covered_records'] for r in monthly)),
        },
    }
    p = ROOT / 'artifacts/s3a_crosscheck_B.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    (ROOT / 'artifacts/s3a_crosscheck_B_detail.json').write_text(
        json.dumps(details, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out['totals'], ensure_ascii=False))
    print('->', p.relative_to(ROOT))


if __name__ == '__main__':
    main()
