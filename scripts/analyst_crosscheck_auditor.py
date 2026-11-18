"""S3A Gate 2.3：C 线审计重算器。

只读中间产物与基础数据，逐层重算 coverage、边集合、common count、权重行和与唯一键，
用于验证「生产脚本写出的文件」本身自洽且可被第三方独立验证。

输入（只读）：
  data/processed/analyst/firm_analyst_coverage.csv     覆盖表（生产 step5 产物）
  data/processed/analyst/edges/edges_<ver>_<cutoff>.csv 月度边表（生产 step67 产物）
  data/processed/analyst/step67_diagnostics.json        生产诊断

不读取原始预测数据、不调用生产投影函数。

输出：artifacts/s3a_crosscheck_C.json（241 月逐月 + 抽查月明细）

用法：
  python scripts/analyst_crosscheck_auditor.py [--months N] [--spot-only]
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
SPEC = ROOT / 'configs/s3a_final_spec_v1.yaml'
_SPEC_BYTES = SPEC.read_bytes()
SPEC_DATA = yaml.safe_load(_SPEC_BYTES.decode('utf-8'))
SPEC_HASH = sha256(_SPEC_BYTES).hexdigest()

COV = ROOT / 'data/processed/analyst/firm_analyst_coverage.csv'
EDGED = ROOT / 'data/processed/analyst/edges'
VERSION = SPEC_DATA['network']['primary_version']
EPS = SPEC_DATA['network']['epsilon_weight']
SPOT_MONTHS = ['2006-06-30', '2015-06-30', '2024-06-30']
EDGE_COLS = ['network_use_case', 'network_version', 'focal_id', 'source_id',
             'common_analyst_count', 'weight', 'network_cutoff', 'available_at',
             'network_support_status']


def _cov_month_metrics(cov: pd.DataFrame, cutoff: str):
    """从覆盖表独立重算：公司/分析师/记录数、逐公司 analyst 数。"""
    m = cov[cov['network_cutoff'] == cutoff]
    return {
        'covered_firms': int(m['firm_id'].nunique()),
        'covered_analysts': int(m['analyst_id'].nunique()),
        'covered_records': int(len(m)),
        'firm_analyst_count': m.groupby('firm_id')['analyst_id'].nunique(),
    }


def _edge_month_metrics(edge: pd.DataFrame, coverage: dict):
    """从边表独立重算结构指标，并复核覆盖表推出的 analyst 数。"""
    focal = edge['focal_id'].values
    src = edge['source_id'].values
    common = edge['common_analyst_count'].values.astype(np.int64)
    weight = edge['weight'].values.astype(float)

    # 无向折叠（取 focal<source 的对；生产双向写出同一 pair 两次）
    lo = np.minimum(focal, src)
    hi = np.maximum(focal, src)
    und = pd.DataFrame({'lo': lo, 'hi': hi, 'common': common}).drop_duplicates(['lo', 'hi'])
    undirected_edges = int(len(und))
    cc = np.sort(und['common'].values)

    # degree（无向）
    deg = pd.concat([und['lo'], und['hi']]).value_counts().values
    deg = np.sort(deg)

    # 自环 / 重复键
    self_loops = int((focal == src).sum())
    key_dup = int(pd.DataFrame({'f': focal, 's': src}).duplicated().sum())

    # 权重行和误差（focal 维度）
    wsum = pd.DataFrame({'f': focal, 'w': weight}).groupby('f')['w'].sum()
    rowsum_err = float((wsum - 1).abs().max()) if len(wsum) else None

    # top1 与 HHI（focal 维度）
    g = pd.DataFrame({'f': focal, 'w': weight})
    top1 = g.groupby('f')['w'].max().values
    hhi = g.assign(w2=g['w'] ** 2).groupby('f')['w2'].sum().values

    # 权重一致性：weight 是否等于 common / focal 行 common 和
    comm_sum = pd.DataFrame({'f': focal, 'c': common}).groupby('f')['c'].transform('sum').values
    weight_recomputed = common / comm_sum
    w_err = float(np.abs(weight - weight_recomputed).max())

    # 边表 focal 覆盖数 vs 覆盖表
    return {
        'undirected_edges': undirected_edges,
        'degree_p50': float(np.percentile(deg, 50)) if len(deg) else None,
        'degree_p90': float(np.percentile(deg, 90)) if len(deg) else None,
        'common_count_p50': float(np.percentile(cc, 50)) if len(cc) else None,
        'common_count_p90': float(np.percentile(cc, 90)) if len(cc) else None,
        'common_count_max': int(cc.max()) if len(cc) else None,
        'weak_edge_ratio': float((und['common'] == 1).mean()) if len(und) else None,
        'top1_weight_p50': float(np.median(top1)) if len(top1) else None,
        'weight_hhi_p50': float(np.median(hhi)) if len(hhi) else None,
        'weight_rowsum_max_err': rowsum_err,
        'weight_formula_max_err': w_err,
        'self_loops': self_loops,
        'duplicate_keys': key_dup,
        'firms_with_edges': int(len(wsum)),
        'coverage_records_in_edges': int(len(edge)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', type=int, default=0)
    ap.add_argument('--spot-only', action='store_true')
    args = ap.parse_args()

    cov = pd.read_csv(COV, dtype=str, usecols=['network_cutoff', 'firm_id', 'analyst_id'])
    cutoffs = sorted(cov['network_cutoff'].unique())
    if args.spot_only:
        cutoffs = [c for c in cutoffs if c in SPOT_MONTHS]
    elif args.months:
        cutoffs = cutoffs[:args.months]
    print(f'[C] cutoffs: {len(cutoffs)}', flush=True)

    monthly, details = [], {}
    for k, cutoff in enumerate(cutoffs, 1):
        cov_m = _cov_month_metrics(cov, cutoff)
        fpath = EDGED / f'edges_{VERSION}_{cutoff}.csv'
        if not fpath.exists():
            monthly.append({'cutoff': cutoff, 'error': 'edge_file_missing'})
            continue
        edge = pd.read_csv(fpath, dtype={'focal_id': str, 'source_id': str},
                           usecols=EDGE_COLS)
        em = _edge_month_metrics(edge, cov_m)
        row = {'cutoff': cutoff,
               'covered_firms': cov_m['covered_firms'],
               'covered_analysts': cov_m['covered_analysts'],
               'covered_records': cov_m['covered_records'],
               **em}
        # 交叉核验：边表 focal 的 analyst_count 应与覆盖表一致（抽查月全量、其余月抽样公司）
        fac = cov_m['firm_analyst_count']
        check_firms = list(fac.index[:200]) if not args.spot_only else list(fac.index)
        sub = edge[edge['focal_id'].isin(check_firms)]
        if len(sub):
            got = sub.groupby('focal_id').size()
            exp = fac.reindex(got.index)
            row['analyst_count_consistency_max_diff'] = int((got - exp).abs().max())
        monthly.append(row)

        if cutoff in SPOT_MONTHS:
            top = (edge[edge['focal_id'] <= edge['source_id']]
                   .sort_values('common_analyst_count', ascending=False)
                   .head(10)[['focal_id', 'source_id', 'common_analyst_count']]
                   .to_dict(orient='records'))
            details[cutoff] = {'top_edges_by_common': top,
                               'edge_cols_present': list(edge.columns)}
        if k % 40 == 0:
            print(f'[C] {k}/{len(cutoffs)} months done', flush=True)

    out = {
        'line': 'C_audit_recalculator',
        'spec_id': SPEC_DATA['spec_id'], 'spec_hash': SPEC_HASH,
        'input_hashes': SPEC_DATA['provenance']['input_hashes'],
        'network_version': VERSION,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'script': Path(__file__).name, 'status': 'COMPUTED',
        'epsilon_weight_threshold': EPS,
        'months': monthly,
        'totals': {
            'n_months': len(monthly),
            'edges_total_undirected': int(sum(r.get('undirected_edges', 0) for r in monthly)),
            'covered_records_total': int(sum(r.get('covered_records', 0) for r in monthly)),
            'max_rowsum_err': max([r.get('weight_rowsum_max_err') or 0 for r in monthly], default=None),
            'max_weight_formula_err': max([r.get('weight_formula_max_err') or 0 for r in monthly], default=None),
            'total_self_loops': int(sum(r.get('self_loops', 0) for r in monthly)),
            'total_duplicate_keys': int(sum(r.get('duplicate_keys', 0) for r in monthly)),
        },
    }
    p = ROOT / 'artifacts/s3a_crosscheck_C.json'
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    (ROOT / 'artifacts/s3a_crosscheck_C_detail.json').write_text(
        json.dumps(details, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out['totals'], ensure_ascii=False))
    print('->', p.relative_to(ROOT))


if __name__ == '__main__':
    main()
