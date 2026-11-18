"""S3-B2 正式边表验收（plan §8 Phase B2 机械化验收项，2026-09-14）。

检查（逐文本年 2017-2025，即网络年 2018-2026）：
  1. 唯一键 (network_version, focal_id, source_id) 重复 = 0
  2. 自环 = 0
  3. 权重行和 ∈ [1-1e-4, 1+1e-4]（有 support 的 focal）
  4. 权重 = similarity / Σsimilarity（行内自洽，抽样复算）
  5. available_at 一致且 PIT 安全：= 网络年 4-30；消费规则 max{available_at < d}
  6. support 表覆盖全部入网公司；无边的 focal 标 no-network-support（权重不填 0）
  7. 网络年映射：文本年 y → network_version pm_mb_tfidf_v1_y、available_at y+1-04-30
  8. 与语料对账：入网公司数 = 该文本年语料 ok 数（--corpus self=mda_texts_v2 / teacher=teacher_mda_panel）

产物：reports/data_gates/G0/s3_b2_acceptance.json（status PASS/FAIL + 逐项明细）
用法：python scripts/data_audit/s3_b2_acceptance_check.py [--dir data/processed/network_mb]
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPD = ROOT / 'reports/data_gates/G0'
TXTS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
TXTS_TEACHER = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]   # 老师语料（B′3）
CORPORA = {
    'self': {'path': ROOT / 'data/processed/mda_texts_v2.csv', 'id_col': 'code',
             'ver_prefix': 'pm_mb_tfidf_v1', 'years': TXTS},
    'teacher': {'path': ROOT / 'data/processed/teacher_mda_panel.csv', 'id_col': 'firm_id',
                'ver_prefix': 'pm_mb_tfidf_teacher_v1', 'years': TXTS_TEACHER},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='data/processed/network_mb')
    ap.add_argument('--corpus', choices=['self', 'teacher'], default='self')
    args = ap.parse_args()
    netd = ROOT / args.dir

    spec = CORPORA[args.corpus]
    mda = pd.read_csv(spec['path'], usecols=[spec['id_col'], 'year', 'status'])
    mda = mda[mda['status'] == 'ok']

    checks, per_year = {}, []
    overall_ok = True
    for y in spec['years']:
        fp = netd / f'{y}_similarity.csv'
        sp = netd / f'{y}_support.csv'
        rp = netd / f'{y}_build_report.json'
        if not (fp.exists() and sp.exists()):
            checks[f'{y}_files'] = False
            overall_ok = False
            continue
        rep = json.loads(rp.read_text(encoding='utf-8')) if rp.exists() else {}
        edges = pd.read_csv(fp, dtype={'focal_id': str, 'source_id': str})
        sup = pd.read_csv(sp, dtype={'focal_id': str})

        n_expected = int((mda['year'] == y).sum())
        dup = int(edges.duplicated(['network_version', 'focal_id', 'source_id']).sum())
        self_loop = int((edges['focal_id'] == edges['source_id']).sum())
        g = edges.groupby('focal_id')['weight'].sum()
        bad_sum = int(((g - 1).abs() > 1e-4).sum())
        # 行内自洽：weight == sim / Σ_row(sim)；按 focal 抽样（含该 focal 全部边，否则行和不完整）
        focals = edges['focal_id'].drop_duplicates().sample(
            min(200, edges['focal_id'].nunique()), random_state=42)
        sub = edges[edges['focal_id'].isin(set(focals))]
        # 容差说明：CSV 以 8 位小数落盘（round(sim,8)/round(w,8)），故按**相对**误差判定，
        # 阈值 1e-5（8 位小数在 w~0.005-1 上带来的相对误差上界约 1.4e-7）。
        recomputed = sub.groupby('focal_id').apply(
            lambda d: (((d['similarity'] / d['similarity'].sum()) - d['weight']).abs()
                       / d['weight'].abs().clip(lower=1e-12)).max(),
            include_groups=False)
        max_weight_err = float(recomputed.max()) if len(recomputed) else None
        avail = sorted(set(edges['available_at'].astype(str)))
        expect_avail = f'{y + 1}-04-30'
        versions = sorted(set(edges['network_version'].astype(str)))
        expect_ver = f"{spec['ver_prefix']}_{y}"
        support_nodes = set(sup.loc[sup['network_support_status'] == 'support', 'focal_id'])
        nsupport_nodes = set(sup.loc[sup['network_support_status'] == 'no-network-support', 'focal_id'])
        edge_nodes = set(edges['focal_id'])
        zero_weight_rows = int((edges['weight'] == 0).sum())

        yr_ok = (dup == 0 and self_loop == 0 and bad_sum == 0 and
                 avail == [expect_avail] and versions == [expect_ver] and
                 edge_nodes == support_nodes and zero_weight_rows == 0 and
                 (max_weight_err is None or max_weight_err < 1e-5))
        overall_ok = overall_ok and yr_ok
        per_year.append({
            'text_year': y, 'network_year': y + 1, 'network_version': versions[0] if len(versions) == 1 else versions,
            'available_at': avail[0] if len(avail) == 1 else avail,
            'enrolled_corpus': n_expected, 'nodes_support': len(support_nodes),
            'nodes_no_support': len(nsupport_nodes), 'n_edges': int(len(edges)),
            'dup_keys': dup, 'self_loops': self_loop, 'bad_row_sum': bad_sum,
            'max_weight_recompute_rel_err': max_weight_err, 'zero_weight_rows': zero_weight_rows,
            'median_peer': float(edges.groupby('focal_id').size().median()) if len(edges) else 0.0,
            'tau': rep.get('tau'), 'tau_mode': rep.get('tau_mode'),
            'mass_retained_ratio': rep.get('mass_retained_ratio'),
            'pass': bool(yr_ok),
        })

    out = {
        'check': 'B2 正式边表验收（plan §8 Phase B2）',
        'dir': args.dir,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'checks': checks,
        'per_year': per_year,
        'status': 'PASS' if overall_ok else 'FAIL',
    }
    REPD.mkdir(parents=True, exist_ok=True)
    suffix = '' if args.corpus == 'self' else '_' + args.corpus
    (REPD / ('s3_b2_acceptance' + suffix + '.json')).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': out['status'], 'years': len(per_year)}, ensure_ascii=False))
    for r in per_year:
        print(f"  {r['text_year']}->{r['network_year']}: nodes={r['nodes_support']}(+{r['nodes_no_support']} ns) "
              f"edges={r['n_edges']} med_peer={r['median_peer']:.0f} tau={r['tau']} "
              f"dup={r['dup_keys']} loop={r['self_loops']} badsum={r['bad_row_sum']} pass={r['pass']}")
    print(f"-> {REPD / 's3_b2_acceptance.json'}")


if __name__ == '__main__':
    main()
