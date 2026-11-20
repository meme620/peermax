"""analyst-network Step 8：网络诊断汇总 + 验收检查。

检查项:
  1. 月度累计指标（edges/degree/common_count/覆盖公司数）
  2. no-network-support：覆盖但无共同分析师的公司数（v2 起直接取 step67 诊断的
     firms_with_edges，不再逐行重读 97 个月边表）
  3. 抽查 3 个月边表：键重复、自环、权重和误差、对称性
  4. 映射失败率（来自 step3）
输出: data/processed/analyst/analyst_network_diagnostics.json
"""
import json
import statistics as st
from pathlib import Path
import pandas as pd

ANAD = Path('data/processed/analyst')
NET_VER = 'coverage_win12_fy12_v3'  # 与 step5/step67 同源
if __name__ == '__main__':
    D = json.loads((ANAD / 'step67_diagnostics.json').read_text(encoding='utf-8'))
    months = D['months']

    # 1. 汇总
    edges = [m['undirected_edges'] for m in months]
    deg_p50 = [m['degree_p50'] for m in months]
    deg_p90 = [m['degree_p90'] for m in months]
    cc_p50 = [m['common_count_p50'] for m in months]
    summary = {
        'months': len(months),
        'network_version': NET_VER,
        'edges_total_undirected': sum(edges),
        'edges_monthly_mean': int(st.mean(edges)),
        'edges_monthly_min': min(edges), 'edges_monthly_max': max(edges),
        'degree_p50_mean': round(st.mean(deg_p50), 1),
        'degree_p90_mean': round(st.mean(deg_p90), 1),
        'common_count_p50_mean': round(st.mean(cc_p50), 2),
    }
    # 密度/集中度诊断（三轮对抗 review 补充项，2026-08-12）
    weak = [m['weak_edge_ratio'] for m in months if m.get('weak_edge_ratio') is not None]
    top1 = [m['top1_weight_p50'] for m in months if m.get('top1_weight_p50') is not None]
    hhi = [m['weight_hhi_p50'] for m in months if m.get('weight_hhi_p50') is not None]
    summary['weak_edge_ratio_mean'] = round(st.mean(weak), 4) if weak else None
    summary['top1_weight_p50_mean'] = round(st.mean(top1), 6) if top1 else None
    summary['weight_hhi_p50_mean'] = round(st.mean(hhi), 6) if hhi else None
    summary['benchmark_reference'] = {
        'ah2020_us': {'min': 1, 'median': 71, 'mean': 86, 'max': 368,
                      'weak_edge_share': 0.70,
                      'source': 'Ali & Hirshleifer 2020 JFE 136(3) Table 1 Panel A / Table 11'},
        'long2025_cn': {'mean_peers_per_firm': 149, 'mean_common_per_pair': 13,
                        'median_common_per_pair': 8,
                        'source': 'Long, Zhu, Wang, Yao & Zaremba 2025 Modern Finance 3(3), CSMAR 2004-2022'},
        'note': '项目 A 股网络为中位 degree ~150-230，与 Long et al. 2025 中国基准（平均 149 家）同量级；'
                'common=1 弱边占比低于 A&H 美国基准（70%），弱边主导非项目特有。',
    }

    # 2. no-network-support：firms_covered（覆盖表）- firms_with_edges（step67 诊断）
    cov = pd.read_csv(ANAD / 'firm_analyst_coverage.csv', dtype=str, usecols=['network_cutoff', 'firm_id'])
    covered_firms = cov.groupby('network_cutoff')['firm_id'].nunique()
    no_support_vals = []
    for m in months:
        c = m['cutoff']
        n_covered = int(covered_firms[c])
        n_support = int(m.get('firms_with_edges', 0))
        no_support_vals.append(n_covered - n_support)
    summary['no_network_support_monthly_mean'] = round(st.mean(no_support_vals), 1)
    summary['no_network_support_monthly_min'] = min(no_support_vals)
    summary['no_network_support_monthly_max'] = max(no_support_vals)

    # 3. 抽查 3 个月边表质量
    quality = {}
    for c in ['2018-06-30', '2021-06-30', '2024-06-30']:
        df = pd.read_csv(ANAD / 'edges' / f'edges_{NET_VER}_{c}.csv', dtype=str)
        dup = int(df.duplicated(['network_use_case', 'network_version', 'network_cutoff',
                                 'focal_id', 'source_id']).sum())
        self_loop = int((df['focal_id'] == df['source_id']).sum())
        ver_ok = int((df['network_version'] == NET_VER).sum()) == len(df)
        w = pd.to_numeric(df['weight'], errors='coerce')
        wsum_err = float((w.groupby(df['focal_id']).sum() - 1).abs().max())
        # 对称性：抽查 common_analyst_count 对称
        piv = df[['focal_id', 'source_id', 'common_analyst_count']]
        rev = piv.rename(columns={'focal_id': 'source_id', 'source_id': 'focal_id'})
        merged = piv.merge(rev, on=['focal_id', 'source_id'], suffixes=('_f', '_s'))
        asym = int((merged['common_analyst_count_f'] != merged['common_analyst_count_s']).sum())
        quality[c] = {'rows': int(len(df)), 'key_dup': dup, 'self_loop': self_loop,
                      'network_version_ok': ver_ok,
                      'weight_sum_err_max': round(float(wsum_err), 8), 'asym_pairs': asym}

    summary['quality_spotcheck'] = quality

    # 4. 映射失败率 + 姓名对齐率（验收第 3 条：A_code 拆分与姓名对齐率，来自 step2_audit.json）
    step3_doc = json.loads((ANAD / 'step3_clean.json').read_text(encoding='utf-8'))
    summary['mapping'] = {'ok': step3_doc['mapped_ok'], 'total_split': step3_doc['split_rows'],
                          'fail_rate': round(1 - step3_doc['mapped_ok'] / step3_doc['split_rows'], 6)}
    try:
        step2_doc = json.loads((Path('data/raw/analyst') / 'step2_audit.json').read_text(encoding='utf-8'))
        summary['mapping']['name_aligned_rate'] = step2_doc['a_code']['name_aligned_rate']
        summary['mapping']['a_code_with_slash'] = step2_doc['a_code']['with_slash']
    except (FileNotFoundError, KeyError):
        summary['mapping']['name_aligned_rate'] = None
        print('WARN: step2_audit.json 缺失或字段不存在，name_aligned_rate 未纳入', flush=True)

    # 5. 缺失状态汇总（验收第 9 条澄清，2026-08-12 偏差 #7）：
    #    not_covered / not_matched / no-network-support 在诊断层显式分开报告
    try:
        chars = pd.read_csv('data/processed/firm_chars_clean.csv', usecols=['firm_id'], dtype=str)
        universe_firms = int(chars['firm_id'].nunique())
    except Exception:
        universe_firms = None
    step3_status = step3_doc.get('status_counts', {})
    not_matched_records = sum(v for k, v in step3_status.items() if k.startswith('not_matched'))
    summary['missing_status_summary'] = {
        'note': '缺失状态在诊断层分开（规划验收第 9 条）：覆盖表无行=not_covered；records 层 mapping_status 区分 not_matched 子类；no-network-support=覆盖但无边',
        'covered_firms_monthly_mean': round(float(st.mean(covered_firms)), 1),
        'universe_firms': universe_firms,
        'not_matched_records': int(not_matched_records),
        'not_matched_detail': {k: int(v) for k, v in step3_status.items() if k.startswith('not_matched')},
        'no_network_support_monthly_mean': summary['no_network_support_monthly_mean'],
    }

    # 6. 窗口覆盖率与状态字段（诊断 #6）：step5 的 coverage_window_incomplete + 覆盖表 flag/missing 分布
    try:
        spec_no_turnover = json.loads((ANAD / 'step5_snapshot.json').read_text(encoding='utf-8'))
        n_cut = spec_no_turnover.get('cutoffs', len(months))
        inc = spec_no_turnover.get('coverage_window_incomplete_count', 0)
        summary['window_coverage'] = {
            'cutoffs_total': n_cut,
            'coverage_window_incomplete_count': inc,
            'window_complete_ratio': round((n_cut - inc) / n_cut, 6) if n_cut else None,
        }
    except (FileNotFoundError, KeyError):
        summary['window_coverage'] = None
    cov_full = pd.read_csv(ANAD / 'firm_analyst_coverage.csv', dtype=str,
                           usecols=['coverage_flag', 'missing_reason'])
    summary['coverage_flag_dist'] = cov_full['coverage_flag'].value_counts().to_dict()
    # 注意：默认读 CSV 时空串为 NaN，须 fillna('') 再判非空（2026-08-12 修正）
    summary['missing_reason_nonempty'] = int((cov_full['missing_reason'].fillna('').astype(str).str.strip() != '').sum())

    (ANAD / 'analyst_network_diagnostics.json').write_text(
        json.dumps({'summary': summary, 'monthly': months}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
