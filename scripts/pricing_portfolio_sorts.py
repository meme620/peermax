#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A8：M1 组合分位（五/十分位 EW/VW）与函数形式诊断（bins / RCS / (MAX-c)+）。

对应 plan §5.7A P3.4 与 §5.7 Step 2/Step 3。headline 仍是连续 PeerMAXraw；
组合与形状诊断只作 supporting/exploratory 证据。空组与零分母写状态、不填 0。
产出：reports/data_gates/G0/s3_m1_portfolio_deciles.json、s3_m1_diagnostics.json
用法：python scripts/pricing_portfolio_sorts.py [--sample-version {v1,v2}] [--out-suffix _v2]

2026-10-11 F-2（v2 样本重跑批）：新增 `--sample-version`（默认 v2 headline；另一版本自动作敏感性并报）
与 `--out-suffix`（如 `--out-suffix _v2` 产出 s3_m1_portfolio_deciles_v2.json / s3_m1_diagnostics_v2.json，
不覆盖 v1 基线名）；组合与诊断块补**窗口标注**（FM 月范围）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pricing_lib import IQR_POOL_REGISTERED, M1_SAMPLE_VERSION_V1, M1_SAMPLE_VERSION_V2, M1_SAMPLE_VERSION_V3, M1_SAMPLE_VERSION_V4, OUT_DIR, fm_regression, joint_wald, newey_west, one_sided_p, param_value, progress, rank_average_groups, rcs_basis, sample_path, treatment_only_mask, two_sided_p, write_json

P = param_value('m1_sample_and_spec_registry')
NW_LAG = int(param_value('nw_lag'))
MIN_N = P['min_focal_count_per_month']
K_SPREAD = param_value('K_Spread_M1')
CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage', 'NetworkDegree']
BASE_X = ['OwnMAXraw', 'PeerReturn'] + CONTROLS


def portfolio_table(df: pd.DataFrame, n_groups: int) -> dict:
    df = df.copy()
    df['_grp'] = df.groupby('month', group_keys=False)['PeerMAXraw'].apply(
        lambda s: rank_average_groups(s, n_groups))
    rows = []
    status_counts = {}
    for m, sub in df.groupby('month', sort=True):
        sub = sub.dropna(subset=['_grp'])
        for g, gs in sub.groupby('_grp'):
            gs = gs.dropna(subset=['R_next'])
            n = len(gs)
            ew = float(gs['R_next'].mean()) if n > 0 else np.nan
            w = gs['market_equity']
            okv = w.notna() & (w > 0) & gs['R_next'].notna()
            if okv.any():
                vw = float(np.average(gs.loc[okv, 'R_next'], weights=w[okv]))
                vw_status = 'OK'
            else:
                vw, vw_status = np.nan, 'ZERO_WEIGHT_DENOM'
            status_counts[f'ew_{"EMPTY" if n == 0 else "OK"}'] = status_counts.get(
                f'ew_{"EMPTY" if n == 0 else "OK"}', 0) + 1
            status_counts[f'vw_{vw_status}'] = status_counts.get(f'vw_{vw_status}', 0) + 1
            rows.append({'month': m, 'group': int(g), 'n': n, 'ew': ew, 'vw': vw, 'vw_status': vw_status})
    ts = pd.DataFrame(rows)
    hi, lo = float(n_groups), 1.0
    out = {'n_groups': n_groups, 'portfolio_status_counts': status_counts,
           'group_mean_returns': {}, 'spread': {}}
    for g in sorted(ts['group'].unique()):
        s = ts[ts['group'] == g]
        out['group_mean_returns'][int(g)] = {
            'ew_mean': float(s['ew'].mean()), 'vw_mean': float(s['vw'].mean()),
            'n_months': int(s['month'].nunique()),
        }
    for wname in ['ew', 'vw']:
        p_hi = ts[ts['group'] == hi].set_index('month')[wname]
        p_lo = ts[ts['group'] == lo].set_index('month')[wname]
        d = (p_hi - p_lo).dropna()
        nw = newey_west(d.to_numpy(), lag=NW_LAG)
        spread = {'mean': nw['mean'], 'se_nw': nw['se_nw'], 't_nw': nw['t_nw'],
                  'n_months': nw['T'],
                  'p_one_sided_negative': one_sided_p(nw['t_nw'], 'negative'),
                  'p_two_sided': two_sided_p(nw['t_nw']),
                  'high_minus_low': 'group_high - group_low',
                  'vs_K_Spread_M1': {'abs_spread': None if nw['mean'] is None else abs(nw['mean']),
                                     'K_Spread_M1': K_SPREAD}}
        if nw['mean'] is not None:
            spread['vs_K_Spread_M1']['pass'] = bool(abs(nw['mean']) >= K_SPREAD)
        out['spread'][wname] = spread
    out['monthly_spread_series'] = {
        'month': ts[ts['group'] == hi]['month'].tolist(),
    }
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='A8：M1 组合分位（五/十分位）与函数形式诊断')
    ap.add_argument('--sample-version', choices=['v1', 'v2', 'v3', 'v4'], default='v2',
                    help='headline 样本版本：v2（默认，m1_s5_v2_2015_2023）/ v1 / v3 / v4（W2 创建）；'
                         '另一版本自动作敏感性并报')
    ap.add_argument('--out-suffix', default='',
                    help="产物文件名后缀，如 '_v2'（产出 s3_m1_portfolio_deciles_v2.json / "
                         "s3_m1_diagnostics_v2.json，不覆盖 v1 基线名；默认空 = 现有默认名）")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ver = args.sample_version
    other = {'v4': 'v3', 'v3': 'v2', 'v2': 'v1', 'v1': 'v2'}[ver]
    sfx = args.out_suffix
    ver_id = {'v1': M1_SAMPLE_VERSION_V1, 'v2': M1_SAMPLE_VERSION_V2,
              'v3': M1_SAMPLE_VERSION_V3, 'v4': M1_SAMPLE_VERSION_V4}[ver]
    other_id = {'v1': M1_SAMPLE_VERSION_V1, 'v2': M1_SAMPLE_VERSION_V2,
                'v3': M1_SAMPLE_VERSION_V3, 'v4': M1_SAMPLE_VERSION_V4}[other]
    progress(f'A8 开始：组合分位与函数形式诊断（headline={ver_id} + {other} 敏感性；'
             f'out_suffix={sfx or "(空)"}）')
    df_all = pd.read_parquet(sample_path('ma_teacher', ver))
    df = df_all[df_all['m1_in_fm_sample'].astype(bool)].copy()
    fm_window = [str(df['month'].min()), str(df['month'].max())]
    # treatment-only 宇宙（outcome-blind）池化中位/IQR —— 用于 (MAX-c)+ 的 c（事前固定口径）
    to = df_all[treatment_only_mask(df_all)]
    c_med = float(to['PeerMAXraw'].median())
    iqr_to = float(to['PeerMAXraw'].quantile(.75) - to['PeerMAXraw'].quantile(.25))
    progress(f'A8 treatment-only 宇宙：n={len(to):,} median={c_med:.6f} IQR={iqr_to:.8f}'
             f'（登记 IQR={IQR_POOL_REGISTERED:.8f}）；FM 窗口={fm_window[0]}..{fm_window[1]}')

    rep_pf = {'spec_id': 'm1_portfolio_q / m1_portfolio_d',
              'sample_version': ver_id,
              'headline_sample_version': ver_id,
              'sample_version_v1_superseded': M1_SAMPLE_VERSION_V1,
              'sample_version_v2': M1_SAMPLE_VERSION_V2,
              'out_suffix': sfx,
              'fm_window': fm_window, 'n_rows_fm': int(len(df)),
              'window_note': ('组合表按 headline 样本（主口径 ma_teacher）构造；'
                              f'本版本 FM 窗口 = {fm_window[0]}..{fm_window[1]}（{df["month"].nunique()} 月）。'
                              'v1（未施加 microcap 筛除）敏感性块另有其自身窗口，两版窗口与 N 不可混读。'),
              'tie_rule_id': P['portfolio_tie_rule_id'], 'K_Spread_M1': K_SPREAD,
              'headline_spec': 'continuous PeerMAXraw（组合仅作 supporting evidence）',
              'column_role': '本表按**主口径**（ma_teacher＝中文法 × 老师 MD&A，主表列）构造；'
                             '方法稳健性（mb_teacher）/ 文本稳健性（ma）/ 语料对照（mb）四列在主规格 FM 表中并报'
                             '（s3_m1_pricing_table.json / s3_m1_robustness_placebo.json 的 m1_rob_col_*）',
              'outcome': 'raw（R_NEXT_MONTH_COMPLETE_v1）为主；后复权（R_NEXT_POST_ADJ_v1）敏感性见 '
                         's3_m1_pricing_table.json:post_adjusted 与 s3_m1_robustness_placebo.json:m1_rob_postadj',
              'quintile': portfolio_table(df, 5), 'decile': portfolio_table(df, 10)}
    # 另一版本敏感性（v1 = 未施加 microcap 筛除）
    df1_all = pd.read_parquet(sample_path('ma_teacher', other))
    df1 = df1_all[df1_all['m1_in_fm_sample'].astype(bool)].copy()
    rep_pf[f'{other}_sensitivity'] = {
        'status': f'{other}_superseded_registered_deviation'
                  + ('（未施加 microcap/次新股筛除）' if other == 'v1' else ''),
        'sample_version': other_id,
        'fm_window': [str(df1['month'].min()), str(df1['month'].max())],
        'quintile': portfolio_table(df1, 5),
        'n_rows': int(len(df1)),
    }
    write_json(OUT_DIR / f's3_m1_portfolio_deciles{sfx}.json', rep_pf)
    progress(f"A8 五分位 EW spread = {rep_pf['quintile']['spread']['ew']['mean']:.5f} "
             f"(t={rep_pf['quintile']['spread']['ew']['t_nw']:.2f})；"
             f"{other} 五分位 EW spread = {rep_pf[f'{other}_sensitivity']['quintile']['spread']['ew']['mean']:.5f}")

    # ---------------- 形状诊断
    d = df.copy()
    diag = {'sample_version': ver_id, 'headline_sample_version': ver_id,
            'out_suffix': sfx, 'fm_window': fm_window, 'n_rows_fm': int(len(df)),
            'headline': 'continuous PeerMAXraw',
            'column_role': '诊断按主口径（ma_teacher）构造；四列并报见 s3_m1_pricing_table.json',
            'outcome': 'raw 主口径（后复权敏感性见 s3_m1_pricing_table.json:post_adjusted）',
            'kink_threshold_c': c_med, 'kink_rule': '(MAX-c)+ 的 c = 池化中位数（treatment-only 宇宙，'
                                                   'outcome-blind 事前固定口径；2026-10-10 复核修复 I4 同批统一）',
            'kink_threshold_scope': {'treatment_only_n': int(len(to)),
                                     'treatment_only_median': c_med,
                                     'treatment_only_iqr': iqr_to,
                                     'iqr_pool_registered': IQR_POOL_REGISTERED},
            'bins': {}, 'rcs': {}, 'kink': {}, 'double_sort_diagnostic': {}}

    # bins：十分位 dummy（d2..d10，基准 d1）
    d['_g'] = d.groupby('month', group_keys=False)['PeerMAXraw'].apply(
        lambda s: rank_average_groups(s, 10))
    for g in range(2, 11):
        d[f'd{g}'] = (d['_g'] == g).astype(float)
    dum = [f'd{g}' for g in range(2, 11)]
    res_bins, pm_bins = fm_regression(d, 'R_next', BASE_X + dum, min_n=MIN_N, nw_lag=NW_LAG)
    vecs = []
    for row in pm_bins:
        if row.get('used'):
            vecs.append([row['beta'][c] for c in BASE_X + dum + ['const']])
    diag['bins'] = {
        'x': BASE_X + dum,
        'coefs': {c: res_bins['coefs'][c] for c in dum},
        'joint_test_decile_dummies': joint_wald(vecs, list(range(len(BASE_X), len(BASE_X) + len(dum))), lag=NW_LAG),
        'n_months_used': res_bins['n_months_used'],
        'note': '基准组 = 十分位 1（最低 PeerMAXraw）；dummy 只解释形状，不替代 headline',
    }
    progress('A8 bins 完成')

    # RCS：3 / 5 节点（Harrell 分位）
    for k, kn in [(3, [0.10, 0.50, 0.90]), (5, [0.05, 0.275, 0.50, 0.725, 0.95])]:
        knots = np.quantile(d['PeerMAXraw'].to_numpy(dtype=float), kn)
        B = rcs_basis(d['PeerMAXraw'].to_numpy(dtype=float), knots)
        cols = [f'rcs{j}' for j in range(B.shape[1])]
        dd = d.copy()
        for j, c in enumerate(cols):
            dd[c] = B[:, j]
        res, pm = fm_regression(dd, 'R_next', BASE_X + cols, min_n=MIN_N, nw_lag=NW_LAG)
        vecs = [[row['beta'][c] for c in BASE_X + cols + ['const']] for row in pm if row.get('used')]
        diag['rcs'][f'knots_{k}'] = {
            'knots': knots.tolist(), 'knot_quantiles': kn, 'x': cols,
            'coefs': {c: res['coefs'][c] for c in cols},
            'joint_test_nonlinear_terms': joint_wald(vecs, list(range(len(BASE_X) + 1, len(BASE_X) + len(cols))), lag=NW_LAG),
            'n_months_used': res['n_months_used'],
        }
        progress(f'A8 RCS knots={k} 完成')

    # (MAX-c)+
    d['kink_pos'] = np.maximum(d['PeerMAXraw'] - c_med, 0.0)
    res_k, _ = fm_regression(d, 'R_next', ['PeerMAXraw', 'kink_pos'] + BASE_X, min_n=MIN_N, nw_lag=NW_LAG)
    diag['kink'] = {'c': c_med, 'coefs': res_k['coefs'], 'n_months_used': res_k['n_months_used'],
                    'note': '仅解释非线性；不得替代 headline 连续规格'}

    # double sort（描述性）：按 OwnMAXraw 与 PeerMAXraw 的中位数 2×2
    d['peer_hi'] = d.groupby('month')['PeerMAXraw'].transform(lambda s: s > s.median())
    d['own_hi'] = d.groupby('month')['OwnMAXraw'].transform(lambda s: s > s.median())
    cell = (d.groupby(['peer_hi', 'own_hi'])['R_next'].agg(['mean', 'count'])
            .reset_index().rename(columns={'mean': 'ew_mean', 'count': 'n_cell'}))
    diag['double_sort_diagnostic'] = {
        'cells': [{'peer_hi': bool(r.peer_hi), 'own_hi': bool(r.own_hi),
                   'ew_mean': float(r.ew_mean), 'n': int(r.n_cell)} for r in cell.itertuples()],
        'role': 'descriptive only（检验结果是否独立于 OwnMAXraw/PeerReturn 的诊断，不替代 headline）',
    }
    write_json(OUT_DIR / f's3_m1_diagnostics{sfx}.json', diag)
    progress(f'A8 完成：s3_m1_portfolio_deciles{sfx}.json / s3_m1_diagnostics{sfx}.json 落盘'
             f'（headline={ver_id}）')


if __name__ == '__main__':
    main()
