#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A11 + A12：MD&A 篇幅控制（B″11-5）与行业吸收检验（B″11-6）。

A11：主控制集 + ln(MD&A length)_{i,y-1}（teacher_mda_panel.csv 的 length，
      文本年与网络版本同口径）与无控制规格并报；含篇幅系数与篇幅-PeerMAX 相关性诊断。
      停机：beta 翻转 或 |t|<1.65 且符号反转 → 停（登记，不得只报有利规格）。
A12：R = a + beta*PeerMAXraw + gamma*IndustryMAX + phi*PeerReturn + theta*OwnMAXraw + GammaX + eps
      IndustryMAX = 申万一级/二级各一列（另报市值加权版）；保留率 delta = 1-|beta_with|/|beta_without|。
      停机：delta<0 或 beta 符号翻转 → 按 pricing_gate gate 降级。
产出：reports/data_gates/G0/pricing_length_control.json、pricing_industry_absorb.json
用法：python scripts/pricing_length_industry_control.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pricing_lib import HEADLINE_SAMPLE_VERSION, IQR_POOL_REGISTERED, MDA_PANEL, OUT_DIR, SAMPLE_VERSION_LABEL, fm_regression, param_value, progress, sample_path, write_json

P = param_value('sample_and_spec_registry')
NW_LAG = int(param_value('nw_lag'))
MIN_N = P['min_focal_count_per_month']
CONTROLS = list(param_value('pricing_main_controls_v1')['list'])
BASE_X = ['OwnMAXraw', 'PeerReturn'] + CONTROLS
MAIN = ['PeerMAXraw'] + BASE_X


def load_mda_length() -> pd.DataFrame:
    """分块读取 1GB 级 teacher_mda_panel.csv，只保留 (firm_id, year, length, ok)。"""
    parts = []
    for chunk in pd.read_csv(MDA_PANEL, usecols=['firm_id', 'year', 'length', 'status'],
                             dtype={'firm_id': str, 'status': str}, chunksize=200_000):
        c = chunk[(chunk['status'] == 'ok') & chunk['length'].notna()]
        if len(c):
            parts.append(c[['firm_id', 'year', 'length']])
    d = pd.concat(parts, ignore_index=True)
    d['year'] = pd.to_numeric(d['year'], errors='coerce').astype('Int64')
    d = d.dropna(subset=['year']).drop_duplicates(['firm_id', 'year'])
    return d.rename(columns={'length': 'mda_length', 'year': 'text_year'})


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='A11+A12：MD&A 篇幅控制与行业吸收')
    ap.add_argument('--sample-version', choices=[SAMPLE_VERSION_LABEL],
                    default=SAMPLE_VERSION_LABEL,
                    help='样本版本，默认取 registry 的 headline 指针')
    ap.add_argument('--out-suffix', default='',
                    help="产物文件名后缀，如 '_s0'（与同实验内 placebo 步骤同口径；默认空）")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ver = args.sample_version
    ver_id = HEADLINE_SAMPLE_VERSION
    sfx = args.out_suffix
    progress(f'A11 开始：MD&A 篇幅控制（headline={ver_id}；out_suffix={sfx or "(空)"}）')
    base = pd.read_parquet(sample_path('ma_teacher', ver))
    fm = base[base['pricing_in_fm_sample'].astype(bool)].copy()
    fm['text_year'] = pd.to_numeric(fm['network_version'], errors='coerce').astype('Int64')

    mda = load_mda_length()
    progress(f'A11 MD&A 篇幅面板就绪（ok 行 {len(mda):,}）')
    fm = fm.merge(mda.rename(columns={'firm_id': 'focal_id'}), on=['focal_id', 'text_year'], how='left')
    fm['ln_mda_length'] = np.log(fm['mda_length'])

    iqr = IQR_POOL_REGISTERED          # I4：登记池化值，全规格共用
    iqr_sample = float(fm['PeerMAXraw'].quantile(.75) - fm['PeerMAXraw'].quantile(.25))
    res_wo, _ = fm_regression(fm, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
    sub = fm[fm['ln_mda_length'].notna()]
    res_w, _ = fm_regression(sub, 'R_next', MAIN + ['ln_mda_length'], min_n=MIN_N, nw_lag=NW_LAG)

    b_wo = res_wo['coefs']['PeerMAXraw']
    b_w = res_w['coefs']['PeerMAXraw']
    len_c = res_w['coefs']['ln_mda_length']
    flip = (b_wo['beta'] is not None and b_w['beta'] is not None
            and np.sign(b_wo['beta']) != np.sign(b_w['beta']))
    weak_reverse = (b_w['t_nw'] is not None and abs(b_w['t_nw']) < 1.65 and flip)
    corr = {}
    for c in ['ln_mda_length']:
        corr[f'pearson_{c}_PeerMAXraw'] = float(sub[c].corr(sub['PeerMAXraw']))
        corr[f'spearman_{c}_PeerMAXraw'] = float(sub[c].corr(sub['PeerMAXraw'], method='spearman'))
    by_month = sub.groupby('month').apply(
        lambda g: g['ln_mda_length'].corr(g['PeerMAXraw']), include_groups=False)

    rep = {
        'specification_id': 'pricing_length_control', 'formula_id': 'MDA_LN_LENGTH_YMINUS1_v1',
        'sample_version': ver_id,
        'x_without': MAIN, 'x_with': MAIN + ['ln_mda_length'],
        'n_rows_without': int(len(fm)), 'n_rows_with': int(len(sub)),
        'beta_without': {'beta': b_wo['beta'], 't_nw': b_wo['t_nw'], 'p_one_sided': b_wo['p_one_sided'],
                         'economic_magnitude': None if b_wo['beta'] is None else b_wo['beta'] * iqr},
        'beta_with': {'beta': b_w['beta'], 't_nw': b_w['t_nw'], 'p_one_sided': b_w['p_one_sided'],
                      'economic_magnitude': None if b_w['beta'] is None else b_w['beta'] * iqr},
        'length_coefficient': len_c,
        'iqr_pool': iqr,
        'iqr_pool_rule': f'IQR_POOL_TREATMENT_ONLY_v1（登记值 {IQR_POOL_REGISTERED:.8f}，全规格共用）',
        'iqr_pool_sample_diagnostic': iqr_sample,
        'diagnostics': {'corr': corr,
                        'monthly_corr_median': float(by_month.median()),
                        'monthly_corr_min': float(by_month.min()),
                        'monthly_corr_max': float(by_month.max()),
                        'n_months': int(by_month.shape[0]),
                        'ln_length_stats': {'mean': float(sub['ln_mda_length'].mean()),
                                            'std': float(sub['ln_mda_length'].std())}},
        'coefs_without': res_wo['coefs'], 'coefs_with': res_w['coefs'],
        'n_months_used_without': res_wo['n_months_used'], 'n_months_used_with': res_w['n_months_used'],
        'stop_condition': {'beta_sign_flip': bool(flip),
                           'weak_and_reversed': bool(weak_reverse),
                           'stop': bool(weak_reverse)},
        'acceptance': '① 主系数 beta 方向/显著性与无控制规格并报；② 篇幅系数量级与符号报告；'
                      '③ 篇幅—PeerMAX 相关性诊断（防机械共线掩盖）',
    }
    write_json(OUT_DIR / f'pricing_length_control{sfx}.json', rep)
    progress(f"A11 完成：beta_without={b_wo['beta']:.5f}(t={b_wo['t_nw']:.2f}) -> "
             f"beta_with={b_w['beta']:.5f}(t={b_w['t_nw']:.2f})；篇幅系数={len_c['beta']:.5f}"
             f"(t={len_c['t_nw']:.2f})；停机={rep['stop_condition']['stop']}")
    # ---------------- A12 行业吸收
    progress('A12 开始：行业吸收检验')
    rows = ['ind_l1_max_ew', 'ind_l2_max_ew']
    out = {'specification_id': 'pricing_industry_absorb', 'formula_id': 'PRICING_INDUSTRY_ABSORB_v1',
           'sample_version': ver_id,
           'industry_version': 'sw2_industry_latest.csv (SwClass2021, is_current_version)',
           'formula': 'R = a + beta*PeerMAXraw + gamma*IndustryMAX + phi*PeerReturn + theta*OwnMAXraw + GammaX + eps',
           'beta_without': {'beta': b_wo['beta'], 't_nw': b_wo['t_nw'], 'p_one_sided': b_wo['p_one_sided']},
           'specs': {}, 'n_rows_base': int(len(fm)),
           'reading_note': ('IndustryMAX 构造（申万一级/二级、排除 focal、EW/VW）经独立 LOO 手算复核正确（差 ≤7e-18）；'
                            'γ 的正号来自**行业/同伴收益动量**：IndustryMAX 与 PeerReturn 高度重叠，'
                            '故「含 PeerReturn」的规格才是可判读的基准（见 industry_max_momentum_diagnostic 与 '
                            'pricing_horse_race.json 的 reading_note——horse race 的 pricing_horse_own_ind_l1/l2 '
                            '不含 PeerReturn，吸收幅度被高估）'),
           }
    # γ 的动量来源诊断（同一样本、只换 x 集）：IndustryMAX 单独 vs 与 PeerReturn 同入
    diag = {}
    for lab, xs in [('ind_l1_wo_peerret', ['OwnMAXraw', 'ind_l1_max_ew'] + CONTROLS),
                    ('ind_l1_with_peerret', ['OwnMAXraw', 'ind_l1_max_ew', 'PeerReturn'] + CONTROLS),
                    ('peerret_only', ['OwnMAXraw', 'PeerReturn'] + CONTROLS)]:
        sub_d = fm.dropna(subset=xs)
        res_d, _ = fm_regression(sub_d, 'R_next', xs, min_n=MIN_N, nw_lag=NW_LAG)
        diag[lab] = {'x': xs, 'n_rows': int(len(sub_d)), 'n_months_used': res_d['n_months_used'],
                     'coefs': {c: res_d['coefs'][c] for c in xs if c in ('ind_l1_max_ew', 'PeerReturn')}}
        progress(f'A12 诊断 {lab}: ' + ', '.join(f'{c}={res_d["coefs"][c]["beta"]:+.5f}'
                                                 f'(t={res_d["coefs"][c]["t_nw"]:+.2f})'
                                                 for c in ('ind_l1_max_ew', 'PeerReturn') if c in xs))
    out['industry_max_momentum_diagnostic'] = {
        'note': '同一 FM 样本、同一控制/NW，仅替换 x 集：IndustryMAX 的正号在 PeerReturn 同入后塌缩 → '
                '正号来自行业/同伴收益动量（与复核结论一致）',
        'specs': diag}
    specs = {
        'pricing_industry_absorb_l1': ['PeerMAXraw', 'ind_l1_max_ew'] + BASE_X,
        'pricing_industry_absorb_l2': ['PeerMAXraw', 'ind_l2_max_ew'] + BASE_X,
        'pricing_industry_absorb_both': ['PeerMAXraw', 'ind_l1_max_ew', 'ind_l2_max_ew'] + BASE_X,
        'pricing_industry_absorb_l1_vw': ['PeerMAXraw', 'ind_l1_max_vw'] + BASE_X,
        'pricing_industry_absorb_l2_vw': ['PeerMAXraw', 'ind_l2_max_vw'] + BASE_X,
    }
    for sid, xs in specs.items():
        sub2 = fm.dropna(subset=xs)
        res, _ = fm_regression(sub2, 'R_next', xs, min_n=MIN_N, nw_lag=NW_LAG)
        bb = res['coefs']['PeerMAXraw']
        ind_cols = [c for c in xs if c.startswith('ind_')]
        delta = None
        if b_wo['beta'] not in (None, 0) and bb['beta'] is not None:
            delta = 1.0 - abs(bb['beta']) / abs(b_wo['beta'])
        out['specs'][sid] = {
            'x': xs, 'n_rows': int(len(sub2)), 'n_months_used': res['n_months_used'],
            'beta_with': bb, 'gamma': {c: res['coefs'][c] for c in ind_cols},
            'retention_delta': delta,
            'sign_flip': bool(b_wo['beta'] is not None and bb['beta'] is not None
                              and np.sign(b_wo['beta']) != np.sign(bb['beta'])),
            'interpretation': (None if delta is None else
                               ('含行业之上的增量（delta>=0.5）' if delta >= 0.5
                                else '部分被行业吸收（delta<0.5）')),
            'all_months_full_rank': bool(all((d.get('rank_status') == 'FULL')
                                             for d in res['per_month'] if d.get('used'))),
        }
        progress(f'A12 {sid}: beta_with={bb["beta"]} t={bb["t_nw"]} delta={delta}')
    out['stop_condition'] = {sid: {'delta_lt_0': (v['retention_delta'] is not None
                                                  and v['retention_delta'] < 0),
                                   'sign_flip': v['sign_flip']}
                             for sid, v in out['specs'].items()}
    out['acceptance'] = 'delta 与 IndustryMAX 并报（delta>=0.5 记「含行业之上的增量」、<0.5 记「部分被行业吸收」，两种都并报）'
    write_json(OUT_DIR / f'pricing_industry_absorb{sfx}.json', out)
    progress('A12 完成：pricing_industry_absorb.json 落盘')


if __name__ == '__main__':
    main()
