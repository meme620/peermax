#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A10：M1 gate report（M1-1…M1-5 逐条判定 + S5_status + 停机动作）。

gate 定义（registry：alpha_m1 / K_Econ_M1 / K_Spread_M1 / K_Consist_M1）：
  M1-1 符号：beta_FM < 0（主方向；EW 与 VW 并报）
  M1-2 单侧显著性：p_one_sided(beta) < alpha_m1
  M1-3 经济量级：|beta_FM * IQR_pool| >= K_Econ_M1（IQR_pool = treatment-only 宇宙单一**登记**池化值，全规格共用）
  M1-4 portfolio spread：|spread_H - spread_L| >= K_Spread_M1（headline=五分位 EW）
  M1-5 EW/VW 一致性：|beta_EW - beta_VW| <= K_Consist_M1 且符号一致、均显著、TOST 等价
并附硬性数值条件：逐月满秩 / 残差 df>0 / 协方差有限 / T_effective 与每月 N_t 达标 /
主 portfolio 组别无未处理空组。**仅显著 != PASS**（plan §5.7A P3.6）。

2026-10-10 复核修复批：headline 样本 = m1_s5_v2_2015_2023（I3 microcap/次新股筛除已施加）；
v1（未施加 microcap 筛除）gate 逐条重算并报（I4 统一登记 IQR；I5 VIF 作用域 + 跨规格最大值）。

产出：reports/data_gates/G0/s3_m1_gate_report{out_suffix}.{json,md}
用法：python scripts/pricing_gate_report.py [--out-suffix _v2] [--input-suffix _v2]

2026-10-11 F-8 命名归一化：新增 `--out-suffix`（产物名后缀，如 `_v2` → s3_m1_gate_report_v2.{json,md}，
不覆盖无后缀 legacy 名）与 `--input-suffix`（输入名后缀，如 `_v2` → 读 s3_m1_pricing_table_v2.json /
s3_m1_portfolio_deciles_v2.json / s3_m1_horse_race_v2.json / s3_m1_diagnostics_v2.json；缺 `_v2` 或
缺少所需键（如 `samples`）时**自动回退无后缀名**，逐条打印实际读取来源）。两个参数默认空 = 现有默认行为。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pricing_lib import IQR_POOL_REGISTERED, M1_SAMPLE_VERSION_V1, M1_SAMPLE_VERSION_V2, M1_SAMPLE_VERSION_V3, OUT_DIR, ROOT, assert_m1_contract, norm_cdf, param_value, params, progress, require, tost_equivalence, write_json

# 2026-10-27 Phase 0.1（S-1/S-4）：MDE/功效与 TOST 尺度说明只做解释，不参与任何 pass 判定；
# TOST equivalent 本身是 M1-5 AND 条件之一（此处只补充尺度解释，未改 pass 逻辑）。
Z_ONE_SIDED = 1.645 + 0.8416   # 单侧 alpha=0.05、power=0.80
Z_TWO_SIDED = 1.96 + 0.8416    # 双侧 alpha=0.05、power=0.80
TOST_Z = 1.645

ALPHA = param_value('alpha_m1')
K_ECON = param_value('K_Econ_M1')
K_SPREAD = param_value('K_Spread_M1')
K_CONSIST = param_value('K_Consist_M1')
NW_LAG = int(param_value('nw_lag'))
P = param_value('m1_sample_and_spec_registry')
MIN_N = P['min_focal_count_per_month']
MIN_MONTHS = P['min_effective_month_count']
COLUMNS = ['ma_teacher', 'mb_teacher', 'ma', 'mb']


def load(name, suffix='', required_key=None):
    """读产物 JSON。suffix 非空时优先读 <stem><suffix>.json；文件缺失或缺 required_key 时回退无后缀名。

    返回 (obj, 实际读取文件名) —— 实际文件名仅用于 stdout 溯源打印，不写入报告内容。
    """
    stem, _, ext = name.rpartition('.')
    cands = ([f'{stem}{suffix}.{ext}'] if suffix else []) + [name]
    for c in cands:
        p = OUT_DIR / c
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding='utf-8'))
        if required_key is not None and not (isinstance(d, dict) and required_key in d):
            print(f'  [load] {c} 缺 "{required_key}" → 回退', flush=True)
            continue
        return d, c
    return None, None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='A10：M1 gate report（M1-1…M1-5 逐条判定 + S5_status + 停机动作）')
    ap.add_argument('--out-suffix', default='',
                    help="产物文件名后缀，如 '_v2'（产出 s3_m1_gate_report_v2.{json,md}，不覆盖无后缀 legacy 名）")
    ap.add_argument('--input-suffix', default='',
                    help="输入文件名后缀，如 '_v2'（读 *_v2 产物；缺文件或缺所需键时自动回退无后缀名）")
    return ap.parse_args(argv)


def _mde_stats(se, threshold, scale=1.0):
    """MDE(80%) 与阈值处拒绝 beta=0 的功效（解释层；不参与 pass）。

    se 为 beta/收益量纲的 NW SE；scale=1 表示 se 已是月度收益量纲（M1-4 spread），
    scale=IQR_POOL_REGISTERED 表示先把 beta 尺度换算为 |beta*IQR| 收益量纲（M1-3）。
    `se_pct` 按复核文档沿用「se*scale」的小数收益量纲；`mde80_pct_*` 为 %/月。
    """
    keys = ('se_pct', 'mde80_pct_one_sided', 'mde80_pct_two_sided',
            'power_at_threshold_reject_zero_one_sided',
            'power_at_threshold_reject_zero_two_sided')
    if se is None or not np.isfinite(se):
        return {k: None for k in keys}
    se_ret = se * scale
    if se_ret <= 0:
        return {k: None for k in keys}
    return {
        'se_pct': se_ret,
        'mde80_pct_one_sided': Z_ONE_SIDED * se_ret * 100.0,
        'mde80_pct_two_sided': Z_TWO_SIDED * se_ret * 100.0,
        'power_at_threshold_reject_zero_one_sided': norm_cdf(threshold / se_ret - 1.645),
        'power_at_threshold_reject_zero_two_sided': norm_cdf(threshold / se_ret - 1.96),
    }


def _tost_scale(tost):
    """M1-5 TOST 的 beta→%/月尺度换算（尺度字段不参与 pass；TOST equivalent 本身是 M1-5 AND 条件之一）。"""
    se = tost.get('se_nw')
    keys = ('se_nw_diff', 'K_Consist_M1_beta', 'K_Consist_x_IQR_pct_per_month',
            'K_Econ_M1_pct_per_month', 'K_Consist_x_IQR_over_K_Econ',
            'tost_fail_bound_beta', 'tost_fail_bound_pct_per_month',
            'tost_fail_bound_over_K_Econ')
    if se is None or not np.isfinite(se) or se <= 0:
        return {k: None for k in keys}
    fail_beta = K_CONSIST - TOST_Z * se
    return {
        'se_nw_diff': se,
        'K_Consist_M1_beta': K_CONSIST,
        'K_Consist_x_IQR_pct_per_month': K_CONSIST * IQR_POOL_REGISTERED * 100.0,
        'K_Econ_M1_pct_per_month': K_ECON * 100.0,
        'K_Consist_x_IQR_over_K_Econ': (K_CONSIST * IQR_POOL_REGISTERED) / K_ECON,
        'tost_fail_bound_beta': fail_beta,
        'tost_fail_bound_pct_per_month': fail_beta * IQR_POOL_REGISTERED * 100.0,
        'tost_fail_bound_over_K_Econ': (fail_beta * IQR_POOL_REGISTERED) / K_ECON,
    }


def judge(block, spread_block, series, cols_status):
    """对给定样本版本重算 M1-1…M1-5（同一定义、同一阈值、同一 IQR 登记值）。"""
    ew, vw = block['ew']['coefs']['PeerMAXraw'], block['vw']['coefs']['PeerMAXraw']
    b_ew, b_vw = ew['beta'], vw['beta']
    econ = None if b_ew is None else b_ew * IQR_POOL_REGISTERED
    d_series = [a - b for a, b in zip(series['beta_series_ew'], series['beta_series_vw'])]
    tost = tost_equivalence(d_series, bound=K_CONSIST, alpha=ALPHA, lag=NW_LAG)
    sign_same = (b_ew is not None and b_vw is not None and np.sign(b_ew) == np.sign(b_vw))
    both_sig = ((ew['p_one_sided'] is not None and ew['p_one_sided'] < ALPHA)
                and (vw['p_one_sided'] is not None and vw['p_one_sided'] < ALPHA))
    diff = None if (b_ew is None or b_vw is None) else b_ew - b_vw
    sp = (spread_block or {}).get('spread', {}).get('ew', {})
    sp_vw = (spread_block or {}).get('spread', {}).get('vw', {})
    se_ew, se_vw = ew.get('se_nw'), vw.get('se_nw')
    se_sp_ew, se_sp_vw = sp.get('se_nw'), sp_vw.get('se_nw')
    mde_ew = _mde_stats(se_ew, K_ECON, IQR_POOL_REGISTERED)
    mde_vw = _mde_stats(se_vw, K_ECON, IQR_POOL_REGISTERED)
    mde_sp_ew = _mde_stats(se_sp_ew, K_SPREAD, 1.0)
    mde_sp_vw = _mde_stats(se_sp_vw, K_SPREAD, 1.0)
    g = {
        'M1-1_sign': {'expected': 'beta_FM < 0', 'beta_ew': b_ew, 'beta_vw': b_vw,
                      'pass': bool(b_ew is not None and b_ew < 0 and b_vw is not None and b_vw < 0)},
        'M1-2_one_sided_significance': {
            'threshold': f'p_one_sided < {ALPHA}', 'p_one_sided_ew': ew['p_one_sided'],
            'p_one_sided_vw': vw['p_one_sided'], 't_nw_ew': ew['t_nw'], 't_nw_vw': vw['t_nw'],
            'pass': bool(both_sig)},
        'M1-3_economic_magnitude': {
            'threshold': f'|beta*IQR| >= {K_ECON}', 'beta_ew': b_ew,
            'iqr_pool': IQR_POOL_REGISTERED, 'iqr_pool_rule_id': P['iqr_scale_rule_id'],
            'economic_magnitude': econ,
            'iqr_pool_sample_diagnostic': block.get('iqr_pool_sample_diagnostic'),
            'se_nw_ew': se_ew, 'se_nw_vw': se_vw,
            'mde80_beta_one_sided': (None if se_ew is None else Z_ONE_SIDED * se_ew),
            **mde_ew,
            'mde80_vw': mde_vw,
            'mde_note': ('MDE 与功效为解释层，不参与 pass。se_pct = se_beta*IQR（小数收益量纲）；'
                         'mde80_pct_* 为 %/月；阈值处功效是真实效应等于阈值时拒绝 beta=0 的概率，'
                         '不是「过 gate」的概率（gate 用 abs() 且无 alpha）。'),
            'pass': bool(econ is not None and abs(econ) >= K_ECON)},
        'M1-4_portfolio_spread': {
            'threshold': f'|spread_H-spread_L| >= {K_SPREAD}', 'headline': 'quintile EW',
            'spread_ew': sp.get('mean'), 't_nw': sp.get('t_nw'), 'n_months': sp.get('n_months'),
            'se_nw': se_sp_ew, **mde_sp_ew,
            'spread_vw': sp_vw.get('mean'), 't_nw_vw': sp_vw.get('t_nw'),
            'se_nw_vw': se_sp_vw, 'mde80_vw_spread': mde_sp_vw,
            'pass_vw': bool(sp_vw.get('mean') is not None and abs(sp_vw['mean']) >= K_SPREAD),
            'pass_vw_note': ('VW 五分位 spread 并报仅为诊断（同一 abs() 规则），不参与 headline pass；'
                             'headline 登记为 EW，不得事后改用 VW 救 M1。'),
            'portfolio_status_counts': (spread_block or {}).get('portfolio_status_counts'),
            'pass': bool(sp.get('mean') is not None and abs(sp['mean']) >= K_SPREAD)},
        'M1-5_ew_vw_consistency': {
            'threshold': f'|beta_EW-beta_VW| <= {K_CONSIST} 且符号一致、均显著、TOST 等价',
            'beta_diff': diff, 'sign_same': bool(sign_same), 'both_significant': bool(both_sig),
            'tost': tost, 'tost_scale': _tost_scale(tost),
            'pass': bool(diff is not None and abs(diff) <= K_CONSIST and sign_same and both_sig
                         and tost.get('equivalent') is True)},
    }
    fm_ok = bool(cols_status['all_months_full_rank'] and cols_status['all_months_residual_df_positive']
                 and cols_status['all_months_covariance_finite']) if cols_status else None
    return g, {'FM_status': ('PASS' if fm_ok else ('FAIL' if fm_ok is False else 'NOT_EVALUATED')),
               'n_months': block['ew']['n_months_used'],
               'n_rows': block['n_rows'], 'fm_window': block.get('fm_window'),
               'T_effective_ew': block['ew']['n_months_used'], 'T_effective_vw': block['vw']['n_months_used'],
               'min_focal_count_per_month': block['per_month_n']['min']}


def check_sample_version(price, sample, where: str = 's3_m1_6') -> tuple:
    """Phase 1 I-2：pricing/sample 的 sample_version 必须存在且一致；失败 SystemExit(2)。"""
    require(price is not None, f'{where}: pricing table 缺失（s3_m1_pricing_table*.json）')
    require(sample is not None,
            f'{where}: 缺少 sample report（s3_m1_sample_report*.json），无法核验四条硬门')
    price_version = price.get('sample_version')
    sample_version = sample.get('sample_version')
    require(price_version is not None, f'{where}: pricing table 缺 sample_version')
    require(sample_version is not None, f'{where}: sample report 缺 sample_version')
    require(price_version == sample_version,
            f'{where}: pricing sample_version={price_version!r} != '
            f'sample sample_version={sample_version!r}')
    return price_version, sample_version


def main(argv=None):
    args = parse_args(argv)
    sfx = args.out_suffix
    isfx = args.input_suffix
    progress(f'A10 开始：M1 gate report（v3 headline + 前一版本 gate 重算 + I4/I5 修正；'
             f'out_suffix={sfx or "(空)"} / input_suffix={isfx or "(空)"}）')
    price, price_src = load('s3_m1_pricing_table.json', isfx, required_key='main')
    sample, sample_src = load('s3_m1_sample_report.json', isfx, required_key='samples')
    pf, pf_src = load('s3_m1_portfolio_deciles.json', isfx, required_key='quintile')
    diag, diag_src = load('s3_m1_diagnostics.json', isfx)
    hr, hr_src = load('s3_m1_horse_race.json', isfx, required_key='specs')
    rob, rob_src = load('s3_m1_robustness_placebo.json', isfx, required_key='specs')
    lc, lc_src = load('s3_m1_length_control.json', isfx, required_key='stop_condition')
    ia, ia_src = load('s3_m1_industry_absorb.json', isfx, required_key='stop_condition')
    for nm, src in (('pricing_table', price_src), ('sample_report', sample_src),
                    ('portfolio_deciles', pf_src), ('diagnostics', diag_src), ('horse_race', hr_src),
                    ('robustness_placebo', rob_src), ('length_control', lc_src), ('industry_absorb', ia_src)):
        print(f'  [load] {nm}: {src}', flush=True)

    # Phase 1 I-2：版本一致性与四条输入硬门在下方 price 缺失检查之后、任何 gate 判定之前执行。
    _samp = (sample or {}).get('samples', {}).get('ma_teacher') or {}
    _n_fm = _samp.get('fm_sample_rows')
    _n_months = _samp.get('n_months_eligible')
    if _n_months is None:
        _fm_months = _samp.get('fm_months') or []
        _n_months = len(_fm_months) if len(_fm_months) > 1 else _fm_months[0] if _fm_months else None
    if isinstance(_n_fm, (int, float)) and _n_months is not None:
        _universe_note = (
            f'M1 主样本（headline，sample_version = {M1_SAMPLE_VERSION_V3}）= 同时有 AnalystCoverage 与 '
            f'NetworkDegree、且下月 complete_month=1（有效交易日 >=15）的 {_n_fm:,} 个 focal-month / '
            f'{_n_months} 个月（2015-05..2023-12），占 M1 面板 307,503 行的 '
            f'{100.0 * _n_fm / 307_503:.1f}%；control_missing 剔除 80,842 行（AnalystCoverage 缺失 '
            '80,725 行、全样本缺失率 35.12%；NetworkDegree 缺失 80,842 行、全样本缺失率 35.17%）。'
            '该样本即「分析师覆盖宇宙」的严格口径；全 A / 无分析师屏幕样本仅作敏感性，不进入 headline，'
            '也不得写成「全 A 主样本」。')
    else:
        _universe_note = (f'M1 主样本 headline = {M1_SAMPLE_VERSION_V3}；本输入 report 未提供 '
                          'fm_sample_rows/fm_months（契约/版本门将在下方独立执行）。')
    rep = {'spec_id': 'm1_gate_report_v1', 'sample_version': M1_SAMPLE_VERSION_V3,
           # Phase 0.3（2026-10-27）：V-P1-6 / N-7 口径注记（append-only；不改任何判定/数值）
           'sample_universe_note': _universe_note,
           'control_definition_note': (
               'NetworkDegree = S3A 共同分析师覆盖网络中的度数（firms sharing at least one analyst；来源 '
               'data/processed/analyst/firm_analyst_derived.csv，边表 '
               'data/processed/analyst/edges/edges_coverage_win12_fy12_v3_{cutoff}.csv 的 '
               'common_analyst_count），不是 M-A 产品市场网络的度数；产品网络处理变量自己的度数 n_sources '
               '未进入主控制集 m1_main_controls_v1（本次不改控制集、不重跑回归）。M1 FM 181,178 行分布：'
               'NetworkDegree p10/p50/p75/p90 = 40/174/302/449（缺失 0）；n_sources = 4/51/111/193'
               '（min/max 1/285；全 panel n_sources>0 的 296,884 行为 3/49/114/197）。二者相等率 0.238%。'
               'n_sources 仅作分布报告，不加入控制集。'),
           'sample_version_v1_superseded': M1_SAMPLE_VERSION_V1,
           'sample_version_v2_superseded': M1_SAMPLE_VERSION_V2,
           'sample_version_v3': M1_SAMPLE_VERSION_V3,
           'phase': 'P1-01（2026-11-15：complete_month/n_valid_days = volume>0 交易日 >=15）',
           'out_suffix': sfx,
           'naming_note': (f'canonical 命名件（out_suffix={sfx}）；无后缀 s3_m1_gate_report.{{json,md}} 为 '
                           'legacy 文件名（v2 headline），一切以本件为准。' if sfx else
                           'legacy 无后缀命名；内容为 v2 headline，canonical 见 s3_m1_gate_report_v2.{json,md}。'),
           'gate_params': {'alpha_m1': ALPHA, 'K_Econ_M1': K_ECON, 'K_Spread_M1': K_SPREAD,
                           'K_Consist_M1': K_CONSIST, 'nw_lag': NW_LAG,
                           'min_focal_count_per_month': MIN_N,
                           'min_effective_month_count': MIN_MONTHS,
                           'portfolio_tie_rule_id': P['portfolio_tie_rule_id'],
                           'iqr_scale_rule_id': P['iqr_scale_rule_id'],
                           'iqr_pool_registered': IQR_POOL_REGISTERED,
                           'tost_rule_id': P['tost_rule_id'],
                           'mde_power_note': ('MDE80 单侧 = (1.645+0.8416)*se；双侧 = (1.96+0.8416)*se；'
                                              '阈值处功效 = Phi(K/se - z_alpha)；均为解释层，不参与 pass。')},
           'gates': {}, 'supporting': {}, 'four_column_table': {},
           'v2_sensitivity': {},
           'per_month_diagnostics_ref': None,
           'main_column': 'ma_teacher（主口径，主表列）',
           'outcome': 'raw（R_NEXT_MONTH_COMPLETE_v1）为主；后复权（R_NEXT_POST_ADJ_v1）为敏感性',
           'fix_batch': {
               'I1': 'winsorization 口径对称：权重变体/placebo 的 treatment 与 PeerReturn 施加同规则月度 '
                     '[0.01,0.99] clip（修正前未 winsorize：weight_linear +0.04363 → 修正后 ≈ 主规格；'
                     'placebo random −0.0478 → −0.0680）',
               'I2': 'sample 窗口统一：四列事件月裁剪到 2015-01..2023-12（旧 ma/mb 含 2024-01 → superseded）',
               'I3': 'microcap 筛除已实现（形成月全市场横截面 q10，总市值口径=登记偏离）+ 次新股 <6 月筛除；'
                     '新样本版本 m1_s5_v2_2015_2023（v1 = superseded_registered_deviation 敏感性）',
               'I4': 'IQR_pool 统一为 treatment-only 宇宙登记池化值 0.02341287（全规格共用；'
                     '原实现用终样本/各列 IQR）',
               'I5': 'VIF/condition number 作用域注明并并报跨规格最大值（见 vif_scope）',
               'I6': '退化行写 spec_status=NOT_COMPUTABLE + reason；样本报告补新股/停牌状态字段；'
                     'ST 分层代码格式映射缺陷修复（st_flag_focal 原恒为 False）',
           },
           'four_column_roles': {'ma_teacher': '主口径（主表列）', 'mb_teacher': '方法稳健性列',
                                 'ma': '文本稳健性列', 'mb': '语料对照列（仅历史对照）'}}

    if price is None:
        rep['S5_status'] = 'BLOCKED'
        rep['reason'] = 's3_m1_pricing_table.json 缺失'
        write_json(OUT_DIR / f's3_m1_gate_report{sfx}.json', rep)
        return

    # Phase 1 I-2：版本不一致 → 退出且不写 gate 报告；契约失败 → 写 contracts_status=FAIL 的
    # blocked 报告后退出（不把 gate FAIL 结果与输入契约失败混为一谈）。
    check_sample_version(price, sample, where='s3_m1_6')
    contracts_status = 'PASS'
    try:
        assert_m1_contract(sample, where='s3_m1_6:sample_report')
    except SystemExit:
        contracts_status = 'FAIL'
        write_json(OUT_DIR / f's3_m1_gate_report{sfx}.json', {
            'spec_id': 'm1_gate_report_v1',
            'sample_version': (sample or {}).get('sample_version'),
            'contracts_status': 'FAIL',
            'S5_status': 'BLOCKED',
            'status': 'BLOCKED',
            'reason': 'M1 输入契约四条硬门失败（[HARD-GATE] 见 stderr）；未做 gate 判定',
            'out_suffix': sfx, 'input_suffix': isfx,
        })
        raise
    rep['contracts_status'] = contracts_status

    main_col = 'ma_teacher'
    cols_status = price['main'][main_col]
    gates, support = judge(price['main'][main_col], pf['quintile'],
                           price['per_month'][main_col], cols_status)
    rep['gates'] = gates
    sp = pf['quintile']['spread']
    pf_status = pf['quintile']['portfolio_status_counts']
    no_empty = all(('EMPTY' not in k) or v == 0 for k, v in pf_status.items())
    coverage_ok = bool(gates['M1-2_one_sided_significance']['t_nw_ew'] is not None
                       and support['T_effective_ew'] >= MIN_MONTHS
                       and support['T_effective_vw'] >= MIN_MONTHS
                       and support['min_focal_count_per_month'] >= MIN_N)
    rep['supporting'] = {
        'FM_status': support['FM_status'],
        'all_months_full_rank': cols_status['all_months_full_rank'],
        'all_months_residual_df_positive': cols_status['all_months_residual_df_positive'],
        'all_months_covariance_finite': cols_status['all_months_covariance_finite'],
        'T_effective_ew': support['T_effective_ew'], 'T_effective_vw': support['T_effective_vw'],
        'min_effective_month_count': MIN_MONTHS, 'coverage_ok': coverage_ok,
        'min_focal_count_per_month': support['min_focal_count_per_month'],
        'portfolio_no_unhandled_empty_group': bool(no_empty),
        'vif_main_scope': '主规格（ma_teacher）per-month max',
        'vif_max_main_spec': price['vif_scope']['scope_main_spec_only']['vif_per_month_max'],
        'vif_max_main_spec_variable': price['vif_scope']['scope_main_spec_only']['vif_per_month_max_variable'],
        'condition_number_max_main_spec': price['vif_scope']['scope_main_spec_only']['condition_number_max'],
        'vif_scope': price['vif_scope'],
        'sample': (sample or {}).get('samples', {}).get(main_col),
        'horse_race_available': hr is not None,
        'robustness_available': rob is not None,
        'length_control_stop': None if lc is None else lc['stop_condition']['stop'],
        'industry_absorb_stop': None if ia is None else
            any(v['delta_lt_0'] or v['sign_flip'] for v in ia['stop_condition'].values()),
    }

    # ---------------- 前一版本敏感性：同一定义重算 5 条 gate
    sens_key = 'v2_sensitivity' if 'v2_sensitivity' in price else 'v1_sensitivity'
    prev_ver = M1_SAMPLE_VERSION_V2 if sens_key == 'v2_sensitivity' else M1_SAMPLE_VERSION_V1
    v1_block = price[sens_key][main_col]
    v1_spread = ((pf or {}).get(sens_key) or {}).get('quintile', {})
    g1, s1 = judge(v1_block, v1_spread, v1_block['per_month_series'], None)
    rep[sens_key] = {
        'status': (f'{prev_ver}（superseded；旧 complete_month 口径）'
                   if sens_key == 'v2_sensitivity' else
                   'v1_superseded_registered_deviation（未施加 microcap/次新股筛除）'),
        'sample_version': prev_ver,
        'gates': g1,
        'summary': s1,
        'note': ('前一版本与 v3 的 gate 判定逐条同向（均 FAIL）；差异仅来自 P1-01 完整月口径修正 '
                 '→ 结论不因语义修复而改变' if sens_key == 'v2_sensitivity' else
                 'v1 与 v2 的 gate 判定逐条同向（均 FAIL）；差异仅来自 microcap（与次新股）筛除 '
                 '→ 结论不因 I3 修复而改变'),
    }

    # ---------------- 四列并报（同一行）
    prev_blk = price[sens_key]
    prev_prefix = 'v2' if sens_key == 'v2_sensitivity' else 'v1'
    for k in COLUMNS:
        m = price['main'][k]
        rep['four_column_table'][k] = {
            'role': m['role'], 'n_rows': m['n_rows'], 'n_months': m['ew']['n_months_used'],
            'fm_window': m['fm_window'],
            'beta_ew': m['ew']['coefs']['PeerMAXraw']['beta'],
            't_nw_ew': m['ew']['coefs']['PeerMAXraw']['t_nw'],
            'p_one_sided_ew': m['ew']['coefs']['PeerMAXraw']['p_one_sided'],
            'beta_vw': m['vw']['coefs']['PeerMAXraw']['beta'],
            't_nw_vw': m['vw']['coefs']['PeerMAXraw']['t_nw'],
            'iqr_pool': m['iqr_pool'],
            'economic_magnitude': m['ew']['economic_magnitude_beta_x_iqr'],
            f'{prev_prefix}_beta_ew': prev_blk[k]['ew']['coefs']['PeerMAXraw']['beta'],
            f'{prev_prefix}_beta_vw': prev_blk[k]['vw']['coefs']['PeerMAXraw']['beta'],
            f'{prev_prefix}_n_months': prev_blk[k]['ew']['n_months_used'],
        }

    gate_pass = [v['pass'] for v in rep['gates'].values()]
    all_ok = all(gate_pass) and support['FM_status'] == 'PASS' and coverage_ok and bool(no_empty)
    rep['S5_status'] = 'PASS' if all_ok else 'FAIL'

    # ---------------- 2026-10-27 Phase 0.1：MDE/功效与 TOST 尺度（解释层；不参与 pass）
    g3 = gates['M1-3_economic_magnitude']
    g4 = gates['M1-4_portfolio_spread']
    rep['mde_power'] = {
        'alpha': ALPHA,
        'power': 0.80,
        'unit': 'pct_per_month（M1-3 为 |beta*IQR| 的收益量纲；M1-4 为月度 spread）',
        'z_one_sided': Z_ONE_SIDED,
        'z_two_sided': Z_TWO_SIDED,
        'rows': {
            'M1-3_EW': {
                'threshold_pct': K_ECON * 100.0,
                'se_beta': g3.get('se_nw_ew'),
                'se_pct': g3.get('se_pct'),
                'mde80_pct_one_sided': g3.get('mde80_pct_one_sided'),
                'mde80_pct_two_sided': g3.get('mde80_pct_two_sided'),
                'power_at_threshold_reject_zero_one_sided': g3.get('power_at_threshold_reject_zero_one_sided'),
                'power_at_threshold_reject_zero_two_sided': g3.get('power_at_threshold_reject_zero_two_sided'),
            },
            'M1-3_VW': {
                'threshold_pct': K_ECON * 100.0,
                'se_beta': g3.get('se_nw_vw'),
                'se_pct': (g3.get('mde80_vw') or {}).get('se_pct'),
                'mde80_pct_one_sided': (g3.get('mde80_vw') or {}).get('mde80_pct_one_sided'),
                'mde80_pct_two_sided': (g3.get('mde80_vw') or {}).get('mde80_pct_two_sided'),
                'power_at_threshold_reject_zero_one_sided':
                    (g3.get('mde80_vw') or {}).get('power_at_threshold_reject_zero_one_sided'),
                'power_at_threshold_reject_zero_two_sided':
                    (g3.get('mde80_vw') or {}).get('power_at_threshold_reject_zero_two_sided'),
            },
            'M1-4_EW_spread': {
                'threshold_pct': K_SPREAD * 100.0,
                'se_pct': g4.get('se_pct'),
                'mde80_pct_one_sided': g4.get('mde80_pct_one_sided'),
                'mde80_pct_two_sided': g4.get('mde80_pct_two_sided'),
                'power_at_threshold_reject_zero_one_sided': g4.get('power_at_threshold_reject_zero_one_sided'),
                'power_at_threshold_reject_zero_two_sided': g4.get('power_at_threshold_reject_zero_two_sided'),
            },
            'M1-4_VW_spread': {
                'threshold_pct': K_SPREAD * 100.0,
                'se_pct': (g4.get('mde80_vw_spread') or {}).get('se_pct'),
                'mde80_pct_one_sided': (g4.get('mde80_vw_spread') or {}).get('mde80_pct_one_sided'),
                'mde80_pct_two_sided': (g4.get('mde80_vw_spread') or {}).get('mde80_pct_two_sided'),
                'power_at_threshold_reject_zero_one_sided':
                    (g4.get('mde80_vw_spread') or {}).get('power_at_threshold_reject_zero_one_sided'),
                'power_at_threshold_reject_zero_two_sided':
                    (g4.get('mde80_vw_spread') or {}).get('power_at_threshold_reject_zero_two_sided'),
            },
        },
        'note': ('M1-3/M1-4 的 pass 逻辑是 abs(stat)>=K（方向盲、无 alpha）；本表功效 = 真实效应等于阈值时'
                 '拒绝 beta=0 的概率，不是过 gate 的概率。FAIL 不能区分零效应与功效不足。'
                 'se_pct = se*IQR（M1-3）或 spread 的 NW SE（M1-4），为小数收益量纲。'),
    }
    scale = dict(gates['M1-5_ew_vw_consistency'].get('tost_scale') or {})
    se_d = scale.get('se_nw_diff')
    k_grid = params()['K_Consist_M1'].get('sensitivity_grid') or [0.25, 0.50, 0.75]
    bounds = []
    for k in k_grid:
        if se_d is None:
            break
        fb = k - TOST_Z * se_d
        bounds.append({'K': k, 'fail_bound_beta': fb,
                       'fail_bound_pct_per_month': fb * IQR_POOL_REGISTERED * 100.0})
    scale['sensitivity_grid_fail_bounds'] = bounds
    scale['in_sample_binding'] = bool(
        gates['M1-5_ew_vw_consistency'].get('beta_diff') is not None
        and scale.get('tost_fail_bound_beta') is not None
        and abs(gates['M1-5_ew_vw_consistency']['beta_diff']) >= scale['tost_fail_bound_beta'])
    scale['note'] = ('TOST 是 M1-5 AND 条件之一；本样本 |d|=%s < 失败界 %s，TOST 不 binding，'
                     'M1-5 的 FAIL 由 sign_same/both_significant 决定。'
                     'TOST 等价不得与 PASS=False 并列读作「一致性已证明」。'
                     % ('NA' if gates['M1-5_ew_vw_consistency'].get('beta_diff') is None
                        else f"{gates['M1-5_ew_vw_consistency']['beta_diff']:.5f}",
                        'NA' if scale.get('tost_fail_bound_beta') is None
                        else f"{scale['tost_fail_bound_beta']:.5f}"))
    rep['m1_5_scale'] = scale
    rep['interpretation_limits'] = [
        'M1-3/M1-4 的 0.003 门槛低于本设计单侧 80% 功效 MDE（0.422/0.542/0.871 %/月）；'
        'FAIL 不能区分零效应与功效不足。',
        'M1-3/M1-4 的 pass 逻辑是 abs(x)>=K（方向盲、无 alpha）；0.549/0.394/0.215 是阈值处拒绝 beta=0 '
        '的功效，不是过 gate 的概率。',
        'M1-5 的 TOST 等价界是 beta 尺度（0.50 beta = 1.17%/月），失败界反解约 0.99%/月 = 3.3x K_Econ；'
        'TOST 是 M1-5 AND 条件之一；本样本 TOST 不 binding，M1-5 的 FAIL 由 sign_same/both_significant 决定。',
        'S5_status=FAIL 只表示未过预注册 gate；不得写成「零效应」或「折价为零」。',
    ]

    rep['gate_and_rule'] = 'M1-1 AND M1-2 AND M1-3 AND M1-4 AND M1-5 AND 硬性数值与覆盖条件（plan §5.7A P3.6 / registry g0_m1_m3_gates）'
    rep['stop_action'] = ({
        'action': 'STOP standalone MAX-pricing claim（plan §5.7A P3.6）：保留描述性关联，'
                  '不进入 S6–S12 强机制叙事；M1 失败不得用 portfolio / functional-form 诊断补救；'
                  '本停机不因无 turnover 规格为负而解除，也不得用事后更换主控制集（移出 turnover/rev）、'
                  '组合或函数形式补救',
        'descriptive_association_retained': True,
    } if rep['S5_status'] != 'PASS' else {'action': 'PASS 可推进 M2（S6）；仍受 claim ladder 限制（P6.2）'})
    rep['per_month_diagnostics_ref'] = f's3_m1_pricing_table{sfx}.json:per_month.ma_teacher'
    rep['a12_reading'] = ('A12 机械判据（v2）：L1 符号翻转（δ = +0.290，v1 样本为 δ = −0.265 且符号翻转）→ 记「部分被行业吸收」'
                          '并登记停机条件命中；实质读法：该「翻转」不是对显著效应的反转，而是未检出效应的基线'
                          '（点均值接近零且不显著）在加入行业 MAX 后的摆动（M1 主系数本就未过任一 gate）。A12 γ 为正号且来自行业/同伴收益动量'
                          '（PeerReturn 同入后塌缩），horse race 的 m1_horse_own_ind_l1/l2 不含 PeerReturn、'
                          '吸收幅度被高估，判读以 A12 含 PeerReturn 的规格为准。')
    rep['note_methodology'] = ('methodology §7 M1 强调 headline 只绑定 primary estimand（FM 系数/NW/方向/量级）；'
                               'M1-4/M1-5 属 supporting evidence 层。本报告的 AND 判定按 plan §5.7A P3.6 与 '
                               'registry g0_m1_m3_gates 执行，两者差异如实并列。')
    write_json(OUT_DIR / f's3_m1_gate_report{sfx}.json', rep)
    _write_md(rep, lc, ia, rob, OUT_DIR / f's3_m1_gate_report{sfx}.md')
    progress(f"S5_status={rep['S5_status']}；M1-1..M1-5(v3) = {[v['pass'] for v in rep['gates'].values()]}；"
             f"previous_version = {[v['pass'] for v in g1.values()]}")


def _f(x, nd=5):
    if x is None:
        return 'NA'
    try:
        if isinstance(x, float) and not np.isfinite(x):
            return 'NA'
    except (TypeError, ValueError):
        pass
    return f'{x:.{nd}f}' if isinstance(x, (int, float)) else str(x)


def _write_md(rep, lc, ia, rob, out_path):
    L = ['# S5 / M1 Gate Report（月度 PeerMAX 定价；P1-01 v3 重建）\n']
    if rep.get('out_suffix'):
        L.append(f"> 命名：本件 = canonical（`out_suffix={rep['out_suffix']}`）；无后缀 "
                 f"`s3_m1_gate_report.{{json,md}}` 为 legacy 文件名（v2 headline），"
                 f"一切以本件为准。\n")
    L.append(f"- sample_version(headline) = `{rep['sample_version']}` | v2(superseded, 旧 complete_month 口径) = "
             f"`{rep['sample_version_v2_superseded']}` | v1(superseded, 未施加 microcap 筛除) = "
             f"`{rep['sample_version_v1_superseded']}` | **S5_status = {rep['S5_status']}**")
    L.append(f"- gate 参数：alpha_m1={ALPHA} / K_Econ_M1={K_ECON} / K_Spread_M1={K_SPREAD} / "
             f"K_Consist_M1={K_CONSIST} / NW lag={NW_LAG}")
    L.append(f"- IQR 规则：`{P['iqr_scale_rule_id']}`；**登记池化值 {IQR_POOL_REGISTERED:.8f}**（treatment-only 宇宙，"
             f"全规格共用；I4 修复前实现用终样本/各列 IQR）")
    # Phase 0.3（2026-10-27）：V-P1-6 / N-7 头注（append-only）
    L.append(f"> 样本总体（V-P1-6）：{rep['sample_universe_note']}")
    L.append(f"> 控制变量定义（N-7）：{rep['control_definition_note']}")
    L.append(f"- 判定规则：{rep['gate_and_rule']}\n")
    L.append('## Gate 逐条判定（v3 headline；括号内为前一版本敏感性）\n')
    L.append('| gate | 实测（v3） | 前一版本敏感性 | 阈值 | 判定 |')
    L.append('|---|---|---|---|---|')
    sens_key = 'v2_sensitivity' if 'v2_sensitivity' in rep else 'v1_sensitivity'
    g, g1 = rep['gates'], rep[sens_key]['gates']
    L.append(f"| M1-1 符号 | beta_EW={_f(g['M1-1_sign']['beta_ew'])} / beta_VW={_f(g['M1-1_sign']['beta_vw'])} "
             f"| {_f(g1['M1-1_sign']['beta_ew'])} / {_f(g1['M1-1_sign']['beta_vw'])} "
             f"| beta<0（EW 与 VW） | {'PASS' if g['M1-1_sign']['pass'] else 'FAIL'} |")
    L.append(f"| M1-2 单侧显著 | p_EW={_f(g['M1-2_one_sided_significance']['p_one_sided_ew'],4)} / "
             f"p_VW={_f(g['M1-2_one_sided_significance']['p_one_sided_vw'],4)} "
             f"(t={_f(g['M1-2_one_sided_significance']['t_nw_ew'],2)}/{_f(g['M1-2_one_sided_significance']['t_nw_vw'],2)}) "
             f"| {_f(g1['M1-2_one_sided_significance']['p_one_sided_ew'],4)} / "
             f"{_f(g1['M1-2_one_sided_significance']['p_one_sided_vw'],4)} "
             f"| p < {ALPHA} | {'PASS' if g['M1-2_one_sided_significance']['pass'] else 'FAIL'} |")
    L.append(f"| M1-3 经济量级 | beta*IQR = {_f(g['M1-3_economic_magnitude']['economic_magnitude'])} "
             f"(IQR={_f(g['M1-3_economic_magnitude']['iqr_pool'],8)}) "
             f"| {_f(g1['M1-3_economic_magnitude']['economic_magnitude'])} | >= {K_ECON} "
             f"| {'PASS' if g['M1-3_economic_magnitude']['pass'] else 'FAIL'} |")
    L.append(f"| M1-4 组合 spread | EW 五分位 spread = {_f(g['M1-4_portfolio_spread']['spread_ew'])} "
             f"(t={_f(g['M1-4_portfolio_spread']['t_nw'],2)}) "
             f"| {_f(g1['M1-4_portfolio_spread']['spread_ew'])} | abs >= {K_SPREAD} "
             f"| {'PASS' if g['M1-4_portfolio_spread']['pass'] else 'FAIL'} |")
    t5, t51 = g['M1-5_ew_vw_consistency'], g1['M1-5_ew_vw_consistency']
    L.append(f"| M1-5 EW/VW 一致性 | |d|={_f(t5['beta_diff'])}, 同号={t5['sign_same']}, "
             f"均显著={t5['both_significant']} "
             f"| |d|={_f(t51['beta_diff'])}, 同号={t51['sign_same']} "
             f"| |d| <= {K_CONSIST} + 三条件 | {'PASS' if t5['pass'] else 'FAIL'} |")
    # ---------------- 2026-10-27 Phase 0.1：MDE/功效 / TOST 尺度 / 三行拆分（不改判定）
    L.append('\n## MDE / 功效说明（S-1；不改 gate 判定）\n')
    L.append('| gate | 阈值（%/月） | se（%/月） | MDE80 单侧（%/月） | MDE80 双侧（%/月） '
             '| 阈值处拒绝 β=0 的功效（单侧/双侧） |')
    L.append('|---|---:|---:|---:|---:|---:|')
    rows = rep['mde_power']['rows']
    labels = [('M1-3_EW', 'M1-3 EW'), ('M1-3_VW', 'M1-3 VW'),
              ('M1-4_EW_spread', 'M1-4 EW spread'), ('M1-4_VW_spread', 'M1-4 VW spread')]
    for key, lab in labels:
        r = rows[key]
        se_disp = None if r.get('se_pct') is None else r['se_pct'] * 100.0
        L.append(f"| {lab} | {_f(r.get('threshold_pct'),3)} | {_f(se_disp,4)} "
                 f"| {_f(r.get('mde80_pct_one_sided'),3)} | {_f(r.get('mde80_pct_two_sided'),3)} "
                 f"| {_f(r.get('power_at_threshold_reject_zero_one_sided'),3)} / "
                 f"{_f(r.get('power_at_threshold_reject_zero_two_sided'),3)} |")
    g3v, g4v = g1['M1-3_economic_magnitude'], g1['M1-4_portfolio_spread']
    v1_vals = [g3v.get('mde80_pct_one_sided'), (g3v.get('mde80_vw') or {}).get('mde80_pct_one_sided'),
               g4v.get('mde80_pct_one_sided')]
    v1_pow = [g3v.get('power_at_threshold_reject_zero_one_sided'),
              (g3v.get('mde80_vw') or {}).get('power_at_threshold_reject_zero_one_sided'),
              g4v.get('power_at_threshold_reject_zero_one_sided')]
    L.append('')
    L.append('- M1-3/M1-4 的 pass 逻辑是 `abs(stat) >= K`（方向盲、没有 alpha）；上表功效是「真实效应恰为阈值时'
             '拒绝 β=0」的概率，**不是「过 gate」的概率**。')
    L.append(f"- M1-3/M1-4 的登记阈值 {K_ECON*100:.3f}%/月低于本设计单侧 80% 功效 MDE"
             f"（{_f(rows['M1-3_EW']['mde80_pct_one_sided'],3)}/"
             f"{_f(rows['M1-3_VW']['mde80_pct_one_sided'],3)}/"
             f"{_f(rows['M1-4_EW_spread']['mde80_pct_one_sided'],3)} %/月）。**FAIL ≠ 零效应**："
             f"本设计无法区分「零效应」与「功效不足」。")
    g4d = g['M1-4_portfolio_spread']
    L.append(f"- **M1-4 的 VW 并报**：VW 五分位 spread = {_f(g4d.get('spread_vw'))}"
             f"（|·| = {_f(None if g4d.get('spread_vw') is None else abs(g4d['spread_vw'])*100, 4)}%/月 ≥ "
             f"{K_SPREAD*100:.3f}%），按同一 `abs()` 规则 `pass_vw={g4d.get('pass_vw')}`；headline 登记为 EW"
             f"（{_f(g4d.get('spread_ew'))}，FAIL），**不得事后改用 VW 救 M1**。该并报只说明「权重函数决定结论」（S-2），"
             f"不改变 `S5_status`。")
    L.append(f"- v1 敏感性（未施加 microcap 筛除）：MDE80 单侧 {_f(v1_vals[0],3)}/{_f(v1_vals[1],3)}/"
             f"{_f(v1_vals[2],3)}%/月，阈值处功效 {_f(v1_pow[0],3)}/{_f(v1_pow[1],3)}/{_f(v1_pow[2],3)}；"
             f"结论同向。")
    L.append('- `S5_status=FAIL` 只表示未过预注册 gate；不得写成「零效应」或「折价为零」。')
    sc = rep['m1_5_scale']
    L.append('\n## M1-5 尺度与 TOST（S-4；TOST 是 M1-5 AND 条件之一）\n')
    L.append(f"- **M1-5a 幅度**：|beta_EW − beta_VW| = {_f(t5['beta_diff'])} β（阈值 {K_CONSIST} β；"
             f"{'|d| 落在等价界内，但 M1-5 仍因方向/显著性 FAIL' if abs(t5['beta_diff'] or 0) <= K_CONSIST else '|d| 超出等价界，幅度项未过'}）。")
    L.append(f"- **M1-5b 方向/显著性**：同号={t5['sign_same']}、均显著={t5['both_significant']}"
             f"（M1-5 的 FAIL 由 sign_same/both_significant 决定）。")
    L.append(f"- **M1-5c TOST（M1-5 AND 条件之一）**：等价={t5['tost'].get('equivalent')}"
             f"（p_lower={_f(t5['tost'].get('p_lower'),4)}, p_upper={_f(t5['tost'].get('p_upper'),4)}）；"
             f"等价界 = ±{K_CONSIST} β = ±{_f(sc.get('K_Consist_x_IQR_pct_per_month'),4)}%/月；"
             f"本样本 |d| = {_f(t5['beta_diff'])} < 失败界 {_f(sc.get('tost_fail_bound_beta'),5)} β = "
             f"{_f(sc.get('tost_fail_bound_pct_per_month'),4)}%/月 = "
             f"{_f(sc.get('tost_fail_bound_over_K_Econ'),3)}×K_Econ_M1 → **TOST 支线不 binding**。"
             f"**TOST 等价 ≠ M1-5 通过，也不等于「EW/VW 一致性已证明」**。")
    L.append(f"- `K_Consist_M1 = {K_CONSIST}` 是 **β 单位**；× IQR {IQR_POOL_REGISTERED:.8f} = "
             f"**{_f(sc.get('K_Consist_x_IQR_pct_per_month'),4)}%/月 = "
             f"{_f(sc.get('K_Consist_x_IQR_over_K_Econ'),2)} × K_Econ_M1({K_ECON*100:.1f}%)**。")
    if sc.get('sensitivity_grid_fail_bounds'):
        parts = [f"K={_f(b['K'],2)} → {_f(b['fail_bound_beta'],5)}β = "
                 f"{_f(b['fail_bound_pct_per_month'],4)}%/月" for b in sc['sensitivity_grid_fail_bounds']]
        L.append('- `sensitivity_grid` 失败界：' + '；'.join(parts) + '。')
    L.append('- M1-5 的 FAIL 由 `sign_same=False`、`both_significant=False` 决定；TOST 是 M1-5 AND 条件之一'
             '（本样本不 binding），不得与 `PASS=False` 并列读作「一致性已证明」。')
    L.append('\n## 硬性数值与覆盖条件\n')
    s = rep['supporting']
    L.append(f"- FM_status = **{s['FM_status']}**；逐月满秩={s['all_months_full_rank']}、"
             f"残差 df>0={s['all_months_residual_df_positive']}、协方差有限={s['all_months_covariance_finite']}")
    L.append(f"- T_effective = {s['T_effective_ew']}（EW）/ {s['T_effective_vw']}（VW），"
             f"门槛 min_effective_month_count = {s['min_effective_month_count']}；coverage_ok={s['coverage_ok']}")
    L.append(f"- 每月最小 N = {s['min_focal_count_per_month']}（门槛 {MIN_N}）")
    L.append(f"- 主 portfolio 无未处理空组 = {s['portfolio_no_unhandled_empty_group']}；"
             f"组状态计数 = {g['M1-4_portfolio_spread'].get('portfolio_status_counts')}")
    vs = s['vif_scope']
    L.append(f"- **VIF / condition number 作用域（I5）**：主规格（{vs['scope_main_spec_only']['spec']}）"
             f"per-month max VIF = {_f(vs['scope_main_spec_only']['vif_per_month_max'],2)}"
             f"（{vs['scope_main_spec_only']['vif_per_month_max_variable']}）、"
             f"cond max = {_f(vs['scope_main_spec_only']['condition_number_max'],2)}；"
             f"**跨规格最大**（含 horse race 8 规格 + 四列主规格）= per-month max VIF "
             f"{_f(vs['scope_cross_spec_max']['per_month_max_vif']['value'],2)}"
             f"（{vs['scope_cross_spec_max']['per_month_max_vif']['spec']}/"
             f"{vs['scope_cross_spec_max']['per_month_max_vif']['variable']}）、monthly-mean VIF 最大 "
             f"{_f(vs['scope_cross_spec_max']['mean_vif_max']['value'],2)}"
             f"（{vs['scope_cross_spec_max']['mean_vif_max']['spec']}/"
             f"{vs['scope_cross_spec_max']['mean_vif_max']['variable']}）、cond max "
             f"{_f(vs['scope_cross_spec_max']['cond_max']['value'],2)}"
             f"（{vs['scope_cross_spec_max']['cond_max']['spec']}）——"
             f"registry 登的「VIF max 6.13 / cond 5.88」只覆盖主规格；跨规格成因见 "
             f"s3_m1_pricing_table.json:vif_scope.cross_spec_cause（NPM_lin 与 PeerReturn 机械共线，"
             f"v1 样本下对应值 cond 31.41 / mean VIF 30.52）")
    L.append('\n## 四列并报（主规格 EW/VW；v3 headline，前一版本括号对照）\n')
    _first = next(iter(rep['four_column_table'].values()), {})
    prev_prefix = 'v2' if 'v2_beta_ew' in _first else 'v1'
    L.append(f'| 列 | 角色 | N | 月份 | FM 窗口 | beta(EW) | t | 单侧 p | beta(VW) | t | beta·IQR | beta_EW({prev_prefix}) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, v in rep['four_column_table'].items():
        L.append(f"| {k} | {v['role']} | {v['n_rows']:,} | {v['n_months']} | {v['fm_window'][0]}..{v['fm_window'][1]} | "
                 f"{_f(v['beta_ew'])} | {_f(v['t_nw_ew'],2)} | {_f(v['p_one_sided_ew'],4)} | "
                 f"{_f(v['beta_vw'])} | {_f(v['t_nw_vw'],2)} | {_f(v['economic_magnitude'])} | "
                 f"{_f(v[f'{prev_prefix}_beta_ew'])} |")
    extra = []
    if lc is not None:
        extra.append(f"- A11 篇幅控制（v2）：beta 无控制 {_f(lc['beta_without']['beta'])} → 含控制 "
                     f"{_f(lc['beta_with']['beta'])}；篇幅系数 {_f(lc['length_coefficient']['beta'])} "
                     f"(t={_f(lc['length_coefficient']['t_nw'],2)})；停机={lc['stop_condition']['stop']}"
                     f"；v1 对照：{_f(lc['v1_sensitivity']['beta_without']['beta'])} → "
                     f"{_f(lc['v1_sensitivity']['beta_with']['beta'])}")
    if ia is not None:
        for sid, v in ia['specs'].items():
            mech = ('δ<0' if (v['retention_delta'] is not None and v['retention_delta'] < 0) else '') + \
                   ('；β 符号翻转' if v['sign_flip'] else '')
            stop = (f'机械判据命中（{mech.strip("；")}）→ 按 M1 gate 降级处理，不得只报未控制规格；'
                    f'实质读法：β_without={_f(ia["beta_without"]["beta"])} 本身不显著，故「翻转」不是对显著效应的反转，'
                    f'而是未检出效应的基线（点均值接近零且不显著）在加入行业 MAX 后的符号摆动'
                    if (v['retention_delta'] is not None and v['retention_delta'] < 0) or v['sign_flip']
                    else '未触发停机（δ>=0 且无符号翻转）')
            extra.append(f"- A12 {sid}: beta_with={_f(v['beta_with']['beta'])} "
                         f"(t={_f(v['beta_with']['t_nw'],2)}), delta={_f(v['retention_delta'],3)} → "
                         f"{v['interpretation']}；{stop}")
    if ia is not None and 'industry_max_momentum_diagnostic' in ia:
        dg = ia['industry_max_momentum_diagnostic']['specs']
        parts = []
        for lab, v in dg.items():
            parts.append(lab + ': ' + ', '.join(
                f'{c}={_f(s["beta"])}(t={_f(s["t_nw"],2)})' for c, s in v['coefs'].items()))
        extra.append('- A12 γ 动量来源诊断（同一样本、只换 x 集）：' + '；'.join(parts))
    if rob is not None:
        pc = rob.get('placebo_comparison', {})
        picks = ['m1_rob_postadj', 'm1_weight_equal', 'm1_weight_linear', 'm1_weight_logsim',
                 'm1_placebo_random', 'm1_placebo_timeshift']
        for p in picks:
            if p in rob['specs']:
                v = rob['specs'][p]
                extra.append(f"- A9 {p}: status={v.get('spec_status')} beta={_f(v['spec_beta'])} "
                             f"(t={_f(v['t_nw'],2)}, months={v['n_months_used']})")
        for sid, v in rob['specs'].items():
            if v.get('spec_status') == 'NOT_COMPUTABLE':
                extra.append(f"- A9 {sid}: **NOT_COMPUTABLE** — {v.get('status_reason')}")
        if pc:
            extra.append(f"- A9 placebo 对照（I1 修正后）：真实网络 β={_f(pc['real_network']['beta'])} vs "
                         f"随机网络 β={_f(pc['random_network']['spec_beta'])}"
                         f"(t={_f(pc['random_network']['t_nw'],2)}) vs 时间错位 β="
                         f"{_f(pc['timeshift_network']['spec_beta'])}"
                         f"(t={_f(pc['timeshift_network']['t_nw'],2)})")
    if extra:
        L.append('\n## 稳健性 / 登记必做项摘要\n')
        L.extend(extra)
    L.append(f"\n## 停机动作\n\n{rep['stop_action']['action']}\n")
    L.append('> 不得用事后更换主控制集（移出 turnover/rev）、组合或函数形式补救本停机结论；'
             '主控制集按方法论冻结。')
    L.append('> MDE 限制：M1-3/M1-4 的登记门槛低于本设计单侧 80% 功效 MDE'
             f"（{_f(rows['M1-3_EW']['mde80_pct_one_sided'],3)}/"
             f"{_f(rows['M1-3_VW']['mde80_pct_one_sided'],3)}/"
             f"{_f(rows['M1-4_EW_spread']['mde80_pct_one_sided'],3)} %/月）；FAIL ≠ 零效应。")
    L.append(f"> {rep['note_methodology']}")
    L.append(f"> {rep['a12_reading']}")
    Path(out_path).write_text('\n'.join(L), encoding='utf-8')
    print(f'  -> {Path(out_path).relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
