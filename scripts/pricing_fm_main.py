#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A6+A7：pricing_gate 主规格 Fama-MacBeth 回归 与 baseline horse race（Phase A 复核修复批）。

主规格（plan §5.7A economic_magnitude.2）：
  R_{i,t+1} = alpha_t + beta*PeerMAXraw + theta*OwnMAXraw + phi*PeerReturn + Gamma*X + eps
horse race（economic_magnitude.3）：OwnMAX / OwnMAX+IndustryMAX(L1,L2) / +M-A / +M-B / 两网同入 / CF-NPM 分解

2026-10-10 复核修复：
  I3  headline 样本 = pricing_sample_v2_2015_2023（microcap + 次新股筛除已施加）；v1（未施加 microcap 筛除）
      作敏感性并报
  I4  经济量级统一用 treatment-only 池化登记值 IQR_POOL_REGISTERED = 0.02341287（全规格共用）
  I5  VIF/condition number 作用域标注 + 跨规格最大值并报（成因：pricing_cf_npm 的 NPM_lin=PeerReturn−own_R_t
      与 PeerReturn/rev 机械共线）
产出：reports/data_gates/G0/pricing_table.{json,md}、pricing_horse_race.json
用法：python scripts/pricing_fm_main.py [--sample-version {v1,v2}] [--out-suffix _v2]

2026-10-11 F-2（v2 样本重跑批）：
  - 新增 `--sample-version`（默认 v2，headline；另一版本自动作敏感性并报）与 `--out-suffix`
    （如 `--out-suffix _v2` 产出 pricing_table_v2.{json,md} / pricing_horse_race_v2.json，不覆盖 v1 基线名）。
  - 四列并报表加**窗口列 + 窗口说明**（主口径 2015-05..2023-12；ma/mb 2018-05..2023-12，防误读为同窗口）。
  - horse race 说明句的 γ/t 读数一律以产物 JSON 为准，不在源码里复制数字（2026-11-19 审核收口）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_lib import IQR_POOL_REGISTERED, SAMPLE_VERSION_V1, SAMPLE_VERSION_V2, SAMPLE_VERSION_V3, SAMPLE_VERSION_V4, OUT_DIR, ROOT, SAMPLE_VERSION_LABEL, assert_pricing_contract, demean_by, fm_cluster_se, fm_regression, param_value, progress, sample_path, vif_table, write_json

P = param_value('sample_and_spec_registry')
NW_LAG = int(param_value('nw_lag'))
DIRECTION = str(param_value('p_direction'))
MIN_N = P['min_focal_count_per_month']
MIN_MONTHS = P['min_effective_month_count']
CONTROLS = list(param_value('pricing_main_controls_v1')['list'])
MAIN = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + CONTROLS
COLUMNS = ['ma_teacher', 'mb_teacher', 'ma', 'mb']
COL_ROLE = {
    'ma_teacher': '主口径（中文法 × 老师 MD&A；主表列）',
    'mb_teacher': '方法稳健性列（TF-IDF × 同语料）',
    'ma': '文本稳健性列（中文法 × 自建业务描述语料）',
    'mb': '语料对照列（TF-IDF × 自建语料；仅历史对照）',
}


def load_sample(key: str, version: str = 'v3') -> pd.DataFrame:
    return pd.read_parquet(sample_path(key, version))


def sample_report_path(version: str) -> Path:
    """v4/v3 读专用报告；其余版本回退主报告（主报告 = 当前 headline，含 samples/samples_v1）。"""
    if version == 'v4':
        return OUT_DIR / 'pricing_sample_report_v4.json'
    if version == 'v3':
        p = OUT_DIR / 'pricing_sample_report_v3.json'
        if p.is_file():
            return p
    return OUT_DIR / 'pricing_sample_report.json'


def expected_sample_rows(sample_report: dict, key: str, version: str) -> int:
    """登记样本行数：v4/v3 取 samples，v2 取 samples_v2_superseded，v1 取 samples_v1_superseded。"""
    if version in ('v4', 'v3'):
        return int(sample_report['samples'][key]['rows_total'])
    if version == 'v2':
        if 'samples_v2_superseded' in sample_report:
            return int(sample_report['samples_v2_superseded'][key]['rows_total'])
        return int(sample_report['samples'][key]['rows_total'])
    return int(sample_report['samples_v1_superseded'][key]['rows_total'])


def load_checked_sample(key: str, version: str, sample_report: dict) -> pd.DataFrame:
    """读样本 parquet 后立即核验四条硬门（行数/键/PIT/缺失不填 0），失败 SystemExit(2)。"""
    df = load_sample(key, version)
    assert_pricing_contract(df, expected_rows=expected_sample_rows(sample_report, key, version),
                       kind='sample', where=f'pricing_fm_main:{key}:{version}')
    return df


def summarise(res, xcols, iqr=None, extra=None):
    out = {'n_months_used': res['n_months_used'], 'weight_col': res['weight_col'],
           'coefs': res['coefs']}
    if iqr is not None:
        b = res['coefs']['PeerMAXraw']['beta']
        out['iqr_pool'] = iqr
        out['iqr_pool_rule'] = 'IQR_POOL_TREATMENT_ONLY_v1（登记值，全规格共用）'
        out['economic_magnitude_beta_x_iqr'] = (None if b is None else b * iqr)
    if extra:
        out.update(extra)
    return out


def spec_block(df: pd.DataFrame, key: str, iqr: float) -> dict:
    """单列主规格：EW/VW + 后复权 + 逐月状态。"""
    sub = df[df['pricing_in_fm_sample'].astype(bool)].copy()
    res_ew, pm_ew = fm_regression(sub, 'R_next', MAIN, weight_col=None,
                                  min_n=MIN_N, nw_lag=NW_LAG, direction=DIRECTION)
    res_vw, pm_vw = fm_regression(sub, 'R_next', MAIN, weight_col='market_equity',
                                  min_n=MIN_N, nw_lag=NW_LAG, direction=DIRECTION)
    iqr_sample = float(sub['PeerMAXraw'].quantile(.75) - sub['PeerMAXraw'].quantile(.25))
    blk = {
        'role': COL_ROLE[key], 'n_rows': int(len(sub)),
        'n_months': res_ew['n_months_used'],
        'fm_window': [min(d['month'] for d in pm_ew if d.get('used')),
                      max(d['month'] for d in pm_ew if d.get('used'))] if res_ew['n_months_used'] else None,
        'sample_window': str(df['sample_window'].iloc[0]) if 'sample_window' in df.columns and len(df) else None,
        'iqr_pool': iqr,
        'iqr_pool_sample_diagnostic': iqr_sample,
        'ew': summarise(res_ew, MAIN, iqr),
        'vw': summarise(res_vw, MAIN, iqr),
        'per_month_n': {
            'min_used': int(min([d['n'] for d in pm_ew if d.get('used')] or [0])),
            'median_used': float(np.median([d['n'] for d in pm_ew if d.get('used')] or [0])),
            'max_used': int(max([d['n'] for d in pm_ew if d.get('used')] or [0])),
            'min_all_months': int(min(d['n'] for d in pm_ew)),
            'months_below_min_n': int(sum(1 for d in pm_ew if d.get('used') is False
                                          and d.get('reason') == 'below_min_n')),
            'note': 'min/median/max_used 只统计实际进入估计的月份；'
                    'months_below_min_n 为因单月样本 <min_focal_count_per_month 被跳过的月数'},
        'all_months_full_rank': bool(all(d.get('rank_status') == 'FULL' for d in pm_ew if d.get('used'))),
        'all_months_residual_df_positive': bool(all(d.get('residual_df', 0) > 0 for d in pm_ew if d.get('used'))),
        'all_months_covariance_finite': bool(all(d.get('covariance_status') == 'finite' for d in pm_ew if d.get('used'))),
        'vif': vif_table(sub, MAIN),
    }
    sub_post = sub[sub['R_next_postadj'].notna()]
    res_post, _ = fm_regression(sub_post, 'R_next_postadj', MAIN, weight_col=None,
                                min_n=MIN_N, nw_lag=NW_LAG, direction=DIRECTION)
    blk['post_adjusted'] = {
        'role': 'sensitivity_only（raw 为主口径）', 'n_rows': int(len(sub_post)),
        'formula_id': 'R_NEXT_POST_ADJ_v1',
        'coefs': res_post['coefs'], 'n_months_used': res_post['n_months_used'],
    }
    return blk, pm_ew, pm_vw


def vif_scan(entries) -> dict:
    """I5：跨规格 VIF/condition number 扫描（per-month max 与 monthly-mean 两种口径）。"""
    best = {'per_month_max_vif': None, 'mean_vif_max': None, 'cond_max': None}
    for sid, v in entries:
        for var, val in (v.get('vif_max') or {}).items():
            if val is None:
                continue
            if best['per_month_max_vif'] is None or val > best['per_month_max_vif']['value']:
                best['per_month_max_vif'] = {'value': float(val), 'spec': sid, 'variable': var}
        for var, val in (v.get('vif_mean') or {}).items():
            if val is None:
                continue
            if best['mean_vif_max'] is None or val > best['mean_vif_max']['value']:
                best['mean_vif_max'] = {'value': float(val), 'spec': sid, 'variable': var}
        c = v.get('condition_number_max')
        if c is not None and (best['cond_max'] is None or c > best['cond_max']['value']):
            best['cond_max'] = {'value': float(c), 'spec': sid}
    return best


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='A6+A7：pricing_gate 主规格 Fama-MacBeth 回归与 horse race')
    ap.add_argument('--sample-version', choices=['v1', 'v2', 'v3', 'v4'], default=SAMPLE_VERSION_LABEL,
                    help='headline 样本版本：v3（默认，pricing_sample_v3_20261115）/ v4（pricing_sample_v4_20261116，W2 创建）/ '
                         'v2（pricing_sample_v2_2015_2023，superseded）/ v1（pricing_sample_v1_2015_2023）；另一版本自动作敏感性并报')
    ap.add_argument('--out-suffix', default='',
                    help="产物文件名后缀，如 '_v2'（产出 pricing_table_v2.{json,md} / "
                         "pricing_horse_race_v2.json，不覆盖 v1 基线名；默认空 = 现有默认名）")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ver = args.sample_version
    other = {'v4': 'v3', 'v3': 'v2', 'v2': 'v1', 'v1': 'v2'}[ver]
    sfx = args.out_suffix
    ver_id = {'v1': SAMPLE_VERSION_V1, 'v2': SAMPLE_VERSION_V2,
              'v3': SAMPLE_VERSION_V3, 'v4': SAMPLE_VERSION_V4}[ver]
    other_id = {'v1': SAMPLE_VERSION_V1, 'v2': SAMPLE_VERSION_V2,
                'v3': SAMPLE_VERSION_V3, 'v4': SAMPLE_VERSION_V4}[other]
    # Phase 1 I-2：启动即核验登记样本 build report 的四条硬门；失败退出，不写任何新结果。
    sample_report = json.loads(sample_report_path(ver).read_text(encoding='utf-8'))
    assert_pricing_contract(sample_report, where='pricing_fm_main:build_report')
    progress(f'A6 开始：主规格 FM 回归（headline={ver_id} + {other} 敏感性 + 四列并报 + EW/VW + 后复权；'
             f'out_suffix={sfx or "(空)"}）')
    samples = {k: load_checked_sample(k, ver, sample_report) for k in COLUMNS}
    main_df = samples['ma_teacher']
    fm_mask = main_df['pricing_in_fm_sample'].fillna(False).astype(bool)
    n_fm_headline = int(fm_mask.sum())
    n_months_headline = int(main_df.loc[fm_mask, 'month'].nunique())

    report = {
        'spec_id': 'pricing_main_v1', 'sample_version': ver_id,
        'headline_sample_version': ver_id,
        'sample_version_v1_superseded': SAMPLE_VERSION_V1,
        'sample_version_v2_superseded': SAMPLE_VERSION_V2,
        'sample_version_v3': SAMPLE_VERSION_V3,
        'sample_version_v4': SAMPLE_VERSION_V4,
        'out_suffix': sfx,
        'phase': 'sign-01（2026-11-15：complete_month/n_valid_days = volume>0 交易日 >=15）',
        'formula': 'R_{i,t+1} = alpha_t + beta*PeerMAXraw + theta*OwnMAXraw + phi*PeerReturn + Gamma*X + eps',
        'x_controls': CONTROLS, 'nw_lag': NW_LAG, 'min_n_per_month': MIN_N,
        'min_effective_month_count': MIN_MONTHS,
        'estimator': 'Fama-MacBeth：逐月横截面 OLS(EW) / WLS(VW, weight=market_equity) -> 系数序列 -> 均值 + NW(6) SE',
        'iqr_rule': P['iqr_scale_rule'], 'iqr_scale_rule_id': P['iqr_scale_rule_id'],
        'iqr_pool_registered': IQR_POOL_REGISTERED,
        'iqr_pool_rule_note': ('IQR_pool 在 treatment-only 宇宙（PeerMAXraw/OwnMAXraw/PeerReturn/controls 可得，'
                               '不含 outcome 条件）算一次、全规格共用；登记值 0.02341287（registry '
                               'sample_and_spec_registry change_log 2026-10-10）。各规格自身样本 IQR 仅作诊断。'),
        'column_roles': COL_ROLE, 'main': {}, 'post_adjusted': {},
        'per_month': {},
    }
    report[f'{other}_sensitivity'] = {}
    report['window_note'] = (
        '四列 FM 窗口**不同**（防误读为同窗口）：ma_teacher / mb_teacher（主口径与方法稳健性列）= '
        '2015-05..2023-12（104 月）；ma / mb（文本稳健性列与语料对照列）= 2018-05..2023-12（68 月，'
        '自建业务描述语料自 2018 年起可得）。N / 系数 / 显著性不可按同窗口跨列解读，'
        '跨列比较仅在重叠窗口 2018-05..2023-12 内成立。')
    report['horse_race_note'] = (
        'A7 horse race 的 pricing_horse_own_ind_l1 / pricing_horse_own_ind_l2 规格**不含 PeerReturn**（也未含两网同入），'
        '故其对 PeerMAXraw 的吸收幅度被高估；判读以 pricing_industry_absorb 产物里**含 PeerReturn** 的 '
        'L1 规格 γ 为准（当次读数见该产物，本文不复制数值）。')
    # Phase 0.3（2026-10-27）：V-sign-6 / N-7 口径注记（append-only；不改任何数值）
    _n_panel = int(len(main_df))
    _cm = int((main_df['pricing_exclusion_reason'] == 'control_missing').sum()) \
        if 'pricing_exclusion_reason' in main_df.columns else None
    report['sample_universe_note'] = (
        f'pricing_gate 主样本（headline，sample_version = {ver_id}）= 主控制集全部可得、且下月 complete_month=1'
        f'（有效交易日 >=15）的 {n_fm_headline:,} 个 focal-month / {n_months_headline} 个月'
        f'（{report["main"]["ma_teacher"]["fm_window"][0]}..{report["main"]["ma_teacher"]["fm_window"][1]}），'
        f'占样本行集 {_n_panel:,} 行的 {100.0 * n_fm_headline / _n_panel:.1f}%'
        + (f'；control_missing 剔除 {_cm:,} 行' if _cm is not None else '')
        + '。该样本即「分析师覆盖宇宙」的严格口径（AnalystCoverage 与 NetworkDegree 是控制集成员，'
          '缺失即剔除）；全 A / 无分析师屏幕样本仅作敏感性，不进入 headline，也不得写成「全 A 主样本」。'
          '各剔除原因的规模见 sample report 的 exclusion_counts_in_order。')
    _deg = main_df.loc[fm_mask, 'NetworkDegree']
    _src = main_df.loc[fm_mask, 'n_sources'] if 'n_sources' in main_df.columns else None
    _q = (lambda s: '/'.join(f'{s.quantile(q):.0f}' for q in (0.1, 0.5, 0.75, 0.9))) if len(_deg) else \
        (lambda s: 'NA')
    report['control_definition_note'] = (
        'NetworkDegree = analyst-network 共同分析师覆盖网络中的度数（firms sharing at least one analyst；来源 '
        'data/processed/analyst/firm_analyst_derived.csv），不是 M-A 产品市场网络的度数；'
        '主控制集取自 registry pricing_main_controls_v1，产品网络处理变量自己的度数 n_sources 不在其中。'
        f'本样本分布（p10/p50/p75/p90）：NetworkDegree = {_q(_deg)}'
        + (f'；n_sources = {_q(_src)}' if _src is not None else '')
        + '。n_sources 仅作分布报告，不加入控制集。')

    # ---------------- 主规格：四列 × EW/VW（headline 样本版本）
    for key in COLUMNS:
        df = samples[key]
        blk, pm_ew, pm_vw = spec_block(df, key, IQR_POOL_REGISTERED)
        report['main'][key] = blk
        report['post_adjusted'][key] = blk['post_adjusted']   # F-2：填充后复权敏感性块（原为空表）
        if key == 'ma_teacher':
            report['per_month'][key] = {'ew': pm_ew, 'vw': pm_vw,
                                        'beta_series_ew': [d['beta']['PeerMAXraw'] for d in pm_ew if d.get('used')],
                                        'beta_series_vw': [d['beta']['PeerMAXraw'] for d in pm_vw if d.get('used')],
                                        'months_ew': [d['month'] for d in pm_ew if d.get('used')],
                                        'months_vw': [d['month'] for d in pm_vw if d.get('used')]}
        progress(f'A6 {key}({ver}): EW beta_peer={_fmt(blk["ew"])} 月份={blk["ew"]["n_months_used"]} '
                 f'窗口={blk["fm_window"]}')

    # 四列窗口对照（F-2 ②：防误读为同窗口）
    report['fm_window_by_column'] = {k: report['main'][k]['fm_window'] for k in COLUMNS}
    report['fm_window_note_by_column'] = {
        'ma_teacher': '2015-05..2023-12（104 月；主口径）',
        'mb_teacher': '2015-05..2023-12（104 月）',
        'ma': '2018-05..2023-12（68 月；自建业务描述语料自 2018 年起）',
        'mb': '2018-05..2023-12（68 月；自建语料）'}

    # ---------------- 另一版本敏感性（v1 = 未施加 microcap 筛除）
    for key in COLUMNS:
        df1 = load_checked_sample(key, other, sample_report)
        blk1, pm1_ew, pm1_vw = spec_block(df1, key, IQR_POOL_REGISTERED)
        blk1['sample_version'] = other_id
        blk1['status'] = f'{other}_superseded_registered_deviation'
        blk1['role_note'] = (
            f'{other} = 未施加 microcap（与次新股）筛除的旧样本版本；ma/mb 列另含 2024-01 事件月'
            '（未施加 sample 窗口过滤）' if other == 'v1' else
            f'{other} = microcap/次新股筛除 + 统一事件月窗口的 headline 样本版本')
        if key == 'ma_teacher':
            blk1['per_month_series'] = {
                'beta_series_ew': [d['beta']['PeerMAXraw'] for d in pm1_ew if d.get('used')],
                'beta_series_vw': [d['beta']['PeerMAXraw'] for d in pm1_vw if d.get('used')],
                'months_ew': [d['month'] for d in pm1_ew if d.get('used')]}
        report[f'{other}_sensitivity'][key] = blk1
        progress(f'A6 {key}({other} 敏感性): EW beta_peer={_fmt(blk1["ew"])} 月份={blk1["ew"]["n_months_used"]} '
                 f'窗口={blk1["fm_window"]}')


    # ---------------- 治疗变量完整性口径（J05）：完整案例 vs 覆盖源内重新归一化
    # 主列要求「全部 peer 可得」，任一 peer 停牌即整行缺失；renorm 在覆盖源上归一，
    # 缺失 peer 不改变条件均值。两版并报，headline 仍是完整案例，不改门禁。
    renorm_block = {}
    for key in COLUMNS:
        d = samples[key]
        sub = d[d['pricing_in_fm_sample'].astype(bool)].copy()
        if 'PeerMAXraw_renorm' not in sub.columns:
            renorm_block[key] = {'status': 'NOT_AVAILABLE', 'reason': '缺 PeerMAXraw_renorm 列'}
            continue
        sub_r = sub[sub['PeerMAXraw_renorm'].notna()].copy()
        xcols = ['PeerMAXraw_renorm'] + MAIN[1:]
        res_r, pm_r = fm_regression(sub_r, 'R_next', xcols, min_n=MIN_N, nw_lag=NW_LAG,
                                    direction=DIRECTION)
        c_r = res_r['coefs']['PeerMAXraw_renorm']
        c_main = report['main'][key]['ew']['coefs']['PeerMAXraw']
        renorm_block[key] = {
            'n_rows': int(len(sub_r)), 'n_months_used': res_r['n_months_used'],
            'beta_renorm_ew': c_r['beta'], 't_nw_renorm_ew': c_r['t_nw'],
            'p_one_sided_renorm_ew': c_r['p_one_sided'],
            'beta_completecase_ew': c_main['beta'], 't_nw_completecase_ew': c_main['t_nw'],
            'sign_same': (c_r['beta'] is not None and c_main['beta'] is not None
                          and np.sign(c_r['beta']) == np.sign(c_main['beta'])),
            'all_months_full_rank': bool(all(d2.get('rank_status') == 'FULL'
                                             for d2 in pm_r if d2.get('used'))),
            'reading': 'renorm 覆盖源上归一（缺失 peer 不改变条件均值）；主列为完整案例。'
                       '两版并报，不替换 headline',
        }
    report['treatment_completeness'] = renorm_block

    # ---------------- 全 A 对照（J09）：去掉分析师覆盖与网络度数两个控制，并放回
    # 仅因这两个控制缺失而被删的行。用于看"分析师覆盖宇宙"对系数的影响。
    all_a_block = {}
    for key in COLUMNS:
        d = samples[key]
        keep = d['pricing_exclusion_reason'].fillna('').isin(['', 'control_missing'])
        sub = d[keep & d['R_next'].notna() & d['PeerMAXraw'].notna()].copy()
        xcols = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + [
            c for c in CONTROLS if c not in ('AnalystCoverage', 'NetworkDegree')]
        sub = sub.dropna(subset=xcols)
        res_a, pm_a = fm_regression(sub, 'R_next', xcols, min_n=MIN_N, nw_lag=NW_LAG,
                                    direction=DIRECTION)
        c_a = res_a['coefs']['PeerMAXraw']
        all_a_block[key] = {
            'n_rows': int(len(sub)), 'n_months_used': res_a['n_months_used'],
            'x_controls': xcols,
            'beta_all_a_ew': c_a['beta'], 't_nw_all_a_ew': c_a['t_nw'],
            'p_one_sided_all_a_ew': c_a['p_one_sided'],
            'beta_headline_ew': report['main'][key]['ew']['coefs']['PeerMAXraw']['beta'],
            't_nw_headline_ew': report['main'][key]['ew']['coefs']['PeerMAXraw']['t_nw'],
            'all_months_full_rank': bool(all(d2.get('rank_status') == 'FULL'
                                             for d2 in pm_a if d2.get('used'))),
            'reading': '放回 control_missing 行并去掉两个分析师相关控制；不替换 headline，'
                       '只用于量化分析师覆盖宇宙的选择效应',
        }
    report['all_a_comparison'] = all_a_block

# ---------------- microcap 异质性（J10）：现在是最小 10% 直接筛除。
    # 并报「含 microcap」与「仅 microcap」两版，看筛除是否掩盖了效应。不换 headline。
    micro_block = {}
    for key in COLUMNS:
        d = samples[key]
        keep = d['pricing_exclusion_reason'].fillna('').isin(['', 'microcap_q10_exclusion'])
        sub_all = d[keep & d['R_next'].notna() & d['PeerMAXraw'].notna()].dropna(
            subset=['R_next'] + MAIN)
        sub_micro = d[(d['pricing_exclusion_reason'] == 'microcap_q10_exclusion')
                      & d['R_next'].notna() & d['PeerMAXraw'].notna()].dropna(
            subset=['R_next'] + MAIN)
        entry = {'x': MAIN}
        for tag, sub in (('with_microcap', sub_all), ('microcap_only', sub_micro)):
            if len(sub) == 0:
                entry[tag] = {'status': 'NOT_COMPUTABLE', 'reason': '无可用行'}
                continue
            res, pm = fm_regression(sub, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG,
                                    direction=DIRECTION)
            c = res['coefs']['PeerMAXraw']
            entry[tag] = {
                'n_rows': int(len(sub)), 'n_months_used': res['n_months_used'],
                'beta_ew': c['beta'], 't_nw_ew': c['t_nw'],
                'p_one_sided_ew': c['p_one_sided'],
                'all_months_full_rank': bool(all(d2.get('rank_status') == 'FULL'
                                                 for d2 in pm if d2.get('used'))),
            }
        entry['beta_headline_ew'] = report['main'][key]['ew']['coefs']['PeerMAXraw']['beta']
        micro_block[key] = entry
    report['microcap_heterogeneity'] = {
        'note': 'headline 已剔除形成月市值后 10%（microcap_q10_exclusion）；本块并报含入与仅入'
                '两版，只作异质性诊断，不替换 headline',
        'by_column': micro_block,
    }

    # ---------------- A7 horse race（统一样本，v2）
    progress('A7 开始：horse race（同样本/同 outcome/同控制/同估计器/同 NW）')
    main_fm = main_df[main_df['pricing_in_fm_sample'].astype(bool)].copy()
    mb = samples['mb_teacher'][['focal_id', 'month', 'PeerMAXraw', 'PeerReturn']].rename(
        columns={'PeerMAXraw': 'PeerMAXraw_mb', 'PeerReturn': 'PeerReturn_mb'})
    srcm = pd.read_csv(ROOT / 'data/processed/source_month_argmax_panel.csv',
                       usecols=['source_id', 'month', 'R'], dtype={'source_id': str, 'month': str})
    srcm = srcm.rename(columns={'source_id': 'focal_id', 'R': 'own_R_t'})
    hr = main_fm.merge(mb, on=['focal_id', 'month'], how='left')
    hr = hr.merge(srcm, on=['focal_id', 'month'], how='left')
    hr['NPM_lin'] = hr['PeerReturn'] - hr['own_R_t']
    hr = hr[(hr['PeerMAXraw_mb'].notna()) & hr['PeerReturn_mb'].notna()
            & hr['ind_l1_max_ew'].notna() & hr['ind_l2_max_ew'].notna()
            & hr['NPM_lin'].notna() & hr['ind_l1_ret_ew'].notna() & hr['ind_l2_ret_ew'].notna()]
    progress(f'A7 horse race 统一样本 = {len(hr):,} 行（主口径 {ver} FM 样本 ∩ 两网 + 行业 baseline 可得）')

    specs = {
        'pricing_horse_own': ['OwnMAXraw'] + CONTROLS,
        'pricing_horse_own_ind_l1': ['OwnMAXraw', 'ind_l1_max_ew'] + CONTROLS,
        'pricing_horse_own_ind_l2': ['OwnMAXraw', 'ind_l2_max_ew'] + CONTROLS,
        'pricing_horse_ma': MAIN,
        'pricing_horse_mb': ['PeerMAXraw_mb', 'OwnMAXraw', 'PeerReturn_mb'] + CONTROLS,
        'pricing_horse_two_net': ['PeerMAXraw', 'PeerMAXraw_mb', 'OwnMAXraw', 'PeerReturn', 'PeerReturn_mb'] + CONTROLS,
        'pricing_cf_npm': ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'NPM_lin'] + CONTROLS,
        'pricing_horse_ind_ret': ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'ind_l1_ret_ew', 'ind_l2_ret_ew'] + CONTROLS,
    }
    hr_report = {
        'spec_id': 'pricing_horse_race_v1', 'sample_version': ver_id,
        'sample': f'主口径（ma_teacher）pricing_gate {ver} FM 样本 ∩ mb_teacher PeerMAX/PeerReturn ∩ '
                  '申万一级/二级 IndustryMAX/IndustryPeerReturn 可得',
        'n_rows': int(len(hr)), 'specs': {}, 'same_sample_all_specs': True,
        'note': '所有规格使用同一行集、同一 outcome（R_next）、同一控制、同一估计器（逐月 OLS）与 NW lag=6',
        'reading_note': (
            'pricing_horse_own_ind_l1 / pricing_horse_own_ind_l2 规格**不含 PeerReturn**（也未含两网同入），'
            '对 PeerMAXraw 的行业吸收幅度被**高估**（行业与同伴收益动量部分重叠）；'
            '判读以 pricing_industry_absorb 产物里**含 PeerReturn** 的 L1 规格 γ 为准'
            '（当次读数见该产物，本文不复制数值）。'),
    }
    for sid, xs in specs.items():
        res, pm = fm_regression(hr, 'R_next', xs, min_n=MIN_N, nw_lag=NW_LAG,
                                direction=DIRECTION)
        hr_report['specs'][sid] = {
            'x': xs, 'n_months_used': res['n_months_used'],
            'coefs': res['coefs'],
            'all_months_full_rank': bool(all(d.get('rank_status') == 'FULL' for d in pm if d.get('used'))),
            'all_months_covariance_finite': bool(all(d.get('covariance_status') == 'finite' for d in pm if d.get('used'))),
            'vif': vif_table(hr, xs),
        }
        progress(f'A7 {sid}: beta_peer={_fmt(res)}')

    # ---------------- 行业×月固定效应：把「行业共同冲击」从 PeerMAX 里剥出去
    # (month, l1_code) 是饱和虚拟变量：within 去均值后逐月回归即带该组固定效应的估计量。
    fe_src = main_fm.dropna(subset=['R_next', 'l1_code'] + MAIN).copy()
    fe_df = demean_by(fe_src, ['R_next'] + MAIN, ['month', 'l1_code'])
    res_fe, pm_fe = fm_regression(fe_df, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG,
                                  direction=DIRECTION)
    b_main = report['main']['ma_teacher']['ew']['coefs']['PeerMAXraw']['beta']
    b_fe = res_fe['coefs']['PeerMAXraw']['beta']
    fe_delta = (None if (b_main in (None, 0) or b_fe is None)
                else 1.0 - abs(b_fe) / abs(b_main))
    hr_report['industry_month_fe'] = {
        'spec_id': 'pricing_industry_month_fe_v1',
        'formula': 'R_next ~ PeerMAXraw + controls，组内去均值于 (month, 申万一级)；逐月 OLS + NW(6)',
        'n_rows': int(len(fe_df)), 'n_months_used': res_fe['n_months_used'],
        'coefs': res_fe['coefs'],
        'beta_main_no_fe': b_main,
        'beta_with_fe': b_fe,
        'retention_delta': fe_delta,
        'all_months_full_rank': bool(all(d.get('rank_status') == 'FULL'
                                         for d in pm_fe if d.get('used'))),
        'reading': ('retention_delta = 1 - |beta_FE| / |beta_main|；接近 0 表示 PeerMAX 的'
                    '预测力主要来自行业×月共同冲击，接近 1 表示行业内部仍有增量'),
    }
    progress(f'A7b 行业×月 FE: beta={b_fe}（主规格 {b_main}），retention_delta={fe_delta}')

    # ---------------- 月内行业聚类稳健抽样误差（与 NW 并报，不改 headline 判据）
    cluster_block = {}
    for key in COLUMNS:
        d = samples[key]
        sub = d[d['pricing_in_fm_sample'].astype(bool)].copy()
        if 'l1_code' not in sub.columns:
            cluster_block[key] = {'status': 'NOT_AVAILABLE', 'reason': '样本缺 l1_code'}
            continue
        for tag, weight_col in (('ew', None), ('vw', 'market_equity')):
            cluster_block[f'{key}_{tag}'] = fm_cluster_se(
                sub, 'R_next', MAIN, 'l1_code', weight_col=weight_col,
                min_n=MIN_N, nw_lag=NW_LAG)
        progress(f'A7c 聚类 SE {key}: '
                 f'deff={cluster_block[f"{key}_ew"].get("deff_within_month")}')
    report['industry_cluster_se'] = {
        'cluster_col': 'l1_code（申万一级）',
        'formula': 'se_hybrid = sqrt(se_nw² + se_sampling_cluster² − se_sampling_iid²)',
        'note': '只作敏感性并报；显著性门仍用 NW(6)。deff>1 表示同月同行业相关使同方差 SE 偏小',
        'by_column': cluster_block,
    }

    hr_report['n_months_used_common'] = hr_report['specs']['pricing_horse_ma']['n_months_used']
    write_json(OUT_DIR / f'pricing_horse_race{sfx}.json', hr_report)

    # ---------------- I5：VIF / condition number 作用域
    entries = [(f'column:{k}', report['main'][k]['vif']) for k in COLUMNS]
    entries += [(f'horse_race:{sid}', v['vif']) for sid, v in hr_report['specs'].items()]
    scan = vif_scan(entries)
    report['vif_scope'] = {
        'definition': ('VIF = 逐月横截面设计矩阵（标准化后）各列对其它列回归的 1/(1-R²)；'
                       'condition number = 标准化设计矩阵（含常数项）逐月 SVD 最大/最小奇异值比。'
                       '`per_month_max_vif` = 逐月 VIF 的月内最大；`mean_vif_max` = 逐月 VIF 的跨月平均。'),
        'scope_main_spec_only': {'spec': 'column:ma_teacher（主规格）',
                                 'vif_per_month_max': max(v for v in
                                                          report['main']['ma_teacher']['vif']['vif_max'].values()),
                                 'vif_per_month_max_variable': max(
                                     report['main']['ma_teacher']['vif']['vif_max'],
                                     key=lambda c: report['main']['ma_teacher']['vif']['vif_max'][c]),
                                 'condition_number_max': report['main']['ma_teacher']['vif']['condition_number_max']},
        'scope_cross_spec_max': scan,
        'cross_spec_cause': ('跨规格最大值出现在 `pricing_cf_npm`：NPM_lin = PeerReturn − own_R_t 与 PeerReturn 机械共线，'
                             '且 rev（反转因子）与收益结构共线（PeerReturn/NPM_lin/rev 三者同源）→ VIF/condition number 抬升；'
                             '该规格为 CF/NPM 分解的登记诊断规格，不构成 headline 多重共线问题。'),
        'note': ('registry change_log 记录的「VIF max 6.13 / cond 5.88」只覆盖**主规格**（column:ma_teacher，'
                 'per-month max 口径）；跨规格最大值须同时并报（见 scope_cross_spec_max）。'),
    }
    write_json(OUT_DIR / f'pricing_table{sfx}.json', report)
    _write_md(report, hr_report, sfx)
    progress(f'A6+A7 完成：pricing_table{sfx}.{{json,md}} / pricing_horse_race{sfx}.json 落盘'
             f'（headline={ver_id}）')


def _fmt(res):
    c = res['coefs'].get('PeerMAXraw') or res['coefs'].get('PeerMAXraw_mb') or {}
    b, t = c.get('beta'), c.get('t_nw')
    return 'NA' if b is None else f'{b:+.5f} (t={t:+.2f})' if t is not None else f'{b:+.5f}'


def _write_md(report, hr_report, sfx=''):
    ver = report['headline_sample_version']
    other = 'v2' if ver == SAMPLE_VERSION_V3 else ('v1' if ver == SAMPLE_VERSION_V2 else 'v2')
    L = []
    L.append('# spec_no_turnover / pricing_gate 主定价表（Monthly PeerMAX Pricing；sign-01 v3 重建）\n')
    L.append(f"- spec_id: `{report['spec_id']}` | sample_version(headline) = `{report['sample_version']}` "
             f"| v1(未施加 microcap 筛除, superseded) = `{report['sample_version_v1_superseded']}` "
             f"| v2(旧 complete_month 口径, superseded) = `{report['sample_version_v2_superseded']}` "
             f"| NW lag = {report['nw_lag']}")
    L.append(f"- 主公式：`{report['formula']}`")
    L.append(f"- IQR 规则：{report['iqr_scale_rule_id']}；**登记池化值 IQR_pool = {report['iqr_pool_registered']:.8f}**"
             f"（treatment-only 宇宙，全规格共用；各规格自身样本 IQR 仅诊断）")
    L.append(f"- 估计器：{report['estimator']}")
    # Phase 0.3（2026-10-27）：V-sign-6 / N-7 头注（append-only）
    L.append(f"> 样本总体（V-sign-6）：{report['sample_universe_note']}")
    L.append(f"> 控制变量定义（N-7）：{report['control_definition_note']}\n")
    L.append(f'## 四列并报（主规格，headline = `{ver}`，EW / VW）\n')
    L.append('| 列 | 角色 | 窗口（FM 月） | N | 月份 | beta(EW) | NW t | 单侧 p | beta·IQR | beta(VW) | NW t(VW) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for k, v in report['main'].items():
        ew, vw = v['ew']['coefs']['PeerMAXraw'], v['vw']['coefs']['PeerMAXraw']
        L.append(f"| {k} | {v['role']} | {v['fm_window'][0]}..{v['fm_window'][1]} | {v['n_rows']:,} | "
                 f"{v['ew']['n_months_used']} | "
                 f"{ew['beta']:+.5f} | {ew['t_nw']:+.2f} | {ew['p_one_sided']:.4f} | "
                 f"{v['ew']['economic_magnitude_beta_x_iqr']:+.5f} | {vw['beta']:+.5f} | {vw['t_nw']:+.2f} |")
    L.append(f"\n> 窗口说明（防误读）：**四列窗口不同** —— ma_teacher / mb_teacher = 2015-05..2023-12（104 月）；"
             f"ma / mb = 2018-05..2023-12（68 月）。N / 系数 / 显著性**不可按同窗口跨列解读**，"
             f"跨列比较仅在重叠窗口 2018-05..2023-12 内成立。")
    L.append(f'\n## {other} 敏感性（{"未施加 microcap/次新股筛除；ma/mb 另未施加窗口过滤" if other == "v1" else "旧 complete_month 口径（价格非空日数）；其余同主口径"}）\n')
    L.append('| 列 | 窗口（FM 月） | N | 月份 | beta(EW) | NW t | beta(VW) | NW t(VW) |')
    L.append('|---|---|---|---|---|---|---|---|')
    for k, v in report[f'{other}_sensitivity'].items():
        ew, vw = v['ew']['coefs']['PeerMAXraw'], v['vw']['coefs']['PeerMAXraw']
        L.append(f"| {k} | {v['fm_window'][0]}..{v['fm_window'][1]} | {v['n_rows']:,} | "
                 f"{v['ew']['n_months_used']} | {ew['beta']:+.5f} | {ew['t_nw']:+.2f} | "
                 f"{vw['beta']:+.5f} | {vw['t_nw']:+.2f} |")
    L.append('\n### v3 vs v2 对照（主口径 ma_teacher，EW / VW）\n')
    L.append('| 版本 | 样本 | 窗口（FM 月） | N | 月份 | beta(EW) | NW t | beta(VW) | NW t(VW) | 结论方向 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for tag, ver_id, key in [(f'v3(headline)', SAMPLE_VERSION_V3, 'main'),
                             ('v2(旧口径敏感性)', SAMPLE_VERSION_V2, f'{other}_sensitivity')]:
        blk = report['main']['ma_teacher'] if key == 'main' else report[f'{other}_sensitivity']['ma_teacher']
        ew, vw = blk['ew']['coefs']['PeerMAXraw'], blk['vw']['coefs']['PeerMAXraw']
        same_sign = ('同号' if (ew['beta'] > 0) == (vw['beta'] > 0)
                     else f'符号不一致（EW {"正" if ew["beta"] > 0 else "负"} / '
                          f'VW {"正" if vw["beta"] > 0 else "负"}）')
        L.append(f"| {tag} | `{ver_id}` | {blk['fm_window'][0]}..{blk['fm_window'][1]} | {blk['n_rows']:,} | "
                 f"{blk['ew']['n_months_used']} | {ew['beta']:+.5f} | {ew['t_nw']:+.2f} | "
                 f"{vw['beta']:+.5f} | {vw['t_nw']:+.2f} | {same_sign} |")
    _v3 = report['main']['ma_teacher']
    _v2 = report[f'{other}_sensitivity']['ma_teacher']
    _v3_same = ((_v3['ew']['coefs']['PeerMAXraw']['beta'] > 0)
                == (_v3['vw']['coefs']['PeerMAXraw']['beta'] > 0))
    _b3 = _v3['ew']['coefs']['PeerMAXraw']['beta']
    _b2 = _v2['ew']['coefs']['PeerMAXraw']['beta']
    _cross_same = (_b3 > 0) == (_b2 > 0)
    L.append(f"\n> v3 vs v2 一致性：headline 列 EW/VW {'同号' if _v3_same else '**符号不一致**'}；"
             f"跨版本（EW）{'同号' if _cross_same else '**符号翻转**'}"
             f"（v3 {_b3:+.5f} / {other} {_b2:+.5f}）。"
             f"v3 仅因 complete_month 改为有效交易日口径发生样本/量级变化。"
             f"符号是否翻转以本表读数为准，翻转即停机。")
    L.append(f'\n## 主口径全系数（EW，NW lag=6，{ver}）\n')
    L.append('| 变量 | beta | NW SE | NW t | 单侧 p | 双侧 p | 95% CI |')
    L.append('|---|---|---|---|---|---|---|')
    for c, s in report['main']['ma_teacher']['ew']['coefs'].items():
        L.append(f"| {c} | {_n(s['beta'])} | {_n(s['se_nw'])} | {_n(s['t_nw'])} | "
                 f"{_n(s['p_one_sided'],4)} | {_n(s['p_two_sided'],4)} | "
                 f"[{_n(s.get('ci95_low'))}, {_n(s.get('ci95_high'))}] |")
    L.append(f'\n## 后复权敏感性（raw 为主口径，{ver}）\n')
    L.append('| 列 | N | 月份 | beta(EW) | NW t | 单侧 p |')
    L.append('|---|---|---|---|---|---|')
    for k, v in report['post_adjusted'].items():
        c = v['coefs']['PeerMAXraw']
        L.append(f"| {k} | {v['n_rows']:,} | {v['n_months_used']} | {c['beta']:+.5f} | "
                 f"{c['t_nw']:+.2f} | {c['p_one_sided']:.4f} |")
    vs = report['vif_scope']
    L.append('\n## 多重共线（I5：作用域 + 跨规格最大值）\n')
    L.append(f"- 主规格（column:ma_teacher）：per-month max VIF = "
             f"{vs['scope_main_spec_only']['vif_per_month_max']:.2f}"
             f"（{vs['scope_main_spec_only']['vif_per_month_max_variable']}）、"
             f"condition number max = {vs['scope_main_spec_only']['condition_number_max']:.2f}")
    L.append(f"- 跨规格最大（{len(report['main']) + len(hr_report['specs'])} 个规格扫描）："
             f"per-month max VIF = {vs['scope_cross_spec_max']['per_month_max_vif']['value']:.2f}"
             f"（{vs['scope_cross_spec_max']['per_month_max_vif']['spec']} / "
             f"{vs['scope_cross_spec_max']['per_month_max_vif']['variable']}）、"
             f"monthly-mean VIF 最大 = {vs['scope_cross_spec_max']['mean_vif_max']['value']:.2f}"
             f"（{vs['scope_cross_spec_max']['mean_vif_max']['spec']} / "
             f"{vs['scope_cross_spec_max']['mean_vif_max']['variable']}）、"
             f"condition number max = {vs['scope_cross_spec_max']['cond_max']['value']:.2f}"
             f"（{vs['scope_cross_spec_max']['cond_max']['spec']}）")
    L.append(f"- 成因：{vs['cross_spec_cause']}")
    L.append('\n> 四列角色：主口径（中文法 × 老师 MD&A，**主表**）／方法稳健性（TF-IDF × 同语料）／'
             '文本稳健性（中文法 × 自建业务描述语料）／语料对照（TF-IDF × 自建语料）。')
    L.append(f"> horse race 说明：{report['horse_race_note']}")
    L.append('\n> 逐月设计矩阵 rank/df/covariance 状态见 JSON 的 `per_month` 与各列 `all_months_*` 字段。')
    L.append('\n## 逐月设计矩阵验收（headline 四列，EW）\n')
    L.append('| 列 | 月份 | 全月满秩(FULL) | 全月 residual df>0 | 全月协方差有限 | '
             '每月 N（min/median/max，仅 used 月） |')
    L.append('|---|---|---|---|---|---|')
    for k, v in report['main'].items():
        pn = v['per_month_n']
        L.append(f"| {k} | {v['ew']['n_months_used']} | {v['all_months_full_rank']} | "
                 f"{v['all_months_residual_df_positive']} | {v['all_months_covariance_finite']} | "
                 f"{pn['min_used']:,} / {pn['median_used']:,.0f} / {pn['max_used']:,}"
                 f"（跳过月 {pn['months_below_min_n']}） |")
    L.append('')
    p_md = OUT_DIR / f'pricing_table{sfx}.md'
    p_md.write_text('\n'.join(L), encoding='utf-8')
    print(f'  -> {p_md.relative_to(ROOT)}', flush=True)


def _n(x, nd=5):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'NA'
    return f'{x:.{nd}f}'


if __name__ == '__main__':
    main()
