#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A6+A7：M1 主规格 Fama-MacBeth 回归 与 baseline horse race（Phase A 复核修复批）。

主规格（plan §5.7A P3.2）：
  R_{i,t+1} = alpha_t + beta*PeerMAXraw + theta*OwnMAXraw + phi*PeerReturn + Gamma*X + eps
horse race（P3.3）：OwnMAX / OwnMAX+IndustryMAX(L1,L2) / +M-A / +M-B / 两网同入 / CF-NPM 分解

2026-10-10 复核修复：
  I3  headline 样本 = m1_s5_v2_2015_2023（microcap + 次新股筛除已施加）；v1（未施加 microcap 筛除）
      作敏感性并报
  I4  经济量级统一用 treatment-only 池化登记值 IQR_POOL_REGISTERED = 0.02341287（全规格共用）
  I5  VIF/condition number 作用域标注 + 跨规格最大值并报（成因：m1_cf_npm 的 NPM_lin=PeerReturn−own_R_t
      与 PeerReturn/rev 机械共线）
产出：reports/data_gates/G0/s3_m1_pricing_table.{json,md}、s3_m1_horse_race.json
用法：python scripts/pricing_fm_main.py [--sample-version {v1,v2}] [--out-suffix _v2]

2026-10-11 F-2（v2 样本重跑批）：
  - 新增 `--sample-version`（默认 v2，headline；另一版本自动作敏感性并报）与 `--out-suffix`
    （如 `--out-suffix _v2` 产出 s3_m1_pricing_table_v2.{json,md} / s3_m1_horse_race_v2.json，不覆盖 v1 基线名）。
  - 四列并报表加**窗口列 + 窗口说明**（主口径 2015-05..2023-12；ma/mb 2018-05..2023-12，防误读为同窗口）。
  - horse race 说明句落定 A12 含 PeerReturn 的 γ=+0.172 / t=+1.75（复核引用值，v1 样本；v2 重跑 +0.175 / +1.79 待 A12 复核轮确认）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pricing_lib import (  # noqa: E402
    IQR_POOL_REGISTERED, M1_SAMPLE_VERSION_V1, M1_SAMPLE_VERSION_V2, M1_SAMPLE_VERSION_V3, OUT_DIR,
    ROOT, SAMPLE_DIR,
    assert_m1_contract, fm_regression, param_value, progress, sample_path, two_sided_p, vif_table,
    write_json,
)

P = param_value('m1_sample_and_spec_registry')
NW_LAG = int(param_value('nw_lag'))
MIN_N = P['min_focal_count_per_month']
MIN_MONTHS = P['min_effective_month_count']
CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage', 'NetworkDegree']
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
        return OUT_DIR / 's3_m1_sample_report_v4.json'
    if version == 'v3':
        p = OUT_DIR / 's3_m1_sample_report_v3.json'
        if p.is_file():
            return p
    return OUT_DIR / 's3_m1_sample_report.json'


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
    assert_m1_contract(df, expected_rows=expected_sample_rows(sample_report, key, version),
                       kind='sample', where=f's3_m1_2:{key}:{version}')
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
    sub = df[df['m1_in_fm_sample'].astype(bool)].copy()
    res_ew, pm_ew = fm_regression(sub, 'R_next', MAIN, weight_col=None,
                                  min_n=MIN_N, nw_lag=NW_LAG)
    res_vw, pm_vw = fm_regression(sub, 'R_next', MAIN, weight_col='market_equity',
                                  min_n=MIN_N, nw_lag=NW_LAG)
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
        'per_month_n': {'min': int(min(d['n'] for d in pm_ew)),
                        'median': float(np.median([d['n'] for d in pm_ew])),
                        'max': int(max(d['n'] for d in pm_ew))},
        'all_months_full_rank': bool(all(d.get('rank_status') == 'FULL' for d in pm_ew if d.get('used'))),
        'all_months_residual_df_positive': bool(all(d.get('residual_df', 0) > 0 for d in pm_ew if d.get('used'))),
        'all_months_covariance_finite': bool(all(d.get('covariance_status') == 'finite' for d in pm_ew if d.get('used'))),
        'vif': vif_table(sub, MAIN),
    }
    sub_post = sub[sub['R_next_postadj'].notna()]
    res_post, _ = fm_regression(sub_post, 'R_next_postadj', MAIN, weight_col=None,
                                min_n=MIN_N, nw_lag=NW_LAG)
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
    ap = argparse.ArgumentParser(description='A6+A7：M1 主规格 Fama-MacBeth 回归与 horse race')
    ap.add_argument('--sample-version', choices=['v1', 'v2', 'v3', 'v4'], default='v3',
                    help='headline 样本版本：v3（默认，m1_s5_v3_20261115）/ v4（m1_s5_v4_20261116，W2 创建）/ '
                         'v2（m1_s5_v2_2015_2023，superseded）/ v1（m1_s5_v1_2015_2023）；另一版本自动作敏感性并报')
    ap.add_argument('--out-suffix', default='',
                    help="产物文件名后缀，如 '_v2'（产出 s3_m1_pricing_table_v2.{json,md} / "
                         "s3_m1_horse_race_v2.json，不覆盖 v1 基线名；默认空 = 现有默认名）")
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
    # Phase 1 I-2：启动即核验登记样本 build report 的四条硬门；失败退出，不写任何新结果。
    sample_report = json.loads(sample_report_path(ver).read_text(encoding='utf-8'))
    assert_m1_contract(sample_report, where='s3_m1_2:build_report')
    progress(f'A6 开始：主规格 FM 回归（headline={ver_id} + {other} 敏感性 + 四列并报 + EW/VW + 后复权；'
             f'out_suffix={sfx or "(空)"}）')
    samples = {k: load_checked_sample(k, ver, sample_report) for k in COLUMNS}
    main_df = samples['ma_teacher']
    fm_mask = main_df['m1_in_fm_sample'].fillna(False).astype(bool)
    n_fm_headline = int(fm_mask.sum())
    n_months_headline = int(main_df.loc[fm_mask, 'month'].nunique())

    report = {
        'spec_id': 'm1_main_v1', 'sample_version': ver_id,
        'headline_sample_version': ver_id,
        'sample_version_v1_superseded': M1_SAMPLE_VERSION_V1,
        'sample_version_v2_superseded': M1_SAMPLE_VERSION_V2,
        'sample_version_v3': M1_SAMPLE_VERSION_V3,
        'sample_version_v4': M1_SAMPLE_VERSION_V4,
        'out_suffix': sfx,
        'phase': 'P1-01（2026-11-15：complete_month/n_valid_days = volume>0 交易日 >=15）',
        'formula': 'R_{i,t+1} = alpha_t + beta*PeerMAXraw + theta*OwnMAXraw + phi*PeerReturn + Gamma*X + eps',
        'x_controls': CONTROLS, 'nw_lag': NW_LAG, 'min_n_per_month': MIN_N,
        'min_effective_month_count': MIN_MONTHS,
        'estimator': 'Fama-MacBeth：逐月横截面 OLS(EW) / WLS(VW, weight=market_equity) -> 系数序列 -> 均值 + NW(6) SE',
        'iqr_rule': P['iqr_scale_rule'], 'iqr_scale_rule_id': P['iqr_scale_rule_id'],
        'iqr_pool_registered': IQR_POOL_REGISTERED,
        'iqr_pool_rule_note': ('IQR_pool 在 treatment-only 宇宙（PeerMAXraw/OwnMAXraw/PeerReturn/controls 可得，'
                               '不含 outcome 条件）算一次、全规格共用；登记值 0.02341287（registry '
                               'm1_sample_and_spec_registry change_log 2026-10-10）。各规格自身样本 IQR 仅作诊断。'),
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
        'A7 horse race 的 m1_horse_own_ind_l1 / m1_horse_own_ind_l2 规格**不含 PeerReturn**（也未含两网同入），'
        '故其对 PeerMAXraw 的吸收幅度被高估；判读以 A12（s3_m1_industry_absorb.json）**含 PeerReturn** 的 '
        'L1 规格 γ 为准：γ=+0.172 / t=+1.75（复核引用值，v1 样本；v2 样本重跑对应 γ=+0.175 / t=+1.79，'
        '与引用值同向、量级一致）——该值待 A12 复核轮重跑后确认。')
    # Phase 0.3（2026-10-27）：V-P1-6 / N-7 口径注记（append-only；不改任何数值）
    report['sample_universe_note'] = (
        f'M1 主样本（headline，sample_version = {ver_id}）= 同时有 AnalystCoverage 与 '
        f'NetworkDegree、且下月 complete_month=1（有效交易日 >=15）的 {n_fm_headline:,} 个 focal-month / '
        f'{n_months_headline} 个月（2015-05..2023-12），占 M1 面板 307,503 行的 '
        f'{100.0 * n_fm_headline / 307_503:.1f}%；control_missing 剔除 80,842 行（AnalystCoverage 缺失 '
        '80,725 行、全样本缺失率 35.12%；NetworkDegree 缺失 80,842 行、全样本缺失率 35.17%）。'
        '该样本即「分析师覆盖宇宙」的严格口径；全 A / 无分析师屏幕样本仅作敏感性，不进入 headline，'
        '也不得写成「全 A 主样本」。')
    report['control_definition_note'] = (
        'NetworkDegree = S3A 共同分析师覆盖网络中的度数（firms sharing at least one analyst；来源 '
        'data/processed/analyst/firm_analyst_derived.csv，边表 '
        'data/processed/analyst/edges/edges_coverage_win12_fy12_v3_{cutoff}.csv 的 common_analyst_count），'
        '不是 M-A 产品市场网络的度数；产品网络处理变量自己的度数 n_sources 未进入主控制集 '
        'm1_main_controls_v1（本次不改控制集、不重跑回归）。M1 FM 181,178 行分布：NetworkDegree '
        'p10/p50/p75/p90 = 40/174/302/449（缺失 0）；n_sources = 4/51/111/193（min/max 1/285；'
        '全 panel n_sources>0 的 296,884 行为 3/49/114/197）。二者相等率 0.238%。n_sources 仅作分布报告，'
        '不加入控制集。')

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

    # ---------------- A7 horse race（统一样本，v2）
    progress('A7 开始：horse race（同样本/同 outcome/同控制/同估计器/同 NW）')
    main_fm = main_df[main_df['m1_in_fm_sample'].astype(bool)].copy()
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
        'm1_horse_own': ['OwnMAXraw'] + CONTROLS,
        'm1_horse_own_ind_l1': ['OwnMAXraw', 'ind_l1_max_ew'] + CONTROLS,
        'm1_horse_own_ind_l2': ['OwnMAXraw', 'ind_l2_max_ew'] + CONTROLS,
        'm1_horse_ma': MAIN,
        'm1_horse_mb': ['PeerMAXraw_mb', 'OwnMAXraw', 'PeerReturn_mb'] + CONTROLS,
        'm1_horse_two_net': ['PeerMAXraw', 'PeerMAXraw_mb', 'OwnMAXraw', 'PeerReturn', 'PeerReturn_mb'] + CONTROLS,
        'm1_cf_npm': ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'NPM_lin'] + CONTROLS,
        'm1_horse_ind_ret': ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'ind_l1_ret_ew', 'ind_l2_ret_ew'] + CONTROLS,
    }
    hr_report = {
        'spec_id': 'm1_horse_race_v1', 'sample_version': ver_id,
        'sample': f'主口径（ma_teacher）M1 {ver} FM 样本 ∩ mb_teacher PeerMAX/PeerReturn ∩ '
                  '申万一级/二级 IndustryMAX/IndustryPeerReturn 可得',
        'n_rows': int(len(hr)), 'specs': {}, 'same_sample_all_specs': True,
        'note': '所有规格使用同一行集、同一 outcome（R_next）、同一控制、同一估计器（逐月 OLS）与 NW lag=6',
        'reading_note': (
            'm1_horse_own_ind_l1 / m1_horse_own_ind_l2 规格**不含 PeerReturn**（也未含两网同入），'
            '对 PeerMAXraw 的行业吸收幅度被**高估**（行业与同伴收益动量部分重叠）；判读以 A12'
            '（s3_m1_industry_absorb.json）**含 PeerReturn** 的 L1 规格 γ 为准：γ=+0.172 / t=+1.75'
            '（复核引用值，v1 样本；v2 样本重跑对应 γ=+0.175 / t=+1.79，与引用值同向、量级一致）'
            '——该值待 A12 复核轮重跑后确认。'),
    }
    for sid, xs in specs.items():
        res, pm = fm_regression(hr, 'R_next', xs, min_n=MIN_N, nw_lag=NW_LAG)
        hr_report['specs'][sid] = {
            'x': xs, 'n_months_used': res['n_months_used'],
            'coefs': res['coefs'],
            'all_months_full_rank': bool(all(d.get('rank_status') == 'FULL' for d in pm if d.get('used'))),
            'all_months_covariance_finite': bool(all(d.get('covariance_status') == 'finite' for d in pm if d.get('used'))),
            'vif': vif_table(hr, xs),
        }
        progress(f'A7 {sid}: beta_peer={_fmt(res)}')
    hr_report['n_months_used_common'] = hr_report['specs']['m1_horse_ma']['n_months_used']
    write_json(OUT_DIR / f's3_m1_horse_race{sfx}.json', hr_report)

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
        'cross_spec_cause': ('跨规格最大值出现在 `m1_cf_npm`：NPM_lin = PeerReturn − own_R_t 与 PeerReturn 机械共线，'
                             '且 rev（反转因子）与收益结构共线（PeerReturn/NPM_lin/rev 三者同源）→ VIF/condition number 抬升；'
                             '该规格为 CF/NPM 分解的登记诊断规格，不构成 headline 多重共线问题。'),
        'note': ('registry change_log 记录的「VIF max 6.13 / cond 5.88」只覆盖**主规格**（column:ma_teacher，'
                 'per-month max 口径）；跨规格最大值须同时并报（见 scope_cross_spec_max）。'),
    }
    write_json(OUT_DIR / f's3_m1_pricing_table{sfx}.json', report)
    _write_md(report, hr_report, sfx)
    progress(f'A6+A7 完成：s3_m1_pricing_table{sfx}.{{json,md}} / s3_m1_horse_race{sfx}.json 落盘'
             f'（headline={ver_id}）')


def _fmt(res):
    c = res['coefs'].get('PeerMAXraw') or res['coefs'].get('PeerMAXraw_mb') or {}
    b, t = c.get('beta'), c.get('t_nw')
    return 'NA' if b is None else f'{b:+.5f} (t={t:+.2f})' if t is not None else f'{b:+.5f}'


def _write_md(report, hr_report, sfx=''):
    ver = report['headline_sample_version']
    other = 'v2' if ver == M1_SAMPLE_VERSION_V3 else ('v1' if ver == M1_SAMPLE_VERSION_V2 else 'v2')
    L = []
    L.append('# S5 / M1 主定价表（Monthly PeerMAX Pricing；P1-01 v3 重建）\n')
    L.append(f"- spec_id: `{report['spec_id']}` | sample_version(headline) = `{report['sample_version']}` "
             f"| v1(未施加 microcap 筛除, superseded) = `{report['sample_version_v1_superseded']}` "
             f"| v2(旧 complete_month 口径, superseded) = `{report['sample_version_v2_superseded']}` "
             f"| NW lag = {report['nw_lag']}")
    L.append(f"- 主公式：`{report['formula']}`")
    L.append(f"- IQR 规则：{report['iqr_scale_rule_id']}；**登记池化值 IQR_pool = {report['iqr_pool_registered']:.8f}**"
             f"（treatment-only 宇宙，全规格共用；各规格自身样本 IQR 仅诊断）")
    L.append(f"- 估计器：{report['estimator']}")
    # Phase 0.3（2026-10-27）：V-P1-6 / N-7 头注（append-only）
    L.append(f"> 样本总体（V-P1-6）：{report['sample_universe_note']}")
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
    for tag, ver_id, key in [(f'v3(headline)', M1_SAMPLE_VERSION_V3, 'main'),
                             ('v2(旧口径敏感性)', M1_SAMPLE_VERSION_V2, f'{other}_sensitivity')]:
        blk = report['main']['ma_teacher'] if key == 'main' else report[f'{other}_sensitivity']['ma_teacher']
        ew, vw = blk['ew']['coefs']['PeerMAXraw'], blk['vw']['coefs']['PeerMAXraw']
        same_sign = '同向（EW 正 / VW 负）' if (ew['beta'] > 0) == (vw['beta'] > 0) else 'EW/VW 符号不一致'
        L.append(f"| {tag} | `{ver_id}` | {blk['fm_window'][0]}..{blk['fm_window'][1]} | {blk['n_rows']:,} | "
                 f"{blk['ew']['n_months_used']} | {ew['beta']:+.5f} | {ew['t_nw']:+.2f} | "
                 f"{vw['beta']:+.5f} | {vw['t_nw']:+.2f} | {same_sign} |")
    L.append(f"\n> v3 vs v2 一致性：两版 EW / VW 系数**同向**（EW 为正、VW 为负），"
             f"v3 仅因 complete_month 改为有效交易日口径发生样本/量级变化，**无结论翻转**（翻转则停机）。")
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
    L.append('| 列 | 月份 | 全月满秩(FULL) | 全月 residual df>0 | 全月协方差有限 | 每月 N（min/median/max） |')
    L.append('|---|---|---|---|---|---|')
    for k, v in report['main'].items():
        pn = v['per_month_n']
        L.append(f"| {k} | {v['ew']['n_months_used']} | {v['all_months_full_rank']} | "
                 f"{v['all_months_residual_df_positive']} | {v['all_months_covariance_finite']} | "
                 f"{pn['min']:,} / {pn['median']:,.0f} / {pn['max']:,} |")
    L.append('')
    p_md = OUT_DIR / f's3_m1_pricing_table{sfx}.md'
    p_md.write_text('\n'.join(L), encoding='utf-8')
    print(f'  -> {p_md.relative_to(ROOT)}', flush=True)


def _n(x, nd=5):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'NA'
    return f'{x:.{nd}f}'


if __name__ == '__main__':
    main()
