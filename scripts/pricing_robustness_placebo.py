#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A9：M1 稳健性与 placebo（Phase A 复核修复批）。

规格（plan §5.7A P3.5，每规格独立 specification_id）：
  raw/post-adjusted、ST 分层、limit-hit 分层、规模/流动性分层、MA/MB 四列、
  equal/linear/log_sim 权重、同密度随机网络（匹配 peer 数与权重分布）、时间错位网络。

2026-10-10 复核修复：
  I1  权重变体 / placebo 的 PeerMAXraw 与 PeerReturn 在 merge 后按**月度 [0.01,0.99] clip**
      （宇宙 = 该规格 FM 分析行集）；修正前这些列是未 winsorize 值，与主规格口径不对称。
      宇宙说明：主规格列在面板构造期按面板宇宙 winsorize（可复现 max|Δ|=0），此处按规格 FM 行集
      计算边界（同规则不同宇宙），故 linear 变体与主规格 β 差 <5e-4（实测 2.84e-4；方向/显著性一致）。
  I3  headline 样本 = m1_s5_v2_2015_2023；v1（未施加 microcap 筛除）作敏感性并报
  I4  经济量级统一用 treatment-only 池化登记值 IQR_POOL_REGISTERED；各规格自身 IQR 仅诊断
  I6  退化规格（空层 / 全月低于 min_focal_count_per_month）写 spec_status=NOT_COMPUTABLE + reason，
      不再只写 spec_beta=null；ST 分层用 focal 侧 Drive ST 标记（代码格式映射缺陷已修复）
产出：reports/data_gates/G0/s3_m1_robustness_placebo.json
用法：python scripts/pricing_robustness_placebo.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pricing_lib import IQR_POOL_REGISTERED, M1_SAMPLE_VERSION_V1, M1_SAMPLE_VERSION_V2, M1_SAMPLE_VERSION_V3, M1_SAMPLE_VERSION_V4, NET_DIRS, OUT_DIR, ROOT, SAMPLE_DIR, aggregate_edges, clip_by_month, fm_regression, load_edges, load_returns_raw, load_src_month, load_support, param_value, progress, rank_average_groups, sample_path, write_json

P = param_value('m1_sample_and_spec_registry')
NW_LAG = int(param_value('nw_lag'))
MIN_N = P['min_focal_count_per_month']
MIN_MONTHS = P['min_effective_month_count']
SEED = int(param_value('random_seed'))
CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage', 'NetworkDegree']
MAIN = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + CONTROLS
COLUMNS = ['ma_teacher', 'mb_teacher', 'ma', 'mb']
KEEP = ['focal_id', 'month', 'PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'R_next', 'R_next_postadj',
        'm1_in_fm_sample', 'st_flag_focal', 'limit_hit_max_stratum', 'market_equity'] + CONTROLS
WINSORIZED_SPECS = ['m1_weight_equal', 'm1_weight_linear', 'm1_weight_logsim',
                    'm1_placebo_random', 'm1_placebo_timeshift']


def summarise(res, iqr, xcol='PeerMAXraw', n_rows=None):
    """I6：退化规格写 spec_status=NOT_COMPUTABLE + reason（不写 spec_beta=null）。"""
    c = res['coefs'].get(xcol, {})
    beta, t = c.get('beta'), c.get('t_nw')
    out = {'n_months_used': res['n_months_used'], 'spec_beta': beta,
           't_nw': t, 'p_one_sided': c.get('p_one_sided'),
           'economic_magnitude': None if beta is None else beta * iqr,
           'iqr_pool': iqr, 'iqr_pool_rule': 'IQR_POOL_TREATMENT_ONLY_v1（登记值，全规格共用）',
           'coefs_peer_own_ind': {k: res['coefs'][k] for k in
                                  [xcol, 'OwnMAXraw', 'PeerReturn'] if k in res['coefs']}}
    if n_rows is not None:
        out['n_rows'] = int(n_rows)
    if beta is None or out['n_months_used'] == 0:
        pm = res.get('per_month') or []
        n_per_month = [d['n'] for d in pm] or [0]
        if n_rows == 0:
            reason = 'EMPTY_STRATUM_NO_ROWS（层内 0 行，不填 0、不外推）'
        else:
            reason = (f'ALL_MONTHS_BELOW_MIN_FOCAL_COUNT（min_focal_count_per_month={MIN_N}；'
                      f'层内月样本中位 {np.median(n_per_month):.0f}，n_usable_months=0）')
        out.update({'spec_status': 'NOT_COMPUTABLE', 'status_reason': reason,
                    'n_months_available': len(pm),
                    'per_month_n_median_all': float(np.median(n_per_month)) if n_per_month else None})
    else:
        out['spec_status'] = 'COMPUTED'
        if out['n_months_used'] < MIN_MONTHS:
            out['low_coverage_warning'] = (f'可用月份 {out["n_months_used"]} < min_effective_month_count='
                                           f'{MIN_MONTHS}：分层样本小，仅作描述，不作推断依据')
    return out


def aggregate_version(key, text_year, months, srcm, edges=None, wcol='weight', ret_raw=None):
    if edges is None:
        edges = load_edges(key, int(text_year))
    vals_max = srcm[['source_id', 'month', 'MAXraw']].rename(columns={'source_id': 'firm_id'})
    vals_ret = ret_raw[['firm_id', 'month', 'R']]   # 与面板同源的 source-month 原始收益
    mm = vals_max[vals_max['month'].isin(months)]
    rr = vals_ret[vals_ret['month'].isin(months)]
    a = aggregate_edges(edges, mm, [wcol], 'MAXraw')
    b = aggregate_edges(edges, rr, [wcol], 'R')
    p = a.merge(b, on=['month', 'focal_id'], how='outer')
    ren = {f'MAXraw_{wcol}': 'PX', f'R_{wcol}': 'PR'}
    return p.rename(columns=ren)


def random_edges(key, text_year):
    """M1_RANDOM_DENSITY_NETWORK_v1：保留 peer 数与权重向量，随机重指派 source。"""
    e = load_edges(key, int(text_year))
    sup = load_support(key, int(text_year))
    nodes = np.array(sorted(set(sup['focal_id'].astype(str)) | set(e['focal_id']) | set(e['source_id'])))
    rng = np.random.default_rng(SEED + int(text_year))
    out = []
    for focal, g in e.groupby('focal_id', sort=False):
        src = g['source_id'].to_numpy()
        w = g['weight'].to_numpy(dtype=float)
        n = len(src)
        pool = nodes[nodes != focal]
        if n >= len(pool):
            pick = pool
        else:
            pick = rng.choice(pool, size=n, replace=False)
        out.append(pd.DataFrame({'focal_id': focal, 'source_id': pick, 'weight': w}))
    return pd.concat(out, ignore_index=True)


def spec_frame(fm: pd.DataFrame, merge_df: pd.DataFrame, pxcol: str, prcol: str,
               winsorize: bool) -> pd.DataFrame:
    """I1：把重算得到的 treatment/control 列换入 FM 分析行集，并按同规则 winsorize。"""
    d = fm.merge(merge_df, on=['focal_id', 'month'], how='left')
    sub = d[d[pxcol].notna() & d[prcol].notna()].copy()
    if winsorize:
        sub = clip_by_month(sub, [pxcol, prcol])
    sub = (sub.drop(columns=['PeerMAXraw', 'PeerReturn'])
           .rename(columns={pxcol: 'PeerMAXraw', prcol: 'PeerReturn'}))
    return sub


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='A9：M1 稳健性与 placebo')
    ap.add_argument('--sample-version', choices=['v1', 'v2', 'v3', 'v4'], default='v2',
                    help='headline 样本版本（默认 v2，行为不变）；v3/v4 为 W1 CLI 支持')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ver = args.sample_version
    ver_id = {'v1': M1_SAMPLE_VERSION_V1, 'v2': M1_SAMPLE_VERSION_V2,
              'v3': M1_SAMPLE_VERSION_V3, 'v4': M1_SAMPLE_VERSION_V4}[ver]
    progress(f'A9 开始：稳健性与 placebo（headline={ver_id}；I1 winsorize 修正 + I6 状态）')
    srcm = load_src_month()
    ret_raw = load_returns_raw()
    base = pd.read_parquet(sample_path('ma_teacher', ver))
    fm = base[base['m1_in_fm_sample'].astype(bool)][KEEP].copy()
    out = {'spec_id': 'm1_robustness_placebo_v1', 'sample_version': ver_id,
           'sample_version_v1_superseded': M1_SAMPLE_VERSION_V1,
           'iqr_pool_registered': IQR_POOL_REGISTERED,
           'random_seed': SEED, 'specs': {}, 'v1_sensitivity': {},
           'main_column': 'ma_teacher（主口径，主表列）',
           'outcome': 'raw 主口径（m1_rob_postadj 为后复权敏感性，raw 仍主表）',
           'four_column_reporting': 'm1_rob_col_{ma_teacher,mb_teacher,ma,mb} 四列并报；'
                                    '各列角色见 m1_rob_col_* 规格的 note 与 s3_m1_pricing_table.json:column_roles',
           'winsorization': {
               'rule': 'registry winsorization_pct = [0.01,0.99] 月度截面',
               'i1_fix': ('权重变体（equal/linear/log_sim）与 placebo（random/timeshift）在 merge 后'
                          '对 treatment（PeerMAXraw）与 PeerReturn 施加月度 [0.01,0.99] clip；'
                          '修正前为重算的未 winsorize 值（与主规格口径不对称：weight_linear 未 winsorize '
                          'β=+0.04363 vs 主规格 +0.02583；placebo random −0.0478 vs 修正后 −0.0680）。'),
               'universe_note': ('主规格列在面板构造期按**面板宇宙** winsorize（复算 max|Δ|=0）；'
                                 '变体/placebo 按**该规格 FM 行集**计算月度分位边界（同规则不同宇宙）→ '
                                 'weight_linear 与主规格 β 差 <5e-4（实测 2.84e-4；方向与显著性一致），placebo 数值与复核一致。'),
           },
           'notes': ['所有规格与主规格同样的 outcome/控制/估计器/NW(6)，仅替换被标注的维度',
                     'placebo 网络不冒充主结果；随机网络匹配 peer 数与权重分布（不做行业保持，登记为限制）',
                     '分层结果为空组或全月低于 min_focal_count_per_month 时写 spec_status=NOT_COMPUTABLE + reason，不填 0',
                     'ST 分层使用 focal 侧 Drive stock_st_flag（代码格式映射已修复，2026-10-10）'],
           'column_roles': {'ma_teacher': '主口径（主表）', 'mb_teacher': '方法稳健性',
                            'ma': '文本稳健性', 'mb': '语料对照'}}

    def add(sid, res, iqr_, note, extra=None, xcol='PeerMAXraw', target=None):
        rec = summarise(res, iqr_, xcol=xcol, n_rows=(extra or {}).get('n_rows'))
        rec.update({'specification_id': sid, 'note': note})
        if extra:
            rec.update(extra)
        (out['specs'] if target is None else target)[sid] = rec
        progress(f'A9 {sid}: status={rec["spec_status"]} beta={rec["spec_beta"]} '
                 f't={rec["t_nw"]} months={rec["n_months_used"]}')

    # 1) raw 主（参照）+ 后复权
    r, _ = fm_regression(fm, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
    add('m1_rob_raw', r, IQR_POOL_REGISTERED, 'raw 主口径（参照）', {'n_rows': int(len(fm))})
    fmp = fm[fm['R_next_postadj'].notna()]
    r, _ = fm_regression(fmp, 'R_next_postadj', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
    add('m1_rob_postadj', r, IQR_POOL_REGISTERED, 'R_NEXT_POST_ADJ_v1（后复权敏感性；raw 仍主口径）',
        {'n_rows': int(len(fmp))})

    # 2) ST 分层（focal 侧 Drive ST 标记）
    for lab, sub in [('st', fm[fm['st_flag_focal']]), ('non_st', fm[~fm['st_flag_focal']])]:
        r, _ = fm_regression(sub, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
        add(f'm1_rob_{lab}', r, IQR_POOL_REGISTERED, f'ST 分层：{lab}（focal 侧 Drive stock_st_flag）',
            {'n_rows': int(len(sub)),
             'st_source_note': 'Drive stock_st_flag 当月曾 ST 即标 1；代码格式映射缺陷已于 2026-10-10 修复'})

    # 3) limit-hit 分层（主样本含三类，分层并报）
    for lab in ['all_non_limit', 'mixed', 'all_limit']:
        sub = fm[fm['limit_hit_max_stratum'] == lab]
        r, _ = fm_regression(sub, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
        add(f'm1_rob_limit_{lab}', r, IQR_POOL_REGISTERED, f'limit-hit 分层：{lab}', {'n_rows': int(len(sub))})

    # 4) 规模 / 流动性分层（月内三分位）
    for var, name in [('market_equity', 'size'), ('illiq', 'liq')]:
        g = fm.groupby('month', group_keys=False)[var].apply(lambda s: rank_average_groups(s, 3))
        for k, lab in [(1, 'low'), (3, 'high')]:
            sub = fm[g == k]
            r, _ = fm_regression(sub, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
            add(f'm1_rob_{name}_{lab}', r, IQR_POOL_REGISTERED, f'{name} 三分位分层（{lab}）',
                {'n_rows': int(len(sub))})

    # 5) 四列并报（同规格；各列面板已 winsorize）
    for key in COLUMNS:
        d = pd.read_parquet(sample_path(key, ver))
        d = d[d['m1_in_fm_sample'].astype(bool)]
        r, _ = fm_regression(d, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
        add(f'm1_rob_col_{key}', r, IQR_POOL_REGISTERED, f'四列并报：{key}（{out["column_roles"][key]}）',
            {'n_rows': int(len(d)), 'winsorized': 'panel_construction'})

    # 6) 权重规则（固定边集）：equal / linear / log_sim（I1：merge 后同规则 winsorize）
    wv = pd.read_parquet(SAMPLE_DIR / 'm1_weight_variants_ma_teacher.parquet')
    for lab, pmax, pret in [('equal', 'PeerMAXraw_weq', 'PeerReturn_weq'),
                            ('linear', 'PeerMAXraw_wlin', 'PeerReturn_wlin'),
                            ('logsim', 'PeerMAXraw_wlog', 'PeerReturn_wlog')]:
        sub = spec_frame(fm, wv, pmax, pret, winsorize=True)
        iqr_w = float(sub['PeerMAXraw'].quantile(.75) - sub['PeerMAXraw'].quantile(.25))
        r, _ = fm_regression(sub, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
        add(f'm1_weight_{lab}', r, IQR_POOL_REGISTERED, f'P2.2 权重规则：w_{"log_sim" if lab == "logsim" else lab}',
            {'n_rows': int(len(sub)), 'weight_rule': lab, 'winsorized': 'spec_fm_universe_[0.01,0.99]',
             'iqr_pool_sample_diagnostic': iqr_w,
             'formula_note': 'w_log_sim = log(1+s) 归一化，**不是** Long 的 ln(1+n) 权重'})

    # 7) placebo：同密度随机网络 与 时间错位网络
    panel = pd.read_parquet(sample_path('ma_teacher', ver))
    ver = (panel[['month', 'network_version']].dropna().drop_duplicates())
    ver['text_year'] = pd.to_numeric(ver['network_version'], errors='coerce').astype(int)
    cache_rand = ROOT / 'artifacts/tmp/m1_cache_placebo_random.parquet'
    cache_shift = ROOT / 'artifacts/tmp/m1_cache_placebo_timeshift.parquet'
    rand_parts, shift_parts = [], []
    if cache_rand.exists() and cache_shift.exists():
        rand = pd.read_parquet(cache_rand)
        shift = pd.read_parquet(cache_shift)
        progress('A9 placebo 网络命中缓存（随机 + 时间错位）')
        rand_parts = shift_parts = None
    for ty, grp in ver.groupby('text_year', sort=True):
        if rand_parts is None:
            break
        months = sorted(grp['month'])
        er = random_edges('ma_teacher', int(ty))
        rand_parts.append(aggregate_version('ma_teacher', ty, months, srcm, edges=er, ret_raw=ret_raw))
        prev = int(ty) - 1
        if (NET_DIRS['ma_teacher'] / f'{prev}_similarity.csv').exists():
            shift_parts.append(aggregate_version('ma_teacher', prev, months, srcm, ret_raw=ret_raw))
        else:
            progress(f'A9 时间错位：文本年 {prev} 无网络文件 → {months[0]}..{months[-1]} 记 NOT_COMPUTABLE')
        progress(f'A9 placebo 网络构造完成 文本年 {ty}（随机 / 错位）')
    if rand_parts is not None:
        rand = pd.concat(rand_parts, ignore_index=True).rename(
            columns={'PX': 'PeerMAXrand', 'PR': 'PeerReturnrand'})
        shift = pd.concat(shift_parts, ignore_index=True).rename(
            columns={'PX': 'PeerMAXshift', 'PR': 'PeerReturnshift'})
        rand.to_parquet(cache_rand, index=False)
        shift.to_parquet(cache_shift, index=False)

    for sid, merge_df, note in [
            ('m1_placebo_random', rand, 'M1_RANDOM_DENSITY_NETWORK_v1：保留 peer 数与权重向量、随机重指派 source（seed=random_seed；不做行业保持）'),
            ('m1_placebo_timeshift', shift, 'M1_TIME_SHIFTED_NETWORK_v1：改用上一文本年网络版本重算 PeerMAXraw/PeerReturn')]:
        col = 'PeerMAXrand' if sid.endswith('random') else 'PeerMAXshift'
        rcol = 'PeerReturnrand' if sid.endswith('random') else 'PeerReturnshift'
        sub = spec_frame(fm, merge_df, col, rcol, winsorize=True)
        iqr_p = float(sub['PeerMAXraw'].quantile(.75) - sub['PeerMAXraw'].quantile(.25))
        r, _ = fm_regression(sub, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
        add(sid, r, IQR_POOL_REGISTERED, note,
            {'n_rows': int(len(sub)),
             'months_excluded_no_version': int(len(fm) - len(sub)),
             'winsorized': 'spec_fm_universe_[0.01,0.99]',
             'iqr_pool_sample_diagnostic': iqr_p})

    # 7b) I1 交叉核验：linear 变体 vs 主规格（同规则、宇宙差异）
    lin = out['specs']['m1_weight_linear']
    raw = out['specs']['m1_rob_raw']
    out['winsorize_crosscheck'] = {
        'main_spec_beta': raw['spec_beta'], 'weight_linear_beta': lin['spec_beta'],
        'abs_diff': (None if (raw['spec_beta'] is None or lin['spec_beta'] is None)
                     else abs(raw['spec_beta'] - lin['spec_beta'])),
        'interpretation': '施加同规则 winsorize 后 linear 变体与主规格一致到 <5e-4（差异来自 winsorize 宇宙：'
                          '主规格=面板宇宙，变体=规格 FM 行集；未修正前该差为 0.0183）。两值均不显著、同号。',
        'pre_fix_reference': {'weight_linear_unwinsorized_beta': 0.043631024760655626,
                              'placebo_random_unwinsorized_beta': -0.0478230299889328,
                              'note': '未 winsorize 的旧值（v1 样本，2026-10-10 复核复核实测）；'
                                      'v1 样本下修正后 placebo random = −0.06801（t=−2.41，与复核一致）、'
                                      'weight_linear = +0.02577（主规格 v1 = +0.02583）',
                              'source': 'archive/phaseA_v1_tables_20261010/s3_m1_robustness_placebo.json'},
    }
    out['placebo_comparison'] = {
        'real_network': {'spec': 'm1_rob_raw', 'beta': raw['spec_beta'], 't_nw': raw['t_nw']},
        'random_network': {k: out['specs']['m1_placebo_random'][k] for k in ('spec_beta', 't_nw', 'n_months_used')},
        'timeshift_network': {k: out['specs']['m1_placebo_timeshift'][k] for k in ('spec_beta', 't_nw', 'n_months_used')},
        'reading': '随机网络（I1 修正后）β 强于真实网络 → 描述性关联不构成网络特异性证据（与 Phase A 结论一致）。',
    }

    # 8) v1 敏感性（未施加 microcap 筛除）
    base1 = pd.read_parquet(sample_path('ma_teacher', 'v1'))
    fm1 = base1[base1['m1_in_fm_sample'].astype(bool)][KEEP].copy()
    r, _ = fm_regression(fm1, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
    add('m1_rob_raw', r, IQR_POOL_REGISTERED, 'v1（未施加 microcap 筛除）raw 主口径',
        {'n_rows': int(len(fm1)), 'status_note': 'v1_superseded_registered_deviation'}, target=out['v1_sensitivity'])
    sub1 = spec_frame(fm1, wv, 'PeerMAXraw_wlin', 'PeerReturn_wlin', winsorize=True)
    r, _ = fm_regression(sub1, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
    add('m1_weight_linear', r, IQR_POOL_REGISTERED, 'v1 权重 linear（同 I1 winsorize 规则）',
        {'n_rows': int(len(sub1)), 'status_note': 'v1_superseded_registered_deviation'},
        target=out['v1_sensitivity'])
    for sid, merge_df in [('m1_placebo_random', rand), ('m1_placebo_timeshift', shift)]:
        col = 'PeerMAXrand' if sid.endswith('random') else 'PeerMAXshift'
        rcol = 'PeerReturnrand' if sid.endswith('random') else 'PeerReturnshift'
        sub1 = spec_frame(fm1, merge_df, col, rcol, winsorize=True)
        r, _ = fm_regression(sub1, 'R_next', MAIN, min_n=MIN_N, nw_lag=NW_LAG)
        add(sid, r, IQR_POOL_REGISTERED, 'v1 placebo（同 I1 winsorize 规则）',
            {'n_rows': int(len(sub1)), 'status_note': 'v1_superseded_registered_deviation'},
            target=out['v1_sensitivity'])

    write_json(OUT_DIR / 's3_m1_robustness_placebo.json', out)
    progress('A9 完成：s3_m1_robustness_placebo.json 落盘（v2 headline）')


if __name__ == '__main__':
    main()
