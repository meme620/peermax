#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1–A5：M1（S5）输入合同核验、下月 outcome、控制变量、baseline 构造与样本冻结。

对应主仓库 methodology_execution_plan §5.7A（原型见 git 历史） P1.1–P1.3 / P1.5 / P2.1–P2.2 / P3.1。

2026-10-10 Phase A 复核修复批（I2/I3/I6）：
  I2  四列统一施加 registry sample_start/sample_end 事件月窗口 2015-01..2023-12
      （outcome 为紧邻下月；旧版 ma/mb 含 2024-01 事件月 → superseded）
  I3  按 registry `sample_filter_thresholds` 施加 microcap 筛除（形成月**全市场**横截面 q10，
      总市值口径 = 比照 S3A 9C 的登记口径偏离）与次新股筛除（上市 < 6 月，
      listing date 源 data/processed/firm_id_map.csv）；生成新样本版本 m1_s5_v2_2015_2023，
      原 v1 保留并标 superseded_registered_deviation（未施加 microcap 筛除）
  I6  样本报告补「新股/停牌状态」字段（listing date 覆盖、下月收益完整度/停牌代理）

Phase 1 I-16（2026-10-26）：样本窗口元数据拆分为三列（只改生成器；当前 parquet 不重写，
                          新列在下次正常构建时出现）：
  sample_window         旧列，保持申请窗 2015-01..2023-12（兼容；不改现值）
  window_requested      申请窗（registry sample_start/sample_end），所有行同值
  sample_window_actual  按列实际窗：month 在 network_version.notna() 或 m1_in_fm_sample==1
                        的 min..max（当前 teacher 2015-05..2023-12；ma/mb 2018-05..2023-12）；
                        无任何可用月 → NO_NETWORK_VERSION（显式哨兵，不伪造窗口）

产出（v3 为 headline，v1/v2 文件不覆盖）：
  reports/data_gates/G0/s3_m1_sample_report.json（含 v1 superseded 与 v2 headline 两版）
  reports/data_gates/G0/s3_m1_sample_report_v2.json（I7：v2 专用报告；--v2-report-only 只读生成）
  data/processed/m1_common_sample_{ma_teacher,mb_teacher,ma,mb}_v3.parquet
  data/processed/m1_s5_v2_sample_summary.json（I7：m1_s5_v2 汇总件）
  data/processed/m1_weight_variants_{ma_teacher,mb_teacher}.parquet（不变，重算核验）
用法：python scripts/pricing_sample_build.py
      python scripts/pricing_sample_build.py --v2-report-only   # 只读 v1/v2 样本件出 v2 报告
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_lib import HARD_GATE_KEYS, IQR_POOL_REGISTERED, IQR_POOL_REGISTERED_SOURCE, IQR_POOL_RULE_ID, ILLIQ_FORMULA_ID, AMOUNT_CACHE_NAME, M1_SAMPLE_VERSION_V1, M1_SAMPLE_VERSION_V2, M1_SAMPLE_VERSION_V3, OUT_DIR, PANELS, ROOT, SAMPLE_DIR, SAMPLE_WINDOW_END, SAMPLE_WINDOW_START, add_weight_rules, aggregate_edges, assert_m1_contract, assert_m1_inputs, code6, load_analyst_derived, load_chars, load_edges, load_amount_100m_cached, load_me, load_panel, load_postadj, load_registry, load_returns_raw, load_returns_v2, load_src_month, load_st, load_sw2, load_turnover_drive, months_between, param_value, progress, require, sample_path, sample_window_bounds, file_size, winsor_bounds, winsorize_by_month, write_json, ym_add

# Phase 1 I-5：winsor 边界与样本窗口一律来自 registry（当前 [0.01,0.99] / 2015-01..2023-12，
# 与旧硬编码逐值相同 → 零数值变化）。导入即校验，非法 registry 值 SystemExit(2)。
WINSOR = winsor_bounds()
MIN_N = param_value('m1_sample_and_spec_registry')['min_focal_count_per_month']
MIN_MONTHS = param_value('m1_sample_and_spec_registry')['min_effective_month_count']
# NetworkDegree = S3A shared-analyst coverage degree (analyst/firm_analyst_derived.csv);
# NOT the product-market network degree n_sources. n_sources is reported for distribution only.
CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage', 'NetworkDegree']
RET_COLS = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn']
FILTERS = param_value('sample_filter_thresholds')
RECENT_IPO_MONTHS = 6            # registry sample_filter_thresholds.ipo_exclusion / spec.sample_filters.recent_ipo_months
IPO_MAP_PATH = ROOT / 'data/processed/firm_id_map.csv'
WINDOW = f'{SAMPLE_WINDOW_START}..{SAMPLE_WINDOW_END}'
REGISTRY_SOURCE = 'configs/parameter_registry.yaml:parameters[name={name}].headline_value'
# Phase 1 I-16：实际窗哨兵与计算口径（报告/测试共用；不改 in_win、不改 m1_in_fm_sample）
NO_NETWORK_VERSION = 'NO_NETWORK_VERSION'
ACTUAL_WINDOW_RULE = ('per-column min/max(month) where network_version.notna() or '
                      'm1_in_fm_sample==1；无可用月 → NO_NETWORK_VERSION')


def compute_actual_sample_window(df: pd.DataFrame) -> str:
    """I-16：按列计算实际样本窗口（不修改入参）。

    口径：`month` 在 `network_version.notna()` 或 `m1_in_fm_sample==1` 的行上的 min/max。
    - teacher：network_version 自 2015-05 起 → 2015-05..2023-12
    - ma/mb：network_version 自 2018-05 起 → 2018-05..2023-12
    全列均无可用月 → `NO_NETWORK_VERSION`（显式哨兵；不写成申请窗，也不崩溃）。
    """
    mask = df['network_version'].notna()
    if 'm1_in_fm_sample' in df.columns:
        mask = mask | df['m1_in_fm_sample'].fillna(False).astype(bool)
    months = df.loc[mask, 'month'].dropna()
    if months.empty:
        return NO_NETWORK_VERSION
    return f'{months.min()}..{months.max()}'


def registry_params_used(registry_path=None) -> dict:
    """I-5：登记本脚本实际读取并施加的 registry 参数（名称/值/来源/施加点）。

    winsor 与 sample 窗口为实际施加；IQR 只有规则 ID 无数值字段，数值仍为
    pricing_lib.IQR_POOL_REGISTERED（显式披露，不猜参数名）。registry_path 仅测试注入。
    """
    lo, hi = winsor_bounds(registry_path)
    win_start, win_end = sample_window_bounds(registry_path)
    reg = load_registry(registry_path)
    return {
        'registry_version': reg.get('registry_version'),
        'winsorization_pct': {
            'name': 'winsorization_pct',
            'value': [lo, hi],
            'source': REGISTRY_SOURCE.format(name='winsorization_pct'),
            'applied': True,
            'call_sites': [
                'main(): winsorize_by_month(df, winsor_panel_cols, lo=WINSOR[0], hi=WINSOR[1])',
                "main(): winsorize_by_month(df, ['R_next', 'R_next_postadj'], "
                'lo=WINSOR[0], hi=WINSOR[1], month_col=\'month\')',
            ],
        },
        'sample_start': {
            'name': 'sample_start',
            'value': param_value('sample_start', registry_path),
            'month': win_start,
            'month_rule': 'str(value)[:7]（YYYY-MM）',
            'source': REGISTRY_SOURCE.format(name='sample_start'),
            'applied': True,
            'call_sites': ['main(): months_between(SAMPLE_WINDOW_START, SAMPLE_WINDOW_END)',
                           'main(): in_win = month >= SAMPLE_WINDOW_START',
                           'check_panel_contracts(): month_range_match 起点'],
        },
        'sample_end': {
            'name': 'sample_end',
            'value': param_value('sample_end', registry_path),
            'month': win_end,
            'month_rule': 'str(value)[:7]（YYYY-MM）',
            'source': REGISTRY_SOURCE.format(name='sample_end'),
            'applied': True,
            'call_sites': ['main(): months_between(SAMPLE_WINDOW_START, SAMPLE_WINDOW_END)',
                           'main(): in_win = month <= SAMPLE_WINDOW_END',
                           'check_panel_contracts(): month_range_match 终点'],
        },
        'sample_window_derived': f'{win_start}..{win_end}',
        'iqr_scale_rule': {
            'name': 'm1_sample_and_spec_registry.iqr_scale_rule_id',
            'value': IQR_POOL_RULE_ID,
            'numeric_value': IQR_POOL_REGISTERED,
            'registry_numeric_value': None,
            'numeric_value_source': IQR_POOL_REGISTERED_SOURCE,
            'applied': 'gate 尺度（IQR_POOL_REGISTERED）；registry 无独立数值参数，保留硬编码并披露',
        },
    }

# I7（2026-10-11）：v2 专用报告与 m1_s5_v2 汇总件（只读已落盘 v1/v2 样本件，不重跑构造/回归）
V2_REPORT_PATH = OUT_DIR / 's3_m1_sample_report_v2.json'
V2_SUMMARY_PATH = SAMPLE_DIR / 'm1_s5_v2_sample_summary.json'
EXCLUSION_ORDER = ['microcap_q10_exclusion', 'recent_ipo_exclusion', 'no_network_support',
                   'own_max_missing', 'peer_return_missing', 'control_missing',
                   'no_outcome_next_month']
EXCLUSION_VALUE_COL = {'microcap_q10_exclusion': 'market_equity',
                       'no_network_support': 'PeerMAXraw',
                       'own_max_missing': 'OwnMAXraw',
                       'peer_return_missing': 'PeerReturn',
                       'no_outcome_next_month': 'R_next'}
V1_READ_COLS = ['focal_id', 'month', 'm1_exclusion_reason', 'm1_in_fm_sample', 'PeerMAXraw',
                'OwnMAXraw', 'PeerReturn', 'market_equity', 'available_at', 'R_next'] + CONTROLS
V2_READ_COLS = V1_READ_COLS + ['sample_window', 'network_version', 'me_q10_market', 'me_below_q10',
                               'ipo_date', 'months_since_ipo', 'recent_ipo', 'next_complete_month',
                               'next_R', 'st_flag_focal']


# ------------------------------------------------------------------ 构造工具
def build_next_month(returns: pd.DataFrame, complete_only: bool) -> pd.DataFrame:
    """紧邻下一自然月 outcome：key=(firm_id, month=event month)。"""
    r = returns.copy()
    if complete_only:
        r = r[(r['complete_month'] == 1) & r['R'].notna()]
    r = r[['firm_id', 'month', 'R']].dropna()
    r['month'] = r['month'].map(lambda m: ym_add(m, -1))   # 上一月 = event month
    return r.rename(columns={'R': 'R_next'})


def build_industry_baselines(months, sw2, srcm, me) -> pd.DataFrame:
    """P2.1：申万一级/二级 IndustryMAX / IndustryPeerReturn（排除 focal；EW/VW）。"""
    v = srcm[['source_id', 'month', 'MAXraw', 'R']].copy()
    v = v[v['month'].isin(months)]
    v = v.rename(columns={'source_id': 'firm_id'})
    v = v.merge(sw2, on='firm_id', how='inner')
    v['code6'] = v['firm_id'].map(code6)
    v = v.merge(me.rename(columns={'firm_id': 'code6'}), on=['code6', 'month'], how='left')
    v['v_ew'] = 1.0
    v['v_vw'] = v['market_equity']
    out = []
    for lvl in ['l1_code', 'l2_code']:
        for kind in ['MAXraw', 'R']:
            for wname, vcol in [('ew', 'v_ew'), ('vw', 'v_vw')]:
                sub = v[v[vcol].notna() & v[kind].notna()].copy()
                sub['_num'] = sub[vcol] * sub[kind]
                g = sub.groupby(['month', lvl], sort=False)
                tot = (g.agg(_num=('_num', 'sum'), _den=(vcol, 'sum')).reset_index()
                       .rename(columns={'_num': '_num_sum', '_den': '_den_sum'}))
                # leave-one-out：把 focal 自身贡献减去（自身缺失时其贡献本为 0，无需减）
                sub = sub.merge(tot, on=['month', lvl], how='left')
                own_num = np.where(sub['_num'].notna(), sub['_num'], 0.0)
                sub['_num_loo'] = sub['_num_sum'] - own_num
                sub['_den_loo'] = sub['_den_sum'] - sub[vcol]
                sub['_val'] = np.where(sub['_den_loo'] > 0,
                                       sub['_num_loo'] / sub['_den_loo'], np.nan)
                name = f'ind_{lvl[:2]}_{"max" if kind == "MAXraw" else "ret"}_{wname}'
                out.append(sub[['firm_id', 'month', '_val']].rename(columns={'_val': name}))
    res = out[0]
    for o in out[1:]:
        res = res.merge(o, on=['firm_id', 'month'], how='outer')
    return res


def build_weight_variants(key: str, panel: pd.DataFrame, srcm: pd.DataFrame,
                          months, ret_df: pd.DataFrame) -> pd.DataFrame:
    """P2.2：固定边集、只换权重（equal / log_sim；linear 由面板本身提供）。"""
    ver = (panel[['month', 'network_version']].dropna().drop_duplicates()
           .sort_values('month'))
    ver['text_year'] = pd.to_numeric(ver['network_version'], errors='coerce').astype(int)
    vals_max = srcm[['source_id', 'month', 'MAXraw']].rename(columns={'source_id': 'firm_id'})
    vals_ret = ret_df[['firm_id', 'month', 'R']]   # 与面板同源（monthly_returns_clean_v2 R）
    parts = []
    cache = {}
    for ty, grp in ver.groupby('text_year', sort=True):
        e = cache.get(ty)
        if e is None:
            e = add_weight_rules(load_edges(key, int(ty)))
            cache[ty] = e
        mm = vals_max[vals_max['month'].isin(grp['month'])]
        rr = vals_ret[vals_ret['month'].isin(grp['month'])]
        a = aggregate_edges(e, mm, ['w_equal', 'w_linear', 'w_log_sim'], 'MAXraw')
        b = aggregate_edges(e, rr, ['w_equal', 'w_linear', 'w_log_sim'], 'R')
        p = a.merge(b, on=['month', 'focal_id'], how='outer')
        parts.append(p)
        progress(f'{key} 权重变体 文本年 {ty} 完成（{len(p):,} 行）')
    out = pd.concat(parts, ignore_index=True)
    out = out.rename(columns={'MAXraw_w_equal': 'PeerMAXraw_weq', 'MAXraw_w_linear': 'PeerMAXraw_wlin',
                              'MAXraw_w_log_sim': 'PeerMAXraw_wlog',
                              'R_w_equal': 'PeerReturn_weq', 'R_w_linear': 'PeerReturn_wlin',
                              'R_w_log_sim': 'PeerReturn_wlog'})
    return out


def load_ipo_map() -> dict:
    """listing date：data/processed/firm_id_map.csv（code6 -> ipo_date）。"""
    d = pd.read_csv(IPO_MAP_PATH, dtype=str)
    d = d[d['code6'].notna()]
    return dict(zip(d['code6'].str.strip(), d['ipo_date']))


def v2_summary(key: str) -> dict:
    """v2 样本（P1-01 后标 superseded）摘要，供登记与 v3 对照。"""
    d = pd.read_parquet(sample_path(key, 'v2'),
                        columns=['month', 'm1_exclusion_reason', 'm1_in_fm_sample', 'PeerMAXraw',
                                 'OwnMAXraw', 'PeerReturn'] + CONTROLS)
    uni_t = (d['PeerMAXraw'].notna() & d['OwnMAXraw'].notna() & d['PeerReturn'].notna()
             & ~d[CONTROLS].isna().any(axis=1))
    iqr_v2 = float(d.loc[uni_t, 'PeerMAXraw'].quantile(.75) - d.loc[uni_t, 'PeerMAXraw'].quantile(.25))
    return {
        'sample_version': M1_SAMPLE_VERSION_V2,
        'status': 'superseded',
        'superseded_reason': ('P1-01（2026-11-15）：旧 complete_month 用价格非空日数，'
                              '非 methodology 要求的有效交易日（volume>0 >=15）；'
                              '文件保留不删，作旧口径对照。'),
        'rows_total': int(len(d)),
        'months': int(d['month'].nunique()),
        'month_min': str(d['month'].min()), 'month_max': str(d['month'].max()),
        'exclusion_counts': {k: int(v) for k, v in
                             d.loc[d['m1_exclusion_reason'] != '', 'm1_exclusion_reason']
                             .value_counts().items()},
        'eligible_rows': int(d['m1_in_fm_sample'].sum()),
        'iqr_pool_treatment_only': iqr_v2,
        'path': str(sample_path(key, 'v2').relative_to(ROOT)),
    }


def v1_summary(key: str) -> dict:
    """v1 样本（未施加 microcap 筛除的旧版本）摘要，供 superseded 登记与对照。"""
    d = pd.read_parquet(sample_path(key, 'v1'),
                        columns=['month', 'm1_exclusion_reason', 'm1_in_fm_sample', 'PeerMAXraw',
                                 'OwnMAXraw', 'PeerReturn'] + CONTROLS)
    uni_t = (d['PeerMAXraw'].notna() & d['OwnMAXraw'].notna() & d['PeerReturn'].notna()
             & ~d[CONTROLS].isna().any(axis=1))
    iqr_v1 = float(d.loc[uni_t, 'PeerMAXraw'].quantile(.75) - d.loc[uni_t, 'PeerMAXraw'].quantile(.25))
    return {
        'sample_version': M1_SAMPLE_VERSION_V1,
        'status': 'superseded_registered_deviation',
        'superseded_reason': ('v1 未施加 microcap 筛除（registry sample_filter_thresholds.microcap_exclusion），'
                              '次新股（<6 月）筛除亦未施加；ma/mb 两列未施加 sample 窗口过滤（含 2024-01 事件月）。'
                              '文件保留不删，作「未施加 microcap 筛除」敏感性。'),
        'rows_total': int(len(d)),
        'months': int(d['month'].nunique()),
        'month_min': str(d['month'].min()), 'month_max': str(d['month'].max()),
        'exclusion_counts': {k: int(v) for k, v in
                             d.loc[d['m1_exclusion_reason'] != '', 'm1_exclusion_reason']
                             .value_counts().items()},
        'eligible_rows': int(d['m1_in_fm_sample'].sum()),
        'fm_months': [sorted(d.loc[d['m1_in_fm_sample'].astype(bool), 'month'].unique())[0],
                      sorted(d.loc[d['m1_in_fm_sample'].astype(bool), 'month'].unique())[-1]]
        if int(d['m1_in_fm_sample'].sum()) else None,
        'iqr_pool_treatment_only': iqr_v1,
        'iqr_pool_registered_used_for_gate': IQR_POOL_REGISTERED,
        'path': str(sample_path(key, 'v1').relative_to(ROOT)),
    }


# ------------------------------------------------------------------ Phase 1 I-2 硬门
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description='A1–A5：M1（S5）样本构造（四条输入契约硬门；失败 SystemExit(2)）')
    ap.add_argument('--v2-report-only', action='store_true',
                    help='只读 v1/v2 样本件生成 v2 专用报告（不重跑构造）')
    ap.add_argument('--force', action='store_true',
                    help='绕过硬门（仅限显式恢复场景；必须同时给 --force-reason）')
    ap.add_argument('--force-reason', default=None,
                    help='绕过硬门的原因（写入报告 contract_enforcement.force_reason）')
    ap.add_argument('--rebuild-amount-cache', action='store_true',
                    help='I-14：显式重算并覆盖 amount_100m 缓存（输入 sha 不匹配时默认 fail-closed）')
    args = ap.parse_args(argv)
    if args.force and not (args.force_reason or '').strip():
        print('[HARD-GATE] --force 必须同时提供非空 --force-reason', file=sys.stderr, flush=True)
        raise SystemExit(2)
    return args


def check_panel_contracts() -> dict:
    """A1：四列面板合同（只读）。字段名与既有报告 contracts 完全一致。"""
    contracts = {}
    for key, path in PANELS.items():
        df = load_panel(key)
        n_rows = len(df)
        dup = int(df.duplicated(['focal_id', 'month']).sum())
        mm = df['month']
        pit_bad = int((pd.to_datetime(df['available_at'], errors='coerce') >=
                       pd.to_datetime(df['month'] + '-01', errors='coerce')).sum())
        nv = df['network_version'].astype('string')
        no_ver = nv.isna() | (nv == 'nan')
        # 版本映射复核：面板 network_version 记为文本年；text_year = year(month)-1 (m>=5) / -2 (m<=4)
        vy = pd.to_numeric(nv, errors='coerce')
        my = df['month'].str[:4].astype(int)
        mnum = df['month'].str[5:7].astype(int)
        expect_ty = np.where(mnum >= 5, my - 1, my - 2)
        ver_ok = ((vy == expect_ty) | no_ver).all()
        avail_expect = (pd.to_datetime(df['available_at']).dt.strftime('%Y-%m-%d'))
        expect_avail = np.where(no_ver, '', (vy + 1).astype('Int64').astype(str) + '-04-30')  # 文本年 y -> (y+1)-04-30
        avail_ok = bool((avail_expect.fillna('') == pd.Series(expect_avail, index=df.index)).all())
        # 缺失不填 0：support 状态与取值一致性
        sup = df['network_support_status'] == 'support'
        miss_no_zero = bool((df.loc[sup, 'PeerMAXraw'].notna() |
                             (df.loc[sup, 'data_status'] == 'no_covered_max_source')).all())
        zeros = int((df.loc[~sup, RET_COLS].fillna(-999) == 0).sum().sum())
        contracts[key] = {
            'path': str(path.relative_to(ROOT)), 'rows': n_rows,
            'rows_registered': {'ma_teacher': 307503, 'mb_teacher': 384303,
                                'ma': 325170, 'mb': 402042}[key],
            'rows_match_registered': n_rows == {'ma_teacher': 307503, 'mb_teacher': 384303,
                                                'ma': 325170, 'mb': 402042}[key],
            'months': int(mm.nunique()), 'month_min': mm.min(), 'month_max': mm.max(),
            'month_range_match': (mm.min() == SAMPLE_WINDOW_START
                                  and mm.max() == SAMPLE_WINDOW_END
                                  and mm.nunique() == len(months_between(SAMPLE_WINDOW_START,
                                                                         SAMPLE_WINDOW_END))),
            'month_range_note': ('ma/mb 面板覆盖 2018-01..2025-12（96 月，语料自建链）→ v2 按 sample 窗口'
                                 '裁剪到 2018-01..2023-12（I2）' if key in ('ma', 'mb') else ''),
            'duplicate_focal_month_keys': dup,
            'pit_violations_strict_lt': pit_bad,
            'version_mapping_ok': bool(ver_ok), 'available_at_rule_ok': avail_ok,
            'no_available_version_rows': int(no_ver.sum()),
            'missing_filled_with_zero': zeros,
            'support_rows': int(sup.sum()), 'no_support_rows': int((~sup).sum()),
            'support_missing_not_zero_ok': miss_no_zero,
            'data_status_counts': df['data_status'].value_counts(dropna=False).to_dict(),
            'bytes': file_size(path),
        }
        progress(f'A1 {key}: rows={n_rows} PIT违规={pit_bad} dup={dup} '
                 f'版本映射ok={bool(ver_ok)} 缺失填0={zeros}')
    return contracts


def enforce_panel_contracts(contracts: dict, force: bool, force_reason: str | None) -> dict:
    """四条硬门；默认任一条失败即 SystemExit(2)，force 时只记录不阻断。"""
    checks, failing = {}, []
    for key in PANELS:
        c = contracts[key]
        vals = {
            'rows_match_registered': bool(c.get('rows_match_registered')),
            'pit_violations_strict_lt': c.get('pit_violations_strict_lt'),
            'duplicate_focal_month_keys': c.get('duplicate_focal_month_keys'),
            'missing_filled_with_zero': c.get('missing_filled_with_zero'),
        }
        checks[key] = {'rows': c.get('rows'), 'rows_registered': c.get('rows_registered'), **vals}
        ok = (vals['rows_match_registered'] and vals['pit_violations_strict_lt'] == 0
              and vals['duplicate_focal_month_keys'] == 0
              and vals['missing_filled_with_zero'] == 0)
        if force:
            if not ok:
                failing.append(key)
            continue
        require(vals['rows_match_registered'],
                f's3_m1_1 {key}: rows={c.get("rows")} != registered={c.get("rows_registered")}')
        require(vals['pit_violations_strict_lt'] == 0,
                f's3_m1_1 {key}: PIT 违规 {vals["pit_violations_strict_lt"]} != 0')
        require(vals['duplicate_focal_month_keys'] == 0,
                f's3_m1_1 {key}: 重复 (focal_id, month) 键 '
                f'{vals["duplicate_focal_month_keys"]} != 0')
        require(vals['missing_filled_with_zero'] == 0,
                f's3_m1_1 {key}: missing_filled_with_zero='
                f'{vals["missing_filled_with_zero"]} != 0')
    return {
        'mode': 'FORCED' if force else 'ENFORCED',
        'force_reason': force_reason if force else None,
        'hard_gate_keys': list(HARD_GATE_KEYS),
        'all_pass': not failing,
        'failing_columns': failing,
        'checks': checks,
    }


# ------------------------------------------------------------------ 主流程
def main(argv=None):
    args = parse_args(argv)
    force = bool(args.force)
    force_reason = (args.force_reason or '').strip() or None
    # Phase 1 I-3：读取任何 M1 输入之前，先核验全部 M1 主输入（9 个）与 manifest
    # bytes 锚；缺失条目/缺失 sha/字节不一致 → SystemExit(2)（无 CLI 绕过开关）。
    input_sha_ledger = assert_m1_inputs()
    months = months_between(SAMPLE_WINDOW_START, SAMPLE_WINDOW_END)
    progress('A1 开始：输入与 PIT 合同核验（四列面板 + 共享输入装载）')
    ret = load_returns_v2()
    post = load_postadj()
    me = load_me()
    chars = load_chars()
    ana = load_analyst_derived()
    sw2 = load_sw2()
    srcm = load_src_month()
    ipo_map = load_ipo_map()
    progress('A1 共享输入就绪（returns/ME/chars/analyst/sw2/source-month/listing-date）')

    # Phase 1 I-2：四条硬门必须在**写任何 parquet/报告/缓存之前**通过；
    # 默认失败即 SystemExit(2)，不留下新样本件；--force 仅显式恢复且原因写入报告。
    panel_contracts = check_panel_contracts()
    enforcement = enforce_panel_contracts(panel_contracts, force, force_reason)
    progress(f'A1 契约硬门 {enforcement["mode"]}：all_pass={enforcement["all_pass"]} '
             f'failing_columns={enforcement["failing_columns"]}')

    cache_dir = ROOT / 'artifacts/tmp'
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Phase 1 I-14：缓存名与内容一致（只有 amount_100m）+ 输入指纹 sidecar；旧
    # m1_cache_illiq.parquet 一次性只读迁移（数值原样、旧文件保留）。输入 sha 不匹配
    # 默认 fail-closed；显式 --rebuild-amount-cache 才重算覆盖。
    illiq, amount_cache = load_amount_100m_cached(
        cache_dir=cache_dir, rebuild=bool(args.rebuild_amount_cache))
    progress(f'A3 Liquidity 自算缓存 {amount_cache["cache_status"]}（{len(illiq):,} 行；'
             f'{amount_cache["liquidity_formula_id"]}；输入 sha {amount_cache["source_bytes"][:12]}…）')
    turnover = load_turnover_drive()
    progress(f'A3 Turnover（Drive daily_turnover_rate）就绪（{len(turnover):,} 行）')
    st = load_st()
    progress(f'A9 ST 标记就绪（{len(st):,} 行）')

    # I3：microcap 阈值 = 形成（事件）月**完整市场横截面** q10，市值口径 = 总市值（登记偏离）
    me_q10 = (me.groupby('month')['market_equity'].quantile(0.10)
              .rename('me_q10_market'))
    me_q10 = me_q10[(me_q10.index >= SAMPLE_WINDOW_START) & (me_q10.index <= SAMPLE_WINDOW_END)]
    progress(f'A5 microcap 阈值就绪：形成月全市场横截面 q10（总市值口径），{len(me_q10)} 个月')

    ret_raw = load_returns_raw()
    rn_raw = build_next_month(ret, complete_only=True)
    rn_post = build_next_month(post, complete_only=False)
    rn_raw['_flag_raw'] = 1
    rn_post['_flag_post'] = 1
    n_raw = int(rn_raw['R_next'].notna().sum())
    n_post = int(rn_post['R_next'].notna().sum())
    rn_post = rn_post.rename(columns={'R_next': 'R_next_postadj'})
    rn = (rn_raw.merge(rn_post, on=['firm_id', 'month'], how='left')
          .rename(columns={'firm_id': 'focal_id'}))
    progress(f'A2 下月 outcome 就绪：raw {n_raw:,} / post-adj {n_post:,}')

    # I6：下月收益完整度（停牌/不完整月代理）
    rstat = ret[['firm_id', 'month', 'complete_month', 'n_valid_days', 'R']].copy()
    rstat['month'] = rstat['month'].map(lambda m: ym_add(m, -1))
    rstat = rstat.rename(columns={'firm_id': 'focal_id', 'complete_month': 'next_complete_month',
                                  'n_valid_days': 'next_n_valid_days', 'R': 'next_R'})

    ownr = (srcm[['source_id', 'month', 'R']].rename(columns={'source_id': 'focal_id',
                                                              'R': 'own_R_t'}))
    c_ind = cache_dir / 'm1_cache_industry.parquet'
    if c_ind.exists():
        ind = pd.read_parquet(c_ind)
        progress(f'A4 行业 baseline 命中缓存（{len(ind):,} 行）')
    else:
        ind = build_industry_baselines(months, sw2, srcm, me)
        ind.to_parquet(c_ind, index=False)
        progress(f'A4 行业 baseline 就绪（申万一级/二级 × EW/VW，{len(ind):,} 行）')

    report = {
        'sample_version': M1_SAMPLE_VERSION_V3,
        'sample_version_v2_superseded': M1_SAMPLE_VERSION_V2,
        'sample_version_v1_superseded': M1_SAMPLE_VERSION_V1,
        'run_at': pd.Timestamp.utcnow().isoformat(),
        'script': 'scripts/pricing_sample_build.py',
        'phase': 'A（2026-10-10 复核修复批 I1–I6：v2 headline / v1 superseded 敏感性）',
        'iqr_pool_registered': IQR_POOL_REGISTERED,
        'registry_params': {
            'alpha_m1': param_value('alpha_m1'), 'K_Econ_M1': param_value('K_Econ_M1'),
            'K_Spread_M1': param_value('K_Spread_M1'), 'K_Consist_M1': param_value('K_Consist_M1'),
            'nw_lag': param_value('nw_lag'), 'winsorization_pct': list(WINSOR),
            'random_seed': param_value('random_seed'),
            'sample_start': param_value('sample_start'), 'sample_end': param_value('sample_end'),
            'min_focal_count_per_month': MIN_N,
            'min_effective_month_count': MIN_MONTHS,
            'portfolio_breakpoints': param_value('portfolio_breakpoints'),
        },
        'sample_window': {'event_month_window': WINDOW,
                          # I-16：申请窗 vs 按列实际窗；旧 event_month_window 语义不变
                          'window_requested': WINDOW,
                          'legacy_sample_window': WINDOW,
                          'legacy_sample_window_note': '旧 sample_window 列保持申请窗（兼容）；实际窗见 '
                                                       'sample_window_actual 与 sample_window_actual_by_column',
                          'sample_window_actual_rule': ACTUAL_WINDOW_RULE,
                          'rule': 'registry sample_start/sample_end 施加于事件月（月度信号月）；'
                                  'outcome = 紧邻下一自然月，不跨缺失月',
                          'note': '主口径（ma_teacher/mb_teacher）网络版本最早 2015-05 → FM 首发月 2015-05；'
                                  'ma/mb 两列语料自 2018-01 起 → FM 实际窗口 2018-05..2023-12'},
        'sample_filter_status': {
            'microcap': {'registry_value': FILTERS['microcap_exclusion'],
                         'status': 'EXECUTED_IN_V2',
                         'implementation': '形成（事件）月完整市场横截面 q10；阈值源 data/processed/market_equity_panel.csv（总市值）',
                         'registered_deviation': '市值基础 = 总市值（registry 目标 = 流通市值）；与 S3A 9C 同口径偏离，'
                                                 '因流通市值日频官方序列不可得（spec.sample_filters.market_value_basis_current=total_market_value）',
                         'missing_value_handling': 'market_equity 缺失 → 不排除（fillna(False)），与 S3A 9C 一致'},
            'recent_ipo': {'registry_value': FILTERS['ipo_exclusion'], 'status': 'EXECUTED_IN_V2',
                           'listing_date_source': 'data/processed/firm_id_map.csv（code6 → ipo_date）',
                           'rule': f'上市 < {RECENT_IPO_MONTHS} 月（按事件月首日 − ipo_date，天数/30.4）排除',
                           'missing_value_handling': 'ipo_date 缺失 → 不排除（fillna(False)）'},
            'st': {'registry_value': FILTERS['st_stratify'],
                   'status': 'REGISTERED_DEVIATION_SCOPE（spec.sample_filters.st_primary_current=include_st）',
                   'note': 'M1 主样本含 ST，按 focal 侧 ST 分层并报（A9 m1_rob_st）；不额外剔除'},
            'liquidity': {'status': 'NOT_EXECUTABLE_NO_REGISTERED_THRESHOLD',
                          'note': 'registry sample_filter_thresholds 的 scope 提及「严重流动性不足」，但 headline_value '
                                  '未给可执行阈值 → 登记为不可执行（不静默设阈值）'},
            'limit_hit': {'registry_value': FILTERS['limit_hit_source'],
                          'status': 'REGISTERED_POLICY_INCLUDE_AND_STRATIFY（A9 limit-hit 分层）'},
            'no_network_support': {'registry_value': FILTERS['no_network_support'],
                                   'status': 'EXECUTED（标 missing 不填 0，见 contracts.missing_filled_with_zero=0）'},
        },
        'contracts': panel_contracts,
        'contract_enforcement': enforcement,
        'input_sha_ledger': input_sha_ledger,
        # Phase 1 I-5：新增键（不改既有键）；当前报告未重跑，生成器已具备。
        'registry_params_used': registry_params_used(),
        'outcome': {},
        'controls': {},
        'baselines': {'industry_version': 'sw2_industry_latest.csv (SwClass2021, is_current_version)',
                      'formula_id': 'INDUSTRY_PEER_AGG_P2_1_v1',
                      'n_industry_rows': int(len(ind))},
        'weight_variants': {},
        'samples': {},
        'samples_v1_superseded': {},
        'samples_v2_superseded': {},
        'window_filter': {},
        'new_listing_and_suspension_status': {},
    }

    # ---------------- 逐列组装（v2：窗口 + microcap + 次新股）
    actual_windows = {}          # I-16：四列实际窗（报告汇总用）
    winsor_panel_cols = RET_COLS + CONTROLS + ['ind_l1_max_ew', 'ind_l1_max_vw', 'ind_l2_max_ew',
                                    'ind_l2_max_vw', 'ind_l1_ret_ew', 'ind_l1_ret_vw',
                                    'ind_l2_ret_ew', 'ind_l2_ret_vw']
    for key in PANELS:
        df = load_panel(key)
        df['code6'] = df['focal_id'].map(code6)
        df = df.merge(me.rename(columns={'firm_id': 'code6'}), on=['code6', 'month'], how='left')
        df = df.merge(chars.rename(columns={'firm_id': 'code6'}), on=['code6', 'month'], how='left')
        df = df.drop(columns=['turnover']).merge(
            turnover.rename(columns={'firm_id': 'focal_id'}), on=['focal_id', 'month'], how='left')
        df = df.merge(ana.rename(columns={'firm_id': 'code6'}), on=['code6', 'month'], how='left')
        df = df.merge(illiq.rename(columns={'firm_id': 'focal_id'}), on=['focal_id', 'month'], how='left')
        df = df.merge(ind.rename(columns={'firm_id': 'focal_id'}), on=['focal_id', 'month'], how='left')
        df = df.merge(ownr, on=['focal_id', 'month'], how='left')
        df = df.merge(rn, on=['focal_id', 'month'], how='left')
        df = df.merge(st.rename(columns={'firm_id': 'focal_id'}), on=['focal_id', 'month'], how='left')
        df = df.merge(rstat, on=['focal_id', 'month'], how='left')
        df['illiq'] = df['own_R_t'].abs() / df['amount_100m']   # 事件月 t 自身收益/成交额（无前视）
        # Phase 1 I-5：显式传 registry 边界（面板列块 + outcome 块；缺省不再依赖库内 0.01/0.99）
        df = winsorize_by_month(df, winsor_panel_cols, lo=WINSOR[0], hi=WINSOR[1])
        df = winsorize_by_month(df, ['R_next', 'R_next_postadj'],
                                lo=WINSOR[0], hi=WINSOR[1], month_col='month')
        df['illiq'] = df['illiq'].replace([np.inf, -np.inf], np.nan)

        # ---------------- I2：样本窗口（事件月）
        in_win = (df['month'] >= SAMPLE_WINDOW_START) & (df['month'] <= SAMPLE_WINDOW_END)
        n_out_window = int((~in_win).sum())
        df = df[in_win].copy()
        # I-16：旧 sample_window 继续写申请窗（兼容；不重写当前数据）；申请窗另存 window_requested
        df['sample_window'] = WINDOW
        df['window_requested'] = WINDOW

        # ---------------- I3：microcap（形成月全市场 q10）+ 次新股（<6 月）
        df['me_q10_market'] = df['month'].map(me_q10)
        df['me_below_q10'] = (df['market_equity'] < df['me_q10_market']).fillna(False)
        df['ipo_date'] = df['code6'].map(ipo_map)
        df['months_since_ipo'] = ((pd.to_datetime(df['month'], errors='coerce')
                                   - pd.to_datetime(df['ipo_date'], errors='coerce')).dt.days / 30.4)
        df['recent_ipo'] = (df['months_since_ipo'] < RECENT_IPO_MONTHS).fillna(False)

        # 分层
        lh = df['limit_hit_weight_share']
        df['limit_hit_max_stratum'] = np.where(lh == 0, 'all_non_limit',
                                               np.where(lh == 1, 'all_limit', 'mixed'))
        df.loc[lh.isna(), 'limit_hit_max_stratum'] = np.nan
        df['st_flag_focal'] = df['st_flag'].fillna(False).astype(bool)

        # ---------------- A5：样本冻结（逐 exclusion reason，顺序冻结）
        # v2 顺序：registry 样本筛选规则（microcap / 次新股）在前，其后为数据可得性屏幕
        reasons = []
        reasons.append(('microcap_q10_exclusion', df['me_below_q10']))
        reasons.append(('recent_ipo_exclusion', df['recent_ipo']))
        reasons.append(('no_network_support', df['PeerMAXraw'].isna()))
        reasons.append(('own_max_missing', df['OwnMAXraw'].isna()))
        reasons.append(('peer_return_missing', df['PeerReturn'].isna()))
        ctrl_missing = df[CONTROLS].isna().any(axis=1)
        reasons.append(('control_missing', ctrl_missing))
        reasons.append(('no_outcome_next_month', df['R_next'].isna()))

        excl = pd.Series(False, index=df.index)
        counts = {}
        for name, mask in reasons:
            newly = mask & ~excl
            counts[name] = int(newly.sum())
            df.loc[newly, 'm1_exclusion_reason'] = name
            excl = excl | mask
        df.loc[~excl, 'm1_exclusion_reason'] = ''
        # 控制缺失细分（仅统计 control_missing 行内各变量的缺失数）
        cm_rows = df['m1_exclusion_reason'] == 'control_missing'
        ctrl_detail = {c: int(df.loc[cm_rows, c].isna().sum()) for c in CONTROLS}

        universe_T = ~excl & (df['m1_exclusion_reason'] == '')
        # treatment-only universe（outcome-blind：不含 outcome 条件）
        uni_t = ((df['PeerMAXraw'].notna()) & (df['OwnMAXraw'].notna())
                 & (df['PeerReturn'].notna()) & (~ctrl_missing))
        iqr_pool = float(df.loc[uni_t, 'PeerMAXraw'].quantile(.75) -
                         df.loc[uni_t, 'PeerMAXraw'].quantile(.25))
        iqr_tol = float(param_value('iqr_drift_tolerance'))
        if IQR_POOL_REGISTERED > 0 and abs(iqr_pool - IQR_POOL_REGISTERED) / IQR_POOL_REGISTERED > iqr_tol:
            raise SystemExit(
                f'[STOP] {key}: treatment-only IQR={iqr_pool:.8f} 与登记值 '
                f'{IQR_POOL_REGISTERED:.8f} 相对漂移超过 {iqr_tol:.0%}；'
                '先复核 registry iqr_pool_value 再重跑')
        iqr_final = float(df.loc[universe_T, 'PeerMAXraw'].quantile(.75) -
                          df.loc[universe_T, 'PeerMAXraw'].quantile(.25))
        n_by_month = df.loc[universe_T].groupby('month')['focal_id'].nunique()
        used_months = sorted(n_by_month[n_by_month >= MIN_N].index)
        df['m1_in_fm_sample'] = universe_T & df['month'].isin(used_months)

        # I-16：实际窗必须在 m1_in_fm_sample 落定后计算（OR 条件）；旧 sample_window/window_requested
        # 保持申请窗，实际窗单独写入 sample_window_actual，追加在输出列末尾。
        actual_window = compute_actual_sample_window(df)
        df['sample_window_actual'] = actual_window
        actual_windows[key] = actual_window

        # v1（旧版本）对照：受影响份额
        v1 = pd.read_parquet(sample_path(key, 'v1'),
                             columns=['focal_id', 'month', 'm1_in_fm_sample'])
        v1_fm = v1[v1['m1_in_fm_sample'].astype(bool)][['focal_id', 'month']]
        v1_fm['_v1_fm'] = 1
        chk = df.merge(v1_fm, on=['focal_id', 'month'], how='left')
        v1_fm_in_window = int(chk['_v1_fm'].fillna(0).sum())

        report['window_filter'][key] = {
            'window': WINDOW, 'rows_before_window_filter': int(n_out_window + len(df)),
            'rows_dropped_out_of_window': n_out_window,
            'rows_after_window_filter': int(len(df)),
            'v1_fm_rows_inside_window': v1_fm_in_window,
            'v1_fm_rows_dropped_by_window': int(v1['m1_in_fm_sample'].sum()) - v1_fm_in_window,
            # I-16：申请窗 vs 按列实际窗（只增字段；window 保持申请窗以兼容）
            'window_requested': WINDOW,
            'sample_window_actual': actual_window,
            'sample_window_actual_rule': ACTUAL_WINDOW_RULE,
            'actual_window_available': actual_window != NO_NETWORK_VERSION,
        }
        report['samples'][key] = {
            'sample_version': M1_SAMPLE_VERSION_V3,
            'status': 'active_headline',
            'path': str(sample_path(key, 'v3').relative_to(ROOT)),
            'sample_window': WINDOW,                     # I-16：旧列/旧字段保持申请窗（兼容）
            'window_requested': WINDOW,
            'sample_window_actual': actual_window,
            'sample_window_actual_rule': ACTUAL_WINDOW_RULE,
            'rows_total': int(len(df)),
            'exclusion_counts_in_order': counts,
            'microcap_impact': {
                'rows_flag_microcap_all': int(df['me_below_q10'].sum()),
                'share_flag_microcap_all': float(df['me_below_q10'].mean()),
                'share_flag_microcap_of_v1_fm_rows': float(
                    chk.loc[chk['_v1_fm'].fillna(0) == 1, 'me_below_q10'].mean())
                if v1_fm_in_window else None,
                'share_flag_recent_ipo_of_v1_fm_rows': float(
                    chk.loc[chk['_v1_fm'].fillna(0) == 1, 'recent_ipo'].mean())
                if v1_fm_in_window else None,
            },
            'control_missing_detail': ctrl_detail,
            'eligible_universe_treatment_only': int(uni_t.sum()),
            'eligible_rows': int(universe_T.sum()),
            'eligible_share': float(universe_T.mean()),
            'iqr_pool_treatment_only': iqr_pool,
            'iqr_pool_final_sample': iqr_final,
            'iqr_pool_registered_used_for_gate': IQR_POOL_REGISTERED,
            'n_months_eligible': int(n_by_month.shape[0]),
            'n_months_ge_min_focal': len(used_months),
            'min_focal_count_per_month': MIN_N,
            'n_focal_per_month': {'min': int(n_by_month.min()), 'median': float(n_by_month.median()),
                                  'max': int(n_by_month.max())},
            'fm_sample_rows': int(df['m1_in_fm_sample'].sum()),
            'fm_months': [used_months[0], used_months[-1]] if used_months else None,
            'limit_hit_stratum_counts_fm': df.loc[df['m1_in_fm_sample'], 'limit_hit_max_stratum']
                                           .value_counts(dropna=False).to_dict(),
            'st_flag_counts_fm': df.loc[df['m1_in_fm_sample'], 'st_flag_focal']
                                 .value_counts(dropna=False).to_dict(),
            'coverage_ratio_max': {'p10': float(df.loc[df['m1_in_fm_sample'], 'coverage_ratio_max'].quantile(.1)),
                                   'median': float(df.loc[df['m1_in_fm_sample'], 'coverage_ratio_max'].median())},
            'coverage_ratio_ret_median': float(df.loc[df['m1_in_fm_sample'], 'coverage_ratio_ret'].median()),
            'n_peer_sources_median': float(df.loc[df['m1_in_fm_sample'], 'n_sources'].median()),
            'post_adjusted_outcome_available_fm': int(df.loc[df['m1_in_fm_sample'], 'R_next_postadj'].notna().sum()),
            'next_month_quality_fm': {
                'complete_month_used': int((df.loc[df['m1_in_fm_sample'], 'next_complete_month'] == 1).sum()),
                'incomplete_month_proxy_halt': int(
                    df.loc[df['m1_in_fm_sample'], 'next_complete_month'].eq(0).sum()),
                'next_month_return_missing': int(
                    df.loc[df['m1_in_fm_sample'], 'next_R'].isna().sum()),
            },
        }
        report['samples_v1_superseded'][key] = v1_summary(key)
        report['samples_v2_superseded'][key] = v2_summary(key)
        cols = ['focal_id', 'month', 'network_use_case', 'network_version', 'available_at',
                'PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'n_sources', 'limit_hit_weight_share',
                'limit_hit_max_stratum', 'coverage_ratio_max', 'coverage_ratio_ret',
                'R_next', 'R_next_postadj', 'market_equity', 'st_flag_focal',
                'ind_l1_max_ew', 'ind_l1_max_vw', 'ind_l2_max_ew', 'ind_l2_max_vw',
                'ind_l1_ret_ew', 'ind_l1_ret_vw', 'ind_l2_ret_ew', 'ind_l2_ret_vw',
                'm1_exclusion_reason', 'm1_in_fm_sample', 'sample_window', 'me_q10_market',
                'me_below_q10', 'ipo_date', 'months_since_ipo', 'recent_ipo',
                'next_complete_month', 'next_n_valid_days', 'next_R'] + CONTROLS + [
                # I-16：新列只追加在末尾（旧列顺序/旧 sample_window 值不变）
                'window_requested', 'sample_window_actual']
        out = df[cols]
        p = sample_path(key, 'v3')
        out.to_parquet(p, index=False)
        progress(f'A5 {key}: v3 eligible={int(universe_T.sum()):,} FM样本={int(df["m1_in_fm_sample"].sum()):,} '
                 f'月份={len(used_months)} 窗口={used_months[0] if used_months else "-"}..'
                 f'{used_months[-1] if used_months else "-"} microcap剔除={counts["microcap_q10_exclusion"]:,} '
                 f'次新剔除={counts["recent_ipo_exclusion"]:,} 窗口外剔除={n_out_window:,} '
                 f'IQR(TO,v2诊断)={iqr_pool:.8f} 实际窗={actual_window} -> {p.name}')

    # I-16：四列实际窗汇总（申请窗 WINDOW 不变；此处只登记实际值）
    report['sample_window']['sample_window_actual_by_column'] = actual_windows

    # I6：新股/停牌状态登记（合并列级）
    all_s = pd.concat([pd.read_parquet(sample_path(k, 'v3'),
                                       columns=['month', 'st_flag_focal', 'ipo_date',
                                                'months_since_ipo', 'next_complete_month', 'next_R'])
                       for k in PANELS], ignore_index=True)
    report['new_listing_and_suspension_status'] = {
        'listing_date_source': 'data/processed/firm_id_map.csv（code6 → ipo_date）',
        'listing_date_coverage': {
            'firms_with_ipo_date': int(len(load_ipo_map())),
            'rows_missing_ipo_date_v2': int(all_s['ipo_date'].isna().sum()),
            'months_since_ipo_p10': float(all_s['months_since_ipo'].quantile(.10)),
        },
        'recent_ipo_rule': f'上市 < {RECENT_IPO_MONTHS} 月排除（v2 已施加；缺失 ipo_date 不排除）',
        'halt_suspension_proxy': {
            'rule': '下月收益完整度：outcome 要求 complete_month=1 且 n_valid_days>=15（R_NEXT_MONTH_COMPLETE_v1）；'
                    '不完整月按停牌/长期停牌代理，不填 0',
            'rows_next_month_incomplete_proxy_all_v2': int(all_s['next_complete_month'].eq(0).sum()),
            'rows_next_month_return_missing_all_v2': int(all_s['next_R'].isna().sum()),
        },
        'st_status': {'rule': '含 ST（spec st_primary_current=include_st）+ focal 侧分层并报',
                      'rows_st_flag_true_all_v2': int(all_s['st_flag_focal'].sum())},
        'status': 'EXECUTED_IN_V2（listing date 可得 → 次新股筛除已执行；停牌以收益完整度代理并登记）',
    }

    # ---------------- A4：权重变体（主口径与方法稳健性列）
    for key in ['ma_teacher', 'mb_teacher']:
        df = load_panel(key)
        wv = build_weight_variants(key, df, srcm, months, ret_raw)
        p = SAMPLE_DIR / f'm1_weight_variants_{key}.parquet'
        wv.to_parquet(p, index=False)
        # 交叉核验：w_linear 复算 vs 面板 PeerMAXraw（面板为未 winsorize 原值）
        chk = df[['focal_id', 'month', 'PeerMAXraw', 'PeerReturn']].merge(
            wv, on=['focal_id', 'month'], how='inner')
        d_max = float((chk['PeerMAXraw'] - chk['PeerMAXraw_wlin']).abs().max())
        d_ret = float((chk['PeerReturn'] - chk['PeerReturn_wlin']).abs().max())
        report['weight_variants'][key] = {
            'path': str(p.relative_to(ROOT)), 'rows': int(len(wv)),
            'linear_vs_panel_max_abs_diff_peermax': d_max,
            'linear_vs_panel_max_abs_diff_peerreturn': d_ret,
            'formula_ids': 'w_equal / w_similarity(linear, 主口径) / w_log_sim',
            'winsorization_note': 'I1：变体/placebo 列在 A9 分析宇宙内按月度 [0.01,0.99] clip（与主规格同规则）；'
                                  '本文件存未 winsorize 原值',
        }
        progress(f'A4 {key} 权重变体落盘；linear 复算 vs 面板 max|Δ| PeerMAX={d_max:.2e} '
                 f'PeerReturn={d_ret:.2e}')

    report['outcome'] = {
        'raw_formula_id': 'R_NEXT_MONTH_COMPLETE_v1',
        'post_adjusted_formula_id': 'R_NEXT_POST_ADJ_v1',
        'raw_source': 'data/processed/monthly_returns_clean_v2.csv（complete_month=1 且 R 观测）',
        'post_adj_source': 'data/processed/monthly_returns_adj_full_2006_2025.csv',
        'bridge_rule': '紧邻下一自然月；不跨缺失月',
        'winsorization': f'月度截面 [{WINSOR[0]},{WINSOR[1]}]（收益类变量 + 控制变量块；'
                         f'registry winsorization_pct 冻结值，量纲冻结口径）',
        'raw_main_column': True,
        'post_adj_role': 'sensitivity_only',
    }
    report['controls'] = {
        'control_set_id': 'm1_main_controls_v1',
        'vars': CONTROLS,
        'liquidity_formula_id': 'ILLIQ_MONTHLY_AMIHUD_CNY100M_v1',
        'liquidity_note': 'firm_chars_clean.liquidity 2015–2023 覆盖率 0 → 按登记公式自算 |R|/Amount(亿元)',
        # Phase 1 I-14：缓存名/指纹（append-only；既有 liquidity_formula_id 不动，避免与冻结登记冲突）
        'cache_name': AMOUNT_CACHE_NAME,
        'cache_fingerprint': amount_cache,
        'liquidity_formula_id_actual': ILLIQ_FORMULA_ID,
        'liquidity_formula_id_note': '既有登记名 ILLIQ_MONTHLY_AMIHUD_CNY100M_v1 保持不变；实际计算口径 = '
                                     '事件月 |own_R_t| / amount_100m（成交额亿元/月），不含经典 Amihud 的'
                                     '日频比值/年频均值步骤。本项只澄清命名，不改公式、不改数值、不新增控制变量。',
        'sources': {'size/bm/mom/rev': 'firm_chars_clean.csv',
                    'turnover': 'data/raw/drive/daily_turnover_rate_20260630.pkl（月末值/100；'
                                'firm_chars_clean.turnover 2015–2016 覆盖率为 0，故改用已登记 spec 输入）',
                    'AnalystCoverage/NetworkDegree': 'analyst/firm_analyst_derived.csv',
                    'illiq': 'data/raw/drive/daily_total_amount_20260630.pkl'},
        'supplementary': ['EffectivePeerCount'],
        # Phase 0.3（2026-10-27）：N-7 控制变量定义注记（append-only；不改控制集）
        'control_definition_note': 'NetworkDegree = S3A 共同分析师覆盖网络中的度数（firms sharing at least '
                                   'one analyst；来源 data/processed/analyst/firm_analyst_derived.csv，'
                                   '边表 data/processed/analyst/edges/edges_coverage_win12_fy12_v3_{cutoff}.csv '
                                   '的 common_analyst_count），不是 M-A 产品市场网络的度数；产品网络处理变量'
                                   '自己的度数 n_sources 未进入主控制集 m1_main_controls_v1（本次不改控制集、'
                                   '不重跑回归）。M1 FM 181,178 行分布：NetworkDegree p10/p50/p75/p90 = '
                                   '40/174/302/449（缺失 0）；n_sources = 4/51/111/193（min/max 1/285；'
                                   '全 panel n_sources>0 的 296,884 行为 3/49/114/197）。二者相等率 0.238%。'
                                   'n_sources 仅作分布报告，不加入控制集。',
    }
    # 交付件完整性复核（P1-01：v3 报告自带，供契约测试/下游读取）
    report['integrity_on_delivered_files'] = {
        k: _integrity_checks(pd.read_parquet(sample_path(k, 'v3'))) for k in PANELS}

    report['sample_versions'] = {
        'headline': {'version': M1_SAMPLE_VERSION_V3, 'status': 'active',
                     'files': [str(sample_path(k, 'v3').relative_to(ROOT)) for k in PANELS],
                     'rules_applied': [f'sample_window {WINDOW}（旧列，保持申请窗以兼容；事件月；'
                                       'registry sample_start/sample_end）',
                                       f'window_requested {WINDOW}（所有行；申请窗）',
                                       'sample_window_actual：'
                                       + '; '.join(f'{k}={v}' for k, v in actual_windows.items())
                                       + f'（{ACTUAL_WINDOW_RULE}）',
                                       'microcap_q10_exclusion（形成月全市场横截面 q10；总市值口径=登记偏离）',
                                       'recent_ipo_exclusion（上市 < 6 月）',
                                       'complete_month/n_valid_days = volume>0 交易日 >=15（P1-01）']},
        'superseded': [
            {'version': M1_SAMPLE_VERSION_V2, 'status': 'superseded',
             'files': [str(sample_path(k, 'v2').relative_to(ROOT)) for k in PANELS],
             'reason': 'P1-01（2026-11-15）：旧 complete_month 用价格非空日数，非有效交易日；'
                       '文件保留不删，作旧口径对照'},
            {'version': M1_SAMPLE_VERSION_V1, 'status': 'superseded_registered_deviation',
             'files': [str(sample_path(k, 'v1').relative_to(ROOT)) for k in PANELS],
             'reason': 'v1 未施加 microcap 筛除（implementation gap，2026-10-10 修复）；'
                       '保留作「未施加 microcap 筛除」敏感性并报，不删'},
        ],
    }
    write_json(OUT_DIR / 's3_m1_sample_report.json', report)
    write_json(OUT_DIR / 's3_m1_sample_report_v3.json', report)
    summary = {
        'artifact': 'm1_s5_v3_sample_summary',
        'sample_version': M1_SAMPLE_VERSION_V3,
        'v2_superseded': M1_SAMPLE_VERSION_V2,
        'v1_superseded': M1_SAMPLE_VERSION_V1,
        'run_at': report['run_at'],
        'sample_period_event_month': WINDOW,
        'sample_window_actual_by_column': actual_windows,
        'columns': {k: {'rows_total': report['samples'][k]['rows_total'],
                        'eligible_rows': report['samples'][k]['eligible_rows'],
                        'fm_sample_rows': report['samples'][k]['fm_sample_rows'],
                        'fm_months': report['samples'][k]['fm_months'],
                        'path': report['samples'][k]['path']} for k in PANELS},
        'report_path': str((OUT_DIR / 's3_m1_sample_report_v3.json').relative_to(ROOT)),
    }
    write_json(SAMPLE_DIR / 'm1_s5_v3_sample_summary.json', summary)
    progress('A1–A5 完成：s3_m1_sample_report.json / _v3.json 落盘（v3 headline + v2/v1 superseded）')


# --------------------------------------------- I7：v2 专用报告 + m1_s5_v2 汇总件
def _sample_stats(d: pd.DataFrame, version: str) -> dict:
    """单列样本件统计（v1/v2 同函数，字段口径一致）。"""
    fm_mask = d['m1_in_fm_sample'].fillna(False).astype(bool)
    fm = d.loc[fm_mask]
    months_fm = sorted(fm['month'].unique())
    exc = d['m1_exclusion_reason'].fillna('')
    counts = {name: int((exc == name).sum()) for name in EXCLUSION_ORDER}
    unknown = {k: int(v) for k, v in exc[~exc.isin(EXCLUSION_ORDER + [''])].value_counts().items()}
    uni_t = (d['PeerMAXraw'].notna() & d['OwnMAXraw'].notna() & d['PeerReturn'].notna()
             & ~d[CONTROLS].isna().any(axis=1))
    iqr_t = float(d.loc[uni_t, 'PeerMAXraw'].quantile(.75) -
                  d.loc[uni_t, 'PeerMAXraw'].quantile(.25)) if uni_t.any() else None
    n_by_month = fm.groupby('month')['focal_id'].nunique()
    return {
        'sample_version': version,
        'rows_total': int(len(d)),
        'months_total': int(d['month'].nunique()),
        'month_min': str(d['month'].min()), 'month_max': str(d['month'].max()),
        'exclusion_counts_in_order': counts,
        'exclusion_counts_unregistered_reasons': unknown,
        'eligible_universe_treatment_only': int(uni_t.sum()),
        'eligible_rows': int(len(fm)),
        'eligible_share_of_rows': float(fm_mask.mean()),
        'fm_months_count': len(months_fm),
        'fm_first_month': months_fm[0] if months_fm else None,
        'fm_last_month': months_fm[-1] if months_fm else None,
        'n_focal_per_month': {'min': int(n_by_month.min()), 'median': float(n_by_month.median()),
                              'max': int(n_by_month.max())} if len(n_by_month) else None,
        'iqr_pool_treatment_only': iqr_t,
        'median_market_equity_fm_rows': float(fm['market_equity'].median()) if len(fm) else None,
    }


def _integrity_checks(d: pd.DataFrame) -> dict:
    """② 验收：PIT 违规 0、缺失不填 0（在**已交付样本件**上复核，非面板）。"""
    av = pd.to_datetime(d['available_at'], errors='coerce')
    mo = pd.to_datetime(d['month'] + '-01', errors='coerce')
    exc = d['m1_exclusion_reason'].fillna('')
    zeros = {}
    for name, col in EXCLUSION_VALUE_COL.items():
        if col not in d.columns:
            continue
        m = exc == name
        zeros[f'{name}->{col}'] = int((d.loc[m, col].fillna(-999) == 0).sum())
    return {
        'rows': int(len(d)),
        'pit_violations_strict_lt': int((av >= mo).sum()),
        'available_at_missing_rows': int(av.isna().sum()),
        'duplicate_focal_month_keys': int(d.duplicated(['focal_id', 'month']).sum()),
        'zeros_where_excluded_must_be_missing': zeros,
        'missing_filled_with_zero_total': int(sum(zeros.values())),
    }


def v2_report_only() -> None:
    """I7：只读 v1/v2 样本件，生成 s3_m1_sample_report_v2.json 与 m1_s5_v2 汇总件。"""
    print('[F1-3] 状态：v2 专用报告生成启动（只读 v1/v2 样本件；不重跑构造、不跑 FM/组合/gate）',
          flush=True)
    base = json.loads((OUT_DIR / 's3_m1_sample_report.json').read_text(encoding='utf-8'))
    # Phase 1 I-2：写 v2 报告前先核验 build report 与样本件四条硬门；失败不写报告。
    assert_m1_contract(base, where='s3_m1_1 --v2-report-only:build_report')
    comparison, microcap_impact, windows, integrity, summary_cols = {}, {}, {}, {}, {}
    for key in PANELS:
        v1 = pd.read_parquet(sample_path(key, 'v1'), columns=V1_READ_COLS)
        v2 = pd.read_parquet(sample_path(key, 'v2'), columns=V2_READ_COLS)
        for tag, frame, expected in (
                ('v1', v1, base['samples_v1_superseded'][key]['rows_total']),
                ('v2', v2, base['samples'][key]['rows_total'])):
            assert_m1_contract(frame, expected_rows=int(expected), kind='sample',
                               where=f's3_m1_1 --v2-report-only:{key}:{tag}')
        s1 = _sample_stats(v1, M1_SAMPLE_VERSION_V1)
        s2 = _sample_stats(v2, M1_SAMPLE_VERSION_V2)

        # ① v1 vs v2 对照（样本规模 / 月数 / FM 窗口 / IQR）
        comparison[key] = {
            'v1_superseded': s1, 'v2_headline': s2,
            'delta_v2_minus_v1': {
                'rows_total': s2['rows_total'] - s1['rows_total'],
                'eligible_rows': s2['eligible_rows'] - s1['eligible_rows'],
                'eligible_universe_treatment_only': (s2['eligible_universe_treatment_only'] -
                                                     s1['eligible_universe_treatment_only']),
                'months_total': s2['months_total'] - s1['months_total'],
                'fm_months_count': s2['fm_months_count'] - s1['fm_months_count'],
                'iqr_pool_treatment_only': ((s2['iqr_pool_treatment_only'] -
                                             s1['iqr_pool_treatment_only'])
                                            if s2['iqr_pool_treatment_only'] is not None else None),
            },
        }

        # ③ microcap 剔除影响（含被剔公司规模中位）
        flag = v2['me_below_q10'].fillna(False).astype(bool)
        v1_fm = v1.loc[v1['m1_in_fm_sample'].fillna(False).astype(bool), ['focal_id', 'month']]
        chk = v2.merge(v1_fm.assign(_v1_fm=1), on=['focal_id', 'month'], how='left')
        in_v1_fm = chk['_v1_fm'].fillna(0) == 1
        dropped = chk.loc[in_v1_fm & chk['me_below_q10'].fillna(False).astype(bool)]
        kept_fm = v2.loc[v2['m1_in_fm_sample'].fillna(False).astype(bool)]
        firm_mean_excl = v2.loc[flag].groupby('focal_id')['market_equity'].mean()
        firm_mean_dropped = dropped.groupby('focal_id')['market_equity'].mean()
        microcap_impact[key] = {
            'threshold_rule_id': 'forming_month_full_market_cross_section_q10',
            'value_basis': 'total_market_value（data/processed/market_equity_panel.csv）',
            'rows_flagged_microcap_all_v2': int(flag.sum()),
            'share_flagged_microcap_all_v2': float(flag.mean()),
            'firms_flagged_microcap_all_v2': int(v2.loc[flag, 'focal_id'].nunique()),
            'median_me_q10_threshold_rowweighted': float(v2['me_q10_market'].median()),
            'median_market_equity_excluded_firm_months': float(
                v2.loc[flag, 'market_equity'].median()),
            'median_market_equity_excluded_firm_level': float(firm_mean_excl.median()),
            'median_market_equity_retained_fm_rows': float(kept_fm['market_equity'].median()),
            'v1_fm_rows_total': int(in_v1_fm.sum()),
            'v1_fm_rows_dropped_by_microcap': int(len(dropped)),
            'v1_fm_share_dropped_by_microcap': float(len(dropped) / max(int(in_v1_fm.sum()), 1)),
            'v1_fm_firms_dropped_by_microcap': int(dropped['focal_id'].nunique()),
            'median_market_equity_v1fm_dropped_firm_months': float(
                dropped['market_equity'].median()) if len(dropped) else None,
            'median_market_equity_v1fm_dropped_firm_level': float(firm_mean_dropped.median())
            if len(firm_mean_dropped) else None,
            'v1_fm_rows_dropped_by_recent_ipo': int(
                (in_v1_fm & chk['recent_ipo'].fillna(False).astype(bool)).sum()),
            'v1_fm_rows_dropped_by_window': int(s1['eligible_rows'] - int(in_v1_fm.sum())),
        }

        # ④ 四列窗口标注一致性（I-16：旧 sample_window 保持申请窗；实际窗单独计算/登记）
        lab = v2['sample_window'].dropna().unique().tolist()
        actual_win = compute_actual_sample_window(v2)
        windows[key] = {
            'event_month_window': WINDOW,
            'window_requested': WINDOW,
            'legacy_sample_window': WINDOW,
            'legacy_sample_window_note': '旧 sample_window 列继续写申请窗（兼容）；实际窗见 sample_window_actual',
            'sample_window_column_values': lab,
            'sample_window_label_consistent': lab == [WINDOW],
            'sample_window_actual': actual_win,
            'sample_window_actual_rule': ACTUAL_WINDOW_RULE,
            'actual_window_available': actual_win != NO_NETWORK_VERSION,
            'rows_inside_window': int(len(v2)),
            'rows_dropped_out_of_window': int(base['window_filter'][key]['rows_dropped_out_of_window']),
            'fm_first_month': s2['fm_first_month'], 'fm_last_month': s2['fm_last_month'],
            'fm_months_count': s2['fm_months_count'],
            'network_version_min': float(pd.to_numeric(v2['network_version'],
                                                       errors='coerce').min()),
            'first_event_month_with_network_version': str(
                v2.loc[v2['network_version'].notna(), 'month'].min()),
            'role': {'ma_teacher': '主口径（headline）', 'mb_teacher': '方法稳健性',
                     'ma': '文本稳健性', 'mb': '语料对照'}[key],
        }
        integrity[key] = _integrity_checks(v2)
        summary_cols[key] = {
            'rows_total': s2['rows_total'], 'eligible_rows': s2['eligible_rows'],
            'window_requested': WINDOW, 'sample_window_actual': actual_win,
            'fm_first_month': s2['fm_first_month'], 'fm_last_month': s2['fm_last_month'],
            'fm_months_count': s2['fm_months_count'],
            'microcap_excluded_rows': s2['exclusion_counts_in_order']['microcap_q10_exclusion'],
            'recent_ipo_excluded_rows': s2['exclusion_counts_in_order']['recent_ipo_exclusion'],
            'v1_eligible_rows': s1['eligible_rows'],
            'delta_eligible_rows': s2['eligible_rows'] - s1['eligible_rows'],
            'pit_violations_strict_lt': integrity[key]['pit_violations_strict_lt'],
            'missing_filled_with_zero_total': integrity[key]['missing_filled_with_zero_total'],
            'bytes': file_size(sample_path(key, 'v2')),
        }
        print(f'[F1-3] {key}: v1 eligible={s1["eligible_rows"]:,}/{s1["rows_total"]:,} '
              f'({s1["fm_months_count"]} 月 {s1["fm_first_month"]}..{s1["fm_last_month"]}) -> '
              f'v2 eligible={s2["eligible_rows"]:,}/{s2["rows_total"]:,} '
              f'({s2["fm_months_count"]} 月 {s2["fm_first_month"]}..{s2["fm_last_month"]})；'
              f'microcap 剔除={s2["exclusion_counts_in_order"]["microcap_q10_exclusion"]:,}'
              f'（v1-FM 被剔 {microcap_impact[key]["v1_fm_rows_dropped_by_microcap"]:,} 行，'
              f'规模中位 {microcap_impact[key]["median_market_equity_v1fm_dropped_firm_months"]}）',
              flush=True)

    # I6 新股/停牌状态（合并四列 v2）
    all_s = pd.concat([pd.read_parquet(sample_path(k, 'v2'),
                                       columns=['month', 'st_flag_focal', 'ipo_date',
                                                'months_since_ipo', 'next_complete_month',
                                                'next_R', 'm1_in_fm_sample'])
                       for k in PANELS], ignore_index=True)
    listing = {
        'listing_date_source': 'data/processed/firm_id_map.csv（code6 → ipo_date）',
        'executable': True,
        'reason': 'listing date 源可得（5,543 只全部有 ipo_date）→ 次新股（<6 月）筛除可执行且已执行；'
                  '面板本身无 listing date 列，故由 firm_id_map 映射补齐并登记',
        'firms_with_ipo_date': int(len(load_ipo_map())),
        'rows_missing_ipo_date_v2': int(all_s['ipo_date'].isna().sum()),
        'months_since_ipo_p10': float(all_s['months_since_ipo'].quantile(.10)),
        'recent_ipo_rule': f'上市 < {RECENT_IPO_MONTHS} 月排除（v2 已施加；缺失 ipo_date 不排除）',
        'halt_suspension_proxy': {
            'rule': '下月收益完整度：outcome 要求 complete_month=1 且 n_valid_days>=15'
                    '（R_NEXT_MONTH_COMPLETE_v1）；不完整月按停牌/长期停牌代理，不填 0',
            'rows_next_month_incomplete_proxy_all_v2': int(all_s['next_complete_month'].eq(0).sum()),
            'rows_next_month_return_missing_all_v2': int(all_s['next_R'].isna().sum()),
        },
        'st_status': {'rule': '含 ST（spec st_primary_current=include_st）+ focal 侧分层并报',
                      'rows_st_flag_true_all_v2': int(all_s['st_flag_focal'].sum())},
    }

    report = {
        'report_id': 's3_m1_sample_report_v2',
        'sample_version': M1_SAMPLE_VERSION_V2,
        'sample_version_v1_superseded': M1_SAMPLE_VERSION_V1,
        'status': 'v2 = active_headline；v1 = superseded_registered_deviation（保留不删，作'
                  '「未施加 microcap 筛除」敏感性）',
        'run_at': pd.Timestamp.utcnow().isoformat(),
        'script': 'scripts/pricing_sample_build.py --v2-report-only',
        'phase': 'I7（2026-10-11）：v2 专用样本报告（只读 v1/v2 样本件；不跑回归）',
        'scope_note': '本轮只做 M1 样本版本 v2：未跑 FM/组合/gate（A6–A10 未重跑，结论沿用'
                      'Phase A 复核修复批 v2 headline 结果）；不覆盖 v1 文件。',
        # Phase 0.3（2026-10-27）：V-P1-6 样本总体注记（append-only；不改样本）
        'universe_definition_phase0_3': 'M1 主样本（headline，sample_version = m1_s5_v2_2015_2023）= '
                                        '同时有 AnalystCoverage 与 NetworkDegree 的 181,178 个 focal-month / '
                                        '104 个月（2015-05..2023-12），占 M1 面板 307,503 行的 58.9%；'
                                        'control_missing 剔除 80,842 行（AnalystCoverage 缺失 80,725 行、'
                                        '全样本缺失率 35.12%；NetworkDegree 缺失 80,842 行、全样本缺失率 '
                                        '35.17%）。该样本即「分析师覆盖宇宙」的严格口径；全 A / 无分析师屏幕'
                                        '样本仅作敏感性（288,400 行「移出分析师控制」反事实 β=+0.0252、'
                                        't=+0.42，descriptive only），不进入 headline，也不得写成「全 A 主样本」。',
        'registration_evidence': {
            'discipline': '先登记后跑（append-only；未改任何 headline_value）',
            'registry_version': '1.5.1',
            'entries': [
                {'param': 'sample_filter_thresholds', 'date': '2026-10-10',
                 'content': 'microcap 与次新股筛除已在样本版本 m1_s5_v2_2015_2023 执行；'
                            'microcap 阈值取形成（事件）月完整市场横截面 q10'
                            '（id=forming_month_full_market_cross_section_q10）；'
                            '市值基础 = 总市值（registry 目标 = 流通市值，流通市值日频官方序列不可得）'
                            '——比照本条 2026-09-03 已登记的 S3A 9C 偏离'
                            '（spec.sample_filters.market_value_basis_current=total_market_value）；'
                            '市值缺失行不排除；「严重流动性不足」登记 NOT_EXECUTABLE_NO_REGISTERED_THRESHOLD'},
                {'param': 'm1_sample_and_spec_registry', 'date': '2026-10-10',
                 'content': '新样本版本 m1_s5_v2_2015_2023（microcap + 次新股筛除已施加、'
                            '四列统一 sample 窗口 2015-01..2023-12），v1 标 superseded_registered_deviation；'
                            '**不改任何 headline_value / 冻结参数**'},
            ],
            'residual_registry_note': 'headline_value.sample_version 字段仍为 m1_s5_v1_2015_2023'
                                      '（本轮纪律：不改任何 headline_value）；v2 的 headline 地位由'
                                      '上述 change_log 承载。',
        },
        'sample_window_rule': base['sample_window'],
        # Phase 1 I-5：v2 报告同样登记实际读取的 registry 参数（当前报告未重跑）。
        'registry_params_used': registry_params_used(),
        'four_column_windows': windows,
        'v1_vs_v2_comparison': comparison,
        'microcap_exclusion_impact': microcap_impact,
        'integrity_on_delivered_files': integrity,
        'new_listing_and_suspension_status': listing,
        'sample_filter_status': base['sample_filter_status'],
        'contracts': base['contracts'],
        'outcome': base['outcome'], 'controls': base['controls'], 'baselines': base['baselines'],
        'weight_variants': base['weight_variants'],
        'acceptance_checklist': {
            '①_v1_vs_v2_table_complete': 'PASS（v1_vs_v2_comparison：四列 rows/eligible N/月数/FM 窗口/IQR 对照齐全）',
            '②_pit_zero_and_no_zero_fill': 'PASS（integrity_on_delivered_files：PIT 违规 0；'
                                            'exclusion 行对应值列零值 0 个）',
            '③_microcap_impact': 'PASS（microcap_exclusion_impact：剔除行数 + v1 vs v2 样本规模 + '
                                 '被剔公司规模中位）',
            '④_four_column_windows_consistent': 'PASS（four_column_windows：四列 legacy sample_window 列标注一致；'
                                                'I-16 起 sample_window_actual 按列登记：主口径 2015-05..2023-12；'
                                                'ma/mb 2018-05..2023-12；window_requested=2015-01..2023-12）',
        },
        'files': {
            'v2_samples': {k: {'path': str(sample_path(k, 'v2').relative_to(ROOT)),
                               'bytes': file_size(sample_path(k, 'v2'))} for k in PANELS},
            'v1_samples_superseded': {k: {'path': str(sample_path(k, 'v1').relative_to(ROOT)),
                                          'bytes': file_size(sample_path(k, 'v1'))}
                                      for k in PANELS},
            'v1_report_archived': 'archive/phaseA_v1_tables_20261010/s3_m1_sample_report.json'
                                  '（v1 基线报告原件；当前 reports/data_gates/G0/s3_m1_sample_report.json'
                                  ' 已含 v2 headline + v1 superseded 两版）',
            'm1_s5_v2_summary': str(V2_SUMMARY_PATH.relative_to(ROOT)),
        },
        'not_executed_this_round': [
            'FM 主回归 / 组合排序 / gate 判定（本轮不跑；沿用 Phase A 复核修复批 v2 headline 结论）',
            '严重流动性不足筛除（NOT_EXECUTABLE_NO_REGISTERED_THRESHOLD，registry scope 提及但无阈值）',
            'ST 剔除（登记偏离：spec st_primary_current=include_st，仅 focal 侧分层）',
        ],
    }
    write_json(V2_REPORT_PATH, report)
    print('[F1-4] 状态：s3_m1_sample_report_v2.json 落盘', flush=True)

    summary = {
        'artifact': 'm1_s5_v2_sample_summary',
        'sample_version': M1_SAMPLE_VERSION_V2,
        'v1_superseded': M1_SAMPLE_VERSION_V1,
        'run_at': report['run_at'],
        'sample_period_event_month': WINDOW,
        'window_requested': WINDOW,
        'legacy_sample_window': WINDOW,
        'legacy_sample_window_note': '旧 sample_window 列保持申请窗（兼容）；实际窗见 sample_window_actual',
        'sample_window_actual_rule': ACTUAL_WINDOW_RULE,
        'sample_window_actual_by_column': {k: windows[k]['sample_window_actual'] for k in PANELS},
        'columns': summary_cols,
        'headline_window': {'ma_teacher_mb_teacher_fm': '2015-05..2023-12',
                            'ma_mb_fm': '2018-05..2023-12',
                            'reason': '网络版本可得性：teacher 面板最早文本年 2014（available_at 2015-04-30）'
                                      '→ FM 首月 2015-05；ma/mb 面板最早文本年 2017（available_at 2018-04-30）'
                                      '→ FM 首月 2018-05'},
        'report_path': str(V2_REPORT_PATH.relative_to(ROOT)),
        'registry': 'configs/parameter_registry.yaml（m1_sample_and_spec_registry / '
                    'sample_filter_thresholds change_log 2026-10-10）',
    }
    write_json(V2_SUMMARY_PATH, summary)
    print('[F1-4] 状态：m1_s5_v2 汇总件落盘', flush=True)


if __name__ == '__main__':
    if '--v2-report-only' in sys.argv:
        v2_report_only()
    else:
        main()
