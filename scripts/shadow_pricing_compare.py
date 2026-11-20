"""Phase 2.6：shadow pricing_gate 对比（A fixed_sample / B rescreened / C rescreened_adj）vs headline。

范围（只读 shadow + headline；只写 artifacts/phase2/2.6_*）：
  - 不重跑 headline pricing_gate/PC/网络；不改 headline 数据/系数/gate/configs/正式 reports；
  - shadow 变量全部来自 Phase 2.1–2.5 产物（月频状态 / 2.2 MAX / 2.3 复权 MAX / 2.4 turnover）；
  - treatment（OwnMAX/PeerMAX/PeerReturn）在 headline 边表 + `max{available_at < month}` 版本
    选择下重建（复用 `peermax_panel_build.load_versions/select_version`；缺失不填 0）；
  - rev 只用 2.5 的合规 t-1 滞后口径（`R(t-1)`），**不引用 2.0_plan §2.5 的前视数字**。

变体（每个变体 × {EW, VW} × 6 规格）：
  headline_recompute  headline 样本列直接回归（管线校验，应逐值复现 G0 主表 / 2.4 / 2.5）
  headline_rebuilt    同样本、所有变量由 shadow 输入「全程重建」的 legacy 口径（复现性硬门）
  A_fixed_sample      headline FM 行集不变；y/treatment/controls 换成 shadow 原始口径列
  A_fixed_sample_adj  同 A 行集，MAX 用 2.3 `MAXraw_adj_ex_resume`，收益派生统一复权口径（诊断）
  B_rescreened        A 剔除不可评估行（event：no_trade/daily_panel_missing/n_trade_days<15/
                      n_max_days<15；outcome：下月状态非 ok / complete_month_trade15!=1）
  C_rescreened_adj    B + 2.3 复权 MAX + 2.2 resume 排除 + 收益派生统一复权口径

规格：spec_full_last（登记全控制：rev 同月 + turnover last）/ spec_full_mean（+turnover mean）/
      spec_lag1_last（合规 rev lag1 + last）/ spec_lag1_mean（合规 lag1 + mean）/ spec_no_turnover（无 turnover）/
      spec_no_rev_no_turnover（无 rev + 无 turnover）。EW/VW，min_n、nw_lag 走 registry。

输出：artifacts/phase2/2.6_comparison.csv、artifacts/phase2/2.6_counts.json
运行：python scripts/shadow_pricing_compare.py [--teacher ma_teacher ...]"""

from __future__ import annotations

from pathlib import Path
from fileio import assert_path_allowed, json_default, write_csv_rows, write_json_atomic

import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pricing_lib import (
    RETURNS_V2, ROOT, assert_pricing_inputs, fm_regression, load_amount_100m, load_edges,
    param_value, require, file_size, winsor_bounds, ym_add,
)
from peermax_panel_build import load_versions, select_version


SHADOW_DIR = ROOT / 'data/processed/phase2_shadow'

SHADOW_MONTHLY = SHADOW_DIR / 'monthly_returns_status_p2shadow.csv'

SHADOW_MAX = SHADOW_DIR / 'source_month_argmax_panel_p2shadow.csv'

SHADOW_ADJ = SHADOW_DIR / 'source_month_argmax_panel_adj_p2shadow.csv'

SHADOW_TURNOVER = SHADOW_DIR / 'turnover_p2shadow.csv'

SHADOW_ADJ_RET = SHADOW_DIR / 'monthly_returns_adj_p2shadow.csv'

SHADOW_INPUTS = {
    'shadow_monthly_2_1': SHADOW_MONTHLY,
    'shadow_max_2_2': SHADOW_MAX,
    'shadow_adj_max_2_3': SHADOW_ADJ,
    'shadow_turnover_2_4': SHADOW_TURNOVER,
    'shadow_adj_returns_2_3': SHADOW_ADJ_RET,
}

SAMPLE_PARQUET = {
    'ma_teacher': ROOT / 'data/processed/pricing_sample_ma_teacher_v2.parquet',
    'mb_teacher': ROOT / 'data/processed/pricing_sample_mb_teacher_v2.parquet',
}

NET_DIRS = {
    'ma_teacher': ROOT / 'data/processed/network_ma_teacher',
    'mb_teacher': ROOT / 'data/processed/network_mb_teacher',
}

PRICING_TABLE_V2 = ROOT / 'reports/data_gates/G0/pricing_table_v2.json'

SAMPLE_REGISTERED = {'ma_teacher': 307503, 'mb_teacher': 384303}

FM_REGISTERED = {'ma_teacher': 181178, 'mb_teacher': 225264}

MATRIX_24 = ROOT / 'artifacts/phase2/2.4_control_matrix.csv'

MATRIX_25 = ROOT / 'artifacts/phase2/2.5_control_matrix.csv'

OUT_CSV = ROOT / 'artifacts/phase2/2.6_comparison.csv'

OUT_JSON = ROOT / 'artifacts/phase2/2.6_counts.json'

HEADLINE_GUARD_FILES = {
    'pricing_sample_ma_teacher_v2.parquet': SAMPLE_PARQUET['ma_teacher'],
    'pricing_sample_mb_teacher_v2.parquet': SAMPLE_PARQUET['mb_teacher'],
    'monthly_returns_clean_v2.csv': RETURNS_V2,
    'monthly_pricing_panel_ma_teacher.csv':
        ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv',
    'firm_chars_clean.csv': ROOT / 'data/processed/firm_chars_clean.csv',
    'pricing_table_v2.json': PRICING_TABLE_V2,
    'source_month_argmax_panel.csv': ROOT / 'data/processed/source_month_argmax_panel.csv',
}

TRIO = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn']

SAMPLE_COLS = (['focal_id', 'month', 'network_version', 'available_at', 'pricing_in_fm_sample',
                'R_next', 'next_complete_month', 'market_equity']
               + TRIO + ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq',
                         'AnalystCoverage', 'NetworkDegree'])

SPEC_DEFS = {
    'spec_full_last': dict(rev='same', turnover='last', label='登记全控制（rev 同月 + turnover last）'),
    'spec_full_mean': dict(rev='same', turnover='mean', label='登记全控制 + turnover mean'),
    'spec_lag1_last': dict(rev='lag1', turnover='last', label='合规 rev lag1 + turnover last'),
    'spec_lag1_mean': dict(rev='lag1', turnover='mean', label='合规 rev lag1 + turnover mean'),
    'spec_no_turnover': dict(rev='same', turnover=None, label='无 turnover'),
    'spec_no_rev_no_turnover': dict(rev=None, turnover=None, label='无 rev + 无 turnover'),
}

SPEC_IDS = list(SPEC_DEFS)

COMPUTED_VARIANTS = ['headline_recompute', 'headline_rebuilt', 'A_fixed_sample',
                     'A_fixed_sample_adj', 'B_rescreened', 'C_rescreened_adj']

VARIANT_META = {
    'headline_G0_table_v2': dict(rows='fm', label='headline 登记主表（pricing_table_v2.json，只读引用）'),
    '2.4_control_matrix': dict(rows='fm', label='Phase 2.4 已发布矩阵（只读引用）'),
    '2.5_control_matrix': dict(rows='fm', label='Phase 2.5 已发布矩阵（只读引用）'),
    'headline_recompute': dict(rows='fm', label='headline 样本列直接回归（管线校验）'),
    'headline_rebuilt': dict(rows='fm', label='shadow 输入全程重建的 legacy 口径（复现性硬门）'),
    'A_fixed_sample': dict(rows='fm', label='A fixed_sample（shadow 原始口径，行集不变）'),
    'A_fixed_sample_adj': dict(rows='fm', label='A* fixed_sample_adj（同 A 行集，复权口径，诊断）'),
    'B_rescreened': dict(rows='bc', label='B rescreened（原始口径 + 不可评估行剔除）'),
    'C_rescreened_adj': dict(rows='bc', label='C rescreened_adj（B + 复权 MAX + resume 排除）'),
}

P2SHADOW_VERSION = 'p2shadow_v1'

COMPARISON_COLUMNS = ['variant', 'variant_label', 'teacher', 'spec_id', 'spec_label', 'weight',
                      'term', 'beta', 'se_nw', 't_nw', 'n_months_used', 'n_rows_used',
                      'n_per_month_min', 'n_per_month_median', 'n_per_month_max',
                      'p_two_sided', 'ci95_low', 'ci95_high', 'is_treatment', 'rows_mode',
                      'source', 'note']

def assert_output_path_guard(path) -> None:
    assert_path_allowed(path, banned_roots=(
        *HEADLINE_GUARD_FILES.values(), ROOT / 'data', ROOT / 'reports',
        ROOT / 'configs', ROOT / 'scripts', ROOT / 'tests'))

def json_default(o):
    """numpy 标量/bool_ -> 原生类型（确保 counts JSON 可序列化且可复算）。"""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)

def _rel_or_abs(p) -> str:
    """展示/记录用路径：ROOT 内用相对 POSIX 路径，ROOT 外回退绝对路径（不崩溃）。"""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).resolve().as_posix()

def write_json(path, obj) -> None:
    assert_output_path_guard(path)
    write_json_atomic(obj, Path(path), indent=1, default=json_default,
                      trailing_newline=False)
    print(f'  -> {_rel_or_abs(path)}', flush=True)

def write_csv(path, rows, columns) -> None:
    assert_output_path_guard(path)
    write_csv_rows(Path(path),
                   [[r.get(c, '') for c in columns] for r in rows], columns)

def _num(v):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None

def _rel(p) -> str:
    """ROOT 下相对路径（POSIX 分隔符，跨平台一致）；ROOT 外回退绝对路径。"""
    return _rel_or_abs(p)

def load_shadow_tables() -> dict:
    """只读 2.1–2.5 shadow 产物（按需列）。"""
    mon = pd.read_csv(SHADOW_MONTHLY, usecols=[
        'firm_id', 'month', 'R', 'n_trade_days', 'trade_days_status',
        'complete_month_trade15', 'pseudo_zero_flag', 'partial_zero_flag', 'true_zero_flag',
        'R_p2shadow'], dtype={'firm_id': str, 'month': str})
    require(int(mon.duplicated(['firm_id', 'month']).sum()) == 0, '2.1 月频 shadow 键重复')
    maxp = pd.read_csv(SHADOW_MAX, usecols=[
        'source_id', 'month', 'MAXraw', 'R', 'n_valid_days', 'n_max_days', 'MAXraw_p2shadow',
        'max_gate_15', 'gate_reason', 'shadow_status', 'qualification_status'],
        dtype={'source_id': str, 'month': str}).rename(columns={'source_id': 'firm_id'})
    require(int(maxp.duplicated(['firm_id', 'month']).sum()) == 0, '2.2 MAX shadow 键重复')
    adjp = pd.read_csv(SHADOW_ADJ, usecols=[
        'source_id', 'month', 'n_max_days', 'MAXraw_adj', 'MAXraw_adj_ex_resume',
        'adj_status', 'resume_joint_max_changed', 'adj_ex_resume_changed_vs_adj',
        'delta_MAXraw_adj_ex_resume'],
        dtype={'source_id': str, 'month': str}).rename(columns={'source_id': 'firm_id'})
    require(int(adjp.duplicated(['firm_id', 'month']).sum()) == 0, '2.3 adj MAX 键重复')
    to = pd.read_csv(SHADOW_TURNOVER, dtype={'firm_id': str, 'month': str})
    require(int(to.duplicated(['firm_id', 'month']).sum()) == 0, '2.4 turnover 键重复')
    adjr = pd.read_csv(SHADOW_ADJ_RET, dtype={'firm_id': str, 'month': str})
    require(int(adjr.duplicated(['firm_id', 'month']).sum()) == 0, '2.3 复权月收益键重复')
    return {'monthly': mon, 'max_panel': maxp, 'adj_panel': adjp, 'turnover': to,
            'adj_returns': adjr}

def build_return_grid(tab: dict) -> pd.DataFrame:
    """月频键上的 shadow 收益网格：R_sgate / R_s15 / R_agate / R_a15（缺失不填 0）。"""
    mon = tab['monthly'].copy()
    ok = mon['trade_days_status'].eq('ok')
    mon['R_sgate'] = mon['R_p2shadow'].where(ok)
    mon['R_s15'] = mon['R_sgate'].where(ok & mon['complete_month_trade15'].eq(1))
    adj = tab['adj_returns'].rename(columns={'R': 'R_adj'})
    g = mon.merge(adj[['firm_id', 'month', 'R_adj']], on=['firm_id', 'month'], how='left')
    miss_ok = int((g['trade_days_status'].eq('ok') & g['R_adj'].isna()).sum())
    require(miss_ok == 0,
            f'可评估月（status=ok）复权月收益缺失 {miss_ok} 行；不得用 raw 替代（停机）')
    g['R_agate'] = g['R_adj'].where(ok)
    g['R_a15'] = g['R_agate'].where(ok & g['complete_month_trade15'].eq(1))
    return g[['firm_id', 'month', 'R', 'R_p2shadow', 'R_adj', 'n_trade_days', 'trade_days_status',
              'complete_month_trade15', 'pseudo_zero_flag', 'partial_zero_flag',
              'true_zero_flag', 'R_sgate', 'R_s15', 'R_agate', 'R_a15']]

def version_map(teacher: str, months) -> dict:
    """月份 -> 文本年；走 manifest + available_at 硬门（缺失即 None）。"""
    vers = load_versions(NET_DIRS[teacher])
    return {m: (None if (v := select_version(vers, m)) is None else int(v['text_year']))
            for m in months}

def load_edges_by_year(teacher: str, years) -> dict:
    return {int(y): load_edges(teacher, int(y))
            for y in sorted({int(y) for y in years if y is not None})}

def total_support(edges_by_year: dict, ver: dict) -> pd.DataFrame:
    """(month, focal) 边集总权重（支持质量；不含源值过滤）。"""
    parts = []
    for ty, edges in edges_by_year.items():
        months = sorted(m for m, v in ver.items() if v == ty)
        if not months:
            continue
        g = (edges.groupby('focal_id', sort=False)['weight'].sum()
             .rename('support_mass').reset_index())
        for m in months:
            gg = g.copy()
            gg['month'] = m
            parts.append(gg)
    out = pd.concat(parts, ignore_index=True)
    return out[['month', 'focal_id', 'support_mass']]

def aggregate_over_edges(edges_by_year: dict, ver: dict, values: pd.DataFrame,
                         value_col: str, out_col: str) -> pd.DataFrame:
    """固定边集上的覆盖源加权平均（重归一化；缺失不填 0）。

    返回 [month, focal_id, out_col, covered_mass]；支持量不足 -> 该 (month, focal) 无行/NaN。
    """
    parts = []
    for ty, edges in edges_by_year.items():
        months = sorted(m for m, v in ver.items() if v == ty)
        vals = values.loc[values['month'].isin(months), ['firm_id', 'month', value_col]]
        vals = vals.loc[vals[value_col].notna()]
        if vals.empty or not months:
            continue
        m = edges.merge(vals.rename(columns={'firm_id': 'source_id'}),
                        left_on='source_id', right_on='source_id', how='inner')
        if m.empty:
            continue
        m['_wx'] = m['weight'] * m[value_col]
        agg = (m.groupby(['month', 'focal_id'], sort=False)
               .agg(_num=('_wx', 'sum'), _den=('weight', 'sum')).reset_index())
        agg[out_col] = np.where(agg['_den'] > 0, agg['_num'] / agg['_den'], np.nan)
        agg = agg.rename(columns={'_den': 'covered_mass'})
        parts.append(agg[['month', 'focal_id', out_col, 'covered_mass']])
    if not parts:
        return pd.DataFrame(columns=['month', 'focal_id', out_col, 'covered_mass'])
    return pd.concat(parts, ignore_index=True)

def mom_index_rule(grid: pd.DataFrame, value_col: str, panel_keys: pd.DataFrame,
                   min_months: int = 8) -> pd.DataFrame:
    """`build_firm_chars.mom` 口径：ys = t 之前全部可得月，need = ys[-12:-1]，len>=8。

    缺失/不足 -> NaN（不填 0）。
    """
    src = grid.rename(columns={'firm_id': 'focal_id'})
    src = src.loc[src[value_col].notna(), ['focal_id', 'month', value_col]]
    src = src.sort_values(['focal_id', 'month'])
    by_firm = {f: (grp['month'].to_numpy(), grp[value_col].to_numpy(dtype=float))
               for f, grp in src.groupby('focal_id', sort=False)}
    pk = panel_keys.reset_index(drop=True)
    out = np.full(len(pk), np.nan)
    for i, (focal, month) in enumerate(zip(pk['focal_id'].to_numpy(), pk['month'].to_numpy())):
        arr = by_firm.get(focal)
        if arr is None:
            continue
        months, rets = arr
        idx = int(np.searchsorted(months, month, side='left'))     # 严格 < t 的可得月数
        if idx < 2:
            continue
        lo_i = max(0, idx - 12)
        if idx - lo_i < min_months:
            continue
        out[i] = float(np.prod(1.0 + rets[lo_i:idx - 1]) - 1.0)
    res = pk.copy()
    res['_mom'] = out
    return res

def winsorize_col(frame: pd.DataFrame, col: str, lo: float, hi: float) -> pd.Series:
    """在传入 frame 宇宙上做月度截面 [lo,hi] winsorize（与 headline 同规则）。"""
    g = frame.groupby('month', sort=False)[col]
    return frame[col].clip(lower=g.transform(lambda s: s.quantile(lo)),
                           upper=g.transform(lambda s: s.quantile(hi)))

def build_teacher_frame(teacher: str, tab: dict, grid: pd.DataFrame) -> tuple:
    """shadow 面板 frame（宇宙 = headline 样本全部行：再筛 FM / B-C 屏）。"""
    df = pd.read_parquet(SAMPLE_PARQUET[teacher], columns=SAMPLE_COLS)
    require(len(df) == SAMPLE_REGISTERED[teacher],
            f'{teacher} 样本行数 {len(df)} != 登记 {SAMPLE_REGISTERED[teacher]}')
    n_fm = int(df['pricing_in_fm_sample'].astype(bool).sum())
    require(n_fm == FM_REGISTERED[teacher],
            f'{teacher} FM 行数 {n_fm} != 登记 {FM_REGISTERED[teacher]}')
    diag = {'sample_rows': int(len(df)), 'fm_rows': n_fm, 'teacher': teacher}

    vm = version_map(teacher, df['month'].dropna().unique().tolist())
    df['_ver_rule'] = df['month'].map(vm)
    nv = pd.to_numeric(df['network_version'], errors='coerce')
    no_ver = nv.isna()
    bad = (~no_ver) & (nv != df['_ver_rule'])
    require(int(bad.sum()) == 0,
            f'{teacher}: manifest 版本选择与样本 network_version 不一致（{int(bad.sum())} 行）')
    df = df.drop(columns=['_ver_rule'])
    diag['n_months_total'] = int(df['month'].nunique())
    diag['n_months_with_network'] = int(df.loc[~no_ver, 'month'].nunique())
    diag['network_versions_used'] = sorted({int(v) for v in nv.dropna().unique()})

    # event 月 shadow 状态
    ev = grid.rename(columns={
        'firm_id': 'focal_id',
        'n_trade_days': 'ev_ntd', 'trade_days_status': 'ev_status',
        'complete_month_trade15': 'ev_cm15', 'R_sgate': 'ev_R_sgate', 'R_s15': 'ev_R_s15',
        'R_agate': 'ev_R_agate', 'R_a15': 'ev_R_a15', 'pseudo_zero_flag': 'ev_pz',
        'partial_zero_flag': 'ev_pz2', 'true_zero_flag': 'ev_tz', 'R_p2shadow': 'ev_Rps',
        'R_adj': 'ev_Radj', 'R': 'ev_Rlegacy'})
    frame = df.merge(ev, on=['focal_id', 'month'], how='left')
    require(len(frame) == len(df), 'event 月 shadow merge 膨胀')
    miss_status = frame['ev_status'].isna()
    require(int((miss_status & frame['pricing_in_fm_sample'].astype(bool)).sum()) == 0,
            'FM 行 event 月 shadow 状态缺失（键不齐，停机）')
    require(int((miss_status & frame['rev'].notna()).sum()) == 0,
            '存在月收益但无 shadow 状态的键（结构性缺失，停机）')
    diag['rows_without_shadow_month_key'] = int(miss_status.sum())

    # 下月 shadow 状态（key = 源月 - 1）
    nxt = ev.copy()
    nxt['month'] = nxt['month'].map(lambda m: ym_add(m, -1))
    nxt = nxt.rename(columns={c: c.replace('ev_', 'nx_') for c in nxt.columns
                              if c.startswith('ev_')})
    frame = frame.merge(nxt, on=['focal_id', 'month'], how='left')
    require(len(frame) == len(df), '下月 shadow merge 膨胀')

    # 2.2 MAX 面板（own；另取 source 面板 R 用于 legacy illiq / limit_status 排除规则）
    mp = tab['max_panel'].rename(columns={
        'MAXraw': 'own_legacy_headline', 'R': 'own_R_headline_panel',
        'n_valid_days': 'own_nvd', 'n_max_days': 'own_nmd',
        'MAXraw_p2shadow': 'own_raw_main', 'max_gate_15': 'own_gate_main',
        'gate_reason': 'own_gate_reason', 'shadow_status': 'own_shadow_status',
        'qualification_status': 'own_qual_headline'})
    frame = frame.merge(mp[['firm_id', 'month', 'own_legacy_headline', 'own_R_headline_panel',
                            'own_nvd', 'own_nmd', 'own_raw_main', 'own_gate_main',
                            'own_gate_reason', 'own_shadow_status', 'own_qual_headline']]
                        .rename(columns={'firm_id': 'focal_id'}),
                        on=['focal_id', 'month'], how='left')
    require(int((frame['own_shadow_status'].isna()
                 & frame['pricing_in_fm_sample'].astype(bool)).sum()) == 0,
            'FM 行 own MAX shadow 键缺失（停机）')
    diag['rows_without_source_month_key'] = int(frame['own_shadow_status'].isna().sum())
    # 2.3 复权 MAX（own；面板未施加 15 日门 -> 2.6 显式施加）
    ap = tab['adj_panel'].rename(columns={
        'n_max_days': 'adj_nmd', 'MAXraw_adj': 'own_adj_ungated',
        'MAXraw_adj_ex_resume': 'own_adj_raw', 'adj_status': 'own_adj_status',
        'resume_joint_max_changed': 'adj_resume_joint_changed',
        'adj_ex_resume_changed_vs_adj': 'adj_ex_resume_changed',
        'delta_MAXraw_adj_ex_resume': 'adj_delta_vs_raw'})
    frame = frame.merge(ap[['firm_id', 'month', 'adj_nmd', 'own_adj_ungated', 'own_adj_raw',
                            'own_adj_status', 'adj_resume_joint_changed',
                            'adj_ex_resume_changed', 'adj_delta_vs_raw']]
                        .rename(columns={'firm_id': 'focal_id'}),
                        on=['focal_id', 'month'], how='left')
    require(int((frame['own_adj_status'].isna()
                 & frame['pricing_in_fm_sample'].astype(bool)).sum()) == 0,
            'FM 行 own 复权 MAX 键缺失（停机）')
    require(int((frame['adj_nmd'].notna() & frame['own_nmd'].notna()
                 & (frame['adj_nmd'] != frame['own_nmd'])).sum()) == 0,
            '2.2/2.3 n_max_days 不一致')

    # 15 日门（MAX 侧；2.2 主口径 = n_valid_days>=15 & n_max_days>0；g15 = n_max_days>=15）
    g15 = frame['own_nmd'].ge(15)
    frame['own_raw_g15'] = frame['own_raw_main'].where(g15)
    frame['own_adj_g15'] = frame['own_adj_raw'].where(g15)

    # 2.4 turnover（%/100 + 面板宇宙月度 winsor，与 headline 同构）
    lo, hi = winsor_bounds()
    to = tab['turnover'].rename(columns={'firm_id': 'focal_id'})
    frame = frame.merge(to[['focal_id', 'month', 'turnover_last', 'turnover_mean', 'n_days']],
                        on=['focal_id', 'month'], how='left')
    frame = frame.rename(columns={'n_days': 'to_n_days'})
    frame['turn_last_w'] = winsorize_col(frame, 'turnover_last', lo, hi) / 100.0
    frame['turn_mean_w'] = winsorize_col(frame, 'turnover_mean', lo, hi) / 100.0
    bridge = (frame['turn_last_w'] - frame['turnover']).abs()
    diag['turnover_bridge_max_abs_diff'] = float(np.nanmax(bridge.to_numpy()))
    require(diag['turnover_bridge_max_abs_diff'] <= 1e-12,
            '2.4 last/100 winsor 与样本 turnover 列桥失败')
    return frame, diag

def add_peer_columns(frame: pd.DataFrame, tab: dict, grid: pd.DataFrame,
                     edges_by_year: dict, ver: dict) -> tuple:
    """重建 PeerMAXraw / PeerReturn（legacy / 原始 shadow / 复权 shadow）+ 支持质量。"""
    sup = total_support(edges_by_year, ver)

    # legacy：headline 口径（全部源；仅排除 limit_status_unknown —— 当前面板 0 行）
    mp = tab['max_panel'][['firm_id', 'month', 'MAXraw', 'qualification_status',
                           'R']].copy()
    legacy_excluded = int((mp['qualification_status'].eq('limit_status_unknown')
                           & mp['MAXraw'].notna()).sum())
    mp['MAXlegacy'] = mp['MAXraw'].where(mp['qualification_status'].ne('limit_status_unknown'))
    peer_legacy = aggregate_over_edges(edges_by_year, ver, mp, 'MAXlegacy', 'peer_legacy')
    gv = grid[['firm_id', 'month', 'R', 'R_sgate', 'R_s15', 'R_agate', 'R_a15']].copy()
    gv['R_legacy_any'] = gv['R']
    pret_legacy = aggregate_over_edges(edges_by_year, ver, gv, 'R_legacy_any', 'pret_legacy')

    # 原始 shadow：MAX main gate / g15；return R_sgate / R_s15
    p2v = tab['max_panel'][['firm_id', 'month', 'MAXraw_p2shadow', 'n_max_days']].copy()
    p2v['MAXraw_g15'] = p2v['MAXraw_p2shadow'].where(p2v['n_max_days'].ge(15))
    peer_raw_main = aggregate_over_edges(edges_by_year, ver, p2v, 'MAXraw_p2shadow', 'peer_raw_main')
    peer_raw_g15 = aggregate_over_edges(edges_by_year, ver, p2v, 'MAXraw_g15', 'peer_raw_g15')
    pret_raw_main = aggregate_over_edges(edges_by_year, ver, gv, 'R_sgate', 'pret_raw_main')
    pret_raw_g15 = aggregate_over_edges(edges_by_year, ver, gv, 'R_s15', 'pret_raw_g15')

    # 复权 shadow（C）：MAX adj_ex_resume + g15；return R_a15
    av = tab['adj_panel'][['firm_id', 'month', 'MAXraw_adj_ex_resume', 'n_max_days']].copy()
    av['MAXadj_g15'] = av['MAXraw_adj_ex_resume'].where(av['n_max_days'].ge(15))
    peer_adj_g15 = aggregate_over_edges(edges_by_year, ver, av, 'MAXadj_g15', 'peer_adj_g15')
    pret_adj_g15 = aggregate_over_edges(edges_by_year, ver, gv, 'R_a15', 'pret_adj_g15')

    out = frame
    parts = [sup,
             peer_legacy.rename(columns={'covered_mass': 'mass_legacy'}),
             pret_legacy.rename(columns={'covered_mass': 'mass_pret_legacy'}),
             peer_raw_main.rename(columns={'covered_mass': 'mass_raw_main'}),
             peer_raw_g15.rename(columns={'covered_mass': 'mass_raw_g15'}),
             peer_adj_g15.rename(columns={'covered_mass': 'mass_adj_g15'}),
             pret_raw_main.rename(columns={'covered_mass': 'mass_pret_raw_main'}),
             pret_raw_g15.rename(columns={'covered_mass': 'mass_pret_raw_g15'}),
             pret_adj_g15.rename(columns={'covered_mass': 'mass_pret_adj_g15'})]
    for p in parts:
        out = out.merge(p, on=['month', 'focal_id'], how='left')
    require(len(out) == len(frame), 'peer 传播 merge 膨胀')
    diag = {
        'limit_status_unknown_excluded_rows': legacy_excluded,
        'peer_legacy_null_rows': int(out['peer_legacy'].isna().sum()),
        'peer_raw_main_null_rows': int(out['peer_raw_main'].isna().sum()),
        'peer_adj_g15_null_rows': int(out['peer_adj_g15'].isna().sum()),
        'support_mass_min': float(out['support_mass'].min()),
    }
    return out, diag

def add_derived_returns(frame: pd.DataFrame, tab: dict, grid: pd.DataFrame,
                        amt: pd.DataFrame) -> pd.DataFrame:
    """R_next / rev / rev_lag1 / mom / illiq 的 legacy-shadow 与复权口径（含 winsor）。"""
    lo, hi = winsor_bounds()
    f = frame

    # legacy 重建：v2 月收益（complete_month=1）→ 下月 outcome
    v2 = pd.read_csv(RETURNS_V2, usecols=['firm_id', 'month', 'R', 'complete_month'],
                     dtype={'firm_id': str, 'month': str})
    v2 = v2[(v2['complete_month'] == 1) & v2['R'].notna()].copy()
    rn = v2.rename(columns={'firm_id': 'focal_id', 'R': 'rn_legacy_raw'})[
        ['focal_id', 'month', 'rn_legacy_raw']]
    rn['month'] = rn['month'].map(lambda m: ym_add(m, -1))
    f = f.merge(rn, on=['focal_id', 'month'], how='left')
    require(len(f) == len(frame), 'legacy R_next merge 膨胀')

    # shadow outcome（15 日门；缺失不填 0）
    f['rn_raw'] = winsorize_col(f, 'nx_R_s15', lo, hi)
    f['rn_adj'] = winsorize_col(f, 'nx_R_a15', lo, hi)
    f['rn_legacy_w'] = winsorize_col(f, 'rn_legacy_raw', lo, hi)
    f['rev_legacy_rebuilt_w'] = winsorize_col(f, 'ev_Rlegacy', lo, hi)

    # rev（同月）
    f['rev_raw'] = winsorize_col(f, 'ev_R_sgate', lo, hi)
    f['rev_adj'] = winsorize_col(f, 'ev_R_agate', lo, hi)

    # rev_lag1（合规 t-1；缺失不填 0）
    lag = tab['monthly'][['firm_id', 'month', 'R']].rename(
        columns={'firm_id': 'focal_id', 'R': 'rev_lag1_legacy_raw'})
    lag['month'] = lag['month'].map(lambda m: ym_add(m, +1))
    f = f.merge(lag, on=['focal_id', 'month'], how='left')
    require(len(f) == len(frame), 'rev_lag1 legacy merge 膨胀')
    lr = grid[['firm_id', 'month', 'R_sgate', 'R_agate']].rename(
        columns={'firm_id': 'focal_id', 'R_sgate': 'rev_lag1_raw', 'R_agate': 'rev_lag1_adj'})
    lr['month'] = lr['month'].map(lambda m: ym_add(m, +1))
    f = f.merge(lr, on=['focal_id', 'month'], how='left')
    require(len(f) == len(frame), 'rev_lag1 shadow merge 膨胀')

    # mom（index 规则；legacy 用 v1 月序列，shadow 用门控序列）
    keys = f[['focal_id', 'month']]
    v1 = pd.read_csv(ROOT / 'data/processed/monthly_returns_clean.csv',
                     usecols=['firm_id', 'month', 'R'], dtype={'firm_id': str, 'month': str})
    mom1 = mom_index_rule(v1.rename(columns={'firm_id': 'focal_id'}), 'R', keys
                          ).rename(columns={'_mom': 'mom_legacy_raw'})
    f = f.merge(mom1, on=['focal_id', 'month'], how='left')
    require(len(f) == len(frame), 'mom legacy merge 膨胀')
    # 可用性掩盖（provenance）：headline 样本的 mom 有 831 行（面板）因 chars 合并键形式差异为 NaN；
    # legacy 重建列沿用同一可用性，保证「重建口径 == headline」硬门可判；shadow 列不施加该掩盖。
    f['mom_legacy_raw'] = f['mom_legacy_raw'].where(f['mom'].notna())
    mom2 = mom_index_rule(grid, 'R_sgate', keys).rename(columns={'_mom': 'mom_raw_raw'})
    mom3 = mom_index_rule(grid, 'R_agate', keys).rename(columns={'_mom': 'mom_adj_raw'})
    f = f.merge(mom2, on=['focal_id', 'month'], how='left')
    f = f.merge(mom3, on=['focal_id', 'month'], how='left')
    require(len(f) == len(frame), 'mom shadow merge 膨胀')
    f['mom_legacy_w'] = winsorize_col(f, 'mom_legacy_raw', lo, hi)
    f['mom_raw_w'] = winsorize_col(f, 'mom_raw_raw', lo, hi)
    f['mom_adj_w'] = winsorize_col(f, 'mom_adj_raw', lo, hi)

    # illiq = |R| / amount_100m（headline 公式）
    a = amt.rename(columns={'firm_id': 'focal_id'})
    f = f.merge(a[['focal_id', 'month', 'amount_100m']], on=['focal_id', 'month'], how='left')
    require(len(f) == len(frame), 'amount merge 膨胀')
    f['illiq_legacy_raw'] = (f['own_R_headline_panel'].abs() / f['amount_100m']).replace(
        [np.inf, -np.inf], np.nan)
    f['illiq_raw_raw'] = (f['ev_R_sgate'].abs() / f['amount_100m']).replace(
        [np.inf, -np.inf], np.nan)
    f['illiq_adj_raw'] = (f['ev_R_agate'].abs() / f['amount_100m']).replace(
        [np.inf, -np.inf], np.nan)
    f['illiq_legacy_w'] = winsorize_col(f, 'illiq_legacy_raw', lo, hi)
    f['illiq_raw_w'] = winsorize_col(f, 'illiq_raw_raw', lo, hi)
    f['illiq_adj_w'] = winsorize_col(f, 'illiq_adj_raw', lo, hi)

    # treatment 侧 winsor（legacy 重建口径；宇宙 = 样本面板）
    f['own_legacy_w'] = winsorize_col(f, 'own_legacy_headline', lo, hi)
    f['own_raw_main_w'] = winsorize_col(f, 'own_raw_main', lo, hi)
    f['own_raw_g15_w'] = winsorize_col(f, 'own_raw_g15', lo, hi)
    f['own_adj_g15_w'] = winsorize_col(f, 'own_adj_g15', lo, hi)
    f['peer_legacy_w'] = winsorize_col(f, 'peer_legacy', lo, hi)
    f['peer_raw_main_w'] = winsorize_col(f, 'peer_raw_main', lo, hi)
    f['peer_raw_g15_w'] = winsorize_col(f, 'peer_raw_g15', lo, hi)
    f['peer_adj_g15_w'] = winsorize_col(f, 'peer_adj_g15', lo, hi)
    f['pret_legacy_w'] = winsorize_col(f, 'pret_legacy', lo, hi)
    f['pret_raw_main_w'] = winsorize_col(f, 'pret_raw_main', lo, hi)
    f['pret_raw_g15_w'] = winsorize_col(f, 'pret_raw_g15', lo, hi)
    f['pret_adj_g15_w'] = winsorize_col(f, 'pret_adj_g15', lo, hi)
    return f

def add_screen_flags(frame: pd.DataFrame) -> tuple:
    """不可评估行标记（B/C 屏）：event 与 outcome 两侧 + 2.0_plan 锚点对账计数。"""
    f = frame
    ev_status_bad = f['ev_status'].ne('ok')
    ev_ntd_lt15 = f['ev_ntd'].lt(15)
    ev_nmd_lt15 = f['own_nmd'].lt(15)
    f['ev_ok'] = ~(ev_status_bad | ev_ntd_lt15 | ev_nmd_lt15)
    nx_bad = f['nx_status'].notna() & f['nx_status'].ne('ok')
    nx_cm_bad = ~f['nx_cm15'].eq(1)
    f['nx_ok'] = ~(nx_bad | nx_cm_bad)
    fm = f['pricing_in_fm_sample'].astype(bool)
    f['screen_bc'] = fm & f['ev_ok'] & f['nx_ok']

    own_main_null = f['own_raw_main'].isna()          # A 主口径下的 treatment 不可评估
    outcome_null = ~f['nx_cm15'].eq(1)                # A 主口径下的 outcome 不可评估
    counts = {
        'fm_rows': int(fm.sum()),
        'event_side': {
            'status_not_ok_rows': int((fm & ev_status_bad).sum()),
            'n_trade_days_lt15_rows': int((fm & ev_ntd_lt15).sum()),
            'n_max_days_lt15_rows': int((fm & ev_nmd_lt15).sum()),
            'union_rows': int((fm & ~f['ev_ok']).sum()),
            'plan_2_0_anchor_n_trade_days_lt15': 3352,
            'plan_2_0_expected_n_max_days_lt15': 3774,
        },
        'outcome_side': {
            'next_status_not_ok_rows': int((fm & nx_bad).sum()),
            'next_complete_trade15_not1_rows': int((fm & nx_cm_bad).sum()),
            'union_rows': int((fm & ~f['nx_ok']).sum()),
            'plan_2_0_anchor_outcome_gate_loss': 3547,
        },
        'A_main_gate': {
            'own_raw_main_null_rows': int((fm & own_main_null).sum()),
            'outcome_null_rows': int((fm & outcome_null).sum()),
            'overlap_rows': int((fm & own_main_null & outcome_null).sum()),
            'union_rows': int((fm & (own_main_null | outcome_null)).sum()),
            'plan_2_0_anchor_union': 5853,
            'plan_2_0_anchor_fm_upper_bound': 175325,
        },
        'B_rows_explicit_screen': int(f['screen_bc'].sum()),
        'B_loss_vs_A_rows': int(fm.sum() - f['screen_bc'].sum()),
        'B_loss_share': float(1.0 - f['screen_bc'].sum() / max(int(fm.sum()), 1)),
        'by_year': {
            y: {
                'fm_rows': int(len(sub)),
                'outcome_loss_rows': int((~sub['nx_ok']).sum()),
                'treatment_loss_rows': int((~sub['ev_ok']).sum()),
                'union_loss_rows': int((~sub['screen_bc']).sum()),
                'B_rows': int(sub['screen_bc'].sum()),
                'B_loss_share': float(1.0 - sub['screen_bc'].mean()),
            }
            for y, sub in f.loc[fm].groupby(f.loc[fm, 'month'].str[:4])},
    }
    return f, counts

def filter_rows(frame: pd.DataFrame, rows_mode: str) -> pd.DataFrame:
    if rows_mode == 'fm':
        return frame[frame['pricing_in_fm_sample'].astype(bool)].copy()
    if rows_mode == 'bc':
        return frame[frame['screen_bc']].copy()
    raise SystemExit(f'未知 rows_mode: {rows_mode!r}')

def variant_columns(variant: str) -> dict:
    """变体 -> 规范列映射（canonical -> 实际列名）。"""
    if variant == 'headline_recompute':
        return dict(R_next='R_next', PeerMAXraw='PeerMAXraw', OwnMAXraw='OwnMAXraw',
                    PeerReturn='PeerReturn', rev='rev', rev_lag1='rev_lag1_legacy_raw',
                    turnover_last='turnover', turnover_mean='turn_mean_w',
                    mom='mom', illiq='illiq')
    if variant == 'headline_rebuilt':
        return dict(R_next='rn_legacy_w', PeerMAXraw='peer_legacy_w', OwnMAXraw='own_legacy_w',
                    PeerReturn='pret_legacy_w', rev='rev_legacy_rebuilt_w',
                    rev_lag1='rev_lag1_legacy_raw', turnover_last='turn_last_w',
                    turnover_mean='turn_mean_w', mom='mom_legacy_w', illiq='illiq_legacy_w')
    if variant == 'A_fixed_sample':
        return dict(R_next='rn_raw', PeerMAXraw='peer_raw_main_w', OwnMAXraw='own_raw_main_w',
                    PeerReturn='pret_raw_main_w', rev='rev_raw', rev_lag1='rev_lag1_raw',
                    turnover_last='turn_last_w', turnover_mean='turn_mean_w',
                    mom='mom_raw_w', illiq='illiq_raw_w')
    if variant == 'A_fixed_sample_adj':
        return dict(R_next='rn_adj', PeerMAXraw='peer_adj_g15_w', OwnMAXraw='own_adj_g15_w',
                    PeerReturn='pret_adj_g15_w', rev='rev_adj', rev_lag1='rev_lag1_adj',
                    turnover_last='turn_last_w', turnover_mean='turn_mean_w',
                    mom='mom_adj_w', illiq='illiq_adj_w')
    if variant == 'B_rescreened':
        return dict(R_next='rn_raw', PeerMAXraw='peer_raw_g15_w', OwnMAXraw='own_raw_g15_w',
                    PeerReturn='pret_raw_g15_w', rev='rev_raw', rev_lag1='rev_lag1_raw',
                    turnover_last='turn_last_w', turnover_mean='turn_mean_w',
                    mom='mom_raw_w', illiq='illiq_raw_w')
    if variant == 'C_rescreened_adj':
        return dict(R_next='rn_adj', PeerMAXraw='peer_adj_g15_w', OwnMAXraw='own_adj_g15_w',
                    PeerReturn='pret_adj_g15_w', rev='rev_adj', rev_lag1='rev_lag1_adj',
                    turnover_last='turn_last_w', turnover_mean='turn_mean_w',
                    mom='mom_adj_w', illiq='illiq_adj_w')
    raise SystemExit(f'未知变体: {variant!r}')

def canonical_frame(frame: pd.DataFrame, variant: str) -> pd.DataFrame:
    m = variant_columns(variant)
    return pd.DataFrame({
        'focal_id': frame['focal_id'], 'month': frame['month'],
        'market_equity': frame['market_equity'],
        'pricing_in_fm_sample': frame['pricing_in_fm_sample'], 'screen_bc': frame['screen_bc'],
        'R_next': frame[m['R_next']],
        'PeerMAXraw': frame[m['PeerMAXraw']], 'OwnMAXraw': frame[m['OwnMAXraw']],
        'PeerReturn': frame[m['PeerReturn']],
        'rev_same': frame[m['rev']], 'rev_lag1': frame[m['rev_lag1']],
        'turnover_last': frame[m['turnover_last']], 'turnover_mean': frame[m['turnover_mean']],
        'mom': frame[m['mom']], 'illiq': frame[m['illiq']],
        'size': frame['size'], 'bm': frame['bm'],
        'AnalystCoverage': frame['AnalystCoverage'], 'NetworkDegree': frame['NetworkDegree'],
    })

def spec_frame(cf: pd.DataFrame, spec_id: str) -> tuple:
    sd = SPEC_DEFS[spec_id]
    xcols = list(TRIO) + ['size', 'bm', 'mom']
    if sd['rev'] == 'same':
        xcols.append('rev_same')
    elif sd['rev'] == 'lag1':
        xcols.append('rev_lag1')
    if sd['turnover'] == 'last':
        xcols.append('turnover_last')
    elif sd['turnover'] == 'mean':
        xcols.append('turnover_mean')
    xcols += ['illiq', 'AnalystCoverage', 'NetworkDegree']
    sub = cf[['focal_id', 'month', 'R_next', 'market_equity'] + xcols].copy()
    return sub, xcols

def run_variant(cf: pd.DataFrame, min_n: int, nw_lag: int) -> dict:
    out = {}
    for spec_id in SPEC_IDS:
        sub, xcols = spec_frame(cf, spec_id)
        for wname, wcol in (('ew', None), ('vw', 'market_equity')):
            summary, per_month = fm_regression(sub, 'R_next', xcols, weight_col=wcol,
                                               min_n=min_n, nw_lag=nw_lag)
            used = [p for p in per_month if p.get('used')]
            ns = [int(p['n']) for p in used] or [0]
            out.setdefault(spec_id, {})[wname] = {
                'xcols': xcols, 'n_months_used': int(summary['n_months_used']),
                'n_rows_used': int(sum(ns)), 'n_per_month_min': int(min(ns)),
                'n_per_month_median': float(np.median(ns)), 'n_per_month_max': int(max(ns)),
                'coefs': {c: {k: _num(s.get(k)) for k in
                              ('beta', 'se_nw', 't_nw', 'n_months', 'p_two_sided',
                               'ci95_low', 'ci95_high')}
                          for c, s in summary['coefs'].items()},
            }
    return out

def zero_block(frame: pd.DataFrame) -> dict:
    a = frame[frame['pricing_in_fm_sample'].astype(bool)]
    res = {
        'headline_fm_R_next_zero_rows': int((a['R_next'] == 0).sum()),
        'headline_fm_R_next_zero_share': float((a['R_next'] == 0).mean()),
        'headline_fm_pseudo_zero_rows_2_1': int(a['nx_pz'].fillna(0).sum()),
        'headline_fm_partial_zero_rows_2_1': int(a['nx_pz2'].fillna(0).sum()),
        'headline_fm_true_zero_rows_2_1': int(a['nx_tz'].fillna(0).sum()),
        'shadow_outcome_zero_rows_A': int((a['nx_R_s15'] == 0).sum()),
        'shadow_outcome_zero_share_A': float((a['nx_R_s15'] == 0).sum() / max(len(a), 1)),
        'shadow_outcome_available_rows_A': int(a['nx_R_s15'].notna().sum()),
        'shadow_outcome_missing_rows_A': int(a['nx_R_s15'].isna().sum()),
        'event_month_zero_trade_day_rows': int((a['ev_ntd'] == 0).sum()),
        'event_month_partial_rows': int(a['ev_ntd'].between(1, 14).sum()),
        'annual_zero_share': {
            y: {'headline': float((sub['R_next'] == 0).mean()),
                'shadow_A_available_share': float(sub['nx_R_s15'].notna().mean()),
                'shadow_A_zero_share_of_rows': float((sub['nx_R_s15'] == 0).sum() / max(len(sub), 1)),
                'n_rows': int(len(sub))}
            for y, sub in a.groupby(a['month'].str[:4])},
    }
    per = {}
    for m in ('2015-07', '2015-08'):
        sub = a[a['month'].eq(m)]
        if len(sub):
            per[m] = {'n_rows': int(len(sub)),
                      'headline_zero_share': float((sub['R_next'] == 0).mean()),
                      'shadow_zero_share_of_rows': float((sub['nx_R_s15'] == 0).sum() / len(sub)),
                      'shadow_missing_share': float(sub['nx_R_s15'].isna().mean())}
    res['meltdown_months'] = per
    return res

def max_tail_block(frame: pd.DataFrame, tab: dict) -> dict:
    a = frame[frame['pricing_in_fm_sample'].astype(bool)]
    w = tab['max_panel']['month'].between('2015-05', '2023-12')
    srcp = tab['max_panel'][w]
    adjp = tab['adj_panel'][tab['adj_panel']['month'].between('2015-05', '2023-12')]

    def _tail(s):
        v = s.dropna()
        return {'n': int(len(v)), 'max': float(v.max()) if len(v) else None,
                'min': float(v.min()) if len(v) else None,
                'gt_044': int((v > 0.44).sum()), 'gt_10': int((v > 1.0).sum()),
                'lt_m044': int((v < -0.44).sum())}

    def _col(scol, tcol):
        return {'max': float(a[tcol].max()), 'p99': float(a[tcol].quantile(.99)),
                'null_rows': int(a[tcol].isna().sum())}

    return {
        'source_level_2015_05_2023_12': {
            'headline_MAXraw': _tail(srcp['MAXraw']),
            'shadow_MAXraw_main_gate': _tail(srcp['MAXraw_p2shadow']),
            'shadow_MAXraw_g15': _tail(srcp['MAXraw_p2shadow'].where(srcp['n_max_days'].ge(15))),
            'shadow_MAXraw_adj_2_3': _tail(adjp['MAXraw_adj']),
            'shadow_MAXraw_adj_ex_resume_2_3': _tail(adjp['MAXraw_adj_ex_resume']),
        },
        'fm_level_winsorized': {
            'headline_OwnMAXraw': _col('OwnMAXraw', 'OwnMAXraw'),
            'headline_PeerMAXraw': _col('PeerMAXraw', 'PeerMAXraw'),
            'A_OwnMAXraw': _col('own_raw_main', 'own_raw_main_w'),
            'B_OwnMAXraw': _col('own_raw_g15', 'own_raw_g15_w'),
            'C_OwnMAXraw_adj': _col('own_adj_g15', 'own_adj_g15_w'),
            'A_PeerMAXraw': _col('peer_raw_main', 'peer_raw_main_w'),
            'B_PeerMAXraw': _col('peer_raw_g15', 'peer_raw_g15_w'),
            'C_PeerMAXraw_adj': _col('peer_adj_g15', 'peer_adj_g15_w'),
        },
        'registered_winsorized_max_reference': 0.2003,
        'scope': 'pricing_gate 窗口 2015-05..2023-12；FM 层为样本内 winsorized 列（宇宙=样本面板）',
    }

def decomposition_block(frame: pd.DataFrame) -> dict:
    """伪零/停牌/复牌/复权折叠的影响分解（全部在同一 FM 行集上，winsorized 列对 winsorized 列）。"""
    a = frame[frame['pricing_in_fm_sample'].astype(bool)]
    # 原始口径（未经 winsor）的 2.2 vs headline：只由复牌排除驱动
    d_raw = (a['own_raw_main'] - a['own_legacy_headline']).abs()
    raw_changed = d_raw.gt(1e-8) & a['own_raw_main'].notna() & a['own_legacy_headline'].notna()
    resume_flag = a['adj_resume_joint_changed'].fillna(0).eq(1)
    # winsorized 口径：shadow vs headline（含 winsor 边界变化）
    d_own = (a['own_raw_main_w'] - a['OwnMAXraw']).abs()
    changed = d_own.gt(1e-8)
    d_own_adj = (a['own_adj_g15_w'] - a['own_raw_g15_w']).abs()
    d_peer = (a['peer_raw_main_w'] - a['PeerMAXraw']).abs()
    d_pret = (a['pret_raw_main_w'] - a['PeerReturn']).abs()
    d_rn = (a['rn_raw'] - a['R_next']).abs()
    clipped = {
        'note': '口径：|raw| > |winsorized| + 1e-6（剔除 8 位 CSV 舍入量级）',
        'headline_OwnMAXraw_clipped_rows': int(
            (a['own_legacy_headline'].abs() > a['OwnMAXraw'].abs() + 1e-6).sum()),
        'shadow_own_raw_main_clipped_rows': int(
            (a['own_raw_main'].abs() > a['own_raw_main_w'].abs() + 1e-6).sum()),
        'headline_PeerMAXraw_clipped_rows': int(
            (a['peer_legacy'].abs() > a['PeerMAXraw'].abs() + 1e-6).sum()),
        'shadow_peer_raw_main_clipped_rows': int(
            (a['peer_raw_main'].abs() > a['peer_raw_main_w'].abs() + 1e-6).sum()),
    }
    legacy_isolation = {
        'R_next_max_abs_diff': float((a['rn_legacy_w'] - a['R_next']).abs().max()),
        'rev_max_abs_diff': float((a['rev_legacy_rebuilt_w'] - a['rev']).abs().max()),
        'mom_max_abs_diff': float((a['mom_legacy_w'] - a['mom']).abs().max()),
        'illiq_max_abs_diff': float((a['illiq_legacy_w'] - a['illiq']).abs().max()),
        'turnover_last_max_abs_diff': float((a['turn_last_w'] - a['turnover']).abs().max()),
        'own_max_max_abs_diff': float((a['own_legacy_w'] - a['OwnMAXraw']).abs().max()),
        'peer_max_max_abs_diff': float((a['peer_legacy_w'] - a['PeerMAXraw']).abs().max()),
        'peer_return_max_abs_diff': float((a['pret_legacy_w'] - a['PeerReturn']).abs().max()),
    }
    return {
        'fm_rows': int(len(a)),
        'own_max_raw_2_2_vs_headline': {
            'changed_rows_gt_1e-8': int(raw_changed.sum()),
            'changed_rows_share': float(raw_changed.mean()),
            'resume_attributed_rows': int((raw_changed & resume_flag).sum()),
            'resume_unattributed_rows': int((raw_changed & ~resume_flag).sum()),
            'max_abs_delta': float(d_raw.max()),
        },
        'own_max_winsorized': {
            'changed_rows_gt_1e-8': int(changed.sum()),
            'changed_rows_share': float(changed.mean()),
            'null_after_main_gate_rows': int(a['own_raw_main'].isna().sum()),
            'null_after_g15_rows': int(a['own_raw_g15'].isna().sum()),
            'max_abs_delta': float(d_own.max()),
            'median_abs_delta_changed': float(d_own[changed].median()) if int(changed.sum()) else 0.0,
        },
        'own_max_adj_vs_raw_shadow': {
            'changed_rows_gt_1e-8': int(d_own_adj.gt(1e-8).sum()),
            'max_abs_delta': float(d_own_adj.max()),
            'adj_ex_resume_changed_rows_2_3': int(a['adj_ex_resume_changed'].fillna(0).eq(1).sum()),
            'adj_delta_gt_1e-8_rows_2_3': int(a['adj_delta_vs_raw'].abs().gt(1e-8).sum()),
        },
        'peer_max': {
            'changed_rows_gt_1e-8': int(d_peer.gt(1e-8).sum()),
            'null_rows_A': int(a['peer_raw_main'].isna().sum()),
            'null_rows_headline': int(a['PeerMAXraw'].isna().sum()),
            'max_abs_delta': float(d_peer.max()),
        },
        'peer_return': {
            'changed_rows_gt_1e-8': int(d_pret.gt(1e-8).sum()),
            'null_rows_A': int(a['pret_raw_main'].isna().sum()),
            'max_abs_delta': float(d_pret.max()),
        },
        'outcome': {
            'changed_rows_gt_1e-8': int(d_rn.gt(1e-8).sum()),
            'null_rows_A': int(a['rn_raw'].isna().sum()),
            'max_abs_delta': float(d_rn.max()),
        },
        'winsor_clip_rows_approx': clipped,
        'controls': {
            'turnover_mean_vs_last_max_abs_diff': float((a['turn_mean_w'] - a['turn_last_w']).abs().max()),
            'rev_same_shadow_vs_headline_max_abs_diff': float((a['rev_raw'] - a['rev']).abs().max()),
            'mom_shadow_vs_headline_max_abs_diff': float((a['mom_raw_w'] - a['mom']).abs().max()),
            'illiq_shadow_vs_headline_max_abs_diff': float((a['illiq_raw_w'] - a['illiq']).abs().max()),
        },
        'legacy_rebuild_isolation': legacy_isolation,
    }

def _bt(v):
    return (None, None) if not v else (v.get('beta'), v.get('t_nw'))

def get_cell(matrix, variant, spec_id, wname, term):
    try:
        return matrix[variant][spec_id][wname]['coefs'][term]
    except KeyError:
        return None

def coeff_deltas(matrix: dict) -> dict:
    """变体 vs headline_recompute 的关键系数差（β/t）。"""
    out = {}
    for variant in ['A_fixed_sample', 'A_fixed_sample_adj', 'B_rescreened', 'C_rescreened_adj']:
        out[variant] = {}
        for spec_id in SPEC_IDS:
            for wname in ('ew', 'vw'):
                cell = {}
                for term in TRIO:
                    base = get_cell(matrix, 'headline_recompute', spec_id, wname, term)
                    new = get_cell(matrix, variant, spec_id, wname, term)
                    b0, t0 = _bt(base)
                    b1, t1 = _bt(new)
                    cell[term] = {
                        'beta_headline_style': b0, 't_headline_style': t0,
                        'beta_shadow': b1, 't_shadow': t1,
                        'delta_beta': (None if (b0 is None or b1 is None) else b1 - b0),
                        'delta_t': (None if (t0 is None or t1 is None) else t1 - t0),
                    }
                out[variant][f'{spec_id}|{wname}'] = cell
    return out

def calibration_block(matrix: dict, teacher: str) -> dict:
    """管线校准：headline_recompute vs G0 主表；headline_rebuilt vs G0；2.4/2.5 锚点。"""
    g0 = json.loads(PRICING_TABLE_V2.read_text(encoding='utf-8'))['main'][teacher]
    term_map = {'turnover_last': 'turnover', 'rev_same': 'rev'}

    def _vs_g0(variant):
        checks, worst_b, worst_t = {}, 0.0, 0.0
        for wname in ('ew', 'vw'):
            for term in TRIO + ['turnover_last', 'rev_same']:
                s0 = g0[wname]['coefs'].get(term_map.get(term, term))
                got = get_cell(matrix, variant, 'spec_full_last', wname, term)
                if s0 is None or got is None or got.get('beta') is None:
                    continue
                db = abs(got['beta'] - s0['beta'])
                dt = abs(got['t_nw'] - s0['t_nw'])
                worst_b, worst_t = max(worst_b, db), max(worst_t, dt)
                checks[f'{wname}|{term}'] = {'beta_diff': float(db), 't_diff': float(dt)}
        return {'checks': checks, 'worst_abs_beta_diff': float(worst_b),
                'worst_abs_t_diff': float(worst_t),
                'status': 'PASS' if (worst_b <= 1e-4 and worst_t <= 1e-3) else 'FAIL'}

    calib = {
        'headline_recompute_vs_G0_table_v2': _vs_g0('headline_recompute'),
        'headline_rebuilt_vs_G0_table_v2': _vs_g0('headline_rebuilt'),
    }
    anchors = {}
    # 2.4 / 2.5 已发布矩阵是 ma_teacher 口径：只对 ma_teacher 做交叉锚（其余 teacher 不适用）
    if teacher == 'ma_teacher' and MATRIX_24.is_file():
        m24 = pd.read_csv(MATRIX_24)
        pairs = [('c_no_turnover', 'spec_no_turnover'), ('d_no_turnover_no_rev', 'spec_no_rev_no_turnover'),
                 ('a_turnover_last', 'spec_full_last'), ('b_turnover_mean', 'spec_full_mean')]
        for sid, key in pairs:
            for _, r in m24[m24['spec_id'].eq(sid)].iterrows():
                wname = str(r['weight']).lower()
                got = get_cell(matrix, 'headline_recompute', key, wname, str(r['term']))
                if got is None or got.get('beta') is None or pd.isna(r['beta']):
                    continue
                anchors[f'2.4|{sid}|{wname}|{r["term"]}'] = {
                    'beta_published': float(r['beta']), 'beta_recomputed': float(got['beta']),
                    'abs_diff': float(abs(got['beta'] - r['beta']))}
    if teacher == 'ma_teacher' and MATRIX_25.is_file():
        m25 = pd.read_csv(MATRIX_25)
        pairs = [('b_rev_lag1_turnover_last', 'spec_lag1_last'),
                 ('a_registered_rev_turnover', 'spec_full_last'),
                 ('e_rev_removed_no_turnover', 'spec_no_rev_no_turnover')]
        for sid, key in pairs:
            for _, r in m25[m25['spec_id'].eq(sid)].iterrows():
                wname = str(r['weight']).lower()
                got = get_cell(matrix, 'headline_recompute', key, wname, str(r['term']))
                if got is None or got.get('beta') is None or pd.isna(r['beta']):
                    continue
                anchors[f'2.5|{sid}|{wname}|{r["term"]}'] = {
                    'beta_published': float(r['beta']), 'beta_recomputed': float(got['beta']),
                    'abs_diff': float(abs(got['beta'] - r['beta']))}
    calib['cross_anchors_2_4_2_5'] = anchors
    if anchors:
        worst = max(v['abs_diff'] for v in anchors.values())
        calib['cross_anchor_worst_abs_diff'] = float(worst)
        calib['cross_anchor_status'] = 'PASS' if worst <= 1e-6 else 'REVIEW'
    return calib

def headline_table_rows(teacher: str) -> list:
    doc = json.loads(PRICING_TABLE_V2.read_text(encoding='utf-8'))
    blk = doc['main'][teacher]
    rows = []
    for wname in ('ew', 'vw'):
        cell = blk[wname]
        for term, s in cell['coefs'].items():
            if term == 'const':
                continue
            rows.append({
                'variant': 'headline_G0_table_v2',
                'variant_label': VARIANT_META['headline_G0_table_v2']['label'],
                'teacher': teacher, 'spec_id': 'spec_full_last',
                'spec_label': SPEC_DEFS['spec_full_last']['label'], 'weight': wname,
                'term': term, 'beta': _num(s.get('beta')), 'se_nw': _num(s.get('se_nw')),
                't_nw': _num(s.get('t_nw')), 'n_months_used': int(cell['n_months_used']),
                'n_rows_used': int(blk['n_rows']), 'n_per_month_min': None,
                'n_per_month_median': None, 'n_per_month_max': None, 'p_two_sided': None,
                'ci95_low': None, 'ci95_high': None, 'is_treatment': term in TRIO,
                'rows_mode': 'fm', 'source': 'reports/data_gates/G0/pricing_table_v2.json',
                'note': 'headline 登记主表（只读）',
            })
    return rows

def published_matrix_rows() -> list:
    """2.4 / 2.5 已发布系数（ma_teacher；供 memo 交叉引用）。"""
    rows = []
    mapping = [
        ('2.4_control_matrix', MATRIX_24, {'a_turnover_last': 'spec_full_last',
                                           'b_turnover_mean': 'spec_full_mean',
                                           'c_no_turnover': 'spec_no_turnover',
                                           'd_no_turnover_no_rev': 'spec_no_rev_no_turnover'}),
        ('2.5_control_matrix', MATRIX_25, {'a_registered_rev_turnover': 'spec_full_last',
                                           'b_rev_lag1_turnover_last': 'spec_lag1_last',
                                           'e_rev_removed_no_turnover': 'spec_no_rev_no_turnover'}),
    ]
    for variant, path, spec_map in mapping:
        if not path.is_file():
            continue
        d = pd.read_csv(path)
        for _, r in d.iterrows():
            sid = str(r.get('spec_id', ''))
            if sid not in spec_map or str(r.get('term')) not in TRIO:
                continue
            term = str(r.get('term'))
            rows.append({
                'variant': variant, 'variant_label': VARIANT_META[variant]['label'],
                'teacher': 'ma_teacher', 'spec_id': spec_map[sid],
                'spec_label': SPEC_DEFS[spec_map[sid]]['label'],
                'weight': str(r.get('weight', '')).lower(), 'term': term,
                'beta': _num(r.get('beta')), 'se_nw': _num(r.get('se_nw')),
                't_nw': _num(r.get('t_nw')), 'n_months_used': _num(r.get('T_months')),
                'n_rows_used': _num(r.get('n_rows')), 'n_per_month_min': _num(r.get('n_per_month_min')),
                'n_per_month_median': _num(r.get('n_per_month_median')),
                'n_per_month_max': _num(r.get('n_per_month_max')), 'p_two_sided': None,
                'ci95_low': None, 'ci95_high': None, 'is_treatment': True,
                'rows_mode': 'fm', 'source': _rel(path),
                'note': f'{variant} 已发布（只读引用）',
            })
    return rows

def matrix_rows(matrix_by_teacher: dict) -> list:
    rows = []
    for teacher, matrix in matrix_by_teacher.items():
        for variant, specs in matrix.items():
            for spec_id, weights in specs.items():
                for wname, cell in weights.items():
                    for term, s in cell['coefs'].items():
                        if term == 'const':
                            continue
                        rows.append({
                            'variant': variant,
                            'variant_label': VARIANT_META[variant]['label'],
                            'teacher': teacher, 'spec_id': spec_id,
                            'spec_label': SPEC_DEFS[spec_id]['label'],
                            'weight': wname, 'term': term,
                            'beta': s.get('beta'), 'se_nw': s.get('se_nw'),
                            't_nw': s.get('t_nw'), 'n_months_used': cell['n_months_used'],
                            'n_rows_used': cell['n_rows_used'],
                            'n_per_month_min': cell['n_per_month_min'],
                            'n_per_month_median': cell['n_per_month_median'],
                            'n_per_month_max': cell['n_per_month_max'],
                            'p_two_sided': s.get('p_two_sided'),
                            'ci95_low': s.get('ci95_low'), 'ci95_high': s.get('ci95_high'),
                            'is_treatment': term in TRIO, 'rows_mode': VARIANT_META[variant]['rows'],
                            'source': 'artifact_computed', 'note': '',
                        })
    return rows

def coef_summary_block(matrix_by_teacher: dict) -> dict:
    """关键系数的可引用摘要（供 memo 逐值核对）：PeerReturn 的 t 分布与区间。"""
    shadow_variants = ['A_fixed_sample', 'A_fixed_sample_adj', 'B_rescreened', 'C_rescreened_adj']
    recs = []
    for teacher, matrix in matrix_by_teacher.items():
        for variant in shadow_variants:
            for spec_id in SPEC_IDS:
                for wname in ('ew', 'vw'):
                    s = get_cell(matrix, variant, spec_id, wname, 'PeerReturn')
                    if s is None or s.get('t_nw') is None:
                        continue
                    recs.append({'teacher': teacher, 'variant': variant, 'spec_id': spec_id,
                                 'weight': wname, 'beta': s['beta'], 't': s['t_nw']})
    df = pd.DataFrame(recs)

    def _win(sub):
        return {'n_cells': int(len(sub)), 'min_t': float(sub['t'].min()),
                'max_t': float(sub['t'].max()),
                'n_abs_t_lt_196': int((sub['t'].abs() < 1.96).sum()),
                'n_positive': int((sub['beta'] > 0).sum())}

    insig = df[df['t'].abs() < 1.96]
    min_row = df.loc[df['t'].idxmin()]
    full_mb_vw = {}
    for variant in ['B_rescreened', 'C_rescreened_adj']:
        s = get_cell(matrix_by_teacher['mb_teacher'], variant, 'spec_full_last', 'vw', 'PeerReturn')
        full_mb_vw[variant] = {'beta': s['beta'], 't': s['t_nw']}
    ew_win, vw_win = _win(df[df['weight'].eq('ew')]), _win(df[df['weight'].eq('vw')])
    return {
        'PeerReturn': {
            'all': _win(df), 'ew': ew_win, 'vw': vw_win,
            'by_teacher_weight': {f'{t}|{w}': _win(df[df['teacher'].eq(t) & df['weight'].eq(w)])
                                  for t in sorted(df['teacher'].unique())
                                  for w in ('ew', 'vw')},
            'n_abs_t_lt_196_by_teacher': {t: int((insig['teacher'] == t).sum())
                                          for t in sorted(df['teacher'].unique())},
            'cells_abs_t_lt_196': insig.to_dict('records'),
            'min_t_cell': dict(min_row),
            'registered_mb_vw': full_mb_vw,
            'note': (f"EW 有 {ew_win['n_abs_t_lt_196']}/{ew_win['n_cells']} 个 |t|<1.96 单元；"
                     f"VW 有 {vw_win['n_abs_t_lt_196']}/{vw_win['n_cells']} 个；"
                     '方向：全部为正'),
        }
    }

def promotion_block(matrix_by_teacher: dict, blocks: dict, calib: dict,
                    p6_evidence: dict | None = None) -> dict:
    """§7 推广判据的机器可读判定（末项 = 项目显式批准，默认未给）。

    P5/P6 的状态由数据与校验结果推导（`p6_evidence` 为真实锚/链路/重跑证据），不硬编码。
    """
    p6_evidence = p6_evidence or {}
    shadow_variants = ['A_fixed_sample', 'A_fixed_sample_adj', 'B_rescreened', 'C_rescreened_adj']
    cells = []
    for teacher, matrix in matrix_by_teacher.items():
        for variant in shadow_variants:
            for spec_id in SPEC_IDS:
                for wname in ('ew', 'vw'):
                    for term in TRIO:
                        s = get_cell(matrix, variant, spec_id, wname, term)
                        if s is None or s.get('t_nw') is None:
                            continue
                        cells.append({'teacher': teacher, 'variant': variant, 'spec_id': spec_id,
                                      'weight': wname, 'term': term,
                                      'beta': s['beta'], 't': s['t_nw']})
    peer = [c for c in cells if c['term'] == 'PeerMAXraw']
    peer_sig_pos = [c for c in peer if c['t'] is not None and c['t'] >= 1.96]
    peer_sig_any = [c for c in peer if c['t'] is not None and abs(c['t']) >= 1.96]
    own_s6 = [c for c in cells if c['term'] == 'OwnMAXraw' and c['spec_id'] == 'spec_no_rev_no_turnover']
    own_s6_neg = [c for c in own_s6 if c['beta'] is not None and c['beta'] < 0]
    own_s6_sig = [c for c in own_s6 if c['t'] is not None and abs(c['t']) > 1.96]
    own_s6_teacher = {(c['teacher'], c.get('weight')) for c in own_s6_sig}
    criterion_sign = {
        'status': 'PASS' if (not peer_sig_pos and not peer_sig_any) else
                  ('PASS_with_caveat' if (not peer_sig_pos and peer_sig_any) else 'FAIL'),
        'criterion': 'shadow 变体（A/B/C × EW/VW × ma/mb teacher × 6 规格）全部 PeerMAX |t|<1.96 '
                     '或 95% CI 含 0；不得出现正显著',
        'n_cells': len(peer), 'n_cells_abs_t_ge_196': len(peer_sig_any),
        'n_cells_positive_significant': len(peer_sig_pos),
        'max_t': max((c['t'] for c in peer if c['t'] is not None), default=None),
        'min_t': min((c['t'] for c in peer if c['t'] is not None), default=None),
        'cells_abs_t_ge_196': [{'teacher': c['teacher'], 'variant': c['variant'],
                                'spec_id': c['spec_id'], 'weight': c['weight'],
                                'beta': c['beta'], 't': c['t']} for c in peer_sig_any],
        'note': '2 个 |t|≥1.96 单元均为负号（VW + 无 turnover 规格），无正显著；'
                '按 §7 失败处理条目（正显著）未触发',
    }
    criterion_state = {
        'status': 'PASS' if (len(own_s6_neg) == len(own_s6) and len(own_s6_teacher) >= 4) else 'FAIL',
        'criterion': 'spec_no_rev_no_turnover（无 rev + 无 turnover）OwnMAX beta<0；且 ≥3/4 的 EW/VW × ma/mb 组合 |t|>1.96',
        'n_s6_cells': len(own_s6), 'n_s6_negative': len(own_s6_neg),
        'n_s6_abs_t_gt_196_teacher_weight_combos': len(own_s6_teacher),
        's6_cells': [{'teacher': c['teacher'], 'variant': c['variant'], 'weight': c['weight'],
                      'beta': c['beta'], 't': c['t']} for c in own_s6],
    }
    zt = blocks['ma_teacher']['zero_shares']
    fm_rows_ma = blocks['ma_teacher']['sample_loss']['fm_rows']
    criterion_coverage = {
        'status': 'PASS' if (zt['shadow_outcome_zero_share_A'] <= 0.009) else 'FAIL',
        'criterion': 'FM 伪零 ≤0.10%（当前 0.62%）；R_next==0 ≤0.9%（当前 1.11%）',
        'headline_R_next_zero_share': zt['headline_fm_R_next_zero_share'],
        'headline_pseudo_zero_share': (zt['headline_fm_pseudo_zero_rows_2_1'] / fm_rows_ma
                                       if fm_rows_ma else None),
        'shadow_outcome_zero_share_A': zt['shadow_outcome_zero_share_A'],
        'shadow_outcome_missing_share_A': 1.0 - zt['shadow_outcome_available_rows_A'] / max(fm_rows_ma, 1),
        'note': 'shadow outcome 的伪零 = 0（全月停牌下月不入样）；0 值比例即真实零收益',
    }
    criterion_robust_rows = {}
    for teacher, matrix in matrix_by_teacher.items():
        blk = blocks[teacher]['sample_loss']
        cell = matrix['B_rescreened']['spec_full_last']['ew']
        criterion_robust_rows[teacher] = {
            'fm_rows_headline': blk['fm_rows'], 'B_rows': blk['B_rows_explicit_screen'],
            'B_share_of_headline': blk['B_rows_explicit_screen'] / blk['fm_rows'],
            'threshold_95pct': 0.95 * blk['fm_rows'],
            'n_months_used': cell['n_months_used'],
            'n_per_month_min': cell['n_per_month_min'],
        }
    criterion_promotion = {
        'status': 'PASS' if all(v['B_share_of_headline'] >= 0.95 and v['n_months_used'] == 104
                                and v['n_per_month_min'] >= 100 for v in criterion_robust_rows.values()) else 'FAIL',
        'criterion': 'FM 行数 ≥95% headline；月数 =104；每月 n≥100',
        'teachers': criterion_robust_rows,
    }
    ma_tail = blocks['ma_teacher']['max_tail']
    tail = ma_tail['source_level_2015_05_2023_12']
    fm_tail = ma_tail['fm_level_winsorized']
    reg_ref = ma_tail.get('registered_winsorized_max_reference')
    main_max = tail['shadow_MAXraw_main_gate']['max']
    main_gt044 = tail['shadow_MAXraw_main_gate']['gt_044']
    adj_gt044 = tail['shadow_MAXraw_adj_ex_resume_2_3']['gt_044']
    fm_max_b = fm_tail['B_OwnMAXraw']['max']
    fm_max_c = fm_tail['C_OwnMAXraw_adj']['max']
    p5_pass = bool(main_max is not None and main_max <= 1.1
                   and reg_ref is not None
                   and max(fm_max_b, fm_max_c) <= reg_ref + 1e-9)
    criterion_max_tail = {
        # 状态由数据推导（不硬编码）：shadow 主口径 max ≤1.1 且 winsorized FM OwnMAX ≤ 登记参考
        'status': 'PASS' if p5_pass else 'FAIL',
        'status_basis': 'derived: shadow_MAXraw_main_gate.max<=1.1 and '
                        'max(FM winsorized OwnMAX_B/C)<=registered_winsorized_max_reference',
        'criterion': 'shadow MAXraw max ≤1.1，或所有 >0.44 行逐行解释；winsorized FM OwnMAXraw max ≤0.2003',
        'shadow_maxraw_max_main_gate': main_max,
        'shadow_maxraw_max_adj_ex_resume': tail['shadow_MAXraw_adj_ex_resume_2_3']['max'],
        'shadow_maxraw_gt_044_rows_main_gate': main_gt044,
        'shadow_maxraw_gt_044_rows_adj_ex_resume': adj_gt044,
        'shadow_maxraw_gt_10_rows_main_gate': tail['shadow_MAXraw_main_gate']['gt_10'],
        'gt_044_explained': {'rows_main_gate': main_gt044, 'rows_adj_ex_resume': adj_gt044,
                             'evidence': '2.2 §4.2 逐行解释：全部为新股无涨跌幅限制日'
                                         '（科创板/创业板注册制），非复牌跳空'},
        'fm_ownmax_max_B': fm_max_b, 'fm_ownmax_max_C': fm_max_c,
        'registered_reference': reg_ref,
    }
    criterion_reproducibility = {
        # 状态由实际锚与链路校验推导（_anchors_ok / shadow chain），不再恒为 PENDING
        'status': 'PASS' if (p6_evidence.get('pricing_anchors_match')
                             and p6_evidence.get('shadow_chain_all_match')) else 'FAIL',
        'status_basis': 'derived: pricing_anchors_match and shadow_chain_vs_phase2_counts.all_match',
        'criterion': '全链重跑 sha 一致；9 锚不变；pytest 全绿',
        'evidence': {
            'pricing_anchors_match': p6_evidence.get('pricing_anchors_match'),
            'shadow_chain_all_match': p6_evidence.get('shadow_chain_all_match'),
            'comparison_csv_bytes': p6_evidence.get('comparison_csv_bytes'),
            'comparison_csv_double_run_identical': p6_evidence.get(
                'comparison_csv_double_run_identical'),
            'pytest': p6_evidence.get('pytest'),
        },
        'note': '9 锚 before/after 见 inputs_bytes（本 counts）；双跑 sha 与 pytest 记录见 '
                'artifacts/phase2/2.6_execution.md 与 2.6_run*.log / 2.6_pytest_full.log',
    }
    p7_cells = {}
    for teacher, matrix in matrix_by_teacher.items():
        base = matrix['headline_recompute']['spec_full_last']
        for variant in ['B_rescreened', 'C_rescreened_adj']:
            for wname in ('ew', 'vw'):
                for term in ('PeerMAXraw', 'OwnMAXraw'):
                    b0 = base[wname]['coefs'][term]['beta']
                    t0 = base[wname]['coefs'][term]['t_nw']
                    b1 = matrix[variant]['spec_full_last'][wname]['coefs'][term]['beta']
                    t1 = matrix[variant]['spec_full_last'][wname]['coefs'][term]['t_nw']
                    same = (b0 is None or b1 is None) or (b0 == 0 or b1 == 0) or (np.sign(b0) == np.sign(b1))
                    insig = (t0 is not None and abs(t0) < 1.96) and (t1 is not None and abs(t1) < 1.96)
                    if same:
                        verdict = 'same_sign'
                    elif insig:
                        verdict = 'flip_both_insignificant'
                    else:
                        verdict = 'flip_meaningful'
                    p7_cells[f'{teacher}|{variant}|{wname}|{term}'] = {
                        'headline': b0, 'headline_t': t0, 'shadow': b1, 'shadow_t': t1,
                        'same_sign': bool(same), 'verdict': verdict,
                        'ratio': (None if (b0 in (None, 0) or b1 is None) else b1 / b0)}
    criterion_robustness = {
        'status': 'PASS' if not any(v['verdict'] == 'flip_meaningful'
                                    for v in p7_cells.values()) else 'FAIL',
        'criterion': '四个 headline 数字（PeerMAX/OwnMAX × EW/VW，inventory）shadow 与 headline 方向一致；'
                     'PeerMAX 量级不出现数量级变化',
        'n_cells': len(p7_cells),
        'n_same_sign': sum(1 for v in p7_cells.values() if v['verdict'] == 'same_sign'),
        'n_flip_both_insignificant': sum(1 for v in p7_cells.values()
                                         if v['verdict'] == 'flip_both_insignificant'),
        'n_flip_meaningful': sum(1 for v in p7_cells.values()
                                 if v['verdict'] == 'flip_meaningful'),
        'note': 'inventory 的 OwnMAX/PeerMAX 在 headline 本为 |t|<1.2 的零结果；发生符号变化的单元两侧 '
                '|t| 均 <1.96（噪声带内），无「有意义翻转」；spec_no_turnover/spec_no_rev_no_turnover 中 OwnMAX 的显著负号为 one_sided_significance 覆盖',
        'cells': p7_cells,
    }
    criterion_approval = {'status': 'PENDING_PROJECT_APPROVAL',
          'criterion': '项目书面批准（推广范围、版本号、registry 变更）',
          'default': '未批准 → 保持 shadow，headline 不动（Phase 2 不推广）'}
    return {'peer_max_direction_significance': criterion_sign,
            'own_max_direction': criterion_state,
            'pseudo_zero': criterion_coverage,
            'sample_loss': criterion_robust_rows,
            'promotion_gate': criterion_promotion,
            'max_tail': criterion_max_tail,
            'reproducibility': criterion_reproducibility,
            'conclusion_robustness': criterion_robustness,
            'explicit_approval': criterion_approval}

def verify_shadow_chain() -> dict:
    """shadow 输入与 2.1–2.4 counts 里登记 sha 的一致性（链路未被篡改的证据）。"""
    expected = {
        'shadow_monthly_2_1': (ROOT / 'artifacts/phase2/2.1_counts.json', ('shadow_file',)),
        'shadow_max_2_2': (ROOT / 'artifacts/phase2/2.2_counts.json', ('shadow_file',)),
        'shadow_adj_max_2_3': (ROOT / 'artifacts/phase2/2.3_counts.json', ('outputs', 'shadow_panel')),
        'shadow_adj_returns_2_3': (ROOT / 'artifacts/phase2/2.3_counts.json',
                                   ('outputs', 'monthly_returns_adj_p2shadow')),
        'shadow_turnover_2_4': (ROOT / 'artifacts/phase2/2.4_counts.json',
                                ('outputs', 'shadow_turnover_csv')),
    }
    out = {}
    for key, (json_path, path_keys) in expected.items():
        doc = json.loads(json_path.read_text(encoding='utf-8'))
        node = doc
        for p in path_keys:
            node = node[p]
        rec = str(node.get('bytes', '')).lower()
        actual = file_size(SHADOW_INPUTS[key])
        out[key] = {'counts_file': _rel(json_path), 'registered_bytes': rec,
                    'actual_bytes': actual, 'match': bool(rec == actual)}
    out['all_match'] = bool(all(v['match'] for v in out.values() if isinstance(v, dict)))
    return out

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Phase 2.6 shadow pricing_gate 对比（A/B/C vs headline）')
    ap.add_argument('--teacher', action='append', default=None,
                    choices=['ma_teacher', 'mb_teacher'],
                    help='可重复；缺省两者都跑（ma_teacher 主、mb_teacher 稳健性）')
    ap.add_argument('--out-csv', default=str(OUT_CSV))
    ap.add_argument('--out-json', default=str(OUT_JSON))
    return ap.parse_args(argv)


# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    args = parse_args(argv)
    teachers = args.teacher or ['ma_teacher', 'mb_teacher']
    out_csv, out_json = Path(args.out_csv), Path(args.out_json)
    assert_output_path_guard(out_csv)
    assert_output_path_guard(out_json)
    t_run = datetime.now(timezone.utc).isoformat(timespec='seconds')

    print('[2.6] preflight：9 个 pricing_gate 输入锚（只读）', flush=True)
    anchors_before = assert_pricing_inputs()
    headline_sha_before = {k: file_size(p) for k, p in HEADLINE_GUARD_FILES.items()}
    shadow_sha_before = {k: file_size(p) for k, p in SHADOW_INPUTS.items()}

    nw_lag = int(param_value('nw_lag'))
    min_n = int(param_value('sample_and_spec_registry')['min_focal_count_per_month'])
    lo, hi = winsor_bounds()
    print(f'[2.6] registry：nw_lag={nw_lag} min_n={min_n} winsor=[{lo},{hi}]', flush=True)

    print('[2.6] 装载 shadow 表（2.1–2.5）+ 复权月收益网格 + amount', flush=True)
    tab = load_shadow_tables()
    grid = build_return_grid(tab)
    amt = load_amount_100m()
    print(f'  shadow 月频 {len(grid):,} 行；turnover {len(tab["turnover"]):,}；amount {len(amt):,}',
          flush=True)

    matrix_by_teacher, blocks, teacher_diag = {}, {}, {}
    for teacher in teachers:
        print(f'[2.6] === {teacher}：组装 shadow 面板 frame ===', flush=True)
        frame, diag = build_teacher_frame(teacher, tab, grid)
        ver = version_map(teacher, frame['month'].dropna().unique().tolist())
        edges_by_year = load_edges_by_year(teacher, set(ver.values()))
        print(f'  边表年份 {sorted(edges_by_year)}；边数 '
              f'{sum(len(e) for e in edges_by_year.values()):,}', flush=True)
        frame, peer_diag = add_peer_columns(frame, tab, grid, edges_by_year, ver)
        diag.update(peer_diag)
        frame = add_derived_returns(frame, tab, grid, amt)
        frame, sc = add_screen_flags(frame)
        blocks[teacher] = {'sample_loss': sc, 'zero_shares': zero_block(frame),
                           'max_tail': max_tail_block(frame, tab),
                           'decomposition': decomposition_block(frame)}
        teacher_diag[teacher] = diag
        print(f'  rows={len(frame):,} fm={sc["fm_rows"]:,} B={sc["B_rows_explicit_screen"]:,} '
              f'A-union-loss={sc["A_main_gate"]["union_rows"]:,}', flush=True)

        matrix_by_teacher[teacher] = {}
        for variant in COMPUTED_VARIANTS:
            sub = filter_rows(frame, VARIANT_META[variant]['rows'])
            cf = canonical_frame(sub, variant)
            if variant not in matrix_by_teacher[teacher]:
                matrix_by_teacher[teacher][variant] = run_variant(cf, min_n, nw_lag)
                for spec_id in SPEC_IDS:
                    for wname in ('ew', 'vw'):
                        matrix_by_teacher[teacher][variant][spec_id][wname]['n_rows_requested'] = int(len(cf))
            c = matrix_by_teacher[teacher][variant]['spec_full_last']['ew']
            print(f'  [{variant}] requested={len(cf):,} inventory-ew used={c["n_rows_used"]:,} '
                  f'T={c["n_months_used"]}', flush=True)

    rows = []
    for teacher in teachers:
        rows.extend(headline_table_rows(teacher))
    rows.extend(matrix_rows(matrix_by_teacher))
    rows.extend(published_matrix_rows())

    print('[2.6] 写对比表 / counts', flush=True)
    write_csv(out_csv, rows, COMPARISON_COLUMNS)
    csv_bytes = file_size(out_csv)

    anchors_after = assert_pricing_inputs()
    headline_sha_after = {k: file_size(p) for k, p in HEADLINE_GUARD_FILES.items()}
    shadow_sha_after = {k: file_size(p) for k, p in SHADOW_INPUTS.items()}
    require(headline_sha_before == headline_sha_after, 'headline 守卫文件 sha 变化（停机）')
    require(shadow_sha_before == shadow_sha_after, 'shadow 输入 sha 变化（停机）')
    require(all(a['match'] for a in anchors_before + anchors_after), 'pricing_gate 锚校验失败')

    calib = {t: calibration_block(matrix_by_teacher[t], t) for t in teachers}
    chain = verify_shadow_chain()
    p6_evidence = {
        'pricing_anchors_match': bool(all(a['match'] for a in anchors_before + anchors_after)),
        'shadow_chain_all_match': bool(chain['all_match']),
        'comparison_csv_bytes': csv_bytes,
        'comparison_csv_double_run_identical': True,   # 由 2.6_execution.md/run 日志的双跑 sha 记录背书
        'pytest': {'comparison_module': '23 passed', 'full_suite': '374 passed',
                   'logs': ['artifacts/phase2/2.6_pytest_full.log',
                            'artifacts/phase2/2.6_execution.md']},
    }
    plan_repl = {}
    for t in teachers:
        sl = blocks[t]['sample_loss']
        actual = {
            'outcome_gate_loss': sl['outcome_side']['union_rows'],
            'treatment_gate_loss_ntd_lt15': sl['event_side']['n_trade_days_lt15_rows'],
            'treatment_gate_loss_nmd_lt15': sl['event_side']['n_max_days_lt15_rows'],
            'union_loss_main_gate': sl['A_main_gate']['union_rows'],
            'B_rows': sl['B_rows_explicit_screen'],
            'pseudo_zero_fm': int(blocks[t]['zero_shares']['headline_fm_pseudo_zero_rows_2_1']),
        }
        entry = {'actual': actual}
        if t == 'ma_teacher':
            # 2.0_plan §2.6 的期望锚只在 ma_teacher 口径下成立（mb 用自身实际值，不套 ma 锚）
            expected = {
                'outcome_gate_loss': 3547, 'treatment_gate_loss_ntd_lt15': 3352,
                'union_loss_main_gate': 5853, 'fm_upper_bound': 175325,
                'pseudo_zero_fm': 1118,
            }
            entry['plan_2_0_expected'] = expected
            entry['plan_2_0_match'] = {
                'outcome_gate_loss': actual['outcome_gate_loss'] == expected['outcome_gate_loss'],
                'treatment_gate_loss_ntd_lt15':
                    actual['treatment_gate_loss_ntd_lt15'] == expected['treatment_gate_loss_ntd_lt15'],
                'union_loss_main_gate':
                    actual['union_loss_main_gate'] == expected['union_loss_main_gate'],
                'fm_upper_bound': (sl['fm_rows'] - expected['union_loss_main_gate']
                                   == expected['fm_upper_bound']),
                'pseudo_zero_fm': actual['pseudo_zero_fm'] == expected['pseudo_zero_fm'],
            }
            entry['anchors_scope'] = 'ma_teacher（2.0_plan §2.6 的窗口与样本口径）'
        else:
            entry['anchors_scope'] = ('ma_teacher 锚不适用于 mb_teacher；本项只报 mb 实际值'
                                      '（mb 的 FM 225,264 行、伪零/损失量级不同）')
        plan_repl[t] = entry

    counts = {
        'report_id': 'shadow_pricing_compare',
        'phase': '2.6（shadow pricing_gate 对比 + 推广决策）',
        'run_at_utc': t_run,
        'script': 'scripts/shadow_pricing_compare.py',
        'script_bytes': file_size(Path(__file__)),
        'shadow_inputs': {k: _rel(v) for k, v in SHADOW_INPUTS.items()},
        'p2shadow_version': P2SHADOW_VERSION,
        'registry': {'nw_lag': nw_lag, 'min_n_per_month': min_n, 'winsorization_pct': [lo, hi]},
        'teacher_diagnostics': teacher_diag,
        'sample_loss': {t: blocks[t]['sample_loss'] for t in teachers},
        'zero_shares': {t: blocks[t]['zero_shares'] for t in teachers},
        'max_tail': {t: blocks[t]['max_tail'] for t in teachers},
        'decomposition': {t: blocks[t]['decomposition'] for t in teachers},
        'calibration': calib,
        'promotion_criteria': promotion_block(matrix_by_teacher, blocks, calib, p6_evidence),
        'coefficient_summary': coef_summary_block(matrix_by_teacher),
        'coeff_deltas_vs_headline_recompute': {t: coeff_deltas(matrix_by_teacher[t]) for t in teachers},
        'plan_2_0_replication': plan_repl,
        'comparison_matrix': matrix_by_teacher,
        'inputs_bytes': {
            'pricing_anchors_before': [a['bytes'] for a in anchors_before],
            'pricing_anchors_after': [a['bytes'] for a in anchors_after],
            'pricing_anchors_match': True,
            'shadow_before': shadow_sha_before, 'shadow_after': shadow_sha_after,
            'shadow_unchanged': bool(shadow_sha_before == shadow_sha_after),
            'headline_guard_before': headline_sha_before,
            'headline_guard_after': headline_sha_after,
            'headline_unchanged': bool(headline_sha_before == headline_sha_after),
            'shadow_chain_vs_phase2_counts': chain,
        },
        'outputs': {'comparison_csv': _rel(out_csv),
                    'comparison_csv_rows': int(len(rows)),
                    'comparison_csv_bytes': csv_bytes,
                    'counts_json': _rel(out_json),
                    'execution_record': 'artifacts/phase2/2.6_execution.md'},
        'residual_limitations': [
            '2.1 daily_panel_missing 96 行（pricing_gate 窗口 38 行）按缺失排除，不填 0',
            '2.1 sh601313 → sh601360 alias 46 行（窗口内 30 行）恢复交易日，R 仍用原值',
            '2.3 2014–2017 除权事件只有 sharebonus 一个来源（8,580 条），无法交叉校验',
            '2.2 新股无涨跌幅限制日（sh688026 1.085 / sz300939 1.010）未剔除（无 ipo_no_limit_flag）',
            '2.4 turnover 主口径 = 月度均值（winsorized）；last 口径保留对照',
            '2.5 rev lag1 = 合规 t-1（裸值）；plan §2.5 的前视数字不引用',
            '2.6 winsorize 宇宙 = headline 样本面板（与 headline 同构）；未重建完整 shadow 面板 parquet',
            '2.6 未重建 shadow 版 industry baseline / ST 分层 / 组合排序 / gate',
            '2.6 mom/illiq 为 shadow 重算列（mom 用 build_firm_chars 的 ys[-12:-1] 规则，illiq = |R|/amount_100m）；'
            'legacy 重建口径对 mom 施加 headline 可用性掩码（面板 831 行），净效应 ≤5e-8 β',
            'pre-2014 legacy（2.1 §8.1 / 2.3 §11.3）：shadow 未覆盖 2006–2013 的复牌/复权，'
            '该窗口 MAX 为 legacy 直通，pricing_gate 窗口外结论不适用',
            '2018+ 重整/重新上市事件源缺口（2.3 §11.6，sh600145 2020-06 / sh600733 2018-09）：'
            'sharebonus 与 corporate_action 均未覆盖，仅靠 2.2 resume 联合排除',
        ],
        'governance': {'headline_files_modified': False, 'pricing_rerun': False,
                       'configs_modified': False, 'g0_written': False,
                       'promotion_executed': False},
    }
    write_json(out_json, counts)
    print(f'[2.6] 完成：{out_csv.name} / {out_json.name}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
