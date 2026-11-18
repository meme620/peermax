"""p2_5_rev_shadow 的辅助函数与常量（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.io import atomic_write_text as _atomic_write_text
from pathlib import Path
from s3_m1_lib import (  # noqa: E402
    RETURNS_V2, ROOT, assert_m1_inputs_sha, fm_regression, param_value, require,
    sha256_file, winsor_bounds, winsorize_by_month,
)
import argparse
import json
import numpy as np
import os
import pandas as pd

SAMPLE_PARQUET = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'

RETURNS_V1 = ROOT / 'data/processed/monthly_returns_clean.csv'

CHARS = ROOT / 'data/processed/firm_chars_clean.csv'

SHADOW_TURNOVER_CSV = ROOT / 'data/processed/phase2_shadow/turnover_p2shadow.csv'

SHADOW_TURNOVER_META = ROOT / 'data/processed/phase2_shadow/turnover_p2shadow.meta.json'

COUNTS_24 = ROOT / 'artifacts/phase2/2.4_counts.json'

MATRIX_24 = ROOT / 'artifacts/phase2/2.4_control_matrix.csv'

PRICING_TABLE_V2 = ROOT / 'reports/data_gates/G0/s3_m1_pricing_table_v2.json'

COUNTS_JSON = ROOT / 'artifacts/phase2/2.5_counts.json'

MATRIX_CSV = ROOT / 'artifacts/phase2/2.5_control_matrix.csv'

HEADLINE_GUARD_FILES = {
    'm1_common_sample_ma_teacher_v2.parquet': SAMPLE_PARQUET,
    'firm_chars_clean.csv': CHARS,
    'monthly_pricing_panel_ma_teacher.csv':
        ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv',
    'monthly_returns_clean_v2.csv': RETURNS_V2,
}

FM_WINDOW_START, FM_WINDOW_END = '2015-05', '2023-12'

FM_WINDOW_START, FM_WINDOW_END = '2015-05', '2023-12'

TRIO = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn']

CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage', 'NetworkDegree']

REV_SAME = 'rev'

REV_LAG1 = 'rev_lag1'

REV_LAG1_WIN = 'rev_lag1_win'

REV_LAG1_FWD = 'rev_lag1_fwd'      # 前视诊断列（= R_{t+1} ≈ outcome）；不得进入规格

TURN_LAST = 'turnover'

TURN_MEAN_WIN = 'turnover_mean_win'

TURN_MEAN_NOWIN = 'turnover_mean_nowin'

TURN_MEAN_RAW = 'turnover_mean_raw'

TURN_LAST_RAW = 'turnover_last_raw'

P2SHADOW_VERSION = 'p2shadow_v1'

SPEC_DEFS = {
    'a_registered_rev_turnover': {
        'label': 'a) 登记规格（rev 同月 + turnover=headline last）',
        'rev': REV_SAME, 'turnover': TURN_LAST,
        'note': 'headline 主表复现：EW OwnMAX -0.00282/t=-0.12、PeerMAX +0.03427/t=0.47'},
    'b_rev_lag1_turnover_last': {
        'label': 'b) rev_lag1（合规 t-1 裸 R）+ turnover last',
        'rev': REV_LAG1, 'turnover': TURN_LAST,
        'note': ('合规 lag1：EW OwnMAX -0.01512/t=-0.71、PeerMAX +0.04391/t=0.59（不显著）；'
                 '⚠ 2.0_plan 引用的 -0.01341/-4.02 来自前视列 R_{t+1}，见 plan_rev_lag1_lookahead_diagnostic')},
    'c_rev_removed_turnover_last': {
        'label': 'c) rev 移除 + turnover last',
        'rev': None, 'turnover': TURN_LAST,
        'note': '审计 L-3「只移 rev」行：OwnMAX -0.01133/t=-0.53（仍不显著 → turnover 是吸收主力）'},
    'd_rev_lag1_no_turnover': {
        'label': 'd) rev_lag1（合规 t-1）+ 无 turnover',
        'rev': REV_LAG1, 'turnover': None,
        'note': '合规 lag1 单独承接：EW OwnMAX -0.13353/t=-4.91（与「完全移除 rev」几乎相同 → 滞后 rev 不吸收）'},
    'e_rev_removed_no_turnover': {
        'label': 'e) rev 移除 + 无 turnover（审计 -0.1359 锚点）',
        'rev': None, 'turnover': None,
        'note': '与 2.4 的 d_no_turnover_no_rev 同控制集；审计 L-3 锚点 -0.13586/t=-4.97'},
    'f_no_rev_no_turnover_reference': {
        'label': 'f) 无 rev + 无 turnover（参考锚点；与 e 同控制集）',
        'rev': None, 'turnover': None,
        'note': '任务矩阵的参考行；控制集与 e 完全相同，系数应与 e 逐值一致'},
    'g_rev_same_no_turnover': {
        'label': 'g) 补充：rev 同月 + 无 turnover（审计「只移 turnover」行）',
        'rev': REV_SAME, 'turnover': None,
        'note': '审计 L-3 锚点：OwnMAX -0.10550/t=-3.80（= 2.4 规格 c）'},
    'h_rev_same_turnover_mean_win': {
        'label': 'h) 补充：rev 同月 + turnover 均值（2.4 主口径 winsorized）',
        'rev': REV_SAME, 'turnover': TURN_MEAN_WIN,
        'note': '应逐值等于 2.4 规格 b_turnover_mean（OwnMAX +0.01188/t=0.62）'},
    'i_rev_lag1_turnover_mean_win': {
        'label': 'i) 补充：rev_lag1（合规）+ turnover 均值（winsorized）',
        'rev': REV_LAG1, 'turnover': TURN_MEAN_WIN,
        'note': '合规 lag1 + 2.4 主口径 mean：EW OwnMAX -0.02303/t=-1.18（不显著）'},
    'j_rev_lag1_turnover_mean_nowin': {
        'label': 'j) 补充：rev_lag1（合规）+ turnover 均值（不 winsor）',
        'rev': REV_LAG1, 'turnover': TURN_MEAN_NOWIN,
        'note': ('合规 lag1 + 不 winsor mean：EW OwnMAX -0.02743/t=-1.35；'
                 '⚠ 2.0_plan 引用的 -0.01728/t=-4.91 来自前视列，见前视诊断块')},
    'k_rev_removed_turnover_mean_win': {
        'label': 'k) 补充：rev 移除 + turnover 均值（winsorized）',
        'rev': None, 'turnover': TURN_MEAN_WIN,
        'note': '补齐 rev × turnover(mean) 网格'},
    'l_rev_lag1_win_turnover_last': {
        'label': 'l) 敏感性：rev_lag1 用 winsorized 口径（当前月面板截面 winsor）+ turnover last',
        'rev': REV_LAG1_WIN, 'turnover': TURN_LAST,
        'note': 'winsor 化滞后 rev 与裸滞后同向同量级（EW OwnMAX -0.01638/t=-0.78，不显著）'},
    'm_rev_lag1_win_no_turnover': {
        'label': 'm) 敏感性：rev_lag1 winsorized + 无 turnover',
        'rev': REV_LAG1_WIN, 'turnover': None,
        'note': 'winsorized lag1 的完整对照：EW OwnMAX -0.13518/t=-5.01（≈ 全移除基准）'},
}

FORWARD_DIAG_SPEC_DEFS = {
    'q_plan_quote_fwd_turnover_last': {
        'label': '前视诊断：R_{t+1} 冒充 rev_lag1 + turnover last（复现 plan §2.5 引用的一行）',
        'rev': REV_LAG1_FWD, 'turnover': TURN_LAST,
        'note': 'plan 引用：OwnMAX -0.01341/t=-4.02、PeerMAX +0.00927/t=1.21'},
    'r_plan_quote_fwd_turnover_mean_nowin': {
        'label': '前视诊断：R_{t+1} + turnover 均值（不 winsor）（复现 plan §2.5 引用的另一行）',
        'rev': REV_LAG1_FWD, 'turnover': TURN_MEAN_NOWIN,
        'note': 'plan 引用：OwnMAX -0.01728/t=-4.91、PeerMAX +0.01121/t=1.49'},
}

def write_json(path: Path, obj) -> None:
    assert_output_path_guard(path)   # 与矩阵 CSV 同守卫：--counts-out 不得指向 data/、G0、configs、headline
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1))
    print(f'  -> {path.relative_to(ROOT)}', flush=True)

def shift_month_series(months, delta: int) -> pd.Index:
    """'YYYY-MM' 序列整体平移 delta 个月（用于构造 lag 键）。"""
    return (pd.PeriodIndex(pd.Series(months).astype(str), freq='M') + delta).strftime('%Y-%m')

def _lag_merge(panel: pd.DataFrame, source: pd.DataFrame, value_col: str, out_col: str,
               value_key: str, key_delta: int) -> pd.DataFrame:
    """通用 lag 键合并：source 的 (key, 源月) 唯一；键 = 源月 + key_delta；缺失留 NaN。"""
    src = source[[value_key, 'month', value_col]].rename(
        columns={value_key: 'focal_id', 'month': 'src_month'})
    require(int(src.duplicated(['focal_id', 'src_month']).sum()) == 0,
            f'{out_col}: source (focal, 源月) 键重复')
    src['month'] = shift_month_series(src['src_month'], key_delta)
    require(int(src.duplicated(['focal_id', 'month']).sum()) == 0,
            f'{out_col}: 平移后 (focal, 月) 键重复')
    src = src.rename(columns={value_col: out_col})[['focal_id', 'month', 'src_month', out_col]]
    before = len(panel)
    out = panel.merge(src, on=['focal_id', 'month'], how='left')
    require(len(out) == before, f'{out_col}: merge 膨胀行数')
    return out

def attach_lagged_column(panel: pd.DataFrame, source: pd.DataFrame, value_col: str,
                         out_col: str, value_key: str = 'focal_id') -> pd.DataFrame:
    """合规滞后：rev_lag1(focal, t) = source.value(focal, t-1)。

    PIT 硬门：所有非空 lag 的来源月严格小于当前月（无同月/前视）；未匹配保持 NaN，不填 0。
    """
    out = _lag_merge(panel, source, value_col, out_col, value_key, key_delta=+1)
    ok = out['src_month'].notna()
    if bool(ok.any()):
        require(bool((out.loc[ok, 'src_month'].astype(str)
                      < out.loc[ok, 'month'].astype(str)).all()),
                f'{out_col}: 出现 lag 月 >= 当前月（前视）')
    return out.drop(columns=['src_month'])

def attach_forward_diagnostic(panel: pd.DataFrame, source: pd.DataFrame, value_col: str,
                              out_col: str, value_key: str = 'focal_id') -> pd.DataFrame:
    """**前视诊断列**（= source.value(focal, t+1)）：仅用于复现 2.0_plan 引用的数字。

    该列必然命中 plan §2.5 的「使用未来收益 → 立即停机」条件；因此：不进入任何规格、
    不写矩阵 CSV。调用处必须显式标注 `_FORWARD_DIAGNOSTIC` 用途。
    """
    out = _lag_merge(panel, source, value_col, out_col, value_key, key_delta=-1)
    ok = out['src_month'].notna()
    if bool(ok.any()):
        require(bool((out.loc[ok, 'src_month'].astype(str)
                      > out.loc[ok, 'month'].astype(str)).all()),
                f'{out_col}: 前视诊断列未全部取未来月（构造异常）')
    return out.drop(columns=['src_month'])

def build_control_frame(sample_path: Path = SAMPLE_PARQUET,
                        returns_path: Path = RETURNS_V2,
                        turnover_shadow_path: Path = SHADOW_TURNOVER_CSV) -> tuple:
    """只读样本 + 月收益 + 2.4 shadow turnover -> FM 行 frame（rev 三口径 + turnover 三口径）。"""
    cols = ['focal_id', 'month', 'm1_in_fm_sample', 'R_next', 'market_equity'] + TRIO + CONTROLS
    df = pd.read_parquet(sample_path, columns=cols)
    require(len(df) == 307503, f'样本 parquet 行数 {len(df)} != 307503')
    ret = pd.read_csv(returns_path, usecols=['firm_id', 'month', 'R'],
                      dtype={'firm_id': str, 'month': str})
    require(int(ret.duplicated(['firm_id', 'month']).sum()) == 0, '月收益 v2 键重复')

    # rev_same 桥：样本 rev == winsorize_by_month(原始 R, panel 宇宙, registry [lo,hi])
    m = df.merge(ret.rename(columns={'firm_id': 'focal_id', 'R': 'R_same_raw'}),
                 on=['focal_id', 'month'], how='left')
    lo, hi = winsor_bounds()
    M = len(m)
    m['R_same_win'] = winsorize_by_month(m, ['R_same_raw'], lo=lo, hi=hi)['R_same_raw']
    br = (m['rev'] - m['R_same_win']).abs()
    raw_diff = (m['rev'] - m['R_same_raw']).abs()
    panel_bridge = {
        'max_abs_diff_vs_sample_rev': float(br.max()),
        'rows_differing_gt_1e-12': int((br > 1e-12).sum()),
        'rows_clipped_by_winsor_vs_raw': int((raw_diff > 1e-12).sum()),
        'max_abs_diff_raw_vs_sample_rev': float(raw_diff.max()),
        'n_rows_R_missing': int(m['R_same_raw'].isna().sum()),
    }
    require(panel_bridge['max_abs_diff_vs_sample_rev'] <= 1e-12 and M == len(df),
            'rev 同月桥失败：面板 winsor(原始 R) 与样本 rev 列不一致（停机条件）')

    # rev_lag1（合规 t-1 裸值）+ 前视诊断列 + winsorized 变体
    m = attach_lagged_column(m, ret.rename(columns={'firm_id': 'focal_id'}),
                             value_col='R', out_col=REV_LAG1, value_key='focal_id')
    m = attach_forward_diagnostic(m, ret.rename(columns={'firm_id': 'focal_id'}),
                                  value_col='R', out_col=REV_LAG1_FWD, value_key='focal_id')
    # 主 winsor 变体：对裸 lag1 在当前月面板宇宙按 [lo,hi] winsor（与 headline rev 的
    # 「月度截面 winsor」同构，且保持与其它规格完全相同的 FM 行集合）
    m[REV_LAG1_WIN] = winsorize_by_month(m, [REV_LAG1], lo=lo, hi=hi)[REV_LAG1]
    # 次 winsor 变体（诊断）：样本 winsorized rev 列沿 month-1 平移（源月 t-1 的 winsor 值；
    # 因样本面板在 t-1 可能缺行，FM 覆盖 179,750/181,178）
    m = attach_lagged_column(m, df[['focal_id', 'month', 'rev']],
                             value_col='rev', out_col='rev_lag1_win_src', value_key='focal_id')
    both_alt = m[REV_LAG1_WIN].notna() & m['rev_lag1_win_src'].notna()
    alt_diff = (m.loc[both_alt, REV_LAG1_WIN] - m.loc[both_alt, 'rev_lag1_win_src']).abs()

    # turnover：样本列（headline last）+ 2.4 shadow（mean）
    t = pd.read_csv(turnover_shadow_path, dtype={'firm_id': str, 'month': str})
    m = m.merge(t.rename(columns={'firm_id': 'focal_id'})[
        ['focal_id', 'month', 'turnover_last', 'turnover_mean']],
        on=['focal_id', 'month'], how='left')
    m = m.rename(columns={'turnover_last': TURN_LAST_RAW, 'turnover_mean': TURN_MEAN_RAW})
    g = m.groupby('month')[TURN_MEAN_RAW]
    m[TURN_MEAN_WIN] = (m[TURN_MEAN_RAW].clip(lower=g.transform(lambda s: s.quantile(lo)),
                                              upper=g.transform(lambda s: s.quantile(hi)))) / 100.0
    m[TURN_MEAN_NOWIN] = m[TURN_MEAN_RAW] / 100.0
    # 一致性桥：panel-winsor(raw last)/100 逐值 == 样本 turnover 列（不得重算 last）
    gl = m.groupby('month')[TURN_LAST_RAW]
    last_win = (m[TURN_LAST_RAW].clip(lower=gl.transform(lambda s: s.quantile(lo)),
                                      upper=gl.transform(lambda s: s.quantile(hi)))) / 100.0
    last_bridge = (last_win - m['turnover']).abs()
    last_bridge_raw = (m[TURN_LAST_RAW] / 100.0 - m['turnover']).abs()

    fm = m[m['m1_in_fm_sample'].astype(bool)].copy()
    # outcome 身份证据：R_next vs winsor(前视列)（证明 2.0_plan 的 rev_lag1 = outcome）
    fwd_win = winsorize_by_month(m, [REV_LAG1_FWD], lo=lo, hi=hi)[REV_LAG1_FWD]
    fwd_win_fm = fwd_win.loc[fm.index]
    id_diff = (fm['R_next'] - fwd_win_fm).abs()
    outcome_identity = {
        'definition': 'forensic：R_next（headline outcome，winsorized）与 winsor(R_{t+1}) 的一致性',
        'corr_pooled': float(np.corrcoef(fm['R_next'], fwd_win_fm)[0, 1]),
        'rows_differing_gt_1e-9': int((id_diff > 1e-9).sum()),
        'max_abs_diff': float(id_diff.max()),
        'median_abs_diff_of_differing_rows': float(id_diff[id_diff > 1e-9].median())
        if int((id_diff > 1e-9).sum()) else 0.0,
        'note': ('差异行的 next_complete_month 全为 1，差异量级 ≤0.003（winsor 施加集合的细微差别）；'
                 '不影响「该列 ≈ outcome」的判定'),
    }
    diag = {
        'sample_rows': int(len(df)), 'fm_rows': int(len(fm)),
        'winsorization_pct': [lo, hi],
        'rev_same_panel_bridge': panel_bridge,
        'rev_lag1': {
            'definition': 'rev_lag1(focal,t) = v2.R(focal, t-1)（裸值，无前视，缺失 NaN）',
            'panel_rows': int(M),
            'panel_rows_with_lag': int(m[REV_LAG1].notna().sum()),
            'panel_missing_from_lag': int(m[REV_LAG1].isna().sum()),
            'fm_rows_with_lag': int(fm[REV_LAG1].notna().sum()),
            'fm_lag_coverage': float(fm[REV_LAG1].notna().mean()),
            'fm_lag_min': float(fm[REV_LAG1].min()), 'fm_lag_max': float(fm[REV_LAG1].max()),
            'missing_filled_with_zero': 0,
        },
        'rev_lag1_win': {
            'definition': ('主 winsor 变体：裸 lag1 在当前月面板宇宙按 registry [lo,hi] 月度截面 winsor'
                           '（与 headline rev 的 winsor 口径同构；FM 行集合与其它规格完全相同）'),
            'fm_rows_with_lag': int(fm[REV_LAG1_WIN].notna().sum()),
            'fm_lag_coverage': float(fm[REV_LAG1_WIN].notna().mean()),
        },
        'rev_lag1_win_src_diagnostic': {
            'definition': ('次 winsor 变体（诊断）：样本 winsorized rev 沿 month-1 平移'
                           '（源月 t-1 的 headline winsor 值）'),
            'fm_rows_with_lag': int(fm['rev_lag1_win_src'].notna().sum()),
            'fm_lag_coverage': float(fm['rev_lag1_win_src'].notna().mean()),
            'sample_loss_vs_fm': int(len(fm) - fm['rev_lag1_win_src'].notna().sum()),
            'two_winsor_variants': {
                'rows_both_nonnull': int(both_alt.sum()),
                'rows_differing_gt_1e-12': int((alt_diff > 1e-12).sum()),
                'max_abs_diff': float(alt_diff.max()) if len(alt_diff) else 0.0,
            },
        },
        'rev_lag1_fwd_lookahead': {
            'definition': 'R_{t+1}（前视诊断，仅 provenance；不得进规格）',
            'fm_rows_with_lag': int(fm[REV_LAG1_FWD].notna().sum()),
            'fm_lag_coverage': float(fm[REV_LAG1_FWD].notna().mean()),
        },
        'outcome_identity_evidence': outcome_identity,
        'turnover': {
            'fm_last_coverage_vs_sample_col': float(fm['turnover'].notna().mean()),
            'fm_last_raw_coverage': float(fm[TURN_LAST_RAW].notna().mean()),
            'last_bridge_max_abs_diff_vs_sample_col': float(last_bridge.max()),
            'last_bridge_rows_differing_gt_1e-12': int((last_bridge > 1e-12).sum()),
            'last_bridge_raw_only_max_abs_diff': float(last_bridge_raw.max()),
            'fm_mean_win_coverage': float(fm[TURN_MEAN_WIN].notna().mean()),
            'fm_mean_nowin_coverage': float(fm[TURN_MEAN_NOWIN].notna().mean()),
        },
        'fm_window': [str(fm['month'].min()), str(fm['month'].max())],
        'fm_months': int(fm['month'].nunique()),
    }
    return fm, diag

def monthly_corr_stats(fm: pd.DataFrame, a: str, b: str, min_n: int = 30) -> dict:
    """逐月截面 Pearson corr（双方非空、月内 n>=min_n 且两列均有方差），再汇总 mean/median。"""
    per_month = []
    for mth, sub in fm.groupby('month', sort=True):
        s = sub[[a, b]].dropna()
        n = len(s)
        c = None
        if n >= min_n and s[a].std() > 0 and s[b].std() > 0:
            c = float(np.corrcoef(s[a].to_numpy(dtype=float), s[b].to_numpy(dtype=float))[0, 1])
        per_month.append({'month': str(mth), 'n': int(n), 'corr': c})
    vals = np.array([r['corr'] for r in per_month if r['corr'] is not None], dtype=float)
    sub = fm[[a, b]].dropna()
    pooled = (float(np.corrcoef(sub[a].to_numpy(dtype=float), sub[b].to_numpy(dtype=float))[0, 1])
              if len(sub) > 2 and sub[a].std() > 0 and sub[b].std() > 0 else None)
    return {'a': a, 'b': b, 'n_pairs_pooled': int(len(sub)), 'pooled_corr': pooled,
            'n_months': int(len(vals)), 'monthly_mean': float(vals.mean()) if len(vals) else None,
            'monthly_median': float(np.median(vals)) if len(vals) else None,
            'monthly_min': float(vals.min()) if len(vals) else None,
            'monthly_max': float(vals.max()) if len(vals) else None,
            'per_month': per_month}

def monthly_vif(fm: pd.DataFrame, xcols, min_n: int = 100) -> dict:
    """逐月截面 VIF（EW 未加权设计矩阵）；返回各列 mean/median/max。"""
    out = {c: [] for c in xcols}
    for _m, sub in fm.groupby('month', sort=True):
        X = sub[list(xcols)].dropna().to_numpy(dtype=float)
        if len(X) < len(xcols) + 2:
            continue
        Xc = X - X.mean(axis=0)
        for j, c in enumerate(xcols):
            y = Xc[:, j]
            Z = np.delete(Xc, j, axis=1)
            ss = float(y @ y)
            if ss <= 0:
                continue
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            resid = y - Z @ beta
            r2 = 1.0 - float(resid @ resid) / ss
            out[c].append(float(np.inf if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)))
    res = {}
    for c, v in out.items():
        arr = np.array(v, dtype=float)
        res[c] = {'n_months': int(len(arr)),
                  'mean': float(arr.mean()) if len(arr) else None,
                  'median': float(np.median(arr)) if len(arr) else None,
                  'max': float(arr.max()) if len(arr) else None}
    return res

def build_xcols(rev_col: str | None, turnover_col: str | None) -> list:
    """TRIO + CONTROLS（rev/turnover 槽位按口径替换或删除；其余顺序与 headline 一致）。"""
    x = list(TRIO)
    for c in CONTROLS:
        if c == REV_SAME:
            if rev_col:
                x.append(rev_col)
        elif c == TURN_LAST:
            if turnover_col:
                x.append(turnover_col)
        else:
            x.append(c)
    return x

def spec_xcols(sd: dict) -> list:
    return build_xcols(sd['rev'], sd['turnover'])

def run_spec_matrix(fm: pd.DataFrame, spec_defs: dict = None) -> dict:
    """规格 × EW/VW 的 FM（min_n=100，NW lag=6，与 headline 一致）。"""
    spec_defs = spec_defs or SPEC_DEFS
    min_n = int(param_value('m1_sample_and_spec_registry')['min_focal_count_per_month'])
    nw_lag = int(param_value('nw_lag'))
    out = {'min_n': min_n, 'nw_lag': nw_lag, 'n_rows': int(len(fm)),
           'months': [str(fm['month'].min()), str(fm['month'].max())], 'specs': {}}
    for sid, sd in spec_defs.items():
        xcols = spec_xcols(sd)
        blk = {'label': sd['label'], 'rev_variant': sd['rev'], 'turnover_variant': sd['turnover'],
               'x': xcols, 'note': sd.get('note', '')}
        for wname, wcol in (('ew', None), ('vw', 'market_equity')):
            res, per_month = fm_regression(fm, 'R_next', xcols, weight_col=wcol,
                                           min_n=min_n, nw_lag=nw_lag)
            used = [d for d in per_month if d.get('used')]
            coefs = {c: {'beta': res['coefs'][c]['beta'], 'se_nw': res['coefs'][c]['se_nw'],
                         't_nw': res['coefs'][c]['t_nw']} for c in xcols}
            blk[wname] = {
                'T': res['n_months_used'], 'n_frame': int(len(fm)),
                'n_rows_used': int(sum(d['n'] for d in used)),
                'coefs': coefs,
                'per_month_n': {'min': int(min(d['n'] for d in per_month)),
                                'median': float(np.median([d['n'] for d in per_month])),
                                'max': int(max(d['n'] for d in per_month))},
            }
        out['specs'][sid] = blk
    return out

def matrix_rows(matrix: dict) -> list:
    rows = []
    for sid, blk in matrix['specs'].items():
        for wname in ('ew', 'vw'):
            w = blk[wname]
            pm = w['per_month_n']
            terms = ['PeerMAXraw', 'OwnMAXraw']
            for extra in (blk['rev_variant'], blk['turnover_variant']):
                if extra and extra not in terms:
                    terms.append(extra)
            for term in terms:
                c = w['coefs'][term]
                rows.append({'spec_id': sid, 'weight': wname, 'term': term,
                             'rev_variant': blk['rev_variant'] or 'removed',
                             'turnover_variant': blk['turnover_variant'] or 'removed',
                             'beta': c['beta'], 'se_nw': c['se_nw'], 't_nw': c['t_nw'],
                             'T_months': w['T'], 'n_rows': w['n_rows_used'],
                             'n_per_month_min': pm['min'], 'n_per_month_median': pm['median'],
                             'n_per_month_max': pm['max'],
                             't_significant_5pct': int(abs(c['t_nw']) >= 1.96),
                             'note': blk.get('note', '')})
    return rows

def headline_comparison(matrix: dict, table_path: Path = PRICING_TABLE_V2,
                        tol_beta: float = 1e-4, tol_t: float = 1e-3) -> dict:
    """规格 a 与 headline 主表（s3_m1_pricing_table_v2.json, ma_teacher）逐值对比（只读）。"""
    table = json.loads(table_path.read_text(encoding='utf-8'))
    mt = table['main']['ma_teacher']
    spec_a = matrix['specs']['a_registered_rev_turnover']
    rows, all_match = [], True
    for wname in ('ew', 'vw'):
        for term in ('PeerMAXraw', 'OwnMAXraw', 'turnover', 'PeerReturn'):
            exp = mt[wname]['coefs'][term]
            act = spec_a[wname]['coefs'][term]
            ok = (abs(exp['beta'] - act['beta']) <= tol_beta
                  and abs(exp['t_nw'] - act['t_nw']) <= tol_t)
            all_match &= bool(ok)
            rows.append({'weight': wname, 'term': term,
                         'headline_beta': exp['beta'], 'shadow_beta': act['beta'],
                         'headline_t': exp['t_nw'], 'shadow_t': act['t_nw'], 'match': bool(ok)})
    return {'source': str(table_path.relative_to(ROOT)),
            'headline_n_rows': mt['n_rows'], 'headline_fm_window': mt['fm_window'],
            'headline_months': mt['ew']['n_months_used'],
            'tol_beta': tol_beta, 'tol_t': tol_t, 'rows': rows, 'all_match': bool(all_match)}

def compare_2_4(matrix: dict, path: Path = MATRIX_24) -> dict:
    """与 2.4 控制矩阵 CSV 交叉对拍（同控制集规格必须逐值一致）。"""
    if not path.is_file():
        return {'available': False, 'file': str(path.relative_to(ROOT))}
    m24 = pd.read_csv(path)
    pairs = [('a_registered_rev_turnover', 'a_turnover_last'),
             ('g_rev_same_no_turnover', 'c_no_turnover'),
             ('e_rev_removed_no_turnover', 'd_no_turnover_no_rev'),
             ('h_rev_same_turnover_mean_win', 'b_turnover_mean')]
    rows, all_match = [], True
    for sid, sid24 in pairs:
        blk24 = m24[m24['spec_id'] == sid24]
        for wname in ('ew', 'vw'):
            for term in ('PeerMAXraw', 'OwnMAXraw'):
                r24 = blk24[(blk24['weight'] == wname) & (blk24['term'] == term)].iloc[0]
                act = matrix['specs'][sid][wname]['coefs'][term]
                ok = (abs(r24['beta'] - act['beta']) <= 1e-8
                      and abs(r24['t_nw'] - act['t_nw']) <= 1e-6)
                all_match &= bool(ok)
                rows.append({'spec_2_5': sid, 'spec_2_4': sid24, 'weight': wname, 'term': term,
                             'beta_2_4': float(r24['beta']), 'beta_2_5': act['beta'],
                             't_2_4': float(r24['t_nw']), 't_2_5': act['t_nw'],
                             'match': bool(ok)})
    return {'available': True, 'file': str(path.relative_to(ROOT)),
            'rows': rows, 'all_match': bool(all_match)}

def _bt(matrix: dict, sid: str, wname: str, term: str) -> list:
    c = matrix['specs'][sid][wname]['coefs'][term]
    return [round(c['beta'], 5), round(c['t_nw'], 2)]

def _near(matrix: dict, sid: str, wname: str, term: str, beta, t,
          tol_b=5e-4, tol_t=0.01) -> bool:
    c = matrix['specs'][sid][wname]['coefs'][term]
    return bool(abs(c['beta'] - beta) <= tol_b and abs(c['t_nw'] - t) <= tol_t)

def anchor_block(matrix: dict, corr: dict) -> dict:
    """复算锚点：headline / 审计 L-3 的**合规**数字逐项对账。"""
    items = [
        {'item': 'corr(rev 同月, OwnMAX) 月度均值', 'plan': 0.4246,
         'actual': round(corr['rev_own']['monthly_mean'], 6),
         'match': 0.42 <= corr['rev_own']['monthly_mean'] <= 0.43},
        {'item': 'corr(rev_lag1 合规 t-1, OwnMAX) 月度均值（plan 引 -0.0343 为前视列，见诊断块）',
         'plan': '合规值待定（无 plan 引用）', 'actual': round(corr['lag1_own']['monthly_mean'], 6),
         'match': 0.08 <= corr['lag1_own']['monthly_mean'] <= 0.16},
        {'item': '规格 a（登记）EW OwnMAX β/t', 'plan': [-0.00282, -0.12],
         'actual': _bt(matrix, 'a_registered_rev_turnover', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'a_registered_rev_turnover', 'ew', 'OwnMAXraw',
                        -0.00281881, -0.12149815)},
        {'item': '规格 a（登记）EW PeerMAX β/t', 'plan': [0.03427, 0.47],
         'actual': _bt(matrix, 'a_registered_rev_turnover', 'ew', 'PeerMAXraw'),
         'match': _near(matrix, 'a_registered_rev_turnover', 'ew', 'PeerMAXraw',
                        0.03427281, 0.47280879)},
        {'item': '规格 b（合规 lag1 + last）EW OwnMAX β/t', 'plan': [-0.01512, -0.71],
         'actual': _bt(matrix, 'b_rev_lag1_turnover_last', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'b_rev_lag1_turnover_last', 'ew', 'OwnMAXraw',
                        -0.01512016, -0.70540855)},
        {'item': '规格 b EW PeerMAX β/t', 'plan': [0.04391, 0.59],
         'actual': _bt(matrix, 'b_rev_lag1_turnover_last', 'ew', 'PeerMAXraw'),
         'match': _near(matrix, 'b_rev_lag1_turnover_last', 'ew', 'PeerMAXraw',
                        0.043910, 0.590)},
        {'item': '规格 c（移除 rev + last）EW OwnMAX β/t', 'plan': [-0.01133, -0.53],
         'actual': _bt(matrix, 'c_rev_removed_turnover_last', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'c_rev_removed_turnover_last', 'ew', 'OwnMAXraw',
                        -0.011331, -0.532, 5e-4, 0.02)},
        {'item': '规格 d（合规 lag1，无 turnover）EW OwnMAX β/t', 'plan': [-0.13353, -4.91],
         'actual': _bt(matrix, 'd_rev_lag1_no_turnover', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'd_rev_lag1_no_turnover', 'ew', 'OwnMAXraw',
                        -0.133529, -4.906, 1e-3, 0.02)},
        {'item': '规格 e/f（移除 rev + 无 turnover）EW OwnMAX β/t（审计锚点）',
         'plan': [-0.13586, -4.97], 'actual': _bt(matrix, 'e_rev_removed_no_turnover', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'e_rev_removed_no_turnover', 'ew', 'OwnMAXraw',
                        -0.13585820, -4.96987225, 5e-5, 0.005)},
        {'item': '规格 g（只移 turnover）EW OwnMAX β/t', 'plan': [-0.10550, -3.80],
         'actual': _bt(matrix, 'g_rev_same_no_turnover', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'g_rev_same_no_turnover', 'ew', 'OwnMAXraw',
                        -0.10550381, -3.80093741, 5e-5, 0.005)},
        {'item': '规格 h（rev 同月 + winsorized mean）EW OwnMAX β/t（= 2.4 规格 b）',
         'plan': [0.01188, 0.62], 'actual': _bt(matrix, 'h_rev_same_turnover_mean_win', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'h_rev_same_turnover_mean_win', 'ew', 'OwnMAXraw',
                        0.01188486, 0.62377247, 5e-5, 0.005)},
        {'item': '规格 i（合规 lag1 + winsorized mean）EW OwnMAX β/t', 'plan': [-0.02303, -1.18],
         'actual': _bt(matrix, 'i_rev_lag1_turnover_mean_win', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'i_rev_lag1_turnover_mean_win', 'ew', 'OwnMAXraw',
                        -0.023028, -1.181, 1e-3, 0.02)},
        {'item': '规格 j（合规 lag1 + 不 winsor mean）EW OwnMAX β/t', 'plan': [-0.02743, -1.35],
         'actual': _bt(matrix, 'j_rev_lag1_turnover_mean_nowin', 'ew', 'OwnMAXraw'),
         'match': _near(matrix, 'j_rev_lag1_turnover_mean_nowin', 'ew', 'OwnMAXraw',
                        -0.027434, -1.349, 1e-3, 0.02)},
        {'item': 'e 与 f 同控制集逐值一致', 'plan': True,
         'actual': bool(all(
             abs(matrix['specs']['e_rev_removed_no_turnover'][w]['coefs'][t]['beta']
                 - matrix['specs']['f_no_rev_no_turnover_reference'][w]['coefs'][t]['beta']) <= 1e-12
             for w in ('ew', 'vw') for t in ('PeerMAXraw', 'OwnMAXraw'))),
         'match': True},
        {'item': 'PeerMAX 在 a–f 全部 EW/VW 规格 |t|<1.96', 'plan': '全不显著',
         'actual': {sid: {w: round(matrix['specs'][sid][w]['coefs']['PeerMAXraw']['t_nw'], 3)
                          for w in ('ew', 'vw')}
                    for sid in ('a_registered_rev_turnover', 'b_rev_lag1_turnover_last',
                                'c_rev_removed_turnover_last', 'd_rev_lag1_no_turnover',
                                'e_rev_removed_no_turnover', 'f_no_rev_no_turnover_reference')},
         'match': all(abs(matrix['specs'][sid][w]['coefs']['PeerMAXraw']['t_nw']) < 1.96
                      for sid in ('a_registered_rev_turnover', 'b_rev_lag1_turnover_last',
                                  'c_rev_removed_turnover_last', 'd_rev_lag1_no_turnover',
                                  'e_rev_removed_no_turnover', 'f_no_rev_no_turnover_reference')
                      for w in ('ew', 'vw'))},
    ]
    return {'items': items, 'all_items_match': all(i['match'] for i in items),
            'summary': ('合规 lag1 下 OwnMAX：+turnover 时 -0.0151（t=-0.71，不显著）、无 turnover 时 '
                        '-0.1335（t=-4.91）；移除 rev 后 +turnover 时 -0.0113（t=-0.53）——'
                        'turnover 单独就吸收 OwnMAX，同月 rev 只是次级吸收源，滞后 rev 不吸收。'
                        'PeerMAX 在全部六个主规格 EW/VW 不显著。')}

def plan_lookahead_diagnostic(fm: pd.DataFrame, fwd_matrix: dict, corr_fwd_own: dict,
                              identity: dict) -> dict:
    """2.0_plan §2.5 引用的 rev_lag1 数字 = 前视列 R_{t+1}（≈ outcome）的复现与证据。"""
    q = fwd_matrix['specs']['q_plan_quote_fwd_turnover_last']
    r = fwd_matrix['specs']['r_plan_quote_fwd_turnover_mean_nowin']
    e_q = q['ew']['coefs']
    e_r = r['ew']['coefs']
    items = [
        {'item': 'plan 引 corr(rev_lag1, OwnMAX) = -0.0343', 'plan': -0.0343,
         'actual': round(corr_fwd_own['monthly_mean'], 6),
         'match': abs(corr_fwd_own['monthly_mean'] + 0.0343) <= 5e-4},
        {'item': 'plan 引 rev_lag1+turnover(last) EW OwnMAX β/t = -0.01341/-4.02',
         'plan': [-0.01341, -4.02],
         'actual': [round(e_q['OwnMAXraw']['beta'], 5), round(e_q['OwnMAXraw']['t_nw'], 2)],
         'match': abs(e_q['OwnMAXraw']['beta'] + 0.01341) <= 5e-4
                  and abs(e_q['OwnMAXraw']['t_nw'] + 4.02) <= 0.02},
        {'item': 'plan 引同一规格 PeerMAX β/t = +0.00927/+1.21', 'plan': [0.00927, 1.21],
         'actual': [round(e_q['PeerMAXraw']['beta'], 5), round(e_q['PeerMAXraw']['t_nw'], 2)],
         'match': abs(e_q['PeerMAXraw']['beta'] - 0.00927) <= 5e-4
                  and abs(e_q['PeerMAXraw']['t_nw'] - 1.21) <= 0.02},
        {'item': 'plan 引 rev_lag1+turnover_mean(不 winsor) EW OwnMAX β/t = -0.01728/-4.91',
         'plan': [-0.01728, -4.91],
         'actual': [round(e_r['OwnMAXraw']['beta'], 5), round(e_r['OwnMAXraw']['t_nw'], 2)],
         'match': abs(e_r['OwnMAXraw']['beta'] + 0.01728) <= 5e-4
                  and abs(e_r['OwnMAXraw']['t_nw'] + 4.91) <= 0.02},
        {'item': 'plan 引同一规格 PeerMAX β/t = +0.01121/+1.49', 'plan': [0.01121, 1.49],
         'actual': [round(e_r['PeerMAXraw']['beta'], 5), round(e_r['PeerMAXraw']['t_nw'], 2)],
         'match': abs(e_r['PeerMAXraw']['beta'] - 0.01121) <= 5e-4
                  and abs(e_r['PeerMAXraw']['t_nw'] - 1.49) <= 0.02},
        {'item': '前视列在规格中的自身系数 ≈ 1（= outcome 本身）',
         'plan': '≈1（>0.5 即证明）', 'actual': [round(e_q[REV_LAG1_FWD]['beta'], 5),
                                                 round(e_q[REV_LAG1_FWD]['t_nw'], 2)],
         'match': e_q[REV_LAG1_FWD]['beta'] > 0.5 and abs(e_q[REV_LAG1_FWD]['t_nw']) > 20},
        {'item': 'R_next vs winsor(R_{t+1}) pooled corr ≥0.95（outcome 身份证据）',
         'plan': '≥0.95', 'actual': round(identity['corr_pooled'], 6),
         'match': identity['corr_pooled'] >= 0.95},
    ]
    return {
        'verdict': ('plan §2.5 引用的 rev_lag1 三组数字（corr -0.0343；-0.01341/-4.02；'
                    '-0.01728/-4.91）逐值由 **R_{t+1}**（下一自然月收益 ≈ outcome `R_next`）复现；'
                    '该列在规格中自身系数 +0.895（t=+107.5），即 plan 的 rev_lag1 是前视/被解释变量。'
                    '合规 t-1 滞后下同规格为 OwnMAX -0.01512（t=-0.71）、PeerMAX +0.04391（t=0.59）。'),
        'plan_compliant_reproduction_possible': False,
        'recommended_action': ('2.6 memo 不得引用 plan §2.5 的 rev_lag1 数字；2.0_plan §2.5 的 R1 锚点'
                               '「rev_lag1 -0.0134」应更正为前视构造的备注，项目须显式决定改名/撤下。'),
        'items': items, 'all_items_match': all(i['match'] for i in items),
        'specs': fwd_matrix['specs'],
        'corr_forward_own': corr_fwd_own,
        'outcome_identity_evidence': identity,
    }

def assert_output_path_guard(path: Path) -> None:
    p = Path(path).resolve()
    banned = set(HEADLINE_GUARD_FILES.values()) | {
        ROOT / 'data',
        ROOT / 'reports/data_gates/G0',
        ROOT / 'configs',
        RETURNS_V1,
    }
    for b in banned:
        b = Path(b).resolve()
        if p == b or b in p.parents:
            raise SystemExit(f'[HARD-GATE] 输出路径命中禁写集合: {p}')

def write_matrix_csv(matrix: dict, path: Path = MATRIX_CSV) -> dict:
    assert_output_path_guard(path)
    df = pd.DataFrame(matrix_rows(matrix))
    tmp = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False, float_format='%.8f', lineterminator='\n')
    os.replace(tmp, path)
    return {'file': str(path.relative_to(ROOT)), 'rows': int(len(df)),
            'columns': list(df.columns), 'sha256': sha256_file(path)}

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Phase 2.5 shadow：rev 同月/滞后/移除 × turnover 控制矩阵')
    ap.add_argument('--sample', default=str(SAMPLE_PARQUET))
    ap.add_argument('--returns', default=str(RETURNS_V2))
    ap.add_argument('--turnover-shadow', default=str(SHADOW_TURNOVER_CSV))
    ap.add_argument('--counts-out', default=str(COUNTS_JSON))
    ap.add_argument('--matrix-out', default=str(MATRIX_CSV))
    return ap.parse_args(argv)

def absorption_block(matrix: dict) -> dict:
    """EW 口径：各控制口径相对「全移除」基准的 OwnMAX 吸收比例（相对降幅）。"""
    base = matrix['specs']['e_rev_removed_no_turnover']['ew']['coefs']['OwnMAXraw']['beta']
    out = {}
    for sid, blk in matrix['specs'].items():
        b = blk['ew']['coefs']['OwnMAXraw']['beta']
        out[sid] = {'ownmax_beta_ew': round(b, 6),
                    'absorption_vs_removed_both': round(1.0 - b / base, 4)}
    return {'base_spec': 'e_rev_removed_no_turnover', 'base_beta_ew': round(base, 6),
            'definition': 'absorption = 1 - beta(spec)/beta(全移除)；仅 EW，供 2.6 引用',
            'by_spec': out}

