"""p2_4_turnover_shadow 的辅助函数与常量（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.io import atomic_write_text as _atomic_write_text
from pathlib import Path
from s3_m1_lib import (  # noqa: E402
    ROOT, TURNOVER_PKL, assert_m1_inputs_sha, drive_code_to_firm_id, fm_regression,
    param_value, require, sha256_file, winsor_bounds,
)
import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
import warnings

SHADOW_DIR = ROOT / 'data/processed/phase2_shadow'

OUT_CSV = SHADOW_DIR / 'turnover_p2shadow.csv'

META_JSON = SHADOW_DIR / 'turnover_p2shadow.meta.json'

COUNTS_JSON = ROOT / 'artifacts/phase2/2.4_counts.json'

DIFF_CSV = ROOT / 'artifacts/phase2/2.4_turnover_diff.csv'

MATRIX_CSV = ROOT / 'artifacts/phase2/2.4_control_matrix.csv'

SAMPLE_PARQUET = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'

PRICING_TABLE_V2 = ROOT / 'reports/data_gates/G0/s3_m1_pricing_table_v2.json'

HEADLINE_GUARD_FILES = {
    'firm_chars_clean.csv': ROOT / 'data/processed/firm_chars_clean.csv',
    'monthly_pricing_panel_ma_teacher.csv': ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv',
    'm1_common_sample_ma_teacher_v2.parquet': SAMPLE_PARQUET,
}

WINDOW_START, WINDOW_END = '2014-01', '2024-01'

WINDOW_START, WINDOW_END = '2014-01', '2024-01'

FM_WINDOW_START, FM_WINDOW_END = '2015-05', '2023-12'

FM_WINDOW_START, FM_WINDOW_END = '2015-05', '2023-12'

TRIO = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn']

CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage', 'NetworkDegree']

MEAN_COL = 'turnover_mean_win'

MEAN_COL_FM = 'turnover_mean_win_fm'

MEAN_COL_RAW = 'turnover_mean_raw'

LAST_COL_RAW = 'turnover_last_raw'

P2SHADOW_VERSION = 'p2shadow_v1'

SPEC_DEFS = {
    'a_turnover_last': {
        'label': 'a) 登记全控制（turnover=月末最后一日，winsorized 样本列，只读）',
        'x': TRIO + CONTROLS, 'turnover_col': 'turnover',
        'turnover_variant': 'last（headline 现行口径）'},
    'b_turnover_mean': {
        'label': 'b) 月度均值替换（Drive mean/100，panel 宇宙月度 winsor）',
        'x': TRIO + [MEAN_COL if c == 'turnover' else c for c in CONTROLS],
        'turnover_col': MEAN_COL, 'turnover_variant': 'mean（Drive 月度均值）'},
    'c_no_turnover': {
        'label': 'c) 无 turnover（其余控制不变）',
        'x': TRIO + [c for c in CONTROLS if c != 'turnover'],
        'turnover_col': None, 'turnover_variant': 'removed'},
    'd_no_turnover_no_rev': {
        'label': 'd) 无 turnover + 无 rev',
        'x': TRIO + [c for c in CONTROLS if c not in ('turnover', 'rev')],
        'turnover_col': None, 'turnover_variant': 'removed(+rev removed)'},
}

DIAG_SPEC_DEFS = {
    'b2_mean_winsor_fm_universe': ('b2) 诊断：mean winsor 仅在 FM 样本内施加', MEAN_COL_FM),
    'b3_mean_no_winsor': ('b3) 诊断：mean 不做 winsor（原始 Drive 均值/100）', 'turnover_mean_nowin'),
}

def write_json(path: Path, obj) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=1))
    print(f'  -> {path.relative_to(ROOT)}', flush=True)

def load_turnover_wide(path=TURNOVER_PKL) -> pd.DataFrame:
    """只读 Drive 日换手率宽表（index=交易日，columns=Drive code）。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d = pickle.load(open(path, 'rb'))
    d.index = pd.to_datetime(d.index)
    require(isinstance(d, pd.DataFrame) and len(d) > 0, f'turnover pkl 非法: {path}')
    return d

def monthly_frames(wide: pd.DataFrame) -> dict:
    """月末最后一日 / 月度均值 / 月内非空日数（month 为 'YYYY-MM' 字符串索引）。"""
    last = wide.resample('ME').last()
    mean = wide.resample('ME').mean()
    cnt = wide.resample('ME').count()
    out = {}
    for name, f in (('turnover_last', last), ('turnover_mean', mean), ('n_days', cnt)):
        f = f.copy()
        f.index = [c.strftime('%Y-%m') for c in f.index]
        out[name] = f
    return out

def _to_long(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """(month × code) 宽表 -> long[month, code, name]；NaN 格子保留（不丢键）。"""
    try:  # pandas >= 2.1：新 stack 实现默认保留 NaN 格子
        stacked = frame.stack(future_stack=True)
    except TypeError:  # pragma: no cover - 旧 pandas 回退
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stacked = frame.stack(dropna=False)
    o = stacked.rename(name).reset_index()
    o.columns = ['month', 'code', name]
    return o

def build_shadow_frame(wide: pd.DataFrame = None, start: str = WINDOW_START,
                       end: str = WINDOW_END) -> pd.DataFrame:
    """turnover shadow 主表：firm_id/month/turnover_last/turnover_mean/n_days（原始 % 单位）。"""
    if wide is None:
        wide = load_turnover_wide()
    frames = monthly_frames(wide)
    df = _to_long(frames['turnover_last'], 'turnover_last')
    df = df.merge(_to_long(frames['turnover_mean'], 'turnover_mean'), on=['month', 'code'],
                  how='outer')
    df = df.merge(_to_long(frames['n_days'], 'n_days'), on=['month', 'code'], how='outer')
    df = df[(df['month'] >= start) & (df['month'] <= end)].copy()
    df['firm_id'] = df['code'].map(drive_code_to_firm_id)
    require(df['firm_id'].notna().all(), 'shadow turnover: code -> firm_id 映射出现空值')
    df['n_days'] = df['n_days'].fillna(0).astype('int64')
    require(int(df.duplicated(['firm_id', 'month']).sum()) == 0,
            'shadow turnover: (firm_id, month) 出现重复键')
    require(bool((df['n_days'] >= 0).all()), 'shadow turnover: n_days 出现负值')
    both = df['turnover_last'].notna() & df['turnover_mean'].notna()
    require(bool((df.loc[both, 'turnover_last'] >= 0).all())
            and bool((df.loc[both, 'turnover_mean'] >= 0).all()),
            'shadow turnover: turnover_last/turnover_mean 出现负值')
    require(bool(df.loc[df['n_days'] == 0, ['turnover_last', 'turnover_mean']].isna().all().all()),
            'shadow turnover: n_days==0 的行 last/mean 必须为空')
    df = df[['firm_id', 'month', 'turnover_last', 'turnover_mean', 'n_days']]
    return df.sort_values(['month', 'firm_id'], kind='mergesort').reset_index(drop=True)

def window_diff_stats(frame: pd.DataFrame, start: str | None = None,
                      end: str | None = None) -> dict:
    """last vs mean 的 corr / 平均绝对差（pp）/ 分位（pp，原始 % 单位口径下 1pp = 1 个百分点）。"""
    f = frame
    if start is not None:
        f = f[f['month'] >= start]
    if end is not None:
        f = f[f['month'] <= end]
    both = f[f['turnover_last'].notna() & f['turnover_mean'].notna()]
    if len(both) == 0:
        return {'n_rows': int(len(f)), 'n_both': 0}
    a = both['turnover_last'].to_numpy(dtype=float)
    b = both['turnover_mean'].to_numpy(dtype=float)
    diff = a - b
    adiff = np.abs(diff)
    q = np.quantile(adiff, [0.5, 0.9, 0.99])
    sq = np.quantile(diff, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        'n_rows': int(len(f)), 'n_both': int(len(both)),
        'n_n_days_zero': int((f['n_days'] == 0).sum()),
        'corr': float(np.corrcoef(a, b)[0, 1]),
        'mean_abs_diff_pp': float(adiff.mean()), 'median_abs_diff_pp': float(q[0]),
        'p90_abs_diff_pp': float(q[1]), 'p99_abs_diff_pp': float(q[2]),
        'max_abs_diff_pp': float(adiff.max()),
        'mean_signed_diff_pp': float(diff.mean()),
        'signed_q01_pp': float(sq[0]), 'signed_q05_pp': float(sq[1]),
        'signed_q50_pp': float(sq[2]), 'signed_q95_pp': float(sq[3]),
        'signed_q99_pp': float(sq[4]),
        'mean_last_pct': float(a.mean()), 'mean_mean_pct': float(b.mean()),
        'median_n_days': float(both['n_days'].median()) if 'n_days' in both.columns else None,
    }

def yearly_diff_stats(frame: pd.DataFrame, start: str, end: str) -> list:
    rows = []
    f = frame[(frame['month'] >= start) & (frame['month'] <= end)].copy()
    f['year'] = f['month'].str[:4]
    for y, sub in f.groupby('year', sort=True):
        s = window_diff_stats(sub)
        s['year'] = y
        rows.append(s)
    return rows

def build_control_frame(sample_path: Path = SAMPLE_PARQUET, shadow: pd.DataFrame = None) -> tuple:
    """只读 M1 样本 + shadow turnover -> FM 行 frame（含 winsor 变体）与诊断信息。"""
    if shadow is None:
        shadow = build_shadow_frame()
    cols = ['focal_id', 'month', 'm1_in_fm_sample', 'R_next', 'market_equity'] + TRIO + CONTROLS
    df = pd.read_parquet(sample_path, columns=cols)
    require(len(df) == 307503, f'样本 parquet 行数 {len(df)} != 307503')
    m = df.merge(shadow.rename(columns={'firm_id': 'focal_id'})[
        ['focal_id', 'month', 'turnover_last', 'turnover_mean', 'n_days']],
        on=['focal_id', 'month'], how='left')
    m = m.rename(columns={'turnover_last': LAST_COL_RAW, 'turnover_mean': MEAN_COL_RAW})
    # panel 宇宙（parquet 全 307,503 行）月度截面 [lo,hi] winsor —— 与 headline `turnover` 施加方式同构
    lo, hi = winsor_bounds()
    g = m.groupby('month')[MEAN_COL_RAW]
    b_lo = g.transform(lambda s: s.quantile(lo))
    b_hi = g.transform(lambda s: s.quantile(hi))
    m[MEAN_COL] = (m[MEAN_COL_RAW].clip(lower=b_lo, upper=b_hi)) / 100.0
    m['turnover_mean_capped_upper'] = (m[MEAN_COL_RAW] > b_hi).astype(int)
    m['turnover_mean_capped_lower'] = (m[MEAN_COL_RAW] < b_lo).astype(int)
    fm = m[m['m1_in_fm_sample'].astype(bool)].copy()
    # 敏感性变体 A：只在 FM 样本内 winsor（供对比，不进入主规格）
    gf = fm.groupby('month')[MEAN_COL_RAW]
    fm[MEAN_COL_FM] = (fm[MEAN_COL_RAW].clip(lower=gf.transform(lambda s: s.quantile(lo)),
                                             upper=gf.transform(lambda s: s.quantile(hi)))) / 100.0
    # 敏感性变体 B：完全不做 winsor
    fm['turnover_mean_nowin'] = fm[MEAN_COL_RAW] / 100.0
    # 一致性桥：panel-winsor(raw last/100) 必须逐值等于样本 turnover 列（headline 只读桥）
    gl = m.groupby('month')[LAST_COL_RAW]
    last_panel_win = (m[LAST_COL_RAW].clip(lower=gl.transform(lambda s: s.quantile(lo)),
                                           upper=gl.transform(lambda s: s.quantile(hi)))) / 100.0
    bridge = (m['turnover'] - last_panel_win).abs()
    diag = {
        'sample_rows': int(len(df)), 'fm_rows': int(len(fm)),
        'winsorization_pct': [lo, hi],
        'fm_turnover_mean_raw_coverage': float(fm[MEAN_COL_RAW].notna().mean()),
        'fm_turnover_last_coverage': float(fm[LAST_COL_RAW].notna().mean()),
        'fm_turnover_sample_coverage': float(fm['turnover'].notna().mean()),
        'mean_capped_rows_panel': int(m['turnover_mean_capped_upper'].sum()
                                      + m['turnover_mean_capped_lower'].sum()),
        'mean_capped_rows_fm': int(fm['turnover_mean_capped_upper'].sum()
                                   + fm['turnover_mean_capped_lower'].sum()),
        'last_bridge_max_abs_diff_vs_sample_turnover': float(bridge.max()),
        'last_bridge_rows_differing': int((bridge > 1e-12).sum()),
        'mean_win_panel_vs_fm_rows_differing': int(
            (fm[MEAN_COL].notna() & fm[MEAN_COL_FM].notna()
             & ((fm[MEAN_COL] - fm[MEAN_COL_FM]).abs() > 1e-12)).sum()),
        'mean_win_max': float(fm[MEAN_COL].max()),
        'last_win_max': float(fm['turnover'].max()),
    }
    return fm, diag

def run_spec_matrix(fm: pd.DataFrame, spec_defs: dict = None) -> dict:
    """四规格 × EW/VW 的 FM（min_n=100，NW lag=6，与 headline 一致）。"""
    spec_defs = spec_defs or SPEC_DEFS
    min_n = int(param_value('m1_sample_and_spec_registry')['min_focal_count_per_month'])
    nw_lag = int(param_value('nw_lag'))
    out = {'min_n': min_n, 'nw_lag': nw_lag, 'n_rows': int(len(fm)),
           'months': [str(fm['month'].min()), str(fm['month'].max())], 'specs': {}}
    for sid, sd in spec_defs.items():
        xcols = list(sd['x'])
        blk = {'label': sd['label'], 'x': xcols, 'turnover_col': sd.get('turnover_col')}
        for wname, wcol in (('ew', None), ('vw', 'market_equity')):
            res, per_month = fm_regression(fm, 'R_next', xcols, weight_col=wcol,
                                           min_n=min_n, nw_lag=nw_lag)
            used = [d for d in per_month if d.get('used')]
            coefs = {c: {'beta': res['coefs'][c]['beta'], 'se_nw': res['coefs'][c]['se_nw'],
                         't_nw': res['coefs'][c]['t_nw']} for c in xcols}
            blk[wname] = {
                'T': res['n_months_used'], 'n_frame': int(len(fm)),
                'n_used': int(sum(d['n'] for d in used)),
                'coefs': coefs,
                'per_month_n': {'min': int(min(d['n'] for d in per_month)),
                                'median': float(np.median([d['n'] for d in per_month])),
                                'max': int(max(d['n'] for d in per_month))},
            }
        out['specs'][sid] = blk
    return out

def headline_comparison(matrix: dict, table_path: Path = PRICING_TABLE_V2,
                        tol_beta: float = 1e-4, tol_t: float = 1e-3) -> dict:
    """规格 a 与 headline 主表（s3_m1_pricing_table_v2.json, ma_teacher）逐值对比（只读）。"""
    table = json.loads(table_path.read_text(encoding='utf-8'))
    mt = table['main']['ma_teacher']
    spec_a = matrix['specs']['a_turnover_last']
    cmp_rows = []
    all_match = True
    for wname in ('ew', 'vw'):
        for term in ('PeerMAXraw', 'OwnMAXraw', 'turnover', 'PeerReturn'):
            exp = mt[wname]['coefs'][term]
            act = spec_a[wname]['coefs'][term]
            ok = (abs(exp['beta'] - act['beta']) <= tol_beta
                  and abs(exp['t_nw'] - act['t_nw']) <= tol_t)
            all_match &= bool(ok)
            cmp_rows.append({'weight': wname, 'term': term,
                             'headline_beta': exp['beta'], 'shadow_beta': act['beta'],
                             'headline_t': exp['t_nw'], 'shadow_t': act['t_nw'],
                             'match': bool(ok)})
    return {'source': str(table_path.relative_to(ROOT)),
            'headline_n_rows': mt['n_rows'], 'headline_fm_window': mt['fm_window'],
            'headline_months': mt['ew']['n_months_used'],
            'tol_beta': tol_beta, 'tol_t': tol_t,
            'rows': cmp_rows, 'all_match': bool(all_match)}

def write_shadow_csv(df: pd.DataFrame, path: Path = OUT_CSV) -> dict:
    assert_output_path_guard(path)
    tmp = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in ('turnover_last', 'turnover_mean'):
        out[c] = out[c].round(8)
    out.to_csv(tmp, index=False, float_format='%.8f', lineterminator='\n')
    os.replace(tmp, path)
    return {'file': str(path.relative_to(ROOT)), 'rows': int(len(df)),
            'columns': list(df.columns), 'sha256': sha256_file(path)}

def write_diff_csv(yearly: list, path: Path = DIFF_CSV) -> dict:
    cols = ['year', 'n_rows', 'n_both', 'n_n_days_zero', 'corr', 'mean_abs_diff_pp',
            'median_abs_diff_pp', 'p90_abs_diff_pp', 'p99_abs_diff_pp', 'max_abs_diff_pp',
            'mean_signed_diff_pp', 'mean_last_pct', 'mean_mean_pct', 'median_n_days']
    df = pd.DataFrame(yearly)[cols]
    tmp = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False, float_format='%.6f', lineterminator='\n')
    os.replace(tmp, path)
    return {'file': str(path.relative_to(ROOT)), 'rows': int(len(df)),
            'sha256': sha256_file(path)}

def matrix_rows(matrix: dict) -> list:
    rows = []
    for sid, blk in matrix['specs'].items():
        for wname in ('ew', 'vw'):
            w = blk[wname]
            pm = w['per_month_n']
            for term in ('PeerMAXraw', 'OwnMAXraw') + \
                    ((blk['turnover_col'],) if blk['turnover_col'] else ()):
                c = w['coefs'][term]
                rows.append({'spec_id': sid, 'weight': wname, 'term': term,
                             'beta': c['beta'], 'se_nw': c['se_nw'], 't_nw': c['t_nw'],
                             'T_months': w['T'], 'n_rows': w['n_used'],
                             'n_per_month_min': pm['min'], 'n_per_month_median': pm['median'],
                             'n_per_month_max': pm['max']})
    return rows

def write_matrix_csv(matrix: dict, path: Path = MATRIX_CSV) -> dict:
    df = pd.DataFrame(matrix_rows(matrix))
    tmp = path.with_name(path.name + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False, float_format='%.8f', lineterminator='\n')
    os.replace(tmp, path)
    return {'file': str(path.relative_to(ROOT)), 'rows': int(len(df)),
            'sha256': sha256_file(path)}

def assert_output_path_guard(path: Path) -> None:
    p = Path(path).resolve()
    banned = set(HEADLINE_GUARD_FILES.values()) | {
        ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv',
        ROOT / 'reports/data_gates/G0',
        ROOT / 'configs',
    }
    for b in banned:
        b = Path(b).resolve()
        if p == b or b in p.parents:
            raise SystemExit(f'[HARD-GATE] 输出路径命中 headline 禁写集合: {p}')

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Phase 2.4 shadow：turnover last vs mean + 四规格 FM')
    ap.add_argument('--turnover-pkl', default=str(TURNOVER_PKL))
    ap.add_argument('--sample', default=str(SAMPLE_PARQUET))
    ap.add_argument('--out', default=str(OUT_CSV))
    ap.add_argument('--meta-out', default=str(META_JSON))
    ap.add_argument('--counts-out', default=str(COUNTS_JSON))
    ap.add_argument('--diff-out', default=str(DIFF_CSV))
    ap.add_argument('--matrix-out', default=str(MATRIX_CSV))
    return ap.parse_args(argv)

def _to_long_asis(wide: pd.DataFrame) -> pd.DataFrame:
    """full-pkl 差异统计用的最小长表（只含 last/mean/n_days，不切片）。"""
    frames = monthly_frames(wide)
    df = _to_long(frames['turnover_last'], 'turnover_last')
    df = df.merge(_to_long(frames['turnover_mean'], 'turnover_mean'), on=['month', 'code'])
    df = df.merge(_to_long(frames['n_days'], 'n_days'), on=['month', 'code'])
    df['firm_id'] = df['code'].map(drive_code_to_firm_id)
    df['n_days'] = df['n_days'].fillna(0).astype('int64')
    return df

def audit_l8_block(matrix: dict, cmp_headline: dict) -> dict:
    """审计 L-8 判定表（reports/audits/peermax_audit_item_review_20261027.md）的逐值复现。"""
    def pick(sid):
        e = matrix['specs'][sid]['ew']['coefs']
        return {'PeerMAX_beta': round(e['PeerMAXraw']['beta'], 4),
                'PeerMAX_t': round(e['PeerMAXraw']['t_nw'], 2),
                'OwnMAX_beta': round(e['OwnMAXraw']['beta'], 4),
                'OwnMAX_t': round(e['OwnMAXraw']['t_nw'], 2),
                'turnover_beta': round(e[matrix['specs'][sid]['turnover_col']]['beta'], 4),
                'turnover_t': round(e[matrix['specs'][sid]['turnover_col']]['t_nw'], 2)}
    expected_last = {'PeerMAX_beta': 0.0343, 'PeerMAX_t': 0.47, 'OwnMAX_beta': -0.0028,
                     'OwnMAX_t': -0.12, 'turnover_beta': -0.3987, 'turnover_t': -9.64}
    expected_mean = {'PeerMAX_beta': 0.0569, 'PeerMAX_t': 0.80, 'OwnMAX_beta': 0.0119,
                     'OwnMAX_t': 0.62, 'turnover_beta': -0.4163, 'turnover_t': -8.78}
    actual_last, actual_mean = pick('a_turnover_last'), pick('b_turnover_mean')

    def close(exp, act, tol_b=5e-5, tol_t=5e-3):
        for k, v in exp.items():
            tol = tol_t if k.endswith('_t') else tol_b
            if abs(v - act[k]) > tol:
                return False
        return True

    return {
        'reference': 'reports/audits/peermax_audit_item_review_20261027.md L-8 判定表',
        'last_row_expected': expected_last, 'last_row_actual': actual_last,
        'mean_row_expected': expected_mean, 'mean_row_actual': actual_mean,
        'match': bool(cmp_headline['all_match'] and close(expected_last, actual_last)
                      and close(expected_mean, actual_mean)),
    }

def plan_replication(stats_plan_window: dict, matrix: dict, diag_matrix: dict,
                     cmp_headline: dict, diag: dict) -> dict:
    """与 2.0_plan §2.4 的 R1 / 现状数字逐项对账（只报差异，不改 headline）。

    发现：plan §2.4「现状」的 mean 行（OwnMAX +0.00583/t=0.30、PeerMAX +0.05377/t=0.74）
    = **未 winsor** 的 Drive 均值变体（本文件 diagnostic b3 逐值复现）；主口径 b
    （registry [0.01,0.99] 月度 winsor，与 headline turnover 施加方式同构）实测
    OwnMAX +0.01188/t=0.62、PeerMAX +0.05694/t=0.80，与审计 L-8 判定表逐值一致。
    """
    a_ew = matrix['specs']['a_turnover_last']['ew']['coefs']
    b_ew = matrix['specs']['b_turnover_mean']['ew']['coefs']
    c_ew = matrix['specs']['c_no_turnover']['ew']['coefs']
    d_ew = matrix['specs']['d_no_turnover_no_rev']['ew']['coefs']
    b3_ew = diag_matrix['specs']['b3_mean_no_winsor']['ew']['coefs']

    def bt(c):
        return [round(c['beta'], 5), round(c['t_nw'], 2)]

    def near(c, beta, t, tol_b=5e-4, tol_t=0.005):
        return bool(abs(c['beta'] - beta) <= tol_b and abs(c['t_nw'] - t) <= tol_t)

    items = [
        {'item': '匹配行数（2014-01..2024-01）', 'plan': 446301,
         'actual': stats_plan_window['n_both'], 'match': stats_plan_window['n_both'] == 446301},
        {'item': 'corr(last, mean)（R1 区间 0.75–0.81）', 'plan': 0.780,
         'actual': round(stats_plan_window['corr'], 6),
         'match': 0.75 <= stats_plan_window['corr'] <= 0.81},
        {'item': '平均绝对差 pp（R1 区间 1.0–1.4）', 'plan': 1.257,
         'actual': round(stats_plan_window['mean_abs_diff_pp'], 4),
         'match': 1.0 <= stats_plan_window['mean_abs_diff_pp'] <= 1.4},
        {'item': 'FM 覆盖（要求 ≥99%/100%）', 'plan': 1.0,
         'actual': round(diag['fm_turnover_mean_raw_coverage'], 6),
         'match': diag['fm_turnover_mean_raw_coverage'] >= 0.999},
        {'item': '规格 a（last）PeerMAX β/t', 'plan': [0.03427, 0.47],
         'actual': bt(a_ew['PeerMAXraw']), 'match': bool(cmp_headline['all_match'])},
        {'item': '规格 a（last）OwnMAX β/t', 'plan': [-0.00282, -0.12],
         'actual': bt(a_ew['OwnMAXraw']), 'match': bool(cmp_headline['all_match'])},
        {'item': '规格 b 主口径（winsorized mean）OwnMAX β/t（plan 现状行列的是 b3，非本行）',
         'plan': [0.00583, 0.30], 'actual': bt(b_ew['OwnMAXraw']),
         'match': near(b_ew['OwnMAXraw'], 0.011885, 0.624, 5e-4, 0.005),
         'note': '主口径 winsorized mean 与审计 L-8 表一致（+0.0119/t=0.62）'},
        {'item': '规格 b 主口径（winsorized mean）PeerMAX β/t（plan 现状行列的是 b3，非本行）',
         'plan': [0.05377, 0.74], 'actual': bt(b_ew['PeerMAXraw']),
         'match': near(b_ew['PeerMAXraw'], 0.056943, 0.796, 5e-4, 0.005),
         'note': '主口径 winsorized mean 与审计 L-8 表一致（+0.0569/t=0.80）'},
        {'item': '诊断 b3（无 winsor mean）OwnMAX β/t = plan 现状行', 'plan': [0.00583, 0.30],
         'actual': bt(b3_ew['OwnMAXraw']), 'match': near(b3_ew['OwnMAXraw'], 0.00583, 0.30)},
        {'item': '诊断 b3（无 winsor mean）PeerMAX β/t = plan 现状行', 'plan': [0.05377, 0.74],
         'actual': bt(b3_ew['PeerMAXraw']), 'match': near(b3_ew['PeerMAXraw'], 0.05377, 0.74)},
        {'item': '规格 b 方向（两变体：PeerMAX>0 且 |t|<1.96；OwnMAX≈0 且 |t|<1.96）',
         'plan': '仍不显著（方向不变）',
         'actual': {'b_PeerMAX_t': round(b_ew['PeerMAXraw']['t_nw'], 3),
                    'b_OwnMAX_t': round(b_ew['OwnMAXraw']['t_nw'], 3),
                    'b3_PeerMAX_t': round(b3_ew['PeerMAXraw']['t_nw'], 3),
                    'b3_OwnMAX_t': round(b3_ew['OwnMAXraw']['t_nw'], 3)},
         'match': bool(abs(b_ew['PeerMAXraw']['t_nw']) < 1.96
                       and abs(b_ew['OwnMAXraw']['t_nw']) < 1.96
                       and abs(b3_ew['PeerMAXraw']['t_nw']) < 1.96
                       and abs(b3_ew['OwnMAXraw']['t_nw']) < 1.96
                       and b_ew['PeerMAXraw']['beta'] > 0 and b3_ew['PeerMAXraw']['beta'] > 0)},
        {'item': '规格 c（无 turnover）OwnMAX β/t', 'plan': [-0.10550, -3.80],
         'actual': bt(c_ew['OwnMAXraw']), 'match': near(c_ew['OwnMAXraw'], -0.10550, -3.80, 5e-5, 0.005)},
        {'item': '规格 d（无 turnover+无 rev）OwnMAX β/t', 'plan': [-0.13586, -4.97],
         'actual': bt(d_ew['OwnMAXraw']), 'match': near(d_ew['OwnMAXraw'], -0.13586, -4.97, 5e-5, 0.005)},
    ]
    return {'items': items,
            'all_items_match': all(i['match'] for i in items),
            'plan_mean_row_reproduced_by_no_winsor_variant': bool(
                near(b3_ew['OwnMAXraw'], 0.00583, 0.30) and near(b3_ew['PeerMAXraw'], 0.05377, 0.74)),
            'summary': ('plan §2.4 现状的 mean 行 = 未 winsor 均值变体（b3，逐值复现）；'
                        '主口径 winsorized mean（b）与审计 L-8 表逐值一致。两种变体方向相同、'
                        'PeerMAX/OwnMAX 均不显著。')}

