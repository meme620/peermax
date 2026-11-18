"""s3_pc_sign_diag_h1h8 的 stats 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.diag.consts import CONTROLS
from lib.diag.consts import FACTORS
from lib.diag.consts import NW
from lib.diag.util import pct
from lib.diag.util import r4
from s3_m1_lib import (  # noqa: E402
    fm_regression, newey_west, param_value, rank_average_groups,
)
import numpy as np
import pandas as pd

def monthly_factors() -> pd.DataFrame:
    fac = pd.read_csv(FACTORS, dtype={'date': str})
    fac['month'] = fac['date'].str[:7]
    f = fac.groupby('month').agg(
        MKT=('MKT_d', lambda s: float(np.prod(1 + s) - 1)),
        SMB=('SMB_d', lambda s: float(np.prod(1 + s) - 1)),
        HML=('HML_d', lambda s: float(np.prod(1 + s) - 1)),
        MOM=('MOM_d', lambda s: float(np.prod(1 + s) - 1)),
        RF=('rf_d', lambda s: float(np.prod(1 + s) - 1))).reset_index()
    return f

def make_samples(m: pd.DataFrame) -> dict:
    pc_ok = m['PC'].notna() & m['R_next'].notna()
    ctrl_ok = m[CONTROLS].notna().all(axis=1)
    reg = m['m1_in_fm_sample'].astype(bool)
    return {
        'registered': m[pc_ok & reg].copy(),
        'paper_like': m[pc_ok].copy(),
        'paper_like_ctrl': m[pc_ok & ctrl_ok].copy(),
        'registered_ctrl': m[pc_ok & reg & ctrl_ok].copy(),
    }

def fm(samp, ycol, xcols, weight_col=None, nw_lag=NW, min_n=None):
    if min_n is None:
        min_n = int(param_value('m1_sample_and_spec_registry')['min_focal_count_per_month'])
    blk, perm = fm_regression(samp, ycol, list(xcols), weight_col=weight_col,
                              min_n=min_n, nw_lag=nw_lag)
    blk['_per_month'] = perm
    return blk

def portfolio_key(samp, keycol, n_groups, wtag, wcol, lag=NW, fmonth=None):
    """与对表脚本同构的组合函数（分组 = 逐月 rank_average，H−L = 最高 − 最低）。"""
    d = samp[['month', keycol, 'R_next', 'market_equity']].copy()
    d = d[d[keycol].notna() & d['R_next'].notna()]
    if wcol:
        d = d[d[wcol].notna() & (d[wcol] > 0)]
    d['grp'] = d.groupby('month', group_keys=False)[keycol].transform(
        lambda s: rank_average_groups(s, n_groups))
    d = d[d['grp'].notna()]
    rows = []
    for mo, sub in d.groupby('month'):
        rec = {'month': mo, 'n': int(len(sub))}
        for g in range(1, n_groups + 1):
            gg = sub[sub['grp'] == g]
            rec[f'g{g}'] = (float(np.average(gg['R_next'], weights=gg[wcol])) if (wcol and not gg.empty)
                            else (float(gg['R_next'].mean()) if not gg.empty else None))
        rows.append(rec)
    t = pd.DataFrame(rows).sort_values('month')
    t['HL'] = t[f'g{n_groups}'] - t['g1']
    nw = newey_west(t['HL'].tolist(), lag=lag)
    out = {'n_months': int(len(t)), 'HL_raw_pct': pct(nw['mean']), 'HL_raw_t_nw': r4(nw['t_nw']),
           'group_mean_pct': {f'Q{g}': pct(float(t[f'g{g}'].mean())) for g in range(1, n_groups + 1)}}
    if fmonth is not None:
        t2 = t.merge(fmonth, on='month', how='left')
        for atag, cols in (('a3', ['MKT', 'SMB', 'HML']), ('a4', ['MKT', 'SMB', 'HML', 'MOM'])):
            ok = t2[cols].notna().all(axis=1)
            ls = (t2['HL'] - t2['RF']).to_numpy(float)[ok.to_numpy()]
            X = np.column_stack([np.ones(len(ls))] + [t2.loc[ok, c].to_numpy(float) for c in cols])
            b, se, tt, T = _ols_hac(ls, X, lag)
            out[f'alpha_{atag}_pct'] = pct(b[0])
            out[f'alpha_{atag}_t'] = r4(tt[0])
    return out

def _ols_hac(y, X, lag):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    T, k = X.shape
    Ae = X * e[:, None]
    S = Ae.T @ Ae / T
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        G = Ae[l:].T @ Ae[:-l] / T
        S = S + w * (G + G.T)
    V = XtXi @ (T * S) @ XtXi
    se = np.sqrt(np.diag(V))
    with np.errstate(divide='ignore', invalid='ignore'):
        return beta, se, beta / se, T

def ctrl_compare(samp, label, ycol='R_next', nw_lag=NW):
    """加控制 FM：报每个控制变量系数（%/单位）与 t。"""
    blk = fm(samp, ycol, ['PC'] + CONTROLS, nw_lag=nw_lag)
    return {f'{label}': {
        'n_months_used': blk['n_months_used'],
        'coefs': {c: {'beta_pct': pct(blk['coefs'][c]['beta']), 't_nw': r4(blk['coefs'][c]['t_nw'])}
                  for c in ['PC'] + CONTROLS},
    }}

