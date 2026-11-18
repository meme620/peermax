"""s3_m1_lib 的 stats 实现（逐字搬移，逻辑未改）。"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))


from lib.m1.contracts import _registry_fail
from lib.m1.contracts import _registry_param
import math
import numpy as np
import pandas as pd

def winsor_bounds(registry_path=None) -> tuple:
    """registry winsorization_pct -> (lo, hi)；长度 2、0<=lo<hi<=1，否则 SystemExit(2)。"""
    raw = _registry_param('winsorization_pct', registry_path)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        _registry_fail(f'winsorization_pct 必须是长度 2 序列，当前={raw!r}')
    try:
        lo, hi = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        _registry_fail(f'winsorization_pct 含非数值: {raw!r}')
    if not (0.0 <= lo < hi <= 1.0):
        _registry_fail(f'winsorization_pct 需满足 0<=lo<hi<=1，当前=({lo}, {hi})')
    return lo, hi

def winsorize_bounds_by_month(df: pd.DataFrame, col: str, lo: float = 0.01, hi: float = 0.99,
                              month_col: str = 'month'):
    """月度截面 [lo, hi] 分位边界（在该 frame 的宇宙上计算；缺月返回 NaN 边界 = 不裁剪）。"""
    g = df.groupby(month_col, sort=False)[col]
    return g.transform(lambda s: s.quantile(lo)), g.transform(lambda s: s.quantile(hi))

def clip_by_month(df: pd.DataFrame, cols, lo: float = 0.01, hi: float = 0.99,
                  month_col: str = 'month') -> pd.DataFrame:
    """I1 修正：对指定列做与主规格同规则的月度截面 [lo,hi] winsorize（宇宙 = 传入 frame）。

    用于 A9 中由 aggregate_edges 重算的权重变体 / placebo 网络的 PeerMAXraw 与 PeerReturn：
    修正前这些列是未 winsorize 值，与主规格（面板构造期已 winsorize）口径不对称。
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        b_lo, b_hi = winsorize_bounds_by_month(df, c, lo=lo, hi=hi, month_col=month_col)
        df[c] = df[c].clip(lower=b_lo, upper=b_hi)
    return df

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def newey_west(series, lag: int = 6):
    """系数序列的均值 + Newey-West(Bartlett) SE。返回 dict。"""
    a = np.asarray([v for v in series if v is not None and np.isfinite(v)], dtype=float)
    T = len(a)
    out = {'T': T, 'mean': None, 'se_nw': None, 't_nw': None, 'se_iid': None, 't_iid': None}
    if T < 2:
        return out
    m = float(a.mean())
    d = a - m
    gamma0 = float(np.dot(d, d) / T)
    nw_var = gamma0
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        cov = float(np.dot(d[l:], d[:-l]) / T)
        nw_var += 2.0 * w * cov
    se_nw = math.sqrt(max(nw_var, 0.0) / T)
    se_iid = math.sqrt(float(d.var(ddof=1)) / T) if T > 1 else float('nan')
    out.update({'mean': m,
                'se_nw': se_nw if np.isfinite(se_nw) else None,
                't_nw': (m / se_nw) if se_nw > 0 else None,
                'se_iid': se_iid if np.isfinite(se_iid) else None,
                't_iid': (m / se_iid) if se_iid and se_iid > 0 else None})
    return out

def one_sided_p(t, direction: str = 'negative'):
    """H0: beta >= 0 vs H1: beta < 0（主方向 beta<0）-> p = Phi(t)。"""
    if t is None or not np.isfinite(t):
        return None
    return norm_cdf(t) if direction == 'negative' else 1.0 - norm_cdf(t)

def two_sided_p(t):
    if t is None or not np.isfinite(t):
        return None
    return 2.0 * (1.0 - norm_cdf(abs(t)))

def tost_equivalence(diff_series, bound: float, alpha: float, lag: int = 6):
    """TOST：H0 |d| >= bound vs H1 |d| < bound（两单侧 p 均 < alpha 记等价）。"""
    nw = newey_west(diff_series, lag=lag)
    d, se = nw['mean'], nw['se_nw']
    out = {'mean_diff': d, 'se_nw': se, 'bound': bound, 'alpha': alpha,
           'p_lower': None, 'p_upper': None, 'equivalent': False,
           'status': 'NOT_COMPUTABLE'}
    if d is None or not se or se <= 0:
        return out
    t1 = (d + bound) / se          # H0: d <= -bound
    t2 = (bound - d) / se          # H0: d >= +bound
    p1 = 1.0 - norm_cdf(t1)
    p2 = 1.0 - norm_cdf(t2)
    out.update({'p_lower': p1, 'p_upper': p2,
                'equivalent': bool(p1 < alpha and p2 < alpha),
                'status': 'PASS' if (p1 < alpha and p2 < alpha) else 'FAIL'})
    return out

def winsorize_by_month(df: pd.DataFrame, cols, lo: float = 0.01, hi: float = 0.99,
                       month_col: str = 'month') -> pd.DataFrame:
    """月度截面 winsorize（只动给定列；缺失保持缺失）。"""
    df = df.copy()
    g = df.groupby(month_col, sort=False)
    for c in cols:
        if c not in df.columns:
            continue
        q_lo = g[c].transform(lambda s: s.quantile(lo))
        q_hi = g[c].transform(lambda s: s.quantile(hi))
        df[c] = df[c].clip(lower=q_lo, upper=q_hi)
    return df

def treatment_only_mask(df: pd.DataFrame) -> pd.Series:
    """treatment-only eligible 宇宙（outcome-blind）：
    通过全部 treatment/controls 可得性屏幕的行（exclusion 顺序中 outcome 屏之前）；
    v2 样本的 `m1_exclusion_reason ∈ {'', 'no_outcome_next_month'}` 即该宇宙。"""
    if 'm1_exclusion_reason' not in df.columns:
        return pd.Series(True, index=df.index)
    return df['m1_exclusion_reason'].fillna('').isin(['', 'no_outcome_next_month'])

def rank_average_groups(x: pd.Series, n_groups: int) -> pd.Series:
    """冻结 tie rule：rank(method='average') 后按 ceil(rank/N*g) 分组（跨 cutoff 的 tie 同组）。"""
    r = x.rank(method='average')
    n = r.notna().sum()
    if n == 0:
        return pd.Series(np.nan, index=x.index)
    g = np.ceil(r / n * n_groups)
    g = g.clip(lower=1, upper=n_groups)
    g[x.isna()] = np.nan
    return g.astype(float)

def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell restricted cubic spline 基（含线性项 + k-1 个非线性项）。"""
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    nk = len(k)
    out = [x]
    for j in range(nk - 2):
        def cube(t):
            return np.where(t > 0, t ** 3, 0.0)
        term = (cube(x - k[j])
                - cube(x - k[nk - 2]) * (k[nk - 1] - k[j]) / (k[nk - 1] - k[nk - 2])
                + cube(x - k[nk - 1]) * (k[nk - 2] - k[j]) / (k[nk - 1] - k[nk - 2]))
        out.append(term / (k[nk - 1] - k[1]) ** 2)
    return np.column_stack(out)

def _ols(X: np.ndarray, y: np.ndarray, w: np.ndarray | None):
    n, p = X.shape
    if w is None:
        Xw, yw = X, y
    else:
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
    XtX = Xw.T @ Xw
    rank = int(np.linalg.matrix_rank(XtX))
    if rank < p:
        return None
    XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ (Xw.T @ yw)
    resid = yw - Xw @ beta
    df = n - p
    if df <= 0:
        return None
    sigma2 = float(resid @ resid / df)
    cov = sigma2 * XtXi
    finite = bool(np.isfinite(beta).all() and np.isfinite(cov).all() and np.isfinite(sigma2))
    return {'beta': beta, 'cov': cov, 'rank': rank, 'df': df, 'n': n,
            'sigma2': sigma2, 'finite': finite}

def fm_regression(df: pd.DataFrame, ycol: str, xcols, month_col: str = 'month',
                  weight_col: str | None = None, min_n: int = 100, nw_lag: int = 6,
                  direction: str = 'negative'):
    """逐月横截面回归 -> 系数序列 -> NW 汇总。返回 (summary, per_month)。"""
    coefs = {c: [] for c in xcols}
    coefs['const'] = []
    per_month = []
    n_months_used = 0
    for m, sub in df.groupby(month_col, sort=True):
        sub = sub.dropna(subset=[ycol] + list(xcols))
        if weight_col:
            sub = sub[sub[weight_col].notna() & (sub[weight_col] > 0)]
        n = len(sub)
        if n < min_n:
            per_month.append({'month': m, 'n': n, 'used': False, 'reason': 'below_min_n'})
            continue
        y = sub[ycol].to_numpy(dtype=float)
        X = np.column_stack([sub[c].to_numpy(dtype=float) for c in xcols]
                            + [np.ones(n)])
        w = sub[weight_col].to_numpy(dtype=float) if weight_col else None
        fit = _ols(X, y, w)
        if fit is None or not fit['finite']:
            per_month.append({'month': m, 'n': n, 'used': False,
                              'reason': 'rank_or_df_or_covariance_failure'})
            continue
        n_months_used += 1
        for j, c in enumerate(list(xcols) + ['const']):
            coefs[c].append(float(fit['beta'][j]))
        per_month.append({'month': m, 'n': n, 'used': True, 'rank': fit['rank'],
                          'n_params': len(xcols) + 1, 'residual_df': fit['df'],
                          'covariance_status': 'finite',
                          'rank_status': 'FULL' if fit['rank'] == len(xcols) + 1 else 'RANK_DEFICIENT',
                          'beta': {c: float(fit['beta'][j]) for j, c in enumerate(list(xcols) + ['const'])}})
    summary = {'n_months_used': n_months_used, 'min_n': min_n, 'weight_col': weight_col,
               'nw_lag': nw_lag, 'coefs': {}, 'per_month': per_month}
    for c in list(xcols) + ['const']:
        nw = newey_west(coefs[c], lag=nw_lag)
        s = {'beta': nw['mean'], 'se_nw': nw['se_nw'], 't_nw': nw['t_nw'],
             'se_iid': nw['se_iid'], 't_iid': nw['t_iid'], 'n_months': nw['T'],
             'p_one_sided': one_sided_p(nw['t_nw'], direction),
             'p_two_sided': two_sided_p(nw['t_nw'])}
        if s['beta'] is not None and s['se_nw']:
            s['ci95_low'] = s['beta'] - 1.96 * s['se_nw']
            s['ci95_high'] = s['beta'] + 1.96 * s['se_nw']
        summary['coefs'][c] = s
    return summary, per_month

def nw_cov_matrix(vecs, lag: int = 6):
    """系数向量序列的均值 + Newey-West 协方差矩阵（Bartlett 权）。"""
    A = np.asarray([v for v in vecs if v is not None and np.isfinite(v).all()], dtype=float)
    T, k = A.shape
    if T < 2:
        return None, None, T
    mean = A.mean(axis=0)
    D = A - mean
    S = (D.T @ D) / T
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        G = (D[l:].T @ D[:-l]) / T
        S = S + w * (G + G.T)
    cov = S / T
    return mean, cov, T

def joint_wald(vecs, idx, lag: int = 6):
    """对系数向量子集做联合 Wald 检验（H0: 子集全部为 0）。"""
    mean, cov, T = nw_cov_matrix(vecs, lag=lag)
    if mean is None:
        return {'status': 'NOT_COMPUTABLE'}
    sub = cov[np.ix_(idx, idx)]
    b = mean[idx]
    try:
        inv = np.linalg.pinv(sub)
        stat = float(b @ inv @ b)
    except Exception:
        return {'status': 'NOT_COMPUTABLE'}
    from math import erfc, sqrt
    k = len(idx)
    # Wilson-Hilferty 近似 p 值（chi2_k 上尾）
    if stat <= 0:
        p = 1.0
    else:
        z = ((stat / k) ** (1.0 / 3.0) - (1 - 2.0 / (9 * k))) / np.sqrt(2.0 / (9 * k))
        p = 0.5 * erfc(z / sqrt(2.0))
    return {'status': 'COMPUTED', 'wald_chi2': stat, 'df': int(k), 'p_value': float(p),
            'T': int(T), 'note': 'NW 协方差 + Wilson-Hilferty 近似（诊断用）'}

def vif_table(df: pd.DataFrame, xcols, month_col: str = 'month') -> dict:
    """逐月 VIF 的均值（对设计矩阵列，含常数项以外）；条件数同时给出。"""
    vifs = {c: [] for c in xcols}
    conds = []
    for m, sub in df.groupby(month_col, sort=True):
        sub = sub.dropna(subset=list(xcols))
        if len(sub) < 50:
            continue
        X = sub[list(xcols)].to_numpy(dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        Xs = np.column_stack([Xs, np.ones(len(Xs))])   # 常数项单列（不标准化，避免零列退化）
        sv = np.linalg.svd(Xs, compute_uv=False)
        if sv[-1] > 0:
            conds.append(float(sv[0] / sv[-1]))
        for j, c in enumerate(xcols):
            others = [k for k in range(X.shape[1]) if k != j]
            Xo = Xs[:, others]
            fit = _ols(Xo, Xs[:, j], None)
            if fit is None:
                vifs[c].append(np.nan)
                continue
            resid = Xs[:, j] - Xo @ fit['beta']
            ss_tot = float(((Xs[:, j] - Xs[:, j].mean()) ** 2).sum())
            r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
            vifs[c].append(1.0 / (1.0 - r2) if np.isfinite(r2) and r2 < 1 else np.inf)
    return {'vif_mean': {c: (float(np.nanmean(v)) if len(v) else None) for c, v in vifs.items()},
            'vif_max': {c: (float(np.nanmax(v)) if len(v) else None) for c, v in vifs.items()},
            'condition_number_mean': float(np.mean(conds)) if conds else None,
            'condition_number_max': float(np.max(conds)) if conds else None,
            'n_months': len(conds)}

