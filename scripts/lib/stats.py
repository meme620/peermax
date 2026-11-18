"""Newey-West 工具。`nw_t` 与 `nw_ols` 逐字搬自 s3a_cf_mom_replication / s3a_cf_mom_alpha。"""

from __future__ import annotations

import numpy as np


def nw_t(x, L=1):
    """Newey-West t 统计（滞后 L 期自协方差修正）。v2：自协方差分母统一为 T。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 3:
        return np.nan
    mu = x.mean()
    se2 = np.sum((x - mu) ** 2) / T / T
    for l in range(1, min(L, T - 1) + 1):
        gamma_l = np.sum((x[:-l] - mu) * (x[l:] - mu)) / T
        se2 += 2 * (1 - l / (L + 1)) * gamma_l / T
    return mu / np.sqrt(se2)


def nw_ols(X, y, L=1):
    """OLS + Newey-West (L) 协方差。返回 beta, se, t。"""
    T = len(y)
    XpX_inv = np.linalg.inv(X.T @ X)
    beta = XpX_inv @ (X.T @ y)
    eps = y - X @ beta
    S0 = sum(eps[t] ** 2 * np.outer(X[t], X[t]) for t in range(T))
    S = S0.copy()
    for l in range(1, min(L, T - 1) + 1):
        Sl = sum(eps[t] * eps[t + l] * (np.outer(X[t], X[t + l]) + np.outer(X[t + l], X[t]))
                 for t in range(T - l))
        S += (1 - l / (L + 1)) * Sl
    V = XpX_inv @ S @ XpX_inv
    se = np.sqrt(np.diag(V))
    return beta, se, beta / se
