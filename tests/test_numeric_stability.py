"""数值稳定性门：零方差列与近共线月份不得进入 FM 系数序列。

修复前的两处缺陷：
  1. matrix_rank 在未归一化矩阵上判定，列量纲差 10 个数量级时把健康满秩矩阵误判成秩 1；
  2. 近共线（满秩但条件数爆表）月份照常估计，伪逆给出爆炸 beta 并计入均值。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_lib as pl  # noqa: E402


def _panel(eps, n=400, months=("2020-01", "2020-02"), seed=11):
    rows = []
    for m in months:
        rr = np.random.default_rng(seed)
        x1 = rr.normal(size=n)
        x2 = x1 + eps * rr.normal(size=n)
        y = 0.01 + 0.02 * x1 + rr.normal(0, 1e-3, n)
        for i in range(n):
            rows.append({"month": m, "y": y[i], "x1": x1[i], "x2": x2[i]})
    return pd.DataFrame(rows)


def test_scale_heterogeneous_full_rank_is_not_rejected():
    """列量纲差 1e10 时秩必须仍判满：旧实现会误判成秩 1 并整月丢弃。"""
    rr = np.random.default_rng(5)
    n = 300
    x1, x2 = rr.normal(size=n), rr.normal(size=n)
    lev = 1e10 * (1 + 0.1 * rr.normal(size=n))
    X = np.column_stack([x1, x2, lev, np.ones(n)])
    fit = pl._ols(X, 0.01 + 0.02 * x1 - 0.01 * x2, None)
    assert fit is not None and fit.get("status") is None
    assert fit["rank"] == 4


def test_zero_variance_column_is_flagged_not_estimated():
    rr = np.random.default_rng(3)
    X = np.column_stack([rr.normal(size=200), np.zeros(200), np.ones(200)])
    fit = pl._ols(X, rr.normal(size=200), None)
    assert isinstance(fit, dict) and fit.get("status") == "ZERO_VARIANCE"


def test_near_collinear_month_is_rejected():
    res, pm = pl.fm_regression(_panel(1e-11), "y", ["x1", "x2"], min_n=50, nw_lag=1)
    assert res["n_months_used"] == 0
    assert {d.get("reason") for d in pm} == {"ill_conditioned"}


def test_healthy_month_is_estimated():
    res, _ = pl.fm_regression(_panel(1e-3), "y", ["x1", "x2"], min_n=50, nw_lag=1)
    assert res["n_months_used"] == 2
    assert abs(res["coefs"]["x1"]["beta"] - 0.02) < 5e-3


def test_condition_threshold_is_far_above_real_design():
    """阈值必须远高于真实主规格条件数（实测 p95 约 19），否则会把健康月份拦掉。"""
    assert pl.COND_MAX >= 1e8
    rr = np.random.default_rng(7)
    n = 400
    X = np.column_stack([rr.normal(size=n), rr.normal(size=n), np.ones(n)])
    cn = np.linalg.norm(X, axis=0)
    sv = np.linalg.svd(X / cn, compute_uv=False)
    assert sv[0] / sv[-1] < pl.COND_MAX
