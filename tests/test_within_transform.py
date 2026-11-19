"""within 变换（行业×月固定效应）与显式虚拟变量回归对拍。

固定效应规格的可信度取决于去均值是否严格等价于加饱和虚拟变量。
逐月 Fama-MacBeth 里放月份虚拟变量会与截距共线，所以这里用合并 OLS 对拍。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from pricing_lib import demean_by  # noqa: E402


def _panel(n_months=6, n_ind=4, n_per=30, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        month = f'2020-{m + 1:02d}'
        for ind in range(n_ind):
            ind_effect = rng.normal(scale=0.02)
            for _ in range(n_per):
                x = rng.normal(scale=0.05)
                y = -0.4 * x + ind_effect + rng.normal(scale=0.01)
                rows.append({'month': month, 'l1_code': f'I{ind}', 'x': x, 'y': y})
    return pd.DataFrame(rows)


def _ols_beta_x(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[0])


def test_within_equals_dummy():
    """within 去均值 OLS 与 (month×industry) 饱和虚拟变量 OLS 的斜率一致。"""
    df = _panel()
    w = demean_by(df, ['y', 'x'], ['month', 'l1_code'])
    beta_within = _ols_beta_x(w[['x']].to_numpy(), w['y'].to_numpy())

    dummies = pd.get_dummies(df['month'] + '|' + df['l1_code'], dtype=float)
    X = np.column_stack([df['x'].to_numpy(), np.ones(len(df)), dummies.to_numpy()])
    beta_dummy = _ols_beta_x(X, df['y'].to_numpy())

    np.testing.assert_allclose(beta_within, beta_dummy, atol=1e-10)


def test_demean_matches_manual_group_mean():
    df = _panel()
    out = demean_by(df, ['x'], ['month', 'l1_code'])
    manual = df['x'] - df.groupby(['month', 'l1_code'])['x'].transform('mean')
    np.testing.assert_allclose(out['x'].to_numpy(), manual.to_numpy(), atol=1e-12)


def test_demean_keeps_missing():
    df = _panel()
    df.loc[df.index[:3], 'x'] = np.nan
    out = demean_by(df, ['x'], ['month', 'l1_code'])
    assert out['x'].isna().sum() == 3


def test_demean_zero_mean_within_group():
    df = _panel()
    out = demean_by(df, ['x'], ['month', 'l1_code'])
    means = out.groupby(['month', 'l1_code'])['x'].mean().abs()
    assert (means < 1e-12).all()
