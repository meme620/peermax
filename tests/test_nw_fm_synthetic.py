"""Phase 1.12 / C-7：NW / FM 分析模块合成对拍（独立参考实现）。

覆盖：
- `pricing_lib.newey_west` vs 手写 Bartlett Newey-West（均值 / se_nw / t_nw / se_iid / t_iid）。
- `pricing_lib.fm_regression` vs 独立逐月 OLS（numpy lstsq）+ 手写 NW（beta 序列均值 / SE / T）。
- VW：`weight_col` 走 sqrt(w) WLS，与独立参考一致，且与 EW 结果不同（证明权重生效）。
- 退化输入：T<2、全 NaN/全缺失、n<min_n、秩亏——返回值/per_month reason 明确。

全部为合成数据（numpy/pandas 内存构造），不读真实 parquet/CSV，不写任何文件，不重跑 pricing_gate/PC。
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from pricing_lib import fm_regression, newey_west  # noqa: E402


# ---------------------------------------------------------------- 独立参考实现
def bartlett_nw(values, lag: int) -> dict:
    """独立手写 Bartlett NW（不调用被测实现）。"""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    T = int(arr.size)
    out = {'T': T, 'mean': None, 'se_nw': None, 't_nw': None, 'se_iid': None, 't_iid': None}
    if T < 2:
        return out
    mean = float(arr.mean())
    dev = arr - mean
    s = float(np.dot(dev, dev) / T)
    for ell in range(1, min(int(lag), T - 1) + 1):
        weight = 1.0 - ell / (lag + 1.0)
        s += 2.0 * weight * float(np.dot(dev[ell:], dev[:-ell]) / T)
    se_nw = math.sqrt(max(s, 0.0) / T)
    se_iid = math.sqrt(float(np.var(dev, ddof=1)) / T)
    out.update({
        'mean': mean,
        'se_nw': se_nw if np.isfinite(se_nw) else None,
        't_nw': (mean / se_nw) if se_nw > 0 else None,
        'se_iid': se_iid if np.isfinite(se_iid) else None,
        't_iid': (mean / se_iid) if se_iid > 0 else None,
    })
    return out


def reference_fm(df, ycol, xcols, weight_col=None, min_n=100, nw_lag=6):
    """独立逐月 OLS（lstsq）+ 手写 NW；返回 ({coef: nw}, [(month, n), ...])。"""
    names = list(xcols) + ['const']
    coefs = {name: [] for name in names}
    used = []
    for month, sub in df.groupby('month', sort=True):
        sub = sub.dropna(subset=[ycol] + list(xcols))
        if weight_col:
            sub = sub[sub[weight_col].notna() & (sub[weight_col] > 0)]
        n = len(sub)
        if n < min_n:
            continue
        y = sub[ycol].to_numpy(dtype=float)
        X = np.column_stack([sub[c].to_numpy(dtype=float) for c in xcols]
                            + [np.ones(n)])
        if weight_col:
            sw = np.sqrt(sub[weight_col].to_numpy(dtype=float))
            X = X * sw[:, None]
            y = y * sw
        beta, _residuals, rank, _sv = np.linalg.lstsq(X, y, rcond=None)
        if rank < len(names):
            continue
        used.append((month, n))
        for j, name in enumerate(names):
            coefs[name].append(float(beta[j]))
    return {name: bartlett_nw(series, nw_lag) for name, series in coefs.items()}, used


# ---------------------------------------------------------------- 合成面板
MONTHS = [f'2015-{m:02d}' for m in range(1, 13)]


def make_panel(n_months=12, n=150, seed=20261026):
    """y = 1.5*x1 - 0.5*x2 + 0.25 + noise；w 为正权重。"""
    rng = np.random.default_rng(seed)
    frames = []
    for month in MONTHS[:n_months]:
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        noise = rng.normal(scale=0.5, size=n)
        w = rng.uniform(0.5, 2.0, size=n)
        frames.append(pd.DataFrame({
            'month': month,
            'x1': x1,
            'x2': x2,
            'w': w,
            'y': 1.5 * x1 - 0.5 * x2 + 0.25 + noise,
        }))
    return pd.concat(frames, ignore_index=True)


def assert_nw_equal(got, ref, abs_tol=1e-12):
    assert got['T'] == ref['T']
    for field in ('mean', 'se_nw', 't_nw', 'se_iid', 't_iid'):
        if ref[field] is None:
            assert got[field] is None, field
        else:
            assert got[field] == pytest.approx(ref[field], rel=1e-12, abs=abs_tol), field


# ---------------------------------------------------------------- newey_west 对拍
NW_CASES = [
    pytest.param([0.10, -0.20, 0.30, 0.05, -0.15, 0.25, -0.05, 0.12,
                  -0.08, 0.22, 0.03, -0.11], 6, id='T12-lag6'),
    pytest.param([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], 3, id='T8-lag3'),
    pytest.param([0.0, 0.2, -0.1, 0.4, 0.05], 10, id='T5-lag-gt-T'),
    pytest.param([0.3, None, -0.1, np.nan, 0.2, np.inf, 0.05, -0.4], 2,
                 id='T6-with-missing'),
]


@pytest.mark.parametrize('series, lag', NW_CASES)
def test_newey_west_matches_independent_bartlett(series, lag):
    got = newey_west(series, lag=lag)
    ref = bartlett_nw(series, lag=lag)
    assert_nw_equal(got, ref)


def test_newey_west_empty_and_single_not_computable():
    for series in ([], [0.42], [None, np.nan, np.inf]):
        got = newey_west(series, lag=6)
        assert got['T'] == 0 or got['T'] == 1
        assert got['mean'] is None
        assert got['se_nw'] is None
        assert got['t_nw'] is None


def test_newey_west_constant_series_zero_se_no_t():
    got = newey_west([0.5] * 9, lag=3)
    ref = bartlett_nw([0.5] * 9, lag=3)
    assert got['T'] == 9
    assert got['mean'] == pytest.approx(0.5)
    assert got['se_nw'] == pytest.approx(0.0, abs=1e-15)
    assert got['t_nw'] is None
    assert_nw_equal(got, ref)


def test_newey_west_pandas_series_accepted():
    series = pd.Series([0.1, -0.05, 0.2, 0.0, -0.1])
    assert_nw_equal(newey_west(series, lag=2), bartlett_nw(series.tolist(), lag=2))


# ---------------------------------------------------------------- fm_regression 对拍
def test_fm_regression_matches_per_month_ols_and_manual_nw():
    df = make_panel()
    got, per_month = fm_regression(df, 'y', ['x1', 'x2'], min_n=100, nw_lag=3)
    ref, used = reference_fm(df, 'y', ['x1', 'x2'], min_n=100, nw_lag=3)

    assert got['n_months_used'] == 12 == len(used)
    assert [row['month'] for row in per_month] == MONTHS
    assert all(row['used'] for row in per_month)
    assert all(row['n'] == 150 for row in per_month)

    for coef in ('x1', 'x2', 'const'):
        s = got['coefs'][coef]
        r = ref[coef]
        assert s['beta'] == pytest.approx(r['mean'], rel=1e-11, abs=1e-10)
        assert s['se_nw'] == pytest.approx(r['se_nw'], rel=1e-10, abs=1e-10)
        assert s['t_nw'] == pytest.approx(r['t_nw'], rel=1e-9, abs=1e-9)
        assert s['n_months'] == 12
        assert s['se_iid'] == pytest.approx(r['se_iid'], rel=1e-10, abs=1e-10)

    # 合成真值合理性（容忍抽样噪声）
    assert got['coefs']['x1']['beta'] == pytest.approx(1.5, abs=0.15)
    assert got['coefs']['x2']['beta'] == pytest.approx(-0.5, abs=0.15)


def test_fm_regression_vw_matches_sqrt_weight_reference():
    df = make_panel()
    got_vw, per_month_vw = fm_regression(
        df, 'y', ['x1', 'x2'], weight_col='w', min_n=100, nw_lag=3)
    ref_vw, used_vw = reference_fm(
        df, 'y', ['x1', 'x2'], weight_col='w', min_n=100, nw_lag=3)
    got_ew, _ = fm_regression(df, 'y', ['x1', 'x2'], min_n=100, nw_lag=3)

    assert got_vw['weight_col'] == 'w'
    assert got_vw['n_months_used'] == 12 == len(used_vw)
    assert all(row['used'] and row['n'] == 150 for row in per_month_vw)

    for coef in ('x1', 'x2', 'const'):
        s = got_vw['coefs'][coef]
        r = ref_vw[coef]
        assert s['beta'] == pytest.approx(r['mean'], rel=1e-11, abs=1e-10)
        assert s['se_nw'] == pytest.approx(r['se_nw'], rel=1e-10, abs=1e-10)
        assert s['t_nw'] == pytest.approx(r['t_nw'], rel=1e-9, abs=1e-9)
        assert s['n_months'] == 12

    # 权重确实生效：VW 与 EW 的 beta 不应逐位相同
    assert abs(got_vw['coefs']['x1']['beta'] - got_ew['coefs']['x1']['beta']) > 1e-8


# ---------------------------------------------------------------- 退化输入
def test_fm_regression_below_min_n_not_used():
    df = make_panel(n=50)
    got, per_month = fm_regression(df, 'y', ['x1'], min_n=100, nw_lag=3)
    assert got['n_months_used'] == 0
    assert len(per_month) == 12
    for row in per_month:
        assert row['used'] is False
        assert row['reason'] == 'below_min_n'
        assert row['n'] == 50
    for coef in ('x1', 'const'):
        s = got['coefs'][coef]
        assert s['beta'] is None and s['se_nw'] is None and s['t_nw'] is None
        assert s['n_months'] == 0


def test_fm_regression_all_nan_yields_no_months():
    df = make_panel()
    df['y'] = np.nan
    got, per_month = fm_regression(df, 'y', ['x1'], min_n=100, nw_lag=3)
    assert got['n_months_used'] == 0
    assert all(row['used'] is False and row['reason'] == 'below_min_n' for row in per_month)


def test_fm_regression_rank_deficient_marked_not_used():
    df = make_panel()
    df['x_const'] = 1.0
    got, per_month = fm_regression(df, 'y', ['x_const'], min_n=100, nw_lag=3)
    assert got['n_months_used'] == 0
    assert all(row['used'] is False for row in per_month)
    assert all(row['reason'] == 'rank_or_df_or_covariance_failure' for row in per_month)
    assert got['coefs']['x_const']['beta'] is None


def test_fm_regression_single_month_t_less_than_two():
    """T=1：逐月系数存在，但 NW 汇总不可计算（T<2）。"""
    df = make_panel(n_months=1)
    got, per_month = fm_regression(df, 'y', ['x1'], min_n=100, nw_lag=6)
    assert got['n_months_used'] == 1
    assert per_month[0]['used'] is True
    s = got['coefs']['x1']
    assert s['n_months'] == 1
    assert s['beta'] is None
    assert s['se_nw'] is None
    assert s['t_nw'] is None


def test_fm_regression_weight_filter_can_drop_below_min_n():
    df = make_panel()
    df.loc[df.groupby('month').cumcount() < 60, 'w'] = 0.0   # 每月 60 行剔除 -> 90 < 100
    got, per_month = fm_regression(df, 'y', ['x1'], weight_col='w', min_n=100, nw_lag=3)
    assert got['n_months_used'] == 0
    assert all(row['used'] is False and row['reason'] == 'below_min_n' for row in per_month)
    assert all(row['n'] == 90 for row in per_month)


def test_reference_matches_on_vw_when_weights_are_constant():
    """w 常数时 sqrt(w) 对 beta 是纯缩放，VW 应数值上等于 EW（排除归一化误差）。"""
    df = make_panel(seed=7)
    df['w'] = 3.0
    got_vw, _ = fm_regression(df, 'y', ['x1', 'x2'], weight_col='w', min_n=100, nw_lag=3)
    got_ew, _ = fm_regression(df, 'y', ['x1', 'x2'], min_n=100, nw_lag=3)
    for coef in ('x1', 'x2', 'const'):
        assert got_vw['coefs'][coef]['beta'] == pytest.approx(
            got_ew['coefs'][coef]['beta'], rel=1e-10, abs=1e-12)
