"""聚类稳健 FM 抽样误差自检。

fm_cluster_se 与 fm_regression 必须给出同一个 β 与 NW SE（同口径逐月 OLS）。
聚类设计效应只在「回归量与聚类同时组内相关」时出现：regressor 若与聚类独立，
聚类方差与同方差方差渐近相等（deff≈1）。设置按此构造。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from pricing_lib import fm_cluster_se, fm_regression  # noqa: E402

MAIN = ['PeerMAXraw']


def _panel(cluster_corr: bool, n_months=12, n_clusters=10, n_per=30, seed=11):
    """cluster_corr=True：x 与 y 都带同一团成分 → 团内误差相关且 regressor 团内相似。"""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        month = f'2021-{m + 1:02d}'
        for c in range(n_clusters):
            if cluster_corr:
                cx = rng.normal(scale=0.05)      # 团内共同 regressor 成分
                cy = rng.normal(scale=0.05)      # 团内共同误差成分
            else:
                cx = cy = 0.0
            for _ in range(n_per):
                x = cx + rng.normal(scale=0.02)
                y = -0.3 * x + cy + rng.normal(scale=0.02)
                rows.append({'month': month, 'l1_code': f'C{c}',
                             'PeerMAXraw': x, 'R_next': y})
    return pd.DataFrame(rows)


def test_beta_and_nw_match_fm_regression():
    df = _panel(False)
    a = fm_cluster_se(df, 'R_next', MAIN, 'l1_code', min_n=50, nw_lag=1)
    b, _ = fm_regression(df, 'R_next', MAIN, min_n=50, nw_lag=1)
    assert a['T'] == b['n_months_used']
    np.testing.assert_allclose(a['beta'], b['coefs']['PeerMAXraw']['beta'], atol=1e-12)
    np.testing.assert_allclose(a['se_nw'], b['coefs']['PeerMAXraw']['se_nw'], atol=1e-12)


def test_deff_above_one_when_cluster_correlated():
    r = fm_cluster_se(_panel(True), 'R_next', MAIN, 'l1_code', min_n=50, nw_lag=1)
    assert r['status'] == 'COMPUTED'
    assert r['deff_within_month'] > 1.2, r['deff_within_month']


def test_deff_near_one_when_independent():
    r = fm_cluster_se(_panel(False), 'R_next', MAIN, 'l1_code', min_n=50, nw_lag=1)
    assert 0.6 < r['deff_within_month'] < 1.6, r['deff_within_month']


def test_hybrid_above_nw_when_deff_above_one():
    r = fm_cluster_se(_panel(True), 'R_next', MAIN, 'l1_code', min_n=50, nw_lag=1)
    assert r['se_hybrid'] > r['se_nw']


def test_hybrid_never_below_zero_and_finite():
    r = fm_cluster_se(_panel(True), 'R_next', MAIN, 'l1_code', min_n=50, nw_lag=1)
    assert np.isfinite(r['se_hybrid']) and r['se_hybrid'] > 0
    assert r['ratio_hybrid_vs_nw'] is not None


def test_not_computable_when_too_few_months():
    r = fm_cluster_se(_panel(True, n_months=1), 'R_next', MAIN, 'l1_code',
                      min_n=50, nw_lag=1)
    assert r['status'] == 'NOT_COMPUTABLE'


def test_missing_cluster_column_raises():
    df = _panel(True).drop(columns=['l1_code'])
    try:
        fm_cluster_se(df, 'R_next', MAIN, 'l1_code', min_n=50, nw_lag=1)
    except KeyError:
        return
    raise AssertionError('缺 cluster 列应报错而不是静默跳过')
