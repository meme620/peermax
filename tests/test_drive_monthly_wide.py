"""Phase 4 BL-03：`pricing_lib._drive_monthly_wide` how 分支显式化护栏。

背景（`artifacts/phase4/4.0_backlog.md` §2.3；Phase 2.4 `shadow_turnover.py` 已实证）：
修复前 `how != 'sum'` 全部静默落 `.last()`（含 `'mean'`）。修复后：

  - `'sum'`  : `resample('ME').sum(min_count=1)`（口径不变；全 NaN 月/列 = NaN，不是 0）
  - `'last'` : `resample('ME').last()`（口径不变；月内最后**非 NaN**值）
  - `'mean'` : `groupby(月度期).mean()`——部分 NaN 月 = 非 NaN 日均值；
    **重采样宽表层**（`.stack()` 前）全 NaN 月为 NaN、部分 NaN 月为非 NaN 均值
  - 其他     : `raise ValueError(f"unsupported how={how!r}")`

覆盖（合成 pkl：3 个月、每月多日、含 NaN；全部断言落在宽表层或长表逐值）：
  1. sum / last / mean 三分支的宽表值与 NaN 位置；
  2. 全 NaN 月（2014-09）在宽表层为 NaN、因 `.stack()` 默认 dropna 不出现在长表；
  3. mean ≠ last（旧 bug 必然失败）的逐值证据；
  4. mean 与 `shadow_turnover.monthly_frames` 自实现月均值逐值对拍（max|Δ| = 0）；
  5. 未知 how 抛 `ValueError`；`resample('ME').mean(min_count=1)` 抛
     `pandas.errors.UnsupportedFunctionCall`（禁止写法，非 TypeError）；
  6. 两个生产 loader（`load_amount_100m` = sum、`load_turnover_drive` = last）
     在合成 pkl 上的输出口径。

注：合成数据不落在真实 data/ 上、不读写真实产物；月份内至少 1 个交易日。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import pricing_lib as lib  # noqa: E402

P2_4_SCRIPT = ROOT / 'scripts/shadow_turnover.py'

CODES = ['000002.SZ', '600000.SH', '000001.SZ']  # 000001.SZ 全 NaN -> 长表无行
DAYS = ['2014-07-01', '2014-07-02', '2014-07-03', '2014-07-31',
        '2014-08-04', '2014-08-29', '2014-09-05', '2014-09-30']
# 2014-07 部分 NaN（000002 [1,2,NaN,3]、600000 [NaN,2.5,3.5,NaN]）；
# 2014-08 部分 NaN；2014-09 全 NaN（全 NaN 月）。
VALUES = np.array([
    [1.0, np.nan], [2.0, 2.5], [np.nan, 3.5], [3.0, np.nan],      # 2014-07: 000002 / 600000
    [4.5, np.nan], [0.5, 6.0],                                    # 2014-08
    [np.nan, np.nan], [np.nan, np.nan],                           # 2014-09
], dtype='float64')  # 行 = 交易日（DAYS 序）、列 = CODES 前两列
WIDE = pd.DataFrame(
    np.column_stack([VALUES, np.full((len(DAYS), 1), np.nan)]),
    index=pd.to_datetime(DAYS), columns=CODES)

MONTHS = ['2014-07', '2014-08', '2014-09']
# 期望宽表（索引 = 'YYYY-MM'，列序 = CODES，含 NaN 位置）
EXPECT_JUL = {'sum': [6.0, 6.0, np.nan], 'last': [3.0, 3.5, np.nan], 'mean': [2.0, 3.0, np.nan]}
EXPECT_AUG = {'sum': [5.0, 6.0, np.nan], 'last': [0.5, 6.0, np.nan], 'mean': [2.5, 6.0, np.nan]}
EXPECT_SEP = {'sum': [np.nan] * 3, 'last': [np.nan] * 3, 'mean': [np.nan] * 3}


def _expected_wide(how: str) -> pd.DataFrame:
    return pd.DataFrame([EXPECT_JUL[how], EXPECT_AUG[how], EXPECT_SEP[how]],
                        index=MONTHS, columns=CODES, dtype='float64')


def _expected_long(how: str) -> pd.DataFrame:
    wide = _expected_wide(how)
    long = wide.stack().rename('value').reset_index()
    long.columns = ['month', 'code', 'value']
    long['firm_id'] = long['code'].map(lib.drive_code_to_firm_id)
    return long[['firm_id', 'month', 'value']].sort_values(
        ['month', 'firm_id'], kind='mergesort').reset_index(drop=True)


@pytest.fixture
def pkl(tmp_path):
    p = tmp_path / 'daily_synthetic.pkl'
    WIDE.to_pickle(p)
    return p


def _capture_wide(monkeypatch, pkl, how):
    """捕获 `_drive_monthly_wide` 内在 `.stack()` 之前的重采样宽表（字符串索引）。"""
    captured = {}
    orig = pd.DataFrame.stack

    def spy(self, *args, **kwargs):
        captured.setdefault('wide', self.copy())
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, 'stack', spy)
    long = lib._drive_monthly_wide(pkl, how)
    assert 'wide' in captured, '未捕获到宽表层（stack 未被调用）'
    return captured['wide'], long


def _sorted(long: pd.DataFrame) -> pd.DataFrame:
    return long.sort_values(['month', 'firm_id'], kind='mergesort').reset_index(drop=True)


# ------------------------------------------------------- 1) sum / last / mean 宽表层
def test_sum_branch_wide_values_and_nan_positions(monkeypatch, pkl):
    wide, long = _capture_wide(monkeypatch, pkl, 'sum')
    pd.testing.assert_frame_equal(wide, _expected_wide('sum'), check_exact=True)
    # 口径与修复前逐值相同：等价于直接 resample sum(min_count=1)
    ref = WIDE.resample('ME').sum(min_count=1)
    assert np.array_equal(wide.to_numpy(), ref.to_numpy(), equal_nan=True)
    assert wide.index.tolist() == MONTHS and wide.columns.tolist() == CODES
    pd.testing.assert_frame_equal(_sorted(long), _expected_long('sum'), check_exact=True)


def test_last_branch_wide_values_and_nan_positions(monkeypatch, pkl):
    wide, long = _capture_wide(monkeypatch, pkl, 'last')
    pd.testing.assert_frame_equal(wide, _expected_wide('last'), check_exact=True)
    ref = WIDE.resample('ME').last()
    assert np.array_equal(wide.to_numpy(), ref.to_numpy(), equal_nan=True)
    pd.testing.assert_frame_equal(_sorted(long), _expected_long('last'), check_exact=True)


def test_mean_branch_wide_values_and_nan_positions(monkeypatch, pkl):
    """宽表层：部分 NaN 月 = 非 NaN 日均值；全 NaN 月 = NaN（不是 0、不报错）。"""
    wide, long = _capture_wide(monkeypatch, pkl, 'mean')
    pd.testing.assert_frame_equal(wide, _expected_wide('mean'), check_exact=True)
    # 部分 NaN 月为非 NaN
    assert wide.loc['2014-07', '000002.SZ'] == 2.0
    assert wide.loc['2014-07', '600000.SH'] == 3.0
    assert wide.loc['2014-08', '000002.SZ'] == 2.5
    # 全 NaN 列与全 NaN 月在宽表层为 NaN
    assert wide['000001.SZ'].isna().all()
    assert wide.loc['2014-09'].isna().all()
    # 长表：全 NaN 月被 `.stack()` 默认 dropna 丢弃（BL-03 默认不改该语义）
    assert sorted(long['month'].unique()) == ['2014-07', '2014-08']
    assert '2014-09' not in set(long['month'])
    pd.testing.assert_frame_equal(_sorted(long), _expected_long('mean'), check_exact=True)


def test_mean_is_not_last_old_bug_regression(monkeypatch, pkl):
    """旧 bug（mean 落 .last()）逐值反例：两个分支在合成数据上必须不同。"""
    mean_wide, _ = _capture_wide(monkeypatch, pkl, 'mean')
    last_wide = WIDE.resample('ME').last()
    last_wide.index = [c.strftime('%Y-%m') for c in last_wide.index]
    # 2014-07 000002: mean 2.0 vs last 3.0；2014-08 000002: mean 2.5 vs last 0.5
    assert mean_wide.loc['2014-07', '000002.SZ'] != last_wide.loc['2014-07', '000002.SZ']
    assert mean_wide.loc['2014-08', '000002.SZ'] != last_wide.loc['2014-08', '000002.SZ']
    long_mean = _sorted(lib._drive_monthly_wide(pkl, 'mean'))
    long_last = _sorted(lib._drive_monthly_wide(pkl, 'last'))
    merged = long_mean.merge(long_last, on=['firm_id', 'month'], suffixes=('_mean', '_last'))
    assert len(merged) == 4
    assert (merged['value_mean'] != merged['value_last']).any()


# ------------------------------------------------------- 2) 与 Phase 2.4 自实现月均值对拍
def test_mean_matches_p2_4_self_implementation(monkeypatch, pkl):
    """逐值对拍 `shadow_turnover.monthly_frames` 自实现月均值（含 NaN 位置）。"""
    spec = importlib.util.spec_from_file_location('shadow_turnover', P2_4_SCRIPT)
    p2_4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2_4)
    ref = p2_4.monthly_frames(p2_4.load_turnover_wide(pkl))['turnover_mean']
    wide, _ = _capture_wide(monkeypatch, pkl, 'mean')
    pd.testing.assert_frame_equal(wide, ref, check_exact=True)
    assert np.array_equal(wide.to_numpy(), ref.to_numpy(), equal_nan=True)
    assert wide.index.tolist() == ref.index.tolist()
    diff = (wide - ref).abs()
    assert float(diff.max().max()) == 0.0, 'mean 与 p2_4 自实现月均值出现非零差'


# ------------------------------------------------------- 3) 未知 how / 禁止写法
@pytest.mark.parametrize('how', ['median', 'average', 'SUM', 'mean ', ''])
def test_unknown_how_raises_value_error(pkl, how):
    with pytest.raises(ValueError) as ei:
        lib._drive_monthly_wide(pkl, how)
    assert f"unsupported how={how!r}" in str(ei.value)


def test_prohibited_mean_min_count_raises_unsupported_function_call():
    """不得写 `resample('ME').mean(min_count=1)`：pandas 2.3.1 抛
    `pandas.errors.UnsupportedFunctionCall`（**不是** TypeError；memo/断言口径一致）。"""
    with pytest.raises(pd.errors.UnsupportedFunctionCall) as ei:
        WIDE.resample('ME').mean(min_count=1)
    assert 'numpy operations are not valid with resample' in str(ei.value)
    assert not isinstance(ei.value, TypeError)


# ------------------------------------------------------- 4) 生产 loader 口径（合成 pkl 注入）
def test_load_amount_100m_uses_sum_branch(monkeypatch, pkl):
    monkeypatch.setattr(lib, 'AMOUNT_PKL', pkl)
    got = lib.load_amount_100m()
    exp = _expected_long('sum').assign(
        amount_100m=lambda d: d['value'] / 1e5)[['firm_id', 'month', 'amount_100m']]
    pd.testing.assert_frame_equal(_sorted(got), _sorted(exp), check_exact=True)


def test_load_turnover_drive_uses_last_branch(monkeypatch, pkl):
    monkeypatch.setattr(lib, 'TURNOVER_PKL', pkl)
    got = lib.load_turnover_drive()
    exp = _expected_long('last').assign(
        turnover=lambda d: d['value'] / 100.0)[['firm_id', 'month', 'turnover']]
    pd.testing.assert_frame_equal(_sorted(got), _sorted(exp), check_exact=True)
