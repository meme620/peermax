"""Phase 1.10 / I-16：sample_window 申请窗 vs 按列实际窗契约测试。

覆盖：
- 现有 4 个 v2 parquet（只读）：teacher 实际窗 2015-05..2023-12；ma/mb 2018-05..2023-12；
  旧 `sample_window` 列仍为 2015-01..2023-12（兼容，不改当前数据）。
- `compute_actual_sample_window`：`network_version` NaN 的前 4 个月被排除；
  按列分别计算；`m1_in_fm_sample==1` 单独可扩展窗口（OR 口径）；全空 → 哨兵。
- 真实样本件只读，不写入 data/、不跑 M1/PC/报告。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import pricing_sample_build as build  # noqa: E402

PANEL_KEYS = ['ma_teacher', 'mb_teacher', 'ma', 'mb']
LEGACY_WINDOW = '2015-01..2023-12'
EXPECTED_ACTUAL = {
    'ma_teacher': '2015-05..2023-12',
    'mb_teacher': '2015-05..2023-12',
    'ma': '2018-05..2023-12',
    'mb': '2018-05..2023-12',
}
REAL_COLS = ['month', 'network_version', 'm1_in_fm_sample', 'sample_window']


@pytest.fixture(scope='module')
def real_samples():
    """只读加载 4 个 v2 样本件（module 级缓存，避免重复 IO）。"""
    out = {}
    for key in PANEL_KEYS:
        path = ROOT / 'data' / 'processed' / f'm1_common_sample_{key}_v2.parquet'
        assert path.exists(), f'缺少 v2 样本件：{path}'
        out[key] = pd.read_parquet(path, columns=REAL_COLS)
    return out


def _actual_mask(frame: pd.DataFrame) -> pd.Series:
    return (frame['network_version'].notna()
            | frame['m1_in_fm_sample'].fillna(False).astype(bool))


# ---------------------------------------------------------------------------
# 真实样本件（只读）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('key', PANEL_KEYS)
def test_real_actual_window_per_column(real_samples, key):
    frame = real_samples[key]
    expected = EXPECTED_ACTUAL[key]
    start, end = expected.split('..')

    assert build.compute_actual_sample_window(frame) == expected

    months = frame.loc[_actual_mask(frame), 'month'].dropna()
    assert months.min() == start
    assert months.max() == end

    # 实际窗之前不存在任何可用网络版本（teacher 2015-01..04 / ma-mb 2018-01..04）
    pre = frame[frame['month'] < start]
    if len(pre):
        assert pre['network_version'].isna().all()
        assert not pre['m1_in_fm_sample'].fillna(False).astype(bool).any()


@pytest.mark.parametrize('key', PANEL_KEYS)
def test_real_legacy_sample_window_unchanged(real_samples, key):
    values = real_samples[key]['sample_window'].dropna().unique().tolist()
    assert values == [LEGACY_WINDOW]
    assert LEGACY_WINDOW == build.WINDOW  # 生成器旧列仍写 registry 申请窗


def test_real_teacher_pre_window_rows_still_present(real_samples):
    """I-16 不删行：2015-01..04 的 10,619 行仍在样本件中（network_version 全空）。"""
    frame = real_samples['ma_teacher']
    pre = frame[frame['month'] < '2015-05']
    assert len(pre) == 10_619
    assert pre['network_version'].isna().all()
    assert not pre['m1_in_fm_sample'].fillna(False).astype(bool).any()


# ---------------------------------------------------------------------------
# 合成小样本（验证口径；不触碰真实数据）
# ---------------------------------------------------------------------------

def _frame(months, versions, fm_flags):
    return pd.DataFrame({
        'month': months,
        'network_version': versions,
        'm1_in_fm_sample': fm_flags,
    })


def test_synthetic_nan_first_four_months_excluded():
    months = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05', '2015-06', '2023-12']
    frame = _frame(months,
                   [None, None, None, None, 2014.0, 2014.0, 2022.0],
                   [False] * len(months))
    assert build.compute_actual_sample_window(frame) == '2015-05..2023-12'


def test_synthetic_ma_mb_style_window_starts_2018_05():
    months = pd.period_range('2015-01', '2023-12', freq='M').strftime('%Y-%m').tolist()
    frame = _frame(months,
                   [None if m < '2018-05' else 2017.0 for m in months],
                   [False] * len(months))
    assert build.compute_actual_sample_window(frame) == '2018-05..2023-12'


def test_synthetic_fm_sample_alone_extends_window():
    """OR 口径：network_version 缺失但 m1_in_fm_sample=1 的月份也计入实际窗。"""
    months = ['2015-01', '2015-02', '2015-03', '2015-04', '2015-05']
    frame = _frame(months,
                   [None, None, None, None, 2014.0],
                   [False, False, False, True, False])
    assert build.compute_actual_sample_window(frame) == '2015-04..2015-05'


def test_synthetic_all_missing_returns_sentinel():
    months = ['2015-01', '2015-02', '2015-03', '2015-04']
    frame = _frame(months, [None] * 4, [False] * 4)
    assert build.compute_actual_sample_window(frame) == build.NO_NETWORK_VERSION
    assert build.NO_NETWORK_VERSION == 'NO_NETWORK_VERSION'


def test_synthetic_does_not_mutate_input():
    frame = _frame(['2015-01', '2015-05'], [None, 2014.0], [False, False])
    before = frame.copy(deep=True)
    build.compute_actual_sample_window(frame)
    pd.testing.assert_frame_equal(frame, before)
