"""Phase 1.12 / I-12 / C-7 + 口径更正：pricing_gate v3 样本契约测试（只读真实产物）。

覆盖行数/重复键/PIT/零填、winsor、IQR、sample_window_actual、sample_version：
- 行数 307,503/384,303/226,210/282,570；缺文件 skip；重复键/PIT/零填均 0。
- winsor 边界 registry [0.01,0.99]；IQR_POOL_REGISTERED=0.02341287 与 parquet 重算一致。
- sample_version 当前标签 + computed 2000-02..2026-06（只读）。
v3 = 口径更正 重建（complete_month/n_valid_days = volume>0 交易日 >=15）；v2 文件保留为 superseded。
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from conftest import registered_rows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import pricing_sample_build as build  # noqa: E402
from pricing_lib import (  # noqa: E402
    IQR_POOL_REGISTERED, SAMPLE_WINDOW_END, SAMPLE_WINDOW_START, sample_path, winsor_bounds,
)

KEYS = ['ma_teacher', 'mb_teacher', 'ma', 'mb']
V3_ROWS = {'ma_teacher': 307_503, 'mb_teacher': 384_303, 'ma': 226_210, 'mb': 282_570}
ACTUAL_WINDOWS = {'ma_teacher': '2015-05..2023-12', 'mb_teacher': '2015-05..2023-12',
                  'ma': '2018-05..2023-12', 'mb': '2018-05..2023-12'}
REQUESTED_WINDOW = '2015-01..2023-12'
CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq',
            'AnalystCoverage', 'NetworkDegree']
READ_COLS = ['focal_id', 'month', 'available_at', 'pricing_exclusion_reason',
             'PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'network_version',
             'pricing_in_fm_sample', 'sample_window'] + CONTROLS
I16_COLS = ['window_requested', 'sample_window_actual']

SAMPLE_REPORT_V3 = ROOT / 'reports' / 'data_gates' / 'G0' / 'pricing_sample_report_v3.json'
SAMPLE_REPORT = ROOT / 'reports' / 'data_gates' / 'G0' / 'pricing_sample_report.json'
MONTHLY_RETURNS_V2 = ROOT / 'data' / 'processed' / 'monthly_returns_clean_v2.csv'


def _read_columns(schema) -> list:
    """按 parquet schema 追加已落盘的 I-16 新列；当前未重写则不加（不伪造列）。"""
    return READ_COLS + [c for c in I16_COLS if c in schema]


@pytest.fixture(scope='module')
def v3_frames():
    """module 级缓存：先读 schema（不投影），再按需加载 I-16 新列。"""
    frames = {}
    for key in KEYS:
        path = sample_path(key, 'v3')
        if path.exists():
            schema = set(pq.ParquetFile(path).schema_arrow.names)
            frames[key] = pd.read_parquet(path, columns=_read_columns(schema))
    return frames


def _frame(v3_frames, key):
    if key not in v3_frames:
        pytest.skip(f'缺少 v2 样本件: {sample_path(key, "v3")}')
    return v3_frames[key]


@pytest.fixture(scope='module')
def report_v3():
    assert SAMPLE_REPORT_V3.exists(), f'缺少 v2 样本报告: {SAMPLE_REPORT_V3}'
    return json.loads(SAMPLE_REPORT_V3.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def report_main():
    assert SAMPLE_REPORT.exists(), f'缺少样本报告: {SAMPLE_REPORT}'
    return json.loads(SAMPLE_REPORT.read_text(encoding='utf-8'))


# ---------------------------------------------------------------- 行数 / 键 / PIT
@pytest.mark.parametrize('key', KEYS)
def test_v3_sample_rows(v3_frames, key):
    assert len(_frame(v3_frames, key)) == V3_ROWS[key]


@pytest.mark.parametrize('key', KEYS)
def test_v3_no_duplicate_focal_month(v3_frames, key):
    df = _frame(v3_frames, key)
    assert int(df.duplicated(['focal_id', 'month']).sum()) == 0


@pytest.mark.parametrize('key', KEYS)
def test_v3_pit_violations_zero(v3_frames, key):
    """违规口径与 pricing_lib.contract_from_frame 一致：available_at >= month 首日。"""
    df = _frame(v3_frames, key)
    av = pd.to_datetime(df['available_at'], errors='coerce')
    mo = pd.to_datetime(df['month'].astype(str) + '-01', errors='coerce')
    assert int((av >= mo).sum()) == 0
    # 非缺失 available_at 必须全部可解析（防止字符串比较把坏日期静默算成安全）
    assert int((df['available_at'].notna() & av.isna()).sum()) == 0


# ---------------------------------------------------------------- 零填 / winsor
@pytest.mark.parametrize('key', KEYS)
def test_missing_filled_with_zero_zero(report_v3, key):
    contract = report_v3['contracts'][key]
    integrity = report_v3['integrity_on_delivered_files'][key]
    assert contract['missing_filled_with_zero'] == 0
    assert integrity['missing_filled_with_zero_total'] == 0
    # v2 report 的 contracts[ma/mb].rows 是 panel 行数；v2 行数以 integrity 段为准
    assert integrity['rows'] == V3_ROWS[key]
    assert contract['duplicate_focal_month_keys'] == 0
    assert contract['pit_violations_strict_lt'] == 0


def test_winsor_bounds_match_registry(report_main):
    lo, hi = winsor_bounds()
    assert (lo, hi) == (0.01, 0.99)
    assert build.WINSOR == (0.01, 0.99)
    assert [float(v) for v in report_main['registry_params']['winsorization_pct']] == [0.01, 0.99]
    assert (SAMPLE_WINDOW_START, SAMPLE_WINDOW_END) == ('2015-01', '2023-12')


# ---------------------------------------------------------------- IQR
def test_iqr_registered_matches_parquet_recompute(v3_frames, report_main):
    """treatment-only 宇宙（outcome-blind）在 ma_teacher v2 上重算 IQR_pool。"""
    assert IQR_POOL_REGISTERED == 0.02341287
    df = _frame(v3_frames, 'ma_teacher')
    universe_t = (df['PeerMAXraw'].notna()
                  & df['OwnMAXraw'].notna()
                  & df['PeerReturn'].notna()
                  & ~df[CONTROLS].isna().any(axis=1))
    iqr = float(df.loc[universe_t, 'PeerMAXraw'].quantile(.75)
                - df.loc[universe_t, 'PeerMAXraw'].quantile(.25))
    assert abs(iqr - IQR_POOL_REGISTERED) <= 1e-12
    assert report_main['iqr_pool_registered'] == IQR_POOL_REGISTERED


# ---------------------------------------------------------------- sample_window
@pytest.mark.parametrize('key', KEYS)
def test_sample_window_actual_semantics(v3_frames, key):
    df = _frame(v3_frames, key)
    schema = set(pq.ParquetFile(sample_path(key, 'v3')).schema_arrow.names)
    expected = ACTUAL_WINDOWS[key]
    # fixture 必须按 schema 加载新列：存在则 df 有列，缺失则不得伪造列
    expected_new = [c for c in I16_COLS if c in schema]
    assert all(c in df.columns for c in expected_new)
    assert all(c not in df.columns for c in set(I16_COLS) - set(expected_new))

    # 只读 helper：按列实际窗（network_version.notna() | pricing_in_fm_sample）
    assert build.compute_actual_sample_window(df) == expected
    assert build.WINDOW == REQUESTED_WINDOW
    assert (SAMPLE_WINDOW_START, SAMPLE_WINDOW_END) == ('2015-01', '2023-12')

    if set(I16_COLS) <= schema:
        # I-16 metadata 重写已落盘：列已加入读取集合，断言列值
        assert df['sample_window_actual'].dropna().unique().tolist() == [expected]
        assert df['window_requested'].dropna().unique().tolist() == [REQUESTED_WINDOW]
    elif schema.isdisjoint(I16_COLS):
        # 当前未重写（1.13 前）：schema 确无新列；旧 sample_window 保持申请窗
        assert 'sample_window_actual' not in schema
        assert 'window_requested' not in schema
        assert df['sample_window'].dropna().unique().tolist() == [REQUESTED_WINDOW]
    else:
        pytest.fail(f'{key}: I-16 新列仅部分存在: {sorted(schema & set(I16_COLS))}')

    start = expected.split('..')[0]
    pre = df[df['month'] < start]
    if len(pre):
        assert pre['network_version'].isna().all()
        assert not pre['pricing_in_fm_sample'].fillna(False).astype(bool).any()


# ---------------------------------------------------------------- sample_version（只读 CSV）
def test_monthly_returns_sample_version_readonly():
    assert MONTHLY_RETURNS_V2.exists(), f'缺少 {MONTHLY_RETURNS_V2}'
    df = pd.read_csv(MONTHLY_RETURNS_V2, usecols=['month', 'sample_version'], dtype=str)
    assert len(df) == registered_rows('monthly_returns_clean_v2')
    computed = f"{df['month'].min()}..{df['month'].max()}"
    assert computed.count('..') == 1
    # Phase 4 BL-01（I-17）已重写：sample_version = computed 标签（min(month)..max(month)）
    assert df['sample_version'].dropna().unique().tolist() == [computed]
    assert df['sample_version'].iat[0] == computed
    # 旧手写标签已退役（仅生成器追溯列 sample_version_legacy 保留）
    assert computed != 'mainline_2018_2025_fullperiod'
