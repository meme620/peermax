"""Phase 1 I-14：amount_100m 缓存名/内容/指纹契约测试。

覆盖：
  - 旧 `m1_cache_illiq.parquet` → 新 `m1_cache_amount_100m.parquet` 一次性迁移，数值/列/行数逐值不变；
  - sidecar `m1_cache_amount_100m.meta.json` 必填字段（输入路径/sha、公式、单位、行数、dtype、脚本 sha、时间）；
  - 新缓存 + sidecar + 输入 sha 一致 → 命中且不重算；
  - sidecar 缺失 → 重建（不静默复用）；输入 sha 不匹配 → 默认 fail-closed，显式 rebuild=True 才覆盖；
  - 旧缓存列不符 → 明确拒绝；
  - `load_illiq` 返回列/数值保持不变（无 illiq 列）。

只使用 tmp_path 合成缓存/输入；不读写真实 artifacts/tmp 缓存，不跑 M1。
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import pricing_lib as lib  # noqa: E402

SOURCE_BYTES = b'phase1-i14-fake-daily-total-amount-source-v1'

SYNTHETIC = pd.DataFrame({
    'firm_id': ['sz000001', 'sz000001', 'sz000002', 'sz000002', 'sh600000'],
    'month': ['2014-06', '2014-07', '2014-06', '2014-07', '2014-06'],
    'amount_100m': [164.048786, 210.5, float('nan'), 3.75, 999.125],
})


def _recompute(frame: pd.DataFrame):
    calls = {'n': 0}

    def _f():
        calls['n'] += 1
        return frame.copy()

    return _f, calls


@pytest.fixture
def source(tmp_path):
    p = tmp_path / 'daily_total_amount.pkl'
    p.write_bytes(SOURCE_BYTES)
    return p


@pytest.fixture
def frame():
    return SYNTHETIC.copy()


def _write_legacy(tmp_path, frame):
    p = tmp_path / lib.LEGACY_AMOUNT_CACHE_NAME
    frame.to_parquet(p, index=False)
    return p


def test_legacy_cache_migrated_values_unchanged(tmp_path, source, frame):
    legacy = _write_legacy(tmp_path, frame)
    legacy_sha = lib.sha256_file(legacy)
    recompute, calls = _recompute(frame)

    df, meta = lib.load_amount_100m_cached(
        cache_dir=tmp_path, source_path=source, recompute=recompute)

    new_path = tmp_path / lib.AMOUNT_CACHE_NAME
    meta_path = tmp_path / lib.AMOUNT_CACHE_META_NAME
    assert new_path.exists() and meta_path.exists()
    assert meta['cache_status'] == 'legacy_migrated'
    assert calls['n'] == 0, '迁移必须只读旧缓存，不得重算'
    assert legacy.exists() and lib.sha256_file(legacy) == legacy_sha, '旧缓存必须保留且不改'

    old = pd.read_parquet(legacy)
    new = pd.read_parquet(new_path)
    assert list(new.columns) == lib.AMOUNT_CACHE_COLUMNS
    assert 'illiq' not in new.columns
    assert len(new) == len(old) == len(frame)
    pd.testing.assert_frame_equal(new, old, check_exact=True)
    pd.testing.assert_frame_equal(df, old, check_exact=True)


def test_meta_fields_complete(tmp_path, source, frame):
    _write_legacy(tmp_path, frame)
    _, meta = lib.load_amount_100m_cached(cache_dir=tmp_path, source_path=source)

    assert set(lib.AMOUNT_CACHE_META_REQUIRED) <= set(meta)
    assert Path(meta['source_path']) == source.resolve()
    assert meta['source_sha256'] == lib.sha256_file(source)
    assert meta['liquidity_formula_id'] == lib.ILLIQ_FORMULA_ID
    assert meta['formula'] == 'illiq = |R_t| / amount_100m'
    assert 'amount_100m' in meta['formula']
    assert '亿元' in meta['amount_100m_unit']
    assert meta['row_count'] == len(frame)
    assert meta['columns'] == lib.AMOUNT_CACHE_COLUMNS
    assert meta['dtypes'] == {'firm_id': 'object', 'month': 'object', 'amount_100m': 'float64'}
    assert re.fullmatch(r'[0-9a-f]{64}', meta['script_sha256'])
    assert Path(meta['script_path']).name == 'pricing_lib.py'
    assert meta['created_at_utc'].endswith('+00:00')

    on_disk = json.loads((tmp_path / lib.AMOUNT_CACHE_META_NAME).read_text(encoding='utf-8'))
    assert 'cache_status' not in on_disk, 'cache_status 仅返回给调用方，不写入 sidecar'
    for key in lib.AMOUNT_CACHE_META_REQUIRED:
        assert on_disk[key] == meta[key]


def test_cache_hit_does_not_recompute(tmp_path, source, frame):
    _write_legacy(tmp_path, frame)
    first, _ = lib.load_amount_100m_cached(cache_dir=tmp_path, source_path=source)

    def _must_not_run():
        raise AssertionError('有效缓存命中时不得重算')

    got, meta = lib.load_amount_100m_cached(
        cache_dir=tmp_path, source_path=source, recompute=_must_not_run)
    assert meta['cache_status'] == 'hit'
    pd.testing.assert_frame_equal(got, first, check_exact=True)


def test_missing_sidecar_rebuilds_not_reuses(tmp_path, source, frame):
    cache_path = tmp_path / lib.AMOUNT_CACHE_NAME
    stale = frame.copy()
    stale['amount_100m'] = stale['amount_100m'] + 1.0
    stale.to_parquet(cache_path, index=False)   # 有缓存、无 sidecar
    assert not (tmp_path / lib.AMOUNT_CACHE_META_NAME).exists()

    fresh = frame.copy()
    recompute, calls = _recompute(fresh)
    df, meta = lib.load_amount_100m_cached(
        cache_dir=tmp_path, source_path=source, recompute=recompute)

    assert meta['cache_status'] == 'rebuilt_missing_meta'
    assert calls['n'] == 1
    assert (tmp_path / lib.AMOUNT_CACHE_META_NAME).exists()
    pd.testing.assert_frame_equal(df, fresh, check_exact=True)
    pd.testing.assert_frame_equal(pd.read_parquet(cache_path), fresh, check_exact=True)


def test_source_sha_mismatch_fails_closed(tmp_path, source, frame, capsys):
    _write_legacy(tmp_path, frame)
    lib.load_amount_100m_cached(cache_dir=tmp_path, source_path=source)
    cache_sha = lib.sha256_file(tmp_path / lib.AMOUNT_CACHE_NAME)
    source.write_bytes(SOURCE_BYTES + b'-changed')

    with pytest.raises(SystemExit):
        lib.load_amount_100m_cached(cache_dir=tmp_path, source_path=source)
    err = capsys.readouterr().err
    assert '指纹' in err and 'source_sha256' in err
    assert lib.sha256_file(tmp_path / lib.AMOUNT_CACHE_NAME) == cache_sha, '失败路径不得改写缓存'


def test_explicit_rebuild_overrides_sha_mismatch(tmp_path, source, frame):
    _write_legacy(tmp_path, frame)
    lib.load_amount_100m_cached(cache_dir=tmp_path, source_path=source)
    source.write_bytes(SOURCE_BYTES + b'-changed')

    fresh = frame.copy()
    fresh['amount_100m'] = fresh['amount_100m'] * 2.0
    recompute, calls = _recompute(fresh)
    df, meta = lib.load_amount_100m_cached(
        cache_dir=tmp_path, source_path=source, recompute=recompute, rebuild=True)

    assert meta['cache_status'] == 'rebuilt_cache_contract_mismatch'
    assert calls['n'] == 1
    pd.testing.assert_frame_equal(df, fresh, check_exact=True)
    on_disk = json.loads((tmp_path / lib.AMOUNT_CACHE_META_NAME).read_text(encoding='utf-8'))
    assert on_disk['source_sha256'] == lib.sha256_file(source)


def test_legacy_wrong_columns_refused(tmp_path, source, frame):
    bad = frame.copy()
    bad = bad.rename(columns={'amount_100m': 'illiq'})
    bad.to_parquet(tmp_path / lib.LEGACY_AMOUNT_CACHE_NAME, index=False)

    with pytest.raises(SystemExit):
        lib.load_amount_100m_cached(cache_dir=tmp_path, source_path=source)
    assert not (tmp_path / lib.AMOUNT_CACHE_NAME).exists(), '列不符时不得生成新缓存'


def test_load_illiq_contract_unchanged(monkeypatch):
    long = pd.DataFrame({
        'firm_id': ['sz000001', 'sh600000'],
        'month': ['2014-06', '2014-06'],
        'value': [1.0e7, 2.5e6],
    })
    monkeypatch.setattr(lib, '_drive_monthly_wide', lambda pkl, how: long.copy())

    got = lib.load_illiq()
    expected = long[['firm_id', 'month']].copy()
    expected['amount_100m'] = long['value'] / 1e5
    assert list(got.columns) == lib.AMOUNT_CACHE_COLUMNS
    assert 'illiq' not in got.columns
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
    pd.testing.assert_frame_equal(lib.load_amount_100m(), expected, check_exact=True)
