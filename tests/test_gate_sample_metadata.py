"""数据门 sample-metadata：样本窗口列现算、sample_version 由数据算。"""

import pandas as pd
from pathlib import Path

from peerlab import gates


def _write_returns(root: Path, months, sample_version, source='computed_from_data') -> None:
    path = root / 'data' / 'processed' / 'monthly_returns_clean_v2.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({'month': months, 'sample_version': [sample_version] * len(months),
                          'sample_version_source': [source] * len(months)})
    frame.to_csv(path, index=False)


def _write_sample(root: Path, actual: str, requested: str = '2018-01..2018-12') -> None:
    path = root / 'data' / 'processed' / 'pricing_sample_ma_v3.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        'month': ['2018-04', '2018-05', '2018-06', '2018-07'],
        'network_version': [None, 'pm_ma_v1_2018', 'pm_ma_v1_2018', None],
        'pricing_in_fm_sample': [False, True, True, False],
        'window_requested': [requested] * 4,
        'sample_window_actual': [actual] * 4,
    })
    frame.to_parquet(path, index=False)


def test_missing_artifacts_blocked(tmp_path: Path) -> None:
    outcome = gates.check_sample_metadata(tmp_path)
    assert outcome['blocked'] and outcome['problems'] == []


def test_valid_passes(tmp_path: Path) -> None:
    _write_sample(tmp_path, '2018-05..2018-06')
    _write_returns(tmp_path, ['2018-05', '2018-06', '2018-07'], '2018-05..2018-07')
    outcome = gates.check_sample_metadata(tmp_path)
    assert outcome['problems'] == []
    assert outcome['checked'] == 2


def test_wrong_actual_window_fails(tmp_path: Path) -> None:
    _write_sample(tmp_path, '2018-01..2018-12')
    _write_returns(tmp_path, ['2018-05', '2018-06'], '2018-05..2018-06')
    problems = gates.check_sample_metadata(tmp_path)['problems']
    assert any('sample_window_actual' in p for p in problems)


def test_legacy_sample_version_fails(tmp_path: Path) -> None:
    _write_sample(tmp_path, '2018-05..2018-06')
    _write_returns(tmp_path, ['2018-05', '2018-06'], 'mainline_2018_2025_fullperiod')
    problems = gates.check_sample_metadata(tmp_path)['problems']
    assert any('sample_version' in p for p in problems)


def test_static_source_fails(tmp_path: Path) -> None:
    _write_sample(tmp_path, '2018-05..2018-06')
    _write_returns(tmp_path, ['2018-05', '2018-06'], '2018-05..2018-06', source='legacy_label')
    problems = gates.check_sample_metadata(tmp_path)['problems']
    assert any('sample_version_source' in p for p in problems)


def test_provenance_falls_back_to_status_report(tmp_path: Path) -> None:
    """交付 CSV 是 10 列：无溯源列时看 status 报告，缺报告即 fail。"""
    _write_sample(tmp_path, '2018-05..2018-06')
    _write_returns(tmp_path, ['2018-05', '2018-06'], '2018-05..2018-06')
    ret = tmp_path / 'data/processed/monthly_returns_clean_v2.csv'
    keep = pd.read_csv(ret)
    keep.drop(columns=['sample_version_source']).to_csv(ret, index=False)
    problems = gates.check_sample_metadata(tmp_path)['problems']
    assert any('缺' in p and 'report' in p for p in problems), problems

    report = tmp_path / 'reports/data_gates/G0/monthly_returns_status_report.json'
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"sample_version_rule": "min(month)..max(month)"}', encoding='utf-8')
    assert gates.check_sample_metadata(tmp_path)['problems'] == []


def test_missing_columns_fail(tmp_path: Path) -> None:
    path = tmp_path / 'data' / 'processed' / 'pricing_sample_ma_v3.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'month': ['2018-05'], 'network_version': ['pm_ma_v1_2018']}).to_parquet(path, index=False)
    _write_returns(tmp_path, ['2018-05'], '2018-05..2018-05')
    problems = gates.check_sample_metadata(tmp_path)['problems']
    assert any('缺列' in p for p in problems)
