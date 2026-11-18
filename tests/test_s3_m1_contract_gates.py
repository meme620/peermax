"""Phase 1 I-2：M1/PC 主链四条输入硬门测试。

只读真实产物（build report / v2 sample parquet）；失败路径用合成 DataFrame 与截断视图，
不覆盖真实文件、不重跑 M1/PC、不写任何报告。
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'data_audit'))

from s3_m1_lib import (  # noqa: E402
    HARD_GATE_KEYS, ROWS_NOT_APPLICABLE, assert_m1_contract, contract_from_frame, require,
    sample_path,
)

SAMPLE_REPORT = ROOT / 'reports' / 'data_gates' / 'G0' / 's3_m1_sample_report.json'
V2_MA_TEACHER = ROOT / 'data' / 'processed' / 'm1_common_sample_ma_teacher_v2.parquet'
V2_READ_COLS = ['focal_id', 'month', 'available_at', 'm1_exclusion_reason',
                'PeerMAXraw', 'OwnMAXraw', 'PeerReturn', 'R_next', 'market_equity']
V2_MA_TEACHER_ROWS = 307503

GOOD = {'rows': 4, 'rows_registered': 4, 'rows_match_registered': True,
        'pit_violations_strict_lt': 0, 'duplicate_focal_month_keys': 0,
        'missing_filled_with_zero': 0}


def _good_report(**over):
    contract = dict(GOOD)
    contract.update(over)
    return {'contracts': {'ma_teacher': contract}}


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        'focal_id': ['sz000001', 'sz000002', 'sz000003', 'sz000004'],
        'month': ['2020-01'] * 4,
        'available_at': ['2019-04-30'] * 4,
        'm1_exclusion_reason': [''] * 4,
        'PeerMAXraw': [0.1, 0.2, 0.3, 0.4],
        'OwnMAXraw': [0.2, 0.3, 0.4, 0.5],
        'PeerReturn': [0.01, 0.02, 0.03, 0.04],
        'R_next': [0.001, 0.002, 0.003, 0.004],
        'market_equity': [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture(scope='module')
def real_report() -> dict:
    return json.loads(SAMPLE_REPORT.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def real_v2() -> pd.DataFrame:
    return pd.read_parquet(V2_MA_TEACHER, columns=V2_READ_COLS)


# ---------------------------------------------------------------- require
def test_require_passes():
    assert require(True, 'should pass') is True


def test_require_false_exits_2_and_reports(capsys):
    with pytest.raises(SystemExit) as ei:
        require(False, 'truncated rows')
    assert ei.value.code == 2
    assert '[HARD-GATE] truncated rows' in capsys.readouterr().err


# ---------------------------------------------------------------- assert_m1_contract：四条硬门
def test_contract_dict_passes():
    summary = assert_m1_contract(_good_report(), where='unit:pass')
    assert summary['status'] == 'PASS'
    assert summary['n_contracts'] == 1


@pytest.mark.parametrize('field, value', [
    ('rows_match_registered', False),
    ('pit_violations_strict_lt', 1),
    ('duplicate_focal_month_keys', 2),
    ('missing_filled_with_zero', 3),
])
def test_each_hard_gate_fails(field, value):
    with pytest.raises(SystemExit) as ei:
        assert_m1_contract(_good_report(**{field: value}), where=f'unit:{field}')
    assert ei.value.code == 2


def test_missing_gate_field_fails():
    contract = dict(GOOD)
    contract.pop('pit_violations_strict_lt')
    with pytest.raises(SystemExit) as ei:
        assert_m1_contract({'contracts': {'ma_teacher': contract}}, where='unit:missing')
    assert ei.value.code == 2


def test_unknown_source_fails():
    with pytest.raises(SystemExit) as ei:
        assert_m1_contract({'not_a_contract': 1}, where='unit:unknown')
    assert ei.value.code == 2


# ---------------------------------------------------------------- 当前真实产物（只读）
def test_current_real_build_report_passes(real_report):
    summary = assert_m1_contract(real_report, where='real:build_report')
    assert summary['n_contracts'] == 4
    for key, c in summary['contracts'].items():
        assert key in {'ma_teacher', 'mb_teacher', 'ma', 'mb'}
        assert c['rows_gate_enforced'] is True
        assert c['rows_match_registered'] is True
        assert c['pit_violations_strict_lt'] == 0
        assert c['duplicate_focal_month_keys'] == 0
        assert c['missing_filled_with_zero'] == 0


def test_current_real_build_report_path_passes():
    summary = assert_m1_contract(SAMPLE_REPORT, where='real:build_report_path')
    assert summary['n_contracts'] == 4


def test_current_real_v2_sample_passes(real_v2):
    summary = assert_m1_contract(real_v2, expected_rows=V2_MA_TEACHER_ROWS,
                                 kind='sample', where='real:v2_ma_teacher')
    c = summary['contracts']['<frame>']
    assert c['rows'] == V2_MA_TEACHER_ROWS
    assert c['rows_match_registered'] is True
    assert c['pit_violations_strict_lt'] == 0
    assert c['duplicate_focal_month_keys'] == 0
    assert c['missing_filled_with_zero'] == 0


# ---------------------------------------------------------------- 失败路径：截断行/重复键/PIT/零填
def test_truncated_sample_rows_fail(real_v2):
    truncated = real_v2.iloc[:-10].copy()
    with pytest.raises(SystemExit) as ei:
        assert_m1_contract(truncated, expected_rows=V2_MA_TEACHER_ROWS,
                           kind='sample', where='truncated:v2_ma_teacher')
    assert ei.value.code == 2


def test_duplicate_focal_month_keys_fail():
    df = _frame()
    df.loc[1, 'focal_id'] = df.loc[0, 'focal_id']       # 同一 month 下重复 focal_id
    contract = contract_from_frame(df, expected_rows=len(df), kind='sample')
    assert contract['duplicate_focal_month_keys'] == 1
    with pytest.raises(SystemExit) as ei:
        assert_m1_contract(df, expected_rows=len(df), kind='sample', where='dup')
    assert ei.value.code == 2


def test_pit_violation_fails():
    df = _frame()
    df.loc[0, 'available_at'] = '2020-01-31'             # >= month 首日 = 前视
    contract = contract_from_frame(df, expected_rows=len(df), kind='sample')
    assert contract['pit_violations_strict_lt'] == 1
    with pytest.raises(SystemExit) as ei:
        assert_m1_contract(df, expected_rows=len(df), kind='sample', where='pit')
    assert ei.value.code == 2


def test_missing_filled_with_zero_fails():
    df = _frame()
    df.loc[0, 'm1_exclusion_reason'] = 'no_network_support'
    df.loc[0, 'PeerMAXraw'] = 0.0                        # 应缺失却填 0
    contract = contract_from_frame(df, expected_rows=len(df), kind='sample')
    assert contract['missing_filled_with_zero'] == 1
    with pytest.raises(SystemExit) as ei:
        assert_m1_contract(df, expected_rows=len(df), kind='sample', where='zero')
    assert ei.value.code == 2


def test_custom_panel_rows_gate_not_applicable():
    contract = contract_from_frame(_frame(), expected_rows=None, kind='sample')
    assert contract['rows_match_registered'] == ROWS_NOT_APPLICABLE
    assert set(HARD_GATE_KEYS).issubset(contract)


# ---------------------------------------------------------------- 下游脚本包装：截断样本必须失败
def test_load_checked_sample_rejects_truncation(monkeypatch, real_v2):
    import s3_m1_2_fm_main as fm
    report = {'samples': {'ma_teacher': {'rows_total': V2_MA_TEACHER_ROWS}},
              'samples_v1_superseded': {'ma_teacher': {'rows_total': V2_MA_TEACHER_ROWS}}}
    monkeypatch.setattr(fm, 'load_sample',
                        lambda key, version='v2': real_v2.iloc[:-10].copy())
    with pytest.raises(SystemExit) as ei:
        fm.load_checked_sample('ma_teacher', 'v2', report)
    assert ei.value.code == 2


# ---------------------------------------------------------------- s3_m1_1 --force
def _bad_panel_contracts(module):
    return {k: {'rows': 1, 'rows_registered': 884, 'rows_match_registered': False,
                'pit_violations_strict_lt': 0, 'duplicate_focal_month_keys': 0,
                'missing_filled_with_zero': 0} for k in module.PANELS}


def test_s3_m1_1_force_without_reason_rejected():
    import s3_m1_1_build_sample as b1
    with pytest.raises(SystemExit) as ei:
        b1.parse_args(['--force'])
    assert ei.value.code == 2


def test_s3_m1_1_force_with_reason_bypasses_but_records():
    import s3_m1_1_build_sample as b1
    bad = _bad_panel_contracts(b1)
    with pytest.raises(SystemExit) as ei:
        b1.enforce_panel_contracts(bad, force=False, force_reason=None)
    assert ei.value.code == 2
    args = b1.parse_args(['--force', '--force-reason', 'unit-test recovery'])
    rec = b1.enforce_panel_contracts(bad, args.force, args.force_reason)
    assert rec['mode'] == 'FORCED'
    assert rec['all_pass'] is False
    assert rec['force_reason'] == 'unit-test recovery'
    assert set(rec['failing_columns']) == set(b1.PANELS)


# ---------------------------------------------------------------- 4 面板 + 8 样本（只读）
@pytest.fixture(scope='module')
def panel_contracts():
    import s3_m1_1_build_sample as b1
    original = b1.progress
    b1.progress = lambda msg: None
    try:
        return b1.check_panel_contracts()
    finally:
        b1.progress = original


def test_four_panel_contracts_pass(panel_contracts):
    import s3_m1_1_build_sample as b1
    rec = b1.enforce_panel_contracts(panel_contracts, force=False, force_reason=None)
    assert rec['mode'] == 'ENFORCED'
    assert rec['all_pass'] is True
    for key in b1.PANELS:
        c = panel_contracts[key]
        assert c['rows'] == c['rows_registered']
        assert c['rows_match_registered'] is True
        assert c['pit_violations_strict_lt'] == 0
        assert c['duplicate_focal_month_keys'] == 0
        assert c['missing_filled_with_zero'] == 0


@pytest.mark.parametrize('version', ['v2', 'v1'])
@pytest.mark.parametrize('key', ['ma_teacher', 'mb_teacher', 'ma', 'mb'])
def test_eight_sample_contracts_pass(real_report, key, version):
    expected = (real_report['samples'] if version == 'v2'
                else real_report['samples_v1_superseded'])[key]['rows_total']
    df = pd.read_parquet(sample_path(key, version), columns=V2_READ_COLS)
    summary = assert_m1_contract(df, expected_rows=int(expected), kind='sample',
                                 where=f'real:{key}:{version}')
    c = summary['contracts']['<frame>']
    assert c['rows'] == int(expected)
    assert c['rows_match_registered'] is True
    assert c['pit_violations_strict_lt'] == 0
    assert c['duplicate_focal_month_keys'] == 0
    assert c['missing_filled_with_zero'] == 0


# ---------------------------------------------------------------- s3_m1_6 版本/契约失败
def _fake_write_json(store):
    def _write(path, obj):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
        store.append(p)
    return _write


def _s3_m1_6_inputs(tmp_path, price_version='v2', sample_version='v2', contract=None):
    (tmp_path / 's3_m1_pricing_table.json').write_text(
        json.dumps({'main': {}, 'sample_version': price_version}), encoding='utf-8')
    (tmp_path / 's3_m1_sample_report.json').write_text(
        json.dumps({'samples': {}, 'sample_version': sample_version,
                    'contracts': {'ma_teacher': contract or dict(GOOD)}}), encoding='utf-8')


def test_s3_m1_6_version_mismatch_exits_without_report(monkeypatch, tmp_path):
    import s3_m1_6_gate_report as g6
    monkeypatch.setattr(g6, 'OUT_DIR', tmp_path)
    monkeypatch.setattr(g6, 'progress', lambda msg: None)
    written = []
    monkeypatch.setattr(g6, 'write_json', _fake_write_json(written))
    _s3_m1_6_inputs(tmp_path, price_version='v2', sample_version='v1')
    with pytest.raises(SystemExit) as ei:
        g6.main([])
    assert ei.value.code == 2
    assert written == []
    assert not (tmp_path / 's3_m1_gate_report.json').exists()


def test_s3_m1_6_contract_failure_writes_fail_status(monkeypatch, tmp_path):
    import s3_m1_6_gate_report as g6
    monkeypatch.setattr(g6, 'OUT_DIR', tmp_path)
    monkeypatch.setattr(g6, 'progress', lambda msg: None)
    written = []
    monkeypatch.setattr(g6, 'write_json', _fake_write_json(written))
    bad = dict(GOOD)
    bad['pit_violations_strict_lt'] = 1
    _s3_m1_6_inputs(tmp_path, price_version='v2', sample_version='v2', contract=bad)
    with pytest.raises(SystemExit) as ei:
        g6.main([])
    assert ei.value.code == 2
    assert len(written) == 1
    out = json.loads(written[0].read_text(encoding='utf-8'))
    assert out['contracts_status'] == 'FAIL'
    assert out['S5_status'] == 'BLOCKED'
    assert not (tmp_path / 's3_m1_gate_report.md').exists()


# ---------------------------------------------------------------- s3_pc 坏登记面板
def test_s3_pc_bad_registered_panel_exits_without_output(monkeypatch, tmp_path):
    import s3_pc_pricing_correspondence as pc
    report = {'contracts': {'ma_teacher': dict(GOOD)},
              'samples': {'ma_teacher': {'rows_total': 10}},
              'sample_version': 'unit-test'}
    report_path = tmp_path / 'sample_report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    panel_path = tmp_path / 'bad_panel.parquet'
    _frame().iloc[:-1].to_parquet(panel_path, index=False)   # 3 行 < 登记 10 行
    monkeypatch.setattr(pc, 'M1_SAMPLE_REPORT', report_path)
    monkeypatch.setattr(pc, 'PANEL_SAMPLE', panel_path)
    outdir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['s3_pc', '--out-dir', str(outdir),
                                      '--progress', str(tmp_path / 'progress.md')])
    with pytest.raises(SystemExit) as ei:
        pc.main()
    assert ei.value.code == 2
    assert not (outdir / 's3_pc_pricing_correspondence.json').exists()
    assert not (outdir / 's3_pc_pricing_correspondence.md').exists()

