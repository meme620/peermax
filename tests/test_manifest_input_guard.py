"""Phase 1 I-3：M1 主输入 / 下游样本 manifest sha 护栏测试。

真实输入只读验证当前全部 MATCH；失败路径全部使用 tmp_path 合成文件/临时 manifest，
不写入、不替换任何真实数据文件，不重跑 M1/PC/S4。
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'data_audit'))

from s3_m1_lib import (  # noqa: E402
    M1_INPUTS, assert_anchored_file_sha, assert_m1_inputs_sha, manifest_input_key,
)

GOOD = {'rows': 4, 'rows_registered': 4, 'rows_match_registered': True,
        'pit_violations_strict_lt': 0, 'duplicate_focal_month_keys': 0,
        'missing_filled_with_zero': 0}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_input(tmp_path: Path, content: bytes = b'a,b\n1,2\n') -> Path:
    p = tmp_path / 'input.csv'
    p.write_bytes(content)
    return p


def _entry(path: Path, sha=None, status: str = 'active') -> dict:
    e = {'file': manifest_input_key(path), 'version': 'test',
         'date': '2026-10-28', 'status': status, 'role': 'test'}
    if sha is not None:
        e['sha256'] = sha
    return e


def _write_manifest(tmp_path: Path, entries: list) -> Path:
    p = tmp_path / 'version_manifest.yaml'
    p.write_text(yaml.safe_dump({'manifest_version': 'test', 'deliverables': entries},
                                allow_unicode=True, sort_keys=False), encoding='utf-8')
    return p


def _write_spec(tmp_path: Path, input_hashes: dict) -> Path:
    p = tmp_path / 'spec.yaml'
    p.write_text(yaml.safe_dump({'spec_version': 'test',
                                 'provenance': {'input_hashes': input_hashes}},
                                allow_unicode=True, sort_keys=False), encoding='utf-8')
    return p


# ---------------------------------------------------------------- 当前真实 9 输入（只读）
def test_current_inputs_pass():
    ledger = assert_m1_inputs_sha()
    assert len(ledger) == len(M1_INPUTS) == 9
    assert {r['input'] for r in ledger} == set(M1_INPUTS)
    assert all(r['match'] for r in ledger)
    assert all(len(r['sha256']) == 64 and r['sha256'] == r['anchor_sha256'] for r in ledger)
    assert all(r['anchor_source'] == 'manifest' for r in ledger)
    by_name = {r['input']: r for r in ledger}
    # 4 个输入同时有 manifest + spec 锚，已交叉校验且一致。
    for name in ('monthly_returns_adj_full_2006_2025', 'market_equity_panel',
                 'firm_id_map', 'firm_chars_clean'):
        assert by_name[name]['manifest_sha256'] == by_name[name]['spec_sha256']
        assert by_name[name]['anchor_sha256'] == by_name[name]['manifest_sha256']


def test_postadj_included_as_ninth_input():
    assert 'monthly_returns_adj_full_2006_2025' in M1_INPUTS
    ledger = assert_m1_inputs_sha()
    rec = {r['input']: r for r in ledger}['monthly_returns_adj_full_2006_2025']
    assert rec['sha256'] == '5525fcf805ec83e00966bcde159ef7dc0b8a6a587feb447215c0a4cd3259c770'
    assert rec['match'] is True


# ---------------------------------------------------------------- 失败路径（tmp）
def test_byte_flip_fails(tmp_path, capsys):
    p = _write_input(tmp_path)
    m = _write_manifest(tmp_path, [_entry(p, _sha(p))])
    assert len(assert_m1_inputs_sha({'input': p}, manifest_path=m, _allow_subset=True)) == 1

    data = bytearray(p.read_bytes())
    data[0] ^= 0xFF                                                       # 改 1 字节
    p.write_bytes(bytes(data))
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs_sha({'input': p}, manifest_path=m, _allow_subset=True)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert 'sha256 不匹配' in err
    assert f'file={manifest_input_key(p)}' in err
    assert '期望=' in err and '实际=' in err


def test_missing_manifest_entry_fails(tmp_path, capsys):
    p = _write_input(tmp_path)
    other = tmp_path / 'other.csv'
    other.write_bytes(b'x')
    m = _write_manifest(tmp_path, [_entry(other, _sha(other))])
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs_sha({'input': p}, manifest_path=m, _allow_subset=True)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert '无锚' in err
    assert f'file={manifest_input_key(p)}' in err


def test_missing_sha_fails(tmp_path, capsys):
    p = _write_input(tmp_path)
    m = _write_manifest(tmp_path, [_entry(p)])                           # 条目存在但无 sha256
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs_sha({'input': p}, manifest_path=m, _allow_subset=True)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert '无锚' in err and '均未登记 sha256' in err
    assert f'file={manifest_input_key(p)}' in err


def test_missing_input_file_fails(tmp_path, capsys):
    p = tmp_path / 'does_not_exist.csv'
    m = _write_manifest(tmp_path, [_entry(p, '0' * 64)])
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs_sha({'input': p}, manifest_path=m, _allow_subset=True)
    assert ei.value.code == 2
    assert '文件不存在' in capsys.readouterr().err


def test_non_active_entry_fails(tmp_path, capsys):
    p = _write_input(tmp_path)
    m = _write_manifest(tmp_path, [_entry(p, _sha(p), status='superseded')])
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs_sha({'input': p}, manifest_path=m, _allow_subset=True)
    assert ei.value.code == 2
    assert "status='superseded' != active" in capsys.readouterr().err


def test_subset_rejected_without_test_flag(tmp_path, capsys):
    p = _write_input(tmp_path)
    m = _write_manifest(tmp_path, [_entry(p, _sha(p))])
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs_sha({'input': p}, manifest_path=m)              # 生产语义：必须是全集
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert '非测试调用必须校验全部' in err
    assert '缺少=' in err and '多余=' in err


def test_empty_inputs_rejected(tmp_path, capsys):
    m = _write_manifest(tmp_path, [_entry(tmp_path / 'other.csv', '0' * 64)])
    for kwargs in ({}, {'_allow_subset': True}):
        with pytest.raises(SystemExit) as ei:
            assert_m1_inputs_sha({}, manifest_path=m, **kwargs)
        assert ei.value.code == 2
        assert '拒绝 0/0 PASS' in capsys.readouterr().err


def test_ledger_has_expected_actual_and_match(tmp_path):
    p = _write_input(tmp_path)
    m = _write_manifest(tmp_path, [_entry(p, _sha(p))])
    rec = assert_m1_inputs_sha({'input': p}, manifest_path=m, _allow_subset=True)[0]
    assert rec['match'] is True
    assert rec['sha256'] == rec['actual_sha256'] == rec['expected_sha256']
    assert rec['anchor_sha256'] == rec['expected_sha256']
    for key in ('input', 'file', 'path', 'anchor_source'):
        assert key in rec


def test_conflicting_anchors_fail(tmp_path, capsys):
    p = _write_input(tmp_path)
    key = manifest_input_key(p)
    m = _write_manifest(tmp_path, [_entry(p, 'a' * 64)])
    s = _write_spec(tmp_path, {key: 'b' * 64})
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs_sha({'input': p}, manifest_path=m, spec_path=s,
                             _allow_subset=True)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert 'manifest/spec 锚冲突' in err
    assert f'file={key}' in err and 'manifest=' in err and 'spec=' in err


def test_no_sha_refresh_function():
    """生产脚本不得提供 manifest sha 自动刷新函数（防“读不到就重算/刷新”）。"""
    pat = re.compile(r'(sha_refresh|refresh_sha)', re.I)
    hits = []
    for p in sorted((ROOT / 'scripts').rglob('*.py')):
        text = p.read_text(encoding='utf-8', errors='ignore')
        if not pat.search(text):
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and pat.search(node.name):
                hits.append(f'{p.relative_to(ROOT)}:{node.name}')
    assert hits == [], f'发现 sha 刷新生产函数: {hits}'


def test_unanchored_file_rejected(tmp_path, capsys):
    p = _write_input(tmp_path)
    other = tmp_path / 'other.csv'
    other.write_bytes(b'x')
    m = _write_manifest(tmp_path, [_entry(other, _sha(other))])
    s = _write_spec(tmp_path, {'tmp/other.csv': '0' * 64})
    with pytest.raises(SystemExit) as ei:
        assert_anchored_file_sha(p, manifest_key='tmp/no_anchor.csv',
                                 manifest_path=m, spec_path=s)
    assert ei.value.code == 2
    assert '无锚' in capsys.readouterr().err


# ---------------------------------------------------------------- 下游 s4_step2
def _s4_argv(tmp_path: Path):
    return ['s4_step2', '--use-case', 'ma',
            '--out', str(tmp_path / 'out.csv'), '--report', str(tmp_path / 'report.json')]


def test_s4_step2_bad_source_month_sha_exits_without_output(monkeypatch, tmp_path, capsys):
    import s4_step2_peermax_formal as s4
    bad = tmp_path / 'bad_source.csv'
    bad.write_text('not the registered source month\n', encoding='utf-8')
    monkeypatch.setattr(s4, 'ARGMAX', bad)
    monkeypatch.setattr(s4, 'assert_m1_inputs_sha', lambda: [])           # 隔离单文件路径校验
    monkeypatch.setattr(sys, 'argv', _s4_argv(tmp_path))
    with pytest.raises(SystemExit) as ei:
        s4.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert 'source_month_argmax_panel' in err and '期望=' in err and '实际=' in err
    assert not (tmp_path / 'out.csv').exists()
    assert not (tmp_path / 'report.json').exists()


def test_s4_step2_bad_returns_sha_exits_without_output(monkeypatch, tmp_path, capsys):
    import s4_step2_peermax_formal as s4
    bad = tmp_path / 'bad_returns.csv'
    bad.write_text('not the registered returns file\n', encoding='utf-8')
    monkeypatch.setattr(s4, 'RETURNS', bad)
    monkeypatch.setattr(s4, 'assert_m1_inputs_sha', lambda: [])
    monkeypatch.setattr(sys, 'argv', _s4_argv(tmp_path))
    with pytest.raises(SystemExit) as ei:
        s4.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert 'monthly_returns_clean_v2' in err and '期望=' in err and '实际=' in err
    assert not (tmp_path / 'out.csv').exists()
    assert not (tmp_path / 'report.json').exists()


# ---------------------------------------------------------------- 下游 s3_pc
def test_s3_pc_bad_sample_sha_exits_without_output(monkeypatch, tmp_path, capsys):
    import s3_pc_pricing_correspondence as pc
    report = {'contracts': {'ma_teacher': dict(GOOD)},
              'samples': {'ma_teacher': {'rows_total': 10}},
              'files': {'v2_samples': {'ma_teacher': {'sha256': '0' * 64}}}}
    report_path = tmp_path / 'sample_report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    panel = tmp_path / 'bad_panel.parquet'
    panel.write_bytes(b'not a parquet file')
    monkeypatch.setattr(pc, 'M1_SAMPLE_REPORT', report_path)
    monkeypatch.setattr(pc, 'PANEL_SAMPLE', panel)
    outdir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['s3_pc', '--out-dir', str(outdir),
                                      '--progress', str(tmp_path / 'progress.md')])
    with pytest.raises(SystemExit) as ei:
        pc.main()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert '样本 sha256 不匹配' in err and '期望=' in err and '实际=' in err
    assert not (outdir / 's3_pc_pricing_correspondence.json').exists()
    assert not (outdir / 's3_pc_pricing_correspondence.md').exists()
    assert not (tmp_path / 'progress.md').exists()      # guard 在 psay 之前


def test_s3_pc_custom_panel_skips_sample_anchor(monkeypatch, tmp_path):
    import s3_pc_pricing_correspondence as pc
    report = {'contracts': {'ma_teacher': dict(GOOD)},
              'samples': {'ma_teacher': {'rows_total': 10}}}
    report_path = tmp_path / 'sample_report.json'
    report_path.write_text(json.dumps(report), encoding='utf-8')
    panel = tmp_path / 'bad_panel.parquet'
    panel.write_bytes(b'not a parquet file')
    monkeypatch.setattr(pc, 'M1_SAMPLE_REPORT', report_path)
    monkeypatch.setattr(pc, 'PANEL_SAMPLE', panel)
    monkeypatch.setattr(sys, 'argv', ['s3_pc', '--panel', str(panel),
                                      '--out-dir', str(tmp_path / 'out'),
                                      '--progress', str(tmp_path / 'progress.md')])
    # 自定义面板不走样本锚：仍会在读取 parquet 时因坏文件失败，但不是 sha 锚错误。
    with pytest.raises(Exception):
        pc.main()
