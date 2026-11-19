import ast
import re
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from pricing_lib import (  # noqa: E402
    M1_INPUTS, assert_m1_inputs, check_input_contract, manifest_input_key,
)

GOOD = {'rows': 4, 'rows_registered': 4, 'rows_match_registered': True,
        'pit_violations_strict_lt': 0, 'duplicate_focal_month_keys': 0,
        'missing_filled_with_zero': 0}


def _write_csv(tmp_path: Path, name: str = 'input.csv', rows: int = 2,
               header: str = 'a,b') -> Path:
    p = tmp_path / name
    body = [header] + [f'{i},{i}' for i in range(rows)]
    p.write_text('\n'.join(body) + '\n', encoding='utf-8')
    return p


def _write_contracts(tmp_path: Path, entries: dict) -> Path:
    p = tmp_path / 'm1_contracts.yaml'
    p.write_text(yaml.safe_dump({'m1_inputs': entries}, allow_unicode=True,
                                sort_keys=False), encoding='utf-8')
    return p


def _entry(path: Path, rows=None, columns=None) -> dict:
    return {manifest_input_key(path): {'file': manifest_input_key(path),
                                       'rows': rows, 'columns': columns}}


def test_rows_match_passes(tmp_path: Path) -> None:
    p = _write_csv(tmp_path, rows=2)
    c = _write_contracts(tmp_path, _entry(p, rows=2, columns=['a', 'b']))
    rec = check_input_contract(p, contracts_path=c)
    assert rec['rows'] == 2 and rec['match'] is True and rec['columns_missing'] == []


def test_rows_mismatch_exits(tmp_path: Path, capsys) -> None:
    p = _write_csv(tmp_path, rows=3)
    c = _write_contracts(tmp_path, _entry(p, rows=2))
    with pytest.raises(SystemExit) as ei:
        check_input_contract(p, contracts_path=c)
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert '输入行数不匹配' in err and '期望=2 实际=3' in err


def test_missing_column_exits(tmp_path: Path, capsys) -> None:
    p = _write_csv(tmp_path, header='a,b')
    c = _write_contracts(tmp_path, _entry(p, rows=2, columns=['a', 'c']))
    with pytest.raises(SystemExit) as ei:
        check_input_contract(p, contracts_path=c)
    assert ei.value.code == 2
    assert "缺=['c']" in capsys.readouterr().err


def test_missing_file_exits(tmp_path: Path, capsys) -> None:
    p = tmp_path / 'nope.csv'
    c = _write_contracts(tmp_path, _entry(p, rows=1))
    with pytest.raises(SystemExit) as ei:
        check_input_contract(p, contracts_path=c)
    assert ei.value.code == 2
    assert '输入不存在' in capsys.readouterr().err


def test_unregistered_rows_exit(tmp_path: Path, capsys) -> None:
    p = _write_csv(tmp_path, rows=5)
    other = _write_csv(tmp_path, name='other.csv')
    c = _write_contracts(tmp_path, _entry(other, rows=2))
    with pytest.raises(SystemExit) as ei:
        check_input_contract(p, contracts_path=c)
    assert ei.value.code == 2
    assert '未登记行数' in capsys.readouterr().err


def test_subset_and_empty_rejected(tmp_path: Path, capsys) -> None:
    p = _write_csv(tmp_path)
    c = _write_contracts(tmp_path, _entry(p, rows=2))
    with pytest.raises(SystemExit) as ei:
        assert_m1_inputs({'input': p}, manifest_path=None, spec_path=None)
    assert ei.value.code == 2
    assert '非测试调用必须校验全部' in capsys.readouterr().err
    for kwargs in ({}, {'_allow_subset': True}):
        with pytest.raises(SystemExit) as ei2:
            assert_m1_inputs({}, **kwargs)
        assert ei2.value.code == 2
        assert '拒绝 0/0 PASS' in capsys.readouterr().err


def test_parquet_supported(tmp_path: Path) -> None:
    p = tmp_path / 'panel.parquet'
    pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]}).to_parquet(p, index=False)
    c = _write_contracts(tmp_path, _entry(p, rows=3, columns=['a', 'b']))
    rec = check_input_contract(p, contracts_path=c)
    assert rec['rows'] == 3 and rec['columns_missing'] == []


def test_ledger_fields(tmp_path: Path) -> None:
    from pricing_lib import M1_INPUTS
    p = _write_csv(tmp_path, rows=2)
    key = M1_INPUTS['monthly_returns_clean_v2']
    c = _write_contracts(tmp_path, {key: {'file': key, 'rows': 2, 'columns': ['a', 'b']}})
    rec = assert_m1_inputs({'monthly_returns_clean_v2': p}, _allow_subset=True,
                               contracts_path=c)[0]
    assert set(rec) >= {'input', 'file', 'path', 'rows', 'rows_registered',
                        'rows_match_registered', 'columns_missing', 'match'}
    assert rec['input'] == 'monthly_returns_clean_v2'


def test_manifest_input_key_is_posix_relative(tmp_path: Path) -> None:
    p = tmp_path / 'a' / 'b.csv'
    p.parent.mkdir(parents=True)
    p.write_text('x\n', encoding='utf-8')
    key = manifest_input_key(p)
    assert '\\' not in key


def test_no_sha_refresh_function() -> None:
    pat = re.compile(r'(sha_refresh|refresh_sha)', re.I)
    hits = []
    for p in sorted((ROOT / 'scripts').rglob('*.py')):
        text = p.read_text(encoding='utf-8', errors='ignore')
        if not pat.search(text):
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and pat.search(node.name):
                hits.append(f'{p.relative_to(ROOT)}:{node.name}')
    assert hits == [], f'发现 sha 刷新生产函数: {hits}'
