"""preflight 的路径判定与退出码自检。"""

from __future__ import annotations

import pathlib

import pytest

import preflight


VALID = 'key: value\n'


def make_configs(root: pathlib.Path, names: list[str]) -> None:
    (root / 'configs').mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / 'configs' / name).write_text(VALID, encoding='utf-8')


def test_configs_全部就位摆通过(tmp_path: pathlib.Path) -> None:
    make_configs(tmp_path, preflight.CONFIGS)
    ok, lines = preflight.check_configs(tmp_path)
    assert ok is True
    assert all('[ok]' in line for line in lines)


def test_configs_缺一个要报出文件名(tmp_path: pathlib.Path) -> None:
    missing = preflight.CONFIGS[0]
    make_configs(tmp_path, preflight.CONFIGS[1:])
    ok, lines = preflight.check_configs(tmp_path)
    assert ok is False
    assert any(missing in line and '[缺失]' in line for line in lines)


def test_configs_yaml_坏了算失败(tmp_path: pathlib.Path) -> None:
    make_configs(tmp_path, preflight.CONFIGS)
    (tmp_path / 'configs' / preflight.CONFIGS[0]).write_text('a: [1,\n', encoding='utf-8')
    ok, lines = preflight.check_configs(tmp_path)
    assert ok is False
    assert any('[解析失败]' in line for line in lines)


def test_data_只报告不判失败(tmp_path: pathlib.Path) -> None:
    (tmp_path / 'data/raw/drive').mkdir(parents=True)
    (tmp_path / 'data/raw/drive/x.pkl').write_bytes(b'x')
    ok, lines = preflight.check_data(tmp_path)
    assert ok is True
    assert any('[就绪] data/raw/drive: 1 个文件' in line for line in lines)
    assert any('raw 目录 1/14 就绪' in line for line in lines)


def test_run_输出分块且结论随env失败(tmp_path: pathlib.Path) -> None:
    make_configs(tmp_path, preflight.CONFIGS)
    ok, text = preflight.run(['configs'], tmp_path)
    assert ok is True
    assert '== 配置 ==' in text
    assert '== 数据 ==' not in text
