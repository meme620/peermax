"""N-10 护栏：S4 Peermax 面板输出路径不得因未登记 net_dir 落回主件名。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/peermax_panel_build.py'


def _load():
    spec = importlib.util.spec_from_file_location('peermax_panel_build', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fulltext_suffix_registered():
    mod = _load()
    assert mod.NET_DIR_SUFFIX['network_ma_fulltext'] == '_fulltext'


def test_fulltext_default_paths_are_isolated():
    mod = _load()
    args = SimpleNamespace(use_case='ma', out=None, report=None)
    out, rp = mod.resolve_outputs(args, ROOT / 'data/processed/network_ma_fulltext')
    assert out.name == 'monthly_pricing_panel_ma_fulltext.csv'
    assert rp.name == 's4_step2_ma_fulltext_report.json'


def test_canonical_defaults_unchanged():
    mod = _load()
    args = SimpleNamespace(use_case='ma', out=None, report=None)
    out, rp = mod.resolve_outputs(args, ROOT / 'data/processed/network_ma')
    assert out.name == 'monthly_pricing_panel_ma.csv'
    assert rp.name == 's4_step2_ma_report.json'
    out, rp = mod.resolve_outputs(args, ROOT / 'data/processed/network_ma_teacher')
    assert out.name == 'monthly_pricing_panel_ma_teacher.csv'
    assert rp.name == 's4_step2_ma_teacher_report.json'
    out, rp = mod.resolve_outputs(SimpleNamespace(use_case='mb', out=None, report=None),
                                  ROOT / 'data/processed/network_mb')
    assert out.name == 'monthly_pricing_panel_mb.csv'
    assert rp.name == 's4_step2_mb_report.json'


def test_unknown_net_dir_without_explicit_paths_refuses():
    mod = _load()
    args = SimpleNamespace(use_case='ma', out=None, report=None)
    with pytest.raises(SystemExit):
        mod.resolve_outputs(args, ROOT / 'data/processed/network_ma_teacher_df10')


def test_unknown_net_dir_with_explicit_paths_allowed():
    mod = _load()
    args = SimpleNamespace(use_case='ma', out='data/processed/tmp_n10.csv',
                           report='reports/data_gates/G0/tmp_n10.json')
    out, rp = mod.resolve_outputs(args, ROOT / 'data/processed/network_ma_teacher_df10')
    assert out == ROOT / 'data/processed/tmp_n10.csv'
    assert rp == ROOT / 'reports/data_gates/G0/tmp_n10.json'


def test_registry_panel_versions_registered():
    """W1 bundle：--panel-version 缺省来源 = registry panel_versions（4 个 corpus）。"""
    mod = _load()
    pv = mod.load_registry_panel_versions()
    assert set(pv) >= {'ma_teacher', 'mb_teacher', 'ma', 'mb'}
    assert pv['ma_teacher'] == 's4_panel_ma_teacher_2015_01_2023_12_v1'
    assert pv['ma'] == 's4_panel_ma_2018_01_2025_12_v1'


def test_v4_outputs_are_new_filenames_not_v3():
    """v4 面板必须写新文件名，不得覆盖 v3 主件名。"""
    mod = _load()
    args = SimpleNamespace(use_case='ma', out='data/processed/monthly_pricing_panel_ma_v4.csv',
                           report='reports/data_gates/G0/s4_step2_ma_v4_report.json')
    out, rp = mod.resolve_outputs(args, ROOT / 'data/processed/network_ma')
    assert out.name == 'monthly_pricing_panel_ma_v4.csv'
    assert out.name != 'monthly_pricing_panel_ma.csv'
    assert rp.name == 's4_step2_ma_v4_report.json'
