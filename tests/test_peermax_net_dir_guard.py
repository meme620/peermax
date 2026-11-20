"""N-10 护栏：panel Peermax 面板输出路径不得因未登记 net_dir 落回主件名。"""
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
    assert rp.name == 'peermax_panel_build_ma_fulltext_report.json'


def test_canonical_defaults_unchanged():
    """主口径两列的默认输出名（C7：自建语料 ma/mb 两列已退役，不再有默认输出）。"""
    mod = _load()
    out, rp = mod.resolve_outputs(SimpleNamespace(use_case='ma', out=None, report=None),
                                  ROOT / 'data/processed/network_ma_teacher')
    assert out.name == 'monthly_pricing_panel_ma_teacher.csv'
    assert rp.name == 'peermax_panel_build_ma_teacher_report.json'
    out, rp = mod.resolve_outputs(SimpleNamespace(use_case='mb', out=None, report=None),
                                  ROOT / 'data/processed/network_mb_teacher')
    assert out.name == 'monthly_pricing_panel_mb_teacher.csv'
    assert rp.name == 'peermax_panel_build_mb_teacher_report.json'


def test_retired_self_corpus_dir_requires_explicit_paths():
    """已退役的自建语料目录不再有默认输出名：必须显式给 --out/--report。"""
    mod = _load()
    args = SimpleNamespace(use_case='ma', out=None, report=None)
    with pytest.raises(SystemExit):
        mod.resolve_outputs(args, ROOT / 'data/processed/network_ma')


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
    """W1 bundle：--panel-version 缺省来源 = registry panel_versions（4 个 corpus）。

    版本串必须带窗口，且窗口末端锚到 registry sample_end；窗口不写死在测试里。
    """
    import re

    import yaml

    mod = _load()
    pv = mod.load_registry_panel_versions()
    assert set(pv) >= {'ma_teacher', 'mb_teacher', 'ma', 'mb'}
    reg = yaml.safe_load((ROOT / 'configs/parameter_registry.yaml').read_text(encoding='utf-8'))
    params = {p['name']: p['headline_value'] for p in reg['parameters']}
    end = params['sample_end'][:7].replace('-', '_').split('_')
    for key in ('ma_teacher', 'mb_teacher', 'ma', 'mb'):
        value = pv[key]
        assert re.fullmatch(r'panel_[a-z_]+_\d{4}_\d{2}_\d{4}_\d{2}_v1', value), value
        assert value.split('_')[-3:-1] == end, f'{key} 窗口末端 {value} 与 registry sample_end 不一致'


def test_v4_outputs_are_new_filenames_not_v3():
    """v4 面板必须写新文件名，不得覆盖 v3 主件名。"""
    mod = _load()
    args = SimpleNamespace(use_case='ma', out='data/processed/monthly_pricing_panel_ma_v4.csv',
                           report='reports/data_gates/G0/peermax_panel_build_ma_v4_report.json')
    out, rp = mod.resolve_outputs(args, ROOT / 'data/processed/network_ma')
    assert out.name == 'monthly_pricing_panel_ma_v4.csv'
    assert out.name != 'monthly_pricing_panel_ma.csv'
    assert rp.name == 'peermax_panel_build_ma_v4_report.json'
