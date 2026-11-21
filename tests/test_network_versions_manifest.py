"""Phase 1 I-9 护栏：panel Step2 网络版本必须来自 manifest `network_versions` 白名单。

覆盖：
  1. manifest 顶层块结构、登记目录集合（4 主目录 + 6 个有报告变体，共 10）、
     pilot `data/processed/network` 不登记。
  2. 当前每个已登记目录实际 `{year}_similarity.csv` 集合 == manifest years；
     4 主目录与旧 glob 结果逐目录一致（36 年）。
  3. 临时目录：多一个 2013 文件 / 清单少一年 / 实际缺一年 / 未知目录 /
     非法 version_prefix / 非法 years / 非法 available_at_rule /
     非年份 similarity 文件 → `SystemExit(2)`，不被静默当文本年。
"""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/peermax_panel_build.py'
MANIFEST = ROOT / 'configs/version_manifest.yaml'
NET_ROOT = ROOT / 'data/processed'
# 主口径两列（C7：自建语料的 network_ma / network_mb 已退役，不再要求其存在）
MAIN_DIRS = ('network_ma_teacher', 'network_mb_teacher')
# Phase 3 shadow 网络目录（`artifacts/phase3/3.0_plan.md`）：**故意不登记** manifest
# （Phase 3 不改 configs；3.6 用注入 `version_spec` 装载）。仅此白名单允许「目录存在但未登记」。
SHADOW_NET_DIRS = ('network_ma_teacher_p3asof',                                    # 3.1（N1）
                   'network_ma_teacher_p3yt', 'network_ma_teacher_p3ydf',           # 3.2（N2/N3）
                   'network_ma_teacher_p3yt_ydf', 'network_ma_teacher_p3yt_latest')  # 3.2（N4/N5）
# 占位 τ 骨架目录（network_mda_edges_build 的 tau_mode=placeholder 输出）：中间态，
# 不是可被定价链消费的网络版本，同样不登记 manifest。
STUB_NET_DIRS = ('network_mb_stub',)
RULE = '(y+1)-04-30'


def _load():
    spec = importlib.util.spec_from_file_location('peermax_panel_build',
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))


def _write_similarity(net_dir, ty, name=None):
    (net_dir / (name or f'{ty}_similarity.csv')).write_text(
        'focal_id,source_id,weight\n', encoding='utf-8')


def _write_version(net_dir, ty):
    _write_similarity(net_dir, ty)
    (net_dir / f'{ty}_build_report.json').write_text(
        json.dumps({'available_at': f'{ty + 1}-04-30',
                    'network_version': f'pm_test_v1_{ty}',
                    'text_year': ty}), encoding='utf-8')


def _spec(years, prefix='pm_test_v1', rule=RULE):
    return {'version_prefix': prefix, 'years': years, 'available_at_rule': rule}


def _assert_exit2(call, capsys, *fragments):
    with pytest.raises(SystemExit) as ei:
        call()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    for frag in fragments:
        assert frag in err, (frag, err)
    return err


@pytest.fixture()
def mod():
    return _load()


# ---------------------------------------------------------------- manifest 结构
def test_manifest_network_versions_block():
    doc = _manifest()
    assert doc['manifest_version'] == '1.0.110'
    specs = doc['network_versions']
    assert isinstance(specs, dict) and specs
    for name, spec in specs.items():
        assert isinstance(spec, dict), name
        assert isinstance(spec.get('version_prefix'), str) and spec['version_prefix']
        assert isinstance(spec.get('years'), list) and spec['years']
        assert spec.get('available_at_rule') == RULE


def test_manifest_covers_all_dirs_with_build_reports():
    if not list(NET_ROOT.glob('network*')):
        pytest.skip('缺网络产物目录（先跑网络构边）')
    specs = _manifest()['network_versions']
    reported = {d.name for d in NET_ROOT.glob('network*')
                if d.is_dir() and list(d.glob('*_build_report.json'))}
    # Phase 3 shadow 网络目录（`artifacts/phase3/3.0_plan.md` §版本治理）：**故意不登记** manifest
    # （Phase 3 不改 configs）；headline 与既有已登记变体的「目录 ⊆ manifest」断言逐值不变。
    # 占位 τ 骨架目录同理：中间态，不是可消费的网络版本。
    assert set(SHADOW_NET_DIRS).isdisjoint(specs), '已登记的目录不得被当作 shadow 跳过'
    reported = reported - set(SHADOW_NET_DIRS) - set(STUB_NET_DIRS)
    # 在盘的已登记目录必须是 manifest 的子集（未登记目录不得偷偷产出边表）；
    # manifest 里其余登记项只是"还没跑"，不是口径缺陷。
    assert reported <= set(specs), f'出现未登记的边表目录：{sorted(reported - set(specs))}'
    assert set(MAIN_DIRS) <= reported, f'主口径两列必须已落盘：{sorted(set(MAIN_DIRS) - reported)}'


def test_pilot_network_not_registered(mod, capsys):
    assert 'network' not in _manifest()['network_versions']
    _assert_exit2(lambda: mod.load_versions(NET_ROOT / 'network'), capsys,
                  '未登记的网络目录', 'network_versions')


# ---------------------------------------------------------------- 真实目录对账
def test_real_dirs_match_manifest_years(mod):
    if not list(NET_ROOT.glob('network*')):
        pytest.skip('缺网络产物目录（先跑网络构边）')
    specs = _manifest()['network_versions']
    for name, spec in specs.items():
        d = NET_ROOT / name
        if not d.is_dir():
            continue     # 未跑出来的登记目录不算口径缺陷（由 test_manifest_covers 兜上限）
        actual = {int(p.stem.split('_')[0])
                  for p in d.glob('*_similarity.csv')}
        assert actual == set(spec['years']), name
        vers = mod.load_versions(d)
        assert set(vers) == set(spec['years']), name


def test_four_main_dirs_match_legacy_glob(mod):
    total = 0
    for name in MAIN_DIRS:
        legacy = {int(p.stem.split('_')[0])
                  for p in (NET_ROOT / name).glob('*_similarity.csv')}
        vers = mod.load_versions(NET_ROOT / name)
        assert set(vers) == legacy, name
        assert len(vers) == 9, name
        total += len(vers)
    assert total == 18


def test_manifest_prefix_matches_build_report_network_version():
    specs = _manifest()['network_versions']
    for name, spec in specs.items():
        for ty in spec['years']:
            rp = NET_ROOT / name / f'{ty}_build_report.json'
            doc = json.loads(rp.read_text(encoding='utf-8'))
            assert doc['network_version'] == f"{spec['version_prefix']}_{ty}", (
                name, ty, doc.get('network_version'))


# ---------------------------------------------------------------- 正常路径
def test_valid_mock_spec_passes(mod, tmp_path):
    _write_version(tmp_path, 2019)
    _write_version(tmp_path, 2020)
    vers = mod.load_versions(tmp_path, _spec([2019, 2020]))
    assert sorted(vers) == [2019, 2020]
    assert vers[2019]['available_at'] == '2020-04-30'


# ---------------------------------------------------------------- 失败路径
def test_extra_actual_year_rejected(mod, tmp_path, capsys):
    """临时目录多一个 2013_similarity.csv：不得被静默当文本年。"""
    _write_version(tmp_path, 2013)
    _write_version(tmp_path, 2019)
    _assert_exit2(lambda: mod.load_versions(tmp_path, _spec([2019])), capsys,
                  '未登记年份 2013', '2013_similarity.csv')


def test_manifest_missing_year_rejected(mod, tmp_path, capsys):
    """清单少一年（实际 2019+2020，清单只写 2019）→ 退出。"""
    _write_version(tmp_path, 2019)
    _write_version(tmp_path, 2020)
    _assert_exit2(lambda: mod.load_versions(tmp_path, _spec([2019])), capsys,
                  '未登记年份 2020', '2020_similarity.csv')


def test_actual_missing_manifest_year_rejected(mod, tmp_path, capsys):
    """清单有 2020 但目录缺 2020_similarity.csv → 退出（缺失也报）。"""
    _write_version(tmp_path, 2019)
    _assert_exit2(lambda: mod.load_versions(tmp_path, _spec([2019, 2020])),
                  capsys, '清单年份缺', '[2020]')


def test_unknown_net_dir_rejected(mod, tmp_path, capsys):
    _write_version(tmp_path, 2019)
    _assert_exit2(lambda: mod.load_versions(tmp_path), capsys,
                  '未登记的网络目录', 'network_versions')


def test_non_year_similarity_file_rejected(mod, tmp_path, capsys):
    _write_version(tmp_path, 2019)
    _write_similarity(tmp_path, 2014, name='2014_similarity_old.csv')
    _assert_exit2(lambda: mod.load_versions(tmp_path, _spec([2019])), capsys,
                  '非 {year}_similarity.csv 文件', '2014_similarity_old.csv')


def test_manifest_entry_not_mapping_rejected(mod, tmp_path, capsys):
    _assert_exit2(lambda: mod.load_versions(tmp_path, ['2019']), capsys,
                  '清单条目不是映射')


@pytest.mark.parametrize('prefix', [
    '', 'PM_MA_v1', 'pm_ma', 'pm ma v1', 'pm_ma_v1_2019', 123, None,
])
def test_invalid_version_prefix_rejected(mod, tmp_path, capsys, prefix):
    _assert_exit2(lambda: mod.load_versions(tmp_path, _spec([2019], prefix=prefix)),
                  capsys, 'version_prefix 非法')


@pytest.mark.parametrize('years', [
    '2019', [], [2019, 2019], ['2019'], [2019.0], [999], [10000], [True], None,
])
def test_invalid_years_rejected(mod, tmp_path, capsys, years):
    _assert_exit2(lambda: mod.load_versions(tmp_path, _spec(years)), capsys,
                  'years')


@pytest.mark.parametrize('rule', [None, 'y+1-04-30', '{text_year+1}-04-30'])
def test_invalid_available_at_rule_rejected(mod, tmp_path, capsys, rule):
    _assert_exit2(lambda: mod.load_versions(tmp_path, _spec([2019], rule=rule)),
                  capsys, 'available_at_rule')


def test_manifest_without_block_rejected(mod, tmp_path, monkeypatch, capsys):
    fake = tmp_path / 'manifest.yaml'
    fake.write_text('manifest_version: "x"\n', encoding='utf-8')
    monkeypatch.setattr(mod, 'MANIFEST', fake)
    net = tmp_path / 'network_ma'
    net.mkdir()
    _assert_exit2(lambda: mod.load_versions(net), capsys,
                  '缺少顶层 network_versions')
