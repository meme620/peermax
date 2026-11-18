"""Phase 1 I-7 护栏：S4 Step2 的 available_at 不得缺失、空值或伪造。

覆盖：
  1. 当前真实网络目录：4 个主目录 36/36、全部有 build_report 的目录 79/79 通过（只读）。
  2. 正常值通过，`select_version()` 行为不变。
  3. 临时副本：build_report 缺失 / 键缺失 / null / 空串 / 坏格式 / 非法日期 /
     规则不符 / 非法 JSON / 顶层非对象 → 非零退出（`SystemExit(2)`）。
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/data_audit/s4_step2_peermax_formal.py'
NET_ROOT = ROOT / 'data/processed'
MAIN_DIRS = ('network_ma', 'network_mb',
             'network_ma_teacher', 'network_mb_teacher')
# Phase 3 shadow 网络目录（`artifacts/phase3/3.0_plan.md`）：**故意不登记** manifest
# （Phase 3 不改 configs；3.6 用注入 version_spec 装载）。仅此白名单允许「目录存在但未登记」。
SHADOW_NET_DIRS = ('network_ma_teacher_p3asof',                                    # 3.1（N1）
                   'network_ma_teacher_p3yt', 'network_ma_teacher_p3ydf',           # 3.2（N2/N3）
                   'network_ma_teacher_p3yt_ydf', 'network_ma_teacher_p3yt_latest')  # 3.2（N4/N5）
SHADOW_VERSION_PREFIX_BY_DIR = {
    'network_ma_teacher_p3asof': 'pm_ma_ewcos_teacher_p3asof_v1',
    'network_ma_teacher_p3yt': 'pm_ma_ewcos_teacher_p3yt_v1',
    'network_ma_teacher_p3ydf': 'pm_ma_ewcos_teacher_p3ydf_v1',
    'network_ma_teacher_p3yt_ydf': 'pm_ma_ewcos_teacher_p3yt_ydf_v1',
    'network_ma_teacher_p3yt_latest': 'pm_ma_ewcos_teacher_p3yt_latest_v1',
}
SHADOW_VERSION_PREFIX = SHADOW_VERSION_PREFIX_BY_DIR['network_ma_teacher_p3asof']	# 兼容既有引用


def _load():
    spec = importlib.util.spec_from_file_location('s4_step2_peermax_formal',
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_similarity(net_dir, ty):
    (net_dir / f'{ty}_similarity.csv').write_text(
        'focal_id,source_id,weight\n', encoding='utf-8')


def _write_report(net_dir, ty, payload):
    if isinstance(payload, dict):
        payload = dict(payload)
        payload.setdefault('network_version', f'pm_test_v1_{ty}')
        payload.setdefault('text_year', ty)
    (net_dir / f'{ty}_build_report.json').write_text(
        json.dumps(payload), encoding='utf-8')


def _make_version(net_dir, ty=2019, available_at='2020-04-30'):
    _write_similarity(net_dir, ty)
    _write_report(net_dir, ty, {'available_at': available_at})


def _mock_spec(net_dir):
    """I-9 后 load_versions 需 manifest 条目；mock 目录显式注入等价年份清单。"""
    years = sorted(int(p.name.split('_')[0])
                   for p in net_dir.glob('*_similarity.csv'))
    return {'version_prefix': 'pm_test_v1', 'years': years,
            'available_at_rule': '(y+1)-04-30'}


def _load_versions(mod, net_dir):
    """真实登记目录走 manifest；tmp/mock 目录注入等价清单（断言语义不变）。"""
    if net_dir.parent == NET_ROOT:
        return mod.load_versions(net_dir)
    return mod.load_versions(net_dir, _mock_spec(net_dir))


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


def test_source_has_no_fabricated_fallback():
    """静态防回归：旧 `av = av or f'{ty+1}-04-30'` 兜底不得复活。"""
    src = SCRIPT.read_text(encoding='utf-8')
    assert 'av = av or' not in src
    assert "av or f'{ty + 1}-04-30'" not in src
    assert "av or f'{ty+1}-04-30'" not in src


# ---------------------------------------------------------------- 当前真实数据
def test_real_four_main_dirs_36_reports_pass(mod):
    total = 0
    for name in MAIN_DIRS:
        vers = _load_versions(mod, NET_ROOT / name)
        assert len(vers) == 9, name
        for ty, v in vers.items():
            assert v['available_at'] == f'{ty + 1}-04-30', (name, ty)
            assert v['path'].name == f'{ty}_similarity.csv'
        total += len(vers)
    assert total == 36


def test_real_all_network_dirs_79_reports_pass(mod):
    dirs = sorted(
        p for p in NET_ROOT.glob('network*')
        if p.is_dir() and list(p.glob('*_build_report.json')))
    assert len(dirs) == 10 + len(SHADOW_NET_DIRS)
    total = 0
    for d in dirs:
        # Phase 3 shadow 目录（`artifacts/phase3/3.0_plan.md` §版本治理）：**故意不登记** manifest
        # （Phase 3 不改 configs）→ 按 3.6 的装载方式注入 version_spec；available_at 断言不变。
        vers = (_load_versions(mod, d) if d.name not in SHADOW_NET_DIRS else
                mod.load_versions(d, {
                    'version_prefix': SHADOW_VERSION_PREFIX_BY_DIR[d.name],
                    'years': sorted(int(p.name.split('_')[0])
                                    for p in d.glob('*_similarity.csv')),
                    'available_at_rule': '(y+1)-04-30'}))
        assert vers, d
        for ty, v in vers.items():
            assert v['available_at'] == f'{ty + 1}-04-30', (d.name, ty)
        total += len(vers)
    assert total == 79 + 9 * len(SHADOW_NET_DIRS)


# ---------------------------------------------------------------- 正常路径
def test_valid_available_at_passes(mod, tmp_path):
    _make_version(tmp_path)
    vers = _load_versions(mod, tmp_path)
    assert list(vers) == [2019]
    assert vers[2019]['available_at'] == '2020-04-30'
    assert vers[2019]['text_year'] == 2019
    assert vers[2019]['path'].name == '2019_similarity.csv'


def test_select_version_behavior_unchanged(mod, tmp_path):
    _make_version(tmp_path, 2019, '2020-04-30')
    _make_version(tmp_path, 2020, '2021-04-30')
    vers = _load_versions(mod, tmp_path)
    # 严格 < 规则：2020-04 时文本 2019 版本尚不可用
    assert mod.select_version(vers, '2020-04') is None
    assert mod.select_version(vers, '2020-05')['text_year'] == 2019
    assert mod.select_version(vers, '2021-04')['text_year'] == 2019
    assert mod.select_version(vers, '2021-05')['text_year'] == 2020


# ---------------------------------------------------------------- 失败路径
def test_missing_build_report_exits_2(mod, tmp_path, capsys):
    _write_similarity(tmp_path, 2019)
    _assert_exit2(lambda: _load_versions(mod, tmp_path), capsys,
                  'text_year=2019', '2019_build_report.json', 'build_report 缺失')


@pytest.mark.parametrize('payload,reason', [
    ({'network_version': 'pm_2019'}, 'available_at 键缺失'),
    ({'available_at': None}, 'available_at 为 null'),
    ({'available_at': ''}, 'available_at 为空串'),
    ({'available_at': '   '}, 'available_at 为空串'),
    ({'available_at': 20200430}, 'available_at 非字符串'),
    ({'available_at': '2020/04/30'}, '格式非 YYYY-MM-DD'),
    ({'available_at': '2020-4-30'}, '格式非 YYYY-MM-DD'),
    ({'available_at': ' 2020-04-30'}, '格式非 YYYY-MM-DD'),
    ({'available_at': '2020-02-30'}, '非合法日期'),
    ({'available_at': '2021-04-30'}, '与规则不符'),
    ({'available_at': '2020-05-31'}, '与规则不符'),
])
def test_invalid_available_at_exits_2(mod, tmp_path, capsys, payload, reason):
    _write_similarity(tmp_path, 2019)
    _write_report(tmp_path, 2019, payload)
    _assert_exit2(lambda: _load_versions(mod, tmp_path), capsys,
                  'text_year=2019', '2019_build_report.json', reason)


def test_invalid_json_exits_2(mod, tmp_path, capsys):
    _write_similarity(tmp_path, 2019)
    (tmp_path / '2019_build_report.json').write_text('{not json', encoding='utf-8')
    _assert_exit2(lambda: _load_versions(mod, tmp_path), capsys,
                  'text_year=2019', '2019_build_report.json', '非法 JSON')


def test_top_level_not_object_exits_2(mod, tmp_path, capsys):
    _write_similarity(tmp_path, 2019)
    _write_report(tmp_path, 2019, ['2020-04-30'])
    _assert_exit2(lambda: _load_versions(mod, tmp_path), capsys,
                  'text_year=2019', '2019_build_report.json', '顶层不是对象')


def test_one_bad_version_blocks_whole_dir(mod, tmp_path, capsys):
    """目录内一个版本坏即退出，不得跳过坏版本继续使用其余年份。"""
    _make_version(tmp_path, 2019, '2020-04-30')
    _write_similarity(tmp_path, 2020)
    _write_report(tmp_path, 2020, {'available_at': ''})
    _assert_exit2(lambda: _load_versions(mod, tmp_path), capsys,
                  'text_year=2020', '2020_build_report.json', '空串')
