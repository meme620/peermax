"""Phase 1 I-5：registry 参数绑定测试。

验证 `winsorization_pct` / `sample_start` / `sample_end` 真正进入调用方：
- winsor 边界由 registry 读取，且 pricing_sample_build 两处 auc 调用显式传参；
- 样本窗口常量与 registry 截月一致，`months_between` 不再硬编码；
- 临时 registry 注入（tmp_path）时 helper 读到新值；
- 非法 registry 值 fail-closed（SystemExit）。

只读真实 registry / 脚本源码；不跑 pricing_gate、不写 data/reports。
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import pricing_lib as lib  # noqa: E402
import pricing_sample_build as build  # noqa: E402

SCRIPT = ROOT / 'scripts' / 'pricing_sample_build.py'
REGISTRY = ROOT / 'configs' / 'parameter_registry.yaml'

DEFAULTS = {
    'winsorization_pct': [0.01, 0.99],
    'sample_start': '2015-01-01',
    'sample_end': '2023-12-31',
}


def write_registry(tmp_path: Path, **overrides) -> Path:
    """写最小临时 registry；value 为 ... 表示删除该参数。"""
    values = dict(DEFAULTS)
    values.update(overrides)
    params = [{'name': name, 'headline_value': value}
              for name, value in values.items() if value is not ...]
    doc = {'registry_version': 'test-1.4', 'parameters': params}
    p = tmp_path / 'parameter_registry.yaml'
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                 encoding='utf-8')
    return p


def _calls(tree: ast.AST, name: str) -> list:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def _refs_winsor(node: ast.AST) -> bool:
    if isinstance(node, ast.Starred):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id == 'WINSOR'
    if isinstance(node, ast.Subscript):
        return isinstance(node.value, ast.Name) and node.value.id == 'WINSOR'
    return False


# ---------------------------------------------------------------- 当前 registry 值
def test_current_winsor_bounds_match_registry():
    assert lib.winsor_bounds() == (0.01, 0.99)
    assert list(lib.param_value('winsorization_pct')) == [0.01, 0.99]


def test_current_sample_window_matches_registry():
    assert lib.sample_window_bounds() == ('2015-01', '2023-12')
    assert (lib.SAMPLE_WINDOW_START, lib.SAMPLE_WINDOW_END) == ('2015-01', '2023-12')
    assert lib.SAMPLE_WINDOW_START == str(lib.param_value('sample_start'))[:7]
    assert lib.SAMPLE_WINDOW_END == str(lib.param_value('sample_end'))[:7]


def test_caller_module_constants_come_from_registry_helpers():
    assert build.WINSOR == lib.winsor_bounds() == (0.01, 0.99)
    assert build.WINDOW == f'{lib.SAMPLE_WINDOW_START}..{lib.SAMPLE_WINDOW_END}'
    assert build.WINDOW == '2015-01..2023-12'


# ---------------------------------------------------------------- 报告字段 registry_params_used
def test_registry_params_used_reports_three_params():
    used = build.registry_params_used()
    assert used['registry_version'] == lib.load_registry().get('registry_version')
    for name in ('winsorization_pct', 'sample_start', 'sample_end'):
        entry = used[name]
        assert entry['name'] == name
        assert entry['applied'] is True
        assert entry['source'] == (f'configs/parameter_registry.yaml:'
                                   f'parameters[name={name}].headline_value')
    assert used['winsorization_pct']['value'] == list(build.WINSOR)
    assert used['sample_start']['value'] == '2015-01-01'
    assert used['sample_end']['value'] == '2023-12-31'
    assert used['sample_start']['month'] == '2015-01'
    assert used['sample_end']['month'] == '2023-12'
    assert used['sample_window_derived'] == build.WINDOW
    assert used['iqr_scale_rule']['registry_numeric_value'] is None
    assert '无独立数值参数' in used['iqr_scale_rule']['numeric_value_source']


# ---------------------------------------------------------------- winsorize_by_month 行为
def _synthetic_month(n: int = 101) -> pd.DataFrame:
    return pd.DataFrame({'month': ['2020-01'] * n,
                         'x': np.arange(n, dtype=float)})


def test_winsorize_by_month_respects_registry_bounds():
    lo, hi = lib.winsor_bounds()
    df = _synthetic_month()
    out = lib.winsorize_by_month(df, ['x'], lo=lo, hi=hi)
    assert out['x'].min() == pytest.approx(df['x'].quantile(lo))
    assert out['x'].max() == pytest.approx(df['x'].quantile(hi))
    assert out['x'].tolist()[50] == 50.0   # 中段值不变
    # 当前 registry 值 == 旧硬编码 0.01/0.99 → 显式传参与旧缺省调用逐值相同
    legacy = lib.winsorize_by_month(df, ['x'])
    pd.testing.assert_series_equal(out['x'], legacy['x'])


def test_tighter_bounds_change_winsor_result():
    df = _synthetic_month()
    registry = lib.winsorize_by_month(df, ['x'], *lib.winsor_bounds())
    tighter = lib.winsorize_by_month(df, ['x'], lo=0.05, hi=0.95)
    assert tighter['x'].min() > registry['x'].min()
    assert tighter['x'].max() < registry['x'].max()


# ---------------------------------------------------------------- 临时 registry 注入
def test_temp_registry_new_winsor_bounds_read(tmp_path):
    p = write_registry(tmp_path, winsorization_pct=[0.005, 0.995])
    assert lib.winsor_bounds(p) == (0.005, 0.995)
    assert lib.param_value('winsorization_pct', p) == [0.005, 0.995]


def test_temp_registry_new_sample_window_read(tmp_path):
    p = write_registry(tmp_path, sample_start='2016-01-01',
                       sample_end='2022-12-31')
    assert lib.sample_window_bounds(p) == ('2016-01', '2022-12')
    assert lib.param_value('sample_start', p) == '2016-01-01'


def test_caller_helper_reads_temp_registry(tmp_path):
    """调用方 helper（报告生成器）随临时 registry 读到 0.005/0.995 与新窗口。"""
    p = write_registry(tmp_path, winsorization_pct=[0.005, 0.995],
                       sample_start='2016-01-01', sample_end='2022-12-31')
    used = build.registry_params_used(p)
    assert used['registry_version'] == 'test-1.4'
    assert used['winsorization_pct']['value'] == [0.005, 0.995]
    assert used['sample_start']['value'] == '2016-01-01'
    assert used['sample_start']['month'] == '2016-01'
    assert used['sample_end']['month'] == '2022-12'
    assert used['sample_window_derived'] == '2016-01..2022-12'


# ---------------------------------------------------------------- 非法 registry fail-closed
@pytest.mark.parametrize('bad', [
    [0.5, 0.4],          # lo >= hi
    [0.5, 0.5],          # lo == hi
    [-0.1, 0.9],         # lo < 0
    [0.01],              # 长度 != 2
    ['a', 0.99],         # 非数值
    'not-a-list',        # 非序列
])
def test_invalid_winsor_rejected(tmp_path, bad):
    p = write_registry(tmp_path, winsorization_pct=bad)
    with pytest.raises(SystemExit) as ei:
        lib.winsor_bounds(p)
    assert ei.value.code == 2


@pytest.mark.parametrize('field,bad', [
    ('sample_start', '2015/01/01'),
    ('sample_start', '2015-13-01'),
    ('sample_end', ''),
    ('sample_end', 'abc'),
])
def test_invalid_window_format_rejected(tmp_path, field, bad):
    p = write_registry(tmp_path, **{field: bad})
    with pytest.raises(SystemExit) as ei:
        lib.sample_window_bounds(p)
    assert ei.value.code == 2


def test_window_start_after_end_rejected(tmp_path):
    p = write_registry(tmp_path, sample_start='2024-01-01')
    with pytest.raises(SystemExit) as ei:
        lib.sample_window_bounds(p)
    assert ei.value.code == 2


@pytest.mark.parametrize('missing', ['winsorization_pct', 'sample_start', 'sample_end'])
def test_missing_parameter_rejected(tmp_path, missing):
    p = write_registry(tmp_path, **{missing: ...})
    with pytest.raises(SystemExit) as ei:
        if missing == 'winsorization_pct':
            lib.winsor_bounds(p)
        else:
            lib.sample_window_bounds(p)
    assert ei.value.code == 2


def test_missing_registry_file_rejected(tmp_path):
    with pytest.raises(SystemExit) as ei:
        lib.winsor_bounds(tmp_path / 'nope.yaml')
    assert ei.value.code == 2


# ---------------------------------------------------------------- 调用方静态检查（防回归）
def test_ast_all_winsor_calls_pass_registry_bounds():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    calls = _calls(tree, 'winsorize_by_month')
    assert len(calls) >= 2, 'pricing_sample_build 的两处 winsorize_by_month 调用缺失'
    for call in calls:
        kws = {k.arg: k.value for k in call.keywords}
        assert 'lo' in kws and 'hi' in kws, f'line {call.lineno}: 未显式传 lo/hi'
        assert _refs_winsor(kws['lo']) and _refs_winsor(kws['hi']), (
            f'line {call.lineno}: lo/hi 必须引用 registry 派生的 WINSOR')
    second_args = [c.args[1] for c in calls if len(c.args) > 1]
    assert any(isinstance(a, ast.Name) and a.id == 'winsor_panel_cols'
               for a in second_args), '面板列块未走 registry 边界'
    listed = {e.value for a in second_args if isinstance(a, ast.List)
              for e in a.elts if isinstance(e, ast.Constant)}
    assert {'R_next', 'R_next_postadj'} <= listed, 'outcome 列块未走 registry 边界'


def test_ast_months_between_uses_registry_window():
    tree = ast.parse(SCRIPT.read_text(encoding='utf-8'))
    calls = _calls(tree, 'months_between')
    assert calls, 'months_between 调用缺失'
    for call in calls:
        ids = [a.id for a in call.args if isinstance(a, ast.Name)]
        assert ids == ['SAMPLE_WINDOW_START', 'SAMPLE_WINDOW_END'], (
            f'line {call.lineno}: months_between 必须使用 registry 窗口常量')
    src = SCRIPT.read_text(encoding='utf-8')
    assert "months_between('2015-01', '2023-12')" not in src
    assert "WINSOR = winsor_bounds()" in src
    assert "param_value('winsorization_pct')" not in src


# ---------------------------------------------------------------- IQR 披露
def test_iqr_registry_rule_id_and_frozen_value_documented():
    rule_id = lib.param_value('sample_and_spec_registry')['iqr_scale_rule_id']
    assert rule_id == lib.IQR_POOL_RULE_ID == 'IQR_POOL_TREATMENT_ONLY_v1'
    assert lib.IQR_POOL_REGISTERED == 0.02341287
    # 2026-11-19 参数口径核查：数值唯一来源改为 registry 的 iqr_pool_value，
    # pricing_lib 导入即核对（不一致 SystemExit(2)），不再要求 registry 里没有数值键。
    assert float(lib.param_value('iqr_pool_value')) == lib.IQR_POOL_REGISTERED
    assert lib.IQR_POOL_REGISTERED_SOURCE.startswith('configs/parameter_registry.yaml')
