"""outcome 复权口径：主口径与敏感性列必须来自两条不同序列（CO-2026-11-20-outcome-convention）。

背景：主口径曾标 raw 而实际读后复权文件，导致「后复权敏感性」与主表同源、不检验任何东西。
本模块钉住三件事：
  1. registry 的 outcome_return_convention 与 spec/design.yaml 的登记键指向一致；
  2. load_returns_raw() 读的是未复权序列，不是主口径文件；
  3. 两条序列在同一批 (firm_id, month) 上确实不同（否则「敏感性」是空转）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def _registry():
    return yaml.safe_load((ROOT / 'configs/parameter_registry.yaml').read_text(encoding='utf-8'))


def _design():
    return yaml.safe_load((ROOT / 'spec/design.yaml').read_text(encoding='utf-8'))


def test_outcome_convention_registered_and_referenced():
    """registry 登记的键必须是 design 引用的那个键（两处写死不同名即口径漂移）。"""
    params = {p['name']: p for p in _registry()['parameters']}
    assert 'outcome_return_convention' in params, 'outcome 复权口径未登记'
    assert params['outcome_return_convention']['headline_value'] == 'post_adjusted'

    outcome = _design()['sample']['outcome']
    assert outcome['convention_registry_key'] == 'outcome_return_convention'
    assert outcome['main_column'] == 'R_next'
    assert outcome['sensitivity_column'] == 'R_next_raw'
    assert outcome['distinct_sources'] is True


def test_return_loaders_point_at_different_files():
    """主口径与未复权敏感性必须读不同文件；同名即两列同源。"""
    import pricing_lib

    assert pricing_lib.RETURNS_V2 != pricing_lib.RETURNS_RAW
    assert pricing_lib.RETURNS_RAW.name == 'monthly_returns_clean_raw.csv'


def test_raw_loader_reads_raw_file_not_main():
    """load_returns_raw 的历史 bug：名为 raw、实读主口径文件。"""
    import inspect

    import pricing_lib

    src = inspect.getsource(pricing_lib.load_returns_raw)
    body = src.split('"""')[-1]          # 去掉 docstring，只看函数体
    assert 'RETURNS_RAW' in body, 'load_returns_raw 必须读 RETURNS_RAW'
    assert 'RETURNS_V2' not in body, 'load_returns_raw 不得读主口径文件'


@pytest.mark.skipif(
    not (ROOT / 'data/processed/monthly_returns_clean_v2.csv').is_file()
    or not (ROOT / 'data/processed/monthly_returns_clean_raw.csv').is_file(),
    reason='需要主口径与未复权两条月收益序列落盘')
def test_two_series_actually_differ():
    """两条序列在共同键上必须不同；完全相同说明敏感性列是空转。"""
    import pandas as pd

    main = pd.read_csv(ROOT / 'data/processed/monthly_returns_clean_v2.csv',
                       usecols=['firm_id', 'month', 'R'], dtype={'firm_id': str, 'month': str})
    raw = pd.read_csv(ROOT / 'data/processed/monthly_returns_clean_raw.csv',
                      usecols=['firm_id', 'month', 'R'], dtype={'firm_id': str, 'month': str})
    j = main.merge(raw, on=['firm_id', 'month'], suffixes=('_main', '_raw'))
    assert len(j) > 0, '两条序列没有共同键'
    diff = (j['R_main'] - j['R_raw']).abs()
    share = float((diff > 1e-9).mean())
    assert share > 0.01, f'主口径与未复权序列几乎相同（差异占比 {share:.4f}），敏感性不自证'
