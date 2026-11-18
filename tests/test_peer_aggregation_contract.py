"""P1.2-01：共享聚合不变量（07 卡 :2311 点名；P1.4-05 复用同一函数）。

覆盖：
- A：两源 complete、w={0.4,0.6}、x={0.1,0.2} → 0.16；support=covered=1、missing=0、ratio=1、complete；
- B：一源缺失 → headline 空、partial=0.04、missing=0.6、ratio=0.4、partial；
- C：权重和 0.9（未归一化网络）→ headline = Σw·x = 0.15（不是 0.15/0.9）；
- D：support=0 → ratio NaN、status=not_applicable；
- nv==0（zero_trading_days）与 incomplete_month 的排除语义；
- 源码/字节码路径不得出现 `/ covered`、`/w_max`、`/w_ret` 写回 headline。
"""
import ast
import inspect
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts/data_audit/s3_m1_lib.py"


def _load(name="s3_m1_lib_agg"):
    spec = importlib.util.spec_from_file_location(name, LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lib = _load()


def _edges(w, month="2015-01", focal="f1"):
    return pd.DataFrame({"focal_id": [focal] * len(w), "month": [month] * len(w),
                         "weight": w})


def test_case_a_complete_full_coverage():
    e = _edges([0.4, 0.6])
    src = pd.DataFrame({"x": [0.1, 0.2]})
    out = lib.aggregate_stat(e, src, "x", pd.Series([True, True]), "max")
    r = out.iloc[0]
    assert r["weighted_value_max"] == pytest.approx(0.16, abs=1e-12)
    assert r["support_mass_max"] == pytest.approx(1.0)
    assert r["covered_mass_max"] == pytest.approx(1.0)
    assert r["missing_mass_max"] == 0.0
    assert r["coverage_ratio_max"] == pytest.approx(1.0)
    assert r["support_status_max"] == "complete"


def test_case_b_partial_not_headline():
    e = _edges([0.4, 0.6])
    src = pd.DataFrame({"x": [0.1, np.nan]})
    out = lib.aggregate_stat(e, src, "x", pd.Series([True, True]), "max")
    r = out.iloc[0]
    assert np.isnan(r["weighted_value_max"])
    assert r["partial_value_max"] == pytest.approx(0.04)
    assert r["missing_mass_max"] == pytest.approx(0.6)
    assert r["coverage_ratio_max"] == pytest.approx(0.4)
    assert r["support_status_max"] == "partial"
    # headline ≠ partial / ratio
    assert not np.isclose(r["partial_value_max"], 0.04 / 0.4)


def test_case_c_unnormalized_not_divided():
    e = _edges([0.3, 0.6])          # Σw = 0.9
    src = pd.DataFrame({"x": [0.1, 0.2]})
    out = lib.aggregate_stat(e, src, "x", pd.Series([True, True]), "max")
    r = out.iloc[0]
    assert r["weighted_value_max"] == pytest.approx(0.15, abs=1e-12)
    assert r["support_mass_max"] == pytest.approx(0.9)
    assert r["coverage_ratio_max"] == pytest.approx(1.0)
    assert r["support_status_max"] == "complete"


def test_case_d_zero_support_not_applicable():
    e = _edges([0.0])
    src = pd.DataFrame({"x": [0.1]})
    out = lib.aggregate_stat(e, src, "x", pd.Series([True]), "ret")
    r = out.iloc[0]
    assert np.isnan(r["coverage_ratio_ret"])
    assert r["support_status_ret"] == "not_applicable"
    assert np.isnan(r["weighted_value_ret"])


def test_zero_trading_days_and_incomplete_excluded():
    e = _edges([0.4, 0.3, 0.3])
    src = pd.DataFrame({"x": [0.1, 0.2, 0.3]})
    # 第 1 源 covered；第 2 源 nv==0；第 3 源 nv 1–14（incomplete）
    out = lib.aggregate_stat(
        e, src, "x", pd.Series([True, False, False]), "ret",
        zero_trading_days_mask=pd.Series([False, True, False]),
        incomplete_mask=pd.Series([False, False, True]))
    r = out.iloc[0]
    assert np.isnan(r["weighted_value_ret"])
    assert r["covered_mass_ret"] == pytest.approx(0.4)
    assert r["missing_mass_ret"] == pytest.approx(0.6)
    assert r["n_zero_trading_days_ret"] == 1
    assert r["n_incomplete_ret"] == 1
    assert r["partial_value_ret"] == pytest.approx(0.4 * 0.1 + 0.3 * 0.3)
    assert r["support_status_ret"] == "partial"


def test_aggregate_stat_has_no_headline_division():
    tree = ast.parse(inspect.getsource(lib.aggregate_stat))
    divs = [n for n in ast.walk(tree)
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
    assert len(divs) >= 1
    # 除法只允许产生 coverage_ratio 与 limit_hit_weight_share（均带安全除零）
    for n in divs:
        expr = ast.unparse(n)
        assert ("coverage_ratio" in expr) or ("limit_hit_weight_share" in expr)             or ("np.where" in expr), expr
    # headline / partial 赋值路径不得含除式
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target = ast.unparse(node.targets[0])
            if ("weighted_value" in target) or ("partial_value" in target):
                assert not any(isinstance(x, ast.BinOp) and isinstance(x.op, ast.Div)
                               for x in ast.walk(node.value)), target
