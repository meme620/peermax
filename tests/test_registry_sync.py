"""registry 单一来源守卫（CO-2026-11-19-parameters）。

三条不变量：
1. registry_version 与 analyst_spec 的 governance.registry_version 一致（pricing_lib 导入即断言）；
2. K_Consist_Pricing 等于 K_Econ_Pricing / iqr_pool_value（派生关系，不再手填）；
3. headline 样本指针只有一个：sample_and_spec_registry.sample_version，
   pricing_lib.sample_path() 的默认值、SAMPLE_VERSION_LABEL 都由它决定。
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_lib as lib  # noqa: E402


def test_registry_version_matches_s3a_governance():
    reg = lib.load_registry()
    spec = yaml.safe_load((ROOT / "configs" / "analyst_spec.yaml").read_text(encoding="utf-8"))
    assert reg["registry_version"] == spec["governance"]["registry_version"]


def test_k_consist_is_derived_from_k_econ_and_iqr():
    k_econ = float(lib.param_value("K_Econ_Pricing"))
    iqr = float(lib.param_value("iqr_pool_value"))
    k_consist = float(lib.param_value("K_Consist_Pricing"))
    assert abs(k_consist - k_econ / iqr) <= 1e-6


def test_headline_sample_version_is_single_source():
    reg = lib.param_value("sample_and_spec_registry")
    assert lib.HEADLINE_SAMPLE_VERSION == reg["sample_version"]
    assert lib.sample_path("ma_teacher") == lib.sample_path("ma_teacher", lib.SAMPLE_VERSION_LABEL)
    assert lib.SAMPLE_VERSION_LABEL in ("v1", "v2", "v3", "v4", "v5", "v6")
