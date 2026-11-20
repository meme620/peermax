"""K 目标（网络密度）的口径登记与可复算性。

registry 的 ma_k_target_teacher = 44 是 2026-09 的外部锚；两组装网可复算口径
（teacher 全部 ok 行、剔 J/S 与上市不足 2 年）都复算不出 44。本测试钉住三件事：
  1. 两个口径都能现算，且数值与 ma_k_target_definition 登记一致；
  2. 冻结值与可复算值的差异是显式登记的（不是悄悄漂移）；
  3. 读取冻结值的路径 fail-closed。
"""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import network_business_cluster_freeze as freeze  # noqa: E402

REGISTRY = ROOT / "configs" / "parameter_registry.yaml"
YEARS = list(range(2014, 2023))


def _params():
    doc = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    return {p["name"]: p for p in doc["parameters"]}


def test_definition_block_registered():
    blk = _params()["ma_k_target_definition"]["headline_value"]
    for key in ("pooling", "industry_source", "sample_filter",
                "realized_pooled_median", "realized_after_network_filters"):
        assert key in blk, key


def test_realized_pooled_median_matches_registration():
    import network_tau_calibrate as tau
    blk = _params()["ma_k_target_definition"]["headline_value"]
    got = tau.sw2_target(YEARS, "teacher")
    assert got["median_peer_pooled_firm_level"] == blk["realized_pooled_median"]


def test_realized_after_network_filters_matches_registration():
    import network_tau_calibrate as tau
    blk = _params()["ma_k_target_definition"]["headline_value"]
    got = tau.sw2_target(YEARS, "teacher", network_pool=True)
    assert got["median_peer_pooled_firm_level"] == blk["realized_after_network_filters"]
    assert got["n_firm_years"] < tau.sw2_target(YEARS, "teacher")["n_firm_years"]


def test_frozen_value_differs_from_recomputable_and_is_documented():
    p = _params()
    frozen = int(p["ma_k_target_teacher"]["headline_value"])
    blk = p["ma_k_target_definition"]["headline_value"]
    assert frozen not in (blk["realized_pooled_median"], blk["realized_after_network_filters"])
    note = p["ma_k_target_teacher"]["note"]
    assert "45" in note and "37" in note


def test_frozen_target_read_is_fail_closed():
    got, src = freeze.registry_target("teacher", 999)
    assert got == int(_params()["ma_k_target_teacher"]["headline_value"])
    assert src.startswith("registry:")
    got2, src2 = freeze.registry_target("self", 56)
    assert got2 == 56 and src2 == "constant"
