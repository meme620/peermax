"""样本版本单源：pricing_* 与 shadow_* 各自的指针（CO-2026-11-19-parameters）。

pricing_* 读 registry sample_and_spec_registry.sample_version（headline）；
shadow_* 读 shadow_legacy_sample_version（未重跑，显式登记）。两者都不得在脚本里写字面量。
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_lib as lib  # noqa: E402

PRICING = ['pricing_fm_main', 'pricing_portfolio_sorts', 'pricing_robustness_placebo',
           'pricing_gate_report']
SHADOW = ['shadow_rev', 'shadow_weight_matrix', 'shadow_turnover', 'shadow_network_pricing',
          'shadow_segment_break', 'shadow_industry_absorb', 'shadow_asof_industry',
          'shadow_inference']


def test_headline_and_shadow_pointers_are_distinct_registry_entries():
    head = lib.param_value('sample_and_spec_registry')['sample_version']
    shadow = lib.param_value('shadow_legacy_sample_version')
    assert lib.HEADLINE_SAMPLE_VERSION == head
    assert lib.SHADOW_SAMPLE_VERSION_LABEL == 'v2'
    assert lib.sample_path('ma_teacher').name.endswith(f"_{lib.SAMPLE_VERSION_LABEL}.parquet")
    assert lib.shadow_sample_path('ma_teacher').name.endswith(
        f"_{lib.SHADOW_SAMPLE_VERSION_LABEL}.parquet")
    # 两者当前不同版本：headline 已升 v3，shadow 仍在 v2
    assert head != shadow


def test_pricing_scripts_have_no_sample_version_literal():
    pattern = re.compile(r"(pricing_sample|pricing_sample)_ma_teacher_v[0-9]")
    for name in PRICING:
        text = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        hits = [ln for ln in text.splitlines() if pattern.search(ln) and not ln.strip().startswith('#')]
        assert hits == [], f"{name}.py 仍有样本版本字面量: {hits}"


def test_shadow_scripts_use_shadow_sample_path():
    for name in SHADOW:
        text = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        assert 'shadow_sample_path(' in text, f"{name}.py 未使用 shadow_sample_path"
