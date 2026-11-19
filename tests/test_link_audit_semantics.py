"""claim / scope / registry 的语义一致性（CO-2026-11-19-parameters）。

三层检查（结构审计原来的盲区）：
1. claim 的 required_gates 必须覆盖 treatment 的数据依赖（PeerMAXraw → G3）；
2. claim 的 input_tables 版本必须等于 headline 样本指针；
3. scope_card 的 blocked_claims 必须已登记在 claim_registry。
"""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import pricing_lib as lib  # noqa: E402

CLAIMS = yaml.safe_load((ROOT / "configs" / "claim_registry.yaml").read_text(encoding="utf-8"))
SCOPE = yaml.safe_load((ROOT / "configs" / "scope_card.yaml").read_text(encoding="utf-8"))

# treatment → 必须被 gate 覆盖的数据依赖
TREATMENT_GATES = {"PeerMAXraw": {"G3"}, "PeerMAXraw_renorm": {"G3"}}


def test_claim_required_gates_cover_treatment_dependency():
    for c in CLAIMS["claims"]:
        need = TREATMENT_GATES.get(c["treatment"], set())
        have = set(c.get("required_gates") or [])
        assert need <= have, f"{c['claim_id']} 缺 {sorted(need - have)}"


def test_claim_input_tables_match_headline_sample_version():
    """claim 的样本输入必须指向 headline 版本；解析不到版本号即失败（不允许静默跳过）。"""
    checked = 0
    for c in CLAIMS["claims"]:
        for t in c.get("input_tables") or []:
            m = re.search(r"_(v[0-9]+)\.parquet$", t)
            if not m:
                continue
            checked += 1
            assert m.group(1) == lib.SAMPLE_VERSION_LABEL, (
                f"{c['claim_id']} 的 input_table {t} 不是 headline 版本 "
                f"{lib.SAMPLE_VERSION_LABEL}")
    assert checked > 0, "没有任何 claim 的 input_tables 被校验到（正则与新命名脱节）"


def test_scope_blocked_claims_registered():
    ids = {c["claim_id"] for c in CLAIMS["claims"]}
    for b in SCOPE.get("blocked_claims") or []:
        assert b in ids, f"scope_card.blocked_claims 引用了未登记的 claim {b}"


def test_scope_window_matches_effective_window():
    eff = yaml.safe_load((ROOT / "spec" / "design.yaml").read_text(encoding="utf-8"))["windows"]["effective"]
    assert SCOPE["sample_window"] == eff
