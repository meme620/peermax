"""Phase 3.6 shadow 网络 pricing_gate 对比契约测试（加法，只读真实产物）。

覆盖（3.0_plan §2.6 R1/R2/R3 + 3.6 任务书）：
  1. 输出路径守卫（只允许 artifacts/phase3/3.6_* 与 phase3_shadow/shadow_treat_*）；
  2. 月份→文本年映射（available_at 严格小于月份；2015-01..04 无网络）；
  3. 固定样本计数（307,503 / 181,178 / 104）与 headline 复算保真门；
  4. 对比表矩阵完整（7 变体 × 3 规格 × 2 权重 = 42 行；spec_full_controls/spec_no_turnover/spec_no_rev_no_turnover 定义）；
  5. headline 登记值复现 + headline_recompute 逐规格复现；
  6. 覆盖/样本损失、treatment 相关、网络 Jaccard 与 3.2 逐值对账；
  7. 9 个 pricing_gate 输入 + headline 样本/网络 + shadow 网络 sha 前后未变；
  8. 输出 sha 台账与二次运行逐字节一致；
  9. Q 判据与 HOLD 决策。

运行：python -m pytest tests/test_shadow_network_pricing.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import assert_input_anchors, require_artifacts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SCRIPT = ROOT / "scripts/shadow_network_pricing.py"
ART = ROOT / "artifacts/phase3"
COUNTS = ART / "3.6_counts.json"
COMPARISON = ART / "3.6_network_comparison.csv"
OUTPUT_SHA = ART / "3.6_output_sha.json"
ANCHOR_PRE = ART / "3.6_anchor_pre.json"
ANCHOR_POST = ART / "3.6_anchor_post.json"
SAMPLE = ROOT / "data/processed/pricing_sample_ma_teacher_v2.parquet"
HEADLINE_NET = ROOT / "data/processed/network_ma_teacher"
SHADOW_DATA = ROOT / "data/processed/phase3_shadow"
G0_SAMPLE = ROOT / "reports/data_gates/G0/pricing_sample_report_v2.json"
REGISTRY = ROOT / "configs/parameter_registry.yaml"
TEST_FILE = Path(__file__)

SHADOW_IDS = ["p3asof", "p3yt", "p3ydf", "p3yt_ydf", "p3yt_latest"]
ALL_IDS = ["headline"] + SHADOW_IDS
FM_ROWS = 181178
FM_MONTHS = 104
JACCARD_32 = {"p3asof": 0.966815, "p3yt": 0.912737, "p3ydf": 0.529672,
              "p3yt_ydf": 0.520040, "p3yt_latest": 0.823349}


def file_size(path):
    return Path(path).stat().st_size


def _manifest_anchor(rel: str) -> str:
    """当前 manifest 的 active 条目（BL-01 后 headline 样本锚的现盘来源）。"""
    import yaml
    man = yaml.safe_load((ROOT / "configs/version_manifest.yaml").read_text(encoding="utf-8"))
    hits = [e for e in man["deliverables"] if str(e.get("file", "")).replace("\\", "/") == rel]
    assert len(hits) == 1, f"manifest 条目数 != 1: {rel}"
    assert hits[0].get("status") in ("active", "superseded"), f"manifest 条目状态异常: {rel}"
    return str(hits[0]["bytes"])


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_network_pricing", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p36():
    assert SCRIPT.is_file(), f"缺少脚本 {SCRIPT}"
    return load_module()


@pytest.fixture(scope="module")
def counts() -> dict:
    assert COUNTS.is_file(), f"缺少 3.6 产物（先运行 {SCRIPT.name}）：{COUNTS}"
    return json.loads(COUNTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def comp() -> pd.DataFrame:
    assert COMPARISON.is_file(), f"缺少 {COMPARISON}"
    return pd.read_csv(COMPARISON)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_parquet(
        SAMPLE, columns=["focal_id", "month", "network_version", "PeerMAXraw",
                         "PeerReturn", "n_sources", "pricing_in_fm_sample"])


# ------------------------------------------------------------------ 守卫 / 规格
def test_output_guard_allows_only_3_6_outputs(p36):
    assert p36.assert_output_path_guard(ART / "3.6_unit_test.csv") == (ART / "3.6_unit_test.csv").resolve()
    assert p36.assert_output_path_guard(
        SHADOW_DATA / "shadow_treat_unit.parquet") == (
        SHADOW_DATA / "shadow_treat_unit.parquet").resolve()


@pytest.mark.parametrize("path", [
    SAMPLE,
    HEADLINE_NET / "2014_similarity.csv",
    G0_SAMPLE,
    REGISTRY,
    SCRIPT,
    TEST_FILE,
    ART / "memo_not_allowed.txt",
    SHADOW_DATA / "other_not_allowed.parquet",
    ROOT / "reports" / "x.json",
    ROOT / "data/processed/network_ma_teacher_p3asof/2014_similarity.csv",
])
def test_output_guard_blocks_headline_configs_g0_and_non_3_6(p36, path):
    with pytest.raises(SystemExit) as exc:
        p36.assert_output_path_guard(path)
    assert exc.value.code == 2


def test_spec_definitions(p36):
    assert p36.spec_full_controls[:3] == ["PeerMAXraw", "OwnMAXraw", "PeerReturn"]
    assert p36.spec_no_turnover == [c for c in p36.spec_full_controls if c != "turnover"]
    assert p36.spec_no_rev_no_turnover == [c for c in p36.spec_full_controls if c not in ("rev", "turnover")]
    assert len(p36.CONTROLS) == 8
    assert p36.MIN_N == 100 and p36.NW_LAG == 6


# ------------------------------------------------------------------ 映射 / 固定样本
def test_month_mapping_uses_available_at_strictly_less(p36):
    vers = {
        2014: {"text_year": 2014, "available_at": "2015-04-30"},
        2015: {"text_year": 2015, "available_at": "2016-04-30"},
        2022: {"text_year": 2022, "available_at": "2023-04-30"},
    }
    assert p36.select_version(vers, "2015-04") is None
    assert p36.select_version(vers, "2015-05")["text_year"] == 2014
    assert p36.select_version(vers, "2016-04")["text_year"] == 2014
    assert p36.select_version(vers, "2016-05")["text_year"] == 2015
    assert p36.select_version(vers, "2023-12")["text_year"] == 2022
    mapping = p36.map_months(vers, ["2015-05", "2016-04", "2016-05", "2023-12"])
    assert mapping == {"2015-05": 2014, "2016-04": 2014,
                       "2016-05": 2015, "2023-12": 2022}


def test_real_month_mapping_covers_all_fm_months(counts, sample):
    check = counts["month_mapping"]["sample_check"]
    assert check["n_mismatch"] == 0
    assert check["n_unmapped_fm_rows"] == 0
    assert check["n_unmapped_with_version"] == 0
    assert check["n_unmapped_rows_pre_network"] == 10619
    assert check["unmapped_months"] == ["2015-01", "2015-02", "2015-03", "2015-04"]
    by_ty = counts["month_mapping"]["by_text_year"]
    assert sorted(int(k) for k in by_ty) == list(range(2014, 2023))
    assert sum(len(v) for v in by_ty.values()) == FM_MONTHS
    assert [len(by_ty[str(y)]) for y in range(2014, 2023)] == [12] * 8 + [8]
    fm = sample[sample["pricing_in_fm_sample"].astype(bool)]
    assert fm["network_version"].notna().all()
    assert len(fm) == FM_ROWS and fm["month"].nunique() == FM_MONTHS


def test_fixed_sample_counts_and_recompute_gate(counts):
    c = counts["constants"]
    assert c["sample_rows"] == 307503 and c["fm_rows"] == FM_ROWS
    assert c["fm_months"] == FM_MONTHS
    v = counts["headline_recompute_validation"]
    assert v["status"] == "PASS"
    assert v["peer_max_max_abs_diff"] <= 1e-7
    assert v["peer_return_max_abs_diff"] <= 1e-7
    assert v["coverage_ratio_max_max_abs_diff"] <= 2e-6
    assert v["coverage_ratio_ret_max_abs_diff"] <= 2e-6
    assert v["n_sources_mismatch_rows"] == 0
    assert v["peermax_missing_pattern_mismatch_rows"] == 0
    for vid in ALL_IDS:
        assert counts["coverage"][vid]["fm_rows"] == FM_ROWS


def test_attach_treatment_keeps_missing_and_zero_n_sources(p36):
    smp = pd.DataFrame({
        "focal_id": ["a", "a", "b"],
        "month": ["2015-05", "2015-06", "2015-05"],
        "PeerMAXraw": [1.0, 2.0, 3.0],
    })
    raw = pd.DataFrame({
        "focal_id": ["a"], "month": ["2015-05"],
        "PeerMAXraw": [0.5], "PeerReturn": [0.1],
        "n_sources": [2], "coverage_ratio_max": [1.0],
        "coverage_ratio_ret": [1.0],
    })
    t = p36.attach_treatment(smp, raw)
    assert t.loc[0, "PeerMAXraw"] == 0.5
    assert np.isnan(t.loc[1, "PeerMAXraw"])
    assert t.loc[1, "n_sources"] == 0
    assert np.isnan(t.loc[2, "PeerMAXraw"])
    assert t.loc[2, "n_sources"] == 0


def test_year_contract_detects_self_loop_and_duplicates(p36):
    e = pd.DataFrame({
        "focal_id": ["a", "a", "b", "b"],
        "source_id": ["b", "b", "a", "b"],
        "similarity": [1.0, 1.0, 1.0, 1.0],
    })
    support = pd.DataFrame({"focal_id": ["a", "b", "c"], "n_edges": [2, 2, 0]})
    chk = p36._year_contract(e, support, 2014, "synthetic")
    assert chk["duplicate_edges"] == 1
    assert chk["self_loops"] == 1
    assert chk["n_edges_mismatch_support"] == 0
    assert chk["n_support_zero_edge_nodes"] == 1
    assert chk["ok"] is False


# ------------------------------------------------------------------ 对比表
def test_comparison_matrix_complete(comp):
    assert len(comp) == 42
    expected = {(v, s, w) for v in ALL_IDS + ["headline_recompute"]
                for s in ("spec_full_controls", "spec_no_turnover", "spec_no_rev_no_turnover") for w in ("EW", "VW")}
    assert set(zip(comp["variant"], comp["spec"], comp["weight"])) == expected
    assert comp["beta_peermax"].notna().all()
    assert comp["t_peermax"].notna().all()
    assert (comp["T"] == FM_MONTHS).all()
    for vid in ALL_IDS + ["headline_recompute"]:
        s = comp[comp["variant"] == vid]
        assert len(s) == 6
        assert set(s["spec"]) == {"spec_full_controls", "spec_no_turnover", "spec_no_rev_no_turnover"}
        assert set(s["weight"]) == {"EW", "VW"}


def test_headline_registered_values(comp):
    h = comp[(comp["variant"] == "headline") & (comp["spec"] == "spec_full_controls")]
    ew = h[h["weight"] == "EW"].iloc[0]
    vw = h[h["weight"] == "VW"].iloc[0]
    assert abs(ew["beta_peermax"] - 0.03427280848385429) <= 1e-12
    assert abs(ew["t_peermax"] - 0.472809) <= 1e-5
    assert abs(vw["beta_peermax"] - (-0.109956)) <= 1e-6
    assert abs(vw["t_peermax"] - (-1.181193)) <= 1e-4
    assert int(ew["n"]) == FM_ROWS and int(ew["T"]) == FM_MONTHS


def test_headline_recompute_reproduces_headline(comp):
    h = comp[comp["variant"] == "headline"].set_index(["spec", "weight"])
    r = comp[comp["variant"] == "headline_recompute"].set_index(["spec", "weight"])
    assert list(h.index) == list(r.index)
    for idx in h.index:
        assert abs(r.loc[idx, "beta_peermax"] - h.loc[idx, "beta_peermax"]) <= 1e-7
        assert abs(r.loc[idx, "t_peermax"] - h.loc[idx, "t_peermax"]) <= 1e-5
        assert int(r.loc[idx, "n"]) == int(h.loc[idx, "n"])
        assert int(r.loc[idx, "T"]) == int(h.loc[idx, "T"])


def test_no_positive_significant_peermax(comp):
    shadow = comp[~comp["variant"].isin(["headline", "headline_recompute"])]
    bad = shadow[(shadow["beta_peermax"] > 0) & (shadow["t_peermax"] > 1.96)]
    assert len(bad) == 0
    assert shadow["t_peermax"].abs().max() < 1.96


def test_peerreturn_positive_and_ownmax_insignificant_in_s1(comp):
    s1 = comp[(comp["spec"] == "spec_full_controls")
              & (~comp["variant"].isin(["headline_recompute"]))]
    assert (s1["beta_peerreturn"] > 0).all()
    assert (s1["t_peerreturn"] > 1.9).all()
    assert (s1["t_ownmax"].abs() < 1.96).all()


# ------------------------------------------------------------------ 覆盖 / 网络
def test_coverage_counts_and_sample_loss(counts):
    for vid in ALL_IDS:
        c = counts["coverage"][vid]
        assert c["fm_peermax_nonnull"] + c["fm_peermax_missing"] == FM_ROWS
        assert c["fm_peermax_lost_vs_headline"] <= c["fm_peermax_missing"]
        assert c["fm_peermax_lost_vs_headline"] == c["fm_peermax_missing"]
        assert 280000 <= c["fixed_peermax_nonnull"] <= 307503
        assert c["fm_coverage_ratio_max_median"] == 1.0
    expected_missing = {"headline": 0, "p3asof": 0, "p3yt": 107,
                        "p3ydf": 4282, "p3yt_ydf": 3880, "p3yt_latest": 826}
    for vid, miss in expected_missing.items():
        assert counts["coverage"][vid]["fm_peermax_missing"] == miss


def test_treatment_parquet_contract(counts):
    for vid in SHADOW_IDS:
        p = SHADOW_DATA / f"shadow_treat_{vid}.parquet"
        assert p.is_file(), f"缺少 {p}"
        d = pd.read_parquet(p)
        assert len(d) == FM_ROWS
        assert not d.duplicated(["focal_id", "month"]).any()
        assert (d["variant"] == vid).all()
        assert int(d["peermax_missing"].sum()) == counts["coverage"][vid]["fm_peermax_missing"]
        assert int(d["peerreturn_missing"].sum()) == counts["coverage"][vid]["fm_peerreturn_missing"]
        assert d["n_sources"].ge(0).all()
        # 缺失不填 0：缺失行的 treatment 必须为 NaN
        assert d.loc[d["peermax_missing"], "PeerMAXraw"].isna().all()
        assert d.loc[d["peerreturn_missing"], "PeerReturn"].isna().all()


def test_network_jaccard_matches_3_2(counts):
    for vid, expected in JACCARD_32.items():
        got = counts["network_metrics"][vid]["jaccard_median_yearly"]
        assert abs(got - expected) <= 1e-6, f"{vid}: {got} vs {expected}"
    for vid in SHADOW_IDS:
        nm = counts["network_metrics"][vid]
        assert nm["jaccard_min_yearly"] <= nm["jaccard_median_yearly"] <= 1.0
        assert nm["n_edges_total"] > 1_000_000


def test_edge_contracts_all_ok(counts):
    assert len(counts["edge_contracts"]) == 6
    for vid, rows in counts["edge_contracts"].items():
        assert len(rows) == 9, f"{vid}: {len(rows)}"
        for r in rows:
            assert r["ok"] is True
            assert r["duplicate_edges"] == 0
            assert r["self_loops"] == 0
            assert r["max_abs_row_weight_sum_minus_1"] <= 1e-9


def test_treatment_correlation_matrix(counts):
    for col in ("PeerMAXraw", "PeerReturn"):
        m = counts["treatment_correlations"][col]["corr"]
        assert set(m) == set(ALL_IDS)
        for a in ALL_IDS:
            assert abs(m[a][a] - 1.0) <= 1e-12
            for b in ALL_IDS:
                assert abs(m[a][b] - m[b][a]) <= 1e-12
        assert m["headline"]["p3asof"] > 0.98
        assert m["headline"]["p3ydf"] < 0.93


# ------------------------------------------------------------------ 锚 / sha / 决策
def test_9_pricing_inputs_match_now():
    assert_input_anchors()

def test_headline_and_shadow_inputs_sha_unchanged(counts):
    cmp_ = counts["anchors"]["compare"]
    assert cmp_["status"] == "MATCH"
    assert cmp_["n_diff"] == 0
    pre, post = counts["anchors"]["pre"], counts["anchors"]["post"]
    assert pre["pricing_inputs"] == post["pricing_inputs"]
    assert len(pre["pricing_inputs"]) == 9
    assert pre["headline_sample"] == post["headline_sample"]
    assert pre["headline_net"] == post["headline_net"]
    assert pre["shadow_net"] == post["shadow_net"]
    assert len(pre["headline_net"]) == 45
    assert set(pre["shadow_net"]) == set(SHADOW_IDS)
    assert all(len(v) == 36 for v in pre["shadow_net"].values())
    # BL-01（I-16）metadata 重写已改现盘 headline 样本 sha；`3.6_counts.json` 是 Phase 3 历史证据
    # （不得重写），故现盘比对改到当前 manifest 锚（pre==post 不变式见上）。
    assert file_size(SAMPLE) == _manifest_anchor(
        "data/processed/pricing_sample_ma_teacher_v2.parquet")
    for rel, sha in pre["headline_net"].items():
        assert file_size(ROOT / rel) == sha, rel
    for vid, files in pre["shadow_net"].items():
        for rel, sha in files.items():
            assert file_size(ROOT / rel) == sha, f"{vid}:{rel}"


def test_anchor_json_files_match_counts(counts):
    assert ANCHOR_PRE.is_file() and ANCHOR_POST.is_file()
    pre = json.loads(ANCHOR_PRE.read_text(encoding="utf-8"))
    post = json.loads(ANCHOR_POST.read_text(encoding="utf-8"))
    assert pre["pricing_inputs"] == counts["anchors"]["pre"]["pricing_inputs"]
    assert post["pricing_inputs"] == counts["anchors"]["post"]["pricing_inputs"]
    assert pre["headline_net"] == post["headline_net"]


def test_output_sha_ledger_and_rerun(counts):
    ledger = json.loads(OUTPUT_SHA.read_text(encoding="utf-8"))
    files = ledger["files"]
    assert len(files) == 6
    assert ledger["rerun_check"]["status"] == "IDENTICAL"
    assert ledger["rerun_check"]["n_diff"] == 0
    assert counts["rerun_check"]["status"] == "IDENTICAL"
    for rel, sha in files.items():
        assert file_size(ROOT / rel) == sha, rel
    assert files == counts["output_sha"]


def test_q_status_and_hold_decision(counts):
    q = counts["q_status"]
    assert q["Q1"]["status"] == "CONDITIONAL"
    assert q["Q2"]["status"] == "PASS"
    assert q["Q3"]["status"] == "PASS"
    assert q["Q4"]["status"] == "FAIL"
    assert q["Q5"]["status"] == "NOT_EVALUATED"
    assert q["Q6"]["status"] == "PASS"
    assert q["Q7"]["status"] == "CITED_PASS_HEADLINE"
    assert q["Q8"]["status"] == "FAIL"
    assert counts["decision"]["value"] == "HOLD"
    assert q["_overall"]["all_pass"] is False
    assert len(counts["proposed_registry_diff"]) >= 7


def test_memo_exists_and_says_hold():
    memo = ART / "3.6_decision_memo.md"
    require_artifacts(memo, hint='缺 3.6 决策备忘产物')
    assert memo.is_file()
    txt = memo.read_text(encoding="utf-8")
    assert "HOLD" in txt
    assert "proposed_registry_diff" in txt
    assert "3.4" in txt and "3.5" in txt
