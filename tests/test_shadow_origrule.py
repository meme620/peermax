"""Phase 3.3 原文第四步阈值规则 shadow 契约测试（加法，合成 + 只读真实产物）。

覆盖（`3.0_plan.md` §2.3 R1/R2/R3 + 3.3 任务书）：
  1. 原文规则阈值 = 团内最小相似度（含团外候选、边界严格 >、并列、单点团无 peer）；
  2. 项目规则对照（peer = 团规模 − 1）与「候选域 = 全部节点」的实现证据（团外有向边 > 0）；
  3. 退化计数与裁决规则（peer 中位 / lift / 占市场比）；
  4. PC：HHI 合成手算；真实产物中同团规则重算与 headline `{y}_pc.csv` 逐值一致；
     2014–2016 无营收年 → pc 留空不填 0；
  5. 真实产物版本/时间契约：`shadow_version=p3shadow_v1`、`available_at=(y+1)-04-30`、9 年 × 2 规则齐；
  6. R1 矩阵对账（与 headline similarity.csv 逐对）与 freeze 报告对账（同团 peer/lift/K）；
  7. headline 未变：9 锚 MATCH + headline 网络 45 文件 sha 与 `3.3_anchor_{pre,post}.json` 一致；
  8. 输出路径守卫：shadow 写出路径不得命中 headline/G0/configs。

运行：python -m pytest tests/test_shadow_origrule.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import assert_input_anchors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SCRIPT = ROOT / "scripts/shadow_origrule.py"
SHADOW_DIR = ROOT / "data/processed/phase3_shadow"
SHADOW_CSV = SHADOW_DIR / "pc_origrule_shadow.csv"
SHADOW_META = SHADOW_DIR / "pc_origrule_shadow.meta.json"
HEADLINE_NET = ROOT / "data/processed/network_ma_teacher"
P3ASOF_NET = ROOT / "data/processed/network_ma_teacher_p3asof"
ART = ROOT / "artifacts/phase3"
READOUT_CSV = ART / "3.3_origrule_readout.csv"
COUNTS_JSON = ART / "3.3_counts.json"
ANCHOR_PRE = ART / "3.3_anchor_pre.json"
ANCHOR_POST = ART / "3.3_anchor_post.json"
FREEZE_REPORT = ROOT / "reports/data_gates/G0/network_business_param_freeze_teacher.json"

YEARS = list(range(2014, 2023))
HEADLINE_NODES = {2014: 2384, 2015: 2363, 2016: 2503, 2017: 2721, 2018: 2835,
                  2019: 3354, 2020: 3427, 2021: 3570, 2022: 3942}
HEADLINE_K = {2014: 235, 2015: 238, 2016: 322, 2017: 291, 2018: 334,
              2019: 407, 2020: 414, 2021: 434, 2022: 463}
HEADLINE_EDGES = {2014: 157096, 2015: 156974, 2016: 152302, 2017: 214028,
                  2018: 204786, 2019: 238458, 2020: 246728, 2021: 249254,
                  2022: 293938}
RULE_ORIG = "origrule_minthr_global"
RULE_CLUSTER = "same_cluster_members"


def file_size(path):
    return Path(path).stat().st_size


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_origrule", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p33():
    assert SCRIPT.is_file(), f"缺少脚本 {SCRIPT}"
    return load_module()


@pytest.fixture(scope="module")
def counts() -> dict:
    assert COUNTS_JSON.is_file(), f"缺少计数产物（先运行 {SCRIPT.name}）：{COUNTS_JSON}"
    return json.loads(COUNTS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shadow() -> pd.DataFrame:
    assert SHADOW_CSV.is_file(), f"缺少 shadow 产物（先运行 {SCRIPT.name}）：{SHADOW_CSV}"
    return pd.read_csv(SHADOW_CSV, dtype={"focal_id": str, "cluster_id": int,
                                          "text_year": int})


# ---------------------------------------------------------------- 1. 原文规则（合成）
def test_thr_is_within_cluster_min(p33):
    # 团 {0,1} 内最小相似度 0.5；团 {2,3} 内最小 0.2（对角不得入 min）
    C = np.array([
        [1.0, 0.5, 0.9, 0.3],
        [0.5, 1.0, 0.55, 0.1],
        [0.9, 0.55, 1.0, 0.2],
        [0.3, 0.1, 0.2, 1.0],
    ])
    labels = np.array([1, 1, 2, 2])
    thr, st = p33.within_cluster_min_sim(C, labels)
    assert np.allclose(thr, [0.5, 0.5, 0.2, 0.2])
    assert st["n_pairs_within_clusters"] == 2
    assert st["thr_median"] == pytest.approx(0.35, abs=1e-12)
    # 对角为 1.0 也不得污染团内最小（两公司单团：团内最小 = 唯一对 0.7，不是对角 1.0）
    C2 = np.array([[1.0, 0.7], [0.7, 1.0]])
    thr2, _ = p33.within_cluster_min_sim(C2, np.array([1, 1]))
    assert thr2[0] == pytest.approx(0.7)


def test_candidate_domain_is_global_includes_out_of_cluster(p33):
    C = np.array([
        [1.0, 0.5, 0.9, 0.3],
        [0.5, 1.0, 0.55, 0.1],
        [0.9, 0.55, 1.0, 0.2],
        [0.3, 0.1, 0.2, 1.0],
    ])
    labels = np.array([1, 1, 2, 2])
    mask, thr, st = p33.orig_rule_mask(C, labels)
    # 公司 0（团 1，thr=0.5）：团外公司 2（sim 0.9）> 0.5 → 必须入选（字面规则的团外候选）
    assert mask[0, 2] and not mask[0, 1] and not mask[0, 3]
    # 公司 2（团 2，thr=0.2）：团外 0（0.9）、团内 1（0.55）入选；3（0.2）不入选（严格 >）
    assert mask[2, 0] and mask[2, 1] and not mask[2, 3]
    assert st["n_directed_edges_cross_cluster"] == 5
    assert st["candidate_domain"] == "all_nodes"
    assert not np.any(np.diag(mask))          # 对角排除（无自环）
    # 按行阈值 → mask 不必对称：sim(3,0)=0.3 > thr_3=0.2 入选，但 sim(0,3)=0.3 ≤ thr_0=0.5 不入选
    assert mask[3, 0] and not mask[0, 3]
    # 换阈值相同的两团 → 对称
    labels_sym = np.array([1, 1, 1, 1])       # 同团：thr 相同（团内最小 0.1）
    mask_sym, thr_sym, _ = p33.orig_rule_mask(C, labels_sym)
    assert thr_sym[0] == pytest.approx(0.1)
    assert np.array_equal(mask_sym, mask_sym.T)


def test_strict_boundary_excludes_equal_and_ties(p33):
    # 某对相似度恰好等于 thr → 不入 peer 集合（严格大于）；并列值行为确定
    C = np.array([
        [1.0, 0.4, 0.4, 0.4],
        [0.4, 1.0, 0.4, 0.4],
        [0.4, 0.4, 1.0, 0.4],
        [0.4, 0.4, 0.4, 1.0],
    ])
    labels = np.array([1, 1, 2, 3])
    mask, thr, st = p33.orig_rule_mask(C, labels)
    assert np.allclose(thr, [0.4, 0.4, 1.0, 1.0])
    assert mask.sum() == 0                    # 全部对 = thr → 严格 > 无一入选
    assert st["n_directed_edges_cross_cluster"] == 0


def test_singleton_cluster_has_no_peer(p33):
    C = np.array([[1.0, 0.9], [0.9, 1.0]])
    labels = np.array([7, 8])
    mask, thr, st = p33.orig_rule_mask(C, labels)
    assert np.allclose(thr, 1.0)
    assert mask.sum() == 0 and st["n_singleton_clusters"] == 2
    assert int((mask.sum(axis=1) == 0).all())


def test_cluster_rule_peers_are_cluster_size_minus_one(p33):
    labels = np.array([1, 1, 2, 2, 2, 3])
    m = p33.cluster_rule_mask(labels)
    peers = m.sum(axis=1) - 1
    assert list(peers) == [1, 1, 2, 2, 2, 0]


def test_edge_share_lift_matches_manual(p33):
    mask = np.array([
        [False, True, True, False],
        [True, False, True, False],
        [True, True, False, False],
        [False, False, False, False],
    ])
    l2 = np.array([0, 0, 1, 1])
    out = p33.edge_share_lift(mask, l2)
    # 有向边 6 条：同二级 (0,1) 双向 = 2 条 → 份额 2/6；基线 = 同二级无序对 2/6 对
    assert out["n_directed_edges"] == 6
    assert out["n_same_sw2_directed_edges"] == 2
    assert out["same_sw2_edge_share"] == pytest.approx(2 / 6, abs=1e-4)
    assert out["base_same_sw2_rate"] == pytest.approx(2 / 6, abs=1e-5)
    assert out["sw2_lift"] == pytest.approx(1.0, abs=1e-2)


# ---------------------------------------------------------------- 2. 退化计数 / 裁决
def test_degeneracy_counts_and_verdict(p33):
    rows = [{"text_year": 2019, "peer_median": 3000.0, "peer_mean": 2900.0,
             "peer_share_of_market_median": 0.9, "share_peers_ge_90pct_market": 0.5,
             "sw2_lift_latest": 1.2, "sw2_lift_asof": 1.1}]
    v = p33.degeneracy_verdict(rows)
    assert v["degenerate"] is True and v["median_of_year_peer_medians"] == 3000.0
    # 停机判据的互斥面：peer 中位 <500 且 lift >5 → 未退化
    v2 = p33.degeneracy_verdict([dict(rows[0], peer_median=400.0, sw2_lift_latest=6.0)])
    assert v2["degenerate"] is False
    # 任一不成立 → 退化
    assert p33.degeneracy_verdict([dict(rows[0], peer_median=600.0, sw2_lift_latest=6.0)])["degenerate"]
    assert p33.degeneracy_verdict([dict(rows[0], peer_median=400.0, sw2_lift_latest=4.0)])["degenerate"]


def test_peer_stats_share_of_market(p33):
    ps = p33.peer_stats(np.array([0.0, 1.0, 2.0]), np.array([0, 2, 0]))
    assert ps["n_firms"] == 3
    assert ps["peer_median"] == 1.0 and ps["isolated_share"] == pytest.approx(1 / 3, abs=1e-4)
    assert ps["peer_share_of_market_median"] == pytest.approx(0.5, abs=1e-6)   # 1/(3−1)


# ---------------------------------------------------------------- 3. PC（合成手算）
def test_hhi_pc_manual_and_missing_rules(p33):
    # 三成员营收 100/100/200 → HHI = (0.25²+0.25²+0.5²) = 0.375
    r = p33.hhi_pc([100.0, 100.0, 200.0], True)
    assert r["pc"] == pytest.approx(-0.375, abs=1e-8) and r["n_rev_covered"] == 2
    # 缺营收不入分母、不计覆盖，但照报
    r2 = p33.hhi_pc([100.0, None, 300.0], True)
    assert r2["pc"] == pytest.approx(-(0.25 ** 2 + 0.75 ** 2), abs=1e-8)
    assert r2["n_rev_covered"] == 1 and r2["revenue_coverage"] == pytest.approx(2 / 3, abs=1e-4)
    # 覆盖 <2 / focal 缺营收 → 空（不填 0）
    assert p33.hhi_pc([100.0, None], True)["pc"] is None
    assert p33.hhi_pc([100.0, 100.0], False)["pc"] is None
    # 非正营收录入缺失（network_business_competition_hhi 同式）
    assert p33.hhi_pc([0.0, -5.0, 300.0], True)["n_rev_covered"] == 0


# ---------------------------------------------------------------- 4. 真实产物：版本 / available_at / 覆盖
def test_shadow_csv_version_and_available_at(shadow):
    assert set(shadow["shadow_version"]) == {"p3shadow_v1"}
    assert set(shadow["cluster_source"]) == {"headline"}
    for y, g in shadow.groupby("text_year"):
        assert set(g["peer_rule"]) == {RULE_ORIG, RULE_CLUSTER}
        rep = json.loads((HEADLINE_NET / f"{y}_build_report.json").read_text(encoding="utf-8"))
        assert set(g["available_at"]) == {str(rep["available_at"])} == {f"{y + 1}-04-30"}
        assert set(g["headline_network_version"]) == {str(rep["network_version"])}
        assert len(g) == 2 * HEADLINE_NODES[y]


def test_shadow_csv_pc_unavailable_years_blank(shadow):
    early = shadow[shadow["text_year"].isin([2014, 2015, 2016])]
    assert set(early["pc_status"]) == {"revenue_year_not_in_panel"}
    assert early["pc"].isna().all() and early["hhi"].isna().all()
    late = shadow[shadow["text_year"] >= 2017]
    assert late["pc_status"].isin(["support", "singleton_cluster", "focal_revenue_missing",
                                   "no_peer_above_thr"]).all()
    assert not (late["pc"] == 0).any()        # 缺失不填 0


def test_cluster_rule_pc_matches_headline_registered(shadow):
    cl = shadow[(shadow["peer_rule"] == RULE_CLUSTER) & (shadow["text_year"] >= 2017)]
    sup = cl[cl["pc"].notna()]
    assert len(sup) > 15_000
    assert (sup["pc_matches_headline"] == 1).all()
    assert np.allclose(sup["pc"], sup["pc_headline"], atol=1e-7)
    # 原文规则 PC 与 headline PC 并排可得（同 focal 行），且量级显著更小
    orig = shadow[(shadow["peer_rule"] == RULE_ORIG) & (shadow["text_year"] >= 2017)]
    both = orig[orig["pc"].notna() & orig["pc_headline"].notna()]
    assert len(both) > 15_000
    assert (both["pc"].abs() < both["pc_headline"].abs()).mean() > 0.9


# ---------------------------------------------------------------- 5. 真实产物：计数 / 裁决 / 对账
def test_origrule_degenerate_under_current_lexicon(counts):
    d = counts["degeneracy"]
    assert d["degenerate"] is True
    assert d["median_of_year_peer_medians"] >= 500
    assert d["median_of_year_lifts_latest"] <= 5
    for y in YEARS:
        assert d["peer_median_by_year"][str(y)] >= 500
        assert d["peer_share_of_market_median_by_year"][str(y)] >= 0.3
        assert d["sw2_lift_latest_by_year"][str(y)] <= 5
    assert counts["conclusion"]["status"] == "不可比/未复现（Phase 0.2 结论维持）"
    assert counts["conclusion"]["verdict_wording"] == "原文规则在当前词表下不可构造/退化"
    assert counts["conclusion"]["degenerate_under_current_lexicon"] is True


def test_origrule_readout_vs_project_rule_and_freeze(counts):
    rr = counts["readout"]["rows"]
    hl = [r for r in rr if r["cluster_source"] == "headline"]
    orig = {r["text_year"]: r for r in hl if r["peer_rule"] == RULE_ORIG}
    cl = {r["text_year"]: r for r in hl if r["peer_rule"] == RULE_CLUSTER}
    freeze = json.loads(FREEZE_REPORT.read_text(encoding="utf-8"))
    assert set(orig) == set(YEARS) and set(cl) == set(YEARS)
    for y in YEARS:
        assert orig[y]["n_firms"] == HEADLINE_NODES[y]
        assert orig[y]["n_clusters"] == HEADLINE_K[y] == cl[y]["n_clusters"]
        # 同团对照 = freeze 登记读数（peer 中位 / lift / 团数逐值）
        assert cl[y]["peer_median"] == freeze["years"][str(y)]["peer_median"]
        assert cl[y]["sw2_lift_latest"] == freeze["years"][str(y)]["sw2_lift"]
        # 原文规则 peer 远大于同团规则；lift 远小于同团规则
        assert orig[y]["peer_median"] > 10 * cl[y]["peer_median"]
        assert orig[y]["sw2_lift_latest"] < cl[y]["sw2_lift_latest"] / 5
        # 团外候选存在（候选域 = 全部节点）
        assert orig[y]["overlap_with_cluster_median"] is not None
        assert 0.0 <= orig[y]["overlap_with_cluster_median"] <= 1.0
    # 3.1 交叉读数（p3asof 聚类）：同样退化
    p3a = {r["text_year"]: r for r in rr if r["cluster_source"] == "p3asof"
           and r["peer_rule"] == RULE_ORIG}
    if P3ASOF_NET.is_dir():
        assert set(p3a) == set(YEARS)
        assert all(p3a[y]["peer_median"] >= 500 and p3a[y]["sw2_lift_latest"] <= 5 for y in YEARS)


def test_matrix_check_matches_headline_edge_counts(counts):
    mc = counts["readout"]["matrix_check"]
    for y in YEARS:
        assert mc[str(y)]["n_pairs"] == HEADLINE_EDGES[y]
        assert mc[str(y)]["max_abs_diff"] <= 1e-6


def test_pc_compare_summary_shape(counts):
    pcc = counts["pc_compare"]
    assert sorted(int(k) for k in pcc) == list(range(2017, 2023))
    for y, v in pcc.items():
        assert v["match"]["all_match_tol_1e-6"] is True
        assert v["orig"]["pc_median"] > v["headline_file"]["pc_median"]      # 原文规则量级更小
        assert v["orig"]["n_competitors_median"] >= 500
        if "corr" in v:
            assert -1.0 <= v["corr"]["orig_vs_headline"] <= 1.0
            assert v["corr"]["abs_magnitude_ratio_median"] < 0.5


def test_vocab_sensitivity_documents_narrow_lexicon_infeasible(counts):
    vs = counts["vocab_sensitivity"]
    kept = vs["terms_kept_unique_at_df_max"]
    assert kept["0.002"] >= 0.9 * kept["0.25"]           # df 收紧几乎不减词条
    assert kept["0.25"] > 1_000_000
    probes = vs["measured_active_vocab_probes"]
    if probes:
        # 原文 ~1.66 万量级：最严 df 阈值下仍 ≥2×（实测 2016 = 2.6×、2022 = 5.5×）
        assert all(p["active_vocab"] > 2 * vs["original_paper_effective_words_fy2016"]
                   for p in probes)
        assert vs["original_paper_effective_words_fy2016"] == 16598


# ---------------------------------------------------------------- 6. readout CSV 与 counts 一致
def test_readout_csv_consistent_with_counts(counts):
    assert READOUT_CSV.is_file(), "缺少 readout CSV"
    rd = pd.read_csv(READOUT_CSV)
    assert set(rd["cluster_source"]) >= {"headline"}
    s = rd[(rd["cluster_source"] == "headline") & (rd["peer_rule"] == RULE_ORIG)].set_index("text_year")
    for y in YEARS:
        assert s.loc[y, "peer_median"] == pytest.approx(
            counts["origrule_readout"]["per_year"][str(y)]["peer_median"])


# ---------------------------------------------------------------- 7. headline 未变
def test_pricing_inputs_sha_anchors_match_manifest():
    assert_input_anchors()

def test_headline_network_sha_unchanged_and_45_files():
    assert ANCHOR_PRE.is_file() and ANCHOR_POST.is_file(), "缺少 3.3 锚台账"
    pre = json.loads(ANCHOR_PRE.read_text(encoding="utf-8"))
    post = json.loads(ANCHOR_POST.read_text(encoding="utf-8"))
    for snap in (pre, post):
        assert len(snap["headline_network_sha"]) == 45
        for rel, sha in snap["headline_network_sha"].items():
            p = ROOT / rel
            assert p.is_file(), f"headline 文件缺失: {rel}"
            assert file_size(p) == sha, f"headline 文件 sha 变化: {rel}"
    assert pre["headline_network_sha"] == post["headline_network_sha"]
    assert pre["pricing_inputs_9"] == post["pricing_inputs_9"]
    assert pre["extra_inputs"] == post["extra_inputs"]


def test_counts_anchor_compare_match(counts):
    assert counts["anchor_compare"]["status"] == "MATCH"
    assert counts["anchor_compare"]["n_diff"] == 0
    assert counts["anchor_post"]["headline_network_n_files"] == 45


# ---------------------------------------------------------------- 8. 输出路径守卫
def test_output_path_guard_blocks_headline_and_inputs(p33):
    ok = p33.assert_output_path_guard(SHADOW_DIR / "pc_origrule_shadow.csv")
    assert ok == (SHADOW_DIR / "pc_origrule_shadow.csv").resolve()
    for bad in [HEADLINE_NET / "2019_similarity.csv",
                P3ASOF_NET / "2019_similarity.csv",
                ROOT / "reports/data_gates/G0/network_business_pc_report_teacher.json",
                ROOT / "configs/parameter_registry.yaml",
                ROOT / "data/processed/product_lexicon_v2_teacher.csv",
                ROOT / "data/raw/drive/operating_revenue_annual.pkl",
                ROOT / "scripts/network_business_edges_build.py"]:
        with pytest.raises(SystemExit) as e:
            p33.assert_output_path_guard(bad)
        assert e.value.code == 2


def test_shadow_meta_documents_rules():
    meta = json.loads(SHADOW_META.read_text(encoding="utf-8"))
    assert meta["shadow_version"] == "p3shadow_v1"
    assert RULE_ORIG in meta["rule_definitions"] and RULE_CLUSTER in meta["rule_definitions"]
    assert "全部节点" in meta["rule_definitions"][RULE_ORIG]
    assert meta["available_at_rule"] == "(y+1)-04-30（= headline build_report available_at，逐行对账）"
    assert meta["revenue_year_rule"].startswith("PC_{i,y} 需 y−1 年营收")
    assert "不必对称" in meta["origrule_asymmetry"]      # 行阈值 → 有向边集允许不对称
    assert set(meta["columns"]) == set(SHADOW_CSV_COLS_EXPECTED)


SHADOW_CSV_COLS_EXPECTED = [
    "shadow_version", "cluster_source", "peer_rule", "focal_id", "text_year",
    "cluster_id", "cluster_size", "n_competitors", "peer_share_of_market",
    "same_sw2_peer_share_latest", "thr", "sw2_lift_latest", "sw2_lift_asof",
    "base_same_sw2_rate_latest", "available_at", "headline_network_version",
    "headline_source_version", "revenue_year", "pc", "hhi", "n_rev_covered",
    "revenue_coverage", "pc_status", "pc_headline", "pc_matches_headline",
]
