"""Phase 3.1 as-of 行业口径 shadow 契约测试（加法，只读真实产物）。

覆盖（3.0_plan §2.1 R1/R2/R3 + 项目 3.1 任务书）：
  1. as-of 边界：只取 `effective_from <= 月首日`（无前视）、恰落月首日当月生效（allow_exact_matches）；
  2. 2021 版切换（2021-07-30）：firm-month 计数与 L2 组数 104 → 131 的 vintage 断点；
  3. K 目标：as-of 池化 49 / 逐年 38/40/44/49/50/52/68/57/64；latest 池化 44 / 26…62（逐值对账）；
  4. as-of L2 组数逐年 102/103/103/104/104/104/104/131/131；
  5. 标签不一致率（pricing_gate 样本 307,503 行）L2 37.35%/L1 19.62%（±0.05pp）；
  6. shadow 版本名 `pm_ma_ewcos_teacher_p3asof_v1_{y}`，与 headline `pm_ma_ewcos_teacher_v1_{y}` 不同名；
  7. shadow 边表合同：键唯一、无自环、行内权重和 = 1、`available_at` = (y+1)-04-30、9 年齐；
  8. 行业基线 LOO 手算抽查（≤1e-12）+ 列名/行数与 headline 对比表；
  9. headline 未变：9 锚 MATCH（manifest）+ headline 网络全文件 sha 与 `3.1_anchor_pre/post.json` 一致；
 10. 输出路径守卫：任何 shadow 写出路径不得命中 headline/G0/configs。

运行：python -m pytest tests/test_shadow_asof_industry.py -q
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
SCRIPT = ROOT / "scripts/shadow_asof_industry.py"
SHADOW_DIR = ROOT / "data/processed/phase3_shadow"
ASOF_MAP = SHADOW_DIR / "sw2_asof_map_shadow.parquet"
ASOF_BASELINES = SHADOW_DIR / "ind_baselines_p3asof.parquet"
NET_DIR = ROOT / "data/processed/network_ma_teacher_p3asof"
HEADLINE_NET = ROOT / "data/processed/network_ma_teacher"
ART = ROOT / "artifacts/phase3"
K_TARGETS = ART / "3.1_k_targets.json"
NETWORK_COMPARE = ART / "3.1_network_compare.csv"
ANCHOR_PRE = ART / "3.1_anchor_pre.json"
ANCHOR_POST = ART / "3.1_anchor_post.json"
SAMPLE = ROOT / "data/processed/pricing_sample_ma_teacher_v2.parquet"
SRCM = ROOT / "data/processed/source_month_argmax_panel.csv"
SW2_MAP = ROOT / "data/processed/sw2_industry_map.csv"
SW2_LATEST = ROOT / "data/processed/sw2_industry_latest.csv"
TEACHER_PANEL = ROOT / "data/processed/teacher_mda_panel.csv"

YEARS = list(range(2014, 2023))
SHADOW_PREFIX = "pm_ma_ewcos_teacher_p3asof_v1"
HEADLINE_PREFIX = "pm_ma_ewcos_teacher_v1"
K_ASOF_PER_YEAR = {2014: 38, 2015: 40, 2016: 44, 2017: 49, 2018: 50,
                   2019: 52, 2020: 68, 2021: 57, 2022: 64}
K_LATEST_PER_YEAR = {2014: 26, 2015: 27, 2016: 30, 2017: 38, 2018: 37,
                     2019: 46, 2020: 52, 2021: 58, 2022: 62}
ASOF_L2_GROUPS = {2014: 102, 2015: 103, 2016: 103, 2017: 104, 2018: 104,
                  2019: 104, 2020: 104, 2021: 131, 2022: 131}
IND_COLS = ["ind_l1_max_ew", "ind_l1_max_vw", "ind_l1_ret_ew", "ind_l1_ret_vw",
            "ind_l2_max_ew", "ind_l2_max_vw", "ind_l2_ret_ew", "ind_l2_ret_vw"]


def file_size(path):
    return Path(path).stat().st_size


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_asof_industry", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p31():
    assert SCRIPT.is_file(), f"缺少脚本 {SCRIPT}"
    return load_module()


@pytest.fixture(scope="module")
def map_long(p31):
    return p31.load_sw2_map()


@pytest.fixture(scope="module")
def map_artifact() -> pd.DataFrame:
    assert ASOF_MAP.is_file(), f"缺少 as-of 映射产物（先运行 {SCRIPT.name}）：{ASOF_MAP}"
    return pd.read_parquet(ASOF_MAP)


@pytest.fixture(scope="module")
def k_targets() -> dict:
    assert K_TARGETS.is_file(), f"缺少 K 目标产物（先运行 {SCRIPT.name}）：{K_TARGETS}"
    return json.loads(K_TARGETS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. as-of 边界
def test_asof_map_backward_only_no_lookahead(p31):
    """合成有效日序列：as-of 只取 effective_from <= 月首日 的最近一条（无前视）。"""
    m = pd.DataFrame({
        "code6": ["000001"] * 3,
        "effective_from": pd.to_datetime(["1991-04-03", "2014-02-21", "2021-07-30"]),
        "l1_code": ["440000", "480000", "480000"],
        "l2_code": ["440100", "480100", "480300"],
        "l3_code": ["x", "y", "z"],
        "source_version": ["a", "b", "c"],
        "is_current_version": [False, False, True],
    })
    fm = pd.DataFrame({"firm_id": ["sz000001"] * 6,
                       "month": ["1991-04", "1991-05", "2014-02", "2014-03",
                                 "2021-07", "2021-08"],
                       "code6": ["000001"] * 6})
    lab = p31.asof_labels_for_months(m, fm).set_index("month")
    # 1991-04：生效日 1991-04-03 > 1991-04-01 → 当月无可用分类（不得前视取 4/3 的版本）
    assert pd.isna(lab.loc["1991-04", "l2_code"])
    assert lab.loc["1991-05", "l2_code"] == "440100"
    assert lab.loc["2014-02", "l2_code"] == "440100"      # 2014-02-21 > 02-01 → 仍用旧版
    assert lab.loc["2014-03", "l2_code"] == "480100"
    assert lab.loc["2021-07", "l2_code"] == "480100"      # 07-30 > 07-01
    assert lab.loc["2021-08", "l2_code"] == "480300"      # 08-01 >= 07-30 → 新版


def test_asof_boundary_exact_month_first_day(p31):
    """恰落月首日：当月即生效（allow_exact_matches 边界，非严格 >）。"""
    m = pd.DataFrame({
        "code6": ["000002"] * 2,
        "effective_from": pd.to_datetime(["2001-01-01", "2021-08-01"]),
        "l1_code": ["1", "2"], "l2_code": ["10", "20"], "l3_code": ["a", "b"],
        "source_version": ["v1", "v2"], "is_current_version": [False, True],
    })
    fm = pd.DataFrame({"firm_id": ["sz000002"], "month": ["2021-08"], "code6": ["000002"]})
    lab = p31.asof_labels_for_months(m, fm)
    assert lab.loc[0, "l2_code"] == "20"
    assert lab.loc[0, "effective_from"] == pd.Timestamp("2021-08-01")


def test_asof_map_key_unique(map_long):
    assert not map_long.duplicated(["code6", "effective_from"]).any()
    assert map_long["effective_from"].notna().all()
    assert map_long["source_version"].str.startswith("sw2_classification_effective_").all()


# ---------------------------------------------------------------- 2. 2021 vintage 切换
def test_2021_vintage_switch(map_long, map_artifact):
    aud = json.loads((ART / "3.1_map_audit.json").read_text(encoding="utf-8"))
    sw = aud["vintage_switch"]
    assert sw["switch_date"] == "2021-07-30"
    assert sw["per_month"]["2021-07"]["n_l2_groups"] == 104
    assert sw["per_month"]["2021-08"]["n_l2_groups"] == 131
    assert sw["per_month"]["2021-08"]["eff_eq_switch_date"] > 2000
    # 映射产物（firm-month 面板）中切到 2021 版的 firm-month 数
    n_2021 = int((map_artifact["effective_from"] == pd.Timestamp("2021-07-30")).sum())
    assert n_2021 > 2000
    # 无前视：所有有标签的 firm-month 都满足 effective_from <= 月首日（无映射行为 NaT）
    eff = map_artifact["effective_from"]
    have = eff.notna()
    assert have.sum() == len(map_artifact) - 1488
    assert (eff[have] <= pd.to_datetime(map_artifact.loc[have, "month"] + "-01")).all()
    assert map_artifact.loc[~have, "l2_code"].isna().all()


# ---------------------------------------------------------------- 3/4. K 目标 + 组数
def _k_targets_recompute(mode: str) -> dict:
    p31 = load_module()
    m = p31.load_sw2_map()
    panel = pd.read_csv(TEACHER_PANEL, usecols=["firm_id", "year", "status"])
    panel = panel[panel["status"] == "ok"].copy()
    panel["firm_id"] = panel["firm_id"].astype(str)
    return p31.sw2_k_target(m, panel, YEARS, mode)


def test_k_target_asof_matches_registered():
    got = _k_targets_recompute("asof")
    assert got["median_peer_pooled_rounded"] == 49
    assert got["median_peer_pooled_firm_level"] == 49.0
    per = {p["text_year"]: int(round(p["median_peer"])) for p in got["per_year"]}
    assert per == K_ASOF_PER_YEAR
    groups = {p["text_year"]: p["sw2_groups"] for p in got["per_year"]}
    assert groups == ASOF_L2_GROUPS
    assert got["n_firm_years_unmatched"] <= 5          # 2020 单项 1 行（映射生效日晚于键）


def test_k_target_latest_matches_registered():
    got = _k_targets_recompute("latest")
    assert got["median_peer_pooled_rounded"] == 44
    per = {p["text_year"]: int(round(p["median_peer"])) for p in got["per_year"]}
    assert per == K_LATEST_PER_YEAR


def test_k_targets_artifact_consistent(k_targets):
    a = k_targets["asof"]
    assert a["median_peer_pooled_rounded"] == 49
    assert {p["text_year"]: int(round(p["median_peer"])) for p in a["per_year"]} == K_ASOF_PER_YEAR
    assert k_targets["registered_asof_pooled"] == 49
    assert k_targets["headline_k_target_used"] == 44
    assert k_targets["planned_network_variants"]["N1_p3asof"]["ver_prefix"] == SHADOW_PREFIX


# ---------------------------------------------------------------- 5. 不一致率
def test_label_mismatch_rate_vs_latest(p31, map_long):
    """不一致率（pricing_gate 样本 307,503 行，月首日口径）L2 37.35% / L1 19.62%（±0.05pp）。"""
    aud = json.loads((ART / "3.1_map_audit.json").read_text(encoding="utf-8"))
    mis = aud["mismatch_vs_latest_on_m1_sample"]
    assert mis["n_sample_rows"] == 307_503
    assert abs(mis["l2_mismatch_month_first_day"] - 0.3735) <= 0.0005
    assert abs(mis["l1_mismatch_month_first_day"] - 0.1962) <= 0.0005
    # 独立复算：直接由 as-of 映射（code6 × effective_from）+ latest 静态标签现算
    smp = pd.read_parquet(SAMPLE, columns=["focal_id", "month"]).rename(
        columns={"focal_id": "firm_id"})
    smp["code6"] = smp["firm_id"].str[2:].str.zfill(6)
    lab = p31.asof_labels_for_months(map_long, smp)
    lat = pd.read_csv(SW2_LATEST, dtype={"code6": str, "l1_code": str, "l2_code": str})
    lat["code6"] = lat["code6"].str.zfill(6)
    lat = lat.drop_duplicates("code6").set_index("code6")
    lab["latest_l2"] = lab["code6"].map(lat["l2_code"])
    lab["latest_l1"] = lab["code6"].map(lat["l1_code"])
    ok = lab["l2_code"].notna() & lab["latest_l2"].notna()
    assert int(ok.sum()) == mis["n_comparable"]
    assert abs(float((lab.loc[ok, "l2_code"] != lab.loc[ok, "latest_l2"]).mean())
               - mis["l2_mismatch_month_first_day"]) <= 0.0005
    assert abs(float((lab.loc[ok, "l1_code"] != lab.loc[ok, "latest_l1"]).mean())
               - mis["l1_mismatch_month_first_day"]) <= 0.0005
    # 分年：2021-07-30 切换后两侧同版 → 2023 不一致率骤降
    per = mis["per_calendar_year"]
    assert per["2015"]["l2_mismatch"] > 0.5 and per["2023"]["l2_mismatch"] < 0.10
    assert mis["l2_mismatch_month_first_day"] >= 0.373 and mis["l1_mismatch_month_first_day"] < 0.20


# ---------------------------------------------------------------- 6. 版本名
def test_shadow_version_names_distinct_from_headline():
    assert SHADOW_PREFIX != HEADLINE_PREFIX
    for y in YEARS:
        srep = json.loads((NET_DIR / f"{y}_build_report.json").read_text(encoding="utf-8"))
        hrep = json.loads((HEADLINE_NET / f"{y}_build_report.json").read_text(encoding="utf-8"))
        assert srep["network_version"] == f"{SHADOW_PREFIX}_{y}"
        assert hrep["network_version"] == f"{HEADLINE_PREFIX}_{y}"
        assert srep["network_version"] != hrep["network_version"]
        assert srep["available_at"] == f"{y + 1}-04-30"
        sup = pd.read_csv(NET_DIR / f"{y}_support.csv", usecols=["network_version"],
                          dtype=str)["network_version"].unique().tolist()
        assert sup == [f"{SHADOW_PREFIX}_{y}"]
        assert srep["params"]["K_rule"] == "per_year_target_peer_match(49)"
        assert srep["corpus"] == "teacher"


# ---------------------------------------------------------------- 7. 边表合同
def test_shadow_edges_contract_one_year():
    y = 2019
    e = pd.read_csv(NET_DIR / f"{y}_similarity.csv",
                    dtype={"focal_id": str, "source_id": str})
    assert not e.duplicated(["focal_id", "source_id"]).any()
    assert (e["focal_id"] != e["source_id"]).all()
    assert (e["weight"] > 0).all() and (e["weight"] <= 1.0 + 1e-12).all()
    s = e.groupby("focal_id", sort=False)["weight"].sum()
    assert float((s - 1.0).abs().max()) <= 1e-9
    rep = json.loads((NET_DIR / f"{y}_build_report.json").read_text(encoding="utf-8"))
    assert len(e) == int(rep["n_edges"])


def test_shadow_network_nine_years_present():
    for y in YEARS:
        for suffix in ("similarity.csv", "support.csv", "build_report.json"):
            assert (NET_DIR / f"{y}_{suffix}").is_file(), f"缺 {y}_{suffix}"
        assert (NET_DIR / "clusters" / f"{y}_clusters.csv").is_file()


def test_shadow_uses_headline_lexicon_and_filter():
    """唯一改动 = K 目标；词表 / df 阈值 / 筛选 / 相似度方法须与 headline 逐值相同。"""
    for y in YEARS:
        srep = json.loads((NET_DIR / f"{y}_build_report.json").read_text(encoding="utf-8"))
        hrep = json.loads((HEADLINE_NET / f"{y}_build_report.json").read_text(encoding="utf-8"))
        assert srep["lexicon"] == hrep["lexicon"]
        assert srep["n_nodes"] == hrep["n_nodes"]
        assert srep["filter_stats"] == hrep["filter_stats"]
        for k in ("similarity_method", "cluster_linkage", "competitor_set",
                  "min_product_terms", "sample_filter"):
            assert srep["params"][k] == hrep["params"][k], f"{y}.{k} 与 headline 不一致"
        assert srep["params"]["term_weighting"] == "ew_0/1"


def test_rerun_byte_identical():
    """R3：确定性产物（边表/support/聚类/映射/基线）重跑逐字节一致。"""
    p = ART / "3.1_output_sha.json"
    assert p.is_file(), f"缺少 sha 台账：{p}"
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d.get("rerun_check"), "缺 rerun_check（需至少两次 --steps all/accept 运行）"
    assert d["rerun_check"]["status"] == "IDENTICAL", d["rerun_check"]
    # 台账确实覆盖 9 年边表 + support + 聚类 + 映射/基线
    keys = list(d["sha"])
    assert sum(k.endswith("_similarity.csv") for k in keys) == 9
    assert sum(k.endswith("_support.csv") for k in keys) == 9
    assert sum(k.endswith("_clusters.csv") for k in keys) == 9
    assert any(k.endswith("sw2_asof_map_shadow.parquet") for k in keys)
    assert any(k.endswith("ind_baselines_p3asof.parquet") for k in keys)


def test_sample_copy_preserves_headline_columns():
    """3.5 依赖件：headline 样本 + 8 个 _p3asof 列；headline 列一字不动。"""
    p = SHADOW_DIR / "pricing_sample_ma_teacher_p3asof_baselines.parquet"
    assert p.is_file(), f"缺少样本副本：{p}"
    d = pd.read_parquet(p)
    assert len(d) == 307_503
    assert not d.duplicated(["focal_id", "month"]).any()
    assert set(IND_COLS).issubset(d.columns)
    assert set(f"{c}_p3asof" for c in IND_COLS).issubset(d.columns)
    ref = pd.read_parquet(SAMPLE, columns=["focal_id", "month", "PeerMAXraw", "R_next"] + IND_COLS)
    j = d.merge(ref, on=["focal_id", "month"], suffixes=("_new", ""))
    for c in ["PeerMAXraw", "R_next"] + IND_COLS:
        a, b = j[c + "_new"], j[c]
        assert ((a == b) | (a.isna() & b.isna())).all(), f"headline 列 {c} 被改"


def test_version_spec_injection_loads_shadow(p31):
    from peermax_panel_build import load_versions, select_version
    spec = {"version_prefix": SHADOW_PREFIX, "years": YEARS, "available_at_rule": "(y+1)-04-30"}
    vers = load_versions(NET_DIR, version_spec=spec)
    assert sorted(vers) == YEARS
    assert all(vers[y]["available_at"] == f"{y + 1}-04-30" for y in YEARS)
    assert select_version(vers, "2016-06")["text_year"] == 2015
    assert select_version(vers, "2021-06")["text_year"] == 2020
    assert select_version(vers, "2024-01")["text_year"] == 2022
    assert select_version(vers, "2015-01") is None


# ---------------------------------------------------------------- 8. 基线
def test_baselines_columns_and_coverage():
    b = pd.read_parquet(ASOF_BASELINES)
    assert set([f"{c}_p3asof" for c in IND_COLS]).issubset(b.columns)
    assert {"firm_id", "month", "l1_asof", "l2_asof"}.issubset(b.columns)
    assert b["month"].min() == "2015-01" and b["month"].max() == "2023-12"
    # as-of 标签覆盖率 ≥99.5%（R1 停机线）
    assert float(b["l2_asof"].notna().mean()) >= 0.995


def test_baselines_loo_handcheck():
    b = pd.read_parquet(ASOF_BASELINES, columns=["firm_id", "month", "l2_asof", "ind_l2_max_ew_p3asof"])
    srcm = pd.read_csv(SRCM, usecols=["source_id", "month", "MAXraw"],
                       dtype={"source_id": str, "month": str})
    srcm = srcm.rename(columns={"source_id": "firm_id"})
    v = srcm[srcm["month"].between("2015-01", "2023-12") & srcm["MAXraw"].notna()]
    v = b.merge(v, on=["firm_id", "month"], how="inner")
    top = v.groupby(["month", "l2_asof"]).size().sort_values(ascending=False).head(20)
    checked, worst = 0, 0.0
    for (mo, l2), _ in top.items():
        g = v[(v["month"] == mo) & (v["l2_asof"] == l2)]
        for f in g["firm_id"].head(3):
            rest = g.loc[g["firm_id"] != f, "MAXraw"]
            got = g.loc[g["firm_id"] == f, "ind_l2_max_ew_p3asof"]
            if rest.empty or got.empty or not np.isfinite(got.iloc[0]):
                continue
            worst = max(worst, abs(float(got.iloc[0]) - float(rest.mean())))
            checked += 1
    assert checked >= 20
    assert worst <= 1e-12


def test_baseline_diff_registered_values():
    d = pd.read_csv(ART / "3.1_baseline_diff.csv")
    fm = d[d["scope"] == "FM_181178"].set_index("column")
    assert abs(fm.loc["ind_l1_max_ew", "corr_asof_vs_headline"] - 0.9564) <= 0.002
    assert abs(fm.loc["ind_l2_max_ew", "corr_asof_vs_headline"] - 0.9005) <= 0.002
    assert abs(fm.loc["ind_l1_max_ew", "median_abs_diff"] - 0.00111) <= 0.0002
    assert abs(fm.loc["ind_l2_max_ew", "median_abs_diff"] - 0.00231) <= 0.0002
    assert fm.loc["ind_l1_max_ew", "sign_agree_share"] == 1.0
    assert fm.loc["ind_l2_max_ew", "missing_asof"] <= 0.001


# ---------------------------------------------------------------- 9. headline 未变 / 10. 守卫
def test_pricing_inputs_sha_anchors_match_manifest():
    assert_input_anchors()

def test_headline_network_sha_unchanged():
    assert ANCHOR_PRE.is_file() and ANCHOR_POST.is_file(), "缺少 3.1 锚台账"
    pre = json.loads(ANCHOR_PRE.read_text(encoding="utf-8"))
    post = json.loads(ANCHOR_POST.read_text(encoding="utf-8"))
    for snap in (pre, post):
        for rel, sha in snap["headline_network_sha"].items():
            p = ROOT / rel
            assert p.is_file(), f"headline 文件缺失: {rel}"
            assert file_size(p) == sha, f"headline 文件 sha 变化: {rel}"
    assert pre["headline_network_sha"] == post["headline_network_sha"]
    assert pre["pricing_inputs_9"] == post["pricing_inputs_9"]
    assert pre["extra_inputs"] == post["extra_inputs"]
    counts = None
    if (ART / "3.1_counts.json").is_file():
        counts = json.loads((ART / "3.1_counts.json").read_text(encoding="utf-8"))
    if counts and counts.get("anchor_compare"):
        assert counts["anchor_compare"]["status"] == "MATCH"


def test_output_path_guard_blocks_headline(p31):
    ok = p31.assert_output_path_guard(NET_DIR / "2099_similarity.csv")
    assert ok == (NET_DIR / "2099_similarity.csv").resolve()
    for bad in [HEADLINE_NET / "2019_similarity.csv",
                ROOT / "reports/data_gates/G0/pricing_sample_report.json",
                ROOT / "configs/parameter_registry.yaml",
                ROOT / "data/processed/pricing_sample_ma_teacher_v2.parquet",
                ROOT / "data/processed/sw2_industry_map.csv"]:
        with pytest.raises(SystemExit) as e:
            p31.assert_output_path_guard(bad)
        assert e.value.code == 2


# ---------------------------------------------------------------- 验收表
def test_network_compare_matches_build_reports():
    d = pd.read_csv(NETWORK_COMPARE)
    assert len(d) == len(YEARS)
    ladder = None
    for _, r in d.iterrows():
        y = int(r["text_year"])
        rep = json.loads((NET_DIR / f"{y}_build_report.json").read_text(encoding="utf-8"))
        assert int(r["p3asof_edges"]) == int(rep["n_edges"])
        assert int(r["p3asof_K"]) == int(rep["k_clusters"])
        assert r["p3asof_network_version"] == f"{SHADOW_PREFIX}_{y}"
        assert 0.0 <= float(r["jaccard_vs_headline"]) <= 1.0
        assert float(r["p3asof_lift_asof"]) > 1.0 and float(r["p3asof_lift_latest"]) > 1.0
        assert float(r["p3asof_isolated_share"]) < 0.15
        delta = abs(float(r["p3asof_peer_median"]) - 49.0)
        if delta > 2.0:      # 仅当阶梯上 49 不可达时允许（须有 k_ladder 证据）
            assert (ART / "3.1_k_ladder.json").is_file(), "偏离 >2 但缺阶梯诊断"
            if ladder is None:
                ladder = json.loads((ART / "3.1_k_ladder.json").read_text(encoding="utf-8"))
            assert ladder["reachability"][str(y)]["target_49_reachable"] is False


def test_k_ladder_documents_2017_plateau():
    """2017：公司层面中位 peer 阶梯在 K=291 处由 61 跳到 44 → 目标 49 不可达（closest 取 291）。"""
    p = ART / "3.1_k_ladder.json"
    assert p.is_file(), f"缺少阶梯诊断（--steps ladder）：{p}"
    lad = json.loads(p.read_text(encoding="utf-8"))
    r17 = lad["reachability"]["2017"]
    assert r17["target_49_reachable"] is False
    assert r17["K_at_median_peer_49"] == []
    sl = r17["ladder_slice"]
    assert float(sl["290"]) >= 50.0 and float(sl["291"]) < 45.0
    d = pd.read_csv(NETWORK_COMPARE)
    row = d[d["text_year"] == 2017].iloc[0]
    assert int(row["p3asof_K"]) == int(row["headline_K"]) == 291
    assert float(row["jaccard_vs_headline"]) == 1.0      # 同 K → 同标签 → 同边集
