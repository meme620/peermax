"""Phase 2.5 shadow rev 矩阵契约测试（加法，只读真实产物 + 合成序列）。

覆盖：
  1. 合成 lag 键平移：跨年（2019-12 ↔ 2020-01）、上一月缺失 -> NaN（不填 0）、来源键重复 -> 硬门退出。
  2. `attach_lagged_column`（合规 t-1）：只取 t-1、无同月/前视；`attach_forward_diagnostic`
     （前视 t+1）只用于 provenance，且不得进入任何规格 x 列表 / 矩阵 CSV。
  3. 真实样本：rev 同月桥（样本 rev == panel winsor(v2.R) ≤1e-12）、rev_lag1 FM 覆盖 100%、
     逐行等于 v2.R(t-1)。
  4. 相关：corr(rev, OwnMAX) 月度均值≈0.4246、corr(rev_lag1 合规, OwnMAX)≈+0.1192、
     corr(rev, rev_lag1)≈-0.0353；前视列 corr≈-0.0343（= 2.0_plan 引用值，provenance）。
  5. 六规格矩阵（a–f）：T=104、n=181,178；a 复现 headline；c/e/f/g 复现审计锚点；
     b/d 为合规 lag1 数字；e 与 f 同控制集逐值一致；全部六规格 PeerMAX |t|<1.96。
  6. 前视诊断：2.0_plan §2.5 引用的 rev_lag1 数字（-0.01341/-4.02、-0.01728/-4.91、corr -0.0343）
     由 R_{t+1} 逐值复现，且该列自身系数 >0.5（t>20）——即 plan 的 rev_lag1 = outcome。
  7. 产物契约：2.5_control_matrix.csv 的 sha 与 counts 一致、13 规格齐全、全行 T/n 一致。
  8. headline 4 文件（M1 样本 parquet / firm_chars / 面板 / v2 月收益）sha 前后不变且等于登记锚。
  9. 输出路径治理：不得命中 data/、G0、configs、headline 文件。

运行：python -m pytest tests/test_p2_rev_shadow.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/data_audit/p2_5_rev_shadow.py"
COUNTS = ROOT / "artifacts/phase2/2.5_counts.json"
MATRIX_CSV = ROOT / "artifacts/phase2/2.5_control_matrix.csv"
SAMPLE = ROOT / "data/processed/m1_common_sample_ma_teacher_v2.parquet"
RETURNS_V2 = ROOT / "data/processed/monthly_returns_clean_v2.csv"
FIRM_CHARS = ROOT / "data/processed/firm_chars_clean.csv"
PRICING_PANEL = ROOT / "data/processed/monthly_pricing_panel_ma_teacher.csv"
PRICING_TABLE_V2 = ROOT / "reports/data_gates/G0/s3_m1_pricing_table_v2.json"
SAMPLE_REPORT_V2 = ROOT / "reports/data_gates/G0/s3_m1_sample_report_v2.json"
MANIFEST = ROOT / "configs/version_manifest.yaml"

HEADLINE_FILES = {
    "m1_common_sample_ma_teacher_v2.parquet": SAMPLE,
    "firm_chars_clean.csv": FIRM_CHARS,
    "monthly_pricing_panel_ma_teacher.csv": PRICING_PANEL,
    "monthly_returns_clean_v2.csv": RETURNS_V2,
}

SIX_SPEC_IDS = ["a_registered_rev_turnover", "b_rev_lag1_turnover_last",
                "c_rev_removed_turnover_last", "d_rev_lag1_no_turnover",
                "e_rev_removed_no_turnover", "f_no_rev_no_turnover_reference"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("p2_5_rev_shadow", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def sha_before():
    return {name: sha256_file(p) for name, p in HEADLINE_FILES.items()}


@pytest.fixture(scope="module")
def frame(mod):
    fm, diag = mod.build_control_frame()
    return fm, diag


@pytest.fixture(scope="module")
def matrix(mod, frame):
    fm, _ = frame
    return mod.run_spec_matrix(fm)


# ---------------------------------------------------------------- 1. 合成 lag
def test_shift_month_series_cross_year(mod):
    s = pd.Series(["2019-12", "2020-01", "2020-02"])
    assert list(mod.shift_month_series(s, +1)) == ["2020-01", "2020-02", "2020-03"]
    assert list(mod.shift_month_series(s, -1)) == ["2019-11", "2019-12", "2020-01"]


def test_attach_lagged_column_synthetic(mod):
    panel = pd.DataFrame({"focal_id": ["a", "a", "a", "b", "b"],
                          "month": ["2020-01", "2020-02", "2020-03", "2020-01", "2020-02"]})
    source = pd.DataFrame({"focal_id": ["a", "a", "a", "b"],
                           "month": ["2019-12", "2020-01", "2020-02", "2020-01"],
                           "v": [0.5, 0.1, -0.2, 0.7]})
    out = mod.attach_lagged_column(panel, source, "v", "lag1", value_key="focal_id")
    got = out.set_index(["focal_id", "month"])["lag1"].to_dict()
    assert got[("a", "2020-01")] == 0.5     # 跨年：2019-12
    assert got[("a", "2020-02")] == 0.1     # t-1
    assert got[("a", "2020-03")] == -0.2
    assert got[("b", "2020-01")] is None or np.isnan(got[("b", "2020-01")])   # 无 2019-12 -> NaN
    assert got[("b", "2020-02")] == 0.7
    assert len(out) == len(panel)
    # 不得填 0：缺失必须保持 NaN
    assert bool(pd.isna(out.loc[(out["focal_id"] == "b") & (out["month"] == "2020-01"), "lag1"]).all())


def test_attach_lagged_column_rejects_duplicate_source_key(mod):
    panel = pd.DataFrame({"focal_id": ["a"], "month": ["2020-02"]})
    source = pd.DataFrame({"focal_id": ["a", "a"], "month": ["2020-01", "2020-01"], "v": [1.0, 2.0]})
    with pytest.raises(SystemExit):
        mod.attach_lagged_column(panel, source, "v", "lag1", value_key="focal_id")


def test_forward_diagnostic_is_future_and_labeled(mod):
    panel = pd.DataFrame({"focal_id": ["a", "a"], "month": ["2020-01", "2020-02"]})
    source = pd.DataFrame({"focal_id": ["a", "a", "a"],
                           "month": ["2019-12", "2020-01", "2020-02"],
                           "v": [1.0, 2.0, 3.0]})
    out = mod.attach_forward_diagnostic(panel, source, "v", "fwd", value_key="focal_id")
    assert out.loc[out["month"] == "2020-01", "fwd"].iloc[0] == 3.0   # t+1：2020-02
    assert pd.isna(out.loc[out["month"] == "2020-02", "fwd"].iloc[0])  # 2020-03 不存在（不填 0）


# ---------------------------------------------------------------- 2. 前视硬门
def test_lookahead_column_never_in_specs(mod):
    for sid, sd in mod.SPEC_DEFS.items():
        x = mod.spec_xcols(sd)
        assert mod.REV_LAG1_FWD not in x, sid
        assert sd["rev"] != mod.REV_LAG1_FWD, sid
    mat = pd.read_csv(MATRIX_CSV)
    assert not any(str(s).startswith(("q_", "r_")) for s in mat["spec_id"].unique())
    assert mod.REV_LAG1_FWD not in set(mat["term"].unique())


# ---------------------------------------------------------------- 3. 真实样本构造
def test_rev_same_bridge_and_lag_coverage(frame, mod):
    fm, diag = frame
    bridge = diag["rev_same_panel_bridge"]
    assert bridge["max_abs_diff_vs_sample_rev"] <= 1e-12
    assert bridge["rows_differing_gt_1e-12"] == 0
    assert diag["rev_lag1"]["fm_lag_coverage"] == 1.0
    assert diag["rev_lag1"]["missing_filled_with_zero"] == 0
    assert diag["rev_lag1"]["panel_rows"] == 307_503
    assert diag["fm_rows"] == 181_178 and diag["fm_months"] == 104
    assert fm["month"].min() == "2015-05" and fm["month"].max() == "2023-12"


def test_rev_lag1_equals_prior_month_return(frame):
    fm, _ = frame
    ret = pd.read_csv(RETURNS_V2, usecols=["firm_id", "month", "R"],
                      dtype={"firm_id": str, "month": str})
    r = ret.rename(columns={"firm_id": "focal_id"})
    r["month"] = (pd.PeriodIndex(r["month"], freq="M") + 1).strftime("%Y-%m")  # 源月 t-1 -> 键 t
    chk = fm[["focal_id", "month", "rev_lag1"]].merge(r, on=["focal_id", "month"], how="left")
    assert len(chk) == len(fm)
    d = (chk["rev_lag1"] - chk["R"]).abs()
    assert float(d.max()) <= 1e-15
    # 独立抽 20 行逐行核对：rev_lag1 == v2.R(t-1) 且 != R(t)
    sample = fm.sample(20, random_state=20261026)
    for _, row in sample.iterrows():
        prev = (pd.Period(row["month"], freq="M") - 1).strftime("%Y-%m")
        v_prev = ret[(ret["firm_id"] == row["focal_id"]) & (ret["month"] == prev)]["R"]
        assert len(v_prev) == 1
        assert abs(float(v_prev.iloc[0]) - float(row["rev_lag1"])) <= 1e-15


# ---------------------------------------------------------------- 4. 相关
def test_correlations_against_plan_and_audit(frame, mod):
    fm, _ = frame
    c_rev_own = mod.monthly_corr_stats(fm, "rev", "OwnMAXraw")
    c_lag_own = mod.monthly_corr_stats(fm, "rev_lag1", "OwnMAXraw")
    c_rev_lag = mod.monthly_corr_stats(fm, "rev", "rev_lag1")
    c_fwd_own = mod.monthly_corr_stats(fm, "rev_lag1_fwd", "OwnMAXraw")
    # 审计 L-3：corr(rev, OwnMAX) 月度均值 0.42464 / 中位 0.44162
    assert abs(c_rev_own["monthly_mean"] - 0.424644) < 2e-5
    assert abs(c_rev_own["monthly_median"] - 0.441620) < 2e-5
    assert c_rev_own["n_months"] == 104
    # 合规 lag1：+0.1192（plan 引用的 -0.0343 是前视列，见下）
    assert abs(c_lag_own["monthly_mean"] - 0.119207) < 5e-4
    assert c_lag_own["monthly_mean"] > 0
    # corr(rev, rev_lag1)
    assert abs(c_rev_lag["monthly_mean"] + 0.035272) < 5e-4
    # 前视列复现 plan 引用值（provenance）
    assert abs(c_fwd_own["monthly_mean"] + 0.034282) < 2e-5


# ---------------------------------------------------------------- 5. 六规格矩阵
def test_six_specs_run_full_matrix(matrix):
    for sid in SIX_SPEC_IDS:
        blk = matrix["specs"][sid]
        for w in ("ew", "vw"):
            assert blk[w]["T"] == 104, sid
            assert blk[w]["n_rows_used"] == 181_178, sid
    st = set(matrix["specs"])
    assert {"g_rev_same_no_turnover", "h_rev_same_turnover_mean_win",
            "i_rev_lag1_turnover_mean_win", "j_rev_lag1_turnover_mean_nowin",
            "k_rev_removed_turnover_mean_win", "l_rev_lag1_win_turnover_last",
            "m_rev_lag1_win_no_turnover"} <= st
    assert len(st) == 13


def test_spec_a_reproduces_headline(mod, matrix):
    cmp = mod.headline_comparison(matrix)
    assert cmp["all_match"] is True
    e = matrix["specs"]["a_registered_rev_turnover"]["ew"]["coefs"]
    assert abs(e["PeerMAXraw"]["beta"] - 0.03427281) < 5e-5
    assert abs(e["OwnMAXraw"]["beta"] + 0.00281881) < 5e-5
    assert abs(e["turnover"]["beta"] + 0.39870074) < 5e-5


def test_compliant_specs_b_d_numbers(matrix):
    e_b = matrix["specs"]["b_rev_lag1_turnover_last"]["ew"]["coefs"]
    assert abs(e_b["OwnMAXraw"]["beta"] + 0.015120) < 5e-5
    assert abs(e_b["OwnMAXraw"]["t_nw"] + 0.704657) < 0.02
    assert abs(e_b["PeerMAXraw"]["beta"] - 0.043908) < 5e-5
    assert abs(e_b["PeerMAXraw"]["t_nw"] - 0.590176) < 0.02
    e_d = matrix["specs"]["d_rev_lag1_no_turnover"]["ew"]["coefs"]
    assert abs(e_d["OwnMAXraw"]["beta"] + 0.133532) < 5e-5
    assert abs(e_d["OwnMAXraw"]["t_nw"] + 4.906308) < 0.02
    assert abs(e_d["rev_lag1"]["beta"] + 0.015695) < 5e-5


def test_audit_anchor_specs_c_e_f_g(matrix):
    e_c = matrix["specs"]["c_rev_removed_turnover_last"]["ew"]["coefs"]
    assert abs(e_c["OwnMAXraw"]["beta"] + 0.011331) < 5e-5
    e_e = matrix["specs"]["e_rev_removed_no_turnover"]["ew"]["coefs"]
    assert abs(e_e["OwnMAXraw"]["beta"] + 0.13585820) < 5e-5
    assert abs(e_e["OwnMAXraw"]["t_nw"] + 4.96987225) < 0.005
    e_g = matrix["specs"]["g_rev_same_no_turnover"]["ew"]["coefs"]
    assert abs(e_g["OwnMAXraw"]["beta"] + 0.10550381) < 5e-5
    assert abs(e_g["OwnMAXraw"]["t_nw"] + 3.80093741) < 0.005


def test_spec_e_and_f_identical(matrix):
    for w in ("ew", "vw"):
        for t in ("PeerMAXraw", "OwnMAXraw"):
            a = matrix["specs"]["e_rev_removed_no_turnover"][w]["coefs"][t]["beta"]
            b = matrix["specs"]["f_no_rev_no_turnover_reference"][w]["coefs"][t]["beta"]
            assert abs(a - b) <= 1e-12
    assert (matrix["specs"]["e_rev_removed_no_turnover"]["x"]
            == matrix["specs"]["f_no_rev_no_turnover_reference"]["x"])


def test_all_six_specs_peer_max_insignificant(matrix):
    for sid in SIX_SPEC_IDS:
        for w in ("ew", "vw"):
            t = matrix["specs"][sid][w]["coefs"]["PeerMAXraw"]["t_nw"]
            assert abs(t) < 1.96, (sid, w, t)


# ---------------------------------------------------------------- 6. 前视诊断
def test_plan_quote_reproduced_only_by_forward_variant(mod, frame):
    fm, _ = frame
    fwd = mod.run_spec_matrix(fm, mod.FORWARD_DIAG_SPEC_DEFS)
    q = fwd["specs"]["q_plan_quote_fwd_turnover_last"]["ew"]["coefs"]
    r = fwd["specs"]["r_plan_quote_fwd_turnover_mean_nowin"]["ew"]["coefs"]
    # plan §2.5 引用的一行：OwnMAX -0.01341（t=-4.02）、PeerMAX +0.00927（t=1.21）
    assert abs(q["OwnMAXraw"]["beta"] + 0.013414) < 5e-4
    assert abs(q["OwnMAXraw"]["t_nw"] + 4.02) < 0.02
    assert abs(q["PeerMAXraw"]["beta"] - 0.00927) < 5e-4
    # 另一行：-0.01728（t=-4.91）、+0.01121（t=1.49）
    assert abs(r["OwnMAXraw"]["beta"] + 0.01728) < 5e-4
    assert abs(r["OwnMAXraw"]["t_nw"] + 4.91) < 0.02
    assert abs(r["PeerMAXraw"]["beta"] - 0.01121) < 5e-4
    # 该前视列自身系数 ≈1（= outcome R_next），证明 plan 的 rev_lag1 是被解释变量
    assert q["rev_lag1_fwd"]["beta"] > 0.5
    assert abs(q["rev_lag1_fwd"]["t_nw"]) > 20


def test_counts_and_matrix_outputs_contract(matrix):
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    assert counts["p2shadow_version"] == "p2shadow_v1"
    assert counts["outputs"]["matrix_csv"]["sha256"] == sha256_file(MATRIX_CSV)
    assert counts["headline_anchors_match"] is True
    assert counts["headline_files_modified"] is False
    assert counts["m1_rerun"] is False and counts["configs_modified"] is False
    assert counts["compliant_anchor_replication"]["all_items_match"] is True
    fwd = counts["plan_rev_lag1_lookahead_diagnostic"]
    assert fwd["all_items_match"] is True
    assert fwd["plan_compliant_reproduction_possible"] is False
    # counts 与测试内重算的矩阵逐值一致
    for sid, blk in matrix["specs"].items():
        c = counts["spec_matrix"]["specs"][sid]
        assert c["ew"]["T"] == blk["ew"]["T"]
        for term in ("PeerMAXraw", "OwnMAXraw"):
            assert abs(c["ew"]["coefs"][term]["beta"]
                       - blk["ew"]["coefs"][term]["beta"]) < 1e-12
    mat = pd.read_csv(MATRIX_CSV)
    assert set(mat["spec_id"].unique()) == set(counts["spec_matrix"]["specs"])
    assert set(mat["weight"].unique()) == {"ew", "vw"}
    assert bool((mat["T_months"] == 104).all())
    assert bool((mat["n_rows"] == 181_178).all())
    for sid in SIX_SPEC_IDS:
        for w in ("ew", "vw"):
            row = mat[(mat["spec_id"] == sid) & (mat["weight"] == w) & (mat["term"] == "OwnMAXraw")]
            assert len(row) == 1
            assert abs(float(row["beta"].iloc[0])
                       - matrix["specs"][sid][w]["coefs"]["OwnMAXraw"]["beta"]) < 5e-9


# ---------------------------------------------------------------- 7. headline 未变
def test_headline_shas_unchanged(sha_before):
    for name, p in HEADLINE_FILES.items():
        assert sha256_file(p) == sha_before[name], name


def test_headline_shas_equal_registered_anchors():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    anchors = {}
    for e in manifest["deliverables"]:
        f = str(e.get("file", "")).replace("\\", "/")
        sha = str(e.get("sha256") or "")
        if sha:
            anchors[f] = sha
    assert sha256_file(FIRM_CHARS) == anchors["data/processed/firm_chars_clean.csv"]
    assert sha256_file(PRICING_PANEL) == anchors["data/processed/monthly_pricing_panel_ma_teacher.csv"]
    assert sha256_file(RETURNS_V2) == anchors["data/processed/monthly_returns_clean_v2.csv"]
    report = json.loads(SAMPLE_REPORT_V2.read_text(encoding="utf-8"))
    assert sha256_file(SAMPLE) == report["files"]["v2_samples"]["ma_teacher"]["sha256"]


# ---------------------------------------------------------------- 8. 路径治理
def test_output_path_guard(mod):
    mod.assert_output_path_guard(mod.MATRIX_CSV)
    mod.assert_output_path_guard(mod.COUNTS_JSON)
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(SAMPLE)
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(FIRM_CHARS)
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(RETURNS_V2)
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(ROOT / "reports/data_gates/G0/anything.json")
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(ROOT / "configs/parameter_registry.yaml")
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(ROOT / "data/processed/phase2_shadow/x.csv")
