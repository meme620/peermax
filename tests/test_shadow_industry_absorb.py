"""Phase 3.5 N-5/A12 行业吸收残差化 shadow 契约测试（加法，只读真实产物）。

覆盖（3.0_plan §2.5 R1/R2/R3 + 3.5 任务书）：
  1. 合成 industry/PeerMAX：month FE 残差化正确（斜率、正交性、R² 定义）；
  2. 真实样本 corr / 第一阶段 R² 复算（E7：0.7917/0.7769、0.6483）；
  3. A12 规格可跑且 headline 复现（β_without/+0.03427、L1/L2 β/γ/δ、E9 残差化 β）；
  4. headline 样本 / firm_chars / panel / G0 A12 报告 sha 前后未变；9 锚 MATCH；
  5. 残差化 parquet 合同（行数、键唯一、残差与行业列正交、版本串）；
  6. 输出路径守卫（只允许 shadow/artifacts）。

运行：python -m pytest tests/test_shadow_industry_absorb.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

SCRIPT = ROOT / "scripts/shadow_industry_absorb.py"
ART = ROOT / "artifacts/phase3"
COUNTS = ART / "3.5_counts.json"
ABSORB = ART / "3.5_industry_absorb.csv"
COLLIN = ART / "3.5_collinearity.csv"
CROSS = ART / "3.5_cluster_industry_cross.csv"
ANCHOR_PRE = ART / "3.5_anchor_pre.json"
ANCHOR_POST = ART / "3.5_anchor_post.json"
RESID = ROOT / "data/processed/phase3_shadow/m1_peer_resid_p3shadow.parquet"
SAMPLE = ROOT / "data/processed/m1_common_sample_ma_teacher_v2.parquet"
SAMPLE_ASOF = ROOT / "data/processed/phase3_shadow/m1_common_sample_ma_teacher_p3asof_baselines.parquet"
ASOF_BASELINES = ROOT / "data/processed/phase3_shadow/ind_baselines_p3asof.parquet"
HEADLINE_PANEL = ROOT / "data/processed/monthly_pricing_panel_ma_teacher.csv"
FIRM_CHARS = ROOT / "data/processed/firm_chars_clean.csv"
G0_A12 = ROOT / "reports/data_gates/G0/s3_m1_industry_absorb.json"

E7_CORR_L1 = 0.7916556878781287
E7_CORR_L2 = 0.7769299176515353
E7_R2_L2 = 0.6483374839833427
HEADLINE_BETA_WITHOUT = 0.03427280848385429
RESID_E9_BETA_LATEST = -0.03905433621818112


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Phase 3.5 shadow 的输入样本固定为 v2（历史只读）；仅此文件允许 manifest
# status='superseded'，且必须给出理由。不得全局放宽为 active/superseded 皆可。
_SUPERSEDED_SCOPED_ALLOW = {
    "data/processed/m1_common_sample_ma_teacher_v2.parquet":
        "Phase 3.5 industry-absorb shadow 的输入样本是 v2（只读历史）；"
        "v3 已为 headline，但重建 v3 shadow 不在本测试范围",
}


def _manifest_anchor(rel: str) -> str:
    """当前 manifest 条目 sha256；默认只接受 active，superseded 仅限 scoped 例外。"""
    import yaml
    man = yaml.safe_load((ROOT / "configs/version_manifest.yaml").read_text(encoding="utf-8"))
    hits = [e for e in man["deliverables"] if str(e.get("file", "")).replace("\\", "/") == rel]
    assert len(hits) == 1, f"manifest 条目数 != 1: {rel}"
    status = hits[0].get("status")
    if status == "superseded":
        assert rel in _SUPERSEDED_SCOPED_ALLOW, (
            f"manifest 条目 superseded 但不在 scoped 例外: {rel}")
    else:
        assert status == "active", f"manifest 条目非 active: {rel}（status={status!r}）"
    return str(hits[0]["sha256"])


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_industry_absorb", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def p35():
    assert SCRIPT.is_file(), f"缺少脚本 {SCRIPT}"
    return load_module()


@pytest.fixture(scope="module")
def counts() -> dict:
    assert COUNTS.is_file(), f"缺少 3.5 产物（先运行 {SCRIPT.name}）：{COUNTS}"
    return json.loads(COUNTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def absorb() -> pd.DataFrame:
    assert ABSORB.is_file(), f"缺少 {ABSORB}"
    return pd.read_csv(ABSORB)


@pytest.fixture(scope="module")
def collin() -> pd.DataFrame:
    assert COLLIN.is_file(), f"缺少 {COLLIN}"
    return pd.read_csv(COLLIN)


@pytest.fixture(scope="module")
def resid() -> pd.DataFrame:
    assert RESID.is_file(), f"缺少 {RESID}"
    return pd.read_parquet(RESID)


# ---------------------------------------------------------------- 1. 合成残差化
def test_synthetic_month_fe_residualization(p35):
    rng = np.random.default_rng(20261029)
    rows = []
    for m in range(24):
        month = f"{2015 + m // 12}-{m % 12 + 1:02d}"
        x = rng.normal(0.02 * m, 1.0, 160)
        y = 0.60 * x + 0.01 * m + rng.normal(0.0, 0.5, 160)
        rows.append(pd.DataFrame({"month": month, "PeerMAXraw": y, "ind_syn": x}))
    df = pd.concat(rows, ignore_index=True)
    fs = p35.within_month_residual(df, "PeerMAXraw", ["ind_syn"])
    assert fs["n_rows"] == len(df)
    assert fs["n_months"] == 24
    assert abs(fs["coef"]["ind_syn"] - 0.60) < 0.02
    assert fs["max_abs_corr_resid_x"] < 1e-10
    assert fs["max_abs_month_mean_resid"] < 1e-10
    # R² 定义：pooled month FE = 1 - var(resid)/var(y)，且 ≥ 双变量 r²（month FE 吸收组间均值）
    r2 = 1.0 - float(np.var(fs["resid"].to_numpy()) / np.var(df["PeerMAXraw"].to_numpy()))
    assert abs(fs["r2_pooled_month_fe"] - r2) < 1e-12
    assert fs["r2_pooled_month_fe"] >= fs["r2_bivariate"] - 1e-12
    assert fs["r2_bivariate"] > 0.5


def test_synthetic_orthogonality_multivariate(p35):
    rng = np.random.default_rng(7)
    rows = []
    for m in range(12):
        x1 = rng.normal(size=120)
        x2 = 0.4 * x1 + rng.normal(size=120)
        y = 0.3 * x1 - 0.2 * x2 + rng.normal(scale=0.4, size=120)
        rows.append(pd.DataFrame({"month": f"2016-{m + 1:02d}",
                                  "PeerMAXraw": y, "ind1": x1, "ind2": x2}))
    df = pd.concat(rows, ignore_index=True)
    fs = p35.within_month_residual(df, "PeerMAXraw", ["ind1", "ind2"])
    assert fs["max_abs_corr_resid_x"] < 1e-10
    assert abs(fs["coef"]["ind1"] - 0.3) < 0.02
    assert abs(fs["coef"]["ind2"] + 0.2) < 0.02
    assert fs["coef"]["ind1"] is not None and fs["coef"]["ind2"] is not None


# ---------------------------------------------------------------- 2. 真实 corr / R²
def test_real_corr_recompute(p35):
    d = pd.read_parquet(SAMPLE)
    fm = d[d["m1_in_fm_sample"].astype(bool)]
    for col, expected in (("ind_l1_max_ew", E7_CORR_L1), ("ind_l2_max_ew", E7_CORR_L2)):
        s = fm.dropna(subset=["PeerMAXraw", col])
        c = float(np.corrcoef(s["PeerMAXraw"], s[col])[0, 1])
        assert abs(c - expected) < 1e-9, (col, c, expected)


def test_real_first_stage_r2_recompute(p35):
    d = pd.read_parquet(SAMPLE)
    fm = d[d["m1_in_fm_sample"].astype(bool)]
    fs = p35.within_month_residual(fm, "PeerMAXraw", ["ind_l2_max_ew"])
    assert fs["n_rows"] == 179_193
    assert fs["n_months"] == 104
    assert abs(fs["r2_pooled_month_fe"] - E7_R2_L2) < 1e-9
    assert abs(fs["corr_pearson"] - E7_CORR_L2) < 1e-9
    # 与独立 lstsq（month dummies + ind + 常数）复算一致
    sub = fm.dropna(subset=["PeerMAXraw", "ind_l2_max_ew"])
    X = pd.get_dummies(sub["month"], drop_first=True).astype(float)
    X["ind"] = sub["ind_l2_max_ew"].to_numpy()
    Xm = np.column_stack([np.ones(len(sub)), X.to_numpy()])
    y = sub["PeerMAXraw"].to_numpy()
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    r = y - Xm @ b
    r2 = 1.0 - float(np.var(r) / np.var(y))
    assert abs(r2 - E7_R2_L2) < 1e-9


def test_collinearity_artifact_matches(counts, collin):
    row = collin[(collin["basis"] == "latest") & (collin["level"] == "l2")].iloc[0]
    assert abs(row["corr_pearson"] - E7_CORR_L2) < 1e-9
    assert abs(row["r2_pooled_month_fe"] - E7_R2_L2) < 1e-9
    assert row["n_rows"] == 179_193
    assert row["n_months"] == 104
    asof = collin[(collin["basis"] == "p3asof") & (collin["level"] == "l2")].iloc[0]
    assert asof["n_rows"] == 181_098
    assert abs(asof["r2_pooled_month_fe"]
               - counts["first_stage"]["p3asof"]["l2"]["r2_pooled_month_fe"]) < 1e-12


# ---------------------------------------------------------------- 3. A12 规格可跑/复现
def test_a12_spec_matrix_present(absorb):
    assert len(absorb) == 40
    req = [
        ("a12_l1_naked", "latest"), ("a12_l2_naked", "latest"),
        ("a12_l1_naked", "p3asof"), ("a12_l2_naked", "p3asof"),
        ("a12_l1_full", "latest"), ("a12_l2_full", "latest"),
        ("a12_l1_full", "p3asof"), ("a12_l2_full", "p3asof"),
        ("resid_l1_univariate_naked", "latest"),
        ("resid_l2_univariate_naked", "latest"),
        ("resid_l1_plus_ind_full", "p3asof"),
        ("resid_l2_plus_ind_full", "p3asof"),
        ("resid_l2_main_full", "latest"),
        ("resid_l2_main_full", "p3asof"),
    ]
    keys = set(zip(absorb["spec_id"], absorb["basis"]))
    for k in req:
        assert k in keys, f"缺少规格 {k}"
    full = absorb[absorb["family"] == "full"]
    assert full["x_vars"].str.contains("OwnMAXraw").all()
    assert full["x_vars"].str.contains("PeerReturn").all()
    assert full["x_vars"].str.contains("turnover").all()
    assert full["ownmax_beta"].notna().all()
    assert full["ownmax_t"].notna().all()
    assert full["peerreturn_beta"].notna().all()
    naked = absorb[absorb["family"] == "naked"]
    assert naked["ownmax_beta"].isna().all()
    # 全部规格 104 月且无 rank 失败
    assert (absorb["n_months_used"] == 104).all()
    assert absorb["all_months_full_rank"].all()
    assert absorb["beta"].notna().all()


def test_a12_headline_reproduction(absorb):
    g0 = json.loads(G0_A12.read_text(encoding="utf-8"))
    for sid, gid in (("a12_l1_full", "m1_industry_absorb_l1"),
                     ("a12_l2_full", "m1_industry_absorb_l2"),
                     ("a12_both_full", "m1_industry_absorb_both")):
        r = absorb[(absorb["spec_id"] == sid) & (absorb["basis"] == "latest")].iloc[0]
        g = g0["specs"][gid]
        assert abs(r["beta"] - g["beta_with"]["beta"]) < 1e-9
        assert abs(r["t_nw"] - g["beta_with"]["t_nw"]) < 1e-9
        assert abs(r["delta_vs_baseline_fullsample"] - g["retention_delta"]) < 1e-9
        for c, gv in g["gamma"].items():
            col = "gamma_1_beta" if r["gamma_1_var"] == c else "gamma_2_beta"
            assert abs(r[col] - gv["beta"]) < 1e-9
    main = absorb[(absorb["spec_id"] == "main_full") & (absorb["basis"] == "latest")].iloc[0]
    assert abs(main["beta"] - HEADLINE_BETA_WITHOUT) < 1e-9
    assert abs(main["t_nw"] - 0.4728087909862261) < 1e-9


def test_residual_main_matches_e9(absorb):
    r = absorb[(absorb["spec_id"] == "resid_l2_main_full")
               & (absorb["basis"] == "latest")].iloc[0]
    assert abs(r["beta"] - RESID_E9_BETA_LATEST) < 1e-6
    assert r["n_rows"] == 179_193
    assert r["n_months_used"] == 104
    asof = absorb[(absorb["spec_id"] == "resid_l2_main_full")
                  & (absorb["basis"] == "p3asof")].iloc[0]
    assert asof["n_rows"] == 181_098
    assert abs(asof["beta"]) < 0.02  # as-of 主口径残差化后接近 0


def test_incremental_r2_is_small(absorb):
    r = absorb[(absorb["spec_id"] == "a12_l2_full")
               & (absorb["basis"] == "latest")].iloc[0]
    assert abs(r["incremental_r2_mean"] - 0.002832602455) < 1e-9
    assert abs(r["incremental_r2_median"] - 0.001217732141) < 1e-9
    assert 0.0 <= r["incremental_r2_mean"] < 0.01
    asof = absorb[(absorb["spec_id"] == "a12_l2_full")
                  & (absorb["basis"] == "p3asof")].iloc[0]
    assert 0.0 <= asof["incremental_r2_mean"] < 0.01
    main = absorb[(absorb["spec_id"] == "main_full")
                  & (absorb["basis"] == "latest")].iloc[0]
    assert abs(main["incremental_r2_mean"]) < 1e-12


def test_fm_regression_runs_on_synthetic_a12():
    from pricing_lib import fm_regression
    rng = np.random.default_rng(11)
    rows = []
    for m in range(18):
        n = 140
        ind = rng.normal(size=n)
        peermax = 0.6 * ind + rng.normal(size=n)
        rows.append(pd.DataFrame({
            "month": f"{2016 + m // 12}-{m % 12 + 1:02d}", "R_next": rng.normal(size=n),
            "PeerMAXraw": peermax, "ind_l2_max_ew": ind,
            "OwnMAXraw": rng.normal(size=n), "PeerReturn": rng.normal(size=n),
            **{c: rng.normal(size=n) for c in
               ("size", "bm", "mom", "rev", "turnover", "illiq",
                "AnalystCoverage", "NetworkDegree")}}))
    df = pd.concat(rows, ignore_index=True)
    x = ["PeerMAXraw", "ind_l2_max_ew", "OwnMAXraw", "PeerReturn",
         "size", "bm", "mom", "rev", "turnover", "illiq",
         "AnalystCoverage", "NetworkDegree"]
    res, per_month = fm_regression(df, "R_next", x, min_n=50, nw_lag=2)
    assert res["n_months_used"] == 18
    assert res["coefs"]["PeerMAXraw"]["beta"] is not None
    assert len([d for d in per_month if d.get("used")]) == 18


# ---------------------------------------------------------------- 4. headline 未变
def test_anchors_pre_post_match(counts):
    a = counts["anchors"]
    assert a["compare"]["status"] == "MATCH"
    assert a["compare"]["n_diff"] == 0
    assert len(a["pre"]["m1_inputs_9"]) == 9
    assert a["pre"]["m1_inputs_9"] == a["post"]["m1_inputs_9"]
    assert a["pre"]["extra_inputs"] == a["post"]["extra_inputs"]


def test_headline_sample_firmchars_panel_sha_unchanged(counts):
    pre = counts["anchors"]["pre"]["extra_inputs"]
    post = counts["anchors"]["post"]["extra_inputs"]
    for key, path in (("headline_panel_csv", HEADLINE_PANEL),
                      ("firm_chars_clean", FIRM_CHARS),
                      ("g0_a12_report", G0_A12),
                      ("asof_baselines_p3asof", ASOF_BASELINES),
                      ("asof_sample_copy", SAMPLE_ASOF)):
        assert pre[key]["sha256"] == sha256_file(path), f"{key} sha 变化"
        assert pre[key]["sha256"] == post[key]["sha256"]
    # BL-01（I-16）metadata 重写已改现盘 headline 样本 sha；`3.5_counts.json` 是 Phase 3 历史证据
    # （不得重写），故现盘比对改到当前 manifest 锚；pre==post 历史不变式保留。
    assert pre["headline_sample_parquet"]["sha256"] == post["headline_sample_parquet"]["sha256"]
    assert sha256_file(SAMPLE) == _manifest_anchor(
        "data/processed/m1_common_sample_ma_teacher_v2.parquet")


def test_m1_inputs_sha_still_match(counts):
    import yaml
    from pricing_lib import assert_m1_inputs_sha
    ledger = assert_m1_inputs_sha()
    actual = {r["input"]: r["sha256"] for r in ledger}
    assert len(actual) == 9
    pre = counts["anchors"]["pre"]["m1_inputs_9"]
    assert set(pre) == set(actual)
    # BL-01（I-17）允许 clean_v2 变化；W1 bundle（P1.1-01）允许 source_month 变化；9 项都必须等于当前 m1_contracts 锚
    assert {k for k in actual if actual[k] != pre[k]} <= {"monthly_returns_clean_v2", "source_month_argmax_panel"}
    contracts = yaml.safe_load((ROOT / "configs/m1_contracts.yaml").read_text(encoding="utf-8"))
    for key, sha in actual.items():
        assert sha == contracts["m1_inputs"][key]["sha256"], f"{key}: m1_contracts 锚不一致"


def test_headline_sample_counts_unchanged(counts):
    assert counts["inputs"]["sample_rows"] == 307_503
    assert counts["inputs"]["fm_rows"] == 181_178
    assert counts["inputs"]["fm_months"] == 104


# ---------------------------------------------------------------- 5. 残差化 parquet 合同
def test_resid_parquet_contract(resid, counts):
    assert len(resid) == 181_178
    assert resid.duplicated(["focal_id", "month"]).sum() == 0
    assert resid["resid_version"].nunique() == 1
    assert resid["resid_version"].iloc[0] == counts["shadow_version_tag"]
    for basis, ind_cols in (("latest", {"l1": "ind_l1_max_ew", "l2": "ind_l2_max_ew"}),
                            ("p3asof", {"l1": "ind_l1_max_ew_p3asof",
                                        "l2": "ind_l2_max_ew_p3asof"})):
        for lvl, ind in ind_cols.items():
            col = f"PeerMAXraw_p3resid_{lvl}_{basis}"
            assert col in resid.columns
            assert resid[col].notna().sum() > 179_000
            base = pd.read_parquet(SAMPLE_ASOF)[["focal_id", "month", ind]]
            m = resid[["focal_id", "month", col]].merge(base, on=["focal_id", "month"],
                                                        how="left")
            ok = m[col].notna() & m[ind].notna()
            assert ok.sum() > 179_000
            c = float(np.corrcoef(m.loc[ok, col], m.loc[ok, ind])[0, 1])
            assert abs(c) < 1e-10, (col, c)
            mm = m.loc[ok].groupby("month")[col].mean().abs().max()
            assert mm < 1e-10
    for basis in ("latest", "p3asof"):
        assert f"PeerMAXraw_p3resid_l2_{basis}_permonth" in resid.columns


# ---------------------------------------------------------------- 6. 路径守卫
def test_output_path_guard(p35):
    assert p35.assert_output_path_guard(ART / "3.5_counts.json")
    assert p35.assert_output_path_guard(
        ROOT / "data/processed/phase3_shadow/m1_peer_resid_p3shadow.parquet")
    for bad in (SAMPLE, HEADLINE_PANEL, FIRM_CHARS, G0_A12,
                ROOT / "data/processed/network_ma_teacher/2019_similarity.csv",
                ROOT / "configs/parameter_registry.yaml"):
        with pytest.raises(SystemExit) as exc:
            p35.assert_output_path_guard(bad)
        assert exc.value.code == 2


def test_outputs_are_shadow_only(counts):
    allowed = (ROOT / "artifacts/phase3", ROOT / "data/processed/phase3_shadow")
    for name in counts["outputs"]:
        p = ART / name if name.startswith("3.5_") else (
            ROOT / "data/processed/phase3_shadow" / name)
        assert any(str(p.resolve()).startswith(str(a.resolve())) for a in allowed), p
