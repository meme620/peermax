"""Phase 2.6 shadow M1 对比契约测试（加法；合成 + 真实产物只读）。

覆盖：
  1. 变体构造：`variant_columns`/`canonical_frame`/`filter_rows` 的 A/B/C 映射与行集语义；
     `spec_frame` 的 x 列表（S1/S2/S3/S4/S5/S6 的 rev/turnover 开关）；`run_variant` 可运行出系数表。
  2. 样本损失：合成 `add_screen_flags` 语义 + 真实数据复算 3,547 / 3,352 / 3,774 / 1,046 / 5,853 / 175,325 锚。
  3. 传播与 winsor：`aggregate_over_edges` 覆盖源重归一化且缺失不填 0；`winsorize_col` 月度边界；
     `mom_index_rule` 复刻 `build_firm_chars` 的 ys[-12:-1] 规则（缺失不填 0）。
  4. 输出路径守卫：拒绝 data/、reports/、configs/、scripts/、headline 文件；接受 artifacts/phase2/2.6_*。
  5. 产物契约：2.6_comparison.csv 的 sha/行数/列与 counts 一致；headline_G0_table_v2 行数值等于定价表 v2。
  6. headline sha 未变：样本 parquet / 定价表 / firm_chars / source 面板 / v2 月收益 sha 等于登记锚；
     counts 的 m1_anchors_match / headline_unchanged / shadow_unchanged 全 True。
  7. 校准与推广判据：headline_recompute / headline_rebuilt 对 G0 的硬门 PASS、2.4/2.5 交叉锚 ≤1e-6；
     P1–P8 状态记录且无 FAIL（P6/P8 为待项目动作）。

运行：python -m pytest tests/test_shadow_pricing_compare.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/shadow_pricing_compare.py"
COUNTS = ROOT / "artifacts/phase2/2.6_counts.json"
CSV = ROOT / "artifacts/phase2/2.6_comparison.csv"
MEMO = ROOT / "artifacts/phase2/2.6_decision_memo.md"

SAMPLE_MA = ROOT / "data/processed/m1_common_sample_ma_teacher_v2.parquet"
PRICING_TABLE_V2 = ROOT / "reports/data_gates/G0/s3_m1_pricing_table_v2.json"
FIRM_CHARS = ROOT / "data/processed/firm_chars_clean.csv"
SOURCE_PANEL = ROOT / "data/processed/source_month_argmax_panel.csv"
RETURNS_V2 = ROOT / "data/processed/monthly_returns_clean_v2.csv"
SHADOW_MONTHLY = ROOT / "data/processed/phase2_shadow/monthly_returns_status_p2shadow.csv"
SHADOW_MAX = ROOT / "data/processed/phase2_shadow/source_month_argmax_panel_p2shadow.csv"

HEADLINE_ANCHORS = {
    # BL-01（I-16）后 = metadata-only 重写（追加 window_requested / sample_window_actual）的新 sha；
    # 原列逐值不变（metadata-only 改写的验收口径）
    "m1_common_sample_ma_teacher_v2.parquet":
        "5e9a8504b7c7bfff811cecc4c4dc325f9e502108420df245ba38eddea04e96fd",
    "s3_m1_pricing_table_v2.json":
        "d3d6c12b0b2d33bf2471a0b73ee1d6d4b4a7fcf7ad088218e92fd7dc3d97dda3",
    "firm_chars_clean.csv":
        "28f9eedcefd5a1d2759b0068f2ed20b10144d5b29be4f27ad278845093938df8",
    # W1 bundle / P1.1-01（2026-11-17）：source-month 原地重建（complete_month gate + 24 列），
    # 该变化为 07 P1.1-01 授权的唯一原地数据重写；MAX/R/argmax 等旧列语义未变。
    "source_month_argmax_panel.csv":
        "9ef16943104b5556df317da2aa66f0ff1cb782e7ac8ac16e1edbd60b0e1a9b9a",
    # BL-01（I-17）后 = sample_version 由旧标签改 computed 标签（2000-02..2026-06）的 CSV 新 sha
    # P1-01（2026-11-15）后 = n_valid_days/complete_month 改为有效交易日口径的新 sha
    "monthly_returns_clean_v2.csv":
        "29937f599cf5d5fd74acdf2908d043af883455d01dcfd949c23fb06b5218cd30",
}

TRIO = ["PeerMAXraw", "OwnMAXraw", "PeerReturn"]
SHADOW_VARIANTS = ["A_fixed_sample", "A_fixed_sample_adj", "B_rescreened", "C_rescreened_adj"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("shadow_pricing_compare", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def counts():
    assert COUNTS.is_file(), "缺少 artifacts/phase2/2.6_counts.json（先跑 p2_6 脚本）"
    return json.loads(COUNTS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. 变体构造
def test_variant_columns_map_expected_sources(mod):
    a = mod.variant_columns("A_fixed_sample")
    b = mod.variant_columns("B_rescreened")
    c = mod.variant_columns("C_rescreened_adj")
    assert a["OwnMAXraw"] == "own_raw_main_w" and a["PeerMAXraw"] == "peer_raw_main_w"
    assert b["OwnMAXraw"] == "own_raw_g15_w" and b["PeerReturn"] == "pret_raw_g15_w"
    assert c["OwnMAXraw"] == "own_adj_g15_w" and c["PeerReturn"] == "pret_adj_g15_w"
    assert a["R_next"] == "rn_raw" and c["R_next"] == "rn_adj"
    assert a["rev_lag1"] == "rev_lag1_raw" and c["rev_lag1"] == "rev_lag1_adj"
    # headline 重建口径必须取自 shadow 输入重建的 legacy 列
    hr = mod.variant_columns("headline_rebuilt")
    assert hr["OwnMAXraw"] == "own_legacy_w" and hr["PeerMAXraw"] == "peer_legacy_w"
    # 未知变体硬失败
    with pytest.raises(SystemExit):
        mod.variant_columns("nope")


def test_filter_rows_modes(mod):
    df = pd.DataFrame({"m1_in_fm_sample": [True, True, False], "screen_bc": [True, False, False],
                       "v": [1.0, 2.0, 3.0]})
    assert list(mod.filter_rows(df, "fm")["v"]) == [1.0, 2.0]
    assert list(mod.filter_rows(df, "bc")["v"]) == [1.0]
    with pytest.raises(SystemExit):
        mod.filter_rows(df, "xx")


def test_spec_frame_xcols_toggles(mod):
    cols = ["focal_id", "month", "R_next", "market_equity"] + TRIO + [
        "size", "bm", "mom", "rev_same", "rev_lag1", "turnover_last", "turnover_mean",
        "illiq", "AnalystCoverage", "NetworkDegree"]
    cf = pd.DataFrame({c: np.arange(4, dtype=float) for c in cols})
    _, s1 = mod.spec_frame(cf, "S1_full_last")
    _, s2 = mod.spec_frame(cf, "S2_full_mean")
    _, s3 = mod.spec_frame(cf, "S3_lag1_last")
    _, s5 = mod.spec_frame(cf, "S5_no_turnover")
    _, s6 = mod.spec_frame(cf, "S6_no_rev_no_turnover")
    assert s1[:3] == TRIO and "rev_same" in s1 and "turnover_last" in s1
    assert "turnover_mean" in s2 and "turnover_last" not in s2
    assert "rev_lag1" in s3 and "rev_same" not in s3
    assert "turnover_last" not in s5 and "turnover_mean" not in s5 and "rev_same" in s5
    assert "rev_same" not in s6 and "rev_lag1" not in s6 and "turnover_last" not in s6
    for xc in (s1, s2, s3, s5, s6):
        assert "illiq" in xc and "AnalystCoverage" in xc and "NetworkDegree" in xc


def test_run_variant_runs_and_reports_coefs(mod):
    rng = np.random.default_rng(20261029)
    months = ["2020-01", "2020-02", "2020-03"]
    n = 120
    rows = []
    for m in months:
        for i in range(n):
            x1, x2, x3 = rng.normal(size=3)
            rows.append({
                "focal_id": f"sz{i:06d}", "month": m,
                "R_next": 0.01 * x1 + 0.02 * x2 + rng.normal(scale=0.1),
                "market_equity": 1e9 + i, "m1_in_fm_sample": True, "screen_bc": True,
                "PeerMAXraw": x1 * 0.1, "OwnMAXraw": x2 * 0.1, "PeerReturn": x3 * 0.1,
                "rev_same": rng.normal(size=1)[0], "rev_lag1": rng.normal(size=1)[0],
                "turnover_last": abs(rng.normal()) / 10, "turnover_mean": abs(rng.normal()) / 10,
                "mom": rng.normal(), "illiq": abs(rng.normal()),
                "size": 20 + rng.normal(), "bm": 1 + rng.normal() * 0.1,
                "AnalystCoverage": abs(rng.normal()) * 10, "NetworkDegree": abs(rng.normal()) * 100,
            })
    cf = pd.DataFrame(rows)
    matrix = mod.run_variant(cf, min_n=100, nw_lag=2)
    assert set(matrix) == set(mod.SPEC_IDS)
    for spec_id in mod.SPEC_IDS:
        cell = matrix[spec_id]["ew"]
        assert cell["n_months_used"] == 3 and cell["n_rows_used"] == 360
        for term in TRIO:
            assert cell["coefs"][term]["beta"] is not None
            assert cell["coefs"][term]["t_nw"] is not None
        assert matrix[spec_id]["vw"]["n_rows_used"] == 360


def test_add_screen_flags_semantics(mod):
    """event 侧（status/n_trade_days/n_max_days）与 outcome 侧（下月状态/complete）语义。"""
    n = 6
    df = pd.DataFrame({
        "month": ["2020-01"] * n,
        "m1_in_fm_sample": [True] * n,
        "ev_status": ["ok"] * n,
        "ev_ntd": [20, 14, 20, 20, 20, 20],
        "own_nmd": [20, 20, 14, 20, 20, 0],
        "nx_status": ["ok"] * n,
        "nx_cm15": [1, 1, 1, 0.0, 1, 1],
        "own_raw_main": [0.1] * n,
    })
    df.loc[5, "nx_status"] = "no_trade"
    out, cnt = mod.add_screen_flags(df)
    assert list(out["ev_ok"]) == [True, False, False, True, True, False]
    assert list(out["nx_ok"]) == [True, True, True, False, True, False]
    assert list(out["screen_bc"]) == [True, False, False, False, True, False]
    assert cnt["event_side"]["n_trade_days_lt15_rows"] == 1
    assert cnt["event_side"]["n_max_days_lt15_rows"] == 2
    assert cnt["outcome_side"]["union_rows"] == 2
    assert cnt["A_main_gate"]["own_raw_main_null_rows"] == 0
    # A 主口径的 outcome 空 = nx_cm15 != 1（不含 status 维；真实数据里 status 异常 ⊂ complete!=1）
    assert cnt["A_main_gate"]["outcome_null_rows"] == 1
    assert cnt["A_main_gate"]["union_rows"] == 1
    assert cnt["B_rows_explicit_screen"] == 2


def test_mom_index_rule_matches_build_firm_chars(mod):
    """ys[-12:-1] 规则：<8 个月 -> NaN（不填 0）；gap 时窗口后移。"""
    series = pd.DataFrame({
        "focal_id": ["a"] * 10,
        "month": ["2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2019-06",
                  "2019-07", "2019-08", "2019-09", "2019-10"],
        "R": [0.01] * 10,
    })
    keys = pd.DataFrame({"focal_id": ["a", "a", "a", "a"],
                         "month": ["2019-02", "2019-09", "2019-10", "2020-01"]})
    out = mod.mom_index_rule(series, "R", keys).set_index("month")["_mom"]
    assert pd.isna(out["2019-02"])                      # 只有 1 个更早月 -> NaN
    assert out["2019-09"] == pytest.approx((1.01 ** 7) - 1)   # ys[-12:-1] = 2019-01..2019-07
    assert out["2019-10"] == pytest.approx((1.01 ** 8) - 1)
    assert pd.isna(out["2020-01"]) or out["2020-01"] == pytest.approx((1.01 ** 9) - 1)


def test_winsorize_col_monthly_bounds(mod):
    df = pd.DataFrame({"month": ["2020-01"] * 100 + ["2020-02"] * 100,
                       "v": list(np.arange(100, dtype=float)) * 2})
    w = mod.winsorize_col(df, "v", 0.01, 0.99)
    out = df.assign(w=w)
    assert w.isna().sum() == 0
    for _, grp in out.groupby("month"):
        assert grp["w"].max() == pytest.approx(grp["v"].quantile(0.99))
        assert grp["w"].min() == pytest.approx(grp["v"].quantile(0.01))
    # 分位之间不动、之外被截断
    mid = out[(out["v"] > out["v"].quantile(0.05)) & (out["v"] < out["v"].quantile(0.95))]
    assert (mid["w"] == mid["v"]).all()
    assert out["w"].abs().max() <= 99.0


def test_aggregate_over_edges_renormalizes_without_zero_fill(mod):
    edges = pd.DataFrame({"focal_id": ["f"] * 3, "source_id": ["s1", "s2", "s3"],
                          "weight": [0.5, 0.3, 0.2], "similarity": [0.5, 0.3, 0.2]})
    vals = pd.DataFrame({"firm_id": ["s1", "s2"], "month": ["2020-01"] * 2,
                         "v": [1.0, 0.0]})
    out = mod.aggregate_over_edges({2020: edges}, {"2020-01": 2020}, vals, "v", "peer_v")
    row = out.loc[out["focal_id"] == "f"].iloc[0]
    assert row["peer_v"] == pytest.approx(0.5 / 0.8)     # 覆盖源重归一化（s3 缺失不填 0）
    assert row["covered_mass"] == pytest.approx(0.8)
    vals_none = vals.assign(firm_id="zzz")
    out2 = mod.aggregate_over_edges({2020: edges}, {"2020-01": 2020}, vals_none, "v", "peer_v")
    assert out2.empty or bool(pd.isna(out2.iloc[0]["peer_v"]))


# ---------------------------------------------------------------- 2. 样本损失（真实数据）
def test_sample_loss_counts_real_data(mod):
    fm = pd.read_parquet(SAMPLE_MA, columns=["focal_id", "month", "m1_in_fm_sample"])
    fm = fm[fm["m1_in_fm_sample"].astype(bool)][["focal_id", "month"]]
    mon = pd.read_csv(SHADOW_MONTHLY, usecols=["firm_id", "month", "n_trade_days",
                                               "trade_days_status", "complete_month_trade15"],
                      dtype={"firm_id": str, "month": str})
    mp = pd.read_csv(SHADOW_MAX, usecols=["source_id", "month", "n_max_days", "MAXraw_p2shadow"],
                     dtype={"source_id": str, "month": str})
    ev = mon.rename(columns={"firm_id": "focal_id"})
    nx = ev.copy()
    nx["month"] = nx["month"].map(lambda m: str(pd.Period(m, "M") - 1))
    nx = nx.rename(columns={"n_trade_days": "nx_ntd", "trade_days_status": "nx_status",
                            "complete_month_trade15": "nx_cm15"})
    m = (fm.merge(ev, on=["focal_id", "month"], how="left")
           .merge(nx[["focal_id", "month", "nx_ntd", "nx_status", "nx_cm15"]],
                  on=["focal_id", "month"], how="left")
           .merge(mp.rename(columns={"source_id": "focal_id"}), on=["focal_id", "month"], how="left"))
    assert len(m) == 181178
    outcome_loss = int((m["nx_cm15"] != 1).sum())
    ntd_loss = int((m["n_trade_days"] < 15).sum())
    nmd_loss = int((m["n_max_days"] < 15).sum())
    own_null = m["MAXraw_p2shadow"].isna()
    out_null = m["nx_cm15"] != 1
    assert outcome_loss == 3547 and ntd_loss == 3352 and nmd_loss == 3774
    assert int((own_null & out_null).sum()) == 1046
    assert int((own_null | out_null).sum()) == 5853
    assert 181178 - int((own_null | out_null).sum()) == 175325      # plan FM 上限
    screen = (~(m["n_trade_days"].lt(15) | m["n_max_days"].lt(15)
                | m["trade_days_status"].ne("ok")) & (m["nx_cm15"] == 1))
    assert int(screen.sum()) == 174920                            # B 实际保留行数


# ---------------------------------------------------------------- 3. 路径守卫
def test_output_path_guard(mod):
    for bad in [ROOT / "data/processed/x.csv", ROOT / "reports/data_gates/G0/x.json",
                ROOT / "configs/x.yaml", ROOT / "scripts/x.py",
                ROOT / "tests/x.py", ROOT / "data/processed/monthly_returns_clean_v2.csv",
                ROOT / "reports/data_gates/G0/s3_m1_pricing_table_v2.json"]:
        with pytest.raises(SystemExit):
            mod.assert_output_path_guard(bad)
    for ok in [ROOT / "artifacts/phase2/2.6_comparison.csv",
               ROOT / "artifacts/phase2/2.6_counts.json"]:
        assert mod.assert_output_path_guard(ok) is None


def test_path_fallback_outside_root_no_crash(mod, tmp_path):
    """--out-* 指向 ROOT 外：展示/记录路径回退绝对路径，写盘仍成功（不崩溃）。"""
    p = tmp_path / "outside_root.json"
    assert not str(p).startswith(str(ROOT))
    assert mod._rel_or_abs(p) == p.resolve().as_posix()
    assert mod._rel(ROOT / "artifacts/phase2/x.json").startswith("artifacts/")
    mod.write_json(p, {"ok": True})                       # 不得抛异常
    assert json.loads(p.read_text(encoding="utf-8")) == {"ok": True}
    csv_p = tmp_path / "outside_root.csv"
    mod.write_csv(csv_p, [{"a": 1, "b": 2}], ["a", "b"])
    assert csv_p.read_text(encoding="utf-8").splitlines()[0] == "a,b"


# ---------------------------------------------------------------- 4. 产物契约
def test_outputs_exist_and_match_counts(counts):
    assert CSV.is_file() and MEMO.is_file() and COUNTS.is_file()
    assert counts["outputs"]["comparison_csv_sha256"] == sha256_file(CSV)
    df = pd.read_csv(CSV)
    assert len(df) == counts["outputs"]["comparison_csv_rows"] == 1584
    assert set(mod_columns(df)) == set(_expected_columns())
    assert set(df["variant"]) >= set(SHADOW_VARIANTS) | {"headline_G0_table_v2"}
    assert set(df["teacher"]) == {"ma_teacher", "mb_teacher"}
    assert set(df["spec_id"]) == {"S1_full_last", "S2_full_mean", "S3_lag1_last",
                                  "S4_lag1_mean", "S5_no_turnover", "S6_no_rev_no_turnover"}


def mod_columns(df):
    return list(df.columns)


def _expected_columns():
    return ["variant", "variant_label", "teacher", "spec_id", "spec_label", "weight",
            "term", "beta", "se_nw", "t_nw", "n_months_used", "n_rows_used",
            "n_per_month_min", "n_per_month_median", "n_per_month_max", "p_two_sided",
            "ci95_low", "ci95_high", "is_treatment", "rows_mode", "source", "note"]


def test_headline_rows_in_csv_equal_pricing_table(counts):
    df = pd.read_csv(CSV)
    g0 = df[(df["variant"] == "headline_G0_table_v2") & (df["teacher"] == "ma_teacher")
            & (df["spec_id"] == "S1_full_last")]
    cell = g0[(g0["weight"] == "ew") & (g0["term"] == "PeerMAXraw")].iloc[0]
    assert cell["beta"] == pytest.approx(0.034273, abs=1e-6)
    own = g0[(g0["weight"] == "vw") & (g0["term"] == "OwnMAXraw")].iloc[0]
    assert own["beta"] == pytest.approx(0.017150, abs=1e-6)
    assert int(cell["n_months_used"]) == 104 and int(cell["n_rows_used"]) == 181178


def test_shadow_headline_cells_match_counts_matrix(counts):
    """CSV 中的 A/B/C 单元与 counts 的 comparison_matrix 逐值一致（同一产物内一致）。"""
    df = pd.read_csv(CSV)
    M = counts["comparison_matrix"]["ma_teacher"]
    for variant in SHADOW_VARIANTS:
        for spec_id in ["S1_full_last", "S6_no_rev_no_turnover"]:
            for w in ["ew", "vw"]:
                for term in TRIO:
                    row = df[(df["variant"] == variant) & (df["teacher"] == "ma_teacher")
                             & (df["spec_id"] == spec_id) & (df["weight"] == w)
                             & (df["term"] == term)].iloc[0]
                    ref = M[variant][spec_id][w]["coefs"][term]
                    assert row["beta"] == pytest.approx(ref["beta"], abs=1e-9)
                    assert row["t_nw"] == pytest.approx(ref["t_nw"], abs=1e-9)


# ---------------------------------------------------------------- 5. headline 未变
def test_headline_shas_unchanged_and_registered():
    for name, expected in HEADLINE_ANCHORS.items():
        path = {"m1_common_sample_ma_teacher_v2.parquet": SAMPLE_MA,
                "s3_m1_pricing_table_v2.json": PRICING_TABLE_V2,
                "firm_chars_clean.csv": FIRM_CHARS,
                "source_month_argmax_panel.csv": SOURCE_PANEL,
                "monthly_returns_clean_v2.csv": RETURNS_V2}[name]
        assert sha256_file(path) == expected, f"{name} sha 变化（headline 被写）"


def test_counts_governance_flags(counts):
    i = counts["inputs_sha"]
    assert i["m1_anchors_match"] is True
    assert i["headline_unchanged"] is True
    assert i["shadow_unchanged"] is True
    assert i["shadow_chain_vs_phase2_counts"]["all_match"] is True
    assert counts["governance"] == {"headline_files_modified": False, "m1_rerun": False,
                                    "configs_modified": False, "g0_written": False,
                                    "promotion_executed": False}
    assert "reports" not in counts["outputs"]["comparison_csv"].replace("\\", "/")
    assert counts["outputs"]["comparison_csv"].replace("\\", "/").startswith("artifacts/phase2/2.6_")
    assert counts["outputs"]["counts_json"].replace("\\", "/").startswith("artifacts/phase2/2.6_")


# ---------------------------------------------------------------- 6. 校准 / 判据
def test_calibration_hard_gates_pass(counts):
    for teacher in ["ma_teacher", "mb_teacher"]:
        cal = counts["calibration"][teacher]
        for key in ["headline_recompute_vs_G0_table_v2", "headline_rebuilt_vs_G0_table_v2"]:
            assert cal[key]["status"] == "PASS", f'{teacher}/{key} 校准失败'
            assert cal[key]["worst_abs_beta_diff"] <= 1e-4
        if teacher == "ma_teacher":
            # 2.4/2.5 交叉锚只适用于 ma_teacher（已发布矩阵为 ma 口径）
            assert cal["cross_anchor_worst_abs_diff"] <= 1e-6
            assert cal["cross_anchor_status"] == "PASS"
        else:
            assert "cross_anchor_worst_abs_diff" not in cal
    li = counts["decomposition"]["ma_teacher"]["legacy_rebuild_isolation"]
    assert li["own_max_max_abs_diff"] <= 1e-9
    assert li["peer_max_max_abs_diff"] <= 1e-7 and li["peer_return_max_abs_diff"] <= 1e-7
    assert li["R_next_max_abs_diff"] <= 1e-9 and li["rev_max_abs_diff"] <= 1e-9


def test_promotion_criteria_recorded(counts):
    p = counts["promotion_criteria"]
    assert set(p) == {f"P{i}_{n}" for i, n in [
        (1, "peer_max_direction_significance"), (2, "own_max_direction"), (3, "pseudo_zero"),
        (4, "sample_loss"), (5, "max_tail"), (6, "reproducibility"),
        (7, "conclusion_robustness"), (8, "explicit_approval")]}
    assert p["P1_peer_max_direction_significance"]["n_cells_positive_significant"] == 0
    assert p["P2_own_max_direction"]["status"] == "PASS"
    assert p["P2_own_max_direction"]["n_s6_negative"] == p["P2_own_max_direction"]["n_s6_cells"]
    assert p["P3_pseudo_zero"]["shadow_outcome_zero_share_A"] <= 0.009
    assert p["P4_sample_loss"]["status"] == "PASS"
    assert p["P7_conclusion_robustness"]["n_flip_meaningful"] == 0
    assert p["P8_explicit_approval"]["status"] == "PENDING_PROJECT_APPROVAL"
    for k, v in p.items():
        assert v["status"] != "FAIL", f"{k} FAIL（不得推广，需复核）"


def test_a_b_c_reported_and_peer_max_insignificant(counts):
    M = counts["comparison_matrix"]
    for teacher in ["ma_teacher", "mb_teacher"]:
        for variant in SHADOW_VARIANTS:
            assert variant in M[teacher]
    # PeerMAX 全变体无正显著（P1 核心断言）
    for teacher in ["ma_teacher", "mb_teacher"]:
        for variant in SHADOW_VARIANTS:
            for spec_id, weights in M[teacher][variant].items():
                for w, cell in weights.items():
                    t = cell["coefs"]["PeerMAXraw"]["t_nw"]
                    assert t is None or t < 1.96, f"{teacher}/{variant}/{spec_id}/{w} PeerMAX 正显著"


# ---------------------------------------------------------------- 7. 状态推导 / 摘要一致性
def test_p5_p6_status_derived_not_hardcoded(counts):
    p = counts["promotion_criteria"]
    assert "derived" in p["P5_max_tail"]["status_basis"]
    assert p["P5_max_tail"]["status"] == "PASS"
    # 主口径 >0.44 行数与 adj_ex_resume 口径都在 counts 中
    assert p["P5_max_tail"]["shadow_maxraw_gt_044_rows_main_gate"] == 5
    assert p["P5_max_tail"]["shadow_maxraw_gt_044_rows_adj_ex_resume"] == 14
    assert p["P5_max_tail"]["registered_reference"] == \
        counts["max_tail"]["ma_teacher"]["registered_winsorized_max_reference"]
    assert "derived" in p["P6_reproducibility"]["status_basis"]
    ev = p["P6_reproducibility"]["evidence"]
    assert ev["m1_anchors_match"] is True and ev["shadow_chain_all_match"] is True
    assert ev["comparison_csv_sha256"] == counts["outputs"]["comparison_csv_sha256"]
    assert p["P6_reproducibility"]["status"] == "PASS"


def test_plan_replication_mb_uses_own_actuals(counts):
    pr = counts["plan_2_0_replication"]
    assert "plan_2_0_expected" in pr["ma_teacher"]
    assert all(pr["ma_teacher"]["plan_2_0_match"].values())
    mb = pr["mb_teacher"]
    assert "plan_2_0_expected" not in mb and "plan_2_0_match" not in mb
    assert "不适用" in mb["anchors_scope"]
    assert mb["actual"] == {"outcome_gate_loss": 4018, "treatment_gate_loss_ntd_lt15": 3805,
                            "treatment_gate_loss_nmd_lt15": 4296, "union_loss_main_gate": 6646,
                            "B_rows": 218146, "pseudo_zero_fm": 1262}
    # ma 的 actual 与 counts 的 sample_loss 逐值一致
    sl = counts["sample_loss"]["ma_teacher"]
    assert pr["ma_teacher"]["actual"]["B_rows"] == sl["B_rows_explicit_screen"]
    assert pr["ma_teacher"]["actual"]["union_loss_main_gate"] == sl["A_main_gate"]["union_rows"]


def test_coefficient_summary_matches_matrix(counts):
    """coefficient_summary.PeerReturn 与 comparison_matrix 逐值一致（memo 引用源）。"""
    M = counts["comparison_matrix"]
    recs = []
    for teacher in ["ma_teacher", "mb_teacher"]:
        for variant in SHADOW_VARIANTS:
            for spec_id in ["S1_full_last", "S2_full_mean", "S3_lag1_last", "S4_lag1_mean",
                            "S5_no_turnover", "S6_no_rev_no_turnover"]:
                for w in ["ew", "vw"]:
                    s = M[teacher][variant][spec_id][w]["coefs"]["PeerReturn"]
                    recs.append((teacher, variant, spec_id, w, s["beta"], s["t_nw"]))
    ts = np.array([r[5] for r in recs])
    bs = np.array([r[4] for r in recs])
    s = counts["coefficient_summary"]["PeerReturn"]
    assert s["all"]["n_cells"] == len(recs) == 96
    assert s["all"]["n_positive"] == int((bs > 0).sum()) == 96
    assert s["all"]["n_abs_t_lt_196"] == int((np.abs(ts) < 1.96).sum()) == 29
    assert s["all"]["min_t"] == pytest.approx(float(ts.min()))
    assert s["all"]["max_t"] == pytest.approx(float(ts.max()))
    assert s["n_abs_t_lt_196_by_teacher"] == {
        "ma_teacher": int((np.abs(ts[[r[0] == "ma_teacher" for r in recs]]) < 1.96).sum()),
        "mb_teacher": int((np.abs(ts[[r[0] == "mb_teacher" for r in recs]]) < 1.96).sum())}
    assert s["n_abs_t_lt_196_by_teacher"] == {"ma_teacher": 7, "mb_teacher": 22}
    i_min = int(ts.argmin())
    assert s["min_t_cell"]["teacher"] == recs[i_min][0] == "mb_teacher"
    assert s["min_t_cell"]["variant"] == recs[i_min][1]
    assert s["min_t_cell"]["spec_id"] == recs[i_min][2]
    assert s["min_t_cell"]["weight"] == recs[i_min][3] == "vw"
    assert s["min_t_cell"]["t"] == pytest.approx(float(ts.min()))
    for w in ("ew", "vw"):
        sel = np.array([r[3] == w for r in recs])
        assert s[w]["n_cells"] == int(sel.sum()) == 48
        assert s[w]["n_abs_t_lt_196"] == int((np.abs(ts[sel]) < 1.96).sum())
        assert s[w]["min_t"] == pytest.approx(float(ts[sel].min()))
        assert s[w]["max_t"] == pytest.approx(float(ts[sel].max()))
    assert s["ew"]["n_abs_t_lt_196"] == 1 and s["vw"]["n_abs_t_lt_196"] == 28
    for key, teacher, w in [("ma_teacher|vw", "ma_teacher", "vw"), ("mb_teacher|vw", "mb_teacher", "vw")]:
        sel = np.array([(r[0] == teacher and r[3] == w) for r in recs])
        assert s["by_teacher_weight"][key]["min_t"] == pytest.approx(float(ts[sel].min()))
        assert s["by_teacher_weight"][key]["max_t"] == pytest.approx(float(ts[sel].max()))
        assert s["by_teacher_weight"][key]["n_abs_t_lt_196"] == int((np.abs(ts[sel]) < 1.96).sum())
    assert s["s1_registered_mb_vw"]["B_rescreened"]["t"] == pytest.approx(1.844, abs=1e-3)
    assert s["s1_registered_mb_vw"]["C_rescreened_adj"]["t"] == pytest.approx(1.541, abs=1e-3)


def test_execution_record_and_residual_limitations(counts):
    exe = ROOT / "artifacts/phase2/2.6_execution.md"
    assert exe.is_file(), "缺少 2.6_execution.md"
    assert counts["outputs"]["execution_record"] == "artifacts/phase2/2.6_execution.md"
    res = counts["residual_limitations"]
    assert len(res) == 11
    joined = " ".join(res)
    assert "mom/illiq" in joined and "pre-2014" in joined and "2018+ 重整" in joined
    assert "2.6_execution 记录" not in json.dumps(counts["promotion_criteria"], ensure_ascii=False)
