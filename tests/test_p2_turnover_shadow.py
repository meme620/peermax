"""Phase 2.4 shadow turnover 契约测试（加法，只读真实产物 + 合成序列）。

覆盖：
  1. 合成宽表：last（月内最后一个非空值）/ mean（月内非空均值，含 0 值日）/ n_days（月内非空日数）
     语义；NaN 保持 NaN 不填 0；整月无数据 -> n_days=0 且 last/mean 为空；月界归属正确。
  2. 真实 pkl（只读）：plan 窗口 2014-01..2024-01 匹配 446,301 行、corr≈0.780、平均绝对差≈1.257pp；
     全 pkl corr≈0.779 / mad≈1.267pp；FM 窗口 2015-05..2023-12 400,092 行、corr≈0.783。
  3. shadow CSV/meta 契约：669,614 行、5 列、唯一键、窗口、n_days 语义、meta 的 output sha 与
     文件一致、source pkl sha 与当前 pkl 一致；CSV 与脚本内重算逐值一致（≤1e-8）。
  4. 四规格 FM 矩阵：a 逐值复现 headline 主表（s3_m1_pricing_table_v2.json）；b（winsorized mean）
     与审计 L-8 判定表一致、方向与 2.0_plan 一致（PeerMAX>0、OwnMAX≈0、均不显著）；
     b3 诊断（无 winsor mean）逐值复现 2.0_plan §2.4「现状」的 mean 行；c/d 复现 plan 锚点；
     VW 下四规格 PeerMAX 均不显著。
  5. `load_turnover_drive()` 管道未被改动（仍为 last，逐值等于本 shadow 的 turnover_last/100）。
  6. headline 3 文件（firm_chars_clean.csv / monthly_pricing_panel_ma_teacher.csv / M1 样本 parquet）
     sha 前后不变，且等于登记锚；turnover pkl 只读不变。
  7. 输出路径治理：shadow 输出不得命中 headline 禁写集合。

运行：python -m pytest tests/test_p2_turnover_shadow.py -q
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
SCRIPT = ROOT / "scripts/data_audit/p2_4_turnover_shadow.py"
SHADOW_CSV = ROOT / "data/processed/phase2_shadow/turnover_p2shadow.csv"
SHADOW_META = ROOT / "data/processed/phase2_shadow/turnover_p2shadow.meta.json"
COUNTS = ROOT / "artifacts/phase2/2.4_counts.json"
DIFF_CSV = ROOT / "artifacts/phase2/2.4_turnover_diff.csv"
MATRIX_CSV = ROOT / "artifacts/phase2/2.4_control_matrix.csv"
SAMPLE = ROOT / "data/processed/m1_common_sample_ma_teacher_v2.parquet"
PRICING_V2 = ROOT / "reports/data_gates/G0/s3_m1_pricing_table_v2.json"
SAMPLE_REPORT_V2 = ROOT / "reports/data_gates/G0/s3_m1_sample_report_v2.json"
TURNOVER_PKL = ROOT / "data/raw/drive/daily_turnover_rate_20260630.pkl"
MANIFEST = ROOT / "configs/version_manifest.yaml"

EXPECTED_ROWS = 669_614
EXPECTED_BOTH = 446_301
FM_WINDOW = ("2015-05", "2023-12")

HEADLINE_FILES = {
    "firm_chars_clean.csv": ROOT / "data/processed/firm_chars_clean.csv",
    "monthly_pricing_panel_ma_teacher.csv": ROOT / "data/processed/monthly_pricing_panel_ma_teacher.csv",
    "m1_common_sample_ma_teacher_v2.parquet": SAMPLE,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("p2_4_turnover_shadow", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def sha_before():
    out = {name: sha256_file(p) for name, p in HEADLINE_FILES.items()}
    out["__turnover_pkl__"] = sha256_file(TURNOVER_PKL)
    return out


@pytest.fixture(scope="module")
def wide(mod):
    return mod.load_turnover_wide()


@pytest.fixture(scope="module")
def shadow_frame(mod, wide):
    return mod.build_shadow_frame(wide)


@pytest.fixture(scope="module")
def control_bundle(mod, shadow_frame):
    return mod.build_control_frame(SAMPLE, shadow_frame)


@pytest.fixture(scope="module")
def spec_matrix(mod, control_bundle):
    fm, _ = control_bundle
    return mod.run_spec_matrix(fm)


@pytest.fixture(scope="module")
def diag_matrix(mod, control_bundle):
    fm, _ = control_bundle
    defs = {k: {"label": v[0],
                "x": mod.TRIO + [v[1] if c == "turnover" else c for c in mod.CONTROLS],
                "turnover_col": v[1], "turnover_variant": "mean-diagnostic"}
            for k, v in mod.DIAG_SPEC_DEFS.items()}
    return mod.run_spec_matrix(fm, defs)


# ---------------------------------------------------------------- 1. 合成序列
def test_synthetic_last_mean_n_days(mod):
    idx = pd.to_datetime(["2020-01-29", "2020-01-30", "2020-01-31",
                          "2020-02-27", "2020-02-28"])
    w = pd.DataFrame({
        "000001.SZ": [1.0, np.nan, np.nan, 2.0, 4.0],   # 1 月最后一个非空=1.0；2 月 mean=3.0
        "600000.SH": [np.nan, np.nan, np.nan, np.nan, np.nan],   # 全 NaN
        "000002.SZ": [0.0, np.nan, 0.0, np.nan, np.nan],  # 停牌 0 值日计入
    }, index=idx)
    fr = mod.monthly_frames(w)
    assert fr["turnover_last"].loc["2020-01", "000001.SZ"] == 1.0
    assert fr["turnover_mean"].loc["2020-01", "000001.SZ"] == 1.0  # 只有 1 个非空日
    assert fr["n_days"].loc["2020-01", "000001.SZ"] == 1
    assert fr["turnover_last"].loc["2020-02", "000001.SZ"] == 4.0
    assert fr["turnover_mean"].loc["2020-02", "000001.SZ"] == 3.0
    assert fr["n_days"].loc["2020-02", "000001.SZ"] == 2
    # 0 值日是数据（停牌），不是缺失
    assert fr["n_days"].loc["2020-01", "000002.SZ"] == 2
    assert fr["turnover_last"].loc["2020-01", "000002.SZ"] == 0.0
    assert fr["turnover_mean"].loc["2020-01", "000002.SZ"] == 0.0
    # 整月无数据：n_days=0，last/mean 为空（不得填 0）
    for m in ("2020-01", "2020-02"):
        assert fr["n_days"].loc[m, "600000.SH"] == 0
        assert pd.isna(fr["turnover_last"].loc[m, "600000.SH"])
        assert pd.isna(fr["turnover_mean"].loc[m, "600000.SH"])


def test_synthetic_shadow_frame_contract(mod):
    idx = pd.to_datetime(["2020-01-31", "2020-02-28"])
    w = pd.DataFrame({"000001.SZ": [1.5, np.nan], "600000.SH": [np.nan, 2.5]}, index=idx)
    df = mod.build_shadow_frame(w, start="2020-01", end="2020-02")
    assert list(df.columns) == ["firm_id", "month", "turnover_last", "turnover_mean", "n_days"]
    assert len(df) == 4  # 2 code × 2 月
    assert df["firm_id"].tolist() == ["sh600000", "sz000001", "sh600000", "sz000001"]
    assert df["month"].tolist() == ["2020-01", "2020-01", "2020-02", "2020-02"]
    a = df[(df["firm_id"] == "sz000001") & (df["month"] == "2020-01")].iloc[0]
    assert a["turnover_last"] == 1.5 and a["n_days"] == 1
    b = df[(df["firm_id"] == "sh600000") & (df["month"] == "2020-01")].iloc[0]
    assert b["n_days"] == 0 and pd.isna(b["turnover_last"]) and pd.isna(b["turnover_mean"])
    assert df["n_days"].dtype == np.int64


# ---------------------------------------------------------------- 2. 真实 pkl 差异统计
def test_real_pkl_plan_window_stats(mod, shadow_frame):
    s = mod.window_diff_stats(shadow_frame)
    assert s["n_rows"] == EXPECTED_ROWS and s["n_both"] == EXPECTED_BOTH
    assert abs(s["corr"] - 0.780414) < 2e-3
    assert abs(s["mean_abs_diff_pp"] - 1.2572) < 5e-3
    assert abs(s["median_abs_diff_pp"] - 0.4843) < 5e-3
    assert s["mean_signed_diff_pp"] < 0  # last < mean（月末快照系统性略低于月均）


def test_real_pkl_fm_window_stats(mod, shadow_frame):
    s = mod.window_diff_stats(shadow_frame, *FM_WINDOW)
    assert s["n_both"] == 400_092
    assert abs(s["corr"] - 0.783307) < 2e-3
    assert abs(s["mean_abs_diff_pp"] - 1.2635) < 5e-3


def test_real_pkl_full_range_stats(mod, wide):
    s = mod.window_diff_stats(mod._to_long_asis(wide))
    assert s["n_both"] == 862_230
    assert abs(s["corr"] - 0.779217) < 2e-3
    assert abs(s["mean_abs_diff_pp"] - 1.2666) < 5e-3


# ---------------------------------------------------------------- 3. shadow CSV / meta 契约
def test_shadow_csv_contract(shadow_frame):
    df = pd.read_csv(SHADOW_CSV, dtype={"firm_id": str, "month": str})
    assert list(df.columns) == ["firm_id", "month", "turnover_last", "turnover_mean", "n_days"]
    assert len(df) == EXPECTED_ROWS
    assert df["month"].min() == "2014-01" and df["month"].max() == "2024-01"
    assert int(df.duplicated(["firm_id", "month"]).sum()) == 0
    assert bool((df["n_days"] >= 0).all())
    zero = df["n_days"] == 0
    assert bool(df.loc[zero, ["turnover_last", "turnover_mean"]].isna().all().all())
    # CSV 与脚本内重算逐值一致（写出 round(8)）
    chk = df.merge(shadow_frame, on=["firm_id", "month"], suffixes=("_csv", "_mem"))
    assert len(chk) == EXPECTED_ROWS
    for c in ("turnover_last", "turnover_mean"):
        d = (chk[f"{c}_csv"] - chk[f"{c}_mem"]).abs()
        assert float(d.max()) < 1e-8
    assert (chk["n_days_csv"] == chk["n_days_mem"]).all()


def test_shadow_meta_contract():
    meta = json.loads(SHADOW_META.read_text(encoding="utf-8"))
    assert meta["p2shadow_version"] == "p2shadow_v1"
    assert meta["row_count"] == EXPECTED_ROWS
    assert meta["row_count_last_mean_nonnull"] == EXPECTED_BOTH
    assert meta["window"]["start"] == "2014-01" and meta["window"]["end"] == "2024-01"
    assert meta["output_sha256"] == sha256_file(SHADOW_CSV)
    assert meta["source"]["sha256"] == sha256_file(TURNOVER_PKL)
    assert meta["script_sha256"] == sha256_file(SCRIPT)
    assert meta["columns"] == ["firm_id", "month", "turnover_last", "turnover_mean", "n_days"]


def test_counts_and_diff_outputs():
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    assert counts["p2shadow_version"] == "p2shadow_v1"
    assert counts["outputs"]["shadow_turnover_csv"]["sha256"] == sha256_file(SHADOW_CSV)
    assert counts["outputs"]["diff_csv"]["sha256"] == sha256_file(DIFF_CSV)
    assert counts["outputs"]["matrix_csv"]["sha256"] == sha256_file(MATRIX_CSV)
    assert counts["headline_anchors_match"] is True
    assert counts["headline_files_modified"] is False
    diff = pd.read_csv(DIFF_CSV)
    assert set(diff["year"].astype(int)) >= {2014, 2023, 2024}
    assert not diff["mean_abs_diff_pp"].isna().any()
    mat = pd.read_csv(MATRIX_CSV)
    assert set(mat["spec_id"]) == {"a_turnover_last", "b_turnover_mean",
                                   "c_no_turnover", "d_no_turnover_no_rev"}
    assert set(mat["weight"]) == {"ew", "vw"}


# ---------------------------------------------------------------- 4. 四规格矩阵
def test_spec_a_reproduces_headline(mod, spec_matrix):
    cmp = mod.headline_comparison(spec_matrix)
    assert cmp["all_match"] is True
    a = spec_matrix["specs"]["a_turnover_last"]
    assert a["ew"]["T"] == 104 and a["ew"]["n_frame"] == 181_178
    e = a["ew"]["coefs"]
    assert abs(e["PeerMAXraw"]["beta"] - 0.03427281) < 5e-4
    assert abs(e["PeerMAXraw"]["t_nw"] - 0.472809) < 5e-3
    assert abs(e["OwnMAXraw"]["beta"] + 0.00281881) < 5e-4
    assert abs(e["turnover"]["beta"] + 0.39870074) < 1e-3  # turnover 仍强负


def test_spec_b_mean_direction_matches_plan(mod, spec_matrix):
    b = spec_matrix["specs"]["b_turnover_mean"]
    e = b["ew"]["coefs"]
    # 主口径 winsorized mean：与审计 L-8 判定表一致（+0.0569/0.80、+0.0119/0.62、-0.4163/-8.78）
    assert abs(e["PeerMAXraw"]["beta"] - 0.056943) < 1e-3
    assert abs(e["PeerMAXraw"]["t_nw"] - 0.796397) < 0.02
    assert abs(e["OwnMAXraw"]["beta"] - 0.011885) < 5e-4
    assert abs(e["OwnMAXraw"]["t_nw"] - 0.623772) < 0.02
    assert abs(e["turnover_mean_win"]["beta"] + 0.416288) < 1e-3
    assert abs(e["turnover_mean_win"]["t_nw"] + 8.776834) < 0.05
    # 2.0_plan 预期方向：PeerMAX>0、OwnMAX≈0，两者均不显著（|t|<1.96）
    assert e["PeerMAXraw"]["beta"] > 0 and abs(e["PeerMAXraw"]["t_nw"]) < 1.96
    assert e["OwnMAXraw"]["beta"] > 0 and abs(e["OwnMAXraw"]["beta"]) < 0.02
    assert abs(e["OwnMAXraw"]["t_nw"]) < 1.96
    assert b["ew"]["T"] == 104 and b["ew"]["n_frame"] == 181_178


def test_spec_b3_no_winsor_reproduces_plan_quoted_row(diag_matrix):
    """2.0_plan §2.4「现状」mean 行 = 未 winsor 的 Drive 均值变体（逐值复现）。"""
    e = diag_matrix["specs"]["b3_mean_no_winsor"]["ew"]["coefs"]
    assert abs(e["OwnMAXraw"]["beta"] - 0.005833) < 2e-4
    assert abs(e["OwnMAXraw"]["t_nw"] - 0.295509) < 0.01
    assert abs(e["PeerMAXraw"]["beta"] - 0.053768) < 5e-4
    assert abs(e["PeerMAXraw"]["t_nw"] - 0.744490) < 0.01


def test_spec_c_d_reproduce_plan_anchors(spec_matrix):
    c = spec_matrix["specs"]["c_no_turnover"]["ew"]["coefs"]
    d = spec_matrix["specs"]["d_no_turnover_no_rev"]["ew"]["coefs"]
    assert abs(c["OwnMAXraw"]["beta"] + 0.10550381) < 5e-4
    assert abs(c["OwnMAXraw"]["t_nw"] + 3.800937) < 0.02
    assert abs(d["OwnMAXraw"]["beta"] + 0.13585820) < 5e-4
    assert abs(d["OwnMAXraw"]["t_nw"] + 4.969872) < 0.02
    for coefs in (c, d):
        assert abs(coefs["PeerMAXraw"]["t_nw"]) < 1.96  # PeerMAX 零结果不因控制集改变


def test_all_specs_peer_max_insignificant_vw(spec_matrix):
    for sid, blk in spec_matrix["specs"].items():
        e = blk["vw"]["coefs"]
        assert abs(e["PeerMAXraw"]["t_nw"]) < 1.96, sid
        assert blk["vw"]["T"] == 104


# ---------------------------------------------------------------- 5. 既有 last 管道未被改动
def test_load_turnover_drive_still_last(mod, shadow_frame):
    import sys
    sys.path.insert(0, str(ROOT / "scripts/data_audit"))
    from s3_m1_lib import load_turnover_drive  # noqa: WPS433
    drive = load_turnover_drive()
    m = drive.merge(shadow_frame[["firm_id", "month", "turnover_last"]],
                    left_on=["firm_id", "month"], right_on=["firm_id", "month"], how="inner")
    assert len(m) == len(drive)  # shadow 窗口覆盖 headline 全部 (2014-06..2024-01) 键
    assert float((m["turnover"] - m["turnover_last"] / 100.0).abs().max()) < 1e-12


# ---------------------------------------------------------------- 6. headline 未变
def test_headline_shas_unchanged(sha_before, spec_matrix):
    for name, p in HEADLINE_FILES.items():
        assert sha256_file(p) == sha_before[name], name
    assert sha256_file(TURNOVER_PKL) == sha_before["__turnover_pkl__"]


def test_headline_shas_equal_registered_anchors():
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    anchors = {}
    for e in manifest["deliverables"]:
        f = str(e.get("file", "")).replace("\\", "/")
        sha = str(e.get("sha256") or "")
        if sha:
            anchors[f] = sha
    assert sha256_file(HEADLINE_FILES["firm_chars_clean.csv"]) == \
        anchors["data/processed/firm_chars_clean.csv"]
    assert sha256_file(HEADLINE_FILES["monthly_pricing_panel_ma_teacher.csv"]) == \
        anchors["data/processed/monthly_pricing_panel_ma_teacher.csv"]
    report = json.loads(SAMPLE_REPORT_V2.read_text(encoding="utf-8"))
    assert sha256_file(SAMPLE) == report["files"]["v2_samples"]["ma_teacher"]["sha256"]


# ---------------------------------------------------------------- 7. 路径治理
def test_output_path_guard(mod):
    mod.assert_output_path_guard(mod.OUT_CSV)
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(HEADLINE_FILES["monthly_pricing_panel_ma_teacher.csv"])
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(ROOT / "data/processed/firm_chars_clean.csv")
    with pytest.raises(SystemExit):
        mod.assert_output_path_guard(ROOT / "reports/data_gates/G0/anything.json")


def test_counts_spec_matrix_matches_recompute(spec_matrix):
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    for sid, blk in spec_matrix["specs"].items():
        c_blk = counts["spec_matrix"]["specs"][sid]
        assert c_blk["ew"]["T"] == blk["ew"]["T"]
        assert c_blk["ew"]["n_frame"] == blk["ew"]["n_frame"]
        for term in ("PeerMAXraw", "OwnMAXraw"):
            assert abs(c_blk["ew"]["coefs"][term]["beta"] - blk["ew"]["coefs"][term]["beta"]) < 1e-12
