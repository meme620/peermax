"""Phase 2.2 shadow MAX 契约测试（加法，只读真实产物）。

覆盖：
  1. 合成日频：相邻交易日、跨停牌、跨月、跨年、首行、整月复牌；
     `resume_flag` / `MAXraw_p2shadow_ungated` / argmax tie rule。
  2. 官方 limit flag 与阈值口径的差异；官方 pkl 缺失时回退语义。
  3. 真实面板只读计数：734,232 行、键唯一、旧列逐字段一致、行序不变。
  4. 2014–2025 复牌 13,986；2015–2023（日历年）argmax 复牌 2,886、
     MAX 改变 2,666、limit_hit 复牌 1,724；M1 窗口（2015-05..2023-12）
     2,601 / 2,396 / 1,528。
  5. gate：source n_valid<15 = 7,848、可匹配 5,964；n_max<15 = 8,576/6,692。
  6. MAX 尾部：raw 5.93717277→shadow 1.08544087；>0.44 19→14；>1.0 6→2。
     2.0_plan 的 17/3 不可复现，测试锁定实际严格口径与偏差说明。
  7. 复牌日不进入 shadow argmax；非复牌月份零变化；shadow <= legacy。
  8. 9 个 M1 输入 sha 与 `m1_contracts.yaml` 一致；headline source panel sha 未变。

运行：python -m pytest tests/test_shadow_max_contracts.py -q
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/shadow_max_build.py"
SHADOW = ROOT / "data/processed/phase2_shadow/source_month_argmax_panel_p2shadow.csv"
META = ROOT / "data/processed/phase2_shadow/source_month_argmax_panel_p2shadow.meta.json"
COUNTS = ROOT / "artifacts/phase2/2.2_counts.json"
RESUME_AUDIT = ROOT / "artifacts/phase2/2.2_resume_audit.csv"
TAIL = ROOT / "artifacts/phase2/2.2_max_tail_diff.csv"
SOURCE = ROOT / "data/processed/source_month_argmax_panel.csv"
CONTRACTS = ROOT / "configs/m1_contracts.yaml"

EXPECTED_ROWS = 734_232
HISTORICAL_SOURCE_SHA = "944adbbab95a7b30d858f2943c8ec2844a852a9551b06224b21147d3c4c6023a"
OLD_SOURCE_HEADER = [
    "source_id", "month", "MAXraw", "argmax_date", "n_valid_days", "R",
    "tie_rule_version", "formula_id", "limit_hit_at_argmax",
    "qualification_status", "st_flag_at_argmax"]
NEW_COLS = [
    "n_valid_days_p2shadow", "n_resume_days", "n_max_days",
    "MAXraw_p2shadow", "argmax_date_p2shadow",
    "MAXraw_p2shadow_ungated", "argmax_date_p2shadow_ungated",
    "resume_at_argmax", "gap_td_at_argmax",
    "R_includes_resume_gap", "R_includes_cross_year_gap",
    "limit_hit_p2shadow", "limit_source", "limit_status_at_resume",
    "limit_official_status_at_resume", "st_flag_at_argmax_p2shadow",
    "max_gate_15", "gate_reason", "qualification_status_p2shadow",
    "formula_id_p2shadow", "tie_rule_version_p2shadow", "R_p2shadow",
    "shadow_status", "p2shadow_version",
]
CORE_COLS = [
    "source_id", "month", "MAXraw", "argmax_date", "n_valid_days", "R",
    "limit_hit_at_argmax", "n_valid_days_p2shadow", "n_resume_days",
    "n_max_days", "MAXraw_p2shadow", "argmax_date_p2shadow",
    "MAXraw_p2shadow_ungated", "argmax_date_p2shadow_ungated",
    "resume_at_argmax", "gap_td_at_argmax", "limit_hit_p2shadow",
    "limit_source", "limit_status_at_resume",
    "limit_official_status_at_resume", "max_gate_15", "gate_reason",
    "qualification_status_p2shadow", "formula_id_p2shadow", "R_p2shadow",
    "shadow_status",
]


def file_size(path):
    return Path(path).stat().st_size


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_max_build", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


@pytest.fixture(scope="module")
def mod():
    assert SCRIPT.is_file(), f"缺少脚本: {SCRIPT}"
    return load_module()


@pytest.fixture(scope="module")
def counts() -> dict:
    assert COUNTS.is_file(), f"缺少 counts（先运行 {SCRIPT.name}）"
    return json.loads(COUNTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def meta() -> dict:
    assert META.is_file(), f"缺少 meta（先运行 {SCRIPT.name}）"
    return json.loads(META.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shadow() -> pd.DataFrame:
    assert SHADOW.is_file(), f"缺少 shadow 产物（先运行 {SCRIPT.name}）"
    return pd.read_csv(SHADOW, dtype=str, keep_default_na=False,
                       usecols=CORE_COLS)


@pytest.fixture(scope="module")
def resume_audit() -> pd.DataFrame:
    assert RESUME_AUDIT.is_file(), f"缺少 resume audit（先运行 {SCRIPT.name}）"
    return pd.read_csv(RESUME_AUDIT, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- 1. 合成语义
def test_synthetic_resume_flags_and_max(mod):
    calendar = [
        "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08",
        "2020-02-03", "2020-02-04", "2020-03-02",
        "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05",
    ]
    idx = {d: i for i, d in enumerate(calendar)}
    rows = [
        # A：正常相邻日、周末（日历跳 2 天但只隔 1 个交易日）、跨月、跨年
        ("szA", "2020-01-02", np.nan),
        ("szA", "2020-01-03", 0.01),
        ("szA", "2020-01-06", 0.02),   # 周五 -> 周一：gap=1，不是复牌
        ("szA", "2020-01-07", 0.03),
        ("szA", "2020-02-03", 0.50),   # 跨月停牌 -> 复牌
        ("szA", "2020-02-04", 0.04),
        ("szA", "2020-12-30", 0.10),   # 跨长停牌 -> 复牌（本应排除）
        ("szA", "2020-12-31", 0.01),
        ("szA", "2021-01-04", 0.06),   # 跨年但相邻交易日 -> 不是复牌
        ("szA", "2021-01-05", 0.07),
        # B：首行不算复牌
        ("szB", "2020-01-03", 0.09),
        ("szB", "2020-01-06", 0.02),
        # C：跨年复牌，且该月只有复牌日（n_max=0）
        ("szC", "2020-12-30", 0.02),
        ("szC", "2021-01-04", 0.03),   # 12-31 缺行 -> gap=2，跨年复牌
        # D：并列 max，用于 tie rule
        ("szD", "2020-01-02", 0.05),
        ("szD", "2020-01-03", 0.05),
    ]
    frame = pd.DataFrame(rows, columns=["code", "date", "rraw"])
    flagged, _, _ = mod.compute_resume_flags(frame, idx)
    flagged = flagged.sort_values(["code", "date"]).reset_index(drop=True)
    got = dict(zip(zip(flagged["code"], flagged["date"]), flagged["resume_flag"]))
    assert got[("szA", "2020-01-02")] == 0          # 首行
    assert got[("szA", "2020-01-03")] == 0
    assert got[("szA", "2020-01-06")] == 0          # 周末不是复牌
    assert got[("szA", "2020-01-07")] == 0
    assert got[("szA", "2020-02-03")] == 1          # 跨月停牌
    assert got[("szA", "2020-02-04")] == 0
    assert got[("szA", "2020-12-30")] == 1
    assert got[("szA", "2020-12-31")] == 0
    assert got[("szA", "2021-01-04")] == 0          # 跨年但相邻交易日
    assert got[("szB", "2020-01-03")] == 0          # 首行
    assert got[("szC", "2021-01-04")] == 1          # 跨年复牌

    flagged["month"] = flagged["date"].str[:7]
    agg = mod.aggregate_year_months(flagged, "earliest").set_index(
        ["code", "month"])
    assert agg.loc[("szA", "2020-01"), "n_valid_days_p2shadow"] == 3
    assert agg.loc[("szA", "2020-01"), "n_resume_days"] == 0
    assert agg.loc[("szA", "2020-01"), "MAXraw_p2shadow_ungated"] == pytest.approx(0.03)
    assert agg.loc[("szA", "2020-02"), "n_resume_days"] == 1
    assert agg.loc[("szA", "2020-02"), "MAXraw_all"] == pytest.approx(0.50)
    assert agg.loc[("szA", "2020-02"), "MAXraw_p2shadow_ungated"] == pytest.approx(0.04)
    assert agg.loc[("szA", "2020-02"), "argmax_date_p2shadow_ungated"] == "2020-02-04"
    assert agg.loc[("szA", "2020-12"), "n_max_days"] == 1
    assert agg.loc[("szA", "2020-12"), "MAXraw_all"] == pytest.approx(0.10)
    assert agg.loc[("szA", "2020-12"), "MAXraw_p2shadow_ungated"] == pytest.approx(0.01)
    assert agg.loc[("szA", "2021-01"), "n_resume_days"] == 0
    assert agg.loc[("szA", "2021-01"), "MAXraw_p2shadow_ungated"] == pytest.approx(0.07)
    assert agg.loc[("szC", "2021-01"), "n_max_days"] == 0
    assert pd.isna(agg.loc[("szC", "2021-01"), "MAXraw_p2shadow_ungated"])
    assert agg.loc[("szC", "2021-01"), "has_cross_year_resume"] == 1

    agg_last = mod.aggregate_year_months(flagged, "latest").set_index(
        ["code", "month"])
    assert agg_last.loc[("szD", "2020-01"), "argmax_all"] == "2020-01-03"
    assert agg.loc[("szD", "2020-01"), "argmax_all"] == "2020-01-02"


def test_synthetic_prior_state_carries_across_year(mod):
    cal = ["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05"]
    idx = {d: i for i, d in enumerate(cal)}
    # 2021 首行：prior state 在 2020-12-30，gap=2 -> 复牌
    frame = pd.DataFrame({
        "code": ["szA", "szA"],
        "date": ["2021-01-04", "2021-01-05"],
        "rraw": [0.10, 0.01],
    })
    flagged, state_ci, state_date = mod.compute_resume_flags(
        frame, idx, prior_ci={"szA": 0}, prior_date={"szA": "2020-12-30"})
    assert flagged["resume_flag"].tolist() == [1, 0]
    assert state_date["szA"] == "2021-01-05"
    # 首行无 prior -> 不算复牌
    flagged2, _, _ = mod.compute_resume_flags(frame, idx)
    assert flagged2["resume_flag"].tolist() == [0, 0]


# ---------------------------------------------------------------- 2. 官方 limit
def test_official_limit_flag_vs_threshold(mod, tmp_path):
    up = pd.DataFrame(
        {"000001.SZ": [1.0, 0.0], "688001.SH": [0.0, 1.0]},
        index=["2020-01-06", "2020-01-07"])
    dn = pd.DataFrame(
        {"000001.SZ": [0.0, 0.0], "688001.SH": [0.0, 0.0]},
        index=["2020-01-06", "2020-01-07"])
    p = tmp_path / "limit_up.pkl"
    up.to_pickle(p)
    panel = mod.load_bool_panel(p)
    vals = mod.lookup_panel(
        panel,
        pd.Series(["2020-01-06", "2020-01-07"]),
        pd.Series(["000001.SZ", "000001.SZ"]))
    assert vals.tolist() == [1.0, 0.0]

    # 官方 hit 但 10% 阈值未 hit；官方 no hit 但阈值 hit（如无涨跌幅限制日）
    assert mod.threshold_limit_hit("sz000001", 0.03, False) == (0, False)
    assert mod.threshold_limit_hit("sz000001", 0.15, False) == (1, False)
    assert mod.threshold_limit_hit("sz000001", 0.15, True) == (1, False)
    assert mod.threshold_limit_hit("sz000001", 0.05, True) == (1, False)
    assert mod.threshold_limit_hit("sh688001", 0.18, None) == (0, False)
    assert mod.threshold_limit_hit("sh688001", 0.20, None) == (1, False)
    # 无 ST 覆盖且落在 5%–9.5% -> legacy unknown zone
    assert mod.threshold_limit_hit("sz000001", 0.06, np.nan) == (0, True)


# ---------------------------------------------------------------- 3. 只读/键/列
def test_source_panel_rows_and_sha(counts):
    contracts = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    reg = contracts["m1_inputs"]["source_month_argmax_panel"]
    assert file_size(SOURCE) == reg["bytes"]
    assert int(reg["expected_rows"]) == EXPECTED_ROWS
    assert counts["headline_source_panel_sha_match_registered"] is True
    assert counts["inputs"]["source_month_argmax_panel"]["bytes_before"] == HISTORICAL_SOURCE_SHA
    assert counts["inputs"]["source_month_argmax_panel"]["bytes_after"] == HISTORICAL_SOURCE_SHA


def test_shadow_rows_keys_and_appended_columns():
    header = pd.read_csv(SHADOW, nrows=0).columns.tolist()
    assert header == OLD_SOURCE_HEADER + NEW_COLS
    keys = pd.read_csv(SHADOW, usecols=["source_id", "month"],
                       dtype=str, keep_default_na=False)
    assert len(keys) == EXPECTED_ROWS
    assert not keys.duplicated(["source_id", "month"]).any()
    assert keys["month"].min() == "2006-01"
    assert keys["month"].max() == "2025-12"


def test_old_columns_identical_and_row_order_preserved():
    # P1.1-01：source 的 n_valid_days=volume>0；shadow 旧列 n_valid_days=rraw 非空日数
    # （= 新 schema n_rraw_days）。其余旧列逐值/行序一致。
    cols = [c for c in OLD_SOURCE_HEADER if c != "n_valid_days"]
    src = pd.read_csv(SOURCE, dtype=str, keep_default_na=False,
                      usecols=OLD_SOURCE_HEADER + ["n_rraw_days"])
    src["n_valid_days"] = src["n_rraw_days"]
    src = src[OLD_SOURCE_HEADER]
    n = 0
    it_sh = pd.read_csv(SHADOW, dtype=str, keep_default_na=False,
                        usecols=OLD_SOURCE_HEADER, chunksize=50_000)
    for b in it_sh:
        a = src.iloc[n:n + len(b)].reset_index(drop=True)
        n += len(b)
        assert a.equals(b.reset_index(drop=True)), f"旧列/行序在第 {n - len(b) + 1} 行附近变化"
        assert n <= EXPECTED_ROWS
    assert n == EXPECTED_ROWS


# ---------------------------------------------------------------- 4. 复牌/窗口计数
def test_resume_counts_and_windows(counts, resume_audit):
    assert counts["resume_rows_2014_2025"] == 13_986
    assert len(resume_audit) == 13_986
    gap = num(resume_audit["gap_td"])
    assert gap.notna().all() and (gap > 1).all()
    assert not resume_audit.duplicated(["source_id", "date"]).any()
    assert int((resume_audit["is_argmax_legacy"] == "1").sum()) == 3_637
    assert int((resume_audit["cross_year"] == "1").sum()) == 1_066
    assert set(resume_audit["limit_status"].unique()) <= {
        "limit_up", "limit_down", "no_limit"}

    cal = counts["calendar_2015_2023"]
    assert cal["rows"] == 401_776
    assert cal["argmax_resume"] == 2_886
    assert cal["max_changed"] == 2_666
    assert cal["limit_hit_at_resume_legacy"] == 1_724

    m1 = counts["m1_window_2015_05_2023_12"]
    assert m1["rows"] == 391_849
    assert m1["argmax_resume"] == 2_601
    assert m1["max_changed"] == 2_396
    assert m1["limit_hit_at_resume_legacy"] == 1_528


def test_gate_counts_match_plan(counts):
    m1 = counts["gate_counts_m1_window"]
    assert m1["source_n_valid_lt_15"] == 7_848
    assert m1["monthly_matchable_n_valid_lt_15"] == 5_964
    assert m1["source_n_max_lt_15"] == 8_576
    assert m1["monthly_matchable_n_max_lt_15"] == 6_692
    assert m1["source_n_max_lt_10"] == 4_439
    assert m1["monthly_matchable_n_max_lt_10"] == 3_189
    rep = counts["plan_2_0_replication"]
    for k in ("gate_m1_source_n_valid_lt_15", "gate_m1_matchable_n_valid_lt_15",
              "gate_m1_source_n_max_lt_15", "gate_m1_matchable_n_max_lt_15"):
        assert rep[k]["match"] is True


# ---------------------------------------------------------------- 5. 尾部/变化
def test_max_tail_and_plan_deviation(counts):
    cal = counts["calendar_2015_2023"]
    assert cal["tail_raw"]["max"] == pytest.approx(5.93717277)
    assert cal["tail_raw"]["gt_0_44"] == 19
    assert cal["tail_raw"]["gt_1_0"] == 6
    assert cal["tail_shadow_ungated"]["max"] == pytest.approx(1.08544087)
    assert cal["tail_shadow_ungated"]["gt_0_44"] == 14
    assert cal["tail_shadow_ungated"]["gt_1_0"] == 2
    assert cal["max_shadow_ungated_row"]["source_id"] == "sh688026"
    assert cal["max_shadow_ungated_row"]["month"] == "2020-02"
    assert cal["max_raw_row"]["source_id"] == "sz002506"
    assert cal["max_raw_row"]["month"] == "2015-08"
    # 2.0_plan 的 17/3 不可复现：严格口径是 14/2；计数文件必须显式记录偏差。
    rep = counts["plan_2_0_replication"]
    assert rep["tail_shadow_gt_0_44"]["actual"] == 14
    assert rep["tail_shadow_gt_1_0"]["actual"] == 2
    assert rep["tail_shadow_gt_0_44"]["match"] is False
    assert rep["tail_shadow_gt_1_0"]["match"] is False
    assert "17" in rep["tail_shadow_gt_0_44"]["note"]
    assert counts["plan_2_0_all_match"] is False


def test_non_resume_months_zero_change_and_shadow_le_legacy(shadow):
    covered = shadow["shadow_status"].eq("evaluable")
    legacy = num(shadow["MAXraw"])
    sh = num(shadow["MAXraw_p2shadow_ungated"])
    nr = num(shadow["n_resume_days"])
    zero = covered & nr.eq(0) & sh.notna()
    assert int((np.abs(sh[zero] - legacy[zero]) > 1e-12).sum()) == 0
    assert int((covered & (sh > legacy + 1e-9)).sum()) == 0
    # 覆盖行 n_valid 必须与 headline 一致
    nv = num(shadow["n_valid_days_p2shadow"])
    nvl = num(shadow["n_valid_days"])
    assert int((covered & (nv != nvl)).sum()) == 0


def test_pre_2014_passthrough(shadow):
    pre = shadow["shadow_status"].eq("pre_2014_no_daily_coverage")
    assert int(pre.sum()) == 180_369
    legacy = num(shadow["MAXraw"])
    sh = num(shadow["MAXraw_p2shadow"])
    assert int((np.abs(sh[pre] - legacy[pre]) > 1e-12).sum()) == 0
    assert shadow.loc[pre, "n_resume_days"].eq("").all()
    assert shadow.loc[pre, "n_max_days"].eq("").all()
    assert shadow.loc[pre, "formula_id_p2shadow"].eq(
        "MAXRAW_P2SHADOW_V1_PRE2014_PASSTHROUGH").all()
    assert shadow.loc[pre, "limit_source"].eq("official_daily_flag").all()
    assert shadow.loc[pre, "R_p2shadow"].equals(shadow.loc[pre, "R"])


def test_r_p2shadow_is_original_r(shadow):
    assert shadow["R_p2shadow"].equals(shadow["R"])


# ---------------------------------------------------------------- 6. 官方 limit 使用
def test_official_limit_flags_are_used(counts, shadow):
    assert counts["limit_source"] == "official_daily_flag"
    cal = counts["limit_flags_calendar_2015_2023"]
    assert cal["legacy_argmax_official_available"] == 401_776
    assert cal["legacy_argmax_official_vs_headline_threshold"]["official_missing"] == 0
    assert cal["shadow_argmax_official_vs_threshold"]["official_missing"] == 0
    assert cal["limit_source_distribution"] == {
        "official_daily_flag": 393_147, "not_applicable": 8_629}
    # 复牌日统一改标（不进入 qualified）；官方细分状态另存
    rr = cal["resume_argmax"]
    assert rr["rows"] == 2_886
    assert rr["legacy_threshold_hit"] == 1_724
    assert rr["status"] == {"limit_status_unknown": 2_886}
    assert rr["official_status"] == {
        "limit_up": 2_116, "no_limit": 683, "limit_down": 87}
    # 官方 flag 与阈值口径确有差异
    cmp = cal["legacy_argmax_official_vs_headline_threshold"]
    assert cmp["both_hit"] == 66_504
    assert cmp["both_no_hit"] == 320_839
    assert cmp["official_hit_threshold_no"] == 11_699
    assert cmp["official_no_threshold_hit"] == 2_734

    d = shadow.set_index(["source_id", "month"])
    # sh688026 2020-02：legacy 阈值 hit=1，官方 flag=0（科创板 IPO 前 5 日无涨跌幅限制）
    row = d.loc[("sh688026", "2020-02")]
    assert row["limit_hit_at_argmax"] == "1"
    assert row["limit_hit_p2shadow"] == "0"
    assert row["limit_source"] == "official_daily_flag"
    # sz000792 2021-08：复牌日按 unknown 改标；shadow argmax 为非复牌日官方 hit=1
    row = d.loc[("sz000792", "2021-08")]
    assert row["resume_at_argmax"] == "1"
    assert row["limit_status_at_resume"] == "limit_status_unknown"
    assert row["limit_official_status_at_resume"] == "no_limit"
    assert row["limit_hit_p2shadow"] == "1"
    # sz002506 2015-08：复牌日 +594%，official up=1，但 15 日门失败 -> shadow 空
    row = d.loc[("sz002506", "2015-08")]
    assert row["MAXraw_p2shadow"] == ""
    assert row["MAXraw_p2shadow_ungated"] == "0.10037736"
    assert row["max_gate_15"] == "0"
    assert row["gate_reason"] == "insufficient_trade_days"
    assert row["limit_hit_p2shadow"] == ""


def test_gate_blanking_and_reasons(shadow):
    gate0 = shadow["max_gate_15"].eq("0")
    assert int(gate0.sum()) == 10_459
    assert shadow.loc[gate0, "MAXraw_p2shadow"].eq("").all()
    assert shadow.loc[gate0, "limit_hit_p2shadow"].eq("").all()
    assert set(shadow.loc[gate0, "gate_reason"].unique()) == {
        "insufficient_trade_days"}
    gate1 = shadow["max_gate_15"].eq("1")
    assert int(gate1.sum()) == 543_404
    assert shadow.loc[gate1, "MAXraw_p2shadow"].ne("").all()
    assert shadow.loc[gate1, "limit_hit_p2shadow"].isin(["0", "1"]).all()
    pre = shadow["shadow_status"].eq("pre_2014_no_daily_coverage")
    assert shadow.loc[pre, "max_gate_15"].eq("").all()
    assert shadow.loc[pre, "gate_reason"].eq("not_evaluable_pre_2014").all()


def test_shadow_argmax_never_on_resume_day(shadow, resume_audit):
    pairs = set(zip(resume_audit["source_id"], resume_audit["date"]))
    arg = shadow["argmax_date_p2shadow_ungated"]
    hit = sum(1 for c, d in zip(shadow["source_id"], arg)
              if d and (c, d) in pairs)
    assert hit == 0
    # 复牌标记与 gap 一致：resume_at_argmax=1 -> gap_td>1
    r = num(shadow["resume_at_argmax"]).eq(1)
    assert int((r & ~(num(shadow["gap_td_at_argmax"]) > 1)).sum()) == 0


# ---------------------------------------------------------------- 7. 产物/锚
def test_audit_files_and_meta(meta, counts):
    assert len(pd.read_csv(RESUME_AUDIT)) == 13_986
    assert TAIL.is_file() and len(pd.read_csv(TAIL)) > 0
    assert meta["output"]["bytes"] == file_size(SHADOW)
    assert meta["output"]["rows"] == EXPECTED_ROWS
    assert meta["script_bytes"] == file_size(SCRIPT)
    assert counts["script_bytes"] == meta["script_bytes"]
    assert meta["headline_anchors_match"] is True
    assert counts["headline_files_modified"] is False
    assert counts["m1_rerun"] is False
    assert counts["configs_modified"] is False


def test_headline_anchors_unchanged():
    contracts = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    for key, reg in contracts["m1_inputs"].items():
        p = ROOT / reg["file"]
        assert p.is_file(), f"缺锚: {key}"
        assert file_size(p) == reg["bytes"], f"锚变化: {key}"
    # 9 个锚快照在 counts 中前后一致
    counts = json.loads(COUNTS.read_text(encoding="utf-8"))
    before = counts["headline_anchors_before"]
    after = counts["headline_anchors_after"]
    assert set(before) == set(after) and len(before) == 9
    assert all(before[k]["bytes"] == after[k]["bytes"] for k in before)
    assert all(before[k]["sha_match"] for k in before)


def test_shadow_paths_are_not_headline(mod):
    contracts = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    protected = {(ROOT / m["file"]).resolve() for m in contracts["m1_inputs"].values()}
    outputs = {mod.OUT.resolve(), mod.META_OUT.resolve(),
               mod.COUNTS_OUT.resolve(), mod.RESUME_AUDIT_OUT.resolve(),
               mod.TAIL_OUT.resolve()}
    assert not (outputs & protected)
    assert mod.OUT.resolve().parent == (ROOT / "data/processed/phase2_shadow").resolve()
    assert mod.META_OUT.resolve().parent == mod.OUT.resolve().parent
    for p in (mod.COUNTS_OUT, mod.RESUME_AUDIT_OUT, mod.TAIL_OUT):
        assert p.resolve().parent == (ROOT / "artifacts/phase2").resolve()
