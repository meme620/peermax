"""Phase 2.3 复权/除权 shadow MAX 契约测试（加法，只读真实产物）。

覆盖：
  1. 合成拆股/送转/派息：raw 跳空、adj 无跳空；`MAXraw_adj` 与
     `MAXraw_ex_rights` 语义。
  2. sharebonus 事件解析（只保留 `EX_DIVIDEND_DATE` 非空 + 实施分配）；
     corporate_action 交叉；sharebonus 回退公式。
  3. 复牌 + 复权联合：`MAXraw_adj` 含复牌跳空，`MAXraw_adj_ex_resume` 排除；
     argmax 不落在复牌日。
  4. 真实面板只读：734,232 行、键唯一、旧 11 列逐字段一致、行序不变。
  5. 复权覆盖：2014–2025 553,863 行全部有 adj；pre-2014 180,369 行 legacy
     直通、`MAXraw_adj` 为空。
  6. 尾部：raw `<-0.44` 12 → adj 4（全部 n_max_days=0）→ adj_ex_resume 0；
     raw `>0.44` 19 → adj 21；raw `>1.0` 6 → adj 7；adj_ex_resume max 1.0854；
     `sz002506` 2015-08 5.937→9.860→0.1004。
  7. 2.2 联合：`MAXraw_ex_resume` 与 2.2 一致（max_changed 2,666 / 2,396）。
  8. 事件覆盖：sharebonus 2014–2017 8,580；2018–2025 corporate_action
     28,978，与 sharebonus 100% 匹配。
  9. headline 9 输入 sha 与 `m1_contracts.yaml` 一致；headline source panel sha
     未变；shadow 路径不等于 headline；monthly adj 文件为逐字节复制。

运行：python -m pytest tests/test_shadow_adjusted_max_contracts.py -q
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
SCRIPT = ROOT / "scripts/shadow_adjusted_max_build.py"
SHADOW = ROOT / "data/processed/phase2_shadow/source_month_argmax_panel_adj_p2shadow.csv"
META = ROOT / "data/processed/phase2_shadow/source_month_argmax_panel_adj_p2shadow.meta.json"
MONTHLY_SHADOW = ROOT / "data/processed/phase2_shadow/monthly_returns_adj_p2shadow.csv"
MONTHLY_ADJ = ROOT / "data/processed/monthly_returns_adj_full_2006_2025.csv"
COUNTS = ROOT / "artifacts/phase2/2.3_counts.json"
DIFF = ROOT / "artifacts/phase2/2.3_adj_diff.csv"
EXRIGHTS = ROOT / "artifacts/phase2/2.3_exrights_audit.csv"
RESUME_AUDIT = ROOT / "artifacts/phase2/2.2_resume_audit.csv"
SOURCE = ROOT / "data/processed/source_month_argmax_panel.csv"
CONTRACTS = ROOT / "configs/m1_contracts.yaml"

EXPECTED_ROWS = 734_232
EXPECTED_SOURCE_SHA = "9ef16943104b5556df317da2aa66f0ff1cb782e7ac8ac16e1edbd60b0e1a9b9a"
HISTORICAL_SOURCE_SHA = "944adbbab95a7b30d858f2943c8ec2844a852a9551b06224b21147d3c4c6023a"
OLD_COLS = ["source_id", "month", "MAXraw", "argmax_date", "n_valid_days",
            "R", "tie_rule_version", "formula_id", "limit_hit_at_argmax",
            "qualification_status", "st_flag_at_argmax"]
CORE_COLS = [
    "source_id", "month", "MAXraw", "argmax_date", "n_valid_days",
    "MAXraw_adj", "argmax_date_adj", "MAXraw_adj_ex_resume",
    "argmax_date_adj_ex_resume", "MAXraw_ex_resume", "MAXraw_ex_rights",
    "n_resume_days", "n_max_days", "corporate_action_flag", "ca_source",
    "n_ca_events", "argmax_raw_on_ex_date", "adj_status",
    "shadow_status_adj", "resume_joint_max_changed", "adj_max_changed_vs_raw",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_adjusted_max_build", SCRIPT)
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
def shadow() -> pd.DataFrame:
    assert SHADOW.is_file(), f"缺少 shadow 产物（先运行 {SCRIPT.name}）"
    return pd.read_csv(SHADOW, dtype=str, keep_default_na=False,
                       usecols=CORE_COLS, low_memory=False)


# ---------------------------------------------------------------- 1. 合成复权
def test_synthetic_split_bonus_dividend_adj_removes_jump(mod):
    # 10 拆 2：raw 10 -> 5（-50%）；后复权价 20 -> 20（0%）。
    px = pd.DataFrame({"600000.SH": [20.0, 20.0]},
                      index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    adj_ret = mod.build_adj_return_matrix(px).iloc[-1, 0]
    raw_ret = pd.Series([10.0, 5.0]).pct_change().iloc[-1]
    assert raw_ret == pytest.approx(-0.5)
    assert adj_ret == pytest.approx(0.0, abs=1e-12)

    d = pd.DataFrame({
        "code": ["sh600000"], "date": ["2020-01-03"],
        "month": ["2020-01"], "rraw": [raw_ret], "r_adj": [adj_ret],
        "resume_flag": [0], "ex_flag": [1], "adj_fallback": [0],
    })
    agg = mod.aggregate_daily_year(d, "earliest")
    assert agg.loc[0, "MAXraw"] == pytest.approx(-0.5)
    assert agg.loc[0, "MAXraw_adj"] == pytest.approx(0.0, abs=1e-12)

    # 10 派 0.5 元：raw 10 -> 9.5（-5%）；后复权 10 -> 10（0%）。
    px = pd.DataFrame({"600000.SH": [10.0, 10.0]},
                      index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    adj_ret = mod.build_adj_return_matrix(px).iloc[-1, 0]
    raw_ret = pd.Series([10.0, 9.5]).pct_change().iloc[-1]
    assert raw_ret == pytest.approx(-0.05)
    assert adj_ret == pytest.approx(0.0, abs=1e-12)

    # 10 送 10：raw 10 -> 5（-50%）；后复权 10 -> 10（0%）。
    px = pd.DataFrame({"600000.SH": [10.0, 10.0]},
                      index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    adj_ret = mod.build_adj_return_matrix(px).iloc[-1, 0]
    assert adj_ret == pytest.approx(0.0, abs=1e-12)


def test_synthetic_exrights_exclusion(mod):
    # 除权日 -5%，非除权日 -1%；排除除权日后 MAX 应为 -1%。
    d = pd.DataFrame({
        "code": ["sh600000", "sh600000"], "date": ["2020-01-02", "2020-01-03"],
        "month": ["2020-01", "2020-01"], "rraw": [-0.01, -0.05],
        "r_adj": [-0.01, 0.0], "resume_flag": [0, 0],
        "ex_flag": [0, 1], "adj_fallback": [0, 0],
    })
    agg = mod.aggregate_daily_year(d, "earliest")
    assert agg.loc[0, "MAXraw"] == pytest.approx(-0.01)
    assert agg.loc[0, "MAXraw_ex_rights"] == pytest.approx(-0.01)
    assert agg.loc[0, "n_ex_rights_days"] == 1
    assert agg.loc[0, "MAXraw_adj"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------- 2. 事件
def test_sharebonus_event_parsing_and_crosscheck(mod):
    sb = pd.DataFrame({
        "SECUCODE": ["600000.SH", "600000.SH", "600000.SH", "000001.SZ"],
        "EX_DIVIDEND_DATE": ["2020-01-03 00:00:00", "", "2020-01-04 00:00:00",
                             "2020-06-10 00:00:00"],
        "ASSIGN_PROGRESS": ["实施分配", "实施分配", "取消分配", "实施分配"],
        "BONUS_RATIO": ["10", "", "10", "0"],
        "IT_RATIO": ["", "", "", "5"],
        "PRETAX_BONUS_RMB": ["1", "", "1", "2"],
    })
    ev = mod.parse_sharebonus_frame(sb)
    assert len(ev) == 2
    assert set(ev["source_id"]) == {"sh600000", "sz000001"}
    assert ev.loc[ev.source_id.eq("sh600000"), "ex_date"].iloc[0] == "2020-01-03"
    assert ev.loc[ev.source_id.eq("sh600000"), "month"].iloc[0] == "2020-01"
    assert ev.loc[ev.source_id.eq("sh600000"), "bonus_ratio"].iloc[0] == 10

    ca = pd.DataFrame({"code": ["600000"], "ex_date": ["2020-01-03"],
                       "total_shares": ["1"], "bonus_ratio": [""],
                       "it_ratio": [""]})
    ca_ev = mod.parse_corporate_action_frame(ca, {"600000": "sh600000"})
    assert ca_ev["source_id"].tolist() == ["sh600000"]
    combined, month = mod.combine_events(ev, ca_ev)
    assert len(combined) == 2
    row = month[(month.source_id == "sh600000") & (month.month == "2020-01")].iloc[0]
    assert row["ca_source"] == "both"
    assert row["n_ca_events"] == 1


def test_sharebonus_fallback_math(mod):
    # 10 送 10 + 10 派 1：raw -50% → 复权 +1%。
    v = mod.fallback_adj_return(-0.5, 10.0, 10.0, 0.0, 1.0)
    assert v == pytest.approx(0.01, abs=1e-12)
    # 纯派息 5 元/10 股（0.5 元/股）：raw -5% → 复权 0%。
    assert mod.fallback_adj_return(-0.05, 10.0, 0.0, 0.0, 5.0) == pytest.approx(
        0.0, abs=1e-12)
    # 纯转增 10：raw -50% → 复权 0%。
    assert mod.fallback_adj_return(-0.5, 10.0, 0.0, 10.0, 0.0) == pytest.approx(
        0.0, abs=1e-12)
    assert np.isnan(mod.fallback_adj_return(-0.5, 0.0, 10.0, 0.0, 0.0))


# ---------------------------------------------------------------- 3. 复牌联合
def test_resume_and_adj_joint_exclusion(mod):
    d = pd.DataFrame({
        "code": ["sh600000"] * 3,
        "date": ["2020-01-02", "2020-01-03", "2020-01-06"],
        "month": ["2020-01"] * 3,
        "rraw": [0.0, 5.93717277, 0.02],
        "r_adj": [0.0, 9.86072325, 0.02],
        "resume_flag": [0, 1, 0],
        "ex_flag": [0, 0, 0],
        "adj_fallback": [0, 0, 0],
    })
    agg = mod.aggregate_daily_year(d, "earliest")
    assert agg.loc[0, "MAXraw_adj"] == pytest.approx(9.86072325)
    assert agg.loc[0, "MAXraw_adj_ex_resume"] == pytest.approx(0.02)
    assert agg.loc[0, "argmax_date_adj"] == "2020-01-03"
    assert agg.loc[0, "argmax_date_adj_ex_resume"] == "2020-01-06"
    assert agg.loc[0, "n_resume_days"] == 1
    assert agg.loc[0, "n_valid_days"] - agg.loc[0, "n_resume_days"] == 2

    # tie rule earliest：并列最大值取最早日期。
    d2 = d.iloc[[0, 2]].copy()
    d2["r_adj"] = [0.05, 0.05]
    agg2 = mod.aggregate_daily_year(d2, "earliest")
    assert agg2.loc[0, "argmax_date_adj"] == "2020-01-02"


# ---------------------------------------------------------------- 4. 真实面板
def test_real_rows_keys_and_old_columns(shadow):
    assert len(shadow) == EXPECTED_ROWS
    assert not shadow.duplicated(["source_id", "month"]).any()
    assert shadow["month"].min() == "2006-01"
    assert shadow["month"].max() == "2025-12"

    # P1.1-01 后 source 的 n_valid_days = volume>0；shadow 的旧列 n_valid_days = rraw 非空日数，
    # 对应新 schema 的 n_rraw_days。其余旧列逐字段/行序仍必须一致。
    source = pd.read_csv(SOURCE, dtype=str, keep_default_na=False,
                         usecols=OLD_COLS + ["n_rraw_days"])
    source["n_valid_days"] = source["n_rraw_days"]
    source = source[OLD_COLS]
    offset = 0
    for chunk in pd.read_csv(SHADOW, usecols=OLD_COLS, dtype=str,
                             keep_default_na=False, chunksize=200_000):
        expected = source.iloc[offset:offset + len(chunk)][OLD_COLS] \
            .reset_index(drop=True)
        got = chunk[OLD_COLS].reset_index(drop=True)
        assert got.equals(expected), f"旧列在 offset={offset} 不一致"
        offset += len(chunk)
    assert offset == EXPECTED_ROWS


def test_real_adj_coverage_and_invariants(shadow):
    n_valid = num(shadow["n_valid_days"])
    n_max = num(shadow["n_max_days"])
    adj = num(shadow["MAXraw_adj"])
    adj_nr = num(shadow["MAXraw_adj_ex_resume"])
    raw = num(shadow["MAXraw"])
    raw_nr = num(shadow["MAXraw_ex_resume"])
    raw_nx = num(shadow["MAXraw_ex_rights"])
    pre = shadow["month"] < "2014-01"
    covered = ~pre

    assert int(covered.sum()) == 553_863
    assert int(pre.sum()) == 180_369
    assert int(adj.notna().sum()) == 553_863
    assert int(adj[pre].notna().sum()) == 0
    assert shadow.loc[covered, "adj_status"].eq("ok_daily_adj").all()
    assert shadow.loc[pre, "adj_status"].eq("pre_2014_no_daily_coverage").all()
    assert shadow.loc[pre, "shadow_status_adj"].eq("pre_2014_no_daily_coverage").all()

    # 旧列 n_valid_days 是字符串；复权日数应等于有效日数。
    assert int(num(shadow.loc[covered, "n_valid_days"]).notna().sum()) == 553_863
    assert n_max[pre].isna().all()
    assert n_max[covered].notna().all()
    assert int(num(shadow.loc[covered, "n_resume_days"]).sum()) == 13_986
    assert (n_max[covered] <= n_valid[covered]).all()

    assert int((adj_nr.notna() & adj.notna() & (adj_nr > adj + 1e-12)).sum()) == 0
    assert int((raw_nx.notna() & raw.notna() & (raw_nx > raw + 1e-12)).sum()) == 0
    assert int((raw_nr.notna() & raw.notna() & (raw_nr > raw + 1e-12)).sum()) == 0
    assert (shadow.loc[covered, "argmax_date_adj"].str[:7]
            == shadow.loc[covered, "month"]).all()

    resume = pd.read_csv(RESUME_AUDIT, dtype=str, keep_default_na=False)
    resume_pairs = set(zip(resume["source_id"], resume["date"]))
    sub = shadow.loc[shadow["argmax_date_adj_ex_resume"].ne(""),
                     ["source_id", "argmax_date_adj_ex_resume"]]
    assert sum(1 for s, d in zip(sub["source_id"], sub["argmax_date_adj_ex_resume"])
               if (s, d) in resume_pairs) == 0


def test_real_tails_and_resume_joint(shadow, counts):
    w = shadow[shadow["month"].between("2015-01", "2023-12")]
    raw = num(w["MAXraw"])
    adj = num(w["MAXraw_adj"])
    adj_nr = num(w["MAXraw_adj_ex_resume"])
    raw_nr = num(w["MAXraw_ex_resume"])
    n_max = num(w["n_max_days"])

    assert int(raw.lt(-0.44).sum()) == 12
    assert int(raw.gt(0.44).sum()) == 19
    assert int(raw.gt(1.0).sum()) == 6
    assert int(adj.lt(-0.44).sum()) == 4
    assert int(adj[n_max > 0].lt(-0.44).sum()) == 0
    assert int(adj_nr.lt(-0.44).sum()) == 0
    assert int(adj.gt(0.44).sum()) == 21
    assert int(adj.gt(1.0).sum()) == 7
    assert int(adj_nr.gt(0.44).sum()) == 14
    assert adj.max() == pytest.approx(9.86072325, abs=1e-6)
    assert adj_nr.max() == pytest.approx(1.08544087, abs=1e-6)
    assert raw_nr.max() == pytest.approx(1.08544087, abs=1e-6)
    assert int(num(w["resume_joint_max_changed"]).sum()) == 2_666
    assert int(num(w["corporate_action_flag"]).sum()) == 24_252
    assert int(num(w["argmax_raw_on_ex_date"]).sum()) == 387

    row = shadow[(shadow.source_id == "sz002506") & (shadow.month == "2015-08")].iloc[0]
    assert float(row["MAXraw"]) == pytest.approx(5.93717277, abs=1e-8)
    assert float(row["MAXraw_adj"]) == pytest.approx(9.86072325, abs=1e-6)
    assert float(row["MAXraw_adj_ex_resume"]) == pytest.approx(0.10037681, abs=1e-6)
    assert float(row["n_resume_days"]) == 1

    # counts 与面板一致
    t = counts["tails"]["calendar_2015_2023"]
    assert t["raw"]["lt_neg_0_44"] == 12
    assert t["adj"]["lt_neg_0_44"] == 4
    assert t["adj_ex_resume"]["lt_neg_0_44"] == 0
    assert t["adj"]["gt_0_44"] == 21
    assert counts["resume_joint"]["max_changed_calendar_2015_2023"] == 2_666


# ---------------------------------------------------------------- 5. counts
def test_counts_contracts(counts):
    assert counts["rows"]["total"] == EXPECTED_ROWS
    assert counts["rows"]["daily_covered_2014_2025"] == 553_863
    assert counts["rows"]["pre_2014_passthrough"] == 180_369
    assert counts["rows"]["rows_with_adj_2015"] == 29_982
    assert counts["rows"]["rows_adj_missing_in_covered"] == 0

    rep = counts["monthly_adj_replication"]
    assert rep["rows_both"] == 844_952
    assert rep["max_abs_diff"] <= 1e-12
    assert rep["gt_1e_12"] == 0 and rep["gt_1e_6"] == 0

    ev = counts["events"]
    assert ev["sharebonus_ex_nonnull"] == 55_733
    assert ev["sharebonus_valid_implemented"] == 55_720
    assert ev["sharebonus_2014_2017"] == 8_580
    assert ev["sharebonus_2018_2025"] == 28_978
    cross = ev["crosscheck_2018_2025"]
    assert cross["matched"] == 28_978
    assert cross["corporate_action_only"] == 0
    assert cross["sharebonus_only"] == 0
    assert cross["match_rate_corporate_action_to_sharebonus"] == pytest.approx(1.0)

    cov = counts["panel_event_coverage"]
    assert cov["rows_with_any_ca"] == 44_066
    assert cov["ca_source_counts"].get("corporate_action", 0) == 0

    pre = counts["pre_2014"]
    assert pre["rows"] == 180_369
    assert pre["adj_computable"] == 0
    assert pre["legacy_passthrough"] == 180_369
    assert pre["monthly_adj_available"] == 179_287

    checks = counts["checks"]
    for key in ("key_unique", "old_columns_readonly_equal",
                "max_adj_ex_resume_le_max_adj", "max_ex_rights_le_max_raw",
                "max_ex_resume_le_max_raw", "argmax_adj_in_month",
                "adj_neg_tail_evaluable_zero", "adj_ex_resume_neg_tail_zero",
                "inputs_unchanged"):
        assert checks[key] is True, key
    assert checks["n_valid_mismatch_vs_headline"] == 0
    assert checks["max_raw_mismatch_vs_headline"] == 0
    assert checks["resume_count_mismatch_vs_2_2"] == 0
    assert checks["adj_argmax_on_resume_day"] == 0
    assert checks["n_max_days_from_2_2_resume"]["m1_not_gated_lt15"] == 8_576
    assert checks["n_max_days_from_2_2_resume"]["m1_n_valid_lt15"] == 7_848

    rep2 = counts["plan_2_0_replication"]
    assert rep2["raw_lt_neg_0_44_calendar"]["match"] is True
    assert rep2["raw_gt_0_44_calendar"]["match"] is True
    assert rep2["raw_gt_1_0_calendar"]["match"] is True
    assert rep2["resume_max_changed_calendar"]["match"] is True
    assert rep2["resume_max_changed_m1"]["match"] is True
    # 已知偏差：adj 单独负尾部 4（全部 n_max_days=0）；adj >0.44 为 21 不是 22。
    assert rep2["adj_lt_neg_0_44_calendar"]["actual"] == 4
    assert rep2["adj_gt_0_44_calendar"]["actual"] == 21
    assert "n_max_days=0" in rep2["adj_lt_neg_0_44_calendar"]["note"]
    assert rep2["monthly_R_sign_flips_2015_2023"]["actual"] == 1_658

    mr = counts["monthly_R_raw_vs_adj"]["calendar_2015_2023"]
    assert mr["rows"] == 408_056
    assert mr["abs_diff_gt_1e_6"] == 157_098
    assert mr["sign_flips_abs_raw_gt_0_2"] == 1_658
    assert mr["abs_raw_gt_0_44"] == 6_782
    assert mr["abs_adj_gt_0_44"] == 5_754


# ---------------------------------------------------------------- 6. 治理
def test_headline_sha_paths_and_provenance(counts):
    assert sha256_file(SOURCE) == EXPECTED_SOURCE_SHA

    contracts = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    for key, rec in contracts["m1_inputs"].items():
        p = ROOT / rec["file"]
        assert p.is_file(), key
        assert sha256_file(p) == rec["sha256"], key

    for p in (SHADOW, META, MONTHLY_SHADOW, COUNTS, DIFF, EXRIGHTS):
        assert p.is_file(), p
        assert p.resolve() not in {
            (ROOT / r["file"]).resolve()
            for r in contracts["m1_inputs"].values()
        }

    assert sha256_file(MONTHLY_SHADOW) == sha256_file(MONTHLY_ADJ)
    assert counts["outputs"]["monthly_returns_adj_p2shadow"]["recomputed"] is False
    assert counts["headline_anchors_match"] is True
    assert counts["source_panel_sha_before"] == HISTORICAL_SOURCE_SHA
    assert counts["source_panel_sha_after"] == HISTORICAL_SOURCE_SHA
    assert counts["headline_files_modified"] is False
    assert counts["m1_rerun"] is False
    assert counts["configs_modified"] is False

    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["script_sha256"] == sha256_file(SCRIPT)
    assert meta["shadow_panel"]["sha256"] == sha256_file(SHADOW)
    assert meta["counts_file"]["sha256"] == sha256_file(COUNTS)


def test_output_path_guard_rejects_headline(mod):
    with pytest.raises(SystemExit):
        mod.main(["--out-max", str(SOURCE)])


# ---------------------------------------------------------------- 7. 审计表
def test_audit_tables(counts):
    diff = pd.read_csv(DIFF, dtype={"source_id": str, "month": str})
    assert len(diff) == counts["outputs"]["adj_diff"]["rows"]
    assert ((diff.source_id == "sz002506") & (diff.month == "2015-08")).any()
    assert num(diff["MAXraw"]).gt(0.44).any()
    assert num(diff["MAXraw"]).lt(-0.44).any()

    ex = pd.read_csv(EXRIGHTS, dtype={"source_id": str, "ex_date": str,
                                      "month": str})
    assert len(ex) == counts["outputs"]["exrights_audit"]["rows"]
    assert int(ex["is_sharebonus"].sum()) == len(ex)
    assert int(ex.loc[ex.ex_date.str[:4].between("2014", "2017"), :].shape[0]) == 8_580
    assert int(ex["in_corporate_action"].sum()) == 28_978
    assert int(ex["on_trading_day"].sum()) == \
        counts["events"]["combined_event_days_on_trading_day"]
    assert ex["source_id"].str.startswith(("sh", "sz", "bj")).all()
