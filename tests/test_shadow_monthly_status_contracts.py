"""Phase 2.1 shadow 月收益/状态面板契约测试（加法，只读真实产物）。

覆盖：
  1. 行数 856,771、键唯一、原 10 列 + 追加列。
  2. pre-2014 `no_volume_data`；2026 `out_of_daily_coverage`；
     `daily_panel_missing` 与 `no_trade` 分离。
  3. 2015-05..2023-12 修正值 13,980 / 8,014 / 5,966，伪零/残月/真零/缺失。
  4. `R_p2shadow` 仅在 `no_trade` 时为 NA；daily_panel_missing 保留原 R。
  5. 632 = 586（no_trade 非零 R）+ 46（sh601313 alias 恢复）；96 个
     daily_panel_missing 与 632 无交集。
  6. 合成整月停牌/残月/真实零/部分停牌缺失/alias 语义（tmp，不碰真实数据）。
  7. headline 9 输入 sha 与 `m1_contracts.yaml` 一致；shadow 路径不等于 headline。
  8. meta/counts/alias JSON 与实际产物、脚本一致。

运行：python -m pytest tests/test_shadow_monthly_status_contracts.py -q
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
SCRIPT = ROOT / "scripts/shadow_monthly_status.py"
SHADOW = ROOT / "data/processed/phase2_shadow/monthly_returns_status_p2shadow.csv"
META = ROOT / "data/processed/phase2_shadow/monthly_returns_status_p2shadow.meta.json"
ALIAS = ROOT / "artifacts/phase2/2.1_alias_map.json"
COUNTS = ROOT / "artifacts/phase2/2.1_counts.json"
RETURNS = ROOT / "data/processed/monthly_returns_clean_v2.csv"
CONTRACTS = ROOT / "configs/m1_contracts.yaml"
SUSPEND_FLAG = ROOT / "data/raw/drive/stock_suspend_flag_20260630.pkl"

EXPECTED_ROWS = 856_771
WINDOW = ("2015-05", "2023-12")
EVAL_STATUS = {"ok", "no_trade"}
NEW_COLS = [
    "n_price_obs", "n_trade_days", "trade_days_status", "trade_data_status",
    "complete_month_legacy", "complete_month_trade15", "complete_month_trade10",
    "pseudo_zero_flag", "partial_zero_flag", "true_zero_flag", "R_p2shadow",
    "p2shadow_version",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location("shadow_monthly_status", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def shadow() -> pd.DataFrame:
    assert SHADOW.is_file(), f"缺少 shadow 产物（先运行 {SCRIPT.name}）"
    return pd.read_csv(SHADOW, dtype={"firm_id": str, "month": str})


@pytest.fixture(scope="module")
def counts() -> dict:
    assert COUNTS.is_file(), f"缺少 counts JSON（先运行 {SCRIPT.name}）"
    return json.loads(COUNTS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. 行数/键/列
def test_rows_keys_and_appended_columns(shadow):
    assert len(shadow) == EXPECTED_ROWS
    assert not shadow.duplicated(["firm_id", "month"]).any()
    assert list(shadow.columns[:10]) == [
        "firm_id", "month", "R", "n_valid_days", "complete_month",
        "return_status", "month_start", "month_end", "sample_version", "available_at",
    ]
    assert list(shadow.columns[10:]) == NEW_COLS
    assert shadow["p2shadow_version"].eq("p2shadow_v1").all()
    assert shadow["trade_data_status"].fillna("NA").eq(
        shadow["trade_days_status"].fillna("NA")).all()
    assert shadow["month"].min() == "2000-02"
    assert shadow["month"].max() == "2026-06"
    assert set(shadow["trade_days_status"].unique()) == {
        "ok", "no_trade", "daily_panel_missing", "no_volume_data",
        "out_of_daily_coverage",
    }


def test_n_price_obs_and_complete_legacy_are_copies(shadow):
    np.testing.assert_array_equal(shadow["n_price_obs"].to_numpy(),
                                  shadow["n_valid_days"].to_numpy())
    np.testing.assert_array_equal(shadow["complete_month_legacy"].to_numpy(),
                                  shadow["complete_month"].to_numpy())


# ---------------------------------------------------------------- 2. 覆盖状态
def test_pre_2014_is_no_volume_data_not_zero(shadow):
    pre = shadow[shadow["month"] < "2014-01"]
    assert len(pre) == 264_134
    assert pre["trade_days_status"].eq("no_volume_data").all()
    assert pre["n_trade_days"].isna().all()
    assert pre["complete_month_trade15"].isna().all()
    assert pre["complete_month_trade10"].isna().all()
    assert pre[["pseudo_zero_flag", "partial_zero_flag", "true_zero_flag"]].isna().all().all()
    np.testing.assert_allclose(pre["R_p2shadow"].to_numpy(), pre["R"].to_numpy(),
                               rtol=0, atol=1e-15)
    assert not pre["trade_days_status"].isin(["no_trade", "ok",
                                              "daily_panel_missing"]).any()


def test_2026_is_out_of_daily_coverage(shadow):
    post = shadow[shadow["month"] >= "2026-01"]
    assert len(post) == 31_159
    assert post["trade_days_status"].eq("out_of_daily_coverage").all()
    assert post["n_trade_days"].isna().all()
    assert post[["pseudo_zero_flag", "partial_zero_flag", "true_zero_flag"]].isna().all().all()
    np.testing.assert_allclose(post["R_p2shadow"].to_numpy(), post["R"].to_numpy(),
                               rtol=0, atol=1e-15)


def test_evaluable_scope_partition(shadow):
    status = shadow["trade_days_status"]
    in_scope = status.isin(["ok", "no_trade", "daily_panel_missing"])
    ev = status.isin(EVAL_STATUS)
    assert int(in_scope.sum()) == 561_478
    assert int(ev.sum()) == 561_382
    assert int(status.eq("daily_panel_missing").sum()) == 96
    td = shadow["n_trade_days"]
    assert int(status.eq("no_trade").sum()) == 10_234
    assert int((status.eq("ok") & td.gt(0) & td.lt(15)).sum()) == 8_348
    assert int((status.eq("ok") & td.ge(15)).sum()) == 542_800
    assert int(ev.sum()) == 10_234 + 8_348 + 542_800
    dpm = shadow.loc[status.eq("daily_panel_missing")]
    assert dpm["n_trade_days"].isna().all()
    assert dpm["complete_month_trade15"].isna().all()
    assert dpm["complete_month_trade10"].isna().all()
    assert dpm[["pseudo_zero_flag", "partial_zero_flag", "true_zero_flag"]].isna().all().all()


# ---------------------------------------------------------------- 3. 窗口计数
def test_window_2015_05_2023_12_counts(shadow):
    w = shadow[(shadow["month"] >= WINDOW[0]) & (shadow["month"] <= WINDOW[1])]
    assert len(w) == 397_537
    status = w["trade_days_status"]
    legacy = w["complete_month_legacy"].eq(1)
    td = w["n_trade_days"]
    zero = status.eq("no_trade")
    partial = status.eq("ok") & td.gt(0) & td.lt(15)
    true = status.eq("ok") & td.ge(15)
    dpm = status.eq("daily_panel_missing")
    assert int(legacy.sum()) == 397_440
    assert int((legacy & (zero | partial)).sum()) == 13_980
    assert int((legacy & zero).sum()) == 8_014
    assert int((legacy & partial).sum()) == 5_966
    assert int(dpm.sum()) == 38

    r0 = w["R"].eq(0)
    assert int((r0 & legacy).sum()) == 9_561
    assert int((r0 & legacy & zero).sum()) == 7_524
    assert int((r0 & legacy & partial).sum()) == 10
    assert int((r0 & legacy & true).sum()) == 2_017
    assert int((r0 & legacy & dpm).sum()) == 10


def test_flags_partition_evaluable_zero_returns(shadow):
    ev = shadow["trade_days_status"].isin(EVAL_STATUS)
    flags = shadow.loc[ev, ["pseudo_zero_flag", "partial_zero_flag", "true_zero_flag"]]
    assert flags.notna().all().all()
    n_flags = flags.fillna(0).sum(axis=1)
    assert n_flags.le(1).all()
    r0 = ev & shadow["R"].eq(0)
    assert int(n_flags[r0].eq(1).sum()) == int(r0.sum())
    assert n_flags[ev & shadow["R"].ne(0)].eq(0).all()
    # daily_panel_missing 不参与 zero flag，即使 R=0
    dpm = shadow["trade_days_status"].eq("daily_panel_missing")
    assert flags.loc[dpm].isna().all().all()


# ---------------------------------------------------------------- 4. R_p2shadow
def test_r_p2shadow_na_iff_no_trade(shadow):
    status = shadow["trade_days_status"]
    assert int(status.eq("no_trade").sum()) == 10_234
    assert shadow.loc[status.eq("no_trade"), "R_p2shadow"].isna().all()
    assert int(shadow["R_p2shadow"].isna().sum()) == 10_234
    keep = ~status.eq("no_trade")
    np.testing.assert_allclose(shadow.loc[keep, "R_p2shadow"].to_numpy(),
                               shadow.loc[keep, "R"].to_numpy(), rtol=0, atol=1e-15)


# ---------------------------------------------------------------- 5. 586/46/96
def test_nonzero_R_split_586_46_96(shadow):
    status = shadow["trade_days_status"]
    no_trade = status.eq("no_trade")
    dpm = status.eq("daily_panel_missing")
    r_nonzero = shadow["R"].notna() & shadow["R"].ne(0)

    assert int((no_trade & r_nonzero).sum()) == 586
    assert int((no_trade & ~r_nonzero).sum()) == 9_648

    alias = shadow["firm_id"].eq("sh601313")
    assert int(alias.sum()) == 73          # 2012-02..2018-02 全部行
    alias_scope = alias & (shadow["month"] >= "2014-01")
    assert int(alias_scope.sum()) == 50    # 2014–2025
    alias_ok = alias_scope & status.eq("ok")
    assert int(alias_ok.sum()) == 46
    assert int((alias_ok & shadow["n_trade_days"].ge(15)).sum()) == 43
    assert int((alias_ok & shadow["n_trade_days"].gt(0)
                & shadow["n_trade_days"].lt(15)).sum()) == 3
    alias_nt = alias_scope & no_trade
    assert int(alias_nt.sum()) == 4
    assert sorted(shadow.loc[alias_nt, "month"].tolist()) == [
        "2017-07", "2017-08", "2017-09", "2017-10"]
    assert shadow.loc[alias_ok, "R_p2shadow"].notna().all()

    assert int(dpm.sum()) == 96
    assert shadow.loc[dpm, "R"].eq(0).all()
    assert shadow.loc[dpm, "R_p2shadow"].eq(0).all()
    assert int((dpm & r_nonzero).sum()) == 0
    assert int((dpm & (no_trade & r_nonzero)).sum()) == 0
    assert 586 + 46 == 632


# ---------------------------------------------------------------- 6. 合成语义
_MINI_DATES = pd.bdate_range("2014-01-02", periods=20).strftime("%Y-%m-%d").tolist()


def _mini_monthly() -> pd.DataFrame:
    rows = [
        ("sz000001", "2013-12", 0.0, 20, 1),   # pre-2014：no_volume_data
        ("sz000001", "2014-01", 0.0, 20, 1),   # suspend 全日 → no_trade
        ("sz000002", "2014-01", 0.0, 20, 1),   # 20 日成交 → 真实零
        ("sz000003", "2014-01", 0.1, 20, 1),   # 10 日成交 → 残月（R 非零）
        ("sz000004", "2014-01", 0.0, 20, 1),   # daily 行全 volume=0 → no_trade
        ("sz000005", "2014-01", 0.0, 20, 1),   # 部分停牌 5/20、无 daily 行 → dpm
        ("sz000006", "2026-01", 0.0, 20, 1),   # out_of_daily_coverage
        ("sz000007", "2014-01", 0.0, 20, 1),   # 无 daily 行、无 suspend → dpm
        ("sz000009", "2014-01", 0.05, 20, 1),  # alias 到 sz000010 → ok
    ]
    return pd.DataFrame(rows, columns=["firm_id", "month", "R", "n_valid_days",
                                       "complete_month"])


def _mini_daily_panel(tmp_path: Path) -> Path:
    panel = tmp_path / "daily_panel"
    panel.mkdir()
    rows = []
    for d in _MINI_DATES:                    # 20 个交易日
        rows.append(("sz000002", d, 100.0))
    for d in _MINI_DATES[:10]:               # 10 个交易日
        rows.append(("sz000003", d, 100.0))
    for d in _MINI_DATES:                    # 有行但 volume=0
        rows.append(("sz000004", d, 0.0))
    for d in _MINI_DATES:                    # alias target
        rows.append(("sz000010", d, 100.0))
    pd.DataFrame(rows, columns=["code", "date", "volume"]).to_csv(
        panel / "2014.csv", index=False)
    return panel


def _mini_suspend_flag() -> pd.DataFrame:
    idx = pd.to_datetime(_MINI_DATES)
    flag = pd.DataFrame(False, index=idx,
                        columns=["000001.SZ", "000002.SZ", "000003.SZ",
                                 "000004.SZ", "000005.SZ", "000010.SZ"])
    flag["000001.SZ"] = True                  # 整月停牌
    flag.loc[idx[:5], "000005.SZ"] = True     # 部分停牌 5/20
    return flag


def test_synthetic_suspension_alias_and_missing(tmp_path):
    mod = load_module()
    out = mod.build_status_panel(
        _mini_monthly(), _mini_daily_panel(tmp_path),
        start_year=2014, end_year=2025, min_trade_days=15, sensitivity_days=10,
        suspend_flag=_mini_suspend_flag(),
        alias_map={"sz000009": "sz000010"})
    assert len(out) == 9
    g = out.set_index(["firm_id", "month"])

    a_pre = g.loc[("sz000001", "2013-12")]
    assert a_pre["trade_days_status"] == "no_volume_data"
    assert pd.isna(a_pre["n_trade_days"])
    assert pd.isna(a_pre["complete_month_trade15"])
    assert pd.isna(a_pre["pseudo_zero_flag"])
    assert a_pre["R_p2shadow"] == 0.0

    a = g.loc[("sz000001", "2014-01")]
    assert a["trade_days_status"] == "no_trade"
    assert a["n_trade_days"] == 0
    assert a["complete_month_trade15"] == 0
    assert a["pseudo_zero_flag"] == 1
    assert pd.isna(a["R_p2shadow"])

    b = g.loc[("sz000002", "2014-01")]
    assert b["trade_days_status"] == "ok"
    assert b["n_trade_days"] == 20
    assert b["complete_month_trade15"] == 1
    assert b["true_zero_flag"] == 1
    assert b["R_p2shadow"] == 0.0

    c = g.loc[("sz000003", "2014-01")]
    assert c["trade_days_status"] == "ok"
    assert c["n_trade_days"] == 10
    assert c["complete_month_trade15"] == 0
    assert c["complete_month_trade10"] == 1
    assert c["partial_zero_flag"] == 0
    assert c["R_p2shadow"] == pytest.approx(0.1)

    d = g.loc[("sz000004", "2014-01")]
    assert d["trade_days_status"] == "no_trade"
    assert d["n_trade_days"] == 0
    assert d["pseudo_zero_flag"] == 1
    assert pd.isna(d["R_p2shadow"])

    e = g.loc[("sz000006", "2026-01")]
    assert e["trade_days_status"] == "out_of_daily_coverage"
    assert pd.isna(e["n_trade_days"])
    assert e["R_p2shadow"] == 0.0

    # alias：源行无自己的日频，但 target 有 20 个交易日 → ok，R 保留
    al = g.loc[("sz000009", "2014-01")]
    assert al["trade_days_status"] == "ok"
    assert al["n_trade_days"] == 20
    assert al["R_p2shadow"] == pytest.approx(0.05)


def test_synthetic_partial_suspension_is_daily_panel_missing(tmp_path):
    mod = load_module()
    out = mod.build_status_panel(
        _mini_monthly(), _mini_daily_panel(tmp_path),
        suspend_flag=_mini_suspend_flag(),
        alias_map={"sz000009": "sz000010"})
    g = out.set_index(["firm_id", "month"])

    partial = g.loc[("sz000005", "2014-01")]
    assert partial["trade_days_status"] == "daily_panel_missing"
    assert pd.isna(partial["n_trade_days"])
    assert pd.isna(partial["complete_month_trade15"])
    assert pd.isna(partial["complete_month_trade10"])
    assert pd.isna(partial["pseudo_zero_flag"])
    assert pd.isna(partial["partial_zero_flag"])
    assert pd.isna(partial["true_zero_flag"])
    assert partial["R_p2shadow"] == 0.0     # 无法证明无成交，保留原 R

    no_flag = g.loc[("sz000007", "2014-01")]
    assert no_flag["trade_days_status"] == "daily_panel_missing"
    assert pd.isna(no_flag["n_trade_days"])
    assert no_flag["R_p2shadow"] == 0.0


def test_synthetic_alias_vs_no_alias(tmp_path):
    mod = load_module()
    ret = pd.DataFrame(
        [("sz000009", "2014-01", 0.05, 20, 1)],
        columns=["firm_id", "month", "R", "n_valid_days", "complete_month"])
    panel = _mini_daily_panel(tmp_path)
    flag = _mini_suspend_flag()

    with_alias = mod.build_status_panel(
        ret, panel, suspend_flag=flag,
        alias_map={"sz000009": "sz000010"})
    assert with_alias["trade_days_status"].iat[0] == "ok"
    assert with_alias["n_trade_days"].iat[0] == 20

    without_alias = mod.build_status_panel(
        ret, panel, suspend_flag=flag, alias_map={})
    assert without_alias["trade_days_status"].iat[0] == "daily_panel_missing"
    assert pd.isna(without_alias["n_trade_days"].iat[0])


def test_synthetic_columns_appended_in_order(tmp_path):
    mod = load_module()
    ret = _mini_monthly()
    out = mod.build_status_panel(
        ret, _mini_daily_panel(tmp_path), suspend_flag=_mini_suspend_flag(),
        alias_map={"sz000009": "sz000010"})
    assert list(out.columns) == list(ret.columns) + NEW_COLS
    assert out["trade_data_status"].fillna("NA").eq(
        out["trade_days_status"].fillna("NA")).all()


# ---------------------------------------------------------------- 7. 路径/锚
def test_shadow_paths_never_equal_headline():
    mod = load_module()
    contracts = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    protected = mod.protected_paths(contracts)
    assert SHADOW.resolve() not in protected
    assert META.resolve() not in protected
    assert ALIAS.resolve() not in protected
    assert COUNTS.resolve() not in protected
    assert RETURNS.resolve() in protected


def test_output_path_guard_rejects_headline_path():
    mod = load_module()
    with pytest.raises(SystemExit):
        mod.main(["--out", str(RETURNS)])


def test_headline_9_inputs_sha_unchanged():
    contracts = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    assert len(contracts["m1_inputs"]) == 9
    for key, meta in contracts["m1_inputs"].items():
        p = ROOT / meta["file"]
        assert p.is_file(), f"{key}: 文件缺失 {p}"
        assert sha256_file(p) == meta["sha256"], f"{key}: headline sha 变化"
    # BL-01（I-17）后不再钉死 clean_v2 字面 sha：contracts 锚必须等于现盘（逐项比对见上一循环）
    assert contracts["m1_inputs"]["monthly_returns_clean_v2"]["sha256"] == sha256_file(RETURNS)


# ---------------------------------------------------------------- 8. meta/counts
def test_meta_records_provenance(shadow):
    assert META.is_file()
    meta = json.loads(META.read_text(encoding="utf-8"))
    assert meta["p2shadow_version"] == "p2shadow_v1"
    assert meta["output"]["rows"] == EXPECTED_ROWS
    assert meta["output"]["sha256"] == sha256_file(SHADOW)
    assert meta["output"]["columns"] == list(shadow.columns)
    assert meta["inputs"]["monthly_returns"]["rows"] == EXPECTED_ROWS
    assert meta["inputs"]["monthly_returns"]["sha256"] == (
        "430419f60dfe77c57f9bdab2aef51607884240436e5019c4ddad55a74c2ecdc0")
    assert meta["inputs"]["stock_suspend_flag"]["sha256"] == sha256_file(SUSPEND_FLAG)
    assert meta["alias_map_output"]["sha256"] == sha256_file(ALIAS)
    assert meta["script_sha256"] == sha256_file(SCRIPT)
    assert meta["headline_anchors_match"] is True
    assert meta["alias_map"] == {"sh601313": "sh601360"}


def test_alias_map_json_basis():
    assert ALIAS.is_file()
    payload = json.loads(ALIAS.read_text(encoding="utf-8"))
    assert payload["alias"] == {"sh601313": "sh601360"}
    pair = payload["pairs"][0]
    assert pair["monthly_overlap_months"] == 73
    assert pair["monthly_identical_months"] == 72
    assert pair["monthly_non_identical_months"] == 1
    assert pair["monthly_non_identical_detail"][0]["month"] == "2018-02"
    assert pair["daily_panel_source_has_rows"] is False
    assert pair["daily_panel_target_has_rows"] is True
    assert pair["firm_id_map_source_present"] is False
    assert pair["firm_id_map_target_present"] is True
    assert pair["rows_2014_2025"] == 50
    assert pair["recovered_rows"] == 46
    assert pair["recovered_ge15"] == 43
    assert pair["recovered_1_14"] == 3
    assert pair["still_no_trade_months"] == [
        "2017-07", "2017-08", "2017-09", "2017-10"]
    assert pair["daily_panel_missing_rows"] == 0


def test_counts_json_matches_panel(shadow, counts):
    assert counts["rows_total"] == EXPECTED_ROWS
    assert counts["rows_in_scope_2014_2025"] == 561_478
    assert counts["rows_evaluable_2014_2025"] == 561_382
    assert counts["rows_daily_panel_missing"] == 96
    assert counts["n_trade_days_zero"] == 10_234
    assert counts["n_trade_days_1_14"] == 8_348
    assert counts["n_trade_days_ge_15"] == 542_800
    assert counts["R_p2shadow_na_rows"] == 10_234
    assert counts["n_trade_days_gt_n_price_obs"] == 0
    assert counts["window_2015_05_2023_12_all_match"] is True

    w = counts["window_2015_05_2023_12"]
    assert w["complete_legacy_and_trade_days_lt_15"] == 13_980
    assert w["complete_legacy_and_trade_days_zero"] == 8_014
    assert w["complete_legacy_and_trade_days_1_14"] == 5_966
    assert w["rows_daily_panel_missing"] == 38
    assert w["R_zero_and_complete_legacy"] == 9_561
    assert w["R_zero_and_complete_legacy_trade_days_zero"] == 7_524
    assert w["R_zero_and_complete_legacy_trade_days_1_14"] == 10
    assert w["R_zero_and_complete_legacy_trade_days_ge_min"] == 2_017
    assert w["R_zero_and_complete_legacy_daily_panel_missing"] == 10

    mc = counts["missing_classification"]
    assert mc["no_trade_R_nonzero"] == 586
    assert mc["alias_recovered_rows"] == 46
    assert mc["alias_recovered_ge15"] == 43
    assert mc["alias_recovered_1_14"] == 3
    assert mc["alias_still_no_trade_months"] == [
        "2017-07", "2017-08", "2017-09", "2017-10"]
    assert mc["daily_panel_missing_rows"] == 96
    assert mc["daily_panel_missing_R_zero"] == 96
    assert mc["daily_panel_missing_R_nonzero"] == 0
    assert mc["original_nonzero_R_daily_panel_missing_632"] == 632
    assert mc["overlap_daily_panel_missing_with_632"] == 0
    assert mc["missing_rows_before_alias"] == 10_376
    assert mc["missing_rows_after_alias"] == 10_330
    assert mc["no_trade_suspend_eq_market"] == 10_234
    assert mc["no_trade_suspend_lt_market"] == 0
    assert mc["dpm_suspend_gt0_lt_market"] == 96
    assert mc["dpm_suspend_eq_market"] == 0

    fm = counts["fm_ma_teacher"]
    assert fm["rows"] == 181_178
    assert fm["months"] == 104
    assert fm["R_next_zero"] == 2_016
    assert fm["R_next_zero_pseudo_full_month_no_trade"] == 1_118
    assert fm["R_next_zero_partial_1_14_trade_days"] == 4
    assert fm["R_next_zero_true_ge_15_trade_days"] == 894
    assert fm["R_next_zero_daily_panel_missing"] == 0
    assert fm["outcome_gate_15_loss_rows"] == 3_547
    assert fm["max_gate_15_loss_rows"] == 3_352
    assert fm["gate_overlap_rows"] == 1_046
    assert fm["gate_union_loss_rows"] == 5_853
    assert fm["fm_upper_bound_after_gates"] == 175_325
    assert all(fm["matches_2_0_plan"].values())

    bridge = counts["source_panel_bridge_2015_05_2023_12"]
    assert bridge["matched_rows"] == 389_453
    assert bridge["equal_rows"] == 389_452
    assert bridge["diff_rows"] == 1
    assert bridge["matches_2_0_plan"] is True
    assert bridge["diff_detail"][0]["firm_id"] == "sz000520"
    assert bridge["diff_detail"][0]["month"] == "2015-12"
    assert bridge["diff_detail"][0]["n_trade_days"] == 10
    assert bridge["diff_detail"][0]["source_n_valid_days"] == 9

    # counts JSON 前/后锚一致（Phase 2 历史不变式），且与当前锚一致；
    # BL-01（I-17）后 monthly_returns_clean_v2 的现盘 sha 已重登记，故该项只保留历史 before==after
    # 不变式 + 现盘对当前 m1_contracts 锚（见 test_headline_9_inputs_sha_unchanged）。
    assert counts["headline_anchors_match"] is True
    before = counts["headline_anchors_before"]
    after = counts["headline_anchors_after"]
    contracts = yaml.safe_load(CONTRACTS.read_text(encoding="utf-8"))
    rewritten_by_bl01 = {"monthly_returns_clean_v2", "source_month_argmax_panel"}
    for key, meta in contracts["m1_inputs"].items():
        assert before[key]["sha256"] == after[key]["sha256"], f"{key}: Phase 2 前/后锚不一致"
        if key not in rewritten_by_bl01:
            assert before[key]["sha256"] == meta["sha256"], f"{key}: 契约锚与历史锚不一致"
        else:
            assert before[key]["sha256"] != meta["sha256"], f"{key}: BL-01 重写后锚应已变化"
