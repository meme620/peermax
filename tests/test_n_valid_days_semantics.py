"""Phase 1.7 / I-1 + 口径更正：n_valid_days = 有效交易日（volume>0）语义测试。

覆盖：
- build_monthly_returns_status：n_valid_days 与 n_trade_days 一致（volume>0 天数），
  complete_month/return_status 同步按有效交易日判定；旧价格观测数仅作
  n_price_obs / n_price_obs_legacy 审计列。
- 全月 volume=0 → complete_month=0（口径更正 新增用例）。
- peermax_max_build：默认 0 输出与旧逻辑逐字节一致；阈值 15 只标记/排除不足行。
"""
import csv
import importlib.util
import io
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MONTHLY_SCRIPT = ROOT / "scripts/build_monthly_returns_status.py"
MAX_SCRIPT = ROOT / "scripts/peermax_max_build.py"
MONTHLY_V2 = ROOT / "data/processed/monthly_returns_clean_v2.csv"
SOURCE_MONTH = ROOT / "data/processed/source_month_argmax_panel.csv"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# build_monthly_returns_status
# ---------------------------------------------------------------------------

def _monthly_fixture(tmp_path):
    """构造：A 前向填充后 20 个价格观测但只有 10 个真实交易日；B 12/12；B 二月无记录。"""
    dates = pd.bdate_range("2015-01-05", "2015-01-30")     # 20 个工作日
    raw = pd.DataFrame(index=dates, columns=["000001.SZ", "000002.SZ"], dtype=float)
    raw.loc[dates[:10], "000001.SZ"] = np.arange(10.0, 20.0)   # A 仅在 10 天有原始价
    raw.loc[dates[:12], "000002.SZ"] = np.arange(20.0, 32.0)   # B 12 天有原始价
    px = raw.copy()
    px["000001.SZ"] = raw["000001.SZ"].ffill()                 # A 停牌日被前向填充 → 20 个非空
    assert px["000001.SZ"].notna().sum() == 20
    assert px["000002.SZ"].notna().sum() == 12

    ship = pd.DataFrame({
        "firm_id": ["sz000001", "sz000002", "sz000002"],
        "month": ["2015-01", "2015-01", "2015-02"],
        "R": [0.11, 0.22, -0.05],
    })
    # 日频面板：A 10 天（另有 1 条 volume=0 不应计数）；B 12 天
    panel_rows = []
    for d in dates[:10]:
        panel_rows.append(["sz000001", d.strftime("%Y-%m-%d"), "1000"])
    panel_rows.append(["sz000001", dates[10].strftime("%Y-%m-%d"), "0"])
    for d in dates[:12]:
        panel_rows.append(["sz000002", d.strftime("%Y-%m-%d"), "2000"])
    panel_dir = tmp_path / "daily_panel"
    _write_csv(panel_dir / "2015.csv", ["code", "date", "volume"], panel_rows)
    return ship, px, panel_dir


def test_price_obs_vs_trade_days_and_legacy_columns_unchanged(tmp_path):
    mod = _load(MONTHLY_SCRIPT, "build_monthly_returns_status_i1")
    ship, px, panel_dir = _monthly_fixture(tmp_path)

    out = mod.build_status(ship, px, panel_dir=panel_dir)

    # 旧列顺序不变，新列只追加在末尾
    assert list(out.columns) == mod.OLD_COLS + mod.NEW_COLS
    assert mod.NEW_COLS == ["n_price_obs", "n_trade_days", "n_price_obs_legacy",
                            "n_valid_days_semantics"]

    got = out.set_index(["firm_id", "month"])
    # A：前向填充 20 个价格观测，但只有 10 个真实交易日
    assert got.loc[("sz000001", "2015-01"), "n_price_obs"] == 20
    assert got.loc[("sz000001", "2015-01"), "n_price_obs_legacy"] == 20
    assert got.loc[("sz000001", "2015-01"), "n_trade_days"] == 10
    assert got.loc[("sz000001", "2015-01"), "n_valid_days"] == 10
    # B：12/12；二月无任何记录 → 0
    assert got.loc[("sz000002", "2015-01"), "n_price_obs"] == 12
    assert got.loc[("sz000002", "2015-01"), "n_trade_days"] == 12
    assert got.loc[("sz000002", "2015-01"), "n_valid_days"] == 12
    assert got.loc[("sz000002", "2015-02"), "n_price_obs"] == 0
    assert got.loc[("sz000002", "2015-02"), "n_trade_days"] == 0
    assert got.loc[("sz000002", "2015-02"), "n_valid_days"] == 0

    # 口径更正：complete_month / return_status 按有效交易日判定
    assert got.loc[("sz000001", "2015-01"), "complete_month"] == 0
    assert got.loc[("sz000002", "2015-01"), "complete_month"] == 0
    assert got.loc[("sz000002", "2015-02"), "complete_month"] == 0
    assert got.loc[("sz000001", "2015-01"), "return_status"] == "CONDITIONAL"
    assert got.loc[("sz000002", "2015-01"), "return_status"] == "CONDITIONAL"
    assert got.loc[("sz000002", "2015-02"), "return_status"] == "BLOCKED"
    assert set(out["n_valid_days_semantics"]) == {"trade_days_v1"}

    # R / month_start / month_end 未被新列影响
    assert got.loc[("sz000001", "2015-01"), "R"] == 0.11
    assert got.loc[("sz000001", "2015-01"), "month_start"] == "2015-01-05"
    assert got.loc[("sz000001", "2015-01"), "month_end"] == "2015-01-30"


def test_complete_month_follows_trade_days(tmp_path):
    """价格观测足 15 天但真实交易日只有 10 天 → complete_month=0（口径更正 语义）。"""
    mod = _load(MONTHLY_SCRIPT, "build_monthly_returns_status_i1_b")
    ship, px, panel_dir = _monthly_fixture(tmp_path)
    out = mod.build_status(ship, px, panel_dir=panel_dir)
    row = out[(out["firm_id"] == "sz000001") & (out["month"] == "2015-01")].iloc[0]
    assert row["n_price_obs"] == 20 and row["n_trade_days"] == 10
    assert row["complete_month"] == 0


def test_all_zero_volume_month_is_incomplete(tmp_path):
    """全月有价格观测但 volume 全为 0 → n_trade_days=0、complete_month=0、BLOCKED。"""
    mod = _load(MONTHLY_SCRIPT, "build_monthly_returns_status_i1_zero_vol")
    dates = pd.bdate_range("2015-01-05", "2015-01-30")
    px = pd.DataFrame(index=dates, columns=["000001.SZ"], dtype=float)
    px["000001.SZ"] = 10.0                       # 20 个价格观测（前向/原值均非空）
    ship = pd.DataFrame({"firm_id": ["sz000001"], "month": ["2015-01"], "R": [0.01]})
    panel_dir = tmp_path / "daily_panel_zero_volume"
    _write_csv(panel_dir / "2015.csv", ["code", "date", "volume"],
               [["sz000001", d.strftime("%Y-%m-%d"), "0"] for d in dates])
    out = mod.build_status(ship, px, panel_dir=panel_dir)
    row = out.iloc[0]
    assert row["n_price_obs"] == 20
    assert row["n_trade_days"] == 0
    assert row["n_valid_days"] == 0
    assert row["complete_month"] == 0
    assert row["return_status"] == "BLOCKED"


def test_empty_panel_dir_gives_zero_trade_days(tmp_path):
    mod = _load(MONTHLY_SCRIPT, "build_monthly_returns_status_i1_c")
    ship, px, _ = _monthly_fixture(tmp_path)
    empty = tmp_path / "empty_panel"
    empty.mkdir()
    out = mod.build_status(ship, px, panel_dir=empty)
    assert (out["n_trade_days"] == 0).all()
    assert (out["n_valid_days"] == 0).all()
    assert (out["complete_month"] == 0).all()
    assert out["n_price_obs"].tolist() == [20, 12, 0]


# ---------------------------------------------------------------------------
# peermax_max_build（口径更正）：complete_month 边界 / D-02=A / registry fail-closed
# ---------------------------------------------------------------------------

PANEL_HEADER = ["code", "date", "volume", "rraw"]


def _gate_panel(tmp_path):
    """A=15 个 volume>0 日；B=14 个；C=5 个（volume 全空白 → n_valid=0，但 rraw 可观测）。"""
    rows = []
    for i in range(15):
        rows.append(["sz000001", f"2015-01-{i + 1:02d}", "100", f"{0.001 * (i + 1):.8f}"])
    for i in range(14):
        rows.append(["sz000002", f"2015-01-{i + 1:02d}", "100", f"{0.002 * (i + 1):.8f}"])
    for i in range(5):
        rows.append(["sz000003", f"2015-01-{i + 1:02d}", "", f"{0.003 * (i + 1):.8f}"])
    panel_dir = tmp_path / "daily_panel"
    _write_csv(panel_dir / "2015.csv", PANEL_HEADER, rows)
    return panel_dir


def _registry_with_min(tmp_path, entry):
    import yaml
    p = tmp_path / f"registry_{len(list(tmp_path.glob('registry_*.yaml')))}.yaml"
    doc = {"registry_version": "test", "parameters": [
        {"name": "tie_rule", "headline_value": "earliest",
         "sensitivity_grid": ["earliest", "latest"]},
        {"name": "limit_thresholds",
         "headline_value": {"main": 0.095, "main_st": 0.045, "chinext_star": 0.195}},
        {"name": "max_day_policy",
         "headline_value": {"limit_hit": "exclude", "resume": "exclude",
                            "listing_grace_days": 0}},
        entry]}
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def _st_flag_pickle(tmp_path):
    """最小 ST 面板：run() 不接受 ST 缺失（限价判定不得降级），故给一张全 False 的。"""
    import pandas as pd
    panel_dir = _gate_panel(tmp_path)
    dates = sorted({r[1] for f in panel_dir.glob('*.csv')
                    for r in list(csv.reader(open(f, encoding='utf-8')))[1:]})
    codes = sorted({r[0][2:] for f in panel_dir.glob('*.csv')
                    for r in list(csv.reader(open(f, encoding='utf-8')))[1:]})
    df = pd.DataFrame({f"{c}.SZ": [False] * len(dates) for c in codes},
                      index=pd.to_datetime(dates))
    p = tmp_path / "st_flag.pkl"
    df.to_pickle(p)
    return p


def _run_gate(tmp_path, mod, **kwargs):
    return mod.run(panel_dir=_gate_panel(tmp_path), out=tmp_path / "source_month.csv",
                   out_json=tmp_path / "peermax_panel_report.json",
                   st_flag=_st_flag_pickle(tmp_path), **kwargs)


def test_complete_month_boundary_and_d02_keep(tmp_path):
    mod = _load(MAX_SCRIPT, "step1_boundary")
    report = _run_gate(tmp_path, mod, min_valid_days=15)
    df = pd.read_csv(tmp_path / "source_month.csv", dtype=str, keep_default_na=False).set_index("source_id")
    # A: 15 日 → complete
    assert df.loc["sz000001", "n_valid_days"] == "15"
    assert df.loc["sz000001", "complete_month"] == "1"
    assert df.loc["sz000001", "max_status"] == "complete"
    assert df.loc["sz000001", "coverage_flag"] == "covered"
    assert df.loc["sz000001", "headline_max_eligible"] == "1"
    # B: 14 日 → incomplete，但 D-02=A 保留观测 MAX
    assert df.loc["sz000002", "n_valid_days"] == "14"
    assert df.loc["sz000002", "complete_month"] == "0"
    assert df.loc["sz000002", "max_status"] == "incomplete"
    assert df.loc["sz000002", "MAXraw"] != ""
    assert df.loc["sz000002", "missing_reason"] == "incomplete_month"
    assert df.loc["sz000002", "headline_max_eligible"] == "0"
    # C: volume 全空白 → n_valid=0；rraw/n_max 仍可观测
    assert df.loc["sz000003", "n_valid_days"] == "0"
    assert df.loc["sz000003", "n_rraw_days"] == "5"
    assert df.loc["sz000003", "n_max_days"] == "5"
    assert df.loc["sz000003", "complete_month"] == "0"
    assert df.loc["sz000003", "max_status"] == "incomplete"
    assert df.loc["sz000003", "MAXraw"] != ""
    assert df.loc["sz000003", "return_status"] == "BLOCKED"
    # 恒等式
    assert (pd.to_numeric(df["complete_month"]) ==
            (pd.to_numeric(df["n_valid_days"]) >= 15).astype(int)).all()
    assert report["min_valid_days"] == 15
    assert report["complete_month_rows"] == 1
    assert report["incomplete_month_rows"] == 2
    assert report["incomplete_with_max_observed"] == 2
    assert report["n_max_days_zero_rows"] == 0


def test_registry_min_valid_days_drives_gate(tmp_path):
    mod = _load(MAX_SCRIPT, "step1_reg14")
    reg = _registry_with_min(tmp_path, {"name": "min_valid_days", "headline_value": 14})
    report = _run_gate(tmp_path, mod, registry_path=reg)
    df = pd.read_csv(tmp_path / "source_month.csv", dtype=str, keep_default_na=False).set_index("source_id")
    assert report["min_valid_days"] == 14
    assert df.loc["sz000002", "complete_month"] == "1"
    assert df.loc["sz000001", "complete_month"] == "1"


def test_min_valid_days_fail_closed(tmp_path):
    mod = _load(MAX_SCRIPT, "step1_fail")
    missing = _registry_with_min(tmp_path, {"name": "other", "headline_value": 15})
    for bad in (missing,
                _registry_with_min(tmp_path, {"name": "min_valid_days", "headline_value": 0}),
                _registry_with_min(tmp_path, {"name": "min_valid_days", "headline_value": -3}),
                _registry_with_min(tmp_path, {"name": "min_valid_days", "headline_value": "15"})):
        with pytest.raises(SystemExit) as ei:
            mod.run(panel_dir=_gate_panel(tmp_path), out=tmp_path / "o.csv",
                    out_json=tmp_path / "r.json", st_flag=tmp_path / "no.pkl",
                    registry_path=bad)
        assert ei.value.code == 2
    dup = _registry_with_min(tmp_path, {"name": "min_valid_days", "headline_value": 15})
    import yaml
    doc = yaml.safe_load(dup.read_text(encoding="utf-8"))
    doc["parameters"].append({"name": "min_valid_days", "headline_value": 15})
    dup.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        mod.run(panel_dir=_gate_panel(tmp_path), out=tmp_path / "o.csv",
                out_json=tmp_path / "r.json", st_flag=tmp_path / "no.pkl",
                registry_path=dup)
    assert ei.value.code == 2


# ---------------------------------------------------------------------------
# 真实数据只读复算（口径更正：source-month 与 returns v2 的 complete_month 恒等式）
# ---------------------------------------------------------------------------

def test_real_complete_month_invariant():
    """口径更正：source-month / returns v2 的 complete_month == 1{n_valid_days>=15}。"""
    for p in (MONTHLY_V2, SOURCE_MONTH):
        if not p.exists():
            pytest.skip(f"missing {p}")
    m = pd.read_csv(MONTHLY_V2, usecols=["firm_id", "month", "n_valid_days", "complete_month"])
    assert int(((m["complete_month"] == 1) & (m["n_valid_days"] < 15)).sum()) == 0
    s = pd.read_csv(SOURCE_MONTH, dtype={"source_id": str, "month": str})
    assert (s["complete_month"] == (s["n_valid_days"] >= 15).astype(int)).all()
    assert {"n_rraw_days", "n_max_days", "max_status", "return_status",
            "source_month_status", "headline_max_eligible"} <= set(s.columns)
    assert (s["n_max_days"] <= s["n_rraw_days"]).all()
    w = s[(s["month"] >= "2015-01") & (s["month"] <= "2023-12")]
    # 窗口内不变量（不写死行数：行数随数据更新变化，写死会变成永假守卫）
    assert len(w) > 0
    assert (w["n_max_days"] <= w["n_rraw_days"]).all()
    assert (w["complete_month"] == (w["n_valid_days"] >= 15).astype(int)).all()
    assert bool(((w["n_rraw_days"] < 15) & w["MAXraw"].notna()).any())
    # 关系型不变量：n_rraw_days 是 rraw 非空日数、n_valid_days 是有效交易日数，
    # 两者 >=15 的判定允许分歧，但分歧量必须是少数（绝大多数月份两者一致）。
    divergent = int(((s["n_valid_days"] >= 15) != (s["n_rraw_days"] >= 15)).sum())
    assert 0 < divergent < len(s) * 0.1, divergent
