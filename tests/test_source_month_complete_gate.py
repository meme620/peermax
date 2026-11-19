"""口径更正：source-month 完整月 gate（07 卡点名；2 source × 3 月，n=14/15 边界）。

覆盖：
- `complete_month == 1{n_valid_days>=15}` 恒等式；
- `n_valid_days=volume>0` / `n_rraw_days` / `n_max_days` 三列语义；
- D-02=A：`n_valid_days<15` 但 MAX 已观测 → 保留 `MAXraw/argmax_date`，`complete_month=0`；
- `max_status/source_month_status/coverage_flag/missing_reason/headline_max_eligible`；
- registry `min_valid_days` 缺失/重复/非正整数 → SystemExit(2)（不硬编码、不回退 0）。
"""
import csv
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/peermax_max_build.py"


def _load(name="step1_complete_gate"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fixture(tmp_path):
    """2 source × 3 月：s1 = 15/14/5（第三月 volume 空白）；s2 = 16/15/14。"""
    rows = []
    # 2015-01
    for i in range(15):
        rows.append(["sz000001", f"2015-01-{i + 1:02d}", "100", "0.0010"])
    for i in range(16):
        rows.append(["sz000002", f"2015-01-{i + 1:02d}", "100", "0.0020"])
    # 2015-02
    for i in range(14):
        rows.append(["sz000001", f"2015-02-{i + 1:02d}", "100", "0.0030"])
    for i in range(15):
        rows.append(["sz000002", f"2015-02-{i + 1:02d}", "100", "0.0040"])
    # 2015-03：s1 volume 全空白（rraw 可观测）；s2 14 个有效日
    for i in range(5):
        rows.append(["sz000001", f"2015-03-{i + 1:02d}", "", "0.0050"])
    for i in range(14):
        rows.append(["sz000002", f"2015-03-{i + 1:02d}", "100", "0.0060"])
    panel = tmp_path / "daily_panel"
    _write_csv(panel / "2015.csv", ["code", "date", "volume", "rraw"], rows)
    return panel


def _st_panel(tmp_path, panel):
    """合成全 0 ST 面板（代码取面板里出现的 source）；ST 面板缺失会 fail-closed。"""
    codes = sorted({r[0][2:] for r in list(csv.reader(open(panel / "2015.csv", encoding="utf-8")))[1:]})
    dates = sorted({r[1] for r in list(csv.reader(open(panel / "2015.csv", encoding="utf-8")))[1:]})
    path = tmp_path / "st.pkl"
    pd.DataFrame({f"{c}.SZ": [False] * len(dates) for c in codes},
                 index=pd.to_datetime(dates)).to_pickle(path)
    return path


def _run(tmp_path, mod, **kwargs):
    """跑 run()：本文件只验 complete_month 语义，故 day policy 取 include、
    上市宽限取 0（限价/复牌/宽限口径由 test_max_day_policy 与 MAX 侧用例覆盖）。"""
    panel = _fixture(tmp_path)
    mod.LIMIT_HIT_FLAG = tmp_path / "no_limit_up.pkl"   # 走阈值回退，与 HEAD 期口径一致
    if kwargs.get("registry_path") is None:
        tie = {"name": "tie_rule", "headline_value": "earliest",
               "sensitivity_grid": ["earliest"]}
        kwargs["registry_path"] = _bad_registry(tmp_path, [tie], "core_registry.yaml")
    return mod.run(panel_dir=panel, out=tmp_path / "source_month.csv",
                   out_json=tmp_path / "report.json",
                   st_flag=_st_panel(tmp_path, panel), **kwargs)


def _read(tmp_path):
    return pd.read_csv(tmp_path / "source_month.csv", dtype=str,
                       keep_default_na=False).set_index(["source_id", "month"])


def test_complete_month_boundary_identity_and_d02(tmp_path):
    mod = _load()
    report = _run(tmp_path, mod, min_valid_days=15)
    df = _read(tmp_path)

    assert (pd.to_numeric(df["complete_month"]) ==
            (pd.to_numeric(df["n_valid_days"]) >= 15).astype(int)).all()
    # s1: 15/14/volume-blank
    assert df.loc[("sz000001", "2015-01"), "complete_month"] == "1"
    assert df.loc[("sz000001", "2015-02"), "complete_month"] == "0"
    assert df.loc[("sz000001", "2015-03"), "n_valid_days"] == "0"
    assert df.loc[("sz000001", "2015-03"), "n_rraw_days"] == "5"
    assert df.loc[("sz000001", "2015-03"), "n_max_days"] == "5"
    # s2: 16/15/14
    assert set(df.loc["sz000002", "complete_month"]) == {"1", "0"}
    assert df.loc[("sz000002", "2015-03"), "n_valid_days"] == "14"
    # D-02=A：所有 incomplete 行保留 MAX
    inc = df[df["complete_month"] == "0"]
    assert (inc["MAXraw"] != "").all()
    assert (inc["max_status"] == "incomplete").all()
    assert (inc["coverage_flag"] == "partial").all()
    assert (inc["source_month_status"] == "incomplete_month").all()
    assert (inc["missing_reason"].str.contains("incomplete_month")).all()
    assert (inc["headline_max_eligible"] == "0").all()
    ok = df[df["complete_month"] == "1"]
    assert (ok["max_status"] == "complete").all()
    assert (ok["coverage_flag"] == "covered").all()
    assert (ok["source_month_status"] == "ok").all()
    assert (ok["missing_reason"] == "").all()
    # available_at = 当月全市场最后交易日
    assert set(df["available_at"]) == {"2015-01-16", "2015-02-15", "2015-03-14"}
    assert set(df["source_month_version"]) == {"src_month_v2_20261116"}
    assert set(df["source_version"]) == {"tencent_daily_panel_20260630"}
    assert report["min_valid_days"] == 15
    assert report["complete_month_rows"] == 3
    assert report["incomplete_month_rows"] == 3
    assert report["incomplete_with_max_observed"] == 3
    assert report["n_max_days_zero_rows"] == 0


CORE_ENTRIES = [
    {"name": "limit_thresholds",
     "headline_value": {"main": 0.095, "main_st": 0.045, "chinext_star": 0.195}},
    {"name": "max_day_policy",
     "headline_value": {"limit_hit": "include", "resume": "include",
                        "listing_grace_days": 0}},
]


def _bad_registry(tmp_path, params, name):
    """临时 registry：补上 run() 必需的顶层参数（除非用例自己给了同名条目）。

    本文件的用例都在验 min_valid_days 的 fail-closed 行为，不该被 limit_thresholds /
    max_day_policy 缺失抢先触发。
    """
    names = {p.get("name") for p in params}
    full = list(params) + [c for c in CORE_ENTRIES if c["name"] not in names]
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"registry_version": "test", "parameters": full},
                                allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def test_registry_min_valid_days_fail_closed(tmp_path):
    mod = _load("s4_step1_complete_gate_fail")
    tie = {"name": "tie_rule", "headline_value": "earliest",
           "sensitivity_grid": ["earliest"]}
    cases = [
        ("missing.yaml", [tie]),
        ("zero.yaml", [tie, {"name": "min_valid_days", "headline_value": 0}]),
        ("negative.yaml", [tie, {"name": "min_valid_days", "headline_value": -1}]),
        ("text.yaml", [tie, {"name": "min_valid_days", "headline_value": "15"}]),
        ("dup.yaml", [tie, {"name": "min_valid_days", "headline_value": 15},
                      {"name": "min_valid_days", "headline_value": 15}]),
    ]
    for fn, params in cases:
        reg = _bad_registry(tmp_path, params, fn)
        with pytest.raises(SystemExit) as ei:
            mod.run(panel_dir=_fixture(tmp_path), out=tmp_path / "o.csv",
                    out_json=tmp_path / "r.json", st_flag=tmp_path / "no.pkl",
                    registry_path=reg)
        assert ei.value.code == 2, fn
    # no registry file
    with pytest.raises(SystemExit) as ei:
        mod.run(panel_dir=_fixture(tmp_path), out=tmp_path / "o.csv",
                out_json=tmp_path / "r.json", st_flag=tmp_path / "no.pkl",
                registry_path=tmp_path / "nope.yaml")
    assert ei.value.code == 2


def test_registry_14_changes_gate(tmp_path):
    mod = _load("s4_step1_complete_gate_14")
    tie = {"name": "tie_rule", "headline_value": "earliest",
           "sensitivity_grid": ["earliest"]}
    reg = _bad_registry(tmp_path, [tie, {"name": "min_valid_days", "headline_value": 14}],
                        "reg14.yaml")
    panel = _fixture(tmp_path)
    mod.LIMIT_HIT_FLAG = tmp_path / "no_limit_up.pkl"
    rep = mod.run(panel_dir=panel, out=tmp_path / "o.csv",
                  out_json=tmp_path / "r.json", st_flag=_st_panel(tmp_path, panel),
                  registry_path=reg)
    assert rep["min_valid_days"] == 14
    df = pd.read_csv(tmp_path / "o.csv", dtype=str, keep_default_na=False)
    # s1 2015-01/02（15/14）、s2 2015-01/02/03（16/15/14）在 14 阈值下 complete
    assert int((df["complete_month"] == "1").sum()) == 5
