"""MAX 日样本口径：上市宽限列与涨跌停阈值回退（CO-2026-11-19-parameters）。

三个不变量：
1. 供应商 flag 无覆盖时，阈值回退真的生效（曾经 _day_limit 算完 thr 直接 return False）；
2. MAXraw 保持 headline 口径，上市宽限只作用于 MAXraw_la；
3. 宽限窗口内的极值日被剔除后，n_excluded_listing 与 rows_where_la_differs 如实计数。
"""
import csv
import importlib.util
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "peermax_max_build.py"


def _load(name="step1_listing_grace"):
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


def _flag_pickle(path, dates, codes):
    pd.DataFrame({f"{c}.SZ": [False] * len(dates) for c in codes},
                 index=pd.to_datetime(dates)).to_pickle(path)
    return path


def _registry(tmp_path, grace, limit_hit="exclude", resume="exclude"):
    doc = {"registry_version": "test-listing", "parameters": [
        {"name": "min_valid_days", "headline_value": 1},
        {"name": "tie_rule", "headline_value": "earliest", "sensitivity_grid": ["earliest"]},
        {"name": "limit_thresholds",
         "headline_value": {"main": 0.095, "main_st": 0.045, "chinext_star": 0.195}},
        {"name": "max_day_policy",
         "headline_value": {"limit_hit": limit_hit, "resume": resume,
                            "listing_grace_days": grace}},
    ]}
    p = tmp_path / f"reg_grace{grace}_{limit_hit}.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def _panel(tmp_path, rows):
    panel = tmp_path / "daily_panel"
    _write_csv(panel / "2020.csv", ["code", "date", "volume", "rraw"], rows)
    dates = sorted({r[1] for r in rows})
    codes = sorted({r[0][2:] for r in rows})
    st = _flag_pickle(tmp_path / "st.pkl", dates, codes)
    return panel, st


def _run(tmp_path, mod, panel, st, reg, flags_path):
    mod.LIMIT_HIT_FLAG = flags_path
    out = tmp_path / "source_month.csv"
    rep_path = tmp_path / "report.json"
    mod.run(panel_dir=panel, out=out, out_json=rep_path, st_flag=st, registry_path=reg)
    rows = {r["month"]: r for r in csv.DictReader(open(out, encoding="utf-8", newline=""))}
    return rows, json.loads(rep_path.read_text(encoding="utf-8"))


IPO_ROWS = [
    ["sz000001", "2020-01-02", "100", "0.4400"],
    ["sz000001", "2020-01-03", "100", "0.0100"],
    ["sz000001", "2020-01-06", "100", "-0.0200"],
    ["sz000001", "2020-01-07", "100", "0.0300"],
]


def test_listing_grace_keeps_headline_and_flags_la(tmp_path):
    mod = _load()
    panel, st = _panel(tmp_path, IPO_ROWS)
    # limit_hit/resume 取 include：本用例只验上市宽限，避免 0.44 也被阈值回退剔掉
    rows, rep = _run(tmp_path, mod, panel, st,
                     _registry(tmp_path, grace=1, limit_hit="include", resume="include"),
                     tmp_path / "no_flags.pkl")
    r = rows["2020-01"]
    assert r["MAXraw"] == "0.44"
    assert r["MAXraw_la"] == "0.03"
    assert r["n_excluded_listing"] == "1"
    assert rep["listing_grace_days"] == 1
    assert rep["rows_where_la_differs"] == 1
    assert rep["n_excluded_listing_total"] == 1


def test_zero_grace_makes_la_identical_to_headline(tmp_path):
    mod = _load()
    panel, st = _panel(tmp_path, IPO_ROWS)
    rows, rep = _run(tmp_path, mod, panel, st,
                     _registry(tmp_path, grace=0, limit_hit="include", resume="include"),
                     tmp_path / "no_flags.pkl")
    r = rows["2020-01"]
    assert r["MAXraw"] == r["MAXraw_la"] == "0.44"
    assert r["n_excluded_listing"] == "0"
    assert rep["rows_where_la_differs"] == 0


def test_listing_count_ignores_days_already_excluded_by_limit(tmp_path):
    """同时被涨跌停/复牌剔除的日子不重复计入 n_excluded_listing。"""
    mod = _load()
    rows_in = [
        ["sz000001", "2020-01-02", "100", "0.4400"],   # 首日：宽限 + 阈值两边都会剔
        ["sz000001", "2020-01-03", "100", "0.0100"],
        ["sz000001", "2020-01-06", "100", "0.0200"],
    ]
    panel, st = _panel(tmp_path, rows_in)
    rows, _ = _run(tmp_path, mod, panel, st,
                   _registry(tmp_path, grace=1, limit_hit="exclude", resume="include"),
                   tmp_path / "no_flags.pkl")
    # 宽限窗只有首日，而首日已被 limit_hit 剔掉 → 宽限的边际剔除数为 0
    assert rows["2020-01"]["n_excluded_listing"] == "0"
    assert rows["2020-01"]["n_excluded_limit"] == "1"


def test_threshold_fallback_effective_without_vendor_flag(tmp_path):
    """无涨跌停 flag 时阈值回退必须真的判 limit_hit（死代码回归）。"""
    mod = _load()
    rows_in = [
        ["sz000001", "2020-01-02", "100", "0.1000"],
        ["sz000001", "2020-01-03", "100", "0.0200"],
    ]
    panel, st = _panel(tmp_path, rows_in)
    rows, rep = _run(tmp_path, mod, panel, st,
                     _registry(tmp_path, grace=0, limit_hit="exclude", resume="include"),
                     tmp_path / "no_flags.pkl")
    r = rows["2020-01"]
    # 来源计数按 source-month 行计（不是按日）：本用例只有 1 行
    assert rep["limit_hit_source_counts"]["derived_threshold"] == 1
    assert rep["limit_hit_source_counts"]["vendor_flag"] == 0
    assert r["n_excluded_limit"] == "1"
    assert r["MAXraw"] == "0.02"
    assert r["MAXraw_incl_all"] == "0.1"
