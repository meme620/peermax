"""Phase 1 / I-15：tie_rule 从 registry 读取（不再写死 earliest）。

覆盖：
- 当前 registry 的 `tie_rule.headline_value` 解析为 `earliest`；
- 默认 registry 下 CSV 输出与旧逻辑（min + 写死 earliest）逐字节一致；
- 临时 registry “最晚/最新/latest” → `latest`，并列 argmax 取最晚日期；
- 缺参数 / 重名 / 无法识别 / 多义 / random / grid 冲突 → SystemExit(2)；
- tie rule 只影响并列日期，MAX/R/n_valid_days/limit/ST 逻辑不变。

全部使用 tmp_path 合成 panel 与临时 registry；不读写 `data/`、`reports/`。
"""
import csv
import importlib.util
import io
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/peermax_max_build.py"
REGISTRY = ROOT / "configs/parameter_registry.yaml"

PANEL_HEADER = ["code", "date", "rraw"]


def _load(name="s4_step1_max_i15"):
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _tie_panel(tmp_path):
    """sh600001 在 2015-01 最大值 0.10 并列于 01-05 / 01-20；sz000002 无并列。"""
    rows = [
        ["sh600001", "2015-01-05", "0.10000000"],
        ["sh600001", "2015-01-12", "-0.02000000"],
        ["sh600001", "2015-01-20", "0.10000000"],
        ["sz000002", "2015-02-03", "0.01000000"],
        ["sz000002", "2015-02-10", "0.03000000"],
    ]
    panel_dir = tmp_path / "daily_panel"
    _write_csv(panel_dir / "2015.csv", PANEL_HEADER, rows)
    return panel_dir, rows


def _write_registry(tmp_path, parameters, filename="parameter_registry.yaml"):
    """写最小临时 registry；parameters 为 list[dict]。"""
    p = tmp_path / filename
    doc = {"registry_version": "test-1.9", "parameters": parameters}
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


def _tie_entry(headline, grid=("earliest", "latest", "random"), **extra):
    entry = {"name": "tie_rule", "headline_value": headline}
    if grid is not None:
        entry["sensitivity_grid"] = list(grid)
    entry.update(extra)
    return entry


def _legacy_csv_text(panel_rows):
    """I-15 之前的 s4_step1 逻辑：earliest 写死，无 gate。用于逐字节对拍。"""
    monthly = defaultdict(list)
    for code, date, rraw in panel_rows:
        if not rraw:
            continue
        monthly[(code, date[:7])].append((date, float(rraw)))
    out_rows = []
    for (code, month), days in monthly.items():
        days.sort()
        rraws = [x[1] for x in days]
        maxv = max(rraws)
        argmax_d = min(d for d, r in days if r == maxv)  # tie_rule=earliest
        prod = 1.0
        for x in rraws:
            prod *= 1 + x
        out_rows.append({"source_id": code, "month": month,
                         "MAXraw": round(maxv, 8), "argmax_date": argmax_d,
                         "n_valid_days": len(days), "R": round(prod - 1, 8),
                         "tie_rule_version": "earliest", "formula_id": "MAXRAW_v1"})
    for r in out_rows:
        maxv = r["MAXraw"]
        unknown = 0.05 <= maxv < 0.095
        lh = 1 if maxv >= 0.095 else 0
        r.update({"limit_hit_at_argmax": lh,
                  "qualification_status": ("limit_status_unknown" if unknown else
                                           ("qualified_limit_hit" if lh else "qualified_non_limit")),
                  "st_flag_at_argmax": ""})
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
    return buf.getvalue()


def _read_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_text(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _run(mod, tmp_path, **kwargs):
    out = tmp_path / "source_month.csv"
    report_path = tmp_path / "s4_report.json"
    report = mod.run(panel_dir=kwargs.pop("panel_dir"),
                     out=out, out_json=report_path,
                     st_flag=tmp_path / "no_such_st.pkl",
                     min_valid_days=kwargs.pop("min_valid_days", 15), **kwargs)
    return out, report


# ---------------------------------------------------------------------------
# 当前 registry / 默认零变化
# ---------------------------------------------------------------------------

def test_current_registry_resolves_earliest(mod):
    info = mod.resolve_tie_rule()
    assert info["id"] == "earliest"
    assert info["source"] == "registry.tie_rule.headline_value"
    assert mod.load_tie_rule() == "earliest"

    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    entry = [p for p in reg["parameters"] if p.get("name") == "tie_rule"]
    assert len(entry) == 1
    assert info["headline_value"] == entry[0]["headline_value"]
    assert info["sensitivity_grid"] == entry[0]["sensitivity_grid"]
    assert info["allowed_in_sensitivity_grid"] == [str(x).lower() for x in entry[0]["sensitivity_grid"]]
    assert set(info["allowed_in_sensitivity_grid"]) <= {"earliest", "latest"}


def test_default_core_fields_match_legacy_and_new_gate_columns(mod, tmp_path):
    """P1.1-01 后不再逐字节一致（n_valid_days 改 volume>0、追加 gate 列）；
    旧 11 列中 MAX/R/argmax/limit/tie 与旧逻辑一致；合成 fixture 无 volume → n_valid_days=0。"""
    panel_dir, panel_rows = _tie_panel(tmp_path)
    out, report = _run(mod, tmp_path, panel_dir=panel_dir)
    got = {r["month"]: r for r in _read_rows(out)}
    legacy = {r["month"]: r for r in csv.DictReader(io.StringIO(_legacy_csv_text(panel_rows)))}
    assert set(got) == set(legacy)
    for month in got:
        for col in ("source_id", "MAXraw", "argmax_date", "R", "formula_id",
                    "limit_hit_at_argmax", "qualification_status", "tie_rule_version"):
            assert got[month][col] == legacy[month][col], (month, col)
        assert got[month]["n_valid_days"] == "0"          # volume 缺失 → volume>0 计数 0
        assert got[month]["n_rraw_days"] == legacy[month]["n_valid_days"]
        assert got[month]["complete_month"] == "0"
        assert got[month]["max_status"] == "incomplete"   # D-02=A 保留 MAX
        assert got[month]["MAXraw"] != ""
    assert {r["tie_rule_version"] for r in got.values()} == {"earliest"}
    assert report["tie_rule"]["id"] == "earliest"
    assert report["tie_rule"]["source"] == "registry.tie_rule.headline_value"
    assert report["min_valid_days"] == 15


def test_earliest_and_latest_only_differ_on_tied_argmax(mod, tmp_path):
    panel_dir, _ = _tie_panel(tmp_path)
    early_dir = tmp_path / "early"
    early_dir.mkdir()
    out_e, _ = _run(mod, early_dir, panel_dir=panel_dir)

    reg_late = _write_registry(
        tmp_path, [_tie_entry("选取月内最晚达到该最大值的交易日作为 d*_{j,t}")],
        filename="registry_latest.yaml")
    late_dir = tmp_path / "late"
    late_dir.mkdir()
    out_l, report_l = _run(mod, late_dir, panel_dir=panel_dir, registry_path=reg_late)

    e = {r["month"]: r for r in _read_rows(out_e)}
    l = {r["month"]: r for r in _read_rows(out_l)}

    # 并列月：earliest=01-05，latest=01-20
    assert e["2015-01"]["argmax_date"] == "2015-01-05"
    assert l["2015-01"]["argmax_date"] == "2015-01-20"
    assert e["2015-01"]["tie_rule_version"] == "earliest"
    assert l["2015-01"]["tie_rule_version"] == "latest"
    # 无并列月：两种规则同一日期
    assert e["2015-02"]["argmax_date"] == l["2015-02"]["argmax_date"] == "2015-02-10"

    # 除 argmax_date / tie_rule_version 外逐字段一致（MAX/R/limit/ST 逻辑不变）
    for month in e:
        for col in e[month]:
            if col in ("argmax_date", "tie_rule_version"):
                continue
            assert e[month][col] == l[month][col], (month, col)
    assert report_l["tie_rule"]["id"] == "latest"


# ---------------------------------------------------------------------------
# 解析映射
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("headline,expected", [
    ("earliest", "earliest"),
    ("latest", "latest"),
    ("选取月内最早达到该最大值的交易日作为 d*", "earliest"),
    ("选取月内最晚达到该最大值的交易日作为 d*", "latest"),
    ("选取月内最新达到该最大值的交易日作为 d*", "latest"),
])
def test_parse_headline_keywords(mod, tmp_path, headline, expected):
    reg = _write_registry(tmp_path, [_tie_entry(headline)])
    assert mod.load_tie_rule(reg) == expected


def test_explicit_tie_rule_id_wins_over_headline(mod, tmp_path):
    reg = _write_registry(tmp_path, [_tie_entry("选取月内最早达到该最大值", tie_rule_id="latest")])
    info = mod.resolve_tie_rule(reg)
    assert info["id"] == "latest"
    assert info["source"] == "registry.tie_rule.tie_rule_id"


def test_pick_argmax_helper(mod):
    days = [("2015-01-05", 0.1), ("2015-01-20", 0.1), ("2015-01-12", 0.05)]
    assert mod.pick_argmax(days, "earliest") == "2015-01-05"
    assert mod.pick_argmax(days, "latest") == "2015-01-20"
    no_tie = [("2015-01-05", 0.01), ("2015-01-20", 0.2)]
    assert mod.pick_argmax(no_tie, "earliest") == mod.pick_argmax(no_tie, "latest") == "2015-01-20"


# ---------------------------------------------------------------------------
# 失败路径：全部 SystemExit(2)
# ---------------------------------------------------------------------------

def _expect_exit2(fn, *args, **kwargs):
    with pytest.raises(SystemExit) as ei:
        fn(*args, **kwargs)
    assert ei.value.code == 2


def test_unrecognized_headline_exits_2(mod, tmp_path):
    reg = _write_registry(tmp_path, [_tie_entry("按收盘价选择交易日")])
    _expect_exit2(mod.load_tie_rule, reg)


def test_missing_tie_rule_param_exits_2(mod, tmp_path):
    reg = _write_registry(tmp_path, [{"name": "other", "headline_value": "earliest"}])
    _expect_exit2(mod.load_tie_rule, reg)
    _expect_exit2(mod.load_tie_rule, tmp_path / "no_such_registry.yaml")


def test_duplicate_tie_rule_param_exits_2(mod, tmp_path):
    entry = _tie_entry("选取月内最早达到该最大值的交易日")
    reg = _write_registry(tmp_path, [entry, dict(entry)])
    _expect_exit2(mod.load_tie_rule, reg)


def test_missing_headline_and_id_exits_2(mod, tmp_path):
    reg = _write_registry(tmp_path, [{"name": "tie_rule"}])
    _expect_exit2(mod.load_tie_rule, reg)


def test_random_tie_rule_not_implemented_exits_2(mod, tmp_path):
    for headline in ("随机选取并列交易日", "random"):
        reg = _write_registry(tmp_path, [_tie_entry(headline)],
                              filename=f"reg_{headline[:2]}.yaml")
        _expect_exit2(mod.load_tie_rule, reg)


def test_ambiguous_tie_text_exits_2(mod, tmp_path):
    reg = _write_registry(tmp_path, [_tie_entry("选取最早或最晚的交易日")])
    _expect_exit2(mod.load_tie_rule, reg)


def test_grid_mismatch_exits_2(mod, tmp_path):
    reg = _write_registry(tmp_path, [_tie_entry("选取月内最早达到该最大值", grid=["latest"])])
    _expect_exit2(mod.load_tie_rule, reg)


def test_invalid_explicit_tie_rule_id_exits_2(mod, tmp_path):
    reg = _write_registry(tmp_path, [_tie_entry("选取月内最早", tie_rule_id="middle")])
    _expect_exit2(mod.load_tie_rule, reg)


def test_run_fails_before_writing_on_bad_registry(mod, tmp_path):
    panel_dir, _ = _tie_panel(tmp_path)
    reg = _write_registry(tmp_path, [_tie_entry("按收盘价选择")])
    out = tmp_path / "out.csv"
    with pytest.raises(SystemExit) as ei:
        mod.run(panel_dir=panel_dir, out=out, out_json=tmp_path / "r.json",
                st_flag=tmp_path / "no.pkl", registry_path=reg, min_valid_days=15)
    assert ei.value.code == 2
    assert not out.exists()
