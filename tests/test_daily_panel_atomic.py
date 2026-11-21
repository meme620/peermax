"""Phase 1.1 / I-4：daily_panel_build 默认拒绝重跑 + 原子写 + 显式 append。"""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/daily_panel_build.py"

RAW_HEADER = ["date", "code", "close", "open", "high", "low", "volume", "pct", "amount"]
QFQ_HEADER = ["date", "code", "close"]
DATES = ["2015-01-05", "2015-01-06", "2015-01-07"]
CLOSES = [10.0, 10.5, 10.2]


def _load():
    spec = importlib.util.spec_from_file_location("build_daily_panel_atomic", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_tencent(tmp_path):
    base = tmp_path / "tencent"
    for code in ("sz000001", "sz000002"):
        _write_csv(
            base / "daily_raw" / code / "2015.csv",
            RAW_HEADER,
            [
                [d, code, str(c), str(c), str(c), str(c), str(1000 + i), "0", str(20000 + i)]
                for i, (d, c) in enumerate(zip(DATES, CLOSES))
            ],
        )
        _write_csv(
            base / "daily_qfq" / code / "2015.csv",
            QFQ_HEADER,
            [[d, code, str(c)] for d, c in zip(DATES, CLOSES)],
        )
    return base


def _prepare(tmp_path, monkeypatch):
    mod = _load()
    tencent = _make_tencent(tmp_path)
    out = tmp_path / "daily_panel"
    report = tmp_path / "report.json"
    monkeypatch.setattr(mod, "TENCENT", tencent)
    monkeypatch.setattr(mod, "OUT_DIR", out)
    monkeypatch.setattr(mod, "REPORT_PATH", report)
    return mod, out, report


def _read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_first_run_writes_header_and_rows(tmp_path, monkeypatch):
    mod, out, report = _prepare(tmp_path, monkeypatch)
    mod.main(["--years", "2015-2015"])

    rows = _read(out / "2015.csv")
    assert rows[0] == mod.COLUMNS == [
        "code", "date", "close_raw", "close_qfq", "volume", "amount", "rraw", "ca_flag", "r_adj",
    ]
    assert len(rows) == 1 + 2 * len(DATES)
    assert rows[1][:3] == ["sz000001", "2015-01-05", "10.0"]
    assert rows[1][6] == ""  # 首日无前收
    assert rows[2][6] == f"{CLOSES[1] / CLOSES[0] - 1:.8f}"
    assert rows[3][6] == f"{CLOSES[2] / CLOSES[1] - 1:.8f}"

    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["mode"] == "default"
    assert summary["rows"] == 2 * len(DATES)
    assert summary["by_year"]["2015"]["firms"] == 2


def test_default_second_run_refused_and_file_unchanged(tmp_path, monkeypatch):
    mod, out, _ = _prepare(tmp_path, monkeypatch)
    mod.main(["--years", "2015-2015"])
    path = out / "2015.csv"
    before = path.read_bytes()

    with pytest.raises(SystemExit) as exc:
        mod.main(["--years", "2015-2015"])
    assert exc.value.code != 0
    assert path.read_bytes() == before
    assert not list(out.glob("*.tmp"))


def test_append_second_run_adds_rows_header_once(tmp_path, monkeypatch):
    mod, out, report = _prepare(tmp_path, monkeypatch)
    mod.main(["--years", "2015-2015"])
    path = out / "2015.csv"
    before_rows = _read(path)

    mod.main(["--years", "2015-2015", "--append", "--append-reason", "测试断点补写"])

    rows = _read(path)
    assert len(rows) == len(before_rows) + 2 * len(DATES)
    assert rows[0] == mod.COLUMNS
    assert sum(1 for r in rows if r == mod.COLUMNS) == 1  # 表头不重复
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["mode"] == "append"
    assert summary["append_reason"] == "测试断点补写"


def test_no_tmp_residue_after_success_and_refusal(tmp_path, monkeypatch):
    mod, out, _ = _prepare(tmp_path, monkeypatch)
    mod.main(["--years", "2015-2015"])
    assert not list(tmp_path.rglob("*.tmp"))

    with pytest.raises(SystemExit):
        mod.main(["--years", "2015-2015"])
    assert not list(tmp_path.rglob("*.tmp"))

    mod.main(["--years", "2015-2015", "--append", "--append-reason", "补写"])
    assert not list(tmp_path.rglob("*.tmp"))


def test_append_requires_reason(tmp_path, monkeypatch):
    mod, out, _ = _prepare(tmp_path, monkeypatch)
    mod.main(["--years", "2015-2015"])
    path = out / "2015.csv"
    before = path.read_bytes()

    with pytest.raises(SystemExit):
        mod.main(["--years", "2015-2015", "--append"])
    assert path.read_bytes() == before
