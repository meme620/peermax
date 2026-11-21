"""change order 登记校验（peerlab.change_order）的负向用例。

每条不变量都要有一个「确实违规」的用例：只测通过路径等于没测，因为登记表
平时就是通过的，校验器失效不会被发现。
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
REG = ROOT / "configs" / "parameter_registry.yaml"

import sys
sys.path.insert(0, str(ROOT))
from peerlab import change_order  # noqa: E402


def _doc():
    return yaml.safe_load(REG.read_text(encoding="utf-8"))


def _write(tmp_path, doc):
    p = tmp_path / "configs" / "parameter_registry.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return tmp_path


def _base():
    """最小可通过的 registry：一条完整 change order。"""
    return {
        "registry_version": "1.0.1-x",
        "change_log": [{
            "id": "CO-2026-01-01-init",
            "date": "2026-01-01",
            "scope": "初始登记",
            "rationale": "建立基线",
            "params": ["alpha"],
            "version_from": "1.0.0-x",
            "version_to": "1.0.1-x",
            "approval": "test",
        }],
        "parameters": [{"name": "alpha", "headline_value": 1}],
    }


def test_baseline_passes(tmp_path):
    assert change_order.check(_write(tmp_path, _base())) == []


def test_real_registry_passes():
    assert change_order.check(ROOT) == []


def test_missing_field_caught(tmp_path):
    d = _base()
    del d["change_log"][0]["rationale"]
    problems = change_order.check(_write(tmp_path, d))
    assert any("rationale" in p for p in problems), problems


def test_registry_version_must_match_latest(tmp_path):
    d = _base()
    d["registry_version"] = "2.0.0-x"
    problems = change_order.check(_write(tmp_path, d))
    assert any("registry_version" in p for p in problems), problems


def test_version_to_must_increase(tmp_path):
    d = _base()
    d["change_log"][0]["version_to"] = "1.0.0-x"
    problems = change_order.check(_write(tmp_path, d))
    assert any("不大于" in p for p in problems), problems


def test_unknown_param_caught(tmp_path):
    d = _base()
    d["change_log"][0]["params"] = ["nope"]
    problems = change_order.check(_write(tmp_path, d))
    assert any("nope" in p for p in problems), problems


def test_subkey_path_accepted(tmp_path):
    """父参数.子键 是可定位路径；只写子键名必须被拒。"""
    d = _base()
    d["parameters"] = [{"name": "alpha", "headline_value": {"beta": 2}}]
    d["change_log"][0]["params"] = ["alpha.beta"]
    assert change_order.check(_write(tmp_path, d)) == []

    d["change_log"][0]["params"] = ["beta"]
    problems = change_order.check(_write(tmp_path, d))
    assert any("beta" in p for p in problems), problems


def test_empty_change_caught(tmp_path):
    d = _base()
    d["change_log"][0]["params"] = []
    problems = change_order.check(_write(tmp_path, d))
    assert any("空改" in p for p in problems), problems


def test_id_format_and_duplicate(tmp_path):
    d = _base()
    d["change_log"][0]["id"] = "bad-id"
    problems = change_order.check(_write(tmp_path, d))
    assert any("CO-YYYY-MM-DD-slug" in p for p in problems), problems

    d = _base()
    d["change_log"].append(dict(d["change_log"][0]))
    problems = change_order.check(_write(tmp_path, d))
    assert any("重复" in p for p in problems), problems


def test_date_order_latest_on_top(tmp_path):
    d = _base()
    d["change_log"].insert(0, {
        "id": "CO-2026-02-01-later",
        "date": "2026-02-01", "scope": "s", "rationale": "r",
        "params": ["alpha"], "version_from": "1.0.1-x", "version_to": "1.0.2-x",
    })
    d["registry_version"] = "1.0.2-x"
    assert change_order.check(_write(tmp_path, d)) == []   # 最新在上，合法

    d["change_log"] = d["change_log"][::-1]                # 倒置 -> 违规
    problems = change_order.check(_write(tmp_path, d))
    assert any("最新在上" in p for p in problems), problems


def test_missing_change_log_caught(tmp_path):
    d = _base()
    del d["change_log"]
    problems = change_order.check(_write(tmp_path, d))
    assert any("change_log" in p for p in problems), problems


def test_version_chain_orphan_caught(tmp_path):
    """两条互不相接的 change order：两个 version_from 都没有 version_to 指向。"""
    d = _base()
    d["change_log"].append({
        "id": "CO-2026-01-02-other",
        "date": "2026-01-02", "scope": "s", "rationale": "r",
        "params": ["alpha"], "version_from": "9.0.0-x", "version_to": "9.0.1-x",
    })
    d["registry_version"] = "9.0.1-x"
    problems = change_order.check(_write(tmp_path, d))
    assert any("版本链断裂" in p for p in problems), problems
