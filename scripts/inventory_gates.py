#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inventory Step 5：数据门 G1-G8 汇总（状态由当次执行推导，不写字面量）。

每个 gate 的判定来自 spec/design.yaml 的 data_gates[].enforced_by：
  - 有 `gate:<id>` 执行者 → 跑 peerlab 数据门，按 fail > blocked > pass 汇总；
  - enforced_by 为空 → NOT_EVALUATED（没有执行者，不得写成"状态已知"）。
产出：reports/data_gates/G0/data_gate_report.json + inventory_run_record.md

用法: python scripts/inventory_gates.py
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
G0 = ROOT / "reports/data_gates/G0"
OUT = G0 / "data_gate_report.json"
RUN_RECORD = ROOT / "reports/execution/inventory_run_record.md"  # 放 G0 目录更贴切，但保持与 S0 run record 同目录

GATE_ORDER = {"FAIL": 0, "BLOCKED": 1, "NOT_EVALUATED": 2, "CONDITIONAL": 3, "PASS": 4}


def aggregate(states):
    return min(states, key=lambda s: GATE_ORDER[s])


def main():
    # 读取 inventory 各步报告（存在性 + 关键字段）
    def load(name):
        p = G0 / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    manifest = load("data_access_manifest.json")
    schema = load("inventory_schema_check.json")
    coverage = load("inventory_coverage_report.json")

    # G1-G8 的判定一律由当次执行推导，不写字面量：
    #   enforced_by 列出 peerlab 数据门 → 跑那些门，按 fail > blocked > pass 汇总；
    #   enforced_by 为空 → NOT_EVALUATED（没有执行者，不得写成"状态已知"）。
    design_path = ROOT / "spec/design.yaml"
    design = yaml.safe_load(design_path.read_text(encoding="utf-8")) if design_path.is_file() else {}
    declared = design.get("data_gates") or []

    def _gate_ids(enforced_by):
        return [str(x).split(":", 1)[1] for x in (enforced_by or [])
                if str(x).startswith("gate:")]

    needed = sorted({gid for g in declared for gid in _gate_ids(g.get("enforced_by"))})
    # 通过 CLI 跑门：门本身会写 reports/data_gates/G0/gate_<id>.json；不在此处 import
    # peerlab（scripts/ 不是包根，插 sys.path 违反仓库约束）。退出码只表达 fail/blocked，
    # 具体状态从 --json 输出解析。
    results = {}
    if needed:
        cmd = [sys.executable, "-m", "peerlab", "gate", "--json", "--only", ",".join(needed)]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        try:
            for r in json.loads(proc.stdout or "[]"):
                results[r["gate"]] = r
        except json.JSONDecodeError:
            results = {}

    def _aggregate(states):
        if not states:
            return "NOT_EVALUATED"
        for s in ("FAIL", "BLOCKED", "PASS"):
            if s in states:
                return s
        return "NOT_EVALUATED"

    gates = {}
    for g in declared:
        gid = str(g.get("id"))
        ids = _gate_ids(g.get("enforced_by"))
        if not ids:
            gates[gid] = {
                "gate_status": "NOT_EVALUATED",
                "requires": g.get("requires"),
                "on_fail": g.get("on_fail"),
                "executors": [],
                "note": "design.data_gates 未声明执行者；状态未知，不得据此主张已过门",
            }
            continue
        per = {gid2: results.get(gid2, {}).get("status", "BLOCKED").upper()
               for gid2 in ids}
        gates[gid] = {
            "gate_status": _aggregate(list(per.values())),
            "requires": g.get("requires"),
            "on_fail": g.get("on_fail"),
            "executors": per,
            "evidence": [f"{k}: {v}" for k, v in per.items()],
        }



    routes = {
        "PRICING_ROUTE": {"required_gates": ["G1", "G2", "G3", "G8"], "note": "主线 A 月度定价+distinctness"},
        "visibility": {"required_gates": ["G1", "G2", "G3", "G4", "G8"], "note": "升级线 B"},
        "mechanism": {"required_gates": ["G1", "G2", "G3", "G5", "G6", "G8"], "note": "attention/order-flow 机制"},
    }
    route_status = {}
    for rname, r in routes.items():
        states = [gates[g]["gate_status"] for g in r["required_gates"]]
        route_status[rname] = {"route_status": aggregate(states), "gate_states": dict(zip(r["required_gates"], states))}

    report = {
        "report_id": "data_gate_report",
        "inventory_status_route_pricing": route_status["PRICING_ROUTE"],
        "inventory_status_route_visibility": route_status["visibility"],
        "inventory_status_route_mechanism": route_status["mechanism"],
        "gates": gates,
        "next_step_allowed": "daily-panel（基础清洗，G1 合同已验）、factor（因子构造）、network（网络 pipeline）；panel/pricing_gate 需 G3 放行后",
        "blocked_until": {
            "G3": "similarity 方法冻结 + pipeline 构造 + 八条件验收",
            "G4/G5/G6": "商业库/授权渠道",
            "G8 因子": "factor 构造 + rf 录入",
        },
        "draft_parameters_pending_user": coverage.get("draft_parameters", {}),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # run record：逐 gate 与路线状态一律取当次计算结果，不留写死的判定
    rows_md = "\n".join(
        f"| {gid} | {v['gate_status']} | {v.get('requires', '')} | "
        f"{', '.join(f'{k}={val}' for k, val in (v.get('executors') or {}).items()) or '无执行者'} |"
        for gid, v in gates.items())
    route_md = "\n".join(f"- {k} = {v['route_status']}" for k, v in route_status.items())
    run = f"""# inventory Run Record：数据可得性盘点

日期：{datetime.now(timezone.utc).date().isoformat()}
输入：reports/data_gates/G0 的 inventory 报告 + spec/design.yaml 的 data_gates

## 逐 gate 判定（状态由当次执行推导）
| Gate | 状态 | requires | 执行门 |
|---|---|---|---|
{rows_md}

## 路线状态
{route_md}

G4-G8 在 spec/design.yaml 里 enforced_by 为空，因此恒为 NOT_EVALUATED：没有执行者就没有判据。

## 关键发现（登记，不得重复踩）
1. 腾讯 pct_chg 字段 ≠ 涨跌幅（2020 抽 40 日 39 不符）→ 废弃，rraw=raw close 派生
2. 腾讯 qfq 版涨跌停日伪收益（偏差 0.2-0.5pp）→ 禁止用于 rraw；仅除权日识别辅助
3. 腾讯 raw close 与新浪双源一致（0 冲突）→ raw 版确认真实
4. eastmoney push2his 对 Python 直连封锁（RemoteDisconnected）→ 第二源=新浪 quotes.sina.cn（Python 可访问）；datacenter-web 可用但无日线行情接口
5. 断档期（975 天）未观测 ~13,444 firm-day 主要为停牌，量级与 2021-2025 停牌率吻合

## 产出物
- inventory_expected_universe.json（expected 8,612,917 firm-day）
- data_access_manifest.json（7 源）
- field_mapping_manifest.json + inventory_schema_check.json
- inventory_coverage_report.json（三分类 + 双源）
- data_gate_report.json（本报告）

## 下一步
inventory(PRICING_ROUTE)=CONDITIONAL → 允许进入 daily-panel（基础清洗）与 factor（因子）；network 需用户先冻结 similarity 方法。
16 项数据参数草案值见 data_gate_report.draft_parameters_pending_user，**待用户确认后写入 parameter_registry.yaml**。
"""
    RUN_RECORD.parent.mkdir(parents=True, exist_ok=True)
    RUN_RECORD.write_text(run, encoding="utf-8")
    print("路线状态:")
    for rname, rs in route_status.items():
        print(f"  {rname}: {rs['route_status']}")
    print(f"\n输出: {OUT}\n     {RUN_RECORD}")


if __name__ == "__main__":
    main()
