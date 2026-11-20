#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""network Phase B′ 主口径专项核验（老师 MD&A 语料网络，2026-09-22）。

替代"改造三个自建链审计脚本"的方案：对**新主口径**做等价核验，覆盖
主仓库 methodology_execution_plan §8「Phase B′ 后续」第 1 项要求的层次（数据/公式/语义），
并显式记录自建链审计脚本仍指向旧目录这一事实。

核验项：
  A. τ ↔ registry 一致性：各年 build_report.tau == registry `m_b_tau_fixed`（0.0888212137），
     且边表最小 similarity > τ（边集规则 `sim > τ` 的必要条件）。
  B. 时点链：available_at == (text_year+1)-04-30；network_version == pm_mb_tfidf_teacher_v1_{y}；
     source_version == teacher_mda_2023。
  C. 语料对账：各年入网节点数 == teacher 面板该年 ok 行数（与验收脚本独立复算）。
  D. 分层语义代理：按权重层（top1 / 高权 / 随机 / 中位）抽样，比较同申万二级边占比——
     高权层应显著高于随机层（"高相似度更像真竞争者"的代理证据，非人工 precision）。

产物：reports/data_gates/G0/network_mainline_audit.json
用法：python scripts/network_mainline_audit.py
"""
from __future__ import annotations

import csv
import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
# 两个主口径列：M-A（等权 0/1 余弦 + 同团规则，无 τ）与 M-B（TF-IDF + 固定 τ）。
# 旧版本只审 M-B，headline 主口径（M-A）没有等价主线审计。
NET_SPECS = {
    'ma_teacher': {'dir': ROOT / 'data/processed/network_ma_teacher',
                   'version_prefix': 'pm_ma_ewcos_teacher_v1',
                   'source_version': 'teacher_mda_panel+product_lexicon_v2_teacher',
                   'uses_tau': False, 'filter_years': True, 'years': list(range(2014, 2023))},
    'mb_teacher': {'dir': ROOT / 'data/processed/network_mb_teacher',
                   'version_prefix': 'pm_mb_tfidf_teacher_v1',
                   'source_version': 'teacher_mda_2023',
                   'uses_tau': True, 'years': list(range(2014, 2023))},
}
NET = NET_SPECS['mb_teacher']['dir']
PANEL = ROOT / "data/processed/teacher_mda_panel.csv"
SW2 = ROOT / "data/processed/sw2_industry_latest.csv"
REG = ROOT / "configs/parameter_registry.yaml"
REPD = ROOT / "reports/data_gates/G0"
YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
csv.field_size_limit(10**9)


def registry_tau() -> float:
    import yaml
    d = yaml.safe_load(REG.read_text(encoding="utf-8"))
    for p in d["parameters"]:
        if p["name"] == "m_b_tau_fixed":
            return float(p["headline_value"])
    raise KeyError("m_b_tau_fixed 未在 registry")


def audit_column(key: str, tau_reg: float, l2: dict, panel_ok) -> dict:
    """对一列网络做逐年核验（τ 只对 M-B 是判据；M-A 的边集规则是「同团 ∧ sim>0」）。"""
    spec = NET_SPECS[key]
    net_dir, uses_tau = spec['dir'], spec['uses_tau']
    per_year, problems = [], []
    for y in spec['years']:
        rep = json.loads((net_dir / f"{y}_build_report.json").read_text(encoding="utf-8"))
        e = pd.read_csv(net_dir / f"{y}_similarity.csv",
                        usecols=["focal_id", "source_id", "similarity", "weight",
                                 "available_at", "network_version", "source_version"])
        # 节点总数 = 有边 focal（support）+ 无支持 focal（no-network-support），与验收脚本口径一致
        sup = pd.read_csv(net_dir / f"{y}_support.csv", usecols=["focal_id", "network_support_status"])
        n_support = int((sup.network_support_status == "support").sum())
        n_nosup = int((sup.network_support_status == "no-network-support").sum())
        nodes = n_support + n_nosup
        n_expect = int((panel_ok["year"] == y).sum())
        min_sim = float(e.similarity.min())

        # D. 分层语义代理（该年，抽 200/层）
        rng = np.random.default_rng(2026 + y)
        g = e.groupby("focal_id")
        top1 = g.apply(lambda t: t.nlargest(1, "weight"), include_groups=False).reset_index(level=0)
        layers = {}
        for name, df in [
            ("top1", top1),
            ("high", e[e.weight >= e.weight.quantile(0.9)]),
            ("random", e.sample(n=min(200 * 8, len(e)), random_state=2026)),
            ("mid", e[(e.weight >= e.weight.quantile(0.45)) & (e.weight <= e.weight.quantile(0.55))]),
        ]:
            d = df if len(df) <= 200 else df.sample(200, random_state=2026)
            fa = d.focal_id.map(lambda f: l2.get(str(f)[2:].zfill(6)))
            fb = d.source_id.map(lambda f: l2.get(str(f)[2:].zfill(6)))
            ok = fa.notna() & fb.notna()
            layers[name] = round(float((fa[ok] == fb[ok]).mean()), 4) if ok.any() else None

        row = {
            "text_year": y, "network_year": y + 1,
            "tau_in_report": rep.get("tau") if uses_tau else None,
            "tau_registry": tau_reg if uses_tau else None,
            "tau_match": (rep.get("tau") == tau_reg) if uses_tau else None,
            "min_similarity": round(min_sim, 10),
            "all_edges_above_tau": (min_sim > tau_reg) if uses_tau else None,
            "all_edges_positive_sim": bool(min_sim > 0),
            "available_at": sorted(set(e.available_at.astype(str))),
            "version": sorted(set(e.network_version.astype(str))),
            "source_version": sorted(set(e.source_version.astype(str))),
            "nodes": nodes, "nodes_support": n_support, "nodes_no_support": n_nosup,
            "nodes_panel_ok": n_expect,
            # M-B 不剔 J/S 与次新股（仅按 status=ok 入网），故节点数必须等于面板 ok 行数；
            # M-A 先剔 J/S + 上市<2 年 + 词数不足，节点数只能等于 build_report.filter_stats.firms_kept。
            "nodes_expected": (int((rep.get('filter_stats') or {}).get('firms_kept', nodes))
                               if spec.get('filter_years') else n_expect),
            "nodes_match_panel": (nodes == int((rep.get('filter_stats') or {}).get('firms_kept', nodes))
                                  if spec.get('filter_years') else nodes == n_expect),
            "same_sw2_by_layer": layers,
        }
        row["ok"] = (row["available_at"] == [f"{y+1}-04-30"]
                     and row["version"] == [f"{spec['version_prefix']}_{y}"]
                     and row["source_version"] == [spec['source_version']]
                     and row["nodes_match_panel"]
                     and (row["tau_match"] and row["all_edges_above_tau"] if uses_tau
                          else row["all_edges_positive_sim"]))
        if not row["ok"]:
            problems.append(y)
        per_year.append(row)
        print(f"[{y}] tau={row['tau_in_report']} match={row['tau_match']} "
              f"min_sim={row['min_similarity']:.6f} nodes={nodes}/{n_expect} "
              f"layer same_sw2={layers} ok={row['ok']}", flush=True)

    return {"network_dir": str(net_dir.relative_to(ROOT)),
            "uses_tau": uses_tau, "tau_registry": tau_reg if uses_tau else None,
            "per_year": per_year,
            "summary": {
                "years_ok": sum(1 for r in per_year if r["ok"]),
                "years_total": len(per_year),
                "problems": problems,
                "layer_monotonic_evidence": {
                    "top1_median": float(np.median([r["same_sw2_by_layer"]["top1"] for r in per_year])),
                    "random_median": float(np.median([r["same_sw2_by_layer"]["random"] for r in per_year])),
                },
            },
            "status": "PASS" if not problems else "CHECK"}


def main() -> None:
    tau_reg = registry_tau()
    sw = pd.read_csv(SW2, dtype={"code6": str})
    sw["code6"] = sw["code6"].str.zfill(6)
    l2 = dict(zip(sw["code6"], sw["l2_code"].astype(str)))
    panel = pd.read_csv(PANEL, usecols=["firm_id", "year", "status"])
    panel_ok = panel[panel["status"] == "ok"]

    columns = {}
    for key in NET_SPECS:
        if not (NET_SPECS[key]['dir'] / f"{NET_SPECS[key]['years'][0]}_similarity.csv").is_file():
            columns[key] = {"status": "NOT_AVAILABLE",
                            "reason": f"缺 {NET_SPECS[key]['dir'].name} 边表", "per_year": []}
            continue
        print(f"=== {key} ===", flush=True)
        columns[key] = audit_column(key, tau_reg, l2, panel_ok)

    out = {
        "step": "network Phase B′ 主口径专项核验（M-A 与 M-B 两列）",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "tau_registry": tau_reg,
        "columns": columns,
        "status": ("PASS" if all(c["status"] == "PASS" for c in columns.values())
                   else "CHECK"),
        "legacy_audits_status": {
            "note": "自建语料的 network_ma / network_mb 不在库；network_cross_audit 与 "
                    "network_cross_audit_attribution 已改指 teacher 两列，可运行。",
            "scripts": ["network_cross_audit.py", "network_cross_audit_attribution.py"],
        },
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / "network_mainline_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nstatus =", out["status"], "|",
          {k: v["summary"]["years_ok"] for k, v in columns.items() if v.get("summary")})


if __name__ == "__main__":
    main()
