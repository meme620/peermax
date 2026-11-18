#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 Phase B′4：换语料桥接对账（自建 MD&A vs 老师 MD&A，重叠年 2017–2022）。

目的：把"换语料"这件事本身量化为可登记的"语料版本效应"，而不是只看最终结果。
输入：data/processed/network_mb/（自建，pm_mb_tfidf_v1_*）
      data/processed/network_mb_teacher/（老师语料，pm_mb_tfidf_teacher_v1_*）
      data/processed/sw2_industry_latest.csv（同申万二级占比）
产物：reports/data_gates/G0/s3_corpus_bridge_report.json + .md

指标（逐年，仅共同节点集）：
  1. 节点数：两网各自入网公司数、交集、仅自建、仅老师
  2. peer 数分布：中位 / 均值 / p90
  3. 弱边占比：权重 < 全样本边权重中位数 的边占比中，取定义一致口径（此处用 weight < 1/n_edges 的分位定义见实现）
  4. 同申万二级边占比 + 相对随机基线 lift
  5. 边 Jaccard（(focal,source) 对，限共同节点）
  6. 权重集中度：Top1 权重中位

用法：python scripts/corpus_bridge_reconcile.py [--years 2017 ... 2022]
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SELF_DIR = ROOT / "data/processed/network_mb"
TEACH_DIR = ROOT / "data/processed/network_mb_teacher"
SW2 = ROOT / "data/processed/sw2_industry_latest.csv"
REPD = ROOT / "reports/data_gates/G0"
YEARS = [2017, 2018, 2019, 2020, 2021, 2022]
csv.field_size_limit(10**9)


def read_edges(d: Path, y: int) -> pd.DataFrame:
    fp = d / f"{y}_similarity.csv"
    return pd.read_csv(fp, dtype={"focal_id": str, "source_id": str},
                       usecols=["focal_id", "source_id", "similarity", "weight"])


def load_sw2() -> dict[str, str]:
    sw = pd.read_csv(SW2, dtype={"code6": str})
    sw["code6"] = sw["code6"].str.zfill(6)
    return dict(zip(sw["code6"], sw["l2_code"].astype(str)))


def l2_of(firm: str, m: dict[str, str]):
    return m.get(str(firm)[2:].zfill(6)) if isinstance(firm, str) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=YEARS)
    args = ap.parse_args()
    sw2 = load_sw2()

    per_year, jac_all = [], []
    for y in args.years:
        if not (SELF_DIR / f"{y}_similarity.csv").exists() or not (TEACH_DIR / f"{y}_similarity.csv").exists():
            per_year.append({"text_year": y, "status": "missing"})
            continue
        a = read_edges(SELF_DIR, y)      # 自建
        b = read_edges(TEACH_DIR, y)     # 老师
        na = set(a.focal_id) | set(a.source_id)
        nb = set(b.focal_id) | set(b.source_id)
        common = na & nb

        def peer_stats(df):
            c = df.groupby("focal_id").size()
            return {"nodes": int(df.focal_id.nunique()), "edges": int(len(df)),
                    "peer_median": float(c.median()), "peer_mean": round(float(c.mean()), 2),
                    "peer_p90": float(c.quantile(0.9)),
                    "top1_weight_median": float(df.groupby("focal_id")["weight"].max().median()),
                    "weak_edge_share": float((df.weight < (1.0 / c.reindex(df.focal_id).values)).mean())}

        pa, pb = peer_stats(a), peer_stats(b)

        # 同申万二级占比（限共同节点）
        def same_sw2(df):
            d = df[df.focal_id.isin(common) & df.source_id.isin(common)]
            fa = d.focal_id.map(lambda f: l2_of(f, sw2))
            fb = d.source_id.map(lambda f: l2_of(f, sw2))
            ok = fa.notna() & fb.notna()
            if not ok.any():
                return None, None
            same = float((fa[ok] == fb[ok]).mean())
            base = float((fa[ok] == fa[ok].sample(frac=1.0, random_state=7).values).mean())
            return same, (same / base if base > 0 else None)

        sa, la = same_sw2(a)
        sb, lb = same_sw2(b)

        # 边 Jaccard（限共同节点）
        A = set(map(tuple, a.loc[a.focal_id.isin(common) & a.source_id.isin(common),
                                ["focal_id", "source_id"]].values))
        B = set(map(tuple, b.loc[b.focal_id.isin(common) & b.source_id.isin(common),
                                ["focal_id", "source_id"]].values))
        jac = len(A & B) / max(1, len(A | B))
        jac_all.append(jac)

        per_year.append({
            "text_year": y, "network_year": y + 1, "status": "ok",
            "nodes_self": pa["nodes"], "nodes_teacher": pb["nodes"],
            "nodes_common": len(common), "nodes_self_only": len(na - nb), "nodes_teacher_only": len(nb - na),
            "edges_self": pa["edges"], "edges_teacher": pb["edges"],
            "peer_median_self": pa["peer_median"], "peer_median_teacher": pb["peer_median"],
            "peer_mean_self": pa["peer_mean"], "peer_mean_teacher": pb["peer_mean"],
            "peer_p90_self": pa["peer_p90"], "peer_p90_teacher": pb["peer_p90"],
            "top1_weight_median_self": pa["top1_weight_median"],
            "top1_weight_median_teacher": pb["top1_weight_median"],
            "weak_edge_share_self": pa["weak_edge_share"], "weak_edge_share_teacher": pb["weak_edge_share"],
            "same_sw2_self": sa, "same_sw2_teacher": sb,
            "sw2_lift_self": la, "sw2_lift_teacher": lb,
            "edge_jaccard_common_nodes": round(jac, 4),
        })

    ok = [r for r in per_year if r.get("status") == "ok"]
    rep = {
        "step": "S3 Phase B′4 换语料桥接对账",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "years": args.years,
        "self_dir": str(SELF_DIR.relative_to(ROOT)),
        "teacher_dir": str(TEACH_DIR.relative_to(ROOT)),
        "note": "两个 M-B 网络均在各自语料上以同一管线（分词/df≤0.25/TF-IDF/余弦/固定 τ）构建；"
                "本报告只量化语料版本效应，不改变任何冻结值。",
        "per_year": per_year,
        "summary": {
            "n_years_ok": len(ok),
            "edge_jaccard_median": float(np.median(jac_all)) if jac_all else None,
            "peer_median_self_median": float(np.median([r["peer_median_self"] for r in ok])) if ok else None,
            "peer_median_teacher_median": float(np.median([r["peer_median_teacher"] for r in ok])) if ok else None,
            "same_sw2_self_median": float(np.median([r["same_sw2_self"] for r in ok if r["same_sw2_self"]])) if ok else None,
            "same_sw2_teacher_median": float(np.median([r["same_sw2_teacher"] for r in ok if r["same_sw2_teacher"]])) if ok else None,
        },
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / "s3_corpus_bridge_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# S3 换语料桥接对账（B′4）", "",
          f"生成时间：{rep['run_at']}　重叠年：{args.years}", "",
          "| 文本年 | 节点(自建/老师/共) | 边(自建/老师) | peer中位(自建/老师) | 同申万二级(自建/老师) | 边Jaccard |",
          "|---|---|---|---|---|---|"]
    for r in per_year:
        if r.get("status") != "ok":
            md.append(f"| {r['text_year']} | — | — | — | — | 缺失 |")
            continue
        md.append(f"| {r['text_year']} | {r['nodes_self']}/{r['nodes_teacher']}/{r['nodes_common']} | "
                  f"{r['edges_self']:,}/{r['edges_teacher']:,} | "
                  f"{r['peer_median_self']:.0f}/{r['peer_median_teacher']:.0f} | "
                  f"{r['same_sw2_self']:.3f}/{r['same_sw2_teacher']:.3f} | {r['edge_jaccard_common_nodes']} |")
    md += ["", f"汇总：边 Jaccard 中位 = {rep['summary']['edge_jaccard_median']}；"
               f"peer 中位（自建/老师）= {rep['summary']['peer_median_self_median']}/"
               f"{rep['summary']['peer_median_teacher_median']}"]
    (REPD / "s3_corpus_bridge_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(rep["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
