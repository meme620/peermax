#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 Phase B′0：老师 MD&A 语料接入（M-B 新主输入，2026-09-22 决策）。

输入：data/raw/teacher/mda_data_2023.csv（老师提供；sha256 364b1dc5…；字段 f_id/f_stock_id/f_year/f_chapter_content）
输出：data/processed/teacher_mda_panel.csv + reports/data_gates/G0/s3_bp0_corpus_ingest.json

处理门（对应 execution_plan §3 S3「语料来源与版本」块）：
  1. 年份门     f_year ∈ [2014, 2022]
  2. universe 门 剔除北交所（代码以 8/4 开头）与不在 firm_id_map 的代码；code6 → 项目 firm_id
  3. 长度门     len(text) >= 500
  4. 唯一键门   (firm_id, year) 唯一（源侧 (f_stock_id, f_year) 亦须唯一）
  5. 审计列     src_code / src_year / length / status / reason

注意：本脚本**不**做分词、不构网；只把老师语料变成与自建语料同构的中间面板，
供 M-B 构网（B′2）与桥接对账（B′4）使用。自建 mda_texts_v2.csv 保留为对照链。
用法：python scripts/data_audit/s3_teacher_corpus_ingest.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/teacher/mda_data_2023.csv"
IDMAP = ROOT / "data/processed/firm_id_map.csv"
OUT = ROOT / "data/processed/teacher_mda_panel.csv"
REPD = ROOT / "reports/data_gates/G0"

YEAR_LO, YEAR_HI = 2014, 2022
MIN_LEN = 500
BSE_PREFIX = ("8", "4")  # 北交所（8xxxxx/4xxxxx）
COLS = ["firm_id", "year", "text", "src_code", "src_year", "length", "status", "reason"]


def load_idmap() -> dict[str, str]:
    """code6 -> firm_id（项目面板码，含 sz/sh 前缀）。"""
    out: dict[str, str] = {}
    with open(IDMAP, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out[str(r["code6"]).zfill(6)] = r["firm_id"]
    return out


def main() -> None:
    idmap = load_idmap()
    csv.field_size_limit(10**9)

    rows_out: list[tuple] = []
    stats = Counter()
    seen: set[tuple[str, str]] = set()
    per_year: dict[str, Counter] = {}

    with open(SRC, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            code = (r.get("f_stock_id") or "").strip()
            year = (r.get("f_year") or "").strip()
            text = r.get("f_chapter_content") or ""
            stats["read"] += 1

            if not year.isdigit():
                stats["drop_bad_year"] += 1
                continue
            if not (YEAR_LO <= int(year) <= YEAR_HI):
                stats["drop_year"] += 1
                continue
            if code.startswith(BSE_PREFIX):
                stats["drop_bse"] += 1
                continue
            firm = idmap.get(code.zfill(6))
            if firm is None:
                stats["drop_unmatched"] += 1
                continue
            if (code, year) in seen:
                stats["drop_dup_pair"] += 1
                continue
            if len(text) < MIN_LEN:
                stats["drop_short"] += 1
                continue

            seen.add((code, year))
            rows_out.append((firm, int(year), text, code, int(year), len(text), "ok", ""))
            stats["kept"] += 1
            per_year.setdefault(year, Counter())["kept"] += 1

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows(rows_out)

    firms = {r[0] for r in rows_out}
    rep = {
        "step": "S3 Phase B′0 语料接入",
        "source": str(SRC.relative_to(ROOT)),
        "output": str(OUT.relative_to(ROOT)),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "year_range": [YEAR_LO, YEAR_HI],
            "min_len": MIN_LEN,
            "drop_bse_prefix": list(BSE_PREFIX),
            "universe": "firm_id_map.csv（5,543 只）",
        },
        "stats": dict(stats),
        "n_rows_kept": len(rows_out),
        "n_firms": len(firms),
        "n_pairs": len(seen),
        "per_year_kept": {y: c["kept"] for y, c in sorted(per_year.items())},
        "status": "PASS" if stats["kept"] > 30_000 and stats["drop_dup_pair"] == 0 else "CHECK",
    }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / "s3_bp0_corpus_ingest.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in
                      ("status", "n_rows_kept", "n_firms", "stats", "per_year_kept")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
