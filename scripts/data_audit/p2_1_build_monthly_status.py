#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.1：shadow 月收益/状态面板（只读 headline，只写 shadow）。

背景（`artifacts/phase2/2.0_plan.md` §2.1）：
  headline `complete_month = n_valid_days >= 15`，而 `n_valid_days` 是 Drive 价矩阵
  非空格数（前向填充会把停牌日算作“有价”），停牌月同样可能 >=15 → 闸门失效。
  本脚本只追加 shadow 列，用 `daily_panel` 的 `volume > 0` 交易日数重算完整月。

2026-10-28 修正（R1/R3）：
  - `n_trade_days==0` 不再直接等于“全月停牌”。对 daily_panel 无行的 (code, month)，
    用 `stock_suspend_flag` 月度 True 天数交叉验证：只有
    `suspend_true_days == 市场交易日` 才允许 `no_trade`；部分停牌/无法证明无成交
    → `daily_panel_missing`，`n_trade_days`/`complete_month_trade15/10`/三个 zero flag
    置 NA（不再写 0），`R_p2shadow` 保留原 R（只在 `no_trade` 时置 NA）。
  - `sh601313` alias 到 `sh601360` 做日频/成交查询：sh601313 在 daily_panel 无行，
    但月频 R 与 sh601360 在 73 个重叠月中 72 个相同；46 行恢复 `n_trade_days`
    （43 行 >=15、3 行 1–14），2017-07..10 的 4 行仍为真实停牌。
    alias 表写入 `artifacts/phase2/2.1_alias_map.json`。
  - 仍然只写 shadow；不覆盖 headline、不改 configs、不重跑 M1/PC/网络。

口径：
  - 输入 `monthly_returns_clean_v2.csv` 原列只读复制；行数固定 856,771。
  - `n_price_obs` = 原 `n_valid_days`；`complete_month_legacy` = 原 `complete_month`。
  - `n_trade_days` = `daily_panel/{2014..2025}.csv` 中 `(daily_code, month)` 的
    `volume > 0` 天数；仅在 `ok` / `no_trade` 行写出（`no_trade` 为 0）。
  - `trade_days_status` / `trade_data_status`：
      `no_volume_data`        month < 2014-01（含 2000–2005 与 2006–2013）
      `ok`                    2014-01..2025-12 且有 volume>0 交易日
      `no_trade`              2014-01..2025-12 且无成交（daily_panel 有行但 volume 全 0，
                              或 suspend_true_days == 市场交易日）
      `daily_panel_missing`   2014-01..2025-12、daily_panel 无行、部分停牌/无法证明无成交
      `out_of_daily_coverage` month > 2025-12（2026）
  - `complete_month_trade15` = `n_trade_days >= 15`（仅 ok/no_trade；其余 NA）；
    `complete_month_trade10` 为 10 日敏感性。
  - `pseudo_zero_flag` = `R == 0` 且 `n_trade_days == 0`（仅 ok/no_trade）；
    `partial_zero_flag` = `R == 0` 且 `0 < n_trade_days < 15`；
    `true_zero_flag` = `R == 0` 且 `n_trade_days >= 15`。
  - `R_p2shadow`：`no_trade` → NA；其余保留原 `R`（含 `daily_panel_missing` 与 alias 行）。

只写：
  data/processed/phase2_shadow/monthly_returns_status_p2shadow.csv
  data/processed/phase2_shadow/monthly_returns_status_p2shadow.meta.json
  artifacts/phase2/2.1_alias_map.json
  artifacts/phase2/2.1_counts.json

用法：
  python scripts/data_audit/p2_1_build_monthly_status.py
"""
from __future__ import annotations

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import sha256_file
from lib.io import write_json_atomic
from lib.shadow.p2_1_build_monthly_status import ALIAS_MAP, ALIAS_OUT, CONTRACTS, COUNTS_OUT, END_YEAR, EVALUABLE_STATUS, EXPECTED_ROWS, FIRM_ID_MAP, FM_EXPECTED_MONTHS, FM_EXPECTED_ROWS, FM_SAMPLE, META_OUT, MIN_TRADE_DAYS, NEW_COLS, OUT, P2SHADOW_VERSION, PANEL_DIR, RETURNS, ROOT, SENSITIVITY_TRADE_DAYS, SOURCE_PANEL, START_YEAR, STATUS_DAILY_PANEL_MISSING, STATUS_NO_TRADE, STATUS_NO_VOLUME, STATUS_OK, STATUS_OUT_OF_COVERAGE, SUSPEND_FLAG, _alias_payload, _build_status_panel_full, _by_year, _classification_audit, _fm_counts, _fm_event_month_counts, _load_suspend_monthly, _scope_counts, _source_bridge, _window_counts, anchor_snapshot, assert_anchors_unchanged, build_status_panel, daily_panel_audit, load_contracts, protected_paths, write_csv_atomic, ym_add
# --- end scripts/lib bootstrap ---


import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / "scripts" / "data"))

# 复用 Phase 1 的 volume>0 口径，不复制实现（2.0_plan §1.3）。
from build_monthly_returns_status import trade_days_long  # noqa: E402



# sh601313（江南嘉捷）在 daily_panel 中无行；日频/成交在 sh601360（三六零）。
# 月频 2012-02..2018-02 重叠 73 个月，72 个月 R 相同；firm_id_map 只有 sh601360。

# 追加列（旧 10 列顺序不变，新列只追加在末尾）。



# --------------------------------------------------------------------------- IO




# ---------------------------------------------------------------- headline 锚








# ---------------------------------------------------------------- 日频辅助




# ---------------------------------------------------------------- 构造




# ---------------------------------------------------------------- 计数
















# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2.1 shadow 月收益/状态面板")
    ap.add_argument("--returns", default=str(RETURNS))
    ap.add_argument("--panel-dir", default=str(PANEL_DIR))
    ap.add_argument("--suspend-flag", default=str(SUSPEND_FLAG))
    ap.add_argument("--firm-id-map", default=str(FIRM_ID_MAP))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--meta-out", default=str(META_OUT))
    ap.add_argument("--alias-out", default=str(ALIAS_OUT))
    ap.add_argument("--counts-out", default=str(COUNTS_OUT))
    ap.add_argument("--contracts", default=str(CONTRACTS))
    ap.add_argument("--fm-sample", default=str(FM_SAMPLE))
    ap.add_argument("--source-panel", default=str(SOURCE_PANEL))
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--min-trade-days", type=int, default=MIN_TRADE_DAYS)
    ap.add_argument("--sensitivity-days", type=int, default=SENSITIVITY_TRADE_DAYS)
    ap.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    args = ap.parse_args(argv)

    returns_p = Path(args.returns).resolve()
    out_p = Path(args.out).resolve()
    meta_p = Path(args.meta_out).resolve()
    alias_p = Path(args.alias_out).resolve()
    counts_p = Path(args.counts_out).resolve()
    contracts_p = Path(args.contracts).resolve()
    fm_p = Path(args.fm_sample).resolve()
    src_p = Path(args.source_panel).resolve()
    susp_p = Path(args.suspend_flag).resolve()
    firm_map_p = Path(args.firm_id_map).resolve()

    contracts = load_contracts(contracts_p)
    protected = protected_paths(contracts)
    for label, p in (("out", out_p), ("meta-out", meta_p),
                     ("alias-out", alias_p), ("counts-out", counts_p)):
        if p in protected:
            raise SystemExit(f"[P2.1] {label} 指向 headline 路径，拒绝写出: {p}")
    shadow_dir = (ROOT / "data/processed/phase2_shadow").resolve()
    artifacts_dir = (ROOT / "artifacts/phase2").resolve()
    if out_p.parent != shadow_dir or meta_p.parent != shadow_dir:
        raise SystemExit(f"[P2.1] shadow 输出必须位于 {shadow_dir}")
    if alias_p.parent != artifacts_dir or counts_p.parent != artifacts_dir:
        raise SystemExit("[P2.1] alias/counts 输出必须位于 artifacts/phase2/")

    anchors_before = anchor_snapshot(contracts)
    if not all(v["sha_match"] for v in anchors_before.values()):
        bad = [k for k, v in anchors_before.items() if not v["sha_match"]]
        raise SystemExit(f"[P2.1] headline 锚起始不一致: {bad}")

    ret = pd.read_csv(returns_p, dtype={"firm_id": str, "month": str})
    if len(ret) != args.expected_rows:
        raise SystemExit(f"[P2.1] 月频行数 {len(ret)} != {args.expected_rows}")
    print(f"[P2.1] 月频输入 {len(ret):,} 行；读 daily_panel {args.start_year}-{args.end_year}"
          f" 与 suspend flag …", flush=True)

    full, daily_rows = _build_status_panel_full(
        ret, Path(args.panel_dir).resolve(), start_year=args.start_year,
        end_year=args.end_year, min_trade_days=args.min_trade_days,
        sensitivity_days=args.sensitivity_days, suspend_flag=susp_p,
        alias_map=ALIAS_MAP)
    panel = full[list(ret.columns) + NEW_COLS]

    if len(panel) != args.expected_rows:
        raise SystemExit(f"[P2.1] shadow 行数 {len(panel)} != {args.expected_rows}")
    if panel.duplicated(["firm_id", "month"]).any():
        raise SystemExit("[P2.1] shadow 键 (firm_id, month) 不唯一")

    write_csv_atomic(panel, out_p)
    out_sha = sha256_file(out_p)

    scope_counts = _scope_counts(panel)
    window_counts = _window_counts(panel)
    classification = _classification_audit(full, ALIAS_MAP)
    alias_payload = _alias_payload(ret, ALIAS_MAP, daily_rows, firm_map_p, full)
    write_json_atomic(alias_payload, alias_p)
    alias_sha = sha256_file(alias_p)
    fm_counts = _fm_counts(panel, fm_p, src_p)
    fm_event = _fm_event_month_counts(panel, fm_p)
    source_bridge = _source_bridge(panel, src_p)

    anchors_after = anchor_snapshot(contracts)
    assert_anchors_unchanged(anchors_before, anchors_after)

    meta = {
        "p2shadow_version": P2SHADOW_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "monthly_returns": {
                "file": str(returns_p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": anchors_before["monthly_returns_clean_v2"]["sha256"],
                "rows": int(len(ret)),
                "read_only": True,
            },
            "daily_panel": {
                "dir": str(Path(args.panel_dir).resolve().relative_to(ROOT)).replace("\\", "/"),
                "years": list(range(args.start_year, args.end_year + 1)),
                "files": [
                    {
                        "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": sha256_file(p),
                    }
                    for p in sorted(Path(args.panel_dir).glob("*.csv"))
                ],
            },
            "stock_suspend_flag": {
                "file": str(susp_p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(susp_p),
                "read_only": True,
            },
            "firm_id_map": {
                "file": str(firm_map_p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(firm_map_p),
                "read_only": True,
            },
            "alias_map": dict(ALIAS_MAP),
        },
        "output": {
            "file": str(out_p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": out_sha,
            "rows": int(len(panel)),
            "columns": list(panel.columns),
        },
        "alias_map_output": {
            "file": str(alias_p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": alias_sha,
        },
        "alias_map": dict(ALIAS_MAP),
        "definitions": {
            "n_price_obs": "原 n_valid_days（Drive 价矩阵非空格数；前向填充含停牌日）",
            "complete_month_legacy": "原 complete_month（n_price_obs >= 15）",
            "n_trade_days": "daily_panel volume>0 天数；仅 ok/no_trade 写出；daily_panel_missing/pre-2014/2026 为 NA",
            "trade_days_status": {
                "no_volume_data": "month < 2014-01",
                "ok": "2014-01..2025-12 且 n_trade_days > 0",
                "no_trade": "2014-01..2025-12 且无成交（daily_panel 有行但 volume 全 0，或 suspend_true_days == 市场交易日）",
                "daily_panel_missing": "2014-01..2025-12、daily_panel 无行、部分停牌/无法证明无成交",
                "out_of_daily_coverage": "month > 2025-12",
            },
            "trade_data_status": "trade_days_status 的逐值别名（任务文字口径）",
            "complete_month_trade15": "n_trade_days >= 15（仅 ok/no_trade，其余 NA）",
            "complete_month_trade10": "n_trade_days >= 10（10 日敏感性，仅 ok/no_trade）",
            "pseudo_zero_flag": "R==0 且 n_trade_days==0（仅 ok/no_trade）",
            "partial_zero_flag": "R==0 且 0<n_trade_days<15（仅 ok/no_trade）",
            "true_zero_flag": "R==0 且 n_trade_days>=15（仅 ok/no_trade）",
            "R_p2shadow": "no_trade → NA；其余保持原 R（含 daily_panel_missing 与 alias 行）",
            "row_order": "month, firm_id（稳定排序）",
            "headline_mutation": "none；原文件只读，输出仅追加列",
        },
        "alias_map_basis": alias_payload["basis"],
        "headline_anchors_before": anchors_before,
        "headline_anchors_after": anchors_after,
        "headline_anchors_match": all(
            anchors_before[k]["sha256"] == anchors_after[k]["sha256"]
            for k in anchors_before),
    }
    write_json_atomic(meta, meta_p)

    expected_window = {
        "complete_legacy_and_trade_days_lt_15": 13_980,
        "complete_legacy_and_trade_days_zero": 8_014,
        "complete_legacy_and_trade_days_1_14": 5_966,
        "R_zero_and_complete_legacy": 9_561,
        "R_zero_and_complete_legacy_trade_days_zero": 7_524,
        "R_zero_and_complete_legacy_trade_days_1_14": 10,
        "R_zero_and_complete_legacy_trade_days_ge_min": 2_017,
        "rows_daily_panel_missing": 38,
        "R_zero_and_complete_legacy_daily_panel_missing": 10,
    }
    window_replication = {
        k: {
            "actual": int(window_counts[k]),
            "expected": int(v),
            "match": bool(window_counts[k] == v),
        }
        for k, v in expected_window.items()
    }
    window_replication["pseudo_zero_flag_includes_non_complete_legacy_rows"] = {
        "flag_sum_unrestricted": window_counts["pseudo_zero_flag"],
        "complete_legacy_subset": window_counts["R_zero_and_complete_legacy_trade_days_zero"],
        "difference": window_counts["pseudo_zero_flag"]
        - window_counts["R_zero_and_complete_legacy_trade_days_zero"],
        "note": "按任务定义 flag 不附加 complete_month=1；2.0_plan 的 7,534 为 complete=1 子集",
    }
    window_replication["partial_zero_flag_includes_non_complete_legacy_rows"] = {
        "flag_sum_unrestricted": window_counts["partial_zero_flag"],
        "complete_legacy_subset": window_counts["R_zero_and_complete_legacy_trade_days_1_14"],
        "difference": window_counts["partial_zero_flag"]
        - window_counts["R_zero_and_complete_legacy_trade_days_1_14"],
        "note": "同上",
    }

    counts = {
        "p2shadow_version": P2SHADOW_VERSION,
        "generated_at_utc": meta["generated_at_utc"],
        "script": meta["script"],
        "script_sha256": meta["script_sha256"],
        "shadow_file": meta["output"],
        "alias_map_file": meta["alias_map_output"],
        "scope": {
            "start_year": args.start_year,
            "end_year": args.end_year,
            "min_trade_days": args.min_trade_days,
            "sensitivity_trade_days": args.sensitivity_days,
            "window_2015_05_2023_12": ["2015-05", "2023-12"],
            "alias_map": dict(ALIAS_MAP),
        },
        **scope_counts,
        "by_year": _by_year(panel),
        "window_2015_05_2023_12": window_counts,
        "window_2015_05_2023_12_replicates_2_0_plan": window_replication,
        "window_2015_05_2023_12_all_match": all(
            v["match"] for v in window_replication.values() if "match" in v),
        "missing_classification": classification,
        "alias_audit": alias_payload,
        "fm_ma_teacher": fm_counts,
        "fm_ma_teacher_event_month": fm_event,
        "source_panel_bridge_2015_05_2023_12": source_bridge,
        "headline_anchors_before": anchors_before,
        "headline_anchors_after": anchors_after,
        "headline_anchors_match": meta["headline_anchors_match"],
        "headline_files_modified": False,
        "m1_rerun": False,
        "configs_modified": False,
    }
    write_json_atomic(counts, counts_p)

    print(f"[P2.1] shadow -> {out_p.relative_to(ROOT)}  rows={len(panel):,}  sha={out_sha[:12]}…")
    print(f"[P2.1] 2014–2025: ok={scope_counts['n_trade_days_ge_15'] + scope_counts['n_trade_days_1_14']:,} "
          f"no_trade={scope_counts['n_trade_days_zero']:,} "
          f"daily_panel_missing={scope_counts['rows_daily_panel_missing']:,}")
    print(f"[P2.1] window 2015-05..2023-12: legacy complete & td<15="
          f"{window_counts['complete_legacy_and_trade_days_lt_15']:,} "
          f"(zero={window_counts['complete_legacy_and_trade_days_zero']:,}, "
          f"partial={window_counts['complete_legacy_and_trade_days_1_14']:,}) "
          f"all_match={counts['window_2015_05_2023_12_all_match']}")
    if fm_counts.get("available"):
        print(f"[P2.1] FM: R_next==0={fm_counts['R_next_zero']:,} "
              f"split={fm_counts['R_next_zero_pseudo_full_month_no_trade']}/"
              f"{fm_counts['R_next_zero_partial_1_14_trade_days']}/"
              f"{fm_counts['R_next_zero_true_ge_15_trade_days']}/"
              f"{fm_counts['R_next_zero_daily_panel_missing']}; "
              f"gates outcome={fm_counts['outcome_gate_15_loss_rows']:,} "
              f"max={fm_counts['max_gate_15_loss_rows']:,} "
              f"union={fm_counts['gate_union_loss_rows']:,} "
              f"upper={fm_counts['fm_upper_bound_after_gates']:,}; "
              f"all_match={all(fm_counts['matches_2_0_plan'].values())}")
    print(f"[P2.1] split: 586={classification['no_trade_R_nonzero']:,} "
          f"46={classification['alias_recovered_rows']:,} "
          f"96={classification['daily_panel_missing_rows']:,} "
          f"632={classification['original_nonzero_R_daily_panel_missing_632']:,} "
          f"overlap96_632={classification['overlap_daily_panel_missing_with_632']}")
    if source_bridge.get("available"):
        print(f"[P2.1] source bridge 2015-05..2023-12: matched="
              f"{source_bridge['matched_rows']:,} equal={source_bridge['equal_rows']:,} "
              f"diff={source_bridge['diff_rows']} "
              f"all_match={source_bridge['matches_2_0_plan']}")
    print(f"[P2.1] 9 anchors unchanged: {meta['headline_anchors_match']}")
    if not counts["window_2015_05_2023_12_all_match"]:
        print("[P2.1] WARN: 2015-05..2023-12 计数与预期不一致", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
