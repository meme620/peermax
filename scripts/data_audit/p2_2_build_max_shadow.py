#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.2：shadow source-month MAX（排除复牌首日跨期收益；只读 headline）。

背景（`artifacts/phase2/2.0_plan.md` §2.2）：
  headline `source_month_argmax_panel.csv` 的 `MAXraw=max(rraw)`，而 `rraw` 是
  “上一可得行”收盘环比；停牌后复牌首日包含整段停牌期（可跨月/跨年）的累计收益，
  被当成单日收益进入 MAX。本脚本只读 headline，另写 shadow 面板：
    - 从 `daily_panel/{2014..2025}.csv` 建市场交易日历；
    - 对每只股票按日期排序，`gap_td = cal_idx(date) - cal_idx(prev_date)`；
      `resume_flag=1` 当 gap_td>1 且存在上一可得行；首行不算复牌；
    - `MAXraw_p2shadow_ungated` = 非复牌日 rraw 的 max；
    - `MAXraw_p2shadow`（主口径）= 在 15 日门（`n_valid_days>=15` 且
      `n_max_days>0`）下输出，否则留空；10 日敏感性由 `n_max_days` 计算；
    - `argmax_date_p2shadow` 用 registry `tie_rule`（当前 earliest）；
    - `limit_hit_p2shadow` 用官方 `daily_limit_up_flag`（若 pkl 可用）；
      缺失时回退阈值口径并标 `limit_source='threshold_fallback'`；
    - `limit_status_at_resume`：headline argmax 落在复牌日时，按官方
      up/down flag 改标 `limit_up`/`limit_down`/`no_limit`；
    - `R_p2shadow` 保留原 `R`（MAX 与 R 分开，不重算月收益）。

口径边界（必须保留）：
  - `daily_panel` 只覆盖 2014-2025；pre-2014 的 source-month 行无日频覆盖，
    `MAXraw_p2shadow`/`argmax_date_p2shadow` 为 legacy 直通，`shadow_status=
    pre_2014_no_daily_coverage`，不得把直通值当作已排除复牌。
  - 2014-2025 的日频覆盖行：`n_resume_days==0` 的月份 `MAXraw_p2shadow_ungated`
    必须与 headline `MAXraw` 逐值一致（硬验收）。
  - 行数固定 734,232；旧列逐字段只读复制；新列只追加在末尾；行序保持输入顺序。

只写：
  data/processed/phase2_shadow/source_month_argmax_panel_p2shadow.csv
  data/processed/phase2_shadow/source_month_argmax_panel_p2shadow.meta.json
  artifacts/phase2/2.2_counts.json
  artifacts/phase2/2.2_resume_audit.csv
  artifacts/phase2/2.2_max_tail_diff.csv

用法：
  python scripts/data_audit/p2_2_build_max_shadow.py --min-trade-days 15 --sensitivity 10
"""
from __future__ import annotations

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import sha256_file
from lib.io import write_json_atomic
from lib.fmt import fmt_float
from lib.fmt import fmt_int
from lib.shadow.p2_2_build_max_shadow import ARTIFACTS_DIR, CONTRACTS, COUNTS_OUT, END_YEAR, EXPECTED_ROWS, EXPECTED_SOURCE_SHA, FORMULA_EVALUABLE, FORMULA_PRE2014, LIMIT_DOWN_FLAG, LIMIT_UP_FLAG, META_OUT, MIN_TRADE_DAYS, MONTHLY_RETURNS, NEW_COLS, NEW_COL_KINDS, OUT, P2SHADOW_VERSION, PANEL_DIR, RESUME_AUDIT_OUT, ROOT, SENSITIVITY_DAYS, SHADOW_DIR, SOURCE_PANEL, START_YEAR, ST_FLAG, SUSPEND_FLAG, TAIL_OUT, WINDOW_CAL, WINDOW_M1, _argmax_rows, _counts_for_window, _gate_counts, _limit_counts, _max_row, _resume_audit_table, _tail_diff_table, _tail_stats, _window, aggregate_year_months, anchor_snapshot, assert_anchors_unchanged, build_market_calendar, build_shadow_panel, code_column, compute_month_aggregates, compute_resume_flags, format_column, load_bool_panel, load_contracts, lookup_panel, protected_paths, st_at, threshold_limit_hit, write_audit_csv, write_shadow_csv
# --- end scripts/lib bootstrap ---


import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / "scripts" / "data_audit"))

# 只复用 Phase 1 已登记实现：tie rule / ST / 板块 / limit 阈值（不复制逻辑）。
from s4_step1_max import (  # noqa: E402
    LIMIT_CHINEXT_STAR,
    LIMIT_MAIN,
    LIMIT_MAIN_ST,
    board_of,
    load_st_flags,
    pick_argmax,
    resolve_tie_rule,
)




# 追加列（旧列顺序不变，新列只追加在末尾）。
# 每列的输出格式：int / float / str / raw（原样字符串）。


# --------------------------------------------------------------------------- IO




# ---------------------------------------------------------------- headline 锚








# ---------------------------------------------------------------- 日历/复牌










# ---------------------------------------------------------------- official flags










# ---------------------------------------------------------------- 构造 shadow


# ---------------------------------------------------------------- 计数


















# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2.2 shadow source-month MAX")
    ap.add_argument("--source-panel", default=str(SOURCE_PANEL))
    ap.add_argument("--panel-dir", default=str(PANEL_DIR))
    ap.add_argument("--suspend-flag", default=str(SUSPEND_FLAG))
    ap.add_argument("--limit-up-flag", default=str(LIMIT_UP_FLAG))
    ap.add_argument("--limit-down-flag", default=str(LIMIT_DOWN_FLAG))
    ap.add_argument("--monthly-returns", default=str(MONTHLY_RETURNS))
    ap.add_argument("--contracts", default=str(CONTRACTS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--meta-out", default=str(META_OUT))
    ap.add_argument("--counts-out", default=str(COUNTS_OUT))
    ap.add_argument("--resume-audit", default=str(RESUME_AUDIT_OUT))
    ap.add_argument("--tail-out", default=str(TAIL_OUT))
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--min-trade-days", type=int, default=MIN_TRADE_DAYS)
    ap.add_argument("--sensitivity", type=int, default=SENSITIVITY_DAYS)
    ap.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    ap.add_argument("--registry", default=None)
    args = ap.parse_args(argv)

    src_p = Path(args.source_panel).resolve()
    out_p = Path(args.out).resolve()
    meta_p = Path(args.meta_out).resolve()
    counts_p = Path(args.counts_out).resolve()
    resume_p = Path(args.resume_audit).resolve()
    tail_p = Path(args.tail_out).resolve()
    contracts_p = Path(args.contracts).resolve()
    monthly_p = Path(args.monthly_returns).resolve()
    panel_dir = Path(args.panel_dir).resolve()
    susp_p = Path(args.suspend_flag).resolve()
    up_p = Path(args.limit_up_flag).resolve()
    down_p = Path(args.limit_down_flag).resolve()

    contracts = load_contracts(contracts_p)
    protected = protected_paths(contracts)
    shadow_dir = SHADOW_DIR.resolve()
    artifacts_dir = ARTIFACTS_DIR.resolve()
    for label, p in (("out", out_p), ("meta-out", meta_p),
                     ("counts-out", counts_p), ("resume-audit", resume_p),
                     ("tail-out", tail_p)):
        if p in protected:
            raise SystemExit(f"[P2.2] {label} 指向 headline 路径，拒绝写出: {p}")
    if out_p.parent != shadow_dir or meta_p.parent != shadow_dir:
        raise SystemExit(f"[P2.2] shadow 输出必须位于 {shadow_dir}")
    if any(p.parent != artifacts_dir for p in (counts_p, resume_p, tail_p)):
        raise SystemExit("[P2.2] 审计输出必须位于 artifacts/phase2/")

    anchors_before = anchor_snapshot(contracts)
    if not all(v["sha_match"] for v in anchors_before.values()):
        bad = [k for k, v in anchors_before.items() if not v["sha_match"]]
        raise SystemExit(f"[P2.2] headline 锚起始不一致: {bad}")

    source = pd.read_csv(src_p, dtype=str, keep_default_na=False)
    if len(source) != args.expected_rows:
        raise SystemExit(f"[P2.2] source 行数 {len(source)} != {args.expected_rows}")
    if source.duplicated(["source_id", "month"]).any():
        raise SystemExit("[P2.2] source 键 (source_id, month) 不唯一")
    src_sha_before = sha256_file(src_p)
    if src_sha_before != EXPECTED_SOURCE_SHA:
        raise SystemExit(
            f"[P2.2] headline source panel sha 与登记不一致: {src_sha_before}")

    tie = resolve_tie_rule(args.registry)
    tie_rule = tie["id"]
    print(f"[P2.2] tie_rule={tie_rule} source={tie['source']}", flush=True)

    years = range(args.start_year, args.end_year + 1)
    print(f"[P2.2] 读 daily_panel {args.start_year}-{args.end_year} …", flush=True)
    agg, resume, calendar = compute_month_aggregates(panel_dir, years, tie_rule)
    print(f"[P2.2] 月度聚合 {len(agg):,} 行；复牌行 {len(resume):,}；"
          f"交易日历 {len(calendar):,}", flush=True)

    print("[P2.2] 载入官方 limit flags …", flush=True)
    up_panel = load_bool_panel(up_p) if up_p.is_file() else None
    down_panel = load_bool_panel(down_p) if down_p.is_file() else None
    limit_available = up_panel is not None and down_panel is not None
    if not limit_available:
        print("[P2.2] WARN: 官方 limit pkl 缺失，limit_hit 用阈值回退", file=sys.stderr)

    out, resume, verification = build_shadow_panel(
        source, agg, resume, tie_rule, up_panel, down_panel,
        st_flag=ST_FLAG,
        min_trade_days=args.min_trade_days,
        sensitivity_days=args.sensitivity)
    print(f"[P2.2] shadow 行数 {len(out):,}；覆盖 {verification['rows_daily_covered_2014_2025']:,}；"
          f"pre-2014 {verification['rows_pre_2014_passthrough']:,}", flush=True)

    write_shadow_csv(out, list(source.columns), out_p)
    out_sha = sha256_file(out_p)

    # ---- 窗口计数
    w_cal = _window(out[out["shadow_status"].eq("evaluable")], WINDOW_CAL)
    w_m1 = _window(out[out["shadow_status"].eq("evaluable")], WINDOW_M1)
    cal_counts = _counts_for_window(w_cal)
    m1_counts = _counts_for_window(w_m1)

    monthly = pd.read_csv(monthly_p, usecols=["firm_id", "month", "complete_month"],
                          dtype={"firm_id": str, "month": str})
    monthly = monthly[(monthly["month"] >= WINDOW_CAL[0])
                      & (monthly["month"] <= WINDOW_CAL[1])]
    gate_cal = _gate_counts(w_cal, monthly, args.min_trade_days, args.sensitivity)
    gate_m1 = _gate_counts(w_m1, monthly, args.min_trade_days, args.sensitivity)

    limit_cal = _limit_counts(w_cal)
    limit_m1 = _limit_counts(w_m1)

    # ---- 审计表
    resume_audit = _resume_audit_table(
        resume, up_panel, down_panel, *load_st_flags(ST_FLAG))
    write_audit_csv(resume_audit, resume_p)
    tail = write_tail = _tail_diff_table(out)
    write_audit_csv(tail, tail_p)

    anchors_after = anchor_snapshot(contracts)
    assert_anchors_unchanged(anchors_before, anchors_after)
    src_sha_after = sha256_file(src_p)
    if src_sha_after != src_sha_before:
        raise SystemExit("[P2.2] headline source panel 在运行中变化")

    # ---- 与 2.0_plan 预期对账
    plan_expected = {
        "resume_rows_2014_2025": 13_986,
        "calendar_2015_2023_argmax_resume": 2_886,
        "calendar_2015_2023_max_changed": 2_666,
        "calendar_2015_2023_limit_hit_at_resume": 1_724,
        "gate_m1_source_n_valid_lt_15": 7_848,
        "gate_m1_matchable_n_valid_lt_15": 5_964,
        "gate_m1_source_n_max_lt_15": 8_576,
        "gate_m1_matchable_n_max_lt_15": 6_692,
        "tail_raw_gt_0_44": 19,
        "tail_raw_gt_1_0": 6,
        "tail_shadow_gt_0_44": 17,
        "tail_shadow_gt_1_0": 3,
    }
    actual = {
        "resume_rows_2014_2025": int(len(resume)),
        "calendar_2015_2023_argmax_resume": cal_counts["argmax_resume"],
        "calendar_2015_2023_max_changed": cal_counts["max_changed"],
        "calendar_2015_2023_limit_hit_at_resume": cal_counts["limit_hit_at_resume_legacy"],
        "gate_m1_source_n_valid_lt_15": gate_m1["source_n_valid_lt_15"],
        "gate_m1_matchable_n_valid_lt_15": gate_m1["monthly_matchable_n_valid_lt_15"],
        "gate_m1_source_n_max_lt_15": gate_m1["source_n_max_lt_15"],
        "gate_m1_matchable_n_max_lt_15": gate_m1["monthly_matchable_n_max_lt_15"],
        "tail_raw_gt_0_44": cal_counts["tail_raw"]["gt_0_44"],
        "tail_raw_gt_1_0": cal_counts["tail_raw"]["gt_1_0"],
        "tail_shadow_gt_0_44": cal_counts["tail_shadow_ungated"]["gt_0_44"],
        "tail_shadow_gt_1_0": cal_counts["tail_shadow_ungated"]["gt_1_0"],
    }
    replication = {
        k: {"expected": int(v), "actual": int(actual[k]),
            "match": bool(actual[k] == v)}
        for k, v in plan_expected.items()
    }
    replication["tail_shadow_gt_0_44"]["note"] = (
        "2.0_plan 写 17；按“排除复牌日后非复牌日 max”严格口径为 14。"
        "17 无法与 max=1.085 同时成立：多出的 3 行只能是复牌日收益，"
        "而这会把 max 推回 >1（sz000995/sz000792/sz000670）。详见执行记录。")
    replication["tail_shadow_gt_1_0"]["note"] = (
        "2.0_plan 写 3；严格口径为 2（sh688026 1.085441、sz300939 1.009931）。"
        "第三个 >1 只能来自复牌日，与 max=1.085 矛盾。")
    replication["tail_raw_gt_0_44"]["note"] = "2015-01..2023-12（日历年）"
    replication["tail_raw_gt_1_0"]["note"] = "2015-01..2023-12（日历年）"
    replication["tail_shadow_gt_0_44"]["note"] += " 窗口=2015-01..2023-12。"
    replication["tail_shadow_gt_1_0"]["note"] += " 窗口=2015-01..2023-12。"

    counts = {
        "p2shadow_version": P2SHADOW_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "shadow_file": {
            "file": str(out_p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": out_sha,
            "rows": int(len(out)),
            "columns": list(source.columns) + NEW_COLS,
        },
        "inputs": {
            "source_month_argmax_panel": {
                "file": str(src_p.relative_to(ROOT)).replace("\\", "/"),
                "sha256_before": src_sha_before,
                "sha256_after": src_sha_after,
                "rows": int(len(source)),
                "read_only": True,
            },
            "daily_panel": {
                "dir": str(panel_dir.relative_to(ROOT)).replace("\\", "/"),
                "years": [args.start_year, args.end_year],
                "files": [
                    {"file": str(p.relative_to(ROOT)).replace("\\", "/"),
                     "sha256": sha256_file(p)}
                    for p in sorted(panel_dir.glob("*.csv"))
                    if args.start_year <= int(p.stem) <= args.end_year
                ],
            },
            "stock_suspend_flag": {
                "file": str(susp_p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(susp_p) if susp_p.is_file() else None,
            },
            "daily_limit_up_flag": {
                "file": str(up_p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(up_p) if up_p.is_file() else None,
                "available": bool(limit_available),
            },
            "daily_limit_down_flag": {
                "file": str(down_p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256_file(down_p) if down_p.is_file() else None,
                "available": bool(limit_available),
            },
            "monthly_returns_clean_v2": {
                "file": str(monthly_p.relative_to(ROOT)).replace("\\", "/"),
                "read_only": True,
            },
            "tie_rule": tie,
        },
        "scope": {
            "daily_years": [args.start_year, args.end_year],
            "min_trade_days": args.min_trade_days,
            "sensitivity_days": args.sensitivity,
            "window_calendar": list(WINDOW_CAL),
            "window_m1": list(WINDOW_M1),
        },
        "verification": verification,
        "resume_rows_2014_2025": int(len(resume)),
        "resume_rows_by_year": {
            str(k): int(v) for k, v in
            resume_audit.assign(y=resume_audit["date"].str[:4])["y"]
            .value_counts().sort_index().items()
        },
        "calendar_2015_2023": cal_counts,
        "m1_window_2015_05_2023_12": m1_counts,
        "gate_counts_calendar_2015_2023": gate_cal,
        "gate_counts_m1_window": gate_m1,
        "limit_flags_calendar_2015_2023": limit_cal,
        "limit_flags_m1_window": limit_m1,
        "limit_source": ("official_daily_flag" if limit_available
                         else "threshold_fallback"),
        "plan_2_0_replication": replication,
        "plan_2_0_all_match": all(v["match"] for v in replication.values()),
        "audit_files": {
            "resume_audit": {
                "file": str(resume_p.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(resume_audit)),
                "sha256": sha256_file(resume_p),
            },
            "max_tail_diff": {
                "file": str(tail_p.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(tail)),
                "sha256": sha256_file(tail_p),
            },
        },
        "headline_anchors_before": anchors_before,
        "headline_anchors_after": anchors_after,
        "headline_anchors_match": all(
            anchors_before[k]["sha256"] == anchors_after[k]["sha256"]
            for k in anchors_before),
        "headline_source_panel_sha_match_registered": bool(
            src_sha_after == EXPECTED_SOURCE_SHA),
        "headline_files_modified": False,
        "m1_rerun": False,
        "configs_modified": False,
    }

    meta = {
        "p2shadow_version": P2SHADOW_VERSION,
        "generated_at_utc": counts["generated_at_utc"],
        "script": counts["script"],
        "script_sha256": counts["script_sha256"],
        "inputs": counts["inputs"],
        "output": counts["shadow_file"],
        "rows_total": int(len(out)),
        "rows_daily_covered_2014_2025": verification["rows_daily_covered_2014_2025"],
        "rows_pre_2014_passthrough": verification["rows_pre_2014_passthrough"],
        "definitions": {
            "resume_flag": "gap_td>1 且存在上一可得行；首行不算复牌；上一可得行含 rraw 为空的行",
            "gap_td": "市场交易日历索引差（daily_panel 日期并集，2014-2025）",
            "MAXraw_p2shadow_ungated": "非复牌日 rraw 的 max；无合格日 -> 空",
            "MAXraw_p2shadow": "15 日门（n_valid_days>=15 且 n_max_days>0）下的非复牌日 max；否则空",
            "argmax_date_p2shadow": "registry tie_rule（earliest）下非复牌日 argmax；15 日门",
            "n_valid_days_p2shadow": "日频有效收益日数（rraw 非空）；应等于 headline n_valid_days",
            "n_resume_days": "月内复牌日数（rraw 非空且 resume_flag=1）",
            "n_max_days": "n_valid_days_p2shadow - n_resume_days（可进入 MAX 的交易日）",
            "limit_hit_p2shadow": "shadow argmax 日的官方 daily_limit_up_flag；缺失回退阈值",
            "limit_status_at_resume": "headline argmax 落在复牌日时统一标 limit_status_unknown（不进入 qualified_limit_hit）",
            "limit_official_status_at_resume": "复牌日的官方细分状态 limit_up/limit_down/no_limit/unknown（仅审计）",
            "qualification_status_p2shadow": "gate 状态或官方 limit 分层；pre-2014 复制 legacy 状态",
            "R_p2shadow": "原 R 只读复制（MAX 与 R 分开）",
            "shadow_status": "evaluable（2014-2025 日频覆盖）/ pre_2014_no_daily_coverage（legacy 直通）",
            "row_order": "保持输入 source-month 面板行序",
            "headline_mutation": "none",
        },
        "headline_anchors_before": anchors_before,
        "headline_anchors_after": anchors_after,
        "headline_anchors_match": counts["headline_anchors_match"],
    }
    write_json_atomic(meta, meta_p)
    counts["meta_file"] = {
        "file": str(meta_p.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(meta_p),
    }
    write_json_atomic(counts, counts_p)

    print(f"[P2.2] shadow -> {out_p.relative_to(ROOT)}  rows={len(out):,}  sha={out_sha[:12]}…")
    print(f"[P2.2] resume 2014–2025={len(resume):,}; "
          f"calendar 2015–2023 argmax_resume={cal_counts['argmax_resume']:,} "
          f"max_changed={cal_counts['max_changed']:,} "
          f"limit_resume={cal_counts['limit_hit_at_resume_legacy']:,}")
    print(f"[P2.2] tail calendar 2015–2023: raw max={cal_counts['tail_raw']['max']:.6f} "
          f">.44={cal_counts['tail_raw']['gt_0_44']} >1={cal_counts['tail_raw']['gt_1_0']}; "
          f"shadow max={cal_counts['tail_shadow_ungated']['max']:.6f} "
          f">.44={cal_counts['tail_shadow_ungated']['gt_0_44']} "
          f">1={cal_counts['tail_shadow_ungated']['gt_1_0']}")
    print(f"[P2.2] gate M1: n_valid<15 {gate_m1['source_n_valid_lt_15']:,}/"
          f"{gate_m1['monthly_matchable_n_valid_lt_15']:,}; n_max<15 "
          f"{gate_m1['source_n_max_lt_15']:,}/{gate_m1['monthly_matchable_n_max_lt_15']:,}")
    print(f"[P2.2] 9 anchors unchanged: {counts['headline_anchors_match']}")
    if not counts["plan_2_0_all_match"]:
        print("[P2.2] WARN: 与 2.0_plan 预期不完全一致（见 counts.plan_2_0_replication）",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
