#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.3：shadow MAX 侧复权/除权面板（只读 headline，只写 shadow）。

背景（`artifacts/phase2/2.0_plan.md` §2.3，审计 V-P1-3）：
  headline `source_month_argmax_panel.csv` 的 `MAXraw` 来自未复权 `rraw`；
  除权日（送转/派息）会产生“除权断崖”，被当成单日负收益进入 MAX。
  `daily_panel` 的 `ca_flag` 在 2014–2017 全为 0（`close_qfq` 全空），
  `corporate_action_events.csv` 只覆盖 2018–2025；因此本脚本：
    - 主复权源：`daily_adj_close_post_20260630.pkl`（后复权价，2001 起），
      由它计算日度复权收益 `r_adj = adj_px_t / adj_px_prev - 1`；
    - 月频对拍：由同一 pkl 月末 last 比值复算 `monthly_returns_adj_full_2006_2025.csv`，
      要求 max abs diff ≤1e-12；
    - 事件源：`sharebonus_events.csv`（`EX_DIVIDEND_DATE` 非空且
      `ASSIGN_PROGRESS=实施分配`，1991+）为主；`corporate_action_events.csv`
      （2018–2025）做交叉验证，并写 `ca_source`；
    - 日频后复权缺失时，用 sharebonus 的送转/派息字段回退，标
      `adj_status=..._with_fallback` / `adj_source=sharebonus_fallback`；
    - `MAXraw_adj` = 月内 `r_adj` max；`MAXraw_adj_ex_resume` = 再排除
      2.2 的 `resume_flag=1` 日；`MAXraw_ex_resume` = 原始收益排除复牌日；
      `MAXraw_ex_rights` = 原始收益排除除权日（分离“除权断崖”与“真极值”）；
    - 旧 11 列逐字段只读复制，新列只追加；行序保持输入顺序。

口径边界（必须保留）：
  - `daily_panel` 只覆盖 2014–2025；pre-2014（180,369 行）没有日度复权收益，
    `MAXraw_adj*` 留空，`shadow_status_adj=pre_2014_no_daily_coverage`，
    不得把 legacy `MAXraw` 当作复权 MAX。
  - 复权 MAX 不得单独使用：后复权价会把复牌首日跳空放大
    （`sz002506` 2015-08 raw 5.937 → adj 9.860），主口径必须与 2.2 的
    `resume_flag` 联合，即 `MAXraw_adj_ex_resume`。
  - 2014–2017 只有 `sharebonus_events` 可用；`corporate_action_events` 的
    2018+ 事件是 `sharebonus` 的子集（本数据 100% 命中），不反过来用
    `corporate_action` 推断 2014–2017。
  - `R` 列只读复制；月频复权收益单独输出（复制既有
    `monthly_returns_adj_full_2006_2025.csv`，不重算）。

只写：
  data/processed/phase2_shadow/source_month_argmax_panel_adj_p2shadow.csv
  data/processed/phase2_shadow/source_month_argmax_panel_adj_p2shadow.meta.json
  data/processed/phase2_shadow/monthly_returns_adj_p2shadow.csv
  artifacts/phase2/2.3_counts.json
  artifacts/phase2/2.3_adj_diff.csv
  artifacts/phase2/2.3_exrights_audit.csv

用法：
  python scripts/data_audit/p2_3_build_adj_max_shadow.py
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
from lib.shadow.p2_3_build_adj_max_shadow import ADJ_PX, ARTIFACTS_DIR, CA_EVENTS, CONTRACTS, COUNTS_OUT, DIFF_OUT, END_YEAR, EXPECTED_ROWS, EXPECTED_SOURCE_SHA, EXRIGHTS_OUT, FORMULA_EVALUABLE, FORMULA_PRE2014, META_OUT, MONTHLY_ADJ, MONTHLY_RAW, NEW_COLS, NEW_COL_KINDS, OUT_MAX, OUT_RETURNS, P2SHADOW_VERSION, PANEL_DIR, RESUME_AUDIT, ROOT, SHADOW_DIR, SHAREBONUS, SHAREBONUS_VALID_STATUS, SOURCE_PANEL, START_YEAR, WINDOW_CAL, WINDOW_M1, _monthly_adj_source, _monthly_adj_status, _num, aggregate_daily_year, anchor_snapshot, assert_anchors_unchanged, build_adj_return_matrix, build_exrights_audit, build_shadow_panel, build_tail_diff, code_column, combine_events, compare_monthly_raw_adj, compute_monthly_aggregates, fallback_adj_return, finalize_panel, format_column, load_contracts, load_resume_monthly, max_row, monthly_adj_from_px, order_book_to_source, parse_corporate_action_frame, parse_sharebonus_frame, process_daily_year, protected_paths, tail_stats, window_mask, write_audit_csv, write_shadow_csv
# --- end scripts/lib bootstrap ---


import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / "scripts" / "data_audit"))

# 只复用 Phase 1 已登记的 tie_rule 解析（不复制逻辑）。
from s4_step1_max import resolve_tie_rule  # noqa: E402




# 追加列（旧列顺序不变，新列只追加在末尾）。


# --------------------------------------------------------------------------- IO






# ---------------------------------------------------------------- headline 锚








# ---------------------------------------------------------------- 代码/事件












# ---------------------------------------------------------------- 复权收益












# ---------------------------------------------------------------- shadow 面板










# ---------------------------------------------------------------- 统计/审计












# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 2.3 shadow adjusted MAX")
    ap.add_argument("--source-panel", default=str(SOURCE_PANEL))
    ap.add_argument("--panel-dir", default=str(PANEL_DIR))
    ap.add_argument("--adj-px", default=str(ADJ_PX))
    ap.add_argument("--sharebonus", default=str(SHAREBONUS))
    ap.add_argument("--ca-events", default=str(CA_EVENTS))
    ap.add_argument("--monthly-adj", default=str(MONTHLY_ADJ))
    ap.add_argument("--monthly-raw", default=str(MONTHLY_RAW))
    ap.add_argument("--resume-audit", default=str(RESUME_AUDIT))
    ap.add_argument("--contracts", default=str(CONTRACTS))
    ap.add_argument("--out-max", default=str(OUT_MAX))
    ap.add_argument("--meta-out", default=str(META_OUT))
    ap.add_argument("--out-returns", default=str(OUT_RETURNS))
    ap.add_argument("--counts-out", default=str(COUNTS_OUT))
    ap.add_argument("--diff-out", default=str(DIFF_OUT))
    ap.add_argument("--exrights-out", default=str(EXRIGHTS_OUT))
    ap.add_argument("--start-year", type=int, default=START_YEAR)
    ap.add_argument("--end-year", type=int, default=END_YEAR)
    ap.add_argument("--expected-rows", type=int, default=EXPECTED_ROWS)
    ap.add_argument("--registry", default=None)
    args = ap.parse_args(argv)

    src_p = Path(args.source_panel).resolve()
    panel_dir = Path(args.panel_dir).resolve()
    adj_px_p = Path(args.adj_px).resolve()
    sb_p = Path(args.sharebonus).resolve()
    ca_p = Path(args.ca_events).resolve()
    monthly_adj_p = Path(args.monthly_adj).resolve()
    monthly_raw_p = Path(args.monthly_raw).resolve()
    resume_p = Path(args.resume_audit).resolve()
    contracts_p = Path(args.contracts).resolve()
    out_p = Path(args.out_max).resolve()
    meta_p = Path(args.meta_out).resolve()
    out_returns_p = Path(args.out_returns).resolve()
    counts_p = Path(args.counts_out).resolve()
    diff_p = Path(args.diff_out).resolve()
    exrights_p = Path(args.exrights_out).resolve()

    contracts = load_contracts(contracts_p)
    protected = protected_paths(contracts)
    shadow_dir = SHADOW_DIR.resolve()
    artifacts_dir = ARTIFACTS_DIR.resolve()
    for label, p in (("out-max", out_p), ("meta-out", meta_p),
                     ("out-returns", out_returns_p), ("counts-out", counts_p),
                     ("diff-out", diff_p), ("exrights-out", exrights_p)):
        if p in protected:
            raise SystemExit(f"[P2.3] {label} 指向 headline 路径，拒绝写出: {p}")
    if out_p.parent != shadow_dir or meta_p.parent != shadow_dir \
            or out_returns_p.parent != shadow_dir:
        raise SystemExit(f"[P2.3] shadow 输出必须位于 {shadow_dir}")
    if any(p.parent != artifacts_dir for p in (counts_p, diff_p, exrights_p)):
        raise SystemExit("[P2.3] 审计输出必须位于 artifacts/phase2/")

    anchors_before = anchor_snapshot(contracts)
    if not all(v["sha_match"] for v in anchors_before.values()):
        bad = [k for k, v in anchors_before.items() if not v["sha_match"]]
        raise SystemExit(f"[P2.3] headline 锚起始不一致: {bad}")

    input_paths = {
        "source_month_argmax_panel": src_p,
        "sharebonus_events": sb_p,
        "corporate_action_events": ca_p,
        "monthly_returns_adj_full_2006_2025": monthly_adj_p,
        "monthly_returns_clean_v2": monthly_raw_p,
        "resume_audit_2_2": resume_p,
        "daily_adj_close_post": adj_px_p,
    }
    for key, p in input_paths.items():
        if not p.is_file():
            raise SystemExit(f"[P2.3] 输入缺失 {key}: {p}")
    inputs_sha_before = {k: sha256_file(p) for k, p in input_paths.items()}
    daily_sha_before = {
        y: sha256_file(panel_dir / f"{y}.csv")
        for y in range(args.start_year, args.end_year + 1)
    }

    source = pd.read_csv(src_p, dtype=str, keep_default_na=False)
    if len(source) != args.expected_rows:
        raise SystemExit(f"[P2.3] source 行数 {len(source)} != {args.expected_rows}")
    if source.duplicated(["source_id", "month"]).any():
        raise SystemExit("[P2.3] source 键 (source_id, month) 不唯一")
    src_sha_before = sha256_file(src_p)
    if src_sha_before != EXPECTED_SOURCE_SHA:
        raise SystemExit(f"[P2.3] headline source panel sha 与登记不一致: {src_sha_before}")
    old_cols = list(source.columns)
    if old_cols != ["source_id", "month", "MAXraw", "argmax_date", "n_valid_days",
                    "R", "tie_rule_version", "formula_id", "limit_hit_at_argmax",
                    "qualification_status", "st_flag_at_argmax"]:
        raise SystemExit(f"[P2.3] source 列序与预期不一致: {old_cols}")

    tie = resolve_tie_rule(args.registry)
    tie_rule = tie["id"]
    print(f"[P2.3] tie_rule={tie_rule} source={tie['source']}", flush=True)

    # ---- 事件：sharebonus 主 + corporate_action 2018+ 交叉
    sb_raw = pd.read_csv(sb_p, dtype=str, keep_default_na=False)
    sb_ev = parse_sharebonus_frame(sb_raw)
    id_map = dict(zip(source["source_id"].str[2:], source["source_id"]))
    ca_raw = pd.read_csv(ca_p, dtype=str, keep_default_na=False)
    ca_ev = parse_corporate_action_frame(ca_raw, id_map)
    ev, ev_month = combine_events(sb_ev, ca_ev)
    event_pairs = set(zip(ev["source_id"], ev["ex_date"]))
    event_meta = {(r.source_id, r.ex_date):
                  (r.bonus_ratio, r.it_ratio, r.dividend)
                  for r in ev.itertuples(index=False)}
    print(f"[P2.3] events: sharebonus={len(sb_ev):,} "
          f"corporate_action={len(ca_ev):,} combined={len(ev):,}", flush=True)

    # ---- 2.2 复牌集
    resume_month, resume_df = load_resume_monthly(resume_p, set(source["source_id"]))
    resume_pairs = set(zip(resume_df["source_id"], resume_df["date"]))
    print(f"[P2.3] resume rows={len(resume_df):,} "
          f"months={len(resume_month):,}", flush=True)

    # ---- 后复权价：月频对拍 + 日度收益
    print("[P2.3] 载入后复权 pkl …", flush=True)
    px = pd.read_pickle(adj_px_p)
    if not isinstance(px, pd.DataFrame):
        raise SystemExit("[P2.3] 后复权 pkl 不是 DataFrame")
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    monthly_calc = monthly_adj_from_px(px)
    ref_adj = pd.read_csv(monthly_adj_p, dtype={"firm_id": str, "month": str})
    cmp = ref_adj.merge(monthly_calc, on=["firm_id", "month"],
                        how="outer", indicator=True)
    both = cmp[cmp["_merge"] == "both"]
    abs_diff = (both["R"] - both["R_calc"]).abs()
    monthly_replication = {
        "rows_reference": int(len(ref_adj)),
        "rows_recomputed": int(len(monthly_calc)),
        "rows_both": int(len(both)),
        "rows_left_only": int((cmp["_merge"] == "left_only").sum()),
        "rows_right_only": int((cmp["_merge"] == "right_only").sum()),
        "max_abs_diff": float(abs_diff.max()) if len(abs_diff) else None,
        "gt_1e_12": int((abs_diff > 1e-12).sum()),
        "gt_1e_6": int((abs_diff > 1e-6).sum()),
    }
    if monthly_replication["gt_1e_12"]:
        raise SystemExit(f"[P2.3] 月频复权对拍失败: {monthly_replication}")
    print(f"[P2.3] 月频复权对拍: rows={len(both):,} "
          f"max_abs_diff={monthly_replication['max_abs_diff']:.3e}", flush=True)

    adj_ret = build_adj_return_matrix(px)
    print(f"[P2.3] 日度复权矩阵 {adj_ret.shape[0]:,} x {adj_ret.shape[1]:,}", flush=True)

    # ---- 日频聚合
    years = range(args.start_year, args.end_year + 1)
    agg, year_stats, event_days, resume_days_seen = compute_monthly_aggregates(
        panel_dir, years, adj_ret, event_pairs, event_meta, resume_pairs, tie_rule)
    del adj_ret, px
    print(f"[P2.3] 月度聚合 {len(agg):,} 行；除权交易日 {len(event_days):,}；"
          f"复牌日 {len(resume_days_seen):,}", flush=True)

    # ---- 组装 shadow
    out, covered, n_valid, max_raw, build_checks = build_shadow_panel(
        source, agg, ev_month, resume_month, resume_pairs, tie_rule)
    out, invariant_checks = finalize_panel(
        out, covered, n_valid, max_raw, ev, resume_pairs, tie_rule)
    if len(out) != args.expected_rows:
        raise SystemExit(f"[P2.3] shadow 行数 {len(out)} != {args.expected_rows}")
    write_shadow_csv(out, old_cols, out_p)
    out_sha = sha256_file(out_p)
    print(f"[P2.3] shadow CSV 已写出 sha={out_sha[:16]}…", flush=True)

    # 旧列只读：回读 shadow 并与 source 逐行比对（分块）。
    old_digest_src = hashlib.sha256()
    for start in range(0, len(source), 100_000):
        old_digest_src.update(
            source.iloc[start:start + 100_000][old_cols]
            .to_csv(index=False, header=False, lineterminator="\n")
            .encode("utf-8"))
    old_digest_out = hashlib.sha256()
    for chunk in pd.read_csv(out_p, usecols=old_cols, dtype=str,
                             keep_default_na=False, chunksize=100_000):
        old_digest_out.update(
            chunk[old_cols].to_csv(index=False, header=False,
                                   lineterminator="\n").encode("utf-8"))
    old_readonly_ok = old_digest_src.hexdigest() == old_digest_out.hexdigest()
    if not old_readonly_ok:
        raise SystemExit("[P2.3] shadow 旧列回读与 source 不一致")
    print("[P2.3] 旧列回读一致", flush=True)

    # ---- 月频复权文件：复制既有登记文件，不重算
    out_returns_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_returns_p.with_name(out_returns_p.name + ".tmp")
    shutil.copyfile(monthly_adj_p, tmp)
    os.replace(tmp, out_returns_p)
    returns_sha = sha256_file(out_returns_p)
    if returns_sha != inputs_sha_before["monthly_returns_adj_full_2006_2025"]:
        raise SystemExit("[P2.3] monthly_returns_adj_p2shadow 与输入不一致")

    # ---- 计数
    def wmask(w):
        return window_mask(out, w)

    cal = out[wmask(WINDOW_CAL)]
    m1 = out[wmask(WINDOW_M1)]
    cal_eval = cal[pd.to_numeric(cal["n_max_days"], errors="coerce").gt(0)]
    m1_eval = m1[pd.to_numeric(m1["n_max_days"], errors="coerce").gt(0)]
    covered_2014_2025 = int(covered.sum())
    pre_2014 = int(out["month"].lt("2014-01").sum())

    tails = {}
    max_rows = {}
    for label, frame in (("all_rows", out), ("calendar_2015_2023", cal),
                         ("calendar_2015_2023_evaluable", cal_eval),
                         ("m1_window", m1), ("m1_window_evaluable", m1_eval)):
        tails[label] = {
            "raw": tail_stats(frame["MAXraw"]),
            "raw_ex_resume": tail_stats(frame["MAXraw_ex_resume"]),
            "raw_ex_rights": tail_stats(frame["MAXraw_ex_rights"]),
            "adj": tail_stats(frame["MAXraw_adj"]),
            "adj_ex_resume": tail_stats(frame["MAXraw_adj_ex_resume"]),
        }
        max_rows[label] = {
            "raw": max_row(frame, "MAXraw"),
            "raw_ex_resume": max_row(frame, "MAXraw_ex_resume"),
            "raw_ex_rights": max_row(frame, "MAXraw_ex_rights"),
            "adj": max_row(frame, "MAXraw_adj"),
            "adj_ex_resume": max_row(frame, "MAXraw_adj_ex_resume"),
        }

    # 2.2 联合：raw 排除复牌日后的 MAX 变化
    resume_changed_cal = int(cal["resume_joint_max_changed"].sum())
    resume_changed_m1 = int(m1["resume_joint_max_changed"].sum())
    adj_changed_cal = int(cal["adj_max_changed_vs_raw"].sum())
    adj_nr_changed_cal = int(cal["adj_ex_resume_changed_vs_adj"].sum())

    # 除权影响
    ca_rows_cal = int(cal["corporate_action_flag"].sum())
    ca_rows_m1 = int(m1["corporate_action_flag"].sum())
    argmax_ex_cal = int(cal["argmax_raw_on_ex_date"].sum())
    ex_rights_changed_cal = int(
        (pd.to_numeric(cal["MAXraw_ex_rights"], errors="coerce").notna()
         & pd.to_numeric(cal["MAXraw"], errors="coerce").notna()
         & (pd.to_numeric(cal["MAXraw_ex_rights"], errors="coerce")
            - pd.to_numeric(cal["MAXraw"], errors="coerce")).abs().gt(1e-8)).sum())
    ex_rights_changed_m1 = int(
        (pd.to_numeric(m1["MAXraw_ex_rights"], errors="coerce").notna()
         & pd.to_numeric(m1["MAXraw"], errors="coerce").notna()
         & (pd.to_numeric(m1["MAXraw_ex_rights"], errors="coerce")
            - pd.to_numeric(m1["MAXraw"], errors="coerce")).abs().gt(1e-8)).sum())

    # 事件覆盖
    sb_years = sb_ev["ex_date"].str[:4].astype(int)
    ca_years = ca_ev["ex_date"].str[:4].astype(int)
    sb_2014_2017 = int(((sb_years >= 2014) & (sb_years <= 2017)).sum())
    sb_2018_2025 = int(((sb_years >= 2018) & (sb_years <= 2025)).sum())
    ca_2018_2025 = int(((ca_years >= 2018) & (ca_years <= 2025)).sum())
    sb_set = set(zip(sb_ev.loc[(sb_years >= 2018) & (sb_years <= 2025), "source_id"],
                     sb_ev.loc[(sb_years >= 2018) & (sb_years <= 2025), "ex_date"]))
    ca_set = set(zip(ca_ev.loc[(ca_years >= 2018) & (ca_years <= 2025), "source_id"],
                     ca_ev.loc[(ca_years >= 2018) & (ca_years <= 2025), "ex_date"]))
    inter = sb_set & ca_set
    crosscheck = {
        "sharebonus_2018_2025": int(len(sb_set)),
        "corporate_action_2018_2025": int(len(ca_set)),
        "matched": int(len(inter)),
        "corporate_action_only": int(len(ca_set - sb_set)),
        "sharebonus_only": int(len(sb_set - ca_set)),
        "match_rate_corporate_action_to_sharebonus": (
            float(len(inter) / len(ca_set)) if ca_set else None),
    }

    # 面板事件覆盖
    ca_source_counts = out["ca_source"].value_counts().to_dict()
    ca_flag_counts = out["corporate_action_flag"].value_counts().to_dict()
    adj_status_counts = out["adj_status"].value_counts().to_dict()
    adj_source_counts = out["adj_source"].value_counts().to_dict()
    pre_2014_ca = int(out.loc[out["month"].lt("2014-01"),
                             "corporate_action_flag"].sum())
    covered_2014_2017 = int(out["month"].between("2014-01", "2017-12").sum())
    covered_2018_2025 = int(out["month"].between("2018-01", "2025-12").sum())
    adj_2015 = int(out.loc[out["month"].between("2015-01", "2015-12"),
                          "MAXraw_adj"].notna().sum())
    adj_2014_2017 = int(out.loc[out["month"].between("2014-01", "2017-12"),
                               "MAXraw_adj"].notna().sum())
    adj_2018_2025 = int(out.loc[out["month"].between("2018-01", "2025-12"),
                               "MAXraw_adj"].notna().sum())

    # 月频 raw vs adj
    monthly_compare = compare_monthly_raw_adj(monthly_raw_p, monthly_adj_p)
    ref_keys = set(zip(ref_adj["firm_id"], ref_adj["month"]))
    pre_keys = list(zip(out.loc[out["month"].lt("2014-01"), "source_id"],
                        out.loc[out["month"].lt("2014-01"), "month"]))

    # 复权对拍窗口（与 2.0_plan 数字对账）
    plan_expected = {
        "raw_lt_neg_0_44_calendar": 12,
        "adj_lt_neg_0_44_calendar": 0,
        "raw_gt_0_44_calendar": 19,
        "adj_gt_0_44_calendar": 22,
        "raw_gt_1_0_calendar": 6,
        "resume_max_changed_calendar": 2666,
        "resume_max_changed_m1": 2396,
    }
    actual = {
        "raw_lt_neg_0_44_calendar": tails["calendar_2015_2023"]["raw"]["lt_neg_0_44"],
        "adj_lt_neg_0_44_calendar": tails["calendar_2015_2023"]["adj"]["lt_neg_0_44"],
        "raw_gt_0_44_calendar": tails["calendar_2015_2023"]["raw"]["gt_0_44"],
        "adj_gt_0_44_calendar": tails["calendar_2015_2023"]["adj"]["gt_0_44"],
        "raw_gt_1_0_calendar": tails["calendar_2015_2023"]["raw"]["gt_1_0"],
        "resume_max_changed_calendar": resume_changed_cal,
        "resume_max_changed_m1": resume_changed_m1,
    }
    replication = {
        k: {"expected": int(v), "actual": int(actual[k]),
            "match": bool(actual[k] == v)}
        for k, v in plan_expected.items()
    }
    replication["adj_lt_neg_0_44_calendar"]["note"] = (
        "2.0_plan 写 adj 0。实际 adj 单独为 4 行，全部 n_max_days=0（整月只有"
        "复牌日）；n_max_days>0 的可评估行为 0，MAXraw_adj_ex_resume 为 0。"
        "这 4 行正是 2.3 必须与 2.2 联合的证据。")
    replication["adj_gt_0_44_calendar"]["note"] = (
        "2.0_plan 写 22。实际 21 = raw 19 + sh600733 2018-09（复牌日 adj 1.525）"
        "+ sh600145 2020-06（n_max_days=0，adj 0.449）；无第 22 行。"
        "adj_ex_resume >0.44 仍为 14（与 2.2 排除复牌后一致）。")
    replication["monthly_R_sign_flips_2015_2023"] = {
        "expected": 1658,
        "actual": monthly_compare["calendar_2015_2023"]["sign_flips_abs_raw_gt_0_2"],
        "match": bool(monthly_compare["calendar_2015_2023"]
                      ["sign_flips_abs_raw_gt_0_2"] == 1658),
        "note": "严格对立符号（raw*adj<0）且 |raw|>0.2；若把 adj==0 也算翻转会得到 1,691。",
    }
    replication["adj_max_calendar"] = {
        "expected_approx": 9.860,
        "actual": tails["calendar_2015_2023"]["adj"]["max"],
        "match": bool(abs((tails["calendar_2015_2023"]["adj"]["max"] or 0) - 9.860) < 1e-2),
    }
    replication["adj_ex_resume_max_le_1_1"] = {
        "expected": "<=1.1",
        "actual": tails["calendar_2015_2023"]["adj_ex_resume"]["max"],
        "match": bool((tails["calendar_2015_2023"]["adj_ex_resume"]["max"] or 0) <= 1.1),
    }

    # 输入 sha after
    inputs_sha_after = {k: sha256_file(p) for k, p in input_paths.items()}
    daily_sha_after = {
        y: sha256_file(panel_dir / f"{y}.csv")
        for y in range(args.start_year, args.end_year + 1)
    }
    inputs_unchanged = all(inputs_sha_before[k] == inputs_sha_after[k]
                           for k in inputs_sha_before)
    daily_unchanged = all(daily_sha_before[y] == daily_sha_after[y]
                          for y in daily_sha_before)
    if not inputs_unchanged or not daily_unchanged:
        raise SystemExit("[P2.3] 输入在运行中变化")
    src_sha_after = sha256_file(src_p)
    if src_sha_after != src_sha_before:
        raise SystemExit("[P2.3] headline source panel 在运行中变化")

    anchors_after = anchor_snapshot(contracts)
    assert_anchors_unchanged(anchors_before, anchors_after)

    # 审计表
    diff = build_tail_diff(out)
    write_audit_csv(diff, diff_p)
    ex_audit = build_exrights_audit(ev, source, out, event_days, id_map)
    write_audit_csv(ex_audit, exrights_p)
    diff_sha = sha256_file(diff_p)
    ex_sha = sha256_file(exrights_p)

    counts = {
        "p2shadow_version": P2SHADOW_VERSION,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "script": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "tie_rule": tie,
        "outputs": {
            "shadow_panel": {
                "file": str(out_p.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(out)),
                "columns": old_cols + NEW_COLS,
                "sha256": out_sha,
                "old_columns_readonly_equal": bool(old_readonly_ok),
            },
            "monthly_returns_adj_p2shadow": {
                "file": str(out_returns_p.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(ref_adj)),
                "sha256": returns_sha,
                "copied_from": "monthly_returns_adj_full_2006_2025",
                "recomputed": False,
            },
            "adj_diff": {
                "file": str(diff_p.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(diff)),
                "sha256": diff_sha,
                "note": "全部 MAX 尾部行 + 各 delta top 500",
            },
            "exrights_audit": {
                "file": str(exrights_p.relative_to(ROOT)).replace("\\", "/"),
                "rows": int(len(ex_audit)),
                "sha256": ex_sha,
                "note": "2014-2025 sharebonus 实施分配事件 + 2018+ corporate_action 交叉",
            },
        },
        "inputs": {
            k: {
                "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256_before": inputs_sha_before[k],
                "sha256_after": inputs_sha_after[k],
                "unchanged": bool(inputs_sha_before[k] == inputs_sha_after[k]),
            }
            for k, p in input_paths.items()
        },
        "daily_panel": {
            "dir": str(panel_dir.relative_to(ROOT)).replace("\\", "/"),
            "years": [args.start_year, args.end_year],
            "files": [
                {"year": y, "sha256_before": daily_sha_before[y],
                 "sha256_after": daily_sha_after[y],
                 "unchanged": bool(daily_sha_before[y] == daily_sha_after[y])}
                for y in sorted(daily_sha_before)
            ],
        },
        "headline_anchors_before": anchors_before,
        "headline_anchors_after": anchors_after,
        "headline_anchors_match": bool(
            all(v["sha_match"] for v in anchors_after.values())),
        "source_panel_sha_before": src_sha_before,
        "source_panel_sha_after": src_sha_after,
        "source_panel_sha_match_registered": bool(src_sha_before == EXPECTED_SOURCE_SHA),
        "headline_files_modified": False,
        "m1_rerun": False,
        "configs_modified": False,
        "rows": {
            "total": int(len(out)),
            "daily_covered_2014_2025": covered_2014_2025,
            "pre_2014_passthrough": pre_2014,
            "covered_2014_2017": covered_2014_2017,
            "covered_2018_2025": covered_2018_2025,
            "rows_with_adj_2014_2025": int(out["MAXraw_adj"].notna().sum()),
            "rows_with_adj_2014_2017": adj_2014_2017,
            "rows_with_adj_2018_2025": adj_2018_2025,
            "rows_with_adj_2015": adj_2015,
            "rows_adj_missing_in_covered": int(
                (covered & out["MAXraw_adj"].isna()).sum()),
        },
        "monthly_adj_replication": monthly_replication,
        "events": {
            "sharebonus_ex_nonnull": int(sb_raw["EX_DIVIDEND_DATE"].fillna("")
                                         .astype(str).str.strip().ne("").sum()),
            "sharebonus_valid_implemented": int(len(sb_ev)),
            "sharebonus_duplicate_code_date_removed": int(
                len(sb_raw[(sb_raw["EX_DIVIDEND_DATE"].fillna("").astype(str)
                            .str.strip().ne(""))
                           & (sb_raw["ASSIGN_PROGRESS"].fillna("").astype(str)
                              .str.strip().isin(SHAREBONUS_VALID_STATUS))])
                - len(sb_ev)),
            "sharebonus_2014_2017": sb_2014_2017,
            "sharebonus_2018_2025": sb_2018_2025,
            "corporate_action_rows": int(len(ca_raw)),
            "corporate_action_2018_2025": ca_2018_2025,
            "combined_unique_events": int(len(ev)),
            "combined_event_days_on_trading_day": int(len(event_days)),
            "crosscheck_2018_2025": crosscheck,
        },
        "panel_event_coverage": {
            "corporate_action_flag_counts": {
                str(k): int(v) for k, v in ca_flag_counts.items()},
            "ca_source_counts": {str(k): int(v) for k, v in ca_source_counts.items()},
            "rows_with_any_ca": int(out["corporate_action_flag"].sum()),
            "rows_with_ca_2014_2017": int(
                out.loc[out["month"].between("2014-01", "2017-12"),
                        "corporate_action_flag"].sum()),
            "rows_with_ca_2018_2025": int(
                out.loc[out["month"].between("2018-01", "2025-12"),
                        "corporate_action_flag"].sum()),
            "rows_with_ca_pre_2014": pre_2014_ca,
        },
        "adj_coverage": {
            "adj_status_counts": {str(k): int(v) for k, v in adj_status_counts.items()},
            "adj_source_counts": {str(k): int(v) for k, v in adj_source_counts.items()},
            "n_adj_fallback_days_total": int(out["n_adj_fallback_days"].fillna(0).sum()),
            "daily_lookup_missing_total": int(sum(s["adj_lookup_missing"]
                                                  for s in year_stats)),
            "daily_fallback_total": int(sum(s["adj_fallback_days"]
                                            for s in year_stats)),
            "by_year": year_stats,
        },
        "tails": tails,
        "max_rows": max_rows,
        "resume_joint": {
            "resume_rows_in_2_2": int(len(resume_df)),
            "resume_days_seen_in_daily": int(len(resume_days_seen)),
            "max_changed_calendar_2015_2023": resume_changed_cal,
            "max_changed_m1_window": resume_changed_m1,
            "adj_vs_raw_changed_calendar": adj_changed_cal,
            "adj_changed_gt_1e_4_calendar": int(
                (pd.to_numeric(cal["delta_MAXraw_adj"], errors="coerce")
                 .abs() > 1e-4).sum()),
            "adj_changed_gt_1e_2_calendar": int(
                (pd.to_numeric(cal["delta_MAXraw_adj"], errors="coerce")
                 .abs() > 1e-2).sum()),
            "adj_ex_resume_vs_adj_changed_calendar": adj_nr_changed_cal,
            "adj_ex_resume_vs_adj_gt_1e_2_calendar": int(
                (pd.to_numeric(cal["delta_MAXraw_adj_ex_resume"], errors="coerce")
                 .abs() > 1e-2).sum()),
            "resume_only_months_calendar": int(
                (pd.to_numeric(cal["n_max_days"], errors="coerce") == 0).sum()),
            "resume_only_months_m1": int(
                (pd.to_numeric(m1["n_max_days"], errors="coerce") == 0).sum()),
            "adj_neg_tail_all_calendar": tails["calendar_2015_2023"]["adj"]["lt_neg_0_44"],
            "adj_neg_tail_evaluable_calendar": tails["calendar_2015_2023_evaluable"]["adj"]["lt_neg_0_44"],
            "adj_ex_resume_neg_tail_calendar": tails["calendar_2015_2023"]["adj_ex_resume"]["lt_neg_0_44"],
            "adj_gt_0_44_all_calendar": tails["calendar_2015_2023"]["adj"]["gt_0_44"],
            "adj_gt_0_44_evaluable_calendar": tails["calendar_2015_2023_evaluable"]["adj"]["gt_0_44"],
            "adj_ex_resume_gt_0_44_calendar": tails["calendar_2015_2023"]["adj_ex_resume"]["gt_0_44"],
            "adj_neg_tail_rows_calendar": [
                {
                    "source_id": str(r["source_id"]),
                    "month": str(r["month"]),
                    "MAXraw": float(r["MAXraw"]),
                    "MAXraw_adj": float(r["MAXraw_adj"]),
                    "argmax_date": str(r["argmax_date"]),
                    "n_valid_days": int(r["n_valid_days"]),
                    "n_max_days": int(r["n_max_days"]),
                    "corporate_action_flag": int(r["corporate_action_flag"]),
                    "note": "整月只有复牌日（n_max_days=0）；adj 单独仍含复牌跳空，"
                            "adj_ex_resume 为空（无合格日）",
                }
                for _, r in cal.loc[
                    pd.to_numeric(cal["MAXraw_adj"], errors="coerce") < -0.44,
                    ["source_id", "month", "MAXraw", "MAXraw_adj", "argmax_date",
                     "n_valid_days", "n_max_days", "corporate_action_flag"]
                ].iterrows()
            ],
            "sz002506_2015_08": {
                "raw_MAXraw": float(out.loc[(out["source_id"] == "sz002506")
                                            & (out["month"] == "2015-08"),
                                            "MAXraw"].iloc[0]),
                "adj_MAXraw": (
                    float(out.loc[(out["source_id"] == "sz002506")
                                  & (out["month"] == "2015-08"),
                                  "MAXraw_adj"].iloc[0])
                    if out.loc[(out["source_id"] == "sz002506")
                               & (out["month"] == "2015-08"),
                               "MAXraw_adj"].notna().any() else None),
                "adj_ex_resume_MAXraw": (
                    float(out.loc[(out["source_id"] == "sz002506")
                                  & (out["month"] == "2015-08"),
                                  "MAXraw_adj_ex_resume"].iloc[0])
                    if out.loc[(out["source_id"] == "sz002506")
                               & (out["month"] == "2015-08"),
                               "MAXraw_adj_ex_resume"].notna().any() else None),
                "resume_days": int(out.loc[(out["source_id"] == "sz002506")
                                           & (out["month"] == "2015-08"),
                                           "n_resume_days"].iloc[0]),
            },
        },
        "corporate_action_impact": {
            "rows_with_ca_calendar": ca_rows_cal,
            "rows_with_ca_m1": ca_rows_m1,
            "argmax_raw_on_ex_date_calendar": argmax_ex_cal,
            "max_changed_by_excluding_ex_dates_calendar": ex_rights_changed_cal,
            "max_changed_by_excluding_ex_dates_m1": ex_rights_changed_m1,
            "max_raw_calendar": tails["calendar_2015_2023"]["raw"]["max"],
            "max_ex_rights_calendar": tails["calendar_2015_2023"]["raw_ex_rights"]["max"],
            "neg_tail_raw_calendar": tails["calendar_2015_2023"]["raw"]["lt_neg_0_44"],
            "neg_tail_ex_rights_calendar": tails["calendar_2015_2023"]["raw_ex_rights"]["lt_neg_0_44"],
            "note": "MAXraw_ex_rights 只剔除除权日，不解复牌；12 个 raw 负极值全部"
                    "是 n_max_days=0 的复牌日（argmax_raw_on_ex_date=0），因此"
                    "ex_rights 的负尾部仍为 12。除权修复必须用 adj；复牌放大必须用"
                    "2.2 resume 排除。",
        },
        "monthly_R_raw_vs_adj": monthly_compare,
        "pre_2014": {
            "rows": pre_2014,
            "adj_computable": int(out.loc[out["month"].lt("2014-01"),
                                          "MAXraw_adj"].notna().sum()),
            "legacy_passthrough": int(out.loc[out["month"].lt("2014-01"),
                                             "shadow_status_adj"]
                                      .eq("pre_2014_no_daily_coverage").sum()),
            "monthly_adj_available": int(sum(1 for k in pre_keys if k in ref_keys)),
        },
        "checks": {
            "rows_expected": EXPECTED_ROWS,
            "key_unique": bool(not out.duplicated(["source_id", "month"]).any()),
            "old_columns_readonly_equal": bool(old_readonly_ok),
            "n_valid_mismatch_vs_headline": build_checks["n_valid_mismatch_vs_headline"],
            "max_raw_mismatch_vs_headline": build_checks["max_raw_mismatch_vs_headline"],
            "resume_count_mismatch_vs_2_2": build_checks["resume_count_mismatch_vs_2_2"],
            "max_adj_ex_resume_le_max_adj": True,
            "max_ex_rights_le_max_raw": True,
            "max_ex_resume_le_max_raw": True,
            "argmax_adj_in_month": True,
            "adj_argmax_on_resume_day": invariant_checks["adj_argmax_on_resume_day"],
            "adj_neg_tail_evaluable_zero": bool(
                tails["calendar_2015_2023_evaluable"]["adj"]["lt_neg_0_44"] == 0),
            "adj_ex_resume_neg_tail_zero": bool(
                tails["calendar_2015_2023"]["adj_ex_resume"]["lt_neg_0_44"] == 0),
            "n_max_days_from_2_2_resume": {
                "m1_not_gated_lt15": int(
                    (m1["n_max_days"].notna()
                     & pd.to_numeric(m1["n_max_days"], errors="coerce").lt(15)).sum()),
                "m1_n_valid_lt15": int(
                    (pd.to_numeric(m1["n_valid_days"], errors="coerce").lt(15)).sum()),
            },
            "inputs_unchanged": bool(inputs_unchanged and daily_unchanged),
        },
        "plan_2_0_replication": replication,
        "plan_2_0_all_match": bool(all(v.get("match", False) for v in replication.values())),
    }
    write_json_atomic(counts, counts_p)

    meta = {
        "p2shadow_version": P2SHADOW_VERSION,
        "generated_at_utc": counts["generated_at_utc"],
        "script": counts["script"],
        "script_sha256": counts["script_sha256"],
        "shadow_panel": counts["outputs"]["shadow_panel"],
        "monthly_returns_adj_p2shadow": counts["outputs"]["monthly_returns_adj_p2shadow"],
        "inputs": counts["inputs"],
        "daily_panel": counts["daily_panel"],
        "headline_anchors_match": counts["headline_anchors_match"],
        "source_panel_sha_match_registered": counts["source_panel_sha_match_registered"],
        "counts_file": {
            "file": str(counts_p.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256_file(counts_p),
        },
        "columns": old_cols + NEW_COLS,
        "new_columns": NEW_COLS,
    }
    write_json_atomic(meta, meta_p)

    print(f"[P2.3] shadow rows={len(out):,} sha={out_sha[:16]}…", flush=True)
    print(f"[P2.3] adj coverage={int(out['MAXraw_adj'].notna().sum()):,} "
          f"pre2014={pre_2014:,}", flush=True)
    print(f"[P2.3] raw tail={tails['calendar_2015_2023']['raw']}", flush=True)
    print(f"[P2.3] adj tail={tails['calendar_2015_2023']['adj']}", flush=True)
    print(f"[P2.3] adj_ex_resume tail="
          f"{tails['calendar_2015_2023']['adj_ex_resume']}", flush=True)
    print(f"[P2.3] counts={counts_p} meta={meta_p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
