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
  python scripts/shadow_max_build.py --min-trade-days 15 --sensitivity 10"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from peermax_max_build import LIMIT_CHINEXT_STAR, LIMIT_MAIN, LIMIT_MAIN_ST, board_of, load_st_flags, resolve_tie_rule


from pathlib import Path
from peermax_max_build import LIMIT_CHINEXT_STAR, LIMIT_MAIN, LIMIT_MAIN_ST, board_of, load_st_flags, resolve_tie_rule
import csv
import math
import numpy as np
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PANEL = ROOT / "data/processed/source_month_argmax_panel.csv"

PANEL_DIR = ROOT / "data/processed/daily_panel"

SUSPEND_FLAG = ROOT / "data/raw/drive/stock_suspend_flag_20260630.pkl"

ST_FLAG = ROOT / "data/raw/drive/stock_st_flag_20260630.pkl"

LIMIT_UP_FLAG = ROOT / "data/raw/drive/daily_limit_up_flag_20260630.pkl"

LIMIT_DOWN_FLAG = ROOT / "data/raw/drive/daily_limit_down_flag_20260630.pkl"

MONTHLY_RETURNS = ROOT / "data/processed/monthly_returns_clean_v2.csv"

CONTRACTS = ROOT / "configs/m1_contracts.yaml"

SHADOW_DIR = ROOT / "data/processed/phase2_shadow"

ARTIFACTS_DIR = ROOT / "artifacts/phase2"

OUT = SHADOW_DIR / "source_month_argmax_panel_p2shadow.csv"

META_OUT = OUT.with_suffix(".meta.json")

COUNTS_OUT = ARTIFACTS_DIR / "2.2_counts.json"

RESUME_AUDIT_OUT = ARTIFACTS_DIR / "2.2_resume_audit.csv"

TAIL_OUT = ARTIFACTS_DIR / "2.2_max_tail_diff.csv"

P2SHADOW_VERSION = "p2shadow_v1"

FORMULA_EVALUABLE = "MAXRAW_P2SHADOW_V1_EX_RESUME"

FORMULA_PRE2014 = "MAXRAW_P2SHADOW_V1_PRE2014_PASSTHROUGH"

START_YEAR = 2014

END_YEAR = 2025

MIN_TRADE_DAYS = 15

SENSITIVITY_DAYS = 10

EXPECTED_ROWS = 734_232

EXPECTED_SOURCE_SHA = "944adbbab95a7b30d858f2943c8ec2844a852a9551b06224b21147d3c4c6023a"

WINDOW_CAL = ("2015-01", "2023-12")

WINDOW_M1 = ("2015-05", "2023-12")

NEW_COLS = [
    "n_valid_days_p2shadow",
    "n_resume_days",
    "n_max_days",
    "MAXraw_p2shadow",
    "argmax_date_p2shadow",
    "MAXraw_p2shadow_ungated",
    "argmax_date_p2shadow_ungated",
    "resume_at_argmax",
    "gap_td_at_argmax",
    "R_includes_resume_gap",
    "R_includes_cross_year_gap",
    "limit_hit_p2shadow",
    "limit_source",
    "limit_status_at_resume",
    "limit_official_status_at_resume",
    "st_flag_at_argmax_p2shadow",
    "max_gate_15",
    "gate_reason",
    "qualification_status_p2shadow",
    "formula_id_p2shadow",
    "tie_rule_version_p2shadow",
    "R_p2shadow",
    "shadow_status",
    "p2shadow_version",
]

NEW_COL_KINDS = {
    "n_valid_days_p2shadow": "int",
    "n_resume_days": "int",
    "n_max_days": "int",
    "MAXraw_p2shadow": "float",
    "argmax_date_p2shadow": "str",
    "MAXraw_p2shadow_ungated": "float",
    "argmax_date_p2shadow_ungated": "str",
    "resume_at_argmax": "int",
    "gap_td_at_argmax": "int",
    "R_includes_resume_gap": "int",
    "R_includes_cross_year_gap": "int",
    "limit_hit_p2shadow": "int",
    "limit_source": "str",
    "limit_status_at_resume": "str",
    "limit_official_status_at_resume": "str",
    "st_flag_at_argmax_p2shadow": "int",
    "max_gate_15": "int",
    "gate_reason": "str",
    "qualification_status_p2shadow": "str",
    "formula_id_p2shadow": "str",
    "tie_rule_version_p2shadow": "str",
    "R_p2shadow": "raw",
    "shadow_status": "str",
    "p2shadow_version": "str",
}

def format_column(series: pd.Series, kind: str) -> list[str]:
    if kind == "int":
        return [fmt_int(v) for v in series.tolist()]
    if kind == "float":
        return [fmt_float(v) for v in series.tolist()]
    if kind == "raw":
        return ["" if v is None or (isinstance(v, float) and not math.isfinite(v))
                else str(v) for v in series.tolist()]
    return ["" if v is None or (isinstance(v, float) and not math.isfinite(v))
            else str(v) for v in series.tolist()]

def write_shadow_csv(df: pd.DataFrame, old_cols: list[str], path: Path,
                     chunksize: int = 50_000) -> None:
    """旧列原样字符串 + 新列固定格式，分块写出，避免全表转字符串的内存峰值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(old_cols + NEW_COLS)
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start:start + chunksize]
            old_rows = chunk[old_cols].to_numpy(dtype=object)
            new_cols = [format_column(chunk[c], NEW_COL_KINDS[c])
                        for c in NEW_COLS]
            for i in range(len(chunk)):
                w.writerow(list(old_rows[i]) + [col[i] for col in new_cols])
    os.replace(tmp, path)

def load_contracts(path: Path = CONTRACTS) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def anchor_snapshot(contracts: dict) -> dict:
    snap = {}
    for key, meta in contracts["m1_inputs"].items():
        p = ROOT / meta["file"]
        if not p.is_file():
            raise SystemExit(f"[P2.2] headline 锚缺失: {p}")
        h = sha256_file(p)
        snap[key] = {
            "file": meta["file"],
            "sha256": h,
            "sha256_registered": meta["sha256"],
            "sha_match": bool(h == meta["sha256"]),
            "rows_expected": int(meta["expected_rows"]),
        }
    return snap

def assert_anchors_unchanged(before: dict, after: dict) -> None:
    for key in before:
        if not before[key]["sha_match"]:
            raise SystemExit(f"[P2.2] 起始锚已不一致: {key}")
        if before[key]["sha256"] != after[key]["sha256"]:
            raise SystemExit(f"[P2.2] headline 锚在运行中变化: {key}")

def protected_paths(contracts: dict) -> set[Path]:
    return {(ROOT / m["file"]).resolve() for m in contracts["m1_inputs"].values()}

def build_market_calendar(panel_dir: Path, years: range) -> list[str]:
    dates: set[str] = set()
    for y in years:
        p = Path(panel_dir) / f"{y}.csv"
        if not p.is_file():
            raise SystemExit(f"[P2.2] daily_panel 缺文件: {p}")
        d = pd.read_csv(p, usecols=["date"], dtype={"date": str})
        dates.update(d["date"].dropna().astype(str).tolist())
    if not dates:
        raise SystemExit("[P2.2] daily_panel 日期为空")
    return sorted(dates)

def compute_resume_flags(frame: pd.DataFrame, calendar_index: dict[str, int],
                         prior_ci: dict[str, int] | None = None,
                         prior_date: dict[str, str] | None = None):
    """单年日频 -> 加 gap_td / resume_flag，并返回跨年状态。

    定义：按 (code, date) 排序；prev 为上一可得行（含 rraw 为空的行）；
    gap_td = 市场交易日索引差；resume_flag=1 当 gap_td>1 且 prev 存在。
    首行（无 prev）不算复牌。
    """
    required = {"code", "date", "rraw"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"[P2.2] 日频输入缺列: {sorted(missing)}")
    d = frame.copy()
    d["_ci"] = d["date"].map(calendar_index)
    if d["_ci"].isna().any():
        bad = d.loc[d["_ci"].isna(), "date"].head(3).tolist()
        raise SystemExit(f"[P2.2] 日期不在市场日历: {bad}")
    d["_ci"] = d["_ci"].astype("int32")
    d = d.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    g = d.groupby("code", sort=False)
    d["_prev_ci"] = g["_ci"].shift(1).astype("float64")
    d["_prev_date"] = g["date"].shift(1).astype(object)
    first = d["_prev_ci"].isna()
    prior_ci = prior_ci or {}
    prior_date = prior_date or {}
    if prior_ci:
        d.loc[first, "_prev_ci"] = d.loc[first, "code"].map(prior_ci)
    if prior_date:
        d.loc[first, "_prev_date"] = d.loc[first, "code"].map(prior_date)
    d["gap_td"] = d["_ci"] - d["_prev_ci"]
    d["resume_flag"] = ((d["gap_td"] > 1) & d["_prev_ci"].notna()).astype("int8")
    last = d.groupby("code", sort=False).tail(1)
    new_ci = dict(prior_ci)
    new_date = dict(prior_date)
    for c, ci_, dt_ in zip(last["code"], last["_ci"], last["date"]):
        new_ci[str(c)] = int(ci_)
        new_date[str(c)] = str(dt_)
    return d, new_ci, new_date

def _argmax_rows(df: pd.DataFrame, keys: list[str], rule: str) -> pd.DataFrame:
    """每组 rraw 最大的行（tie rule 决定并列取最早/最晚日期）。"""
    cols = keys + ["date", "gap_td", "resume_flag", "_prev_date", "rraw"]
    if df.empty:
        return pd.DataFrame(columns=cols).set_index(keys)
    g = df.groupby(keys, sort=False)
    if rule == "earliest":
        idx = g["rraw"].idxmax()
        sub = df.loc[idx, cols]
        return sub.set_index(keys)
    if rule == "latest":
        mx = g["rraw"].transform("max")
        tie = df[df["rraw"].eq(mx)]
        dates = tie.groupby(keys, sort=False)["date"].max().rename("date").reset_index()
        sub = df.merge(dates, on=keys + ["date"], how="inner").drop_duplicates(keys)
        return sub.set_index(keys)
    raise SystemExit(f"[P2.2] 未支持的 tie_rule: {rule}")

def aggregate_year_months(flagged: pd.DataFrame, rule: str) -> pd.DataFrame:
    """单年日频 -> 每 (code, month) 的 MAX/argmax/复牌计数。"""
    r = flagged[flagged["rraw"].notna()].copy()
    if r.empty:
        return pd.DataFrame(columns=[
            "code", "month", "n_valid_days_p2shadow", "n_resume_days",
            "n_max_days", "MAXraw_all", "argmax_all", "gap_td_at_argmax",
            "resume_at_argmax", "argmax_cross_year", "MAXraw_p2shadow_ungated",
            "argmax_date_p2shadow_ungated", "has_resume",
            "has_cross_year_resume"])
    keys = ["code", "month"]
    out = r.groupby(keys, sort=False).agg(
        n_valid_days_p2shadow=("rraw", "size"),
        n_resume_days=("resume_flag", "sum"),
        MAXraw_all=("rraw", "max"),
    )
    out["n_resume_days"] = out["n_resume_days"].astype("int64")
    out["n_max_days"] = out["n_valid_days_p2shadow"] - out["n_resume_days"]

    all_arg = _argmax_rows(r, keys, rule)
    out["argmax_all"] = all_arg["date"]
    out["gap_td_at_argmax"] = all_arg["gap_td"]
    out["resume_at_argmax"] = all_arg["resume_flag"].astype("int64")

    nr = r[r["resume_flag"] == 0]
    if nr.empty:
        out["MAXraw_p2shadow_ungated"] = np.nan
        out["argmax_date_p2shadow_ungated"] = ""
    else:
        nr_arg = _argmax_rows(nr, keys, rule)
        out["MAXraw_p2shadow_ungated"] = nr_arg["rraw"]
        out["argmax_date_p2shadow_ungated"] = nr_arg["date"]

    out["has_resume"] = (out["n_resume_days"] > 0).astype("int64")
    res = r[r["resume_flag"] == 1]
    if res.empty:
        out["has_cross_year_resume"] = 0
    else:
        cross = (
            res["date"].str[:4] != res["_prev_date"].astype(object).str[:4]
        )
        cross_month = (cross.groupby([res["code"], res["month"]], sort=False)
                       .any().rename("has_cross_year_resume"))
        out = out.join(cross_month)
        out["has_cross_year_resume"] = (
            out["has_cross_year_resume"].eq(True).astype("int64"))
    return out.reset_index()

def compute_month_aggregates(panel_dir: Path, years: range, rule: str):
    """按年流式读 daily_panel，返回 (月度聚合, 复牌行审计)。"""
    calendar = build_market_calendar(panel_dir, years)
    calendar_index = {d: i for i, d in enumerate(calendar)}
    prior_ci: dict[str, int] = {}
    prior_date: dict[str, str] = {}
    agg_parts, resume_parts = [], []
    for y in years:
        p = Path(panel_dir) / f"{y}.csv"
        d = pd.read_csv(p, usecols=["code", "date", "rraw"],
                        dtype={"code": str, "date": str, "rraw": "float64"})
        flagged, prior_ci, prior_date = compute_resume_flags(
            d, calendar_index, prior_ci, prior_date)
        flagged["month"] = flagged["date"].str[:7]
        agg_parts.append(aggregate_year_months(flagged, rule))
        res = flagged[flagged["resume_flag"] == 1].copy()
        if not res.empty:
            res["cross_year"] = (
                res["date"].str[:4] != res["_prev_date"].astype(object).str[:4]
            )
            res["cross_month"] = (
                res["date"].str[:7] != res["_prev_date"].astype(object).str[:7]
            )
            resume_parts.append(res[[
                "code", "date", "_prev_date", "gap_td", "rraw", "month",
                "cross_year", "cross_month"]])
        print(f"[P2.2] {y}: daily={len(d):,} resume={int(flagged['resume_flag'].sum()):,}",
              flush=True)
        del d, flagged
    agg = pd.concat(agg_parts, ignore_index=True, sort=False)
    resume = (pd.concat(resume_parts, ignore_index=True, sort=False)
              if resume_parts else pd.DataFrame())
    return agg, resume, calendar

def load_bool_panel(path: Path) -> pd.DataFrame:
    d = pd.read_pickle(path)
    if not isinstance(d, pd.DataFrame):
        raise SystemExit(f"[P2.2] flag pkl 不是 DataFrame: {path}")
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    d.index = d.index.strftime("%Y-%m-%d")
    for c in d.columns:
        s = d[c]
        if s.dtype == object:
            s = s.map({"True": True, "False": False, True: True, False: False})
        d[c] = s.where(s.notna(), np.nan).astype("float64")
    return d

def lookup_panel(panel: pd.DataFrame, dates: pd.Series,
                 columns: pd.Series) -> np.ndarray:
    """按 (date, column) 批量取值；缺失 -> NaN。"""
    di = pd.Index(["" if (v is None or (isinstance(v, float) and not math.isfinite(v)))
                   else str(v) for v in dates.tolist()])
    ci = pd.Index(columns.tolist())
    rows = panel.index.get_indexer(di)
    cols = panel.columns.get_indexer(ci)
    vals = panel.to_numpy(dtype="float64")
    out = np.full(len(dates), np.nan, dtype="float64")
    ok = (rows >= 0) & (cols >= 0)
    out[ok] = vals[rows[ok], cols[ok]]
    return out

def code_column(code: str) -> str:
    return code[2:] + (".SH" if code.startswith("sh") else ".SZ")

def threshold_limit_hit(code: str, maxv: float, st_val) -> tuple[int, bool]:
    """Phase 1 阈值口径（与 peermax_max_build 一致）；返回 (hit, unknown)。"""
    board = board_of(code)
    if board in ("chinext", "star"):
        thr = LIMIT_CHINEXT_STAR
    elif st_val is True:
        thr = LIMIT_MAIN_ST
    elif st_val is False:
        thr = LIMIT_MAIN
    else:
        thr = LIMIT_MAIN
        if 0.05 <= maxv < LIMIT_MAIN:
            return 0, True
    return (1 if maxv >= thr else 0), False

def st_at(st_df, st_cols: dict, date: str, code: str, cache: dict):
    if st_df is None or code[2:] not in st_cols:
        return np.nan
    row = cache.get(date)
    if row is None:
        try:
            row = st_df.loc[date]
        except KeyError:
            row = False
        cache[date] = row
    if row is False:
        return np.nan
    col = st_cols[code[2:]]
    if col not in row.index:
        return np.nan
    v = row[col]
    if v is None:
        return np.nan
    try:
        if v != v:
            return np.nan
    except Exception:
        return np.nan
    return bool(v)

def build_shadow_panel(source: pd.DataFrame, agg: pd.DataFrame, resume: pd.DataFrame,
                       tie_rule: str, up_panel: pd.DataFrame | None,
                       down_panel: pd.DataFrame | None, st_flag=ST_FLAG,
                       min_trade_days: int = MIN_TRADE_DAYS,
                       sensitivity_days: int = SENSITIVITY_DAYS):
    """把月度聚合/复牌审计接到 source-month 面板；返回完整内部表与审计表。"""
    old_cols = list(source.columns)
    collisions = [c for c in NEW_COLS if c in old_cols]
    if collisions:
        raise SystemExit(f"[P2.2] 输入已含 shadow 列，拒绝覆盖: {collisions}")

    agg = agg.rename(columns={"code": "source_id"})
    out = source.copy()
    out["_row"] = np.arange(len(out))
    out = out.merge(agg, on=["source_id", "month"], how="left",
                    validate="one_to_one", indicator=True)
    if (out["_merge"] == "right_only").any():
        raise SystemExit("[P2.2] 月度聚合存在 source 面板外键")
    out = out.sort_values("_row", kind="mergesort").reset_index(drop=True)
    covered = out["_merge"].eq("both")
    if int(covered.sum()) != len(agg):
        raise SystemExit("[P2.2] 日频覆盖行数与聚合行数不一致")

    n_valid_legacy = pd.to_numeric(out["n_valid_days"], errors="coerce")
    max_legacy = pd.to_numeric(out["MAXraw"], errors="coerce")
    n_valid = out["n_valid_days_p2shadow"].where(covered, n_valid_legacy)
    n_max = out["n_max_days"].where(covered, np.nan)
    max_nr = out["MAXraw_p2shadow_ungated"].where(covered, max_legacy)
    argmax_nr = out["argmax_date_p2shadow_ungated"].where(covered, out["argmax_date"])
    # 覆盖行里没有非复牌日 -> 无 shadow argmax。
    max_nr = max_nr.where(~(covered & (n_max.fillna(0) == 0)), np.nan)
    argmax_nr = argmax_nr.where(~(covered & (n_max.fillna(0) == 0)), "")

    gate_ok = covered & n_valid.ge(min_trade_days) & n_max.gt(0)
    insufficient = covered & n_valid.lt(min_trade_days)
    resume_excluded = covered & n_valid.ge(min_trade_days) & n_max.eq(0)
    pre = ~covered

    # 构造正确性：覆盖行必须与 headline 逐值对齐。
    nvalid_mismatch = int((covered & (n_valid != n_valid_legacy)).sum())
    max_mismatch = int((covered & (np.abs(out["MAXraw_all"] - max_legacy) > 1e-8)).sum())
    argmax_mismatch = int((covered & (out["argmax_all"] != out["argmax_date"])).sum())
    if nvalid_mismatch or max_mismatch or argmax_mismatch:
        raise SystemExit(
            f"[P2.2] headline 对齐失败: n_valid={nvalid_mismatch}, "
            f"MAX={max_mismatch}, argmax={argmax_mismatch}")

    # ---- 复牌审计 + legacy argmax 标记
    if resume.empty:
        resume = pd.DataFrame(columns=[
            "code", "date", "_prev_date", "gap_td", "rraw", "month",
            "cross_year", "cross_month"])
    resume = resume.copy()
    resume["source_id"] = resume["code"]
    resume["prev_date"] = resume["_prev_date"].astype(object)
    resume = resume.merge(
        source[["source_id", "month", "argmax_date", "limit_hit_at_argmax"]],
        on=["source_id", "month"], how="left", suffixes=("", "_panel"),
        validate="many_to_one")
    resume["is_argmax_legacy"] = (
        resume["argmax_date"].notna() & resume["date"].eq(resume["argmax_date"])
    ).astype("int64")
    resume["limit_hit_legacy_threshold"] = pd.to_numeric(
        resume["limit_hit_at_argmax"], errors="coerce")

    # ---- 官方 limit flags（legacy argmax / shadow argmax / 复牌日）
    has_up = up_panel is not None
    has_down = down_panel is not None
    if has_up:
        up_legacy = lookup_panel(
            up_panel, out["argmax_date"].replace("", np.nan),
            out["source_id"].map(code_column))
        down_legacy = (lookup_panel(
            down_panel, out["argmax_date"].replace("", np.nan),
            out["source_id"].map(code_column)) if has_down else np.nan)
        up_shadow = lookup_panel(
            up_panel, argmax_nr.replace("", np.nan),
            out["source_id"].map(code_column))
        down_shadow = (lookup_panel(
            down_panel, argmax_nr.replace("", np.nan),
            out["source_id"].map(code_column)) if has_down else np.nan)
    else:
        up_legacy = np.full(len(out), np.nan)
        down_legacy = np.full(len(out), np.nan)
        up_shadow = np.full(len(out), np.nan)
        down_shadow = np.full(len(out), np.nan)

    out["_up_legacy"] = up_legacy
    out["_down_legacy"] = down_legacy
    out["_up_shadow"] = up_shadow
    out["_down_shadow"] = down_shadow

    st_df, st_cols = load_st_flags(st_flag)
    st_cache: dict = {}

    def st_for(date: str, code: str):
        return st_at(st_df, st_cols, date, code, st_cache)

    # legacy argmax 的阈值口径（用于官方 vs 阈值差异；不改变 legacy 列）
    legacy_thr = []
    legacy_unknown = []
    for code, date, mx in zip(out["source_id"], out["argmax_date"], max_legacy):
        if not date:
            legacy_thr.append(np.nan)
            legacy_unknown.append(False)
            continue
        hit, unk = threshold_limit_hit(code, float(mx), st_for(date, code))
        legacy_thr.append(hit)
        legacy_unknown.append(unk)
    out["_legacy_thr_hit"] = legacy_thr
    out["_legacy_thr_unknown"] = legacy_unknown

    # shadow argmax 的阈值口径（敏感性/差异计数；主口径用官方）
    shadow_thr = []
    shadow_unknown = []
    for code, date, mx, ok in zip(out["source_id"], argmax_nr,
                                  out["MAXraw_p2shadow_ungated"], gate_ok):
        if not ok or not date:
            shadow_thr.append(np.nan)
            shadow_unknown.append(False)
            continue
        hit, unk = threshold_limit_hit(code, float(mx), st_for(date, code))
        shadow_thr.append(hit)
        shadow_unknown.append(unk)
    out["_shadow_thr_hit"] = shadow_thr
    out["_shadow_thr_unknown"] = shadow_unknown

    # ---- 新列
    out["n_valid_days_p2shadow"] = n_valid.round(0)
    out["n_resume_days"] = out["n_resume_days"].where(covered, np.nan)
    out["n_max_days"] = n_max
    out["MAXraw_p2shadow"] = max_nr.where(gate_ok | pre, np.nan)
    out["argmax_date_p2shadow"] = argmax_nr.where(gate_ok | pre, "")
    out["MAXraw_p2shadow_ungated"] = max_nr
    out["argmax_date_p2shadow_ungated"] = argmax_nr
    out["resume_at_argmax"] = out["resume_at_argmax"].where(covered, np.nan)
    out["gap_td_at_argmax"] = out["gap_td_at_argmax"].where(covered, np.nan)
    out["R_includes_resume_gap"] = out["has_resume"].where(covered, np.nan)
    out["R_includes_cross_year_gap"] = out["has_cross_year_resume"].where(covered, np.nan)

    # 官方 up flag 优先；缺失 -> 阈值回退。
    official_shadow = np.isfinite(out["_up_shadow"]).to_numpy()
    hit_shadow = np.where(official_shadow, out["_up_shadow"],
                          out["_shadow_thr_hit"])
    hit_shadow = pd.Series(hit_shadow, index=out.index).where(gate_ok, np.nan)
    out["limit_hit_p2shadow"] = hit_shadow

    official_legacy = np.isfinite(out["_up_legacy"]).to_numpy()
    hit_legacy = np.where(official_legacy, out["_up_legacy"], out["_legacy_thr_hit"])
    hit_legacy = pd.Series(hit_legacy, index=out.index)
    out["limit_hit_p2shadow"] = out["limit_hit_p2shadow"].where(
        ~pre, hit_legacy.where(pre, np.nan))

    source_official = np.where(gate_ok, official_shadow, official_legacy)
    out["limit_source"] = np.where(
        source_official, "official_daily_flag", "threshold_fallback")
    out["limit_source"] = out["limit_source"].where(
        gate_ok | pre, "not_applicable")

    # 复牌日 limit 状态：按 2.0_plan 统一标 limit_status_unknown（不进入 qualified），
    # 另存官方细分状态供审计（limit_up / limit_down / no_limit / unknown）。
    official_status = []
    for resume_flag, up, down, date in zip(
            out["resume_at_argmax"], out["_up_legacy"], out["_down_legacy"],
            out["argmax_date"]):
        if not (resume_flag == 1):
            official_status.append("")
            continue
        if not date or not np.isfinite(up):
            official_status.append("unknown")
        elif up == 1:
            official_status.append("limit_up")
        elif np.isfinite(down) and down == 1:
            official_status.append("limit_down")
        else:
            official_status.append("no_limit")
    out["limit_official_status_at_resume"] = official_status
    out["limit_status_at_resume"] = np.where(
        out["resume_at_argmax"].eq(1), "limit_status_unknown", "")

    st_shadow = pd.Series([
        st_for(date, code) if (ok and date) else np.nan
        for code, date, ok in zip(out["source_id"], argmax_nr, gate_ok)
    ], index=out.index)
    out["st_flag_at_argmax_p2shadow"] = st_shadow.map(
        lambda v: np.nan if v is None or (isinstance(v, float) and not math.isfinite(v))
        else int(bool(v)))

    out["max_gate_15"] = np.where(covered, gate_ok.astype(int), np.nan)
    out["gate_reason"] = np.select(
        [pre, insufficient, resume_excluded],
        ["not_evaluable_pre_2014", "insufficient_trade_days", "resume_excluded"],
        default="none")

    qual = out["qualification_status"].where(pre, None)
    qual = qual.where(pre, np.where(
        insufficient, "insufficient_trade_days",
        np.where(resume_excluded, "resume_excluded",
                 np.where(pd.Series(hit_shadow, index=out.index).eq(1),
                          "qualified_limit_hit", "qualified_non_limit"))))
    out["qualification_status_p2shadow"] = qual
    out["formula_id_p2shadow"] = np.where(pre, FORMULA_PRE2014, FORMULA_EVALUABLE)
    out["tie_rule_version_p2shadow"] = tie_rule
    out["R_p2shadow"] = out["R"]
    out["shadow_status"] = np.where(pre, "pre_2014_no_daily_coverage", "evaluable")
    out["p2shadow_version"] = P2SHADOW_VERSION

    # 硬验收：非复牌月份逐值一致；shadow <= legacy；shadow argmax 不是复牌日。
    zero_mask = covered & out["n_resume_days"].eq(0)
    nonresume_change = int((zero_mask & (
        np.abs(out["MAXraw_p2shadow_ungated"] - max_legacy) > 1e-12)).sum())
    gt_legacy = int((covered & (
        out["MAXraw_p2shadow_ungated"] > max_legacy + 1e-9)).sum())
    # 用 (source,date) 精确判断 shadow argmax 是否复牌日
    resume_pairs = set(zip(resume["source_id"], resume["date"]))
    shadow_argmax_resume = int(sum(
        1 for code, d in zip(out["source_id"], out["argmax_date_p2shadow_ungated"])
        if d and (code, d) in resume_pairs))
    if nonresume_change or gt_legacy or shadow_argmax_resume:
        raise SystemExit(
            f"[P2.2] 硬验收失败: nonresume_change={nonresume_change}, "
            f"gt_legacy={gt_legacy}, shadow_argmax_resume={shadow_argmax_resume}")

    verification = {
        "rows_total": int(len(out)),
        "rows_daily_covered_2014_2025": int(covered.sum()),
        "rows_pre_2014_passthrough": int(pre.sum()),
        "n_valid_mismatch_vs_headline": nvalid_mismatch,
        "max_mismatch_vs_headline": max_mismatch,
        "argmax_mismatch_vs_headline": argmax_mismatch,
        "non_resume_months_changed": nonresume_change,
        "shadow_gt_legacy_violations": gt_legacy,
        "shadow_argmax_on_resume_day": shadow_argmax_resume,
        "all_resume_months_no_eligible_day": int((covered & n_max.eq(0)).sum()),
        "resume_excluded_gate15": int(resume_excluded.sum()),
        "months_insufficient_trade_days": int(insufficient.sum()),
        "months_gate15_pass": int(gate_ok.sum()),
    }
    return out, resume, verification

def _tail_stats(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n_nonnull": int(len(v)),
        "max": None if v.empty else float(v.max()),
        "min": None if v.empty else float(v.min()),
        "gt_0_44": int((v > 0.44).sum()),
        "gt_1_0": int((v > 1.0).sum()),
        "lt_neg_0_44": int((v < -0.44).sum()),
        "lt_neg_1_0": int((v < -1.0).sum()),
    }

def _max_row(df: pd.DataFrame, col: str) -> dict:
    v = pd.to_numeric(df[col], errors="coerce")
    if v.notna().sum() == 0:
        return {}
    i = v.idxmax()
    return {
        "source_id": str(df.at[i, "source_id"]),
        "month": str(df.at[i, "month"]),
        "value": float(v.at[i]),
        "argmax_date": str(df.at[i, "argmax_date"]),
        "argmax_date_p2shadow": str(df.at[i, "argmax_date_p2shadow_ungated"]),
        "n_valid_days": int(df.at[i, "n_valid_days"]),
        "n_max_days": None if pd.isna(df.at[i, "n_max_days"])
        else int(df.at[i, "n_max_days"]),
    }

def _window(df: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    return df[(df["month"] >= window[0]) & (df["month"] <= window[1])]

def _counts_for_window(w: pd.DataFrame) -> dict:
    max_legacy = pd.to_numeric(w["MAXraw"], errors="coerce")
    max_nr = pd.to_numeric(w["MAXraw_p2shadow_ungated"], errors="coerce")
    max_gated = pd.to_numeric(w["MAXraw_p2shadow"], errors="coerce")
    n_max = pd.to_numeric(w["n_max_days"], errors="coerce")
    changed = w["shadow_status"].eq("evaluable") & n_max.gt(0) & (
        np.abs(max_nr - max_legacy) > 1e-12)
    all_resume = w["shadow_status"].eq("evaluable") & n_max.eq(0)
    return {
        "rows": int(len(w)),
        "argmax_resume": int(w["resume_at_argmax"].eq(1).sum()),
        "max_changed": int(changed.sum()),
        "max_all_resume_excluded": int(all_resume.sum()),
        "limit_hit_at_resume_legacy": int((
            pd.to_numeric(w["limit_hit_at_argmax"], errors="coerce").eq(1)
            & w["resume_at_argmax"].eq(1)).sum()),
        "tail_raw": _tail_stats(max_legacy),
        "tail_shadow_ungated": _tail_stats(max_nr),
        "tail_shadow_gated": _tail_stats(max_gated),
        "max_raw_row": _max_row(w, "MAXraw"),
        "max_shadow_ungated_row": _max_row(w, "MAXraw_p2shadow_ungated"),
    }

def _limit_counts(w: pd.DataFrame) -> dict:
    legacy = pd.to_numeric(w["limit_hit_at_argmax"], errors="coerce")
    off_legacy = pd.to_numeric(w["_up_legacy"], errors="coerce")
    thr_legacy = pd.to_numeric(w["_legacy_thr_hit"], errors="coerce")
    off_shadow = pd.to_numeric(w["_up_shadow"], errors="coerce")
    thr_shadow = pd.to_numeric(w["_shadow_thr_hit"], errors="coerce")
    gate_ok = pd.to_numeric(w["max_gate_15"], errors="coerce").eq(1)
    both_l = off_legacy.notna() & legacy.notna()
    both_s = gate_ok & off_shadow.notna() & thr_shadow.notna()
    resume = w["resume_at_argmax"].eq(1)
    return {
        "legacy_argmax_official_available": int(off_legacy.notna().sum()),
        "legacy_argmax_official_vs_headline_threshold": {
            "both_hit": int((both_l & off_legacy.eq(1) & legacy.eq(1)).sum()),
            "both_no_hit": int((both_l & off_legacy.eq(0) & legacy.eq(0)).sum()),
            "official_hit_threshold_no": int((
                both_l & off_legacy.eq(1) & legacy.eq(0)).sum()),
            "official_no_threshold_hit": int((
                both_l & off_legacy.eq(0) & legacy.eq(1)).sum()),
            "official_missing": int(off_legacy.isna().sum()),
        },
        "shadow_argmax_official_vs_threshold": {
            "rows_with_both": int(both_s.sum()),
            "both_hit": int((both_s & off_shadow.eq(1) & thr_shadow.eq(1)).sum()),
            "both_no_hit": int((both_s & off_shadow.eq(0) & thr_shadow.eq(0)).sum()),
            "official_hit_threshold_no": int((
                both_s & off_shadow.eq(1) & thr_shadow.eq(0)).sum()),
            "official_no_threshold_hit": int((
                both_s & off_shadow.eq(0) & thr_shadow.eq(1)).sum()),
            "threshold_unknown_zone": int((
                both_s & w["_shadow_thr_unknown"]).sum()),
            "official_missing": int((gate_ok & off_shadow.isna()).sum()),
        },
        "limit_source_distribution": {
            str(k): int(v) for k, v in
            w["limit_source"].value_counts(dropna=False).items()
        },
        "resume_argmax": {
            "rows": int(resume.sum()),
            "legacy_threshold_hit": int((
                resume & legacy.eq(1)).sum()),
            "official_up": int((resume & off_legacy.eq(1)).sum()),
            "official_down": int((
                resume & pd.to_numeric(w["_down_legacy"], errors="coerce").eq(1)).sum()),
            "status": {
                str(k): int(v) for k, v in
                w.loc[resume, "limit_status_at_resume"].value_counts().items()
            },
            "official_status": {
                str(k): int(v) for k, v in
                w.loc[resume, "limit_official_status_at_resume"].value_counts().items()
            },
        },
    }

def _gate_counts(w: pd.DataFrame, monthly: pd.DataFrame,
                 min_trade_days: int, sensitivity_days: int) -> dict:
    n_valid = pd.to_numeric(w["n_valid_days_p2shadow"], errors="coerce")
    n_max = pd.to_numeric(w["n_max_days"], errors="coerce")
    m = monthly.rename(columns={"firm_id": "source_id"})
    j = w[["source_id", "month"]].copy()
    j["n_valid"] = n_valid.to_numpy()
    j["n_max"] = n_max.to_numpy()
    j = j.merge(m[["source_id", "month", "complete_month"]],
                on=["source_id", "month"], how="left", validate="one_to_one")
    complete = pd.to_numeric(j["complete_month"], errors="coerce").eq(1)
    return {
        "source_rows": int(len(j)),
        "monthly_matched_rows": int(j["complete_month"].notna().sum()),
        f"source_n_valid_lt_{min_trade_days}": int((j["n_valid"] < min_trade_days).sum()),
        f"monthly_matchable_n_valid_lt_{min_trade_days}": int((
            (j["n_valid"] < min_trade_days) & complete).sum()),
        f"source_n_max_lt_{min_trade_days}": int((j["n_max"] < min_trade_days).sum()),
        f"monthly_matchable_n_max_lt_{min_trade_days}": int((
            (j["n_max"] < min_trade_days) & complete).sum()),
        f"source_n_max_lt_{sensitivity_days}": int((j["n_max"] < sensitivity_days).sum()),
        f"monthly_matchable_n_max_lt_{sensitivity_days}": int((
            (j["n_max"] < sensitivity_days) & complete).sum()),
        f"source_n_valid_lt_{sensitivity_days}": int((j["n_valid"] < sensitivity_days).sum()),
        f"monthly_matchable_n_valid_lt_{sensitivity_days}": int((
            (j["n_valid"] < sensitivity_days) & complete).sum()),
    }

def _tail_diff_table(out: pd.DataFrame) -> pd.DataFrame:
    d = out[out["shadow_status"].eq("evaluable")].copy()
    max_legacy = pd.to_numeric(d["MAXraw"], errors="coerce")
    max_nr = pd.to_numeric(d["MAXraw_p2shadow_ungated"], errors="coerce")
    delta = max_nr - max_legacy
    mask = (d["month"] >= WINDOW_CAL[0]) & (d["month"] <= WINDOW_CAL[1])
    tail = mask & ((max_legacy > 0.44) | (max_legacy < -0.44)
                   | (max_nr > 0.44) | (max_nr < -0.44))
    top = delta.abs().nlargest(50).index
    sel = d.loc[tail | d.index.isin(top)].copy()
    sel["max_delta"] = delta.loc[sel.index]
    cols = ["source_id", "month", "MAXraw", "argmax_date", "n_valid_days",
            "MAXraw_p2shadow_ungated", "argmax_date_p2shadow_ungated",
            "n_max_days", "n_resume_days", "resume_at_argmax",
            "gap_td_at_argmax", "limit_hit_at_argmax", "limit_hit_p2shadow",
            "limit_status_at_resume", "limit_official_status_at_resume",
            "max_delta"]
    return sel[cols].sort_values(["month", "source_id"], kind="mergesort")

def _resume_audit_table(resume: pd.DataFrame, up_panel, down_panel,
                        st_df, st_cols) -> pd.DataFrame:
    d = resume.copy()
    if d.empty:
        return d
    up = lookup_panel(up_panel, pd.Series(d["date"].tolist(), index=d.index),
                      d["source_id"].map(code_column)) if up_panel is not None else np.nan
    dn = lookup_panel(down_panel, pd.Series(d["date"].tolist(), index=d.index),
                      d["source_id"].map(code_column)) if down_panel is not None else np.nan
    d["limit_up_official"] = up
    d["limit_down_official"] = dn
    d["limit_status"] = np.select(
        [np.isfinite(up) & (up == 1), np.isfinite(dn) & (dn == 1)],
        ["limit_up", "limit_down"], default=np.where(np.isfinite(up), "no_limit", "unknown"))
    out = pd.DataFrame({
        "source_id": d["source_id"],
        "date": d["date"],
        "month": d["month"],
        "prev_date": d["prev_date"],
        "gap_td": d["gap_td"],
        "rraw": d["rraw"],
        "cross_month": d["cross_month"].astype(int),
        "cross_year": d["cross_year"].astype(int),
        "is_argmax_legacy": d["is_argmax_legacy"],
        "limit_up_official": d["limit_up_official"],
        "limit_down_official": d["limit_down_official"],
        "limit_status": d["limit_status"],
        "legacy_limit_hit_threshold": d["limit_hit_legacy_threshold"],
    })
    return out.sort_values(["date", "source_id"], kind="mergesort")

def write_audit_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


import argparse
import csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 只复用 Phase 1 已登记实现：tie rule / ST / 板块 / limit 阈值（不复制逻辑）。
from peermax_max_build import LIMIT_CHINEXT_STAR, LIMIT_MAIN, LIMIT_MAIN_ST, board_of, load_st_flags, resolve_tie_rule




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
