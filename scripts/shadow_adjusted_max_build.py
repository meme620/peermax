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
  python scripts/shadow_adjusted_max_build.py"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from peermax_max_build import resolve_tie_rule  # noqa: E402


from pathlib import Path
import csv
import math
import numpy as np
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_PANEL = ROOT / "data/processed/source_month_argmax_panel.csv"

PANEL_DIR = ROOT / "data/processed/daily_panel"

ADJ_PX = ROOT / "data/raw/drive/daily_adj_close_post_20260630.pkl"

SHAREBONUS = ROOT / "data/processed/sharebonus_events.csv"

CA_EVENTS = ROOT / "data/processed/corporate_action_events.csv"

MONTHLY_ADJ = ROOT / "data/processed/monthly_returns_adj_full_2006_2025.csv"

MONTHLY_RAW = ROOT / "data/processed/monthly_returns_clean_v2.csv"

RESUME_AUDIT = ROOT / "artifacts/phase2/2.2_resume_audit.csv"

CONTRACTS = ROOT / "configs/m1_contracts.yaml"

SHADOW_DIR = ROOT / "data/processed/phase2_shadow"

ARTIFACTS_DIR = ROOT / "artifacts/phase2"

OUT_MAX = SHADOW_DIR / "source_month_argmax_panel_adj_p2shadow.csv"

META_OUT = OUT_MAX.with_suffix(".meta.json")

OUT_RETURNS = SHADOW_DIR / "monthly_returns_adj_p2shadow.csv"

COUNTS_OUT = ARTIFACTS_DIR / "2.3_counts.json"

DIFF_OUT = ARTIFACTS_DIR / "2.3_adj_diff.csv"

EXRIGHTS_OUT = ARTIFACTS_DIR / "2.3_exrights_audit.csv"

P2SHADOW_VERSION = "p2shadow_v1"

FORMULA_EVALUABLE = "MAXRAW_P2SHADOW_ADJ_EX_RESUME_V1"

FORMULA_PRE2014 = "MAXRAW_P2SHADOW_ADJ_PRE2014_PASSTHROUGH"

START_YEAR = 2014

END_YEAR = 2025

EXPECTED_ROWS = 734_232

EXPECTED_SOURCE_SHA = "944adbbab95a7b30d858f2943c8ec2844a852a9551b06224b21147d3c4c6023a"

WINDOW_CAL = ("2015-01", "2023-12")

WINDOW_M1 = ("2015-05", "2023-12")

SHAREBONUS_VALID_STATUS = {"实施分配"}

NEW_COLS = [
    "n_resume_days",
    "n_max_days",
    "MAXraw_adj",
    "argmax_date_adj",
    "MAXraw_adj_ex_resume",
    "argmax_date_adj_ex_resume",
    "MAXraw_ex_resume",
    "MAXraw_ex_rights",
    "argmax_date_ex_rights",
    "delta_MAXraw_adj",
    "delta_MAXraw_adj_ex_resume",
    "delta_MAXraw_ex_resume",
    "delta_MAXraw_ex_rights",
    "n_adj_days",
    "n_adj_fallback_days",
    "adj_status",
    "adj_source",
    "corporate_action_flag",
    "ex_rights_flag",
    "ca_source",
    "n_ca_events",
    "argmax_raw_on_ex_date",
    "adj_max_changed_vs_raw",
    "resume_joint_max_changed",
    "adj_ex_resume_changed_vs_adj",
    "shadow_status_adj",
    "formula_id_adj_p2shadow",
    "p2shadow_version",
]

NEW_COL_KINDS = {
    "n_resume_days": "int",
    "n_max_days": "int",
    "MAXraw_adj": "float",
    "argmax_date_adj": "str",
    "MAXraw_adj_ex_resume": "float",
    "argmax_date_adj_ex_resume": "str",
    "MAXraw_ex_resume": "float",
    "MAXraw_ex_rights": "float",
    "argmax_date_ex_rights": "str",
    "delta_MAXraw_adj": "float",
    "delta_MAXraw_adj_ex_resume": "float",
    "delta_MAXraw_ex_resume": "float",
    "delta_MAXraw_ex_rights": "float",
    "n_adj_days": "int",
    "n_adj_fallback_days": "int",
    "adj_status": "str",
    "adj_source": "str",
    "corporate_action_flag": "int",
    "ex_rights_flag": "int",
    "ca_source": "str",
    "n_ca_events": "int",
    "argmax_raw_on_ex_date": "int",
    "adj_max_changed_vs_raw": "int",
    "resume_joint_max_changed": "int",
    "adj_ex_resume_changed_vs_adj": "int",
    "shadow_status_adj": "str",
    "formula_id_adj_p2shadow": "str",
    "p2shadow_version": "str",
}

def format_column(series: pd.Series, kind: str) -> list[str]:
    if kind == "int":
        return [fmt_int(v) for v in series.tolist()]
    if kind == "float":
        return [fmt_float(v) for v in series.tolist()]
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

def write_audit_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n", float_format="%.8f")
    os.replace(tmp, path)

def load_contracts(path: Path = CONTRACTS) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def anchor_snapshot(contracts: dict) -> dict:
    snap = {}
    for key, meta in contracts["m1_inputs"].items():
        p = ROOT / meta["file"]
        if not p.is_file():
            raise SystemExit(f"[P2.3] headline 锚缺失: {p}")
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
            raise SystemExit(f"[P2.3] 起始锚已不一致: {key}")
        if before[key]["sha256"] != after[key]["sha256"]:
            raise SystemExit(f"[P2.3] headline 锚在运行中变化: {key}")

def protected_paths(contracts: dict) -> set[Path]:
    return {(ROOT / m["file"]).resolve() for m in contracts["m1_inputs"].values()}

def code_column(code: str) -> str:
    """source_id(sh600000) -> 后复权 pkl 列名(600000.SH)。"""
    if code.startswith("sh"):
        return code[2:] + ".SH"
    if code.startswith("sz"):
        return code[2:] + ".SZ"
    if code.startswith("bj"):
        return code[2:] + ".BJ"
    return str(code)

def order_book_to_source(col: str) -> str:
    code6, suffix = str(col).split(".")[0], str(col).split(".")[-1].upper()
    return ("sh" if suffix == "SH" else "sz" if suffix == "SZ" else "bj") + code6

def parse_sharebonus_frame(sb: pd.DataFrame,
                           valid_status: set[str] | None = None) -> pd.DataFrame:
    """sharebonus -> 去重后的 (source_id, ex_date, 送转/派息字段)。

    只保留 `EX_DIVIDEND_DATE` 非空且 `ASSIGN_PROGRESS` 为实施类的事件；
    预披露/取消/否决不进入除权表。
    """
    valid_status = valid_status or SHAREBONUS_VALID_STATUS
    need = {"SECUCODE", "EX_DIVIDEND_DATE", "ASSIGN_PROGRESS",
            "BONUS_RATIO", "IT_RATIO", "PRETAX_BONUS_RMB"}
    missing = need - set(sb.columns)
    if missing:
        raise SystemExit(f"[P2.3] sharebonus 缺列: {sorted(missing)}")
    d = sb.copy()
    ex_raw = d["EX_DIVIDEND_DATE"].fillna("").astype(str).str.strip()
    status = d["ASSIGN_PROGRESS"].fillna("").astype(str).str.strip()
    mask = ex_raw.ne("") & status.isin(valid_status)
    v = d.loc[mask].copy()
    ex = ex_raw.loc[mask].str[:10]
    ok = ex.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    v = v.loc[ok].copy()
    ex = ex.loc[ok]
    sec = v["SECUCODE"].fillna("").astype(str).str.strip().str.upper()
    suffix = sec.str[-2:]
    code6 = sec.str[:6]
    ok2 = code6.str.match(r"^\d{6}$", na=False) & suffix.isin(["SH", "SZ", "BJ"])
    v = v.loc[ok2].copy()
    ex = ex.loc[ok2]
    sec = sec.loc[ok2]
    v["source_id"] = np.where(sec.str.endswith(".SH"), "sh",
                              np.where(sec.str.endswith(".SZ"), "sz", "bj")) + sec.str[:6]
    v["ex_date"] = ex.to_numpy()
    v["month"] = v["ex_date"].str[:7]
    v["bonus_ratio"] = pd.to_numeric(v["BONUS_RATIO"], errors="coerce").fillna(0.0)
    v["it_ratio"] = pd.to_numeric(v["IT_RATIO"], errors="coerce").fillna(0.0)
    v["dividend"] = pd.to_numeric(v["PRETAX_BONUS_RMB"], errors="coerce").fillna(0.0)
    v["event_source"] = "sharebonus"
    v["status"] = status.loc[v.index].to_numpy()
    v = v[["source_id", "ex_date", "month", "bonus_ratio", "it_ratio",
           "dividend", "event_source", "status"]]
    v = v.drop_duplicates(["source_id", "ex_date"], keep="first")
    return v.reset_index(drop=True)

def parse_corporate_action_frame(ca: pd.DataFrame,
                                 id_map: dict[str, str]) -> pd.DataFrame:
    """corporate_action_events -> (source_id, ex_date)；code 为 6 位无交易所。"""
    need = {"code", "ex_date"}
    missing = need - set(ca.columns)
    if missing:
        raise SystemExit(f"[P2.3] corporate_action 缺列: {sorted(missing)}")
    d = ca.copy()
    d["code6"] = d["code"].fillna("").astype(str).str.strip().str.zfill(6)
    d["ex_date"] = d["ex_date"].fillna("").astype(str).str[:10]
    ok = (d["code6"].str.match(r"^\d{6}$", na=False)
          & d["ex_date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False))
    d = d.loc[ok].copy()
    d["source_id"] = d["code6"].map(id_map)
    prefix = np.where(d["code6"].str.startswith("6"), "sh",
                      np.where(d["code6"].str.startswith(("0", "3")), "sz", "bj"))
    fallback = pd.Series(prefix, index=d.index) + d["code6"]
    d["source_id"] = d["source_id"].fillna(fallback)
    d["month"] = d["ex_date"].str[:7]
    d["event_source"] = "corporate_action"
    d["bonus_ratio"] = pd.to_numeric(d.get("bonus_ratio"), errors="coerce")
    d["it_ratio"] = pd.to_numeric(d.get("it_ratio"), errors="coerce")
    d["dividend"] = np.nan
    d["status"] = ""
    d = d[["source_id", "ex_date", "month", "bonus_ratio", "it_ratio",
           "dividend", "event_source", "status"]]
    d = d.drop_duplicates(["source_id", "ex_date"], keep="first")
    return d.reset_index(drop=True)

def combine_events(sb_ev: pd.DataFrame, ca_ev: pd.DataFrame):
    """合并事件，返回 (逐事件表, 逐月覆盖表)。"""
    ev = pd.concat([sb_ev, ca_ev], ignore_index=True, sort=False)
    ev["is_sharebonus"] = ev["event_source"].eq("sharebonus").astype("int8")
    ev["is_corporate_action"] = ev["event_source"].eq("corporate_action").astype("int8")
    ev = ev.groupby(["source_id", "ex_date"], as_index=False).agg(
        month=("month", "first"),
        bonus_ratio=("bonus_ratio", "max"),
        it_ratio=("it_ratio", "max"),
        dividend=("dividend", "max"),
        is_sharebonus=("is_sharebonus", "max"),
        is_corporate_action=("is_corporate_action", "max"),
    )
    month = ev.groupby(["source_id", "month"], as_index=False).agg(
        n_ca_events=("ex_date", "nunique"),
        has_sharebonus=("is_sharebonus", "max"),
        has_corporate_action=("is_corporate_action", "max"),
    )
    both = month["has_sharebonus"].eq(1) & month["has_corporate_action"].eq(1)
    sb_only = month["has_sharebonus"].eq(1) & month["has_corporate_action"].eq(0)
    ca_only = month["has_sharebonus"].eq(0) & month["has_corporate_action"].eq(1)
    month["ca_source"] = np.select(
        [both, sb_only, ca_only],
        ["both", "sharebonus", "corporate_action"], default="none")
    return ev, month

def load_resume_monthly(path: Path, source_ids: set[str]):
    """2.2 复牌集 -> (逐月复牌日数, 复牌行表)。"""
    ra = pd.read_csv(path, dtype=str, keep_default_na=False)
    need = {"source_id", "date", "month"}
    missing = need - set(ra.columns)
    if missing:
        raise SystemExit(f"[P2.3] resume audit 缺列: {sorted(missing)}")
    if ra.duplicated(["source_id", "date"]).any():
        raise SystemExit("[P2.3] resume audit 键 (source_id,date) 不唯一")
    bad = ~ra["source_id"].isin(source_ids)
    if bad.any():
        raise SystemExit(f"[P2.3] resume audit 含 source 面板外 code: "
                         f"{ra.loc[bad, 'source_id'].head(3).tolist()}")
    ra["month"] = ra["month"].astype(str).str[:7]
    monthly = ra.groupby(["source_id", "month"], as_index=False).size()
    monthly = monthly.rename(columns={"size": "n_resume_days"})
    return monthly, ra

def build_adj_return_matrix(px: pd.DataFrame) -> pd.DataFrame:
    """后复权价宽表 -> 日度收益。

    使用 `pct_change(fill_method=None)`；本数据的 pkl 在停牌期重复上一收盘价，
    复牌首日比值即包含整段停牌收益（与 headline `rraw` 同构，供 2.2 resume
    排除）。不在这里做 ffill，避免把上市前 NaN 误当可得价。
    """
    if not isinstance(px.index, pd.DatetimeIndex):
        px = px.copy()
        px.index = pd.to_datetime(px.index)
    if px.index.has_duplicates:
        raise SystemExit("[P2.3] 后复权 pkl 日期索引重复")
    return px.sort_index().pct_change(fill_method=None)

def monthly_adj_from_px(px: pd.DataFrame) -> pd.DataFrame:
    """后复权价宽表 -> 月频收益（月末 last 比值），用于与登记月频文件对拍。"""
    m = px.resample("ME").last()
    r = m.pct_change(fill_method=None)
    s = r.stack(future_stack=True).dropna().rename("R_calc").reset_index()
    s["month"] = s["date"].dt.strftime("%Y-%m")
    s["firm_id"] = s["order_book_id"].map(order_book_to_source)
    return s[["firm_id", "month", "R_calc"]]

def fallback_adj_return(rraw: float, prev_close_raw: float, bonus_ratio: float,
                        it_ratio: float, dividend: float) -> float:
    """sharebonus 回退：每 10 股送/转 + 每 10 股派息（税前）。

    `r_adj = (1+rraw) * (1 + (送转)/10) + (派息/10)/prev_close - 1`。
    只在日频后复权价缺失时使用；实际数据未触发（覆盖率 100%）。
    """
    if not all(isinstance(v, (int, float, np.floating)) and np.isfinite(v)
               for v in (rraw, prev_close_raw, bonus_ratio, it_ratio, dividend)):
        return float("nan")
    if prev_close_raw <= 0:
        return float("nan")
    share = 1.0 + (float(bonus_ratio) + float(it_ratio)) / 10.0
    return (1.0 + float(rraw)) * share + (float(dividend) / 10.0) / float(prev_close_raw) - 1.0

def aggregate_daily_year(d: pd.DataFrame, tie_rule: str) -> pd.DataFrame:
    """单年日频 -> 每 (code, month) 的 raw/adj/ex-resume/ex-rights MAX。"""
    keys = ["code", "month"]
    if d.empty:
        return pd.DataFrame(columns=keys + [
            "n_valid_days", "n_adj_days", "n_adj_fallback_days",
            "n_resume_days", "n_ex_rights_days", "MAXraw", "MAXraw_adj",
            "MAXraw_adj_ex_resume", "MAXraw_ex_resume", "MAXraw_ex_rights",
            "argmax_date_adj", "argmax_date_adj_ex_resume",
            "argmax_date_ex_rights"])
    d = d.sort_values(["code", "month", "date"], kind="mergesort").reset_index(drop=True)
    d["_r_adj_nr"] = d["r_adj"].where(d["resume_flag"] == 0)
    d["_rraw_nr"] = d["rraw"].where(d["resume_flag"] == 0)
    d["_rraw_nx"] = d["rraw"].where(d["ex_flag"] == 0)
    if tie_rule == "latest":
        # 反转后 idxmax 取组内最后一条（并列取最晚日期）。
        d = d.iloc[::-1].reset_index(drop=True)
    g = d.groupby(keys, sort=False)
    out = g.agg(
        n_valid_days=("rraw", "size"),
        n_adj_days=("r_adj", "count"),
        n_adj_fallback_days=("adj_fallback", "sum"),
        n_resume_days=("resume_flag", "sum"),
        n_ex_rights_days=("ex_flag", "sum"),
        MAXraw=("rraw", "max"),
        MAXraw_adj=("r_adj", "max"),
        MAXraw_adj_ex_resume=("_r_adj_nr", "max"),
        MAXraw_ex_resume=("_rraw_nr", "max"),
        MAXraw_ex_rights=("_rraw_nx", "max"),
    )
    for col, out_col in (("r_adj", "argmax_date_adj"),
                         ("_r_adj_nr", "argmax_date_adj_ex_resume"),
                         ("_rraw_nx", "argmax_date_ex_rights")):
        idx = g[col].idxmax()
        dates = pd.Series(index=idx.index, dtype=object)
        ok = idx.notna()
        if ok.any():
            dates.loc[ok] = d.loc[idx[ok].astype("int64"), "date"].to_numpy()
        out[out_col] = dates
    return out.reset_index()

def process_daily_year(path: Path, adj_ret: pd.DataFrame, adj_np: np.ndarray,
                       event_pairs: set, event_meta: dict, resume_pairs: set,
                       tie_rule: str):
    """读单年 daily_panel，算 r_adj，返月聚合 + 审计计数。"""
    d = pd.read_csv(path, usecols=["code", "date", "close_raw", "rraw"],
                    dtype={"code": str, "date": str,
                           "close_raw": "float64", "rraw": "float64"})
    n_rows_all = len(d)
    d = d[d["rraw"].notna()].copy()
    d["month"] = d["date"].str[:7]
    d["order_book_id"] = d["code"].map(code_column)

    ri = adj_ret.index.get_indexer(pd.to_datetime(d["date"]))
    ci = adj_ret.columns.get_indexer(d["order_book_id"])
    r_adj = np.full(len(d), np.nan, dtype="float64")
    ok = (ri >= 0) & (ci >= 0)
    r_adj[ok] = adj_np[ri[ok], ci[ok]]
    d["r_adj"] = r_adj
    d["adj_fallback"] = np.int8(0)

    d = d.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
    d["_prev_close_raw"] = d.groupby("code", sort=False)["close_raw"].shift(1)
    miss = d["r_adj"].isna()
    n_fallback = 0
    if miss.any():
        fb_idx = np.flatnonzero(miss.to_numpy())
        for i in fb_idx:
            code = str(d.at[i, "code"])
            date = str(d.at[i, "date"])
            meta = event_meta.get((code, date))
            if meta is None:
                continue
            b, it, div = meta
            prev = d.at[i, "_prev_close_raw"]
            v = fallback_adj_return(d.at[i, "rraw"], prev, b, it, div)
            if np.isfinite(v):
                d.at[i, "r_adj"] = v
                d.at[i, "adj_fallback"] = np.int8(1)
                n_fallback += 1

    pairs = list(zip(d["code"].tolist(), d["date"].tolist()))
    d["ex_flag"] = np.fromiter((1 if k in event_pairs else 0 for k in pairs),
                               dtype=np.int8, count=len(pairs))
    d["resume_flag"] = np.fromiter((1 if k in resume_pairs else 0 for k in pairs),
                                   dtype=np.int8, count=len(pairs))
    event_days = [k for k, flag in zip(pairs, d["ex_flag"].tolist()) if flag == 1]
    resume_days = [k for k, flag in zip(pairs, d["resume_flag"].tolist()) if flag == 1]

    agg = aggregate_daily_year(d, tie_rule)
    stats = {
        "year": int(str(d["date"].iloc[0])[:4]) if len(d) else None,
        "daily_rows_all": int(n_rows_all),
        "daily_rows_with_rraw": int(len(d)),
        "adj_lookup_missing": int(miss.sum()),
        "adj_fallback_days": int(n_fallback),
        "event_days": int(len(event_days)),
        "resume_days": int(len(resume_days)),
    }
    return agg, stats, event_days, resume_days

def compute_monthly_aggregates(panel_dir: Path, years: range, adj_ret: pd.DataFrame,
                               event_pairs: set, event_meta: dict,
                               resume_pairs: set, tie_rule: str):
    adj_np = adj_ret.to_numpy(dtype="float64")
    parts, stats, event_days_all, resume_days_all = [], [], set(), set()
    for y in years:
        p = panel_dir / f"{y}.csv"
        if not p.is_file():
            raise SystemExit(f"[P2.3] daily_panel 缺文件: {p}")
        agg, st, ev_days, res_days = process_daily_year(
            p, adj_ret, adj_np, event_pairs, event_meta, resume_pairs, tie_rule)
        parts.append(agg)
        stats.append(st)
        event_days_all.update(ev_days)
        resume_days_all.update(res_days)
        print(f"[P2.3] {y}: rows={st['daily_rows_with_rraw']:,} "
              f"adj_missing={st['adj_lookup_missing']:,} "
              f"fallback={st['adj_fallback_days']:,} "
              f"event_days={st['event_days']:,} resume_days={st['resume_days']:,}",
              flush=True)
    agg = pd.concat(parts, ignore_index=True, sort=False)
    return agg, stats, event_days_all, resume_days_all

def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _monthly_adj_status(n_adj: pd.Series, n_valid: pd.Series,
                        n_fallback: pd.Series) -> np.ndarray:
    return np.select(
        [
            (n_adj == n_valid) & (n_fallback == 0),
            (n_adj == n_valid) & (n_fallback > 0),
            (n_adj > 0) & (n_adj < n_valid),
            (n_adj == 0),
        ],
        ["ok_daily_adj", "ok_daily_adj_with_fallback",
         "partial_daily_adj", "no_daily_adj"],
        default="unknown",
    )

def _monthly_adj_source(n_adj: pd.Series, n_valid: pd.Series,
                        n_fallback: pd.Series) -> np.ndarray:
    return np.select(
        [
            (n_fallback == 0),
            (n_fallback > 0) & (n_adj == n_valid),
            (n_fallback > 0) & (n_adj < n_valid),
        ],
        ["daily_adj_close_post", "sharebonus_fallback",
         "mixed_daily_and_fallback"],
        default="none",
    )

def build_shadow_panel(source: pd.DataFrame, agg: pd.DataFrame,
                       ev_month: pd.DataFrame, resume_month: pd.DataFrame,
                       resume_pairs: set, tie_rule: str):
    old_cols = list(source.columns)
    collisions = [c for c in NEW_COLS if c in old_cols]
    if collisions:
        raise SystemExit(f"[P2.3] 输入已含 shadow 列，拒绝覆盖: {collisions}")

    # 避免与 source 的 MAXraw / n_valid_days / month 列冲突。
    agg = agg.rename(columns={
        "code": "_code",
        "n_valid_days": "n_valid_days_recomputed",
        "MAXraw": "MAXraw_recomputed",
        "n_resume_days": "n_resume_days_daily",
    })
    out = source.copy()
    out["_row"] = np.arange(len(out))
    out = out.merge(agg, left_on=["source_id", "month"],
                    right_on=["_code", "month"], how="left",
                    validate="one_to_one", indicator=True)
    if (out["_merge"] == "right_only").any():
        raise SystemExit("[P2.3] 日频聚合存在 source 面板外键")
    if int((out["_merge"] == "both").sum()) != len(agg):
        raise SystemExit("[P2.3] 日频覆盖行数与聚合行数不一致")
    out = out.sort_values("_row", kind="mergesort").reset_index(drop=True)
    out = out.drop(columns=["_code"])
    covered = out["_merge"].eq("both")

    ev_month = ev_month.rename(columns={"ca_source": "_ca_source"})
    out = out.merge(ev_month, on=["source_id", "month"], how="left",
                    validate="one_to_one")
    out = out.merge(resume_month, on=["source_id", "month"], how="left",
                    validate="one_to_one")

    n_valid = _num(out["n_valid_days"])
    max_raw = _num(out["MAXraw"])
    n_adj = out["n_adj_days"].where(covered)
    n_fb = out["n_adj_fallback_days"].where(covered)
    n_resume = out["n_resume_days"].where(covered).fillna(0)
    n_max = (n_valid - n_resume).where(covered)

    # 构造正确性：覆盖行必须与 headline 逐值对齐。
    mismatch_valid = int((covered & (
        _num(out["n_valid_days_recomputed"]) != n_valid)).sum())
    raw_mismatch = int((covered & (
        np.abs(_num(out["MAXraw_recomputed"]) - max_raw) > 1e-8)).sum())
    if mismatch_valid or raw_mismatch:
        raise SystemExit(f"[P2.3] headline 对齐失败: n_valid={mismatch_valid}, "
                         f"MAXraw={raw_mismatch}")

    # 与 2.2 复牌集交叉：日频重算的复牌日数必须与 2.2 审计表一致。
    resume_mismatch = int((covered & (
        _num(out["n_resume_days_daily"]).fillna(0) != n_resume)).sum())
    if resume_mismatch:
        raise SystemExit(f"[P2.3] 2.2 resume 计数与日频复算不一致: "
                         f"{resume_mismatch}")

    is_pre = out["month"].lt("2014-01")
    out["n_resume_days"] = n_resume
    out["n_max_days"] = n_max

    for col in ("MAXraw_adj", "argmax_date_adj", "MAXraw_adj_ex_resume",
                "argmax_date_adj_ex_resume", "MAXraw_ex_resume",
                "MAXraw_ex_rights", "argmax_date_ex_rights"):
        out.loc[~covered, col] = np.nan if not col.startswith("argmax") else ""

    out["delta_MAXraw_adj"] = _num(out["MAXraw_adj"]) - max_raw
    out["delta_MAXraw_adj_ex_resume"] = _num(out["MAXraw_adj_ex_resume"]) - max_raw
    out["delta_MAXraw_ex_resume"] = _num(out["MAXraw_ex_resume"]) - max_raw
    out["delta_MAXraw_ex_rights"] = _num(out["MAXraw_ex_rights"]) - max_raw

    out["n_adj_days"] = n_adj
    out["n_adj_fallback_days"] = n_fb
    status = np.where(is_pre, "pre_2014_no_daily_coverage", "unknown")
    status = np.where(covered, _monthly_adj_status(n_adj, n_valid, n_fb), status)
    source_status = np.where(covered, _monthly_adj_source(n_adj, n_valid, n_fb), "none")
    out["adj_status"] = status
    out["adj_source"] = source_status

    out["n_ca_events"] = out["n_ca_events"].fillna(0).astype("int64")
    out["corporate_action_flag"] = (out["n_ca_events"] > 0).astype("int64")
    out["ex_rights_flag"] = out["corporate_action_flag"]
    out["ca_source"] = out["_ca_source"].fillna("none")
    out = out.drop(columns=["_ca_source"])

    checks = {
        "n_valid_mismatch_vs_headline": mismatch_valid,
        "max_raw_mismatch_vs_headline": raw_mismatch,
        "resume_count_mismatch_vs_2_2": resume_mismatch,
    }
    return out, covered, n_valid, max_raw, checks

def finalize_panel(out: pd.DataFrame, covered: pd.Series, n_valid: pd.Series,
                   max_raw: pd.Series, ev: pd.DataFrame, resume_pairs: set,
                   tie_rule: str):
    """补齐 argmax 日事件/变化标记/状态/版本列；返回 (面板, 检查计数)。"""
    is_pre = out["month"].lt("2014-01")

    ex_pairs = set(zip(ev["source_id"], ev["ex_date"]))
    out["argmax_raw_on_ex_date"] = np.fromiter(
        (1 if (s, d) in ex_pairs else 0
         for s, d in zip(out["source_id"], out["argmax_date"])),
        dtype=np.int64, count=len(out))

    adj = _num(out["MAXraw_adj"])
    adj_nr = _num(out["MAXraw_adj_ex_resume"])
    raw_nr = _num(out["MAXraw_ex_resume"])
    raw_nx = _num(out["MAXraw_ex_rights"])

    out["adj_max_changed_vs_raw"] = (
        adj.notna() & max_raw.notna() & (np.abs(adj - max_raw) > 1e-8)).astype("int64")
    out["resume_joint_max_changed"] = (
        raw_nr.notna() & max_raw.notna() & (np.abs(raw_nr - max_raw) > 1e-8)).astype("int64")
    out["adj_ex_resume_changed_vs_adj"] = (
        adj_nr.notna() & adj.notna() & (np.abs(adj_nr - adj) > 1e-8)).astype("int64")

    out["shadow_status_adj"] = np.where(is_pre, "pre_2014_no_daily_coverage",
                                        np.where(covered, "evaluable", "unknown"))
    out["formula_id_adj_p2shadow"] = np.where(
        is_pre, FORMULA_PRE2014,
        np.where(covered, FORMULA_EVALUABLE, "MAXRAW_P2SHADOW_ADJ_UNDEFINED"))
    out["p2shadow_version"] = P2SHADOW_VERSION

    # 构造不变量
    violations = {}
    violations["adj_ex_resume_gt_adj"] = int(
        (adj_nr.notna() & adj.notna() & (adj_nr > adj + 1e-12)).sum())
    violations["ex_rights_gt_raw"] = int(
        (raw_nx.notna() & max_raw.notna() & (raw_nx > max_raw + 1e-12)).sum())
    violations["ex_resume_gt_raw"] = int(
        (raw_nr.notna() & max_raw.notna() & (raw_nr > max_raw + 1e-12)).sum())
    violations["argmax_adj_not_in_month"] = int(
        (out["argmax_date_adj"].ne("")
         & out["argmax_date_adj"].str[:7].ne(out["month"])).sum())
    sub = out.loc[out["argmax_date_adj_ex_resume"].ne(""),
                  ["source_id", "argmax_date_adj_ex_resume"]]
    violations["adj_argmax_on_resume_day"] = int(sum(
        1 for s, d in zip(sub["source_id"], sub["argmax_date_adj_ex_resume"])
        if (s, d) in resume_pairs))
    if any(violations.values()):
        raise SystemExit(f"[P2.3] 构造不变量失败: {violations}")
    return out, violations

def window_mask(df: pd.DataFrame, window: tuple[str, str]) -> pd.Series:
    return df["month"].ge(window[0]) & df["month"].le(window[1])

def tail_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n_nonnull": 0, "max": None, "min": None,
                "gt_0_44": 0, "gt_1_0": 0, "lt_neg_0_44": 0, "lt_neg_1_0": 0,
                "p99": None, "p01": None, "median": None}
    return {
        "n_nonnull": int(len(s)),
        "max": float(s.max()),
        "min": float(s.min()),
        "gt_0_44": int((s > 0.44).sum()),
        "gt_1_0": int((s > 1.0).sum()),
        "lt_neg_0_44": int((s < -0.44).sum()),
        "lt_neg_1_0": int((s < -1.0).sum()),
        "p99": float(s.quantile(0.99)),
        "p01": float(s.quantile(0.01)),
        "median": float(s.median()),
    }

def max_row(df: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return {}
    i = s.idxmax()
    row = df.loc[i]
    nv = row.get("n_valid_days", np.nan)
    nm = row.get("n_max_days", np.nan)
    return {
        "source_id": str(row["source_id"]),
        "month": str(row["month"]),
        "value": float(s.loc[i]),
        "argmax_date": str(row.get("argmax_date", "")),
        "argmax_date_adj": str(row.get("argmax_date_adj", "")),
        "n_valid_days": int(float(nv)) if pd.notna(nv) else None,
        "n_max_days": int(float(nm)) if pd.notna(nm) else None,
        "adj_status": str(row.get("adj_status", "")),
        "ca_source": str(row.get("ca_source", "")),
    }

def compare_monthly_raw_adj(raw_path: Path, adj_path: Path) -> dict:
    raw = pd.read_csv(raw_path, usecols=["firm_id", "month", "R"],
                      dtype={"firm_id": str, "month": str})
    adj = pd.read_csv(adj_path, usecols=["firm_id", "month", "R"],
                      dtype={"firm_id": str, "month": str})
    j = raw.merge(adj, on=["firm_id", "month"], suffixes=("_raw", "_adj"),
                  how="inner", validate="one_to_one")
    out = {"rows_common": int(len(j)), "rows_raw": int(len(raw)),
           "rows_adj": int(len(adj))}
    for label, window in (("calendar_2015_2023", WINDOW_CAL),
                          ("m1_window", WINDOW_M1)):
        w = j[j["month"].ge(window[0]) & j["month"].le(window[1])].copy()
        d = (w["R_adj"] - w["R_raw"]).abs()
        flip = ((w["R_raw"].abs() > 0.2)
                & (w["R_raw"] * w["R_adj"] < 0))
        out[label] = {
            "rows": int(len(w)),
            "abs_diff_gt_1e_6": int((d > 1e-6).sum()),
            "abs_diff_p99": float(d.quantile(0.99)) if len(d) else None,
            "abs_diff_max": float(d.max()) if len(d) else None,
            "sign_flips_abs_raw_gt_0_2": int(flip.sum()),
            "abs_raw_gt_0_44": int((w["R_raw"].abs() > 0.44).sum()),
            "abs_adj_gt_0_44": int((w["R_adj"].abs() > 0.44).sum()),
        }
    return out

def build_tail_diff(df: pd.DataFrame, n_top: int = 500) -> pd.DataFrame:
    """审计表：全部尾部行 + 各 delta 的 top N。"""
    def flag(series, thr):
        s = pd.to_numeric(series, errors="coerce")
        return s.gt(thr) | s.lt(-thr)

    mask = (flag(df["MAXraw"], 0.44) | flag(df["MAXraw_adj"], 0.44)
            | flag(df["MAXraw_adj_ex_resume"], 0.44)
            | flag(df["MAXraw_ex_resume"], 0.44)
            | flag(df["MAXraw_ex_rights"], 0.44))
    for col in ("delta_MAXraw_adj", "delta_MAXraw_adj_ex_resume",
                "delta_MAXraw_ex_resume", "delta_MAXraw_ex_rights"):
        v = pd.to_numeric(df[col], errors="coerce").abs()
        mask.loc[v.nlargest(n_top).index] = True
    sub = df.loc[mask].copy()
    cols = ["source_id", "month", "MAXraw", "argmax_date",
            "MAXraw_adj", "argmax_date_adj",
            "MAXraw_adj_ex_resume", "argmax_date_adj_ex_resume",
            "MAXraw_ex_resume", "MAXraw_ex_rights", "argmax_date_ex_rights",
            "delta_MAXraw_adj", "delta_MAXraw_adj_ex_resume",
            "delta_MAXraw_ex_resume", "delta_MAXraw_ex_rights",
            "n_resume_days", "n_max_days", "corporate_action_flag",
            "ca_source", "n_ca_events", "adj_status", "shadow_status_adj"]
    return sub[cols].sort_values(["month", "source_id"], kind="mergesort")

def build_exrights_audit(ev: pd.DataFrame, source: pd.DataFrame,
                         df: pd.DataFrame, event_days: set,
                         id_map: dict[str, str]) -> pd.DataFrame:
    """2014–2025 除权事件审计；含 2018+ corporate_action 交叉命中。"""
    e = ev[(ev["ex_date"] >= "2014-01-01") & (ev["ex_date"] <= "2025-12-31")].copy()
    e["in_corporate_action"] = e["is_corporate_action"].astype("int64")
    e["on_trading_day"] = [
        int((s, d) in event_days) for s, d in zip(e["source_id"], e["ex_date"])]
    src_keys = set(zip(source["source_id"], source["month"]))
    e["source_panel_match"] = [
        int((s, m) in src_keys) for s, m in zip(e["source_id"], e["month"])]

    panel_cols = ["source_id", "month", "MAXraw", "argmax_date",
                  "MAXraw_ex_rights", "argmax_date_ex_rights",
                  "delta_MAXraw_ex_rights", "n_ca_events", "ca_source"]
    j = e.merge(df[panel_cols], on=["source_id", "month"], how="left",
                validate="many_to_one")
    j["argmax_on_ex_date"] = (
        j["argmax_date"].fillna("").eq(j["ex_date"])).astype("int64")
    j["year"] = j["ex_date"].str[:4]
    cols = ["source_id", "ex_date", "month", "year", "bonus_ratio", "it_ratio",
            "dividend", "is_sharebonus", "is_corporate_action",
            "in_corporate_action", "on_trading_day", "source_panel_match",
            "argmax_on_ex_date", "MAXraw", "argmax_date", "MAXraw_ex_rights",
            "argmax_date_ex_rights", "delta_MAXraw_ex_rights",
            "n_ca_events", "ca_source"]
    return j[cols].sort_values(["ex_date", "source_id"], kind="mergesort")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


import argparse
import csv
import hashlib
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


# 只复用 Phase 1 已登记的 tie_rule 解析（不复制逻辑）。
from peermax_max_build import resolve_tie_rule  # noqa: E402




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
