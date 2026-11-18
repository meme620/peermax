"""p2_3_build_adj_max_shadow 的辅助函数与常量（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.fmt import fmt_float
from lib.fmt import fmt_int
from lib.io import sha256_file
from pathlib import Path
import csv
import math
import numpy as np
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

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

