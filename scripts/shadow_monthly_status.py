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
  python scripts/shadow_monthly_status.py"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from build_monthly_returns_status import trade_days_long  # noqa: E402


from build_monthly_returns_status import trade_days_long  # noqa: E402
from pathlib import Path
import numpy as np
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RETURNS = ROOT / "data/processed/monthly_returns_clean_v2.csv"

PANEL_DIR = ROOT / "data/processed/daily_panel"

SUSPEND_FLAG = ROOT / "data/raw/drive/stock_suspend_flag_20260630.pkl"

FIRM_ID_MAP = ROOT / "data/processed/firm_id_map.csv"

OUT = ROOT / "data/processed/phase2_shadow/monthly_returns_status_p2shadow.csv"

META_OUT = OUT.with_suffix(".meta.json")

ALIAS_OUT = ROOT / "artifacts/phase2/2.1_alias_map.json"

COUNTS_OUT = ROOT / "artifacts/phase2/2.1_counts.json"

CONTRACTS = ROOT / "configs/m1_contracts.yaml"

FM_SAMPLE = ROOT / "data/processed/m1_common_sample_ma_teacher_v2.parquet"

SOURCE_PANEL = ROOT / "data/processed/source_month_argmax_panel.csv"

P2SHADOW_VERSION = "p2shadow_v1"

START_YEAR = 2014

END_YEAR = 2025

MIN_TRADE_DAYS = 15

SENSITIVITY_TRADE_DAYS = 10

EXPECTED_ROWS = 856_771

FM_EXPECTED_ROWS = 181_178

FM_EXPECTED_MONTHS = 104

ALIAS_MAP = {"sh601313": "sh601360"}

NEW_COLS = [
    "n_price_obs",
    "n_trade_days",
    "trade_days_status",
    "trade_data_status",
    "complete_month_legacy",
    "complete_month_trade15",
    "complete_month_trade10",
    "pseudo_zero_flag",
    "partial_zero_flag",
    "true_zero_flag",
    "R_p2shadow",
    "p2shadow_version",
]

STATUS_NO_VOLUME = "no_volume_data"

STATUS_OUT_OF_COVERAGE = "out_of_daily_coverage"

STATUS_OK = "ok"

STATUS_NO_TRADE = "no_trade"

STATUS_DAILY_PANEL_MISSING = "daily_panel_missing"

EVALUABLE_STATUS = (STATUS_OK, STATUS_NO_TRADE)

def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)

def ym_add(month: str, k: int) -> str:
    """'YYYY-MM' + k 个月；只用于紧邻下一自然月，不引入前视。"""
    y, mo = (int(x) for x in str(month).split("-"))
    idx = y * 12 + (mo - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"

def load_contracts(path: Path = CONTRACTS) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def anchor_snapshot(contracts: dict) -> dict:
    """9 个 M1 主输入 sha 前后快照；sha 与登记不一致立即停机。"""
    snap = {}
    for key, meta in contracts["m1_inputs"].items():
        p = ROOT / meta["file"]
        if not p.is_file():
            raise SystemExit(f"[P2.1] headline 锚缺失: {p}")
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
            raise SystemExit(f"[P2.1] 起始锚已不一致: {key}")
        if before[key]["sha256"] != after[key]["sha256"]:
            raise SystemExit(f"[P2.1] headline 锚在运行中变化: {key}")

def protected_paths(contracts: dict) -> set[Path]:
    return {(ROOT / m["file"]).resolve() for m in contracts["m1_inputs"].values()}

def daily_panel_audit(panel_dir: Path, start_year: int = START_YEAR,
                      end_year: int = END_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2014–2025 每 (code, month) 的日频行数与市场交易日数（日期并集）。"""
    frames, calendars = [], []
    for y in range(start_year, end_year + 1):
        p = Path(panel_dir) / f"{y}.csv"
        if not p.is_file():
            continue
        d = pd.read_csv(p, usecols=["code", "date"],
                        dtype={"code": str, "date": str})
        d = d.dropna(subset=["code", "date"])
        if d.empty:
            continue
        d["month"] = d["date"].str[:7]
        frames.append(d.groupby(["code", "month"]).size()
                      .rename("n_daily_rows").reset_index())
        calendars.append(d[["month", "date"]].drop_duplicates())
    if not frames:
        return (pd.DataFrame(columns=["firm_id", "month", "n_daily_rows"]),
                pd.DataFrame(columns=["month", "market_trade_days"]))
    rows = (pd.concat(frames, ignore_index=True)
            .groupby(["code", "month"], as_index=False)["n_daily_rows"].sum()
            .rename(columns={"code": "firm_id"}))
    cal = (pd.concat(calendars, ignore_index=True)
           .groupby("month", as_index=False)["date"].nunique()
           .rename(columns={"date": "market_trade_days"}))
    return rows, cal

def _load_suspend_monthly(suspend_flag, start_year: int = START_YEAR,
                          end_year: int = END_YEAR) -> tuple[pd.DataFrame, dict]:
    """stock_suspend_flag -> (月度 True 天数, code6->列名)。"""
    if isinstance(suspend_flag, pd.DataFrame):
        flag = suspend_flag.copy()
    else:
        p = Path(suspend_flag)
        if not p.is_file():
            raise SystemExit(f"[P2.1] suspend flag 缺失: {p}")
        flag = pd.read_pickle(p)
    if not isinstance(flag.index, pd.DatetimeIndex):
        flag.index = pd.to_datetime(flag.index)
    flag = flag.loc[(flag.index >= f"{start_year}-01-01")
                    & (flag.index <= f"{end_year}-12-31")]
    if flag.empty:
        raise SystemExit("[P2.1] suspend flag 在覆盖期为空")
    for c in flag.columns:
        s = flag[c]
        if s.dtype == object:
            s = s.map({"True": True, "False": False,
                       True: True, False: False})
        flag[c] = s.where(s.notna(), False).astype(bool)
    monthly = flag.groupby(flag.index.to_period("M")).sum()
    monthly.index = [str(x) for x in monthly.index]
    return monthly, {c.split(".")[0]: c for c in flag.columns}

def _build_status_panel_full(ret: pd.DataFrame, panel_dir: Path,
                             start_year: int = START_YEAR,
                             end_year: int = END_YEAR,
                             min_trade_days: int = MIN_TRADE_DAYS,
                             sensitivity_days: int = SENSITIVITY_TRADE_DAYS,
                             suspend_flag=SUSPEND_FLAG,
                             alias_map: dict | None = None):
    """原月频表 + shadow 列 + 内部审计列；不覆盖原列。"""
    required = {"firm_id", "month", "R", "n_valid_days", "complete_month"}
    missing = required - set(ret.columns)
    if missing:
        raise SystemExit(f"[P2.1] 月频输入缺列: {sorted(missing)}")
    internal = ["_daily_code", "n_daily_rows", "market_trade_days",
                "_suspend_true_days"]
    collisions = [c for c in NEW_COLS + internal if c in ret.columns]
    if collisions:
        raise SystemExit(f"[P2.1] 月频输入已含 shadow/内部列，拒绝覆盖: {collisions}")

    alias = {str(k): str(v) for k, v in (alias_map or {}).items()}
    out = ret.copy()
    out["_daily_code"] = out["firm_id"].map(alias).fillna(out["firm_id"])

    td = trade_days_long(panel_dir)
    if td.duplicated(["firm_id", "month"]).any():
        raise SystemExit("[P2.1] trade_days_long 返回重复 (firm_id, month)")
    td = td.rename(columns={"firm_id": "_daily_code"})
    out = out.merge(td, on=["_daily_code", "month"], how="left",
                    validate="many_to_one")

    daily_rows, market = daily_panel_audit(panel_dir, start_year, end_year)
    daily_lookup = daily_rows.rename(columns={"firm_id": "_daily_code"})
    if daily_lookup.duplicated(["_daily_code", "month"]).any():
        raise SystemExit("[P2.1] daily_panel_audit 返回重复 (code, month)")
    out = out.merge(daily_lookup, on=["_daily_code", "month"], how="left",
                    validate="many_to_one")
    out = out.merge(market, on="month", how="left", validate="many_to_one")

    if len(out) != len(ret):
        raise SystemExit("[P2.1] merge 后行数变化")
    if out.duplicated(["firm_id", "month"]).any():
        raise SystemExit("[P2.1] 月频键 (firm_id, month) 不唯一")

    year = out["month"].str[:4].astype(int)
    in_scope = (year >= start_year) & (year <= end_year)
    pre_scope = year < start_year
    post_scope = year > end_year

    ntd_raw = pd.to_numeric(out["n_trade_days"], errors="coerce")
    nrows = pd.to_numeric(out["n_daily_rows"], errors="coerce")
    has_rows = nrows.fillna(0).gt(0)

    susp_monthly, colmap = _load_suspend_monthly(suspend_flag, start_year, end_year)
    needed_cols = [colmap[c] for c in out["_daily_code"].str[2:].dropna().unique()
                   if c in colmap]
    if needed_cols:
        susp_monthly = susp_monthly[needed_cols]
    susp_lookup = {(str(mo), col): float(v)
                   for (mo, col), v in susp_monthly.stack().items()}
    out["_suspend_true_days"] = [
        susp_lookup.get((str(mo), colmap.get(str(c6))), np.nan)
        for mo, c6 in zip(out["month"], out["_daily_code"].str[2:])
    ]

    td_zero_with_rows = has_rows & ntd_raw.fillna(0).eq(0)
    no_rows = ~has_rows
    susp_full = no_rows & out["_suspend_true_days"].eq(out["market_trade_days"])
    ok = in_scope & ntd_raw.gt(0)
    no_trade = in_scope & (td_zero_with_rows | susp_full)
    dpm = in_scope & no_rows & ~susp_full
    if not (ok | no_trade | dpm).loc[in_scope].all():
        raise SystemExit("[P2.1] 2014–2025 状态未完全覆盖")
    if ((ok & no_trade) | (ok & dpm) | (no_trade & dpm)).any():
        raise SystemExit("[P2.1] 状态互斥失败")

    status = pd.Series(pd.NA, index=out.index, dtype="object")
    status.loc[pre_scope] = STATUS_NO_VOLUME
    status.loc[post_scope] = STATUS_OUT_OF_COVERAGE
    status.loc[ok] = STATUS_OK
    status.loc[no_trade] = STATUS_NO_TRADE
    status.loc[dpm] = STATUS_DAILY_PANEL_MISSING
    out["trade_days_status"] = status
    # 任务口径列名别名（2.0_plan 用 trade_days_status；本任务文字用 trade_data_status）。
    out["trade_data_status"] = status

    out["n_price_obs"] = pd.to_numeric(out["n_valid_days"], errors="coerce").astype("int64")
    out["complete_month_legacy"] = pd.to_numeric(
        out["complete_month"], errors="coerce").astype("int64")

    evaluable = out["trade_days_status"].isin(EVALUABLE_STATUS)

    ntd_out = pd.Series(pd.NA, index=out.index, dtype="Int64")
    ntd_out.loc[evaluable] = ntd_raw.loc[evaluable].fillna(0).astype(int)
    out["n_trade_days"] = ntd_out

    def _complete_like(threshold: int) -> pd.Series:
        s = pd.Series(pd.NA, index=out.index, dtype="Int64")
        s.loc[evaluable] = (ntd_out.loc[evaluable] >= threshold).astype(int)
        return s

    out["complete_month_trade15"] = _complete_like(min_trade_days)
    out["complete_month_trade10"] = _complete_like(sensitivity_days)

    def _flag(mask: pd.Series) -> pd.Series:
        s = pd.Series(pd.NA, index=out.index, dtype="Int64")
        s.loc[evaluable] = mask.loc[evaluable].fillna(False).astype(int)
        return s

    r_zero = out["R"].eq(0)
    out["pseudo_zero_flag"] = _flag(r_zero & ntd_out.eq(0))
    out["partial_zero_flag"] = _flag(
        r_zero & ntd_out.gt(0) & ntd_out.lt(min_trade_days))
    out["true_zero_flag"] = _flag(r_zero & ntd_out.ge(min_trade_days))

    # 只有 no_trade（可证明无成交）才把 R 置 NA；daily_panel_missing 保留原 R。
    out["R_p2shadow"] = out["R"].mask(out["trade_days_status"].eq(STATUS_NO_TRADE))
    out["p2shadow_version"] = P2SHADOW_VERSION

    # 构造正确性：ok/no_trade 内 n_trade_days 不得大于价格观测数（Phase 1 锚）。
    viol = int((out.loc[evaluable, "n_trade_days"]
                > out.loc[evaluable, "n_price_obs"]).sum())
    if viol:
        raise SystemExit(f"[P2.1] n_trade_days > n_price_obs 行数 {viol} != 0")

    out = out.sort_values(["month", "firm_id"], kind="mergesort").reset_index(drop=True)
    return out, daily_rows

def build_status_panel(ret: pd.DataFrame, panel_dir: Path,
                       start_year: int = START_YEAR, end_year: int = END_YEAR,
                       min_trade_days: int = MIN_TRADE_DAYS,
                       sensitivity_days: int = SENSITIVITY_TRADE_DAYS,
                       suspend_flag=SUSPEND_FLAG,
                       alias_map: dict | None = None) -> pd.DataFrame:
    """公开构造入口：只返回旧列 + shadow 列（内部审计列已剔除）。"""
    full, _ = _build_status_panel_full(
        ret, panel_dir, start_year=start_year, end_year=end_year,
        min_trade_days=min_trade_days, sensitivity_days=sensitivity_days,
        suspend_flag=suspend_flag, alias_map=alias_map)
    return full[list(ret.columns) + NEW_COLS]

def _scope_counts(out: pd.DataFrame) -> dict:
    status = out["trade_days_status"]
    in_scope = status.isin([STATUS_OK, STATUS_NO_TRADE, STATUS_DAILY_PANEL_MISSING])
    ev = status.isin(EVALUABLE_STATUS)
    td = out["n_trade_days"]
    return {
        "rows_total": int(len(out)),
        "rows_in_scope_2014_2025": int(in_scope.sum()),
        "rows_evaluable_2014_2025": int(ev.sum()),
        "rows_daily_panel_missing": int(status.eq(STATUS_DAILY_PANEL_MISSING).sum()),
        "rows_no_volume_data_pre_2014": int(status.eq(STATUS_NO_VOLUME).sum()),
        "rows_out_of_daily_coverage_post_2025": int(
            status.eq(STATUS_OUT_OF_COVERAGE).sum()),
        "n_trade_days_zero": int(status.eq(STATUS_NO_TRADE).sum()),
        "n_trade_days_1_14": int((status.eq(STATUS_OK)
                                  & td.gt(0) & td.lt(MIN_TRADE_DAYS)).sum()),
        "n_trade_days_ge_15": int((status.eq(STATUS_OK)
                                   & td.ge(MIN_TRADE_DAYS)).sum()),
        "R_p2shadow_na_rows": int(out["R_p2shadow"].isna().sum()),
        "R_p2shadow_na_from_R_zero": int(
            (out["R_p2shadow"].isna() & out["R"].eq(0)).sum()),
        "R_p2shadow_na_from_R_nonzero": int(
            (out["R_p2shadow"].isna() & out["R"].notna() & out["R"].ne(0)).sum()),
        "R_null_rows": int(out["R"].isna().sum()),
        "n_trade_days_gt_n_price_obs": int(
            (out["n_trade_days"] > out["n_price_obs"]).sum()),
        "pseudo_zero_flag_sum": int(out["pseudo_zero_flag"].fillna(0).sum()),
        "partial_zero_flag_sum": int(out["partial_zero_flag"].fillna(0).sum()),
        "true_zero_flag_sum": int(out["true_zero_flag"].fillna(0).sum()),
    }

def _window_counts(out: pd.DataFrame) -> dict:
    w = out[(out["month"] >= "2015-05") & (out["month"] <= "2023-12")]
    legacy = w["complete_month_legacy"].eq(1)
    status = w["trade_days_status"]
    td = w["n_trade_days"]
    r0 = w["R"].eq(0)
    zero = status.eq(STATUS_NO_TRADE)
    partial = status.eq(STATUS_OK) & td.gt(0) & td.lt(MIN_TRADE_DAYS)
    true = status.eq(STATUS_OK) & td.ge(MIN_TRADE_DAYS)
    dpm = status.eq(STATUS_DAILY_PANEL_MISSING)
    return {
        "rows": int(len(w)),
        "complete_month_legacy": int(legacy.sum()),
        "complete_month_legacy_true": int(w["complete_month_legacy"].sum()),
        "rows_daily_panel_missing": int(dpm.sum()),
        "complete_legacy_and_trade_days_lt_15": int(
            (legacy & (zero | partial)).sum()),
        "complete_legacy_and_trade_days_zero": int((legacy & zero).sum()),
        "complete_legacy_and_trade_days_1_14": int((legacy & partial).sum()),
        "R_zero_and_complete_legacy": int((r0 & legacy).sum()),
        "R_zero_and_complete_legacy_trade_days_zero": int((r0 & legacy & zero).sum()),
        "R_zero_and_complete_legacy_trade_days_1_14": int(
            (r0 & legacy & partial).sum()),
        "R_zero_and_complete_legacy_trade_days_ge_min": int(
            (r0 & legacy & true).sum()),
        "R_zero_and_complete_legacy_daily_panel_missing": int(
            (r0 & legacy & dpm).sum()),
        "pseudo_zero_flag": int(w["pseudo_zero_flag"].fillna(0).sum()),
        "partial_zero_flag": int(w["partial_zero_flag"].fillna(0).sum()),
        "true_zero_flag": int(w["true_zero_flag"].fillna(0).sum()),
    }

def _classification_audit(full: pd.DataFrame, alias_map: dict | None = None) -> dict:
    """586/46/96 拆分 + suspend 验证；只读 full 的 2014–2025 可评估范围。"""
    alias_map = alias_map or {}
    in_scope = full["trade_days_status"].isin(
        [STATUS_OK, STATUS_NO_TRADE, STATUS_DAILY_PANEL_MISSING])
    d = full.loc[in_scope]
    status = d["trade_days_status"]
    no_trade = status.eq(STATUS_NO_TRADE)
    dpm = status.eq(STATUS_DAILY_PANEL_MISSING)
    alias_rows = d["firm_id"].isin(set(alias_map))
    alias_ok = alias_rows & status.eq(STATUS_OK)
    alias_no_trade = alias_rows & no_trade
    alias_dpm = alias_rows & dpm
    ntd = d["n_trade_days"]
    susp = d["_suspend_true_days"]
    mkt = d["market_trade_days"]

    no_trade_r0 = no_trade & d["R"].eq(0)
    no_trade_rnz = no_trade & d["R"].notna() & d["R"].ne(0)
    dpm_r0 = dpm & d["R"].eq(0)
    dpm_rnz = dpm & d["R"].notna() & d["R"].ne(0)
    alias_ok_rnz = alias_ok & d["R"].notna() & d["R"].ne(0)

    return {
        "alias_rows_2014_2025": int(alias_rows.sum()),
        "alias_recovered_rows": int(alias_ok.sum()),
        "alias_recovered_ge15": int((alias_ok & ntd.ge(MIN_TRADE_DAYS)).sum()),
        "alias_recovered_1_14": int((alias_ok & ntd.gt(0)
                                     & ntd.lt(MIN_TRADE_DAYS)).sum()),
        "alias_still_no_trade_rows": int(alias_no_trade.sum()),
        "alias_still_no_trade_months": sorted(
            d.loc[alias_no_trade, "month"].astype(str).tolist()),
        "alias_daily_panel_missing_rows": int(alias_dpm.sum()),
        "alias_recovered_nonzero_R": int(alias_ok_rnz.sum()),
        "no_trade_rows": int(no_trade.sum()),
        "no_trade_R_zero": int(no_trade_r0.sum()),
        "no_trade_R_nonzero": int(no_trade_rnz.sum()),
        "daily_panel_missing_rows": int(dpm.sum()),
        "daily_panel_missing_R_zero": int(dpm_r0.sum()),
        "daily_panel_missing_R_nonzero": int(dpm_rnz.sum()),
        "original_nonzero_R_daily_panel_missing_632": int(
            no_trade_rnz.sum() + alias_ok_rnz.sum()),
        "overlap_daily_panel_missing_with_632": int(
            (dpm & (no_trade_rnz | alias_ok_rnz)).sum()),
        "missing_rows_before_alias": int(no_trade.sum() + dpm.sum() + alias_ok.sum()),
        "missing_rows_after_alias": int(no_trade.sum() + dpm.sum()),
        "no_trade_with_daily_rows_volume_zero": int(
            (no_trade & d["n_daily_rows"].fillna(0).gt(0)).sum()),
        "no_trade_suspend_eq_market": int((no_trade & susp.eq(mkt)).sum()),
        "no_trade_suspend_lt_market": int((no_trade & susp.lt(mkt)).sum()),
        "dpm_suspend_gt0_lt_market": int(
            (dpm & susp.gt(0) & susp.lt(mkt)).sum()),
        "dpm_suspend_zero": int((dpm & susp.eq(0)).sum()),
        "dpm_suspend_missing": int((dpm & susp.isna()).sum()),
        "dpm_suspend_eq_market": int((dpm & susp.eq(mkt)).sum()),
        "note_96_overlap": ("按实际：96 个 daily_panel_missing 行全部 R=0，"
                            "与 632（586 no_trade 非零 R + 46 alias 恢复）无交集；"
                            "632 已被 586+46 完全解释。"),
    }

def _alias_payload(ret: pd.DataFrame, alias_map: dict, daily_rows: pd.DataFrame,
                   firm_id_map_path: Path, full: pd.DataFrame) -> dict:
    firm_ids = set()
    if Path(firm_id_map_path).is_file():
        fim = pd.read_csv(firm_id_map_path, dtype=str)
        if "firm_id" in fim.columns:
            firm_ids = set(fim["firm_id"].dropna())
    pairs = []
    for src, dst in alias_map.items():
        a = ret.loc[ret["firm_id"].eq(src), ["month", "R"]].rename(
            columns={"R": "R_alias"})
        b = ret.loc[ret["firm_id"].eq(dst), ["month", "R"]].rename(
            columns={"R": "R_target"})
        ov = a.merge(b, on="month", how="inner")
        same = np.isclose(ov["R_alias"], ov["R_target"], rtol=0, atol=1e-12,
                          equal_nan=True)
        diff = ov.loc[~same]
        sub = full.loc[full["firm_id"].eq(src)
                       & full["trade_days_status"].isin(
                           [STATUS_OK, STATUS_NO_TRADE, STATUS_DAILY_PANEL_MISSING])]
        pairs.append({
            "alias": src,
            "target": dst,
            "monthly_overlap_months": int(len(ov)),
            "monthly_identical_months": int(same.sum()),
            "monthly_non_identical_months": int((~same).sum()),
            "monthly_non_identical_detail": [
                {"month": str(r.month),
                 "R_alias": float(r.R_alias) if pd.notna(r.R_alias) else None,
                 "R_target": float(r.R_target) if pd.notna(r.R_target) else None}
                for r in diff.itertuples(index=False)
            ],
            "daily_panel_source_has_rows": bool(
                daily_rows["firm_id"].eq(src).any() if len(daily_rows) else False),
            "daily_panel_target_has_rows": bool(
                daily_rows["firm_id"].eq(dst).any() if len(daily_rows) else False),
            "firm_id_map_source_present": bool(src in firm_ids),
            "firm_id_map_target_present": bool(dst in firm_ids),
            "rows_2014_2025": int(len(sub)),
            "recovered_rows": int(sub["trade_days_status"].eq(STATUS_OK).sum()),
            "recovered_ge15": int((sub["trade_days_status"].eq(STATUS_OK)
                                   & sub["n_trade_days"].ge(MIN_TRADE_DAYS)).sum()),
            "recovered_1_14": int((sub["trade_days_status"].eq(STATUS_OK)
                                   & sub["n_trade_days"].gt(0)
                                   & sub["n_trade_days"].lt(MIN_TRADE_DAYS)).sum()),
            "still_no_trade_months": sorted(
                sub.loc[sub["trade_days_status"].eq(STATUS_NO_TRADE), "month"]
                .astype(str).tolist()),
            "daily_panel_missing_rows": int(
                sub["trade_days_status"].eq(STATUS_DAILY_PANEL_MISSING).sum()),
        })
    return {
        "p2shadow_version": P2SHADOW_VERSION,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "alias": dict(alias_map),
        "pairs": pairs,
        "basis": [
            "daily_panel 2014–2025 无 sh601313 行；sh601360 有完整日频/成交记录。",
            "monthly_returns_clean_v2 中 sh601313 与 sh601360 重叠 73 个月，"
            "72 个月 R 相同；仅 2018-02 不同（0.233710 vs 0.110417）。",
            "firm_id_map 只有 sh601360，无 sh601313。",
            "因此 sh601313 的日频/成交查询 alias 到 sh601360；输出 firm_id/R 不变。",
        ],
        "note": "alias 只影响 n_trade_days/状态；不重算 R、不合并行、不改 headline。",
    }

def _fm_counts(out: pd.DataFrame, sample_path: Path, source_path: Path) -> dict:
    """FM 零值拆分 + 15 日 outcome/MAX 门损失；只读 headline 样本/面板。"""
    if not sample_path.is_file():
        return {"available": False, "reason": f"missing: {sample_path}"}
    fm = pd.read_parquet(sample_path)
    need = {"focal_id", "month", "m1_in_fm_sample", "R_next"}
    if not need.issubset(fm.columns):
        return {"available": False,
                "reason": f"missing columns: {sorted(need - set(fm.columns))}"}
    fm = fm.loc[fm["m1_in_fm_sample"].fillna(False).astype(bool),
                ["focal_id", "month", "R_next"]].copy()
    fm["next_month"] = fm["month"].map(lambda m: ym_add(m, 1))

    nxt = out[["firm_id", "month", "n_trade_days", "trade_days_status"]].rename(
        columns={"firm_id": "focal_id", "month": "next_month",
                 "n_trade_days": "next_n_trade_days",
                 "trade_days_status": "next_status"})
    if nxt.duplicated(["focal_id", "next_month"]).any():
        raise SystemExit("[P2.1] shadow 面板键重复，无法映射下月")
    fm = fm.merge(nxt, on=["focal_id", "next_month"], how="left",
                  validate="many_to_one")

    zero = fm["R_next"].eq(0)
    ntd_next = fm["next_n_trade_days"]
    nxt_status = fm["next_status"]
    pseudo = zero & nxt_status.eq(STATUS_NO_TRADE)
    partial = zero & nxt_status.eq(STATUS_OK) & ntd_next.gt(0) & ntd_next.lt(MIN_TRADE_DAYS)
    true = zero & nxt_status.eq(STATUS_OK) & ntd_next.ge(MIN_TRADE_DAYS)
    missing = zero & ~nxt_status.isin(EVALUABLE_STATUS)

    # outcome 门：下月不满足 >=15 交易日，或状态不可评估（daily_panel_missing）→ 无法定义 R_next。
    outcome_loss = ~nxt_status.isin(EVALUABLE_STATUS) | ntd_next.lt(MIN_TRADE_DAYS)

    src = pd.read_csv(source_path, usecols=["source_id", "month", "n_valid_days"],
                      dtype={"source_id": str, "month": str}).rename(
        columns={"source_id": "focal_id", "n_valid_days": "src_n_valid_days"})
    if src.duplicated(["focal_id", "month"]).any():
        raise SystemExit("[P2.1] source_month_argmax_panel 键重复")
    fm = fm.merge(src, on=["focal_id", "month"], how="left", validate="many_to_one")
    max_loss = fm["src_n_valid_days"].isna() | fm["src_n_valid_days"].lt(MIN_TRADE_DAYS)

    union = outcome_loss | max_loss
    n = int(len(fm))
    return {
        "available": True,
        "sample": str(sample_path.relative_to(ROOT)),
        "rows": n,
        "months": int(fm["month"].nunique()),
        "month_min": str(fm["month"].min()),
        "month_max": str(fm["month"].max()),
        "next_status_counts": {str(k): int(v) for k, v in
                               nxt_status.value_counts(dropna=False).items()},
        "R_next_zero": int(zero.sum()),
        "R_next_zero_pseudo_full_month_no_trade": int(pseudo.sum()),
        "R_next_zero_partial_1_14_trade_days": int(partial.sum()),
        "R_next_zero_true_ge_15_trade_days": int(true.sum()),
        "R_next_zero_daily_panel_missing": int(missing.sum()),
        "R_next_zero_missing_next_trade_days": int(
            (zero & ntd_next.isna()).sum()),
        "missing_next_trade_days_all": int(ntd_next.isna().sum()),
        "missing_next_status_all": int(nxt_status.isna().sum()),
        "missing_source_n_valid_days": int(fm["src_n_valid_days"].isna().sum()),
        "outcome_gate_15_loss_rows": int(outcome_loss.sum()),
        "outcome_gate_15_loss_rows_R_next_zero": int((outcome_loss & zero).sum()),
        "outcome_gate_15_loss_rows_R_next_nonzero": int((outcome_loss & ~zero).sum()),
        "max_gate_15_loss_rows": int(max_loss.sum()),
        "gate_overlap_rows": int((outcome_loss & max_loss).sum()),
        "gate_union_loss_rows": int(union.sum()),
        "fm_upper_bound_after_gates": int(n - union.sum()),
        "fm_loss_pct_union": round(float(union.sum()) / n * 100.0, 4) if n else None,
        "expected_from_2_0_plan": {
            "rows": FM_EXPECTED_ROWS,
            "months": FM_EXPECTED_MONTHS,
            "R_next_zero": 2016,
            "pseudo": 1118,
            "partial": 4,
            "true": 894,
            "daily_panel_missing": 0,
            "outcome_gate_loss": 3547,
            "max_gate_loss": 3352,
            "overlap": 1046,
            "union": 5853,
            "fm_upper_bound": 175325,
        },
        "matches_2_0_plan": {
            "rows": n == FM_EXPECTED_ROWS,
            "months": int(fm["month"].nunique()) == FM_EXPECTED_MONTHS,
            "R_next_zero": int(zero.sum()) == 2016,
            "zero_split": (int(pseudo.sum()), int(partial.sum()),
                           int(true.sum()), int(missing.sum())) == (1118, 4, 894, 0),
            "outcome_gate_loss": int(outcome_loss.sum()) == 3547,
            "max_gate_loss": int(max_loss.sum()) == 3352,
            "gate_overlap": int((outcome_loss & max_loss).sum()) == 1046,
            "union": int(union.sum()) == 5853,
            "fm_upper_bound": int(n - union.sum()) == 175325,
        },
    }

def _fm_event_month_counts(out: pd.DataFrame, sample_path: Path) -> dict:
    """事件月状态分布（供 2.6 解释变量可用性）。"""
    if not sample_path.is_file():
        return {"available": False}
    fm = pd.read_parquet(sample_path, columns=["focal_id", "month", "m1_in_fm_sample"])
    fm = fm.loc[fm["m1_in_fm_sample"].fillna(False).astype(bool), ["focal_id", "month"]]
    key = out[["firm_id", "month", "trade_days_status", "n_trade_days"]].rename(
        columns={"firm_id": "focal_id"})
    j = fm.merge(key, on=["focal_id", "month"], how="left", validate="many_to_one")
    status = j["trade_days_status"]
    td = j["n_trade_days"]
    return {
        "available": True,
        "fm_rows": int(len(j)),
        "event_month_status_counts": {str(k): int(v) for k, v in
                                      status.value_counts(dropna=False).items()},
        "event_month_n_trade_days_zero": int(status.eq(STATUS_NO_TRADE).sum()),
        "event_month_n_trade_days_1_14": int(
            (status.eq(STATUS_OK) & td.gt(0) & td.lt(MIN_TRADE_DAYS)).sum()),
        "event_month_daily_panel_missing": int(
            status.eq(STATUS_DAILY_PANEL_MISSING).sum()),
        "event_month_missing_trade_days": int(td.isna().sum()),
    }

def _source_bridge(out: pd.DataFrame, source_path: Path) -> dict:
    """2.1→2.2 口径桥：窗口内 n_trade_days vs source 面板 n_valid_days。"""
    if not source_path.is_file():
        return {"available": False, "reason": f"missing: {source_path}"}
    src = pd.read_csv(source_path, usecols=["source_id", "month", "n_valid_days"],
                      dtype={"source_id": str, "month": str}).rename(
        columns={"source_id": "firm_id", "n_valid_days": "source_n_valid_days"})
    if src.duplicated(["firm_id", "month"]).any():
        raise SystemExit("[P2.1] source_month_argmax_panel 键重复")
    w = out[(out["month"] >= "2015-05") & (out["month"] <= "2023-12")][
        ["firm_id", "month", "n_trade_days", "R"]]
    j = w.merge(src, on=["firm_id", "month"], how="inner", validate="one_to_one")
    diff_mask = j["n_trade_days"].ne(j["source_n_valid_days"])
    diff = j.loc[diff_mask].sort_values(["month", "firm_id"])
    return {
        "available": True,
        "matched_rows": int(len(j)),
        "equal_rows": int((~diff_mask).sum()),
        "diff_rows": int(diff_mask.sum()),
        "diff_detail": [
            {
                "firm_id": str(r.firm_id),
                "month": str(r.month),
                "n_trade_days": int(r.n_trade_days),
                "source_n_valid_days": int(r.source_n_valid_days),
                "R": float(r.R) if pd.notna(r.R) else None,
            }
            for r in diff.itertuples(index=False)
        ],
        "expected_from_2_0_plan": {"matched_rows": 389_453, "equal_rows": 389_452,
                                   "diff_rows": 1},
        "matches_2_0_plan": bool(len(j) == 389_453
                                and int((~diff_mask).sum()) == 389_452
                                and int(diff_mask.sum()) == 1),
    }

def _by_year(out: pd.DataFrame) -> list[dict]:
    d = out.assign(_year=out["month"].str[:4])
    rows = []
    for year, g in d.groupby("_year", sort=True):
        status = g["trade_days_status"]
        td = g["n_trade_days"]
        legacy = g["complete_month_legacy"].eq(1)
        r0 = g["R"].eq(0)
        zero = status.eq(STATUS_NO_TRADE)
        partial = status.eq(STATUS_OK) & td.gt(0) & td.lt(MIN_TRADE_DAYS)
        rows.append({
            "year": int(year),
            "rows": int(len(g)),
            "evaluable_rows": int(status.isin(EVALUABLE_STATUS).sum()),
            "daily_panel_missing": int(status.eq(STATUS_DAILY_PANEL_MISSING).sum()),
            "n_trade_days_na": int(td.isna().sum()),
            "n_trade_days_zero": int(zero.sum()),
            "n_trade_days_1_14": int(partial.sum()),
            "n_trade_days_ge_15": int((status.eq(STATUS_OK)
                                       & td.ge(MIN_TRADE_DAYS)).sum()),
            "complete_legacy_and_trade_days_lt_15": int(
                (legacy & (zero | partial)).sum()),
            "R_zero_and_complete_legacy": int((r0 & legacy).sum()),
            "pseudo_zero_flag": int(g["pseudo_zero_flag"].fillna(0).sum()),
            "partial_zero_flag": int(g["partial_zero_flag"].fillna(0).sum()),
            "true_zero_flag": int(g["true_zero_flag"].fillna(0).sum()),
            "R_p2shadow_na": int(g["R_p2shadow"].isna().sum()),
        })
    return rows


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
