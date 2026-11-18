"""D3：月度收益质量列补充（2026-09-13）。

背景（审计 E4）：`data/processed/monthly_returns_clean.csv` 仅 3 列
（firm_id, month, R），缺 plan §7.4 与 §5.6 Step 3 要求的质量字段：
`n_valid_days` / `complete_month` / `return_status` / `month_start` / `month_end`。

**关键约束**：主文件被 S3A spec 钉住 input hash（8715be47…），直接改写会破坏
S3A 全链锚点。故本脚本**只读主文件、输出扩展版新文件**，主文件保持逐位不变。

阈值（已在 execution_protocol_b0_5_to_s6a.md 冻结为 pilot_frozen）：
  min_valid_days = 15（月均 20.3 交易日，75%；Jansen 2021 / Hou 口径）
  return_status: n_valid_days>=15 → PASS；5..14 → CONDITIONAL；<5 → BLOCKED

Phase 1 / I-1（2026-10-26）：
  - 新增 `n_price_obs`：Drive 日频价矩阵非空格数（前向填充会把停牌日也计为“有价”）。
  - 新增 `n_trade_days`：按 `data/processed/daily_panel/{year}.csv` 的
    `(code, month)` 统计 `volume > 0` 的天数（真实交易日；覆盖期外为 0）。
  - 旧列顺序/旧列值不变，新列只追加在末尾。

P1-01（2026-11-15）：
  - `n_valid_days` 语义改为**有效交易日**（`volume > 0` 天数），与 methodology
    plan L4830–4837 / L4507 一致；`complete_month` 与 `return_status` 同步改用
    `n_trade_days` 判定。
  - 旧“价格非空日数”保留为 `n_price_obs` / `n_price_obs_legacy` 审计列，
    不再冒充 `n_valid_days`；`n_valid_days_semantics='trade_days_v1'`。
  - 交付 CSV 保持 10 列（`OLD_COLS`）；审计列写入
    `reports/data_gates/G0/monthly_returns_status_report.json`。

Phase 1 / I-17（2026-10-26）：
  - 删除手写常量 `SAMPLE_VERSION="mainline_2018_2025_fullperiod"`；写出前按
    `min(month)..max(month)` 计算 `sample_version`（当前数据应为 2000-02..2026-06）。
  - 新增 `sample_version_source='computed_from_data'` 与 `sample_version_legacy`
    （旧标签 mainline_2018_2025_fullperiod，仅追溯；不再参与 sample_version）。
  - 孪生文件登记：`monthly_returns_clean.csv` 与
    `monthly_returns_full_2006_2025.csv` 字节完全相同（sha256 8715be47…、
    md5 a7c050ba…）；后者 `alias_of=monthly_returns_clean.csv`。Phase 1 不删除、
    不改名这两个文件。

产出：data/processed/monthly_returns_clean_v2.csv（10 列，`n_valid_days` = 有效交易日）
  firm_id, month, R, n_valid_days, complete_month, return_status,
  month_start, month_end, sample_version, available_at
审计报告：reports/data_gates/G0/monthly_returns_status_report.json
  （n_price_obs / n_trade_days / n_price_obs_legacy / 一致性计数）

用法：python scripts/build_monthly_returns_status.py
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = ROOT / "data/processed/monthly_returns_clean.csv"
DRIVE_PX = ROOT / "data/raw/drive/daily_close_none_20260630.pkl"
PANEL_DIR = ROOT / "data/processed/daily_panel"
OUT = ROOT / "data/processed/monthly_returns_clean_v2.csv"
REPORT = ROOT / "reports/data_gates/G0/monthly_returns_status_report.json"

MIN_VALID_DAYS = 15        # registry 待登记；来源 protocol pilot_frozen
SAMPLE_VERSION_RULE = "min(month)..max(month)"
SAMPLE_VERSION_SOURCE = "computed_from_data"
LEGACY_SAMPLE_VERSION = "mainline_2018_2025_fullperiod"  # I-17 旧标签，仅写入 legacy 列

OLD_COLS = ["firm_id", "month", "R", "n_valid_days", "complete_month",
            "return_status", "month_start", "month_end", "sample_version",
            "available_at"]
NEW_COLS = ["n_price_obs", "n_trade_days", "n_price_obs_legacy", "n_valid_days_semantics"]
SAMPLE_VERSION_COLS = ["sample_version_source", "sample_version_legacy"]


def compute_sample_version(months) -> str:
    """按数据首末月计算 sample_version；空输入拒绝写出空标签。"""
    s = pd.Series(months).dropna().astype(str)
    if s.empty:
        raise SystemExit("[SAMPLE-VERSION] month 为空，拒绝写出空 sample_version")
    return f"{s.min()}..{s.max()}"


def annotate_sample_version(out: pd.DataFrame) -> pd.DataFrame:
    """写出前按 out['month'] 的 min/max 标注 sample_version 与追溯列（不改其他列）。"""
    out = out.copy()
    out["sample_version"] = compute_sample_version(out["month"])
    out["sample_version_source"] = SAMPLE_VERSION_SOURCE
    out["sample_version_legacy"] = LEGACY_SAMPLE_VERSION
    return out


def fid(c: str) -> str:
    """Drive 列名 '000001.SZ' → daily_panel/monthly firm_id 'sz000001'。"""
    code, exch = c.split(".")
    return exch.lower() + code


def _to_long(df: pd.DataFrame, name: str) -> pd.DataFrame:
    s = df.stack(future_stack=True).rename(name).reset_index()
    # reset_index 后列为 [date, order_book_id, <name>]；统一改名为 month/col/value
    s.columns = ["month", "col", name]
    return s


def price_obs_long(px: pd.DataFrame) -> pd.DataFrame:
    """Drive 价矩阵 → (firm_id, month) 的价格观测数（非空格数）与月首/月末。

    语义与旧 `n_valid_days` 完全一致：前向填充的停牌日也算“有价”。
    """
    cnt = px.notna().resample("ME").sum()
    cnt.index = [d.strftime("%Y-%m") for d in cnt.index]
    first = px.apply(lambda c: c.dropna().resample("ME").apply(lambda s: s.index[0] if len(s) else np.nan))
    last = px.apply(lambda c: c.dropna().resample("ME").apply(lambda s: s.index[-1] if len(s) else np.nan))
    first.index = [d.strftime("%Y-%m") for d in first.index]
    last.index = [d.strftime("%Y-%m") for d in last.index]

    cnt_l = _to_long(cnt, "n_price_obs")
    first_l = _to_long(first, "month_start")
    last_l = _to_long(last, "month_end")
    for d in (cnt_l, first_l, last_l):
        d["firm_id"] = d["col"].map(fid)

    q = cnt_l[["firm_id", "month", "n_price_obs"]]
    q = q.merge(first_l[["firm_id", "month", "month_start"]], on=["firm_id", "month"], how="left")
    q = q.merge(last_l[["firm_id", "month", "month_end"]], on=["firm_id", "month"], how="left")
    q["month_start"] = pd.to_datetime(q["month_start"]).dt.strftime("%Y-%m-%d")
    q["month_end"] = pd.to_datetime(q["month_end"]).dt.strftime("%Y-%m-%d")
    return q


def trade_days_long(panel_dir=PANEL_DIR) -> pd.DataFrame:
    """daily_panel 年度文件 → (firm_id, month) 的真实交易日数（volume > 0）。

    只统计日频面板中有记录且 `volume > 0` 的日期；停牌日通常无行，因此该数
    小于或等于价格观测数（前向填充口径）。覆盖期外无记录 → 0（由调用方 fillna）。
    """
    frames = []
    for yf in sorted(Path(panel_dir).glob("*.csv")):
        d = pd.read_csv(yf, usecols=["code", "date", "volume"],
                        dtype={"code": str, "date": str})
        d = d[d["volume"].fillna(0) > 0]
        if d.empty:
            continue
        d = d.assign(month=d["date"].str[:7])
        g = (d.groupby(["code", "month"]).size()
             .rename("n_trade_days").reset_index()
             .rename(columns={"code": "firm_id"}))
        frames.append(g)
    if not frames:
        return pd.DataFrame(columns=["firm_id", "month", "n_trade_days"])
    return (pd.concat(frames, ignore_index=True)
            .groupby(["firm_id", "month"], as_index=False)["n_trade_days"].sum())


def build_status(ship: pd.DataFrame, px: pd.DataFrame, panel_dir=PANEL_DIR,
                 min_valid_days: int = MIN_VALID_DAYS) -> pd.DataFrame:
    """构造月度质量表；n_valid_days/complete_month/return_status 按有效交易日判定。

    P1-01：`n_valid_days = n_trade_days`（`volume>0` 天数），旧价格非空日数仅作
    `n_price_obs` / `n_price_obs_legacy` 审计列保留。
    """
    q = price_obs_long(px)
    out = ship.merge(q, on=["firm_id", "month"], how="left")
    out["n_price_obs"] = out["n_price_obs"].fillna(0).astype(int)
    out["available_at"] = out["month_end"]        # 月末即可得（保守）

    td = trade_days_long(panel_dir)
    out = out.merge(td, on=["firm_id", "month"], how="left")
    out["n_trade_days"] = pd.to_numeric(out["n_trade_days"], errors="coerce").fillna(0).astype(int)

    # P1-01：有效交易日口径；旧价格观测数只保留审计
    out["n_price_obs_legacy"] = out["n_price_obs"]
    out["n_valid_days"] = out["n_trade_days"]
    out["complete_month"] = (out["n_trade_days"] >= min_valid_days).astype(int)
    out["return_status"] = np.select(
        [out["n_trade_days"] >= min_valid_days, out["n_trade_days"] >= 5],
        ["PASS", "CONDITIONAL"], default="BLOCKED")
    out["n_valid_days_semantics"] = "trade_days_v1"

    out = out.sort_values(["month", "firm_id"]).reset_index(drop=True)
    out["sample_version"] = compute_sample_version(out["month"])
    out = out[OLD_COLS + NEW_COLS]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()

    # 1) 仅读主文件（不写）
    ship = pd.read_csv(SHIPPED, dtype={"firm_id": str, "month": str})

    # 2) 从 Drive 日频价还原每个 (firm, month) 的价格观测数与月首/月末
    px = pickle_load(DRIVE_PX)
    out = build_status(ship, px, PANEL_DIR)
    # 写出前补齐 I-17 追溯列（sample_version 已在 build_status 内按 min/max 计算）
    out = annotate_sample_version(out)

    # P1-01：交付 CSV 保持 10 列；审计列只进报告
    csv_out = out[OLD_COLS]
    Path(args.out).write_text("", encoding="utf-8")     # 确保可写
    csv_out.to_csv(args.out, index=False)

    with open(args.out, "rb") as f:
        csv_sha = hashlib.sha256(f.read()).hexdigest()

    cm = csv_out["complete_month"].astype(int)
    ntd = out["n_trade_days"].astype(int)
    npo = out["n_price_obs"].astype(int)
    report = {
        "report_id": "monthly_returns_status_report",
        "run_at": pd.Timestamp.utcnow().isoformat(),
        "script": "scripts/build_monthly_returns_status.py",
        "purpose": "P1-01: n_valid_days/complete_month 按有效交易日（volume>0）判定",
        "csv_path": str(Path(args.out).relative_to(ROOT)),
        "csv_sha256": csv_sha,
        "rows": int(len(csv_out)),
        "sample_version": csv_out["sample_version"].iat[0],
        "sample_version_rule": SAMPLE_VERSION_RULE,
        "columns_delivered": OLD_COLS,
        "audit_columns_in_memory_only": NEW_COLS,
        "min_valid_days": MIN_VALID_DAYS,
        "n_valid_days_semantics": "trade_days_v1",
        "n_valid_days_rule": "n_valid_days = n_trade_days = count(daily_panel volume>0 days in month)",
        "n_price_obs_legacy_rule": "n_price_obs_legacy = n_price_obs = Drive price matrix non-null days (ffill 含停牌日)",
        "n_valid_days_equals_n_trade_days": bool((csv_out["n_valid_days"].astype(int) == ntd).all()),
        "complete_month_rows": int(cm.sum()),
        "complete_month_violations_n_trade_days_lt_min": int(((cm == 1) & (ntd < MIN_VALID_DAYS)).sum()),
        "incomplete_month_rows": int((cm == 0).sum()),
        "n_price_obs_ne_n_trade_days_rows": int((npo != ntd).sum()),
        "n_price_obs_and_n_trade_days_both_zero_rows": int(((npo == 0) & (ntd == 0)).sum()),
        "legacy_definition": {
            "complete_month_rule": f"n_price_obs >= {MIN_VALID_DAYS}",
            "complete_month_rows_legacy": int((npo >= MIN_VALID_DAYS).sum()),
            "disagreement_rows": int(((npo >= MIN_VALID_DAYS) != (ntd >= MIN_VALID_DAYS)).sum()),
        },
        "return_status_counts": csv_out["return_status"].value_counts().to_dict(),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"rows={len(csv_out)} -> {Path(args.out).relative_to(ROOT)}")
    print("sample_version:", csv_out["sample_version"].iat[0],
          "| rule:", SAMPLE_VERSION_RULE,
          "| source:", out["sample_version_source"].iat[0])
    print("return_status:", report["return_status_counts"],
          "| complete_month=1:", report["complete_month_rows"])
    print("n_valid_days==n_trade_days:", report["n_valid_days_equals_n_trade_days"],
          "| complete_month violations:", report["complete_month_violations_n_trade_days_lt_min"])
    print("n_price_obs != n_trade_days:", report["n_price_obs_ne_n_trade_days_rows"],
          "| both zero:", report["n_price_obs_and_n_trade_days_both_zero_rows"])
    print("report ->", Path(args.report).relative_to(ROOT))
    return 0


def pickle_load(p):
    import pickle
    with open(p, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    raise SystemExit(main())
