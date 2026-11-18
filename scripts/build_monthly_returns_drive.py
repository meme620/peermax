"""S1 全期月收益面板 builder（Drive raw close 派生，2026-09-03 审计 F5 持久化）。

重建 2026-08-29 全期化交付面板 `data/processed/monthly_returns_clean.csv`
（交付时为一次性会话产物，builder 未落盘——本脚本即该补齐；首月 2000-02）。

口径：
  - raw close 月末环比（close_none，不含分红；除权日断崖保留在 R 中，
    见 s3a_report.md §3.3 除权口径注 / 审计报告 s3a_data_usage_review_20260903.md F2）
  - 质量列契约沿用 build_monthly_returns.py：n_trading_days + monthly_return_status
    （PASS>=15d / CONDITIONAL 5-14d / BLOCKED <5d）。
    注意：交付 CSV 仅 firm_id,month,R 三列（未带质量列）——消费方若需质量
    过滤，应使用本脚本的 --with-status 输出，勿默认交付面板已过滤。

用法：
  python scripts/build_monthly_returns_drive.py                # 重建并与交付对账
  python scripts/build_monthly_returns_drive.py --with-status  # 输出带质量列的增强面板
输出：
  artifacts/monthly_returns_drive_rebuild.csv（--with-status 时含质量列）
  stdout 对账结论（行数/逐位一致率）
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DRIVE_PX = ROOT / "data/raw/drive/daily_close_none_20260630.pkl"
SHIPPED = ROOT / "data/processed/monthly_returns_clean.csv"
OUT = ROOT / "artifacts/monthly_returns_drive_rebuild.csv"


def build(with_status: bool) -> pd.DataFrame:
    px = pickle.load(open(DRIVE_PX, "rb"))
    me = px.resample("ME").last()
    cnt = px.notna().resample("ME").sum()
    ret = me / me.shift(1) - 1.0
    ret.index = [d.strftime("%Y-%m") for d in ret.index]
    cnt.index = ret.index
    # future_stack 实现保留 NaN 值（dropna 默认 False），故 stack 后显式丢 NaN
    long_r = ret.stack(future_stack=True).dropna().rename("R").reset_index()
    long_r.columns = ["month", "col", "R"]
    def fid(c: str) -> str:
        code, exch = c.split(".")
        return exch.lower() + code
    long_r["firm_id"] = long_r["col"].map(fid)
    out = long_r[["firm_id", "month", "R"]].copy()
    if with_status:
        cmap = cnt.stack(future_stack=True).rename("n_trading_days").reset_index()
        cmap.columns = ["month", "col", "n_trading_days"]
        cmap["firm_id"] = cmap["col"].map(fid)
        out = out.merge(cmap[["firm_id", "month", "n_trading_days"]],
                        on=["firm_id", "month"], how="left")
        out["monthly_return_status"] = np.select(
            [out["n_trading_days"] >= 15, out["n_trading_days"] >= 5],
            ["PASS", "CONDITIONAL"], default="BLOCKED")
    return out.sort_values(["month", "firm_id"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-status", action="store_true")
    args = ap.parse_args()
    out = build(args.with_status)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"rebuilt rows={len(out)} -> {OUT}")

    ship = pd.read_csv(SHIPPED)
    key = ["firm_id", "month"]
    m = ship.merge(out[key + ["R"]], on=key, how="outer",
                   suffixes=("_ship", "_reb"), indicator=True)
    both = m[m["_merge"] == "both"]
    only_s = int((m["_merge"] == "left_only").sum())
    only_r = int((m["_merge"] == "right_only").sum())
    diff = (both["R_ship"] - both["R_reb"]).abs()
    bad = int((diff > 1e-9).sum())
    print(f"shipped rows={len(ship)} | matched={len(both)} "
          f"only_shipped={only_s} only_rebuilt={only_r} R_mismatch(>1e-9)={bad}")
    if len(ship) == len(out) and only_s == 0 and only_r == 0 and bad == 0:
        print("RECONCILE: EXACT MATCH（重建与交付逐位一致）")
        return 0
    print("RECONCILE: MISMATCH — 交付面板非本口径精确重建，登记差异后决定")
    if bad:
        print(both.loc[diff > 1e-9, key].head(10))
    if only_s or only_r:
        print(m[m["_merge"] != "both"].head(10))
    return 1


if __name__ == "__main__":
    sys.exit(main())
