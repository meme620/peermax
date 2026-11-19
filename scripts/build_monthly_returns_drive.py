"""inventory 全期月收益面板 builder（Drive raw close 派生，2026-09-03 审计 F5 持久化）。

重建 2026-08-29 全期化交付面板 `data/processed/monthly_returns_clean.csv`
（交付时为一次性会话产物，builder 未落盘——本脚本即该补齐；首月 2000-02）。

口径：
  - raw close 月末环比（close_none，不含分红；除权日断崖保留在 R 中，
    见 analyst_report.md §3.3 除权口径注 / 审计报告 analyst_data_usage_review_20260903.md F2）
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
from fileio import read_pickle

ROOT = Path(__file__).resolve().parents[1]
DRIVE_PX = ROOT / "data/raw/drive/daily_close_none_20260630.pkl"
SHIPPED = ROOT / "data/processed/monthly_returns_clean.csv"
PX_POST = ROOT / "data/raw/drive/daily_adj_close_post_20260630.pkl"
REPORT = ROOT / "reports/data_gates/G0/return_convention.json"
CLIFF = -0.215  # A 股涨跌停制度下不可能的日跌幅，出现即为公司行为
CONVENTIONS = {'post_adjusted': PX_POST, 'raw': DRIVE_PX}
# 每个口径写自己的文件：后复权是 headine 交付面板，raw 只作敏感性（同名会互相覆盖）
OUT_BY_CONVENTION = {
    'post_adjusted': ROOT / "data/processed/monthly_returns_clean.csv",
    'raw': ROOT / "data/processed/monthly_returns_clean_raw.csv",
}


def build(with_status: bool, convention: str = 'post_adjusted') -> pd.DataFrame:
    if convention not in CONVENTIONS:
        raise SystemExit(f'[STOP] 未知收益口径 {convention!r}；可选 {sorted(CONVENTIONS)}')
    px = read_pickle(CONVENTIONS[convention])
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


def cliff_stats(px) -> dict:
    r = px.pct_change(fill_method=None)
    bad = r < CLIFF
    return {'cells_below_cliff': int(bad.to_numpy().sum()),
            'firm_days_below_cliff': int(bad.any(axis=0).sum()),
            'cliff_threshold': CLIFF}


def write_report(convention: str, stats: dict, rows: int) -> None:
    import json
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        'return_convention': convention,
        'source_file': str(CONVENTIONS[convention].relative_to(ROOT)),
        'rows': rows,
        'cliff_stats': stats,
        'note': 'post_adjusted = 后复权收盘（含公司行为调整）；raw = 未复权，除权日断崖保留，仅作敏感性',
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-status", action="store_true")
    ap.add_argument("--convention", choices=sorted(CONVENTIONS), default='post_adjusted')
    args = ap.parse_args()
    stats = cliff_stats(read_pickle(CONVENTIONS[args.convention]))
    out = build(args.with_status, args.convention)
    out_path = OUT_BY_CONVENTION[args.convention]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    write_report(args.convention, stats, len(out))
    print(f"rebuilt rows={len(out)} convention={args.convention} -> {out_path}")

    # 独立对拍：后复权口径应与老师交付的 monthly_returns_adj.csv 逐位一致（不再与自身对账）
    teacher = ROOT / "data/raw/drive/monthly_returns_adj.csv"
    if args.convention == 'post_adjusted' and teacher.is_file():
        t = pd.read_csv(teacher, dtype={'firm_id': str}).rename(columns={'R_adj': 'R'})
        # teacher 文件用裸 6 位代码；本仓 firm_id 为 sz/sh 前缀
        t['firm_id'] = t['firm_id'].str.zfill(6).map(
            lambda c: ('sh' if c[0] in '69' else 'sz') + c)
        m = out.merge(t, on=['firm_id', 'month'], how='inner', suffixes=('_new', '_tch'))
        d = (m['R_new'] - m['R_tch']).abs()
        worse = int((d > 1e-9).sum())
        print(f"对拍 teacher: matched={len(m)} max|diff|={float(d.max()):.3e} mismatched={worse}")
        if worse:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
