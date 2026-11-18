#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2 Step 1-2：firm-day 面板构造 + rraw 派生 + 除权日检测。

输入：腾讯 daily_raw + daily_qfq（854.6 万行/版）
输出（按年分片）：
  data/processed/daily_panel/{year}.csv
    code,date,close_raw,close_qfq,volume,amount,rraw,ca_flag
  - rraw = raw_close_d/raw_close_{d-1}-1（raw 版主路径，新浪双源已验）
  - ca_flag = 1 当 qfq/raw 比值相邻日跳变 > 1%（除权事件候选；qfq 噪声 <0.5% 不误触）
  - 停牌日无行（腾讯原始语义）；上市首日 rraw=NaN（valid_trade=0 由 S2 质量门处理）

写入语义（Phase 1.1 / I-4）：
  - 默认：目标年度文件已存在则拒绝写入（SystemExit 2），不改动现有文件；
  - --append：仅显式传入才允许追加，且必须同时给 --append-reason；
  - 所有模式先写 {year}.csv.tmp，成功后 os.replace 原子替换；失败清理 tmp。

用法:
  python scripts/data_audit/s2_build_daily_panel.py [--years 2018-2025]
  python scripts/data_audit/s2_build_daily_panel.py --append --append-reason "断点补写"
"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TENCENT = ROOT / "data/raw/tencent"
OUT_DIR = ROOT / "data/processed/daily_panel"
REPORT_PATH = ROOT / "reports/data_gates/G0/s2_daily_panel_report.json"
RATIO_JUMP = 0.01  # qfq/raw 比值跳变阈值（除权日检测）
COLUMNS = ["code", "date", "close_raw", "close_qfq", "volume", "amount", "rraw", "ca_flag"]


def _refuse(msg):
    print(f"[S2][REFUSE] {msg}", file=sys.stderr)
    raise SystemExit(2)


def _write_json_atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _open_year_writer(target, mode):
    """打开目标年度文件的临时写句柄；返回 (fh, csv.writer, tmp_path)。

    append 且目标非空时先把现有内容复制到 tmp（保证失败不半写），
    表头仍只在新文件/空文件写一次。
    """
    tmp = target.with_name(target.name + ".tmp")
    f = None
    try:
        if mode == "append" and target.exists() and target.stat().st_size > 0:
            shutil.copyfile(target, tmp)
            with open(tmp, "rb+") as raw:
                raw.seek(-1, os.SEEK_END)
                if raw.read(1) != b"\n":
                    raw.write(b"\n")
            f = open(tmp, "a", encoding="utf-8", newline="")
            return f, csv.writer(f), tmp
        f = open(tmp, "w", encoding="utf-8", newline="")
        w = csv.writer(f)
        w.writerow(COLUMNS)
        return f, w, tmp
    except BaseException:
        if f is not None:
            f.close()
        tmp.unlink(missing_ok=True)
        raise


def run(tencent_dir=TENCENT, out_dir=OUT_DIR, years=None, *, mode="default",
        append_reason=None, report_path=REPORT_PATH):
    """构造年度日频面板；返回 summary（同时按 report_path 原子写 JSON）。"""
    tencent_dir = Path(tencent_dir)
    out_dir = Path(out_dir)
    if years is None:
        years = {str(y) for y in range(2018, 2026)}
    else:
        years = {str(y) for y in years}
    if mode not in ("default", "append"):
        _refuse(f"未知写入模式: {mode!r}（仅支持 default / append）")
    if mode == "append" and not (append_reason or "").strip():
        _refuse("--append 必须同时提供 --append-reason，说明追加原因")

    targets = {yr: out_dir / f"{yr}.csv" for yr in sorted(years)}
    existing = [p for p in targets.values() if p.exists()]
    if mode == "default" and existing:
        names = ", ".join(sorted(p.name for p in existing))
        _refuse(
            f"目标年度文件已存在，拒绝写入（防止重复追加/覆盖）：{names}。"
            "如确需追加，请显式使用 --append --append-reason <原因>；"
            "如确需重建，请先备份并移走现有文件。"
        )

    raw_root = tencent_dir / "daily_raw"
    if not raw_root.is_dir():
        _refuse(f"输入目录不存在：{raw_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(lambda: [0, 0, 0])  # rows, ca_flags, firms
    writers = {}  # yr -> (fh, csv.writer, tmp_path)
    try:
        for code_dir in sorted(raw_root.iterdir()):
            if not code_dir.is_dir():
                continue
            code = code_dir.name
            qfq_dir = tencent_dir / "daily_qfq" / code
            if not qfq_dir.is_dir():
                continue
            last_close = None  # 跨年保留前一年最后收盘
            for year_file in sorted(code_dir.iterdir()):
                yr = year_file.stem
                if yr not in years:
                    continue
                qfq_file = qfq_dir / year_file.name
                # 读两版
                rows = {}
                for p, tag in ((year_file, "raw"), (qfq_file, "qfq")):
                    if not p.exists():
                        continue
                    with open(p, encoding="utf-8") as f:
                        for i, row in enumerate(csv.reader(f)):
                            if i == 0 or not row or not row[0]:
                                continue
                            d = row[0]
                            rec = rows.setdefault(d, {"raw": None, "qfq": None, "vol": None, "amt": None})
                            if tag == "raw" and len(row) >= 9:
                                rec["raw"] = float(row[2]); rec["vol"] = float(row[5]); rec["amt"] = float(row[8])
                            elif tag == "qfq" and len(row) >= 3:
                                rec["qfq"] = float(row[2])
                dates = sorted(rows)
                if not dates:
                    continue
                if yr not in writers:
                    writers[yr] = _open_year_writer(targets[yr], mode)
                _, w, _ = writers[yr]
                prev_raw = last_close  # 跨年保留前一年最后收盘（修复：每年首个交易日 rraw 缺失）
                prev_ratio = None
                for d in dates:
                    rec = rows[d]
                    r = rec["raw"]
                    if r is None:
                        continue
                    rraw = (r / prev_raw - 1) if prev_raw else None
                    ca = 0
                    if rec["qfq"] and r > 0:
                        ratio = rec["qfq"] / r
                        if prev_ratio is not None and ratio and abs(ratio - prev_ratio) / prev_ratio > RATIO_JUMP:
                            ca = 1
                        prev_ratio = ratio
                    w.writerow([code, d, r, rec["qfq"] if rec["qfq"] is not None else "", rec["vol"] or "", rec["amt"] or "",
                                f"{rraw:.8f}" if rraw is not None else "", ca])
                    stats[yr][0] += 1
                    stats[yr][1] += ca
                    prev_raw = r
                stats[yr][2] += 1
                last_close = rows[dates[-1]]["raw"]

        for fh, _, _ in writers.values():
            fh.close()
        for yr in sorted(writers):
            _, _, tmp = writers[yr]
            os.replace(tmp, targets[yr])
    except BaseException:
        for fh, _, _ in writers.values():
            if not fh.closed:
                fh.close()
        for _, _, tmp in writers.values():
            Path(tmp).unlink(missing_ok=True)
        raise

    # 汇总
    total_rows = sum(v[0] for v in stats.values())
    total_ca = sum(v[1] for v in stats.values())
    summary = {
        "report_id": "s2_daily_panel",
        "mode": mode,
        "append_reason": append_reason if mode == "append" else None,
        "rows": total_rows,
        "ca_flags": total_ca,
        "ca_rate": round(total_ca / total_rows, 6) if total_rows else None,
        "by_year": {y: {"rows": v[0], "ca_flags": v[1], "firms": v[2]} for y, v in sorted(stats.items())},
        "note": f"rraw=raw close 派生；ca_flag=qfq/raw 比值跳变>{RATIO_JUMP}（除权事件候选，S2.5 mv 交叉校验复核）",
    }
    if report_path is not None:
        _write_json_atomic(report_path, summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2018-2025")
    ap.add_argument("--append", action="store_true", help="显式允许向已存在的年度文件追加（必须同时给 --append-reason）")
    ap.add_argument("--append-reason", default=None, help="追加原因（--append 必填，用于审计）")
    args = ap.parse_args(argv)
    y0, y1 = (int(x) for x in args.years.split("-"))
    years = {str(y) for y in range(y0, y1 + 1)}

    summary = run(
        TENCENT,
        OUT_DIR,
        years,
        mode="append" if args.append else "default",
        append_reason=args.append_reason,
        report_path=REPORT_PATH,
    )
    print(f"面板完成: {summary['rows']:,} 行, ca_flag {summary['ca_flags']:,} ({summary['ca_rate']})")
    for y, v in sorted(summary["by_year"].items()):
        print(f"  {y}: {v['rows']:,} 行, ca {v['ca_flags']:,}, firms {v['firms']}")


if __name__ == "__main__":
    main()
