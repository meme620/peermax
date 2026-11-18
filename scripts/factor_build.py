#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S2.5：自建 FF4 因子 factor_returns_daily（MKT/SMB/HML/MOM，日度）。

构造规则（FF/CH 方法，universe 规则登记于 factor_version）：
  1. 排序时点：月末最后交易日收盘信息 → 次月组合（无前视）
  2. 市值 = close × 股本（SHAREBONUS 事件重构，事件间恒定；缺初始股本股票不参与排序）
  3. BM = TOTAL_EQUITY_PK（东财，NOTICE_DATE ≤ 排序日）/ 市值
  4. 2×3：size 中位数 × BM 30/70 分位 → 6 组合（市值加权日收益）
  5. SMB = (S/L+S/M+S/H)/3 − (B/L+B/M+B/H)/3；HML = (S/H+B/H)/2 − (S/L+B/L)/2
  6. MKT = 全市场市值加权（rraw − rf_d）；MOM = 12-1 动量（跳过最近 1 月）按 FF 方法 2×3
  7. rf_d = 一年期存款基准利率 1.5%（2015-10-24 起生效，日度 = 0.015/365；待央行公告核实登记）
输出：data/processed/factor_returns_daily.csv + reports/data_gates/G0/factor_construction_report.json
用法: python scripts/factor_build.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from fileio import read_pickle

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/daily_panel"
SHAREBONUS = ROOT / "data/processed/sharebonus_events.csv"
FINANCE = ROOT / "data/raw/eastmoney/finance"
OUT_CSV = ROOT / "data/processed/factor_returns_daily.csv"
OUT_JSON = ROOT / "reports/data_gates/G0/factor_construction_report.json"
RF_ANNUAL = 0.015  # 一年期存款基准利率（2015-10-24 起；待央行公告核实）


def load_panel():
    """面板 → {code: {date: (close, rraw)}}。"""
    by_code = defaultdict(dict)
    for yf in sorted(PANEL.glob("*.csv")):
        with open(yf, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rr = r["rraw"]
                if rr:
                    by_code[r["code"]][r["date"]] = (float(r["close_raw"]), float(rr))
    return by_code


def load_shares():
    """SHAREBONUS → {code: [(ex_date, total_shares)]}（全量，含 2016 前初始）"""
    ev = defaultdict(list)
    with open(SHAREBONUS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r["EX_DIVIDEND_DATE"] or "")[:10]
            ts = r["TOTAL_SHARES"] or ""
            if d and ts:
                sec = r.get("SECUCODE") or ""
                prefix = "sh" if sec.endswith(".SH") else ("sz" if sec.endswith(".SZ") else "")
                if prefix:
                    ev[prefix + r["SECURITY_CODE"]].append((d, ts))
    return {k: sorted(v) for k, v in ev.items()}


def load_bm():
    """2026-08-29 全期化：改用 Drive 官方 TTM BM（2001 起，batch2 决定 TTM 口径）；
    原东财 TOTAL_EQUITY_PK 2011 前覆盖不足（<30 家/月）。返回 {code: [(date, bm_ratio)]}（月末值）。"""
    bm_df = read_pickle(ROOT / "data/raw/drive/daily_book_to_market_ratio_ttm_20260630.pkl")
    bm_df.columns = [c.split(".")[0] for c in bm_df.columns]
    bm_m = bm_df.resample("ME").last()
    bm = defaultdict(list)
    for d, row in bm_m.iterrows():
        ds = d.strftime("%Y-%m-%d")
        for c, v in row.items():
            if v == v and v > 0:
                prefix = "sh" if c.startswith(("6", "9")) else "sz"
                bm[prefix + c].append((ds, float(v)))
    return {k: sorted(v) for k, v in bm.items()}


def main():
    by_code = load_panel()
    shares = load_shares()
    bm = load_bm()
    print(f"面板 {len(by_code)} 只；股本事件 {len(shares)} 只；BM {len(bm)} 只")

    # 交易日历（面板出现过的全部日期，排序）
    all_dates = sorted({d for c in by_code.values() for d in c})
    # 月末最后交易日
    month_ends = {}
    for d in all_dates:
        month_ends[d[:7]] = d
    print(f"交易日 {len(all_dates)}，月末 {len(month_ends)}")

    # 月度截面构造（排序月）
    # 对每个月末 t：为每只股票计算 size_t/bm_t/mom_t → 排序 → 下月组合
    def shares_on(code, date):
        evs = shares.get(code)
        if not evs:
            return None
        cur = None
        for d, ts in evs:
            if d <= date:
                cur = ts
            else:
                break
        if cur is None:
            return None  # 排序日前无事件 → 初始股本未知（不参与排序）
        try:
            return float(cur)
        except ValueError:
            return None

    def bm_on(code, date):
        rows = bm.get(code)
        if not rows:
            return None
        cur = None
        for d, v in rows:
            if d <= date:
                cur = v
            else:
                break
        if cur is None:
            return None
        return cur

    # 月度动量（12-1，跳过最近 1 月）：按月收益累计
    monthly_ret = defaultdict(dict)  # code -> {month: R}
    for code, days in by_code.items():
        for d in sorted(days):
            m = d[:7]
            monthly_ret[code].setdefault(m, 1.0)
            monthly_ret[code][m] *= 1 + days[d][1]

    def mom_on(code, month):
        """month 的动量 = prod(m-12..m-2)(1+R)-1"""
        try:
            y, mo = int(month[:4]), int(month[5:7])
        except ValueError:
            return None
        prod = 1.0
        for k in range(2, 13):
            yy, mm = y, mo - k
            while mm <= 0:
                mm += 12
                yy -= 1
            key = f"{yy:04d}-{mm:02d}"
            r = monthly_ret.get(code, {}).get(key)
            if r is None:
                return None
            prod *= 1 + r
        return prod - 1

    # 逐月排序 → 组合收益
    sorted_months = sorted(month_ends)
    factor_rows = {}  # date -> [rf, mkt, smb, hml, mom]
    mom_sorted_months = sorted_months  # 动量与 size/bm 同排序月（都用 t 月末信息）

    for idx, month in enumerate(sorted_months):
        if idx == 0:
            continue  # 首月无前月排序
        sort_month = sorted_months[idx - 1]  # 排序月 = 前一个月末（t 月末信息）
        sort_date = month_ends[sort_month]
        # 截面特征（t 月末）
        chars = {}
        for code, days in by_code.items():
            rec = days.get(sort_date)
            if not rec:
                continue
            close, _ = rec
            s = shares_on(code, sort_date)
            if not s:
                continue
            size = close * s
            b = bm_on(code, sort_date)
            mom = mom_on(code, sort_month)
            if mom is None:
                continue
            chars[code] = {"size": size, "bm": b if b is not None else None, "mom": mom}  # b 已是 Drive TTM 比值
        n = len(chars)
        if n < 100:
            continue
        # 排序
        size_cut = sorted(v["size"] for v in chars.values())[n // 2]
        bm_vals = sorted(v["bm"] for v in chars.values() if v["bm"] is not None)
        if len(bm_vals) < 30:
            continue
        bm_lo = bm_vals[max(0, int(len(bm_vals) * 0.3) - 1)]
        bm_hi = bm_vals[min(len(bm_vals) - 1, int(len(bm_vals) * 0.7))]
        mom_vals = sorted(v["mom"] for v in chars.values())
        mom_lo = mom_vals[max(0, int(len(mom_vals) * 0.3) - 1)]
        mom_hi = mom_vals[min(len(mom_vals) - 1, int(len(mom_vals) * 0.7))]

        def portfolio(codes):
            """返回组合内股票的 (code → size) 子集。"""
            return {c: chars[c]["size"] for c in codes}

        def group(codes, key):
            g = defaultdict(list)
            for c in codes:
                g[key(chars[c])].append(c)
            return g

        s_group = lambda v: "S" if v["size"] <= size_cut else "B"
        b_group = lambda v: "H" if v["bm"] is not None and v["bm"] > bm_hi else ("M" if v["bm"] is not None and v["bm"] > bm_lo else ("L" if v["bm"] is not None else None))
        m_group = lambda v: "U" if v["mom"] > mom_hi else ("D" if v["mom"] <= mom_lo else None)

        # 组合分类（市值加权组合日收益，组合月 = month）
        combos = {}
        for c in chars:
            sg = s_group(chars[c])
            bg = b_group(chars[c])
            combos.setdefault(sg, {}).setdefault(bg, []).append(c)
            combos.setdefault(sg, {}).setdefault("ALL", []).append(c)
        # 收集下月（组合月）交易日
        combo_dates = [d for d in all_dates if d[:7] == month[:7]]
        if not combo_dates:
            continue
        for d in combo_dates:
            # 组合日收益
            def wret(codes):
                num = den = 0.0
                for c in codes:
                    rec = by_code[c].get(d)
                    if rec:
                        w = chars[c]["size"]
                        num += w * rec[1]
                        den += w
                return num / den if den > 0 else None
            mkt_num = mkt_den = 0.0
            for c in chars:
                rec = by_code[c].get(d)
                if rec:
                    mkt_num += chars[c]["size"] * rec[1]
                    mkt_den += chars[c]["size"]
            mkt = mkt_num / mkt_den if mkt_den > 0 else None
            smb = hml = momf = None
            sl, sm, sh, bl, bm_, bh = (None,) * 6
            if combos.get("S") and combos["S"].get("L"):
                sl = wret(combos["S"]["L"])
            if combos.get("S") and combos["S"].get("M"):
                sm = wret(combos["S"].get("M", []))
            if combos.get("S") and combos["S"].get("H"):
                sh = wret(combos["S"]["H"])
            if combos.get("B") and combos["B"].get("L"):
                bl = wret(combos["B"]["L"])
            if combos.get("B") and combos["B"].get("M"):
                bm_ = wret(combos["B"].get("M", []))
            if combos.get("B") and combos["B"].get("H"):
                bh = wret(combos["B"]["H"])
            if sl is not None and bl is not None:
                smb = ((sl + sm + sh) / 3 - (bl + bm_ + bh) / 3) if None not in (sm, sh, bm_, bh) else None
            if sl is not None and bh is not None and sh is not None and bl is not None:
                hml = (sh + bh) / 2 - (sl + bl) / 2
            # MOM（2×3：size × mom）
            dl, du = (None,) * 2
            # 简化：MOM 用全样本 3 分位（不 split size，登记简化）
            mom_codes = [c for c in chars if m_group(chars[c])]
            if len(mom_codes) > 30:
                ups = [c for c in chars if m_group(chars[c]) == "U"]
                dns = [c for c in chars if m_group(chars[c]) == "D"]
                du = wret(ups) if ups else None
                dl = wret(dns) if dns else None
                if du is not None and dl is not None:
                    momf = du - dl
            rf_d = RF_ANNUAL / 365
            factor_rows[d] = {
                "date": d, "rf_d": round(rf_d, 8),
                "MKT_d": round(mkt - rf_d, 8) if mkt is not None else None,
                "SMB_d": round(smb, 8) if smb is not None else None,
                "HML_d": round(hml, 8) if hml is not None else None,
                "MOM_d": round(momf, 8) if momf is not None else None,
                "n_stocks": n,
            }

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "rf_d", "MKT_d", "SMB_d", "HML_d", "MOM_d", "n_stocks", "factor_version"])
        w.writeheader()
        for d in sorted(factor_rows):
            r = factor_rows[d]
            w.writerow({**r, "factor_version": "FF4_SELF_BUILT_v1"})
    print(f"因子日度: {len(factor_rows)} 天 → {OUT_CSV}")
    dates = sorted(factor_rows)
    if dates:
        print("覆盖:", dates[0], "~", dates[-1])
    # 描述统计
    import statistics
    stats = {}
    for k in ["MKT_d", "SMB_d", "HML_d", "MOM_d"]:
        vals = [factor_rows[d][k] for d in factor_rows if factor_rows[d][k] is not None]
        if vals:
            stats[k] = {"n": len(vals), "mean": round(statistics.mean(vals), 6),
                        "sd": round(statistics.stdev(vals), 6)}
    report = {
        "report_id": "factor_construction", "version": "FF4_SELF_BUILT_v1",
        "method": "FF 2×3 月末排序/次月组合；size=close×SHAREBONUS 股本重构；BM=东财 TOTAL_EQUITY_PK(≥NOTICE_DATE)/size；MOM=12-1（简化：全样本 3 分位不 split size，登记简化待升级）",
        "rf": {"annual": RF_ANNUAL, "note": "一年期存款基准利率 1.5%，2015-10-24 起生效；日度=1.5%/365；待央行公告核实登记"},
        "coverage": {"first": dates[0] if dates else None, "last": dates[-1] if dates else None,
                     "n_days": len(factor_rows)},
        "descriptive": stats,
        "limitations": [
            "市值缺初始股本股票不参与排序（SHAREBONUS 覆盖 86% 股票）；覆盖率随 S2 股本源补全而改善",
            "MOM 为全样本 3 分位简化版（未 split size）——sensitivity 升级待办",
            "rf 常数 1.5% 待央行公告核实",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("描述统计:", json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"输出: {OUT_JSON}")


if __name__ == "__main__":
    main()
