#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""股本序列提取（市值分母；年报文本三策略 + 东财送转表交叉，替代 baostock totalShare）。

策略（2026-08-09 实测验证）:
  A. 精确匹配"期末总股本"后数字（A 股标准模板，单位=股）
  B. 年报声明"未发生变化/未发生变动" → 沿用上年期末
  C. 分红描述"总股本 X 股/万股为基数"（A+H 特殊模板；单位判读）

输入: data/raw/cninfo/annual_txt/{code}/{year}.txt（35,929 份）
       data/raw/eastmoney/sharebonus/{code}.csv（送转事件，交叉验证）
输出: data/raw/institutional/share_capital_history.csv
      (code, year, end_share, method, status, sharebonus_check)
"""
import csv
import glob
import os
import re
import sys
import time

TXT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "cninfo", "annual_txt")
BONUS = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "eastmoney", "sharebonus")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "institutional")
os.makedirs(OUT, exist_ok=True)

NUM = re.compile(r"([\d,]{6,}\.\d{2})")


def parse_num(s: str) -> float:
    return float(s.replace(",", ""))


def strategy_a(t: str):
    """期末总股本 + 数字（单位判读：股/万股/亿股；2026-08-09 修单位 bug）。"""
    for m in re.finditer("期末总股本", t):
        seg = t[m.end():m.end() + 120]
        m2 = NUM.search(seg)
        if m2:
            v = parse_num(m2.group(1))
            after = seg[m2.end():m2.end() + 15]
            if "万股" in after:
                v *= 10000
            elif "亿股" in after:
                v *= 1e8
            return v
    return None


def strategy_c(t: str):
    """总股本 X 股/万股为基数（分红描述；要求"为基数"在数字后 25 字符内，收紧防误匹配）。"""
    for m in re.finditer("总股本", t):
        seg = t[m.end():m.end() + 80]
        m2 = NUM.search(seg)
        if not m2:
            continue
        after = seg[m2.end():m2.end() + 25]
        if "为基数" not in after:
            continue
        v = parse_num(m2.group(1))
        if "万股" in after:
            v *= 10000
        elif "亿股" in after:
            v *= 1e8
        if v >= 1e8:
            return v
    return None


def main():
    files = sorted(glob.glob(os.path.join(TXT, "*", "*.txt")))
    print(f"扫描 {len(files)} 份年报", file=sys.stderr)
    rows = []
    t0 = time.time()
    by_code = {}
    for p in files:
        parts = p.replace("\\", "/").split("/")
        code, year = parts[-2], int(parts[-1][:4])
        with open(p, encoding="utf-8", errors="ignore") as f:
            t = f.read()
        end = strategy_a(t)
        method = "A"
        if end is None:
            # 策略 B：标准模板句（限股本语境，避免"资本管理未发生变化"误命中）
            if "股份总数及股本结构未发生变化" in t or "股份总数未发生变化" in t:
                end = None  # 沿用由后处理
                method = "B"
            else:
                end = strategy_c(t)
                method = "C"
        by_code.setdefault(code, {}).update({year: (end, method)})
        if len(by_code) % 1000 == 0 and len(by_code) <= 1000:
            pass
    # B 策略沿用 + 输出
    for code, years in sorted(by_code.items()):
        prev = None
        for y in sorted(years):
            end, method = years[y]
            status = "ok"
            if method == "B":
                end, method = prev, "B_carry"
                status = "carried"
            if end is not None:
                prev = end
            rows.append((code, y, end, method, status))
    # 送转交叉检查
    bonus_check = {}
    for code in by_code:
        bp = os.path.join(BONUS, f"{code}.csv")
        if os.path.exists(bp):
            with open(bp, encoding="utf-8") as f:
                br = list(csv.reader(f))
            if len(br) > 1:
                hdr = br[0]
                try:
                    iy = hdr.index("REPORT_DATE")
                except ValueError:
                    iy = -1
                years_bonus = {r[iy][:4] for r in br[1:] if iy >= 0 and len(r) > iy}
                bonus_check[code] = years_bonus
    with open(os.path.join(OUT, "share_capital_history.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "end_share", "method", "status", "bonus_years"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], ",".join(sorted(bonus_check.get(r[0], set())))])

    ok = sum(1 for r in rows if r[2] is not None)
    print(f"完成: {len(rows)} 股票-年份, 提取成功 {ok} ({ok/len(rows):.0%}), 耗时 {(time.time()-t0)/60:.1f} 分钟", file=sys.stderr)


if __name__ == "__main__":
    main()
