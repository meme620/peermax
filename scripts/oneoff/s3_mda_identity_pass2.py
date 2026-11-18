"""MD&A 身份校验第二遍：压低 unknown 占比（扩展窗口 + 公司名词表扫描）。

第一遍（s3_mda_integrity_audit.py）只用前 3000 字 + "证券代码/公司代码" 关键字，
33.9% 文件无信号（大量年报正文首页不含代码行，如平安银行）。本遍对 identity!=match
的行追加两种信号：
  1. 关键字窗口放宽到前 30000 字（公司简介章节通常在文首 2 万字符内）；
  2. 公司名词表扫描（a_stocks 现名 + name_history_refined 历史名，长度>=3），
     前 3000 字中命中唯一样本代码 -> 判定；命中多代码 -> ambiguous 保守置 unknown。
输出：原地更新 artifacts/review/mda_src_identity.csv（新增 method2 列；identity 取强信号）
用法: python scripts/data_audit/s3_mda_identity_pass2.py
"""
import csv
import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data/raw/cninfo/annual_txt"
IDENT = ROOT / "artifacts/review/mda_src_identity.csv"
A_STOCKS = ROOT / "data/raw/baostock/stock_basic/a_stocks.csv"
NAME_HIST = ROOT / "data/processed/name_history_refined.csv"

CODE_RE = re.compile(r"(?:证券代码|股票代码|公司代码|证券简称及代码|股票简称及代码)[：:\s]*([0-9]{6})")
NAME_LINE_RE = re.compile(r"(?:证券简称|股票简称|公司简称)[：:\s]*([\u4e00-\u9fa5A-Za-z0-9]{2,12})")
CODE_PAIR_RE = re.compile(r"([0-9]{6})\s*[,、]\s*g[sz][sh][0-9]{7}")
VALID_PREFIX = ("000", "001", "002", "003", "300", "301", "302",
                "600", "601", "603", "605", "688", "689")
STOP_NAMES = {"中国", "公司", "集团", "股份", "国际", "科技", "实业", "发展", "控股", "投资", "银行"}


def valid(c):
    return bool(c) and c.startswith(VALID_PREFIX)


def load_names():
    name2codes = collections.defaultdict(set)
    with open(A_STOCKS, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 1 and row[0]:
                n = re.sub(r"[ \u3000]", "", row[1])
                if len(n) >= 3 and n not in STOP_NAMES:
                    name2codes[n].add(row[0].split(".")[-1])
    if NAME_HIST.exists():
        with open(NAME_HIST, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) > 3 and row[0]:
                    n = re.sub(r"[ \u3000]", "", row[3])
                    if len(n) >= 3 and n not in STOP_NAMES:
                        name2codes[n].add(row[0].split(".")[-1])
    pat = re.compile("|".join(sorted((re.escape(n) for n in name2codes), key=len, reverse=True)))
    return name2codes, pat


def main():
    name2codes, name_pat = load_names()
    rows = list(csv.DictReader(open(IDENT, encoding="utf-8")))
    updated = 0
    method2 = collections.Counter()
    for row in rows:
        if row["identity"] == "match":
            row["method2"] = row.get("method2", "")
            continue
        p = TXT / row["code6"] / f"{row['year']}.txt"
        if not p.exists():
            row["method2"] = "file_missing"
            continue
        t = p.read_text(encoding="utf-8", errors="replace")
        head30 = re.sub(r"[ \u3000\t]", "", t[:30000])
        head3 = re.sub(r"[ \u3000\t]", "", t[:3000])
        m = CODE_RE.search(head30) or CODE_PAIR_RE.search(head30)
        if m and valid(m.group(1)):
            row["content_code"] = m.group(1)
            row["identity"] = "match" if m.group(1) == row["code6"] else "mismatch"
            row["method"] = "code30k"
            row["method2"] = "code30k"
            updated += 1
            method2["code30k"] += 1
            continue
        hits = collections.Counter()
        for mm in name_pat.finditer(head3):
            for c in name2codes.get(mm.group(0), ()):
                hits[c] += 1
        if len(hits) == 1:
            c = next(iter(hits))
            row["content_code"] = c
            row["identity"] = "match" if c == row["code6"] else "mismatch"
            row["method"] = "name_scan"
            row["method2"] = "name_scan"
            updated += 1
            method2["name_scan"] += 1
        elif len(hits) > 1:
            row["method2"] = f"ambiguous:{len(hits)}"
            method2["ambiguous"] += 1
        else:
            row["method2"] = "no_signal"
            method2["no_signal"] += 1
    fields = list(rows[0].keys())
    if "method2" not in fields:
        fields.append("method2")
    with open(IDENT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("更新行数:", updated)
    print("method2:", dict(method2))
    print("identity 分布:", dict(collections.Counter(r["identity"] for r in rows)))
    print("identity_by_year:")
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["year"]][r["identity"]] += 1
    for y in sorted(by):
        print(" ", y, dict(by[y]))


if __name__ == "__main__":
    main()
