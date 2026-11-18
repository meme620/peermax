"""MD&A 身份确认第三遍：区分真错位与"正文引用他司代码"假阳性。

背景：pass2 的 30k 窗口会在合法年报里命中参股/关联公司代码行（如 TCL科技正文引用
TCL中环 002129），产生假 mismatch。本遍对 identity=mismatch 的行做名称确认：
  正文前 5000 字（封面+公司简介）出现目录码公司名且不含内容码公司名 -> 假阳性（回 match）
  出现内容码公司名且不含目录码公司名 -> 真错位（保持 mismatch）
  其余 -> mismatch_weak（保留文本、产物侧标 suspect_source，不静默剔除）
输出：原地更新 artifacts/review/mda_src_identity.csv（method2=confirm_*）
用法: python scripts/corpus_mda_identity_confirm.py
"""
import csv
import collections
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "data/raw/cninfo/annual_txt"
IDENT = ROOT / "artifacts/review/mda_src_identity.csv"
A_STOCKS = ROOT / "data/raw/baostock/stock_basic/a_stocks.csv"
NAME_HIST = ROOT / "data/processed/name_history_refined.csv"
HEAD_CHARS = 5000


def code6_of(raw):
    """兼容 'sh.600000'（baostock）与 'sz000001'（面板）两种代码串。"""
    if "." in raw:
        return raw.split(".")[-1]
    return raw[2:] if raw[:2] in ("sz", "sh") else raw


def load_names():
    m = collections.defaultdict(set)
    with open(A_STOCKS, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 1 and row[0].startswith(("sh.", "sz.")):
                n = re.sub(r"[ \u3000]", "", row[1])
                if len(n) >= 3:
                    m[code6_of(row[0])].add(n)
    if NAME_HIST.exists():
        with open(NAME_HIST, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) > 3 and row[0]:
                    n = re.sub(r"[ \u3000]", "", row[3])
                    if len(n) >= 3:
                        m[code6_of(row[0])].add(n)
    return m


def main():
    names = load_names()
    rows = list(csv.DictReader(open(IDENT, encoding="utf-8")))
    stat = collections.Counter()
    for row in rows:
        if row["identity"] != "mismatch":
            continue
        p = TXT / row["code6"] / f"{row['year']}.txt"
        if not p.exists():
            stat["file_missing"] += 1
            continue
        head = re.sub(r"[ \u3000\t\r\n]", "", p.read_text(encoding="utf-8", errors="replace")[:HEAD_CHARS])
        own = names.get(row["code6"], set())
        other = names.get(row["content_code"] or "", set())
        own_hit = any(n in head for n in own)
        other_hit = any(n in head for n in other)
        if own_hit and not other_hit:
            row["identity"] = "match"
            row["method2"] = "confirm_own"
            stat["reverted_false_positive"] += 1
        elif other_hit and not own_hit:
            row["method2"] = "confirm_other"
            stat["confirmed_mismatch"] += 1
        else:
            row["identity"] = "mismatch_weak"
            row["method2"] = "confirm_ambiguous"
            stat["weak"] += 1
    fields = list(rows[0].keys())
    with open(IDENT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("确认结果:", dict(stat))
    print("identity 分布:", dict(collections.Counter(r["identity"] for r in rows)))


if __name__ == "__main__":
    main()
