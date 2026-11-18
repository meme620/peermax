"""生成 MD&A 年报源重下清单（供 fetch_cninfo_annual.py --pairs）。

入队条件（任一）：
  1. identity == mismatch                      —— 正文代码与目录代码不符
  2. doc_type in (announcement, other) 且非 match —— 源文件不是年报正文
  3. 文件 < 8000 字节                            —— 残文/空壳（PDF 提取失败或错误页）
  4. identity == unknown 且 dup_n > 1            —— 与他人字节相同但无身份信号
输出：artifacts/review/mda_redownload_pairs.csv（code,year,reason）
用法: python scripts/corpus_mda_redownload_queue.py
"""
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "data/raw/cninfo/annual_txt"
IDENT = ROOT / "artifacts/review/mda_src_identity.csv"
OUT = ROOT / "artifacts/review/mda_redownload_pairs.csv"
MIN_BYTES = 8000


def main():
    rows = list(csv.DictReader(open(IDENT, encoding="utf-8")))
    queue = []
    reason_c = Counter()
    for r in rows:
        code6, year = r["code6"], r["year"]
        p = TXT / code6 / f"{year}.txt"
        reasons = []
        if r["identity"] == "mismatch":
            reasons.append(f"identity_mismatch:{r['content_code']}")
        if r["doc_type"] in ("announcement", "other") and r["identity"] != "match":
            reasons.append(f"doc_type:{r['doc_type']}")
        if r["doc_type"] == "announcement" and r["identity"] == "match":
            # 公告正文里出现本公司代码会被黄金校验判 match，但源文件不是年报 -> 仍需重下
            reasons.append(f"doc_type:{r['doc_type']}")
        if p.exists() and p.stat().st_size < MIN_BYTES:
            reasons.append(f"tiny:{p.stat().st_size}B")
        if r["identity"] == "unknown" and int(r.get("dup_n") or 1) > 1:
            reasons.append(f"dup_unknown:n={r['dup_n']}")
        if reasons:
            queue.append({"code": code6, "year": year, "reason": ";".join(reasons)})
            for x in reasons:
                reason_c[x.split(":")[0]] += 1
    queue.sort(key=lambda x: (x["code"], x["year"]))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code", "year", "reason"])
        w.writeheader()
        w.writerows(queue)
    print(f"重下清单 {len(queue)} 条 -> {OUT}")
    print("原因分布:", dict(reason_c))
    by_year = Counter(q["year"] for q in queue)
    print("按年:", dict(sorted(by_year.items())))


if __name__ == "__main__":
    main()
