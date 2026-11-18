"""MD&A v2 文本级跨公司近重复扫描（区别于源文件 md5 重复：这里比对提取后的正文）。

方法：5-gram 字符 shingle + bottom-k(128) minhash，按公司分组两两估计 Jaccard。
输出疑似同文（>=0.95 高置信 / 0.80-0.95 待查），供模板/抄袭/污染复核。
输出：artifacts/review/mda_text_dups.csv、reports/data_gates/G0/mda_text_dup_summary.json
用法: python scripts/data_audit/s3_mda_text_dups.py
"""
import csv
import collections
import hashlib
import heapq
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data/processed/mda_texts_v2.csv"
ART = ROOT / "artifacts/review"
OUTD = ROOT / "reports/data_gates/G0"
K = 128
CAP = 30000


def sig(text):
    t = re.sub(r"\s+", "", text)[:CAP]
    if len(t) < 6:
        return None
    hs = heapq.nsmallest(K, (hash(t[i:i + 5]) for i in range(len(t) - 4)))
    return frozenset(hs)


def main():
    csv.field_size_limit(10 ** 9)
    per_year = collections.defaultdict(list)
    with open(V2, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            s = sig(row["text"])
            if s:
                per_year[row["year"]].append((row["code"], s))
    pairs = []
    for year, items in per_year.items():
        n = len(items)
        for i in range(n):
            ci, si = items[i]
            for j in range(i + 1, n):
                cj, sj = items[j]
                if ci == cj:
                    continue
                inter = len(si & sj)
                if inter == 0:
                    continue
                est = inter / K
                if est >= 0.80:
                    pairs.append({"year": year, "code_a": ci, "code_b": cj, "est_jaccard": round(est, 3)})
        print(f"  {year}: {n} 篇扫描完，累计疑似 {len(pairs)} 对", flush=True)
    pairs.sort(key=lambda x: -x["est_jaccard"])
    with open(ART / "mda_text_dups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "code_a", "code_b", "est_jaccard"])
        w.writeheader()
        w.writerows(pairs)
    high = [p for p in pairs if p["est_jaccard"] >= 0.95]
    summary = {"ok_texts_scanned": sum(len(v) for v in per_year.values()),
               "dup_pairs_ge_0.80": len(pairs), "dup_pairs_ge_0.95": len(high),
               "firms_in_high_pairs": len({p["code_a"] for p in high} | {p["code_b"] for p in high})}
    (OUTD / "mda_text_dup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
