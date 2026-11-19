"""network-mda 前置：MD&A 文本源完整性审计（黄金代码校验 + 跨公司重复 + 跨年错位）。

背景：reports/data_gates/G0/text_closure_report.md（2026-08-16）确认下载管线
内容错位，其对正式 M-B 提取的验收要求为：黄金代码校验（全量、必过项）+ 错位双信号
+ 模板检测。本脚本把这三项落到 MD&A 语料上，输出可比对的机器可读清单。

方法：
  A. 黄金代码校验：年报正文前 3000 字内 "证券代码/公司代码：XXXXXX"（有效 A 股前缀）
     与目录代码比对 -> match / mismatch / unknown；辅以名称法（证券简称）低置信信号。
  B. 跨公司完全重复：文件字节 md5 分组，同组跨不同 code -> 模板/污染组。
  C. 跨年错位（限身份可信文本）：同公司各年 MD&A 文本两两 minhash(bottom-k=128)
     估计 Jaccard，max<0.05 记疑似错位（沿用 text_layer_audit.py 阈值）。
输出：
  artifacts/review/mda_src_identity.csv    逐文件身份（code,year,content_code,method,dup_n,doc_type）
  artifacts/review/mda_dup_groups.csv      重复组明细
  artifacts/review/mda_misalign.csv        疑似错位清单
  reports/data_gates/G0/mda_integrity_audit.json  汇总
用法: python scripts/corpus_mda_integrity_audit.py [--limit N]
"""
import argparse
import csv
import collections
import heapq
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "data/raw/cninfo/annual_txt"
V2 = ROOT / "data/processed/mda_texts.csv"
ART = ROOT / "artifacts/review"
OUTD = ROOT / "reports/data_gates/G0"
A_STOCKS = ROOT / "data/raw/baostock/stock_basic/a_stocks.csv"
NAME_HIST = ROOT / "data/processed/name_history_refined.csv"

VALID_PREFIX = ("000", "001", "002", "003", "300", "301", "302",
                "600", "601", "603", "605", "688", "689")
CODE_RE = re.compile(r"(?:证券代码|股票代码|公司代码|证券简称及代码|股票简称及代码)[：:\s]*([0-9]{6})")
NAME_RE = re.compile(r"(?:证券简称|股票简称|公司简称)[：:\s]*([\u4e00-\u9fa5A-Za-z0-9]{2,12})")
ANN_RE = re.compile(r"公告编号[：:\s]*\d|关于.{0,30}的公告")
# 兼容 "600540、gssh600540" 之类 cninfo 拼接样式
CODE_PAIR_RE = re.compile(r"([0-9]{6})\s*[,、]\s*g[sz][sh][0-9]{7}")

MINHASH_K = 128
STOPLINE = {"页码": None}

csv.field_size_limit(10 ** 9)   # v2 文本无截断，单字段可超 131072 默认上限


def valid_code(c):
    return c is not None and c.startswith(VALID_PREFIX)


def minhash_sig(text, k=MINHASH_K, n=5, cap=40000):
    """bottom-k minhash 签名（进程内 hash，5-gram，取文本前 cap 字符控成本）。"""
    t = re.sub(r"[ \u3000\t\r\n]+", "", text or "")[:cap]
    if len(t) < n:
        return None
    return frozenset(heapq.nsmallest(k, (hash(t[i:i + n]) for i in range(len(t) - n + 1))))


def est_jaccard(a, b):
    if a is None or b is None:
        return 0.0
    inter = len(a & b)
    # bottom-k 估计：|A∩B| / k
    return inter / float(MINHASH_K)


def load_name_map():
    """code -> set(names)（当前简称 + 历史名）。"""
    m = collections.defaultdict(set)
    with open(A_STOCKS, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) > 1 and row[0]:
                m[row[0].split(".")[-1]].add(re.sub(r"[ \u3000]", "", row[1]))
    if NAME_HIST.exists():
        with open(NAME_HIST, encoding="utf-8") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) > 3 and row[0]:
                    m[row[0].split(".")[-1]].add(re.sub(r"[ \u3000]", "", row[3]))
    return m


def check_identity(text, code6, dir_code, name_map):
    """返回 (content_code, method, identity)。identity ∈ match/mismatch/unknown"""
    head = re.sub(r"[ \u3000\t]", "", text[:3000])
    m = CODE_RE.search(head)
    content = m.group(1) if (m and valid_code(m.group(1))) else None
    if content is None:
        code_match = CODE_PAIR_RE.search(head)
        if code_match and valid_code(code_match.group(1)):
            content = code_match.group(1)
    if content is not None:
        return content, "code", ("match" if content == code6 else "mismatch")
    nm = NAME_RE.search(re.sub(r"[ \u3000\t]", "", text[:1500]))
    if nm:
        name = nm.group(1)
        hit = {c for c, names in name_map.items() if name in names}
        if len(hit) == 1:
            c = next(iter(hit))
            return c, "name", ("match" if c == code6 else "mismatch")
    return None, "none", "unknown"


def year_signal(text, year):
    """年报年度一致性（v6，按出现次数）：头部 "{20xx}年年度报告" 计数中目标年缺席或
    被他年压制 -> mismatch:<主导年>；无信号 -> none。

    判据经四轮收敛（2026-09-12）：单次命中会误报"封面残留往年版式"
    （云铝 2022 年报首页含 2021 年报字样，44/49 误报）；弱词"20xx年报"需 >=3 次
    （1-2 次多为正文历史引用）。终态高精度：全量 39,778 只标 2 份真错位
    （002220/2020、002447/2022，退市公司该年确无年报）。
    """
    head = re.sub(r"\s+", "", text[:3000])
    cnt = collections.Counter(m.group(1) for m in re.finditer(r"(20\d\d)年年度报告", head))
    if not cnt:
        cnt = collections.Counter(m.group(1) for m in re.finditer(r"(20\d\d)年报", head))
        cnt = collections.Counter({k: v for k, v in cnt.items() if v >= 3})
    if not cnt:
        return "none"
    my = cnt.get(year, 0)
    others = {k: v for k, v in cnt.items() if k != year}
    if my > 0 and my >= max(others.values(), default=0):
        return "ok"
    return f"mismatch:{max(cnt, key=lambda k: cnt[k])}"


def doc_type(text):
    """年报正文 vs 公告/其他（只看文首，防正文误伤）。"""
    first = re.sub(r"[ \u3000\t\r\n]", "", text[:600])
    head = text[:1500]
    # 公告强信号：保证本公告内容 / 文首"…的公告"
    if "保证本公告内容" in first:
        return "announcement"
    if ANN_RE.search(first[:400]) and not re.search(r"\d{4}年年度报告", first[:60]):
        return "announcement"
    if len(text) < 20000 and "公告编号" in first[:200]:
        return "announcement"
    if re.search(r"\d{4}\s*年年度报告", head):
        return "annual_report"
    if "年度报告" in head:
        return "annual_report?"
    return "other"


def iter_v2_texts():
    """v2 文本（status=ok / src_identity=match / dup_n=1）；缺 v2 直接停，不回退 v1。"""
    if not V2.is_file():
        raise SystemExit("[MDA] 缺 data/processed/mda_texts.csv；先跑 text_mda_extract.py")
    with open(V2, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok" or row["src_identity"] != "match" or row["dup_n"] != "1":
                continue
            code = row["code"]
            yield (code[2:] if code[:2] in ("sz", "sh") else code), row["year"], row["text"]


def misalign_only():
    """只跑跨年错位扫描，不回写身份列（读 v2 产物，status=ok 且 src_identity=match 且 dup_n==1）。"""
    mis = []
    n_scanned = 0
    sig = {}
    for num, year, text in iter_v2_texts():
        s = minhash_sig(text)
        if s:
            sig[(num, year)] = s
        if text:
            n_scanned += 1
    bycode = collections.defaultdict(list)
    for k in sig:
        bycode[k[0]].append(k)
    for code6, keys in bycode.items():
        if len(keys) < 2:
            continue
        for k in keys:
            mx = max((est_jaccard(sig[k], sig[o]) for o in keys if o != k), default=0.0)
            if mx < 0.05:
                mis.append({"code6": code6, "year": k[1], "max_jaccard_other_years": round(mx, 4)})
    with open(ART / "mda_misalign.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code6", "year", "max_jaccard_other_years"])
        w.writeheader()
        w.writerows(mis)
    jp = OUTD / "mda_integrity_audit.json"
    summary = json.loads(jp.read_text(encoding="utf-8")) if jp.exists() else {}
    summary["misalign_suspect"] = len(mis)
    summary["misalign_firms"] = len({m["code6"] for m in mis})
    summary["misalign_scanned_texts"] = n_scanned
    summary["misalign_source"] = "mda_texts.csv"
    jp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"跨年错位: {len(mis)} 条 / {summary['misalign_firms']} 家（扫描 {n_scanned} 篇，源 "
          f"{summary['misalign_source']}）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-misalign", action="store_true",
                    help="跳过跨年错位（minhash 慢）；仅刷新身份/重复清单")
    ap.add_argument("--misalign-only", action="store_true",
                    help="只跑跨年错位（读既有 mda_src_identity.csv，不回写身份列）")
    args = ap.parse_args()
    if args.misalign_only:
        misalign_only()
        return
    ART.mkdir(parents=True, exist_ok=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    name_map = load_name_map()

    files = []
    for d in sorted(os.listdir(TXT)):
        dp = os.path.join(TXT, d)
        if not os.path.isdir(dp):
            continue
        for f in sorted(os.listdir(dp)):
            if f.endswith(".txt"):
                files.append((d, f[:-4]))
    if args.limit:
        files = files[: args.limit]
    print(f"扫描 {len(files)} 个文件", flush=True)

    rows = []
    h2 = collections.defaultdict(list)
    for i, (code6, year) in enumerate(files):
        p = os.path.join(TXT, code6, year + ".txt")
        b = Path(p).read_bytes()
        h2[len(b)].append((code6, year))
        t = b.decode("utf-8", "replace")
        content, method, ident = check_identity(t, code6, code6, name_map)
        rows.append({"code6": code6, "year": year, "content_code": content or "",
                     "method": method, "identity": ident, "doc_type": doc_type(t),
                     "year_signal": year_signal(t, year), "md5": len(b)})
        if (i + 1) % 5000 == 0:
            print(f"  身份校验 {i+1}/{len(files)}", flush=True)

    # B. 跨公司重复组
    dup_rows = []
    dup_n = {}
    groups = 0
    for h, members in h2.items():
        codes = {c for c, _ in members}
        if len(members) > 1 and len(codes) > 1:
            groups += 1
            for c, y in members:
                dup_n[(c, y)] = len(members)
                dup_rows.append({"md5": h[:12], "code6": c, "year": y,
                                 "group_size": len(members), "group_codes": len(codes)})
    for r in rows:
        r["dup_n"] = dup_n.get((r["code6"], r["year"]), 1)

    with open(ART / "mda_src_identity.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code6", "year", "content_code", "method",
                                          "identity", "doc_type", "year_signal", "dup_n", "md5"])
        w.writeheader()
        w.writerows(rows)
    with open(ART / "mda_dup_groups.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["md5", "code6", "year", "group_size", "group_codes"])
        w.writeheader()
        w.writerows(dup_rows)

    # C. 跨年错位（身份 match 且非重复组的文本）
    good = {(r["code6"], r["year"]) for r in rows if r["identity"] == "match" and r["dup_n"] == 1}
    sig = {}
    mis = []
    if not args.skip_misalign:
        for num, year, text in iter_v2_texts():
            if (num, year) in good:
                sig[(num, year)] = minhash_sig(text)
        bycode = collections.defaultdict(list)
        for k in sig:
            bycode[k[0]].append(k)
        for code6, keys in bycode.items():
            if len(keys) < 2:
                continue
            for k in keys:
                mx = max((est_jaccard(sig[k], sig[o]) for o in keys if o != k), default=0.0)
                if mx < 0.05:
                    mis.append({"code6": code6, "year": k[1], "max_jaccard_other_years": round(mx, 4)})
    with open(ART / "mda_misalign.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["code6", "year", "max_jaccard_other_years"])
        w.writeheader()
        w.writerows(mis)

    ys_c = collections.Counter((r["year_signal"] or "none").split(":")[0] for r in rows)
    ident_c = collections.Counter(r["identity"] for r in rows)
    type_c = collections.Counter(r["doc_type"] for r in rows)
    ident_by_year = collections.defaultdict(collections.Counter)
    for r in rows:
        ident_by_year[r["year"]][r["identity"]] += 1
    summary = {
        "files_scanned": len(rows),
        "identity": dict(ident_c),
        "identity_by_year": {y: dict(c) for y, c in sorted(ident_by_year.items())},
        "doc_type": dict(type_c),
        "year_signal": dict(ys_c),
        "dup_groups": groups,
        "dup_files": len(dup_rows),
        "dup_files_identity_match": sum(1 for r in rows if r["dup_n"] > 1 and r["identity"] == "match"),
        "misalign_suspect": len(mis),
        "misalign_firms": len({m["code6"] for m in mis}),
        "note": "黄金代码校验=前3000字证券代码与目录代码比对；跨年错位用 minhash bottom-k=128 估计 Jaccard<0.05（限身份 match 且非重复文件）",
    }
    (OUTD / "mda_integrity_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
