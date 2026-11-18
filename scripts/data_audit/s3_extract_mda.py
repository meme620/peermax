"""S3-MB：年报 MD&A（管理层讨论与分析）章节提取（M-B 文本源）。

锚点（按年度格式覆盖）：
  START: "管理层讨论与分析" / "经营情况讨论与分析" / "公司经营情况讨论与分析"
  STOP : "公司治理" / "重要事项" / "监事会报告" / "监事报告"
          / "董事、监事、高级管理人员和员工情况" / "公司治理及内部控制" / "财务报告"
截取：遍历所有 START 出现位置，跳过目录/引用行，取首个正文长度 >= MIN_LEN 的切片；
      无 STOP 时截取到 MAX_LEN 上限。
输出：data/processed/mda_texts.csv（code, year, text, anchor, length, status）
"""
import argparse, csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TXT = ROOT / "data/raw/cninfo/annual_txt"
OUT = ROOT / "data/processed/mda_texts.csv"
MAX_LEN = int(3e4)
MIN_LEN = 500

START_ANCHORS = ["管理层讨论与分析", "经营情况讨论与分析", "公司经营情况讨论与分析",
                 "管理层讨论与展望", "董事会报告之经营情况讨论与分析"]
STOP_TITLES = ["公司治理", "重要事项", "监事会报告", "监事报告",
               "董事、监事、高级管理人员和员工情况", "公司治理及内部控制",
               "财务报告", "备查文件"]

DOT_CHARS = ["....", "…", "……"]


def to_panel_code(num):
    if num.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + num
    if num.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "sz" + num
    return num


def clean(t):
    t = re.sub(r"[ \u3000]+", "", t)
    t = re.sub(r"\.{4,}|…{4,}|。{4,}", "", t)
    t = re.sub(r"\n+", "\n", t)
    return t.strip()


def is_toc_line(line):
    if any(c in line for c in DOT_CHARS):
        return True
    if re.search(r'。"\s*\d+\.?\s*、?', line[:60]):
        return True
    return False


def extract_mda(t):
    for anchor in START_ANCHORS:
        prev = 0
        while True:
            idx = t.find(anchor, prev)
            if idx < 0:
                break
            prev = idx + 1
            line_end = t.find("\n", idx)
            line = t[idx:line_end if line_end > 0 else idx + 200]
            if is_toc_line(line):
                continue
            stops = [t.find(s, idx) for s in STOP_TITLES]
            stops = [s for s in stops if s > 0]
            end = min(stops) if stops else min(len(t), idx + MAX_LEN)
            text = clean(t[idx:end])
            if len(text) >= MIN_LEN:
                return text[:MAX_LEN], anchor
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rows = []
    for f in sorted(TXT.glob("*/*.txt")):
        code, year = f.parent.name, f.stem
        try:
            t = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            rows.append((to_panel_code(code), year, "", "", 0, f"read_error:{e}"))
            continue
        text, anchor = extract_mda(t)
        if text:
            rows.append((to_panel_code(code), year, text, anchor, len(text), "ok"))
        else:
            rows.append((to_panel_code(code), year, "", "", 0, "no_mda_anchor"))
        if args.limit and len(rows) >= args.limit:
            break
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "text", "anchor", "length", "status"])
        w.writerows(rows)
    from collections import Counter
    print(f"输出 {len(rows)} 行 -> {args.out}")
    print("状态分布:", dict(Counter(r[5] for r in rows)))
    print("锚点分布:", dict(Counter(r[3] for r in rows if r[3])))


if __name__ == "__main__":
    main()
