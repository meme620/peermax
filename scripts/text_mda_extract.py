"""network-mda v2：年报 MD&A 章节提取（修复 v1 的 STOP 半句误切 + 去 30k 截断 + 来源身份列）。

v1 提取器审计发现的三处缺陷（2026-09-11，见 reports/data_gates/G0/mda_integrity_audit.json）：
  1. STOP 标题用裸 str.find，正文半句命中即截断（"为规范公司治理，"）→ 262 条锚点在却提取失败；
  2. MAX_LEN=3e4 硬截断 → 17.6% 文本被腰斩，M-B TF-IDF 词频视野受损；
  3. 无来源身份列 → 下载管线错位文本（text_closure_report.md）无法在产物层剔除。

v2 规则：
  - 标题必须独占一行：去空白后长度 <= 40 且无点线（目录行），其后 300 字内须有正文；
  - START 阶梯：新格式章节标题 → 老格式"董事会报告"（extract_format=old）→ 章内小节标题（=weak）；
  - STOP：同级章节标题行（第X节 标记 / 具名章节名），不再做半句匹配；
  - 不截断（仅 30 万字符安全上限）；
  - 合并 artifacts/review/mda_src_identity.csv：
      identity=mismatch 或 doc_type=announcement -> status=bad_source（text 置空）；
      dup_n>1 且无身份信号 -> status=suspect_source（保留文本待复核）。
输出：data/processed/mda_texts.csv
  code,year,text,anchor,length,status,extract_format,src_identity,content_code,dup_n,year_signal,reason
用法：python scripts/text_mda_extract.py [--limit N] [--out PATH]
"""
import argparse
import collections
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "data/raw/cninfo/annual_txt"
OUT = ROOT / "data/processed/mda_texts.csv"
IDENT = ROOT / "artifacts/review/mda_src_identity.csv"

MIN_LEN = 500
MIN_SLICE = 1000   # 最小可信切片：短于此的候选视为页眉/引用噪声（2026-09-12）
MAX_LEN = int(3e5)
HEAD_MAX = 40          # 标题行去空白后最大长度
CONTENT_LOOKAHEAD = 300  # 标题后需出现正文的窗口

START_NEW_RE = re.compile(
    r"(?:第[一二三四五六七八九十]+[章节])?"
    r"(?:董事会报告及)?(?:管理层讨论与分析|经营情况讨论与分析|经营情况讨论及分析|"
    r"公司经营情况讨论与分析|管理层讨论与展望|经营层讨论与分析|"
    r"董事会报告之经营情况讨论与分析)")
START_OLD_RE = re.compile(r"(?:第[一二三四五六七八九十]+[章节])?董事会报告")
START_WEAK_RE = re.compile(r"(?:[一二三四五六七八九十]+、)?(?:经营情况讨论与分析|报告期内经营情况回顾|主营业务分析)")

STOP_NAMES = {"公司治理", "重要事项", "监事会报告", "监事报告",
              "董事、监事、高级管理人员和员工情况", "公司治理及内部控制",
              "财务报告", "备查文件", "环境和社会责任", "社会责任报告",
              "股份变动及股东情况", "审计报告", "独立董事述职报告",
              "内部控制评价报告", "内部控制自我评价报告",
              "募集资金存放与使用情况", "优先股相关情况", "债券相关情况",
              "董事会报告"}
STOP_CAND_RE = re.compile(
    r"^[ \u3000]*(?:第[一二三四五六七八九十]+[章节]|公司治理|重要事项|监事会报告|监事报告|"
    r"董事、监事|财务报告|备查文件|环境和社会责任|社会责任报告|股份变动|审计报告|"
    r"董事会报告|独立董事|内部控制|募集资金|优先股|债券相关情况|利润分配|员工情况)",
    re.M)
CHAPTER_MARK_RE = re.compile(r"^第[一二三四五六七八九十]+[章节]$")


def to_panel_code(num):
    if num.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + num
    if num.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "sz" + num
    return num


def norm_line(line):
    # \s 覆盖半角/全角/NBSP(\xa0) 等全部空白（2026-09-11：'第四节\xa0经营情况讨论与分析' 实测）
    return re.sub(r"\s+", "", line).strip()


def line_bounds(text, idx):
    s = text.rfind("\n", 0, idx) + 1
    e = text.find("\n", idx)
    return s, (e if e > 0 else len(text))


def is_toc_line(raw_line):
    return (".." in raw_line) or ("…" in raw_line) or (len(norm_line(raw_line)) == 0)


def chapter_marker_counts(text):
    """统计"独占一行的短标题"出现次数：章节标记（第X节）或 STOP_NAMES 具名标题。

    A+H 年报每页有页眉（单独一行"第四节"/"财务报告" + 页码），若当章节边界会把正文
    切成几百字（实测 300589/601598/601628/601899）。判据：同一标题全文 >= 3 次 -> 页眉。
    """
    c = collections.Counter()
    for m in re.finditer(r"^[ \u3000]*([^\n]{2,12})[ \u3000]*$", text, re.M):
        raw = norm_line(m.group(1))
        core = raw.rstrip("0123456789")
        if re.fullmatch(r"第[一二三四五六七八九十]+[章节]", core) or core in STOP_NAMES:
            c[core] += 1
    return c


def is_running_header(counts, s, e, text):
    n = norm_line(text[s:e])
    core = re.sub(r"[\s]+", "", n).rstrip("0123456789")
    return counts.get(core, 0) >= 3


def heading_ok(text, idx, pat, counts=None, exclude_norm=None):
    """候选标题：独占一行、去空白后匹配 pat、非目录、非页眉、其后有正文。"""
    s, e = line_bounds(text, idx)
    raw = text[s:e]
    if is_toc_line(raw) or (counts and is_running_header(counts, s, e, text)):
        return False
    norm = norm_line(raw)
    if not norm or len(norm) > HEAD_MAX:
        return False
    norm = re.sub(r"[（(][^）)]{0,10}[）)]$", "", norm)   # 容 '（董事会报告）' 等尾部括注
    if not pat.fullmatch(norm.rstrip("0123456789")):
        return False
    if exclude_norm and norm.rstrip("0123456789") in exclude_norm:
        return False
    # 目录页识别（2026-09-12）：候选紧邻区（后 150 / 前 250 字）出现
    #   ① 全数字短行（目录页码，如 "162"）>= 3，或 ② 点线行 >= 2
    # 财务表格数字多为长数值（-4,303,058.19），不满足 ^\d{1,4}$，故不误伤正文。
    nxt, prv = text[e:e + 150], text[max(0, s - 250):s]
    nums = len(re.findall(r"^[ \u3000]*[0-9]{1,4}[ \u3000]*$", nxt, re.M)) + \
        len(re.findall(r"^[ \u3000]*[0-9]{1,4}[ \u3000]*$", prv, re.M))
    dots = (nxt + prv).count("....") + (nxt + prv).count("…")
    if nums >= 8 or dots >= 4:
        return False
    tail = text[e:e + CONTENT_LOOKAHEAD]
    han = len(re.findall(r"[\u4e00-\u9fa5]", tail))
    if han >= 100:
        return True
    # 表格容忍：标题后紧跟财务表格时汉字少（如 000553/2019 仅 61 字），
    # 但若近邻 200 字内出现另一个章节标记，则是无点线的目录条目 -> 仍拒绝
    if han >= 40 and not re.search(r"第[一二三四五六七八九十]+[章节]", text[e:e + 200]):
        return True
    return False


def find_stop(text, start, counts=None, exclude_norm=None, min_pos=None):
    """start 之后首个**有效**同级章节标题行；页眉重复与过近的标记跳过。

    min_pos：允许的最小停止位置（= start + MIN_SLICE），用于跳过页眉/正文引用造成的误切。
    """
    pos = start
    while True:
        m = STOP_CAND_RE.search(text, pos)
        if not m:
            return None
        s, e = line_bounds(text, m.start())
        raw = text[s:e]
        norm = norm_line(raw)
        core = norm.rstrip("0123456789")
        ok = False
        if (len(norm) <= HEAD_MAX and not is_toc_line(raw)
                and not (counts and is_running_header(counts, s, e, text))):
            if min_pos is not None and s < min_pos:
                ok = False
            elif CHAPTER_MARK_RE.fullmatch(core):
                ok = core not in (exclude_norm or set())
            else:
                body = re.sub(r"^第[一二三四五六七八九十]+[章节]", "", core)
                ok = body in STOP_NAMES and body not in (exclude_norm or set())
        if ok:
            return s
        pos = m.start() + 1


def clean(t):
    t = re.sub(r"[^\S\n]+", "", t)   # 去空白（含 NBSP/全角空格）但保留换行结构
    t = re.sub(r"\.{4,}|…{4,}|。{4,}", "", t)
    # 去 PDF 字形残渣（2026-09-12）：PUA 私用区（Wingdings √/□ 占位，11,486 行）、
    # 控制字符（√ 被编码为 \x00 等，68 行）、替换符 U+FFFD（1 行）——不剥离会在分词时变垃圾词
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ue000-\uf8ff\ufffd]", "", t)
    t = re.sub(r"\n+", "\n", t)
    return t.strip()


def extract_mda(t):
    """返回 (text, anchor, fmt) 或 (None, None, reason)。

    多个候选标题时取**最长**切片（2026-09-12）：页眉/正文引用会制造短切片噪声，
    最长切片对应真正的章节正文。
    """
    best = None
    counts = chapter_marker_counts(t)
    for pat, fmt in ((START_NEW_RE, "new"), (START_OLD_RE, "old")):
        exclude = {"董事会报告"} if fmt == "new" else set()
        for m in pat.finditer(t):
            if not heading_ok(t, m.start(), pat, counts):
                continue
            s, e = line_bounds(t, m.start())
            stop = find_stop(t, e, counts=counts, exclude_norm=exclude, min_pos=s + MIN_SLICE)
            text = clean(t[s:stop if stop else min(len(t), s + MAX_LEN)])
            if len(text) >= MIN_LEN and (best is None or len(text) > len(best[0])):
                best = (text[:MAX_LEN], norm_line(t[s:e]), fmt)
    if best is not None:
        return best
    for m in START_WEAK_RE.finditer(t):
        if not heading_ok(t, m.start(), START_WEAK_RE, counts):
            continue
        s, e = line_bounds(t, m.start())
        stop = find_stop(t, e, counts=counts, min_pos=s + MIN_SLICE)
        text = clean(t[s:stop if stop else min(len(t), s + MAX_LEN)])
        if len(text) >= MIN_LEN and (best is None or len(text) > len(best[0])):
            best = (text[:MAX_LEN], norm_line(t[s:e]), "weak")
    return best if best is not None else (None, None, "no_mda_anchor")


def load_identity():
    ident = {}
    if not IDENT.exists():
        return ident
    with open(IDENT, encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            ident[(row["code6"], row["year"])] = row
    return ident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    ident = load_identity()
    rows = []
    files = []
    for d in sorted(TXT.iterdir()):
        if d.is_dir():
            for f in sorted(d.glob("*.txt")):
                files.append((d.name, f.stem))
    if args.limit:
        files = files[: args.limit]
    print(f"提取 {len(files)} 个文件", flush=True)
    for i, (code6, year) in enumerate(files):
        p = TXT / code6 / f"{year}.txt"
        try:
            t = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            rows.append((to_panel_code(code6), year, "", "", 0, "read_error", "", "", "", 1, "", str(exc)))
            continue
        meta = ident.get((code6, year), {})
        src_identity = meta.get("identity", "unknown")
        content_code = meta.get("content_code", "")
        dup_n = meta.get("dup_n", "1")
        doc_type = meta.get("doc_type", "")
        year_signal = meta.get("year_signal", "")
        text, anchor, fmt = extract_mda(t)
        if not text:
            rows.append((to_panel_code(code6), year, "", "", 0, "no_mda_anchor", fmt, src_identity,
                         content_code, dup_n, year_signal,
                         "alt_format" if doc_type != "announcement" else "announcement"))
            continue
        status, reason = "ok", ""
        if src_identity == "mismatch" or doc_type == "announcement":
            status, reason = "bad_source", f"identity:{src_identity};doc:{doc_type}"
            text = ""
        elif str(year_signal).startswith("mismatch"):
            status, reason = "suspect_source", f"year_signal:{year_signal}（源文件年份错位，待重下）"
        elif src_identity == "mismatch_weak":
            status, reason = "suspect_source", "identity:mismatch_weak（名称确认歧义，保留文本待复核）"
        elif int(dup_n or 1) > 1 and src_identity == "unknown":
            status, reason = "suspect_source", f"dup_n:{dup_n};identity:unknown"
        rows.append((to_panel_code(code6), year, text, anchor, len(text), status, fmt,
                     src_identity, content_code, dup_n, year_signal, reason))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "text", "anchor", "length", "status",
                    "extract_format", "src_identity", "content_code", "dup_n",
                    "year_signal", "reason"])
        w.writerows(rows)
    print(f"输出 {len(rows)} 行 -> {args.out}")
    print("status:", dict(Counter(r[5] for r in rows)))
    print("format:", dict(Counter(r[6] for r in rows)))
    print("anchor:", dict(Counter(r[3] for r in rows if r[3])))


if __name__ == "__main__":
    main()
