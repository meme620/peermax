#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3-2 v7：年报业务章节提取（公司业务概要/主营业务）——去 900 截断 + 审计列。

相对 v6 的变更（2026-09-13，P0.2）：
  1. **移除 900 字硬截断**：改为章节边界截取（STOP 标题），安全上限 3e4；
     v6 下 88.4% 的文本被截在 900 字，产品词召回受损。
  2. **新增审计列**：src_identity / content_code / dup_n / year_signal / status / reason，
     复用 M-B 的身份判定（artifacts/review/mda_src_identity.csv），与 M-B 可交叉核验。
  3. **身份门**：mismatch / announcement 源 → status=bad_source 且文本置空，
     不得进入 M-A 网络（v6 无此门）。
  4. 表格误杀判据保留；文档级去重保留（模板文本剔除）。

锚点优先序（沿用 v6）：
  1. 报告期内公司从事的主要业务（深交所格式）
  2. 主要业务（跳过目录行）
  3. 公司业务概要 / 业务概要（2020 前格式）

输出：data/processed/business_texts.csv
  code, year, text, anchor, length, status, extract_format, src_identity,
  content_code, dup_n, year_signal, reason
用法: python scripts/text_business_extract.py [--out PATH] [--limit N]
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TXT = ROOT / "data/raw/cninfo/annual_txt"
IDENT = ROOT / "artifacts/review/mda_src_identity.csv"
OUT = ROOT / "data/processed/business_texts.csv"

SAFE_MAX = int(3e4)          # 安全上限（防异常长文）
MIN_LEN = 40                 # 最小可信长度（2026-09-13 由 100 降至 40：部分银行年报业务章节仅一句话）
MAX_OCC = 12                 # 每个锚点最多尝试的出现位置数（2026-09-13 v7.1）

# 锚点优先序（2026-09-13 v7.8 补新版格式）：
#   新版年报（2021+ 证监会格式）用「报告期内公司从事的业务情况」；
#   旧版用「报告期内公司从事的主要业务」/「主要业务」/「公司业务概要」。
ANCHORS = ["报告期内公司从事的主要业务", "报告期内公司从事的业务情况",
           "报告期内本行从事的主要业务", "公司从事的业务情况",
           "公司业务发展情况", "主要业务", "公司业务概要", "业务概要"]
STOP_TITLES = ["经营情况讨论与分析", "主营业务分析", "报告期内公司经营", "公司未来发展的展望",
               "报告期内主要经营情况", "公司经营情况", "核心竞争力", "公司未来发展战略",
               "重要事项", "公司治理", "财务报告", "董事会报告"]

CID_BAD_RE = re.compile(
    r'[\u0a00-\u0aff\u0b00-\u0b7f\u0c00-\u0c7f\u0d00-\u0d7f\u0e00-\u0e7f'
    r'\u1100-\u11ff\u1780-\u17ff\ua960-\ua97f\uac00-\ud7af\ue000-\uf8ff'
    r'\ufff0-\uffff\u2e80-\u2eff]')


def to_panel_code(num_code):
    if num_code.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh" + num_code
    if num_code.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "sz" + num_code
    return num_code


def is_table_text(text):
    """表格/目录误命中判据。

    v6 用绝对计数（子公司>=8 或 有限公司>=10）是因为它先截到 900 字再判断；
    v7 去掉了 900 截断，绝对计数在长章节里必然触发（正常业务章节也会多次提到
    子公司/有限公司），故改为**密度 + 短文本**双条件：
      - 短文本（<600 字）保留绝对计数（同 v6 场景）；
      - 长文本改用密度阈值，只有整段被表格吃掉才判为表格。
    """
    n = len(text)
    if n == 0:
        return True
    if n < 600:
        return text.count("子公司") >= 8 or text.count("有限公司") >= 10
    return (text.count("子公司") + text.count("有限公司")) / n > 0.06


def load_identity():
    ident = {}
    if not IDENT.exists():
        return ident
    with open(IDENT, encoding="utf-8") as f:
        for d in csv.DictReader(f):
            ident[(d["code6"], d["year"])] = d
    return ident


def extract_business(t):
    """返回 (text, anchor, fmt) 或 (None, None, fmt)。

    2026-09-13 v7.1：每个锚点扫描**全部出现位置**（上限 MAX_OCC 次），
    逐个尝试直至通过判据。此前只试首次出现，命中表格列头被拒后即放弃，
    会漏掉后文真正的业务章节（如工行"12. 董事会报告 主要业务"）。
    """
    fmt = None
    for anchor in ANCHORS:
        positions = []
        start = 0
        while True:
            i = t.find(anchor, start)
            if i < 0 or len(positions) >= MAX_OCC:
                break
            positions.append(i)
            start = i + len(anchor)
        if not positions:
            continue
        if fmt is None:
            fmt = "new" if anchor == "报告期内公司从事的主要业务" else "old"
        for idx in positions:
            r = _try_anchor(t, anchor, idx)
            if r is not None:
                return r[0], anchor, fmt
    return None, None, fmt


def _try_anchor(t, anchor, idx):
    """尝试在给定位置按锚点提取；不通过判据返回 None（由外层试下一候选）。"""
    seg_start = idx
    # 标题判据（2026-09-13 v7.2）：锚点须位于行首（前导仅编号/空白），
    # 或整行基本等于锚点。正文/表格中出现的锚点（如"本行及本行子公司的主要业务"）排除。
    _ls = t.rfind(chr(10), 0, idx) + 1
    _pre = t[_ls:idx]
    _le = t.find(chr(10), idx)
    _line = t[_ls:_le if _le > 0 else idx + 120]
    _lead = _pre.strip()
    _ok_head = (len(_lead) <= 6 and not _lead.endswith(('的', '和', '及', '，', '。')))
    if not _ok_head:
        return None
    # 目录行判据（v7.4，2026-09-13）：若锚点所在行本身是目录行（点线 / 锚点后仅页码 /
    # 锚点后紧跟其它章节名），直接判该候选无效——此前"跳过目录行再提取"会从目录中段继续，
    # 产出目录残段（实测 1,210 例）。
    _le0 = t.find(chr(10), idx)
    _line0 = t[idx:_le0 if _le0 > 0 else idx + 200]
    if "...." in _line0 or "…" in _line0:
        return None
    _after0 = _line0[len(anchor):].strip()
    if _after0 and __import__("re").match(r"^[0-9\s.]+$", _after0):
        return None
    _NEXT_SEC = ("第四节", "第五节", "第六节",
                 "第七节", "第八节", "第九节",
                 "第十节", "重要事项", "公司治理",
                 "股份变动", "财务报告")
    if sum(_after0.count(s) for s in _NEXT_SEC) >= 1 and len(_after0) < 60:
        return None
    end_candidates = [t.find(st, seg_start + 20) for st in STOP_TITLES
                      if t.find(st, seg_start + 20) > 0]
    end = min(end_candidates) if end_candidates else seg_start + SAFE_MAX
    text = t[seg_start:end]
    text = re.sub(r"[ 	　]+", "", text)
    text = re.sub(r"\.{4,}", " ", text)
    text = re.sub(r"[\r\n]+", " ", text)
    text = text.strip()
    if len(text) < MIN_LEN:
        return None
    head = text[len(anchor):len(anchor) + 40]
    if head:
        col_names = ("注册资本", "总资产", "净资产",
                     "营业收入", "净利润", "营业利润",
                     "项目", "指标", "本期", "上年同期",
                     "变化比例", "总股本", "资产总额")
        if sum(head.count(c) for c in col_names) >= 2:
            return None
        if sum(ch.isdigit() for ch in head) / len(head) > 0.25:
            return None
        # 投资表列头判据（v7.7，2026-09-13）：锚点常作为"对外投资情况"表的列头出现
        # （主要业务 | 投资方式 | 投资金额 | 持股比例 …），此类候选须排除。
        _COLS = ("投资方式", "投资金额", "持股比例", "资金", "合作方", "投资期限",
                 "产品类型", "进展情况", "投资盈亏", "披露日期", "披露索引")
        if sum(head.count(c) for c in _COLS) >= 2:
            return None
    if len(text) > SAFE_MAX:
        cut = text.rfind("。", 0, SAFE_MAX)
        text = text[:cut + 1] if cut > MIN_LEN else text[:SAFE_MAX]
    if is_table_text(text):
        return None
    return (text, anchor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ident = load_identity()
    files = []
    for d in sorted(TXT.iterdir()):
        if d.is_dir():
            for f in sorted(d.glob("*.txt")):
                files.append((d.name, f.stem))
    if args.limit:
        files = files[: args.limit]
    print(f"提取 {len(files)} 个文件", flush=True)

    rows = []
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

        if len(t) < 500:
            rows.append((to_panel_code(code6), year, "", "", 0, "too_short", "", src_identity,
                         content_code, dup_n, year_signal, "source_lt_500"))
            continue

        text, anchor, fmt = extract_business(t)
        if not text:
            rows.append((to_panel_code(code6), year, "", "", 0, "no_anchor", fmt or "",
                         src_identity, content_code, dup_n, year_signal, "no_business_anchor"))
            continue

        status, reason = "ok", ""
        if src_identity == "mismatch" or doc_type == "announcement":
            status, reason = "bad_source", f"identity:{src_identity};doc:{doc_type}"
            text = ""
        elif str(year_signal).startswith("mismatch"):
            status, reason = "suspect_source", f"year_signal:{year_signal}"
        elif src_identity == "mismatch_weak":
            status, reason = "suspect_source", "identity:mismatch_weak"
        # CID 字形乱码隔离（同 M-B 判据）
        if text and len(CID_BAD_RE.findall(text)) / len(text) > 0.005:
            status, reason = "bad_source", (reason + ";" if reason else "") + "cid_glyph_corruption"
            text = ""
        rows.append((to_panel_code(code6), year, text, anchor, len(text), status, fmt or "",
                     src_identity, content_code, dup_n, year_signal, reason))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    # 文档级去重：同 (year, text) 完全相同视为模板/误提取，保留首现
    seen, dedup, removed = {}, [], 0
    for r in rows:
        if r[5] != "ok" or not r[2]:
            dedup.append(r); continue
        key = (r[1], r[2])
        if key in seen:
            # 2026-09-13：模板重复行**标记**而非删除，保持与语料一一对应（契约要求 39,778 行）
            removed += 1
            r = (r[0], r[1], "", r[3], 0, "duplicate_text", r[6], r[7], r[8], r[9], r[10],
                 "identical_to:" + str(seen[key]))
            dedup.append(r)
            continue
        seen[key] = r[0]
        dedup.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "year", "text", "anchor", "length", "status",
                    "extract_format", "src_identity", "content_code", "dup_n",
                    "year_signal", "reason"])
        w.writerows(dedup)
    st = Counter(r[5] for r in dedup)
    print(f"输出 {len(dedup)} 行 -> {out}")
    print("status:", dict(st), "| 去重剔除:", removed)
    print("anchor:", dict(Counter(r[3] for r in dedup if r[3])))
    ok = sorted(r[4] for r in dedup if r[5] == "ok" and r[4])
    if ok:
        print(f"长度 p10={ok[len(ok)//10]} p50={ok[len(ok)//2]} p90={ok[len(ok)*9//10]} max={max(ok)}"
              f" | =900 占比 {sum(1 for x in ok if x==900)/len(ok)*100:.1f}%")


if __name__ == "__main__":
    main()
