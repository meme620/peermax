"""s3_network_prospectus_anchor 的 text 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.prospectus.consts import BLACK
from lib.prospectus.consts import CJK_CH
from lib.prospectus.consts import CONN_SUB
from lib.prospectus.consts import DRIFT_EXPLICIT
from lib.prospectus.consts import FRAG_EXPLICIT
from lib.prospectus.consts import FRAG_WORD
from lib.prospectus.consts import KWS
from lib.prospectus.consts import KW_RE
from lib.prospectus.consts import NAME_START_BAD
from lib.prospectus.consts import NAME_TAIL
from lib.prospectus.consts import NOISE_CHAR
from lib.prospectus.consts import NOISE_SUB
from lib.prospectus.consts import NOISE_TAIL
from lib.prospectus.consts import NOISE_TOKEN
from lib.prospectus.consts import NOISE_WEAK
from lib.prospectus.consts import NON_ENTITY_WORD
from lib.prospectus.consts import PLACE_PREFIX
from lib.prospectus.consts import R1
from lib.prospectus.consts import R2
from lib.prospectus.consts import R3
from lib.prospectus.consts import R4
from lib.prospectus.consts import R4_PEER_TRIG
from lib.prospectus.consts import R4_SEP
from lib.prospectus.consts import R5_NAME
from lib.prospectus.consts import R5_NUM
from lib.prospectus.consts import REJECT_LOG
from lib.prospectus.consts import REPAIR_LOG
from lib.prospectus.consts import RESIDUAL_CLS
from lib.prospectus.consts import RESIDUAL_EV
from lib.prospectus.consts import ROOT
from lib.prospectus.consts import TBL_HDR_TOK
from lib.prospectus.consts import TBL_HEAD
from lib.prospectus.consts import TBL_ROW
from lib.prospectus.consts import WIN_POST
from lib.prospectus.consts import WIN_PRE
from lib.prospectus.consts import _CID_CACHE
from lib.prospectus.consts import _GID2UNI
from lib.prospectus.consts import _OCR_CHUNK
from lib.prospectus.consts import _OCR_DPI
from lib.prospectus.consts import _OCR_LANG
from lib.prospectus.consts import _OCR_MAX_CHUNKS
from lib.prospectus.consts import _OCR_PSM
from lib.prospectus.consts import _OCR_SH
from lib.prospectus.consts import _SIMSUN
from lib.prospectus.consts import _WSL_EXE
from lib.prospectus.consts import _WSL_USER
from lib.prospectus.names import base_name
from lib.prospectus.names import is_generic_name
from lib.prospectus.names import is_self_name
from lib.prospectus.names import norm
from lib.prospectus.names import r3_shape_ok
from lib.prospectus.names import r4_member_ok
from lib.prospectus.names import r4_peer_list_ok
from lib.prospectus.names import r4_strip_head
from lib.prospectus.names import repair_r2_left
from lib.prospectus.names import strip_st
import os
import pymupdf
import re
import shutil
import subprocess
import time

def _gid2uni():
    """SimSun（宋体）GID → Unicode 反查表（修 3 备选通道用；子集字体无 ToUnicode 时，
    Identity-H 编码的 2 字节码 = GID，可经同版本宋体反查回真字符）。"""
    global _GID2UNI
    if _GID2UNI is None:
        from fontTools.ttLib import TTCollection
        f = TTCollection(_SIMSUN, lazy=True).fonts[0]
        rev = {}
        for u, g in sorted(f.getBestCmap().items()):
            gid = f.getGlyphID(g)
            if gid not in rev or u < rev[gid]:
                rev[gid] = u
        _GID2UNI = rev
    return _GID2UNI

def cid_recover_text(code, pdf_path):
    """修 3 备选提取通道：pdfminer.six 抽取 + `(cid:N)` → SimSun GID → Unicode 解码。

    适用对象 = 文本层 CID 编码损坏（无 ToUnicode CMap 的子集字体，pymupdf 读出 `chr(cid)` 乱码、
    「公司/发行人/万元」计数 = 0）。解码为**确定性字符还原**（非 OCR）。缓存
    artifacts/tmp/tier2_prosp/cid_recover/<code>.txt（键 = PDF 字节数），命中即复用。
    """
    os.makedirs(_CID_CACHE, exist_ok=True)
    cache = os.path.join(_CID_CACHE, f"{code}.txt")
    key = os.path.join(_CID_CACHE, f"{code}.key")
    size = str(os.path.getsize(pdf_path))
    if os.path.exists(cache) and os.path.exists(key) and open(key).read().strip() == size:
        return open(cache, encoding="utf-8").read(), "pdfminer+cid_gid（缓存命中）"
    import pdfminer.high_level
    rev = _gid2uni()
    raw = pdfminer.high_level.extract_text(pdf_path)
    n_unmapped = [0]

    def sub(m):
        cid = int(m.group(1))
        if cid < 0x80:
            return chr(cid)
        u = rev.get(cid)
        if u is None:
            n_unmapped[0] += 1
            return "\ufffd"
        return chr(u)

    dec = re.sub(r"\(cid:(\d+)\)", sub, raw)
    with open(cache, "w", encoding="utf-8") as f:
        f.write(dec)
    with open(key, "w") as f:
        f.write(size)
    return dec, f"pdfminer+cid_gid（未映射 CID {n_unmapped[0]}）"

def _wsl_path(p):
    """Windows 绝对路径 → WSL /mnt/<drive>/... 路径。"""
    p = os.path.abspath(p)
    return f"/mnt/{p[0].lower()}" + p[2:].replace("\\", "/")

def _insufficient(text, n_pages):
    """文本层「缺失/不足」判据（registry 1.5.13）：`公司` 计数 < 20 或 字符数 < max(5000, 50×页数)。"""
    return text.count("公司") < 20 or len(text) < max(5000, 50 * n_pages)

def ocr_recover_text(code, pdf_path):
    """OCR 回退通道：pymupdf 渲染逐页 PNG → 分 chunk 并行 WSL tesseract → 按页序拼接。

    缓存 = <year_dir>/ocr_txt/{code}.txt（键 = PDF 字节数，命中即复用）；渲染 PNG 存
    artifacts/tmp/tier2_full/ocr_render/<code>/，用后即删（不随件留存）。
    返回 (text, detail_str)；detail_str 即 channel 记录（含页数/耗时/缺页）。
    """
    year_dir = os.path.dirname(os.path.dirname(os.path.abspath(pdf_path)))
    cache_dir = os.path.join(year_dir, "ocr_txt")
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{code}.txt")
    keyf = os.path.join(cache_dir, f"{code}.key")
    size = str(os.path.getsize(pdf_path))
    if os.path.exists(cache) and os.path.exists(keyf) and open(keyf).read().strip() == size:
        return open(cache, encoding="utf-8").read(), "ocr_tesseract_chi_sim（缓存命中）"
    tmpd = os.path.join(ROOT, "artifacts", "tmp", "tier2_full", "ocr_render", code)
    shutil.rmtree(tmpd, ignore_errors=True)
    os.makedirs(tmpd, exist_ok=True)
    t0 = time.time()
    try:
        doc = pymupdf.open(pdf_path)
        n_pages = len(doc)
        for i, pg in enumerate(doc):
            pg.get_pixmap(dpi=_OCR_DPI).save(os.path.join(tmpd, f"p{i:05d}.png"))
        doc.close()
        render_sec = time.time() - t0
        chunks = [(s, min(n_pages, s + _OCR_CHUNK)) for s in range(0, n_pages, _OCR_CHUNK)]
        script = _wsl_path(_OCR_SH)

        def run_chunk(se):
            """单 chunk = 一次 wsl.exe：bash <脚本> <dir> <s> <e> <lang> <psm>（避免内联引号被 Windows→WSL 改写）。"""
            s, e = se
            subprocess.run([_WSL_EXE, "-u", _WSL_USER, "bash", script, _wsl_path(tmpd),
                            str(s), str(e), _OCR_LANG, _OCR_PSM],
                           capture_output=True, timeout=7200)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, min(_OCR_MAX_CHUNKS, len(chunks)))) as ex:
            list(ex.map(run_chunk, chunks))
        # 缺页补跑一次（同脚本按缺失页连续区间重跑；仍缺则计入 missing）
        missing_idx = [i for i in range(n_pages)
                       if not os.path.exists(os.path.join(tmpd, f"p{i:05d}.txt"))]
        if missing_idx:
            runs = []
            for i in missing_idx:
                if runs and i == runs[-1][1]:
                    runs[-1][1] = i + 1
                else:
                    runs.append([i, i + 1])
            for s, e in runs:
                run_chunk((s, e))
        ocr_sec = time.time() - t0 - render_sec
        parts, missing = [], 0
        for i in range(n_pages):
            tp = os.path.join(tmpd, f"p{i:05d}.txt")
            if os.path.exists(tp):
                parts.append(open(tp, encoding="utf-8", errors="replace").read())
            else:
                missing += 1
                parts.append("")
        text = "\n".join(parts)
        with open(cache, "w", encoding="utf-8") as f:
            f.write(text)
        with open(keyf, "w") as f:
            f.write(size)
        return text, (f"ocr_tesseract_chi_sim（{n_pages}p，render {render_sec:.0f}s + "
                      f"ocr {ocr_sec:.0f}s，缺页 {missing}）")
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)

def locate(txt_path, pdf_path, code=None):
    """返回 (located, n_hits, kw_counts, first_page, n_pages, spans, text, page_offsets, channel)。

    通道优先级（registry 1.5.13 全量轮预登记）：
      ① pymupdf 文本层（默认）；
      ② 文本层缺失/不足时 → pdfminer.six + SimSun GID 确定性还原（仅 CID 编码损坏件，沿用试点通道）；
      ③ 仍不足 → OCR 回退（WSL tesseract 5.3.4 chi_sim，--psm 3，200 DPI）。
    「不足」判据 = `公司` 计数 < 20 或 字符数 < max(5000, 50×页数)（在 ①② 之后判定）。
    channel = 最终通道记录；试点 2018 的 100 正常件（pymupdf）/3 CID 件（pdfminer+cid_gid）判定不变。
    """
    doc = pymupdf.open(pdf_path)
    pages = [pg.get_text() for pg in doc]
    n_pages = len(pages)
    doc.close()
    text = "".join(pages)
    channel = "pymupdf"
    text_replaced = False       # 修 4（registry 1.5.17）：文本是否已由 CID/OCR 通道替换（页码不再适用）
    counts = {k: text.count(k) for k in KWS}
    if _insufficient(text, n_pages):
        cid_done = False
        # ② CID 确定性还原：仅当存在疑似编码损坏的文本层（字符数 >2000；扫描件/无文本层直接走 ③）
        if code and len(text) > 2000:
            try:
                rel_text, cid_note = cid_recover_text(code, pdf_path)
                if not _insufficient(rel_text, n_pages):
                    text, channel, cid_done = rel_text, cid_note, True
                    text_replaced = True
                else:
                    channel = f"pdfminer+cid_gid(不足)"
            except Exception as e:
                channel = f"pdfminer+cid_gid(失败:{type(e).__name__})"
        # ③ OCR 回退（仅当仍不足）
        if not cid_done and _insufficient(text, n_pages):
            if code:
                try:
                    ocr_text, ocr_note = ocr_recover_text(code, pdf_path)
                    prefix = "" if channel == "pymupdf" else f"{channel}→"
                    if not _insufficient(ocr_text, n_pages):
                        text, channel = ocr_text, f"{prefix}{ocr_note}"
                        text_replaced = True
                    else:
                        channel = f"{prefix}ocr_insufficient（回退后仍不足；{ocr_note}）"
                except Exception as e:
                    channel = f"{channel}→text_layer_invalid:ocr_fail:{type(e).__name__}"
            else:
                channel = f"{channel}→text_layer_invalid:no_code"
        counts = {k: text.count(k) for k in KWS}
    offsets, acc = [], 0
    for p in pages:
        offsets.append(acc)
        acc += len(p)
    hits = [m.start() for m in KW_RE.finditer(text)]
    spans = []   # (start, end, anchor_hit_position)
    for h in sorted(hits):
        a, b = max(0, h - WIN_PRE), min(len(text), h + WIN_POST)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b, h])
    first_page = None
    if hits:
        if text_replaced:
            # 修 4（registry 1.5.17）：CID 还原 / OCR 通道无 pymupdf 页码映射 → 标 not_applicable_channel
            #（不得以 pymupdf 页码冒充；pymupdf 页码与替换后文本的字符偏移不对应）
            first_page = "not_applicable_channel"
        else:
            h0 = hits[0]
            first_page = max(1, sum(1 for o in offsets if o <= h0))
    return bool(hits), len(hits), counts, first_page, n_pages, spans, text, offsets, channel

def _raw_ok(raw, rule):
    """token 形态过滤：公司名不得含连接词/数字/括号残片、不得以非中文字符开头。"""
    b = raw.strip()
    if re.match(r"^(ST|\*ST|S\*ST|SST)", b):
        b = re.sub(r"^(ST|\*ST|S\*ST|SST)", "", b)
    if not b or not re.match(r"^[一-龥A-Za-z]", b):
        return None
    if re.search(r"[0-9（）()\[\]【】]", b):
        return None
    if any(c in b for c in CONN_SUB):
        return None
    if len(b) < 2 or len(b) > 20:
        return None
    if rule in ("R1_full", "R5_table_row", "R4_enum") and len(b) < 3:
        return None
    return b

def extract_candidates(text, spans, self_exact, self_core, uni_keys=()):
    """返回 [dict(raw, base, rule, adjacent_code)]（未去重）。

    精度设计（解析层，抽取规则；映射规则不变）：
      R1 全称型：形态由「股份有限公司/有限责任公司/有限公司」后缀保证，命中窗口内直接采纳；
      R2/R3 代码锚定型：形态与语境同时由显式 A 股代码保证；
      R4 列举型：仅当**同句前置 ≤40 字**内出现 同行业/可比/同业 语境词时采纳（否则「包括…」会命中科目枚举）；
        修 1(a)(b)（registry 1.5.17）：触发词补「可比公司」类复合词（触发词自证）+ 并列切分补 和/与/及；
      R4 表头列型（R4_tbl_head，修 1(f)，registry 1.5.17）：表头行首 token = 可比公司类词 + 行内其余 token
        均期间/数字类 → 取其后连续 ≥2 行（行首短名 + 行内含数字）的行首名；
      R5 表格行型（R5'）：先切成「表格块」= ≥2 个连续「短名行 + 数字/百分号行」，且**块前 120 字**含
         同行业/可比/同业/证券代码/股票代码 语境词（排除「纳税主体名称/所得税税率」这类同形非同行表）。
    """
    PEER_CTX = ("同行业", "可比", "同业", "证券代码", "股票代码")
    out = []
    for a, b, hit in spans:
        seg = text[a:b]
        hit_off = hit - a
        for m in R2.finditer(seg):
            out.append(dict(raw=m.group(1), base=base_name(m.group(1)), rule="R2_name_code",
                            adjacent_code=m.group(2)))
        for m in R3.finditer(seg):
            out.append(dict(raw=m.group(2), base=base_name(m.group(2)), rule="R3_code_name",
                            adjacent_code=m.group(1)))
        for m in R1.finditer(seg):
            if not (-WIN_PRE <= m.start() - hit_off <= 250):   # 全称型只取命中近邻（防远距表夹带）
                continue
            out.append(dict(raw=m.group(0), base=base_name(m.group(0)), rule="R1_full", adjacent_code=None))
        for m in R4.finditer(seg):
            pre = seg[max(0, m.start() - 20):m.start()]   # 列举型必须紧跟同行语境词（否则命中科目枚举）
            ctx_ok = any(k in pre for k in PEER_CTX)
            peer_trig = m.group(0).startswith(R4_PEER_TRIG)
            # 修 1(a)（registry 1.5.17）：触发词本身为「可比公司」类复合词时按触发词自证；**凡复合触发词路径一律加
            # 名单形态守则**（≥2 成员 ∧ 每成员形如公司简称 ∧ 名单含 ≥1 个 A 股在册名）——防散文续写
            #（「同行业可比公司不存在明显差异，主要由于…」「可比公司上市后发展较快，盈利能力、…」）入名单。
            if not ctx_ok and not peer_trig:
                continue
            # 修 1(b)：并列切分补 和/与/及——**仅复合触发路径**（既有「包括/分别为/有：+ 同行语境」路径逐字未改，
            # 防「产品包括 A 和 B（产品名词组）」被拆出伪名，如「层压焊接装框和检测设备」）。
            raw_toks = ([t for t in R4_SEP.split(m.group(1)) if t] if peer_trig
                        else [t for t in re.split(r"[、,，]", m.group(1)) if t])
            toks = raw_toks
            if peer_trig:
                gap = m.group(0)[:len(m.group(0)) - len(m.group(1))]
                good = [t for t in toks if r4_member_ok(t)]
                # ① 任一成员 >6 字（散文/句片段形态，非换行截断残片）→ 整名丢弃；
                # ② 名单须 ≥2 个形态合规成员 ∧ 触发词后无并列分隔符 ∧ 含 ≥1 个 A 股在册名（名录守卫）；
                # ③ 只把形态合规成员写入候选（≤2 字 OCR 断行残片与含虚词片段不入名单）；
                # ④ 名录守卫的**首字修复**：t[1:] 在 A 股在册名录且 t 本身不在册 → 剥首字（切分残片，如
                #    有海泰科→海泰科、中苏交科→苏交科；真名如 有研新材 因 t[1:] 不在册而不剥）。
                toks = [r4_strip_head(t, uni_keys) for t in good]
                if any(len(t) > 6 for t in toks) or not r4_peer_list_ok(toks, gap, uni_keys):
                    continue
            for tok in toks:
                out.append(dict(raw=tok, base=base_name(tok), rule="R4_enum", adjacent_code=None))
        # ---- R5' 表格块 ----
        lines, pos = [], []
        start = 0
        for ln in seg.split("\n"):
            lines.append(ln.strip())
            pos.append(start)
            start += len(ln) + 1
        rows = [i for i, ln in enumerate(lines)
                if R5_NAME.match(ln) and i + 1 < len(lines)
                and (lambda nx: bool(nx) and bool(R5_NUM.match(nx)))(
                    re.sub(r"[ \u3000]", "", next((x for x in lines[i + 1:i + 4] if x), "")))]
        i = 0
        while i < len(rows):
            j = i
            while j + 1 < len(rows) and rows[j + 1] - rows[j] <= 4:
                j += 1
            block = rows[i:j + 1]
            if len(block) >= 2 and 0 <= pos[block[0]] - hit_off <= 350:
                ctx = seg[max(0, pos[block[0]] - 60):pos[block[0]]]   # 块前 60 字内须有同行语境词
                # 兄弟前缀过滤：≥3 行共享 ≥4 字前缀 = 子公司/客户分部表（非同行名单）
                toks = [lines[r] for r in block]
                pref = ""
                if len(toks) >= 3:     # 兄弟前缀：≥3 行共享 4-8 字前缀 → 子公司/客户分部表
                    for L in range(min(len(toks[0]), 8), 3, -1):
                        cand = toks[0][:L]
                        if sum(1 for t in toks if t.startswith(cand)) >= 3:
                            pref = cand
                            break
                if any(k in ctx for k in PEER_CTX):
                    for r in block:
                        tk = lines[r]
                        if pref and len(pref) >= 4 and tk.startswith(pref):
                            continue
                        out.append(dict(raw=tk, base=base_name(tk),
                                        rule="R5_table_row", adjacent_code=None))
            i = j + 1
        # ---- 修 1(f)（registry 1.5.17）：R4_tbl_head「可比公司表头列」型 ----
        # 表头行（行首 token ∈ 可比公司类词，其余 token 均期间/数字类）后**连续** ≥2 行
        # （行首 2–10 汉字短名 + 行内含数字/百分号）→ 取行首名；走与 R4_enum/R5_table_row 相同的弱证据过滤。
        for hi, hln in enumerate(lines):
            htoks = hln.split()
            if not htoks:
                continue
            if htoks[0].strip("：:，,、") not in TBL_HEAD:
                continue
            if any(not TBL_HDR_TOK.match(t) for t in htoks[1:]):
                continue
            names = []
            for hj in range(hi + 1, min(hi + 26, len(lines))):
                rm = TBL_ROW.match(lines[hj])
                if not rm or not re.search(r"[\d%％]", rm.group(2)):
                    break
                names.append(rm.group(1))
            if len(names) >= 2:
                for nm in names:
                    out.append(dict(raw=nm, base=base_name(nm), rule="R4_tbl_head", adjacent_code=None))
    # 归一化过滤
    clean = []
    # 修 1：R3 与 R1/R2 交叉校验（R3 名不得是本文档 R1/R2 候选名的真片段）
    # 修 5（registry 1.5.15）：R2 左边界修复（语境/连接词前缀逐层剥离）**先行**，再算 strong 集合
    for c in out:
        if c["rule"] == "R2_name_code":
            r = repair_r2_left(c["raw"], uni_keys)      # 修 1(e)：加名录守卫（registry 1.5.17）
            if r != c["raw"] and len(r) >= 2:
                REPAIR_LOG.append(dict(raw=c["raw"], fixed=r))
                c["raw"], c["base"] = r, base_name(r)
    strong = {c["base"] for c in out if c["rule"] in ("R1_full", "R2_name_code")}
    for c in out:
        if _raw_ok(c["raw"], c["rule"]) is None:      # 形态过滤：连接词/数字/括号残片/句片段
            continue
        b = c["base"]
        if c["rule"] == "R3_code_name":
            if not r3_shape_ok(b):                    # 修 1：R3 名称形态约束（长度/起头词/业务描述词）
                REJECT_LOG.append(dict(base=b, raw=c["raw"], code=c["adjacent_code"],
                                       why="形态约束（长度/起头词/业务描述词）"))
                continue
            if any(s != b and b in s for s in strong):  # 修 1：与 R1/R2 交叉校验（真片段 → 弃）
                REJECT_LOG.append(dict(base=b, raw=c["raw"], code=c["adjacent_code"],
                                       why="R1/R2 交叉校验（真片段）"))
                continue
        if c["rule"] == "R2_name_code" and any(b.startswith(x) for x in NAME_START_BAD):
            REJECT_LOG.append(dict(base=b, raw=c["raw"], code=c["adjacent_code"], why="R2 起头词约束"))
            continue                                  # 修 1：R2 名同受起头词约束（如「控股公司奥瑞德」）
        if len(b) < 2 or not re.search(r"[\u4e00-\u9fa5]", b):
            continue
        if b in BLACK or c["raw"] in BLACK:
            continue
        if is_self_name(b, c["raw"], self_exact, self_core, uni_keys):
            REJECT_LOG.append(dict(base=b, raw=c["raw"], code=None,
                                   why="发行人自指（容错：精确/同核/编辑距离/近全包含）"))
            continue                                  # 修 5（M2）：自名过滤加容错，逐条留证据
        if any(s in b for s in NOISE_SUB):
            continue
        if c["rule"] in ("R4_enum", "R5_table_row", "R4_tbl_head"):
            if any(s in b for s in NOISE_WEAK):       # 弱证据规则（无代码锚定）加严
                continue
            if c["rule"] in ("R5_table_row", "R4_tbl_head"):
                # 修 5（M1②）：NOISE_CHAR 单字判据 → **整名判据**（不得因单字命中丢整名，如 红日药业）
                if b in NOISE_TOKEN or all(ch in NOISE_CHAR for ch in b):
                    continue
                if not R5_NAME.match(b):
                    continue
        if b.endswith(NOISE_TAIL) or b.startswith(NOISE_TAIL):
            continue
        if c["rule"] == "R1_full" and len(b) < 3:
            continue
        clean.append(c)
    return clean

def prefer_code(codes, universe, at_year=None):
    """多代码命中时的择一规则：优先 试点年当时已上市且未退市 的公司，其次最小代码。"""
    cs = sorted(codes)
    if at_year:
        act = [c for c in cs if universe[c]["ipo_date"] and int(universe[c]["ipo_date"][:4]) <= at_year
               and (not universe[c].get("out_date") or int(universe[c]["out_date"][:4]) >= at_year)]
        if act:
            return act[0]
    return cs[0]

def map_name(cand, short, full, universe, at_year=None):
    n = cand["base"]
    raw = cand["raw"]
    code = cand.get("adjacent_code")
    if code and code in universe:
        return code, "adjacent_code"
    for key in (norm(raw), n, strip_st(norm(raw)), strip_st(n)):
        if key in short:
            return prefer_code(short[key], universe, at_year), "exact_short"
        if key in full:
            return prefer_code(full[key], universe, at_year), "exact_full"
    # 后缀归一化：抽出的全称去后缀后与全称字典/简称字典的「去后缀」形式比对
    if n in full:
        return prefer_code(full[n], universe, at_year), "exact_full"
    alt = re.sub(r"集团$", "", n)
    for key in (alt, n):
        for cand_key in (alt, n):
            if cand_key in full:
                return prefer_code(full[cand_key], universe, at_year), "suffix_norm"
        if key in short:
            return prefer_code(short[key], universe, at_year), "suffix_norm"
    return None, "unmapped"

def _is_truncated(text, raw):
    """该 token 在正文中是否**总是**前接汉字（= 从更长中文串中间切出来的残片）。"""
    if not text or not raw:
        return False
    hits = 0
    for m in re.finditer(re.escape(raw), text):
        hits += 1
        if m.start() == 0 or not CJK_CH.match(text[m.start() - 1]):
            return False
        if hits >= 5:
            break
    return hits > 0

def classify_unmapped(b, raw, text, uni_keys, self_names=()):
    """返回 (类别, 证据)。类别 ∈ {碎片, 名称漂移, 抽取噪声, 残差桶（未过前三类判据）}。

    判据（**描述性口径**，逐名留证据；主口径失败率不依赖本分类）：
      碎片 = 复核清单 / 起头「公司·发行人」/ 含体裁·科目词 / 正文中总以前接汉字出现（截断残片）/
             为发行人自身全称的截断自指；
      名称漂移 = 与 A 股 universe 名**同头 ≥3 字**，或（同头 2 字 ∧ 公司式收尾词 ∧ 长度 ≥4）；
      抽取噪声 = 整名 ≈ 通用词/科目/句片段词（`GEN_SUB` **整名判据** + **公司后缀豁免**，修 3，registry 1.5.17；
             原「子串命中」判据把真实公司名（上海弘陆物流设备 / 山东联创节能新材料…）误判为噪声）；
      残差桶（未过前三类判据）= 其余——**兜底 return**，其来源需独立判读，**不得**以本桶断言「非 A 股实体」
             （修 2 改名，registry 1.5.17；复核抽样 n=160：真非上市/境外 ≈62.5%、抽取伪影 ≈30.0%、
             A 股名漂移 ≈7.5%）。
    """
    if b in FRAG_EXPLICIT:
        return "碎片", "复核清单（形态实测：截断/句片段/体裁词）"
    if b in DRIFT_EXPLICIT:
        return "名称漂移", "复核清单（A 股改名/ST 前缀漂移）"
    if any(b and b in s for s in self_names):
        return "碎片", "发行人自身名称的自指（抽取伪影，非外部同行）"
    if b.startswith(("公司", "本公司", "发行人", "上述", "该", "其")):
        return "碎片", f"起头为「{b[:2]}」"
    if any(w in b for w in FRAG_WORD):
        return "碎片", "含体裁/科目词"
    if _is_truncated(text, raw):
        return "碎片", "正文中总以「前接汉字」出现（截断残片）"
    for L in (3, 2):
        if len(b) > L and any(k.startswith(b[:L]) and k != b for k in uni_keys):
            if L >= 3 and not any(w in b for w in NON_ENTITY_WORD):
                return "名称漂移", f"A 股 universe 有同头 3 字名（{b[:L]}…）"
            if L == 2 and len(b) >= 4 and b[-2:] in NAME_TAIL and b[:2] not in PLACE_PREFIX:
                return "名称漂移", f"A 股 universe 有同头 2 字名（{b[:L]}…）+ 公司式收尾"
    if is_generic_name(b):     # 修 3（registry 1.5.17）：整名 ≈ 通用词 + 公司后缀豁免
        return "抽取噪声", "整名 ≈ 通用词/科目/句片段词（GEN_SUB 整名判据 + 公司后缀豁免；抽取层伪影，非外部实体）"
    # 修 2（registry 1.5.17）：第 4 类改名——原「非 A 股实体」是兜底 return 的**断言**而非判据证据；
    # 中性命名「残差桶（未过前三类判据）」，来源需独立判读（复核抽样 n=160：真非上市/境外 ≈62.5%、
    # 抽取伪影 ≈30.0%、A 股名漂移 ≈7.5%）。
    return RESIDUAL_CLS, RESIDUAL_EV

