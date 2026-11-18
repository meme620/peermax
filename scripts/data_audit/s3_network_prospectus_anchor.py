#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二梯队 #9 路线 A（plan §8 B″11-7 锚 1）：招股书「同行业可比公司」自报锚试点。

口径真源 = `configs/parameter_registry.yaml` 的 `prospectus_peer_anchor_pilot`（P0 先登记后执行）
+ `reports/audits/s3_remaining_workplan_20261011.md` 第二梯队 #9 路线 A。

步骤（P2–P5 一体）：
  P2 解析定位：pymupdf 页码级扫描 4 个关键词（同行业可比公司 / 可比上市公司 / 同行业上市公司 /
     同行业可比上市公司），记录命中数、首个命中页码；`located = 命中数 > 0`。
  P3 抽名单：在命中窗口（hit−150 .. hit+1500，重叠合并）内按 5 条正则规则抽取公司名：
     R1 全称型（…股份有限公司/有限责任公司/有限公司）；R2 短名+代码型（短名（NNNNNN））；
     R3 代码+短名型（NNNNNN.SZ/SH 后随短名）；R4 列举型（包括/分别为/主要有… ×、×）；
     R5 表格行型（独立成行的 2–10 字短名，下一非空行以数字/百分号开头）。
     归一化：去空白/全角半角统一/去引号；去「股份有限公司/有限责任公司/有限公司」后缀得 base；
     排除发行人自身（全称与简称）、聚合词与通用词（黑名单）、纯数字/长度 <2 的 token。
  P4 映射：名称→代码，模式 = `exact_short`（项目 firm 名单简称精确）/ `exact_full`（全称字典精确）/
     `suffix_norm`（去公司后缀后精确）/ `adjacent_code`（窗口内自带 6 位 A 股代码）；失败 = 上述全不中。
     全称字典 = ① 年报文本头部全称（本地抽取，缓存 artifacts/tmp）+ ② 东财曾用名/全称缓存。
  P5 度量（主口径网络 = `pm_ma_ewcos_teacher_v1_{试点年+2}`，该 focal 上市后第一个其为节点的版本）：
     重合率 share = Σ_i #{j∈D_i : (i,j)∈E} / Σ_i #D_i；基线 d = E/(N(N−1)/2)（N = 入网公司含零对手，
     E = 无向边数，与 lift_pair_density_readout 同口径）；lift = share / d。

产出：`reports/data_gates/G0/s3_network_prospectus_pilot_<year>.json` + `.md`（+ 逐公司明细 CSV）。
停机（已登记）：解析成功率 <80% / 映射失败率 >20% / 覆盖率 <50% → 停（不得宽松匹配或人工补录）。

用法: python scripts/data_audit/s3_network_prospectus_anchor.py --year 2018
      python scripts/data_audit/s3_network_prospectus_anchor.py --year 2018 --codes 601138,300750
"""
# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.prospectus.consts import ANNUAL_TXT, A_STOCKS, BLACK, CACHE, CACHE_INFO, CJK_CH, CONN_SUB, DRIFT_EXPLICIT, EM_NAMES, FIRM_MAP, FRAG_EXPLICIT, FRAG_WORD, FULL_NAME_RE, GEN_SUB, GEN_SUB_MAXLEN, GEN_SUB_SET, GEN_SUB_TAIL_EXEMPT, GEO_EXTRA, KWS, KW_RE, LABEL_CUT, NAME_BIZ_WORD, NAME_HIST, NAME_HIST_CAND, NAME_START_BAD, NAME_TAIL, NET_DIR, NOISE_CHAR, NOISE_GENERIC, NOISE_SUB, NOISE_TAIL, NOISE_TOKEN, NOISE_WEAK, NON_ENTITY_WORD, PLACE_PREFIX, R1, R2, R2_CTX_HEAD, R3, R4, R4_MEMBER_RE, R4_MEMBER_STOP, R4_PEER_TRIG, R4_SEP, R5_NAME, R5_NUM, REJECT_LOG, REPAIR_LOG, RESIDUAL_CLS, RESIDUAL_EV, ROOT, SELF_RE, ST_PREFIX, SUFFIX_END, TBL_HDR_TOK, TBL_HEAD, TBL_ROW, WIN_POST, WIN_PRE, _CID_CACHE, _CTX_LINK_TAIL, _CTX_SUFFIX, _GEO, _GID2UNI, _MEAS, _OCR_CHUNK, _OCR_DPI, _OCR_LANG, _OCR_MAX_CHUNKS, _OCR_PSM, _OCR_SH, _R2_LABEL, _SELF_IDX, _SIMSUN, _WS, _WSL_EXE, _WSL_USER
from lib.prospectus.names import base_name, build_self_index, ctx_suffix_frags, cut_label_tail, geo_prefix, is_generic_name, is_self_name, lev_leq1, norm, r3_shape_ok, r4_member_ok, r4_peer_list_ok, r4_strip_head, rel, repair_r2_left, self_names_for, strip_st, trim_dup_name
from lib.prospectus.universe import build_full_name_map, load_firm_universe, load_universe_name_keys
from lib.prospectus.text import _gid2uni, _insufficient, _is_truncated, _raw_ok, _wsl_path, cid_recover_text, classify_unmapped, extract_candidates, locate, map_name, ocr_recover_text, prefer_code
from lib.prospectus.ingest import _ingest_one, ingest_main, resolve_network_version
from lib.prospectus.report import _f, write_anchor_md, write_md
from lib.prospectus.measure import _dedupe_readings, _init_measure_worker, _load_network, _measure_one, _wilson, _year_rates, measure_main
# --- end scripts/lib bootstrap ---
import argparse
import collections
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import pymupdf



# 修 5（registry 1.5.15）：R2 标签集补「证券代码 / 公司代码」（原仅「股票代码」）——
# 招股书正文常见「名称（证券代码：NNNNNN）」形态（如 300737 正文「东方雨虹（证券代码：002271）…」）。
# 修 1（2026-10-14 修复轮二，registry 1.5.17）：R2 括号内**空白容错**——代码标签内 /
# 标签与数字之间允许空白与换行断行（CID/OCR 件常见「（证\n\n券代码：002271）」；
# 300737 因该断行整条不命中 → 东方雨虹 002271 漏抽）。标签集与匹配形态不变。
# 修 1（registry 1.5.17）(a)：R4 触发词补「可比公司」类**复合触发词**——招股书常见
# 「同行业可比公司A、B、C、D、E和F」直列（无「包括/有：」），原触发词集不命中（300748 原抽 0 名）。
# 修 1（registry 1.5.17）(b)：并列切分补「和/与/及」（末位成员原整 token 含连接词被 `_raw_ok` 丢弃；
# 切分后逐 token 仍走同一形态过滤 → 含连接词的 token 照样被丢，**不放宽匹配口径**）。
# 修 1（registry 1.5.17）(f)：「可比公司表头列」型（`R4_tbl_head`）——表头行首 token = 可比公司类词，
# 行内其余 token 均为期间/数字类；取其后**连续** ≥2 行（行首 2–10 汉字短名 + 行内含数字/百分号）的行首名。
# 修 1(a) 守则（registry 1.5.17）：复合触发词直列时（无可信语境前置），成员须形如**公司简称**——
# 散文续写（「同行业可比公司不存在明显差异，主要由于…」「可比公司上市后发展较快，盈利能力、…」）不得入名单。
# 虚词字符（成员全名含任一即判散文片段）：真实公司简称不含 的/了/较/均/在/于/为/是… 等虚词。
# 注：**不含** 有/会/再/同/大/多/通/达/实/包/中/上/高 等（真实简称可作首字/中字：有研新材 / 会稽山 /
# 再升科技 / 同花顺 / 大族激光 / 多氟多 / 中科三环 …）——只留纯虚词，避免误伤真名。






# 抽取层精度过滤：财务科目/表格通用词的**子串**黑名单（命中即弃；公司名不含这些子串）
# 弱证据规则（R4 列举 / R5 表格行，无代码锚定）的加严词表：科目/资产/费用/客户分部件类
# 连接词/句式子串：出现即说明该 token 是句子片段而非公司名（公司名不含这些连接成分）


# ---- 修 5（2026-10-14 修复轮，registry 1.5.15「先登记后跑」）：H1/M1/M2 三处解析缺陷的规则化修复 ----
# H1-(b) 标签边界切分：年报头部副标题行无空白，「简称」捕获串常与紧随的代码类标签粘连
#（如「公司代码：603501公司简称：韦尔股份转债代码：113616转债简称：韦尔转债…」）→ 按最早标签边界切分。
#（与 §P5 PLACE_PREFIX 同源；PLACE_PREFIX 收录 2 字形态（黑龙/内蒙），此处按登记原文补 3 字前缀
#（黑龙江/内蒙古）——登记原文：「区域/『中国』前缀（与 §P5 PLACE_PREFIX 同源，**含黑龙江/内蒙古等 3 字前缀**）」）
# 修 5（registry 1.5.15）④：归因口径第 4 类「抽取噪声」判据 = 既有通用词/科目/句片段词表
#（与第一轮 `failure_diagnostics.GEN_SUB` 同表；此处提为模块级单一真源）。
# 修 3（registry 1.5.17）：GEN_SUB 判据 = **整名 ≈ 通用词**（+ 公司后缀豁免）。




# 修 2（registry 1.5.17）：unmapped 兜底类**改名**——原「非 A 股实体」是断言而非判据证据。





















# ---- 修 1（2026-10-13 修正轮）：R3「代码+短名」名称形态约束 ----
# 背景：R3 只要求「6 位代码后随 2–10 字中文」，会把**表格/正文片段**当成短名（复核实测 14 例，
# 如 601606 的「属于铁路」「钢丝」「太原重」、300724 的「半导体集成电路制造设」、300741 的「食用香精」）。
# 约束 = ① 长度 3–10 汉字；② 不得为动词/连接词/主语词起头；③ 不含业务/产品/工序描述词（「主营业务」列片段）；
# ④ 与 R1/R2 交叉校验（不得是本文档 R1/R2 候选名的真片段）。













# ---------------------------------------------------------------- dictionaries






# ---------------------------------------------------------------- P2 locate






# ------------------------------------------------- OCR 回退通道（全量轮新增；registry 1.5.13 预登记）
# 引擎实测选型 = WSL Ubuntu tesseract 5.3.4（chi_sim, --psm 3；渲染 200 DPI；经 wsl.exe 调用）。
# 仅当文本层缺失/不足（`公司`<20 或 字符数 < max(5000, 50×页数)）且 CID 确定性还原不可用时启用。
                           # 曾用 4 → 24 并发 tesseract × 默认 OMP 线程 → load 95/32、0.12 页/s，
                           # 见 ocr_tesseract_chunk.sh 的 OMP_THREAD_LIMIT=2）










# ---------------------------------------------------------------- P3 extract




# ---------------------------------------------------------------- P4 map




# ---------------------------------------------------------------- P5 classify
# 修 2（2026-10-13 修正轮）：unmapped 名三分类 = 抽取碎片 / 名称漂移 / 非 A 股实体。
# 碎片判据（规则 + 复核清单）：① 正文中总以「前接汉字」出现（截断残片）；② 起头 公司/本公司/发行人；
# ③ 含体裁词（招股书/说明书）；④ 含科目/句片段词（信息来源/销售数量/竞争力…）。
# 名称尾（公司简称常见收尾词）——用于「与 A 股 universe 名同头 2 字 + 公司式收尾」的改名判读
# 地名/区域前缀（2 字判据排除项：地名同头不是改名证据）与非实体后缀词（子公司/设施/项目实体）








# ---------------------------------------------------------------- 全量轮 ingest（registry 1.5.13）






# ---------------------------------------------------------------- 全量测量（registry 1.5.13 汇总轮）
# 口径（逐字沿用试点，判据零改动）：四项定义 / 停机线 / 匹配口径（精确 + 后缀归一化 + 代码锚定）、
# 覆盖率分母 = 2015–2023 cohort 全量（2,656）；逐 IPO 年 Y 用其 network_version_used（min(Y+2,2023)
# + fallback v1_2022）；边界年子集 {2021,2022,2023} 单列、不混入主读数。




















def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--codes", default="")
    ap.add_argument("--base", default=os.path.join(ROOT, "data", "raw", "cninfo_prospectus"))
    ap.add_argument("--net-dir", default=NET_DIR)
    ap.add_argument("--text-lag", type=int, default=2, help="网络文本年 = 试点年 + lag（冻结样本筛选口径）")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-md", default="")
    ap.add_argument("--detail-csv", default="")
    ap.add_argument("--progress-every", type=int, default=20)
    ap.add_argument("--years", default="", help="全量 ingest 模式（registry 1.5.13）：如 2015-2023；"
                                                "只出解析层记录，不做锚测量（测量留汇总轮）")
    ap.add_argument("--workers", type=int, default=1, help="并行进程数（--years / --measure 模式）")
    ap.add_argument("--ingest-outdir", default="", help="ingest 输出目录（缺省 artifacts/tmp/tier2_full；"
                                                        "分批跑时每年批写独立目录，便于合并/断点续跑）")
    ap.add_argument("--measure", action="store_true",
                    help="全量测量模式（汇总轮）：抽取+映射+四项率+逐年/池化 lift+边界年单列+最终锚报告")
    ap.add_argument("--ingest-detail", default=os.path.join(ROOT, "reports", "data_gates", "G0",
                                                            "s3_network_prospectus_full_ingest_detail.csv"),
                    help="阶段报告逐份明细（--measure 模式下作 channel/status 逐份对照）")
    ap.add_argument("--pre-fix-report", default=os.path.join(ROOT, "reports", "data_gates", "G0",
                                                             "s3_network_prospectus_anchor_v1.json"),
                    help="修复前锚报告（只读；--measure 模式下嵌入 sensitivity_pre_fix 前置/后对照）")
    ap.add_argument("--pre-fix-report2", default=os.path.join(ROOT, "reports", "data_gates", "G0",
                                                              "s3_network_prospectus_anchor_v2_pre_fix.json"),
                    help="上一轮（H1/M1/M2 修复轮）锚报告（只读；修复轮二的前置基线 sensitivity_pre_round2）")
    args = ap.parse_args()

    if args.measure:
        return measure_main(args)
    if args.years:
        return ingest_main(args)

    year = args.year
    text_year = year + args.text_lag
    net_ver = f"pm_ma_ewcos_teacher_v1_{text_year}"
    datadir = os.path.join(args.base, str(year))
    pdfd, txtd = os.path.join(datadir, "pdf"), os.path.join(datadir, "txt")
    out = args.out or os.path.join(ROOT, "reports", "data_gates", "G0",
                                   f"s3_network_prospectus_pilot_{year}.json")
    out_md = args.out_md or out.replace(".json", ".md")
    detail_csv = args.detail_csv or os.path.join(ROOT, "artifacts", "tmp", "tier2_prosp",
                                                 f"pilot_{year}_detail.csv")

    firm, short = load_firm_universe()
    full, short_hist, full_info = build_full_name_map(firm)
    if not os.path.exists(CACHE) or full_info.get("cache_version") != 4:
        raise SystemExit("名称注册表缓存构建失败（cache_version != 4）")
    for nm, cs in short_hist.items():
        short.setdefault(nm, set()).update(cs)
    uni_keys = load_universe_name_keys()
    build_self_index(full, short_hist)     # 修 5（M2-i）：自名集合含代码锚定历史简称（试点路径同链）
    print(f"[P-2] 名称字典：firm universe {len(firm)}；简称键 {len(short)}（含历史简称 {len(short_hist)}）；"
          f"全称键 {len(full)}（{full_info.get('n_codes')} 只覆盖）", flush=True)

    cohort = [c for c, m in firm.items() if m["ipo_date"][:4] == str(year)]
    cohort.sort()
    if args.codes:
        keep = set(args.codes.split(","))
        cohort = [c for c in cohort if c in keep]
    print(f"[P-2] 试点年 {year} IPO 公司（项目 universe 内）= {len(cohort)} 家；网络版本 = {net_ver}", flush=True)

    # 网络
    with open(os.path.join(args.net_dir, f"{text_year}_support.csv"), encoding="utf-8") as f:
        sup = list(csv.DictReader(f))
    with open(os.path.join(args.net_dir, f"{text_year}_similarity.csv"), encoding="utf-8") as f:
        edges = set()
        for r in csv.DictReader(f):
            a, b = r["focal_id"], r["source_id"]
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    N = len(sup)
    E = len(edges)
    base_density = E / (N * (N - 1) / 2) if N > 1 else float("nan")
    nodes = {r["focal_id"] for r in sup}
    print(f"[P-5] 网络 {text_year}：N={N} 无向边 E={E} 基线密度 d={base_density:.6f}", flush=True)

    detail, t0 = [], time.time()
    n_downloaded = 0
    n_located = 0
    all_names, all_unmapped = {}, {}
    pair_tot, pair_hit, pair_tot_in, pair_hit_in = 0, 0, 0, 0
    rejected = []
    per_firm = []
    for i, code in enumerate(cohort):
        pdfp, txtp = os.path.join(pdfd, f"{code}.pdf"), os.path.join(txtd, f"{code}.txt")
        rec = dict(code=code, firm_id=firm[code]["firm_id"], name=firm[code]["name"],
                   ipo_date=firm[code]["ipo_date"], board=firm[code]["board"])
        if not (os.path.exists(pdfp) and os.path.exists(txtp)):
            rec.update(status="no_file", located=False, n_unique_names=0, n_mapped=0, n_unmapped=0,
                       network_node=firm[code]["firm_id"] in nodes, n_pairs=0, n_pairs_edge=0,
                       overlap_share=None, unmapped_names=[], unmapped_classified=[], channel="no_file")
            detail.append(rec)
            per_firm.append(rec)
            continue
        n_downloaded += 1
        (located, n_hits, counts, first_page, n_pages, spans, text,
         offsets, channel) = locate(txtp, pdfp, code=code)
        rec.update(status="parsed" if located else "no_section", located=located, n_kw_hits=n_hits,
                   kw_counts=counts, first_hit_page=first_page, n_pages=n_pages,
                   n_spans=len(spans), text_chars=len(text), channel=channel)
        if not located:
            rec.update(n_unique_names=0, n_mapped=0, n_unmapped=0,
                       network_node=firm[code]["firm_id"] in nodes, n_pairs=0, n_pairs_edge=0,
                       overlap_share=None, unmapped_names=[], unmapped_classified=[])
            detail.append(rec)
            per_firm.append(rec)
            if (i + 1) % args.progress_every == 0 or i + 1 == len(cohort):
                print(f"[P-2] 进度 {i + 1}/{len(cohort)}（located={n_located}）", flush=True)
            continue
        n_located += 1
        self_exact, self_core = self_names_for(firm, code)   # 修 5（M2-i）：含代码锚定历史简称
        REJECT_LOG.clear()
        REPAIR_LOG.clear()
        cands = extract_candidates(text, spans, self_exact, self_core, uni_keys)
        for x in REJECT_LOG:
            rejected.append(dict(focal=code, base=x["base"], peer=x.get("code"), why=x["why"]))
        REJECT_LOG.clear()
        REPAIR_LOG.clear()
        uniq = {}
        for c in cands:
            key = c["base"]
            prev = uniq.get(key)
            if prev is None or (prev["adjacent_code"] is None and c["adjacent_code"]):
                uniq[key] = c
        mapped, modes, unmapped = {}, {}, []
        for b, c in sorted(uniq.items()):
            code6, mode = map_name(c, short, full, firm, at_year=year)
            modes[mode] = modes.get(mode, 0) + 1
            if code6 == code:      # 映射到发行人自身 → 不计入自报同行（释义/正文自指）
                continue
            if code6:
                mapped[b] = dict(code6=code6, mode=mode, raw=c["raw"])
            else:
                unmapped.append(dict(base=b, raw=c["raw"], rule=c["rule"]))
            all_names[b] = all_names.get(b, 0) + 1
            if not code6:
                all_unmapped[b] = all_unmapped.get(b, 0) + 1
        rec.update(n_unique_names=len(uniq), n_mapped=len(mapped), n_unmapped=len(unmapped),
                   map_modes=modes, peers=sorted(mapped.keys()),
                   peers_codes=sorted({v["code6"] for v in mapped.values()}),
                   unmapped_names=[u["base"] for u in unmapped][:20],
                   unmapped_classified=[dict(name=u["base"], cls=classify_unmapped(
                       u["base"], u["raw"], text, uni_keys, self_exact)[0],
                       evidence=classify_unmapped(u["base"], u["raw"], text, uni_keys, self_exact)[1])
                       for u in unmapped])
        # P5 lift 分量（节点与边均为 firm_id 口径）
        fid = firm[code]["firm_id"]
        if fid in nodes:
            pset = sorted({firm[j]["firm_id"] for j in {v["code6"] for v in mapped.values()}} - {fid})
            tot = len(pset)
            hit = sum(1 for j in pset if (min(fid, j), max(fid, j)) in edges)
            rec.update(network_node=True, n_pairs=tot, n_pairs_edge=hit,
                       overlap_share=(hit / tot) if tot else None)
            pair_tot += tot
            pair_hit += hit
            pset_in = {j for j in pset if j in nodes}
            pair_tot_in += len(pset_in)
            pair_hit_in += sum(1 for j in pset_in if (min(fid, j), max(fid, j)) in edges)
        else:
            rec.update(network_node=False, n_pairs=0, n_pairs_edge=0, overlap_share=None)
        per_firm.append(rec)
        detail.append(rec)
        if (i + 1) % args.progress_every == 0 or i + 1 == len(cohort):
            print(f"[P-2] 进度 {i + 1}/{len(cohort)}（located={n_located} "
                  f"names={len(all_names)} unmapped={len(all_unmapped)} elapsed={time.time() - t0:.0f}s）",
                  flush=True)

    # ---------------- 三项率 ----------------
    parse_rate = n_located / n_downloaded if n_downloaded else float("nan")
    n_names = len(all_names)
    map_fail_rate = len(all_unmapped) / n_names if n_names else float("nan")
    covered = [r for r in per_firm if r.get("n_mapped", 0) >= 1]
    coverage_rate = len(covered) / len(cohort) if cohort else float("nan")
    cov_in_net = [r for r in per_firm if r.get("network_node") and r.get("n_mapped", 0) >= 1]
    n_net_nodes = sum(1 for r in per_firm if r.get("network_node"))
    coverage_in_net = len(cov_in_net) / n_net_nodes if n_net_nodes else float("nan")
    share = pair_hit / pair_tot if pair_tot else float("nan")
    share_in = pair_hit_in / pair_tot_in if pair_tot_in else float("nan")
    lift = share / base_density if base_density and base_density == base_density else float("nan")
    lift_in = share_in / base_density if base_density and base_density == base_density else float("nan")

    stop = []
    if parse_rate < 0.80:
        stop.append(f"解析成功率 {parse_rate:.4f} < 0.80")
    if map_fail_rate > 0.20:
        stop.append(f"映射失败率 {map_fail_rate:.4f} > 0.20")
    if coverage_rate < 0.50:
        stop.append(f"覆盖率 {coverage_rate:.4f} < 0.50")

    # 失败案例样本（≤10）：解析失败（无文件/未定位）→ 抽到名单但无可映射同行 → 在网但零重合
    fail_cases = []
    for r in per_firm:
        if r.get("status") == "no_file":
            fail_cases.append(dict(case="no_file_scan_only_or_download_failed", code=r["code"],
                                   name=r["name"], detail="PDF/txt 缺失（实测为无文本层扫描件或候选下载失败）"))
        elif not r.get("located"):
            ch = str(r.get("channel", ""))
            if "备选通道失败" in ch:
                det = f"文本层 CID 编码损坏且备选通道失败（{ch}）；{r.get('n_pages')} 页"
            elif r.get("kw_counts") and not any(r["kw_counts"].values()):
                det = f"4 关键词命中数 = 全 0（{r.get('n_pages')} 页）——非标准/缺失章节标题"
            else:
                det = f"4 关键词命中数 = 全 0（{r.get('n_pages')} 页）"
            fail_cases.append(dict(case="section_not_located", code=r["code"], name=r["name"], detail=det))
    for r in per_firm:
        if r.get("located") and r.get("n_mapped", 0) == 0:
            fail_cases.append(dict(case="no_mapped_peer", code=r["code"], name=r["name"],
                                   detail="、".join(r.get("unmapped_names", [])[:5])
                                   or f"唯一名称 {r.get('n_unique_names', 0)} 个（均被自身/黑名单过滤）"))
    for r in per_firm:
        if r.get("network_node") and (r.get("overlap_share") == 0):
            fail_cases.append(dict(case="zero_overlap_in_network", code=r["code"], name=r["name"],
                                   detail="自报同行（已映射）" + "、".join(r.get("peers", [])[:6])))
    fail_cases = fail_cases[:10]

    unmapped_top = sorted(all_unmapped.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    year_counts = {}
    for r in detail:
        year_counts[r.get("status", "?")] = year_counts.get(r.get("status", "?"), 0) + 1

    # ---------------- 修 2 ②：失败率口径分列（碎片 / 名称漂移 / 非 A 股实体） ----------------
    cls_of = {}
    for r in per_firm:
        for u in (r.get("unmapped_classified") or []):
            cls_of.setdefault(u["name"], u)
    by_cls = {"碎片": [], "名称漂移": [], RESIDUAL_CLS: []}
    for n, u in sorted(cls_of.items()):
        by_cls.setdefault(u["cls"], []).append(u)
    n_frag = len(by_cls["碎片"])
    fail_no_frag = ((len(all_unmapped) - n_frag) / (n_names - n_frag)) if n_names > n_frag else float("nan")
    failure_decomposition = {
        "note": "修 2：unmapped 唯一名三分类（规则判据 + 复核清单；明细含逐名证据）。"
                "「剔碎片后」= 分子分母同剔碎片（碎片不是公司名，不应进入名称口径）",
        "上界（含碎片，主口径）": map_fail_rate, "上界分子分母": f"{len(all_unmapped)}/{n_names}",
        "剔碎片后": fail_no_frag, "剔碎片后分子分母": f"{len(all_unmapped) - n_frag}/{n_names - n_frag}",
        "碎片": len(by_cls["碎片"]), "名称漂移": len(by_cls["名称漂移"]),
        RESIDUAL_CLS: len(by_cls[RESIDUAL_CLS]),
        "明细": {k: [{"name": u["name"], "evidence": u["evidence"]} for u in v] for k, v in by_cls.items()},
        "修正前对照（v4 交付口径）": {"含碎片上界": 0.1869, "含碎片分子分母": "63/337",
                                     "剔碎片后（复核清单 12 碎片）": 51 / 325},
    }

    # 覆盖率拆解与逐公司重合率分布
    cov_breakdown = {
        "no_file（下载失败，不计入解析分母）": year_counts.get("no_file", 0),
        "no_section（解析未定位，计入解析分母）": year_counts.get("no_section", 0),
        "located_but_no_mapped_peer": sum(1 for r in per_firm if r.get("located") and r.get("n_mapped", 0) == 0),
        "covered（非空且 ≥1 已映射）": len(covered),
    }
    ov = [r["overlap_share"] for r in per_firm
          if r.get("network_node") and r.get("overlap_share") is not None]
    ov_stats = {
        "n_firms_in_network_with_peers": len(ov),
        "mean": (sum(ov) / len(ov)) if ov else None,
        "median": (sorted(ov)[len(ov) // 2] if ov else None),
        "min": min(ov) if ov else None, "max": max(ov) if ov else None,
        "n_zero_overlap": sum(1 for x in ov if x == 0),
        "n_ge_50pct": sum(1 for x in ov if x >= 0.5),
        "n_ge_90pct": sum(1 for x in ov if x >= 0.9),
    }
    peers_per_firm = [r["n_mapped"] for r in per_firm if r.get("located")]
    mode_totals = {}
    for r in per_firm:
        for k, v in (r.get("map_modes") or {}).items():
            mode_totals[k] = mode_totals.get(k, 0) + v
    # 解析修复轮次（停机 → 先修解析 → 复测；日志为证据）
    parse_iterations = [
        {"round": "v1 初始抽取规则", "mapping_failure_rate": 0.4681, "n_unmapped": 293,
         "n_unique": 626, "coverage_rate": 0.6762, "lift": 26.0008,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis.log",
         "note": "窗口 hit−150..+1500；无形态过滤（连接词/数字/科目词未过滤）；R5 表格行无同行语境要求"},
        {"round": "v2 收窄窗口 + R1 近邻 + 兄弟前缀", "mapping_failure_rate": 0.3653, "n_unmapped": 183,
         "n_unique": 501, "coverage_rate": 0.6667, "lift": 26.1124,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis_v2.log",
         "note": "窗口 hit−100..+500；R1 只取命中近邻（+250 字内）；R5 增加同行语境词与兄弟前缀过滤"},
        {"round": "v3 形态过滤生效 + 弱证据规则加严", "mapping_failure_rate": 0.2382, "n_unmapped": 81,
         "n_unique": 340, "coverage_rate": 0.5905, "lift": 25.3663,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis_v3.log",
         "note": "接上 `_raw_ok` 形态过滤（连接词/数字/括号/句片段）；R4 列举必须紧跟同行语境词；R5 语境 60 字 + "
                 "科目/资产/费用词表加严"},
        {"round": "v4 名称注册表补历史简称（代码锚定）", "mapping_failure_rate": 0.1869,
         "n_unmapped": 63, "n_unique": 337, "coverage_rate": 0.5905, "lift": 25.7681,
         "log": "artifacts/tmp/tier2_prosp/p2_p5_analysis_v4.log",
         "note": "年报头部「股票简称+股票代码」配对抽取历史简称（代码锚定、非模糊匹配），弥合 2018 名称与"
                 "2026 项目名单的名称漂移（v4 为**交付口径**，其后经独立复核 CONFIRMED_WITH_ISSUES）"},
        {"round": "v5 修正轮（本轮）：R3 名称形态约束 + 名称注册表标签/顺序/更名/释义 + CID 备选通道",
         "mapping_failure_rate": map_fail_rate, "n_unmapped": len(all_unmapped), "n_unique": n_names,
         "coverage_rate": coverage_rate, "lift": lift,
         "log": "artifacts/tmp/tier2_prosp_fix/f1_dump_v2.log / f1_variant_eval3.py / f3_recovery.log",
         "note": "① 修 1：R3「代码+短名」加名称形态约束（3–10 汉字、非动词/连接词/科目词起头、无业务描述词）"
                 "并与 R1/R2 交叉校验（真片段弃），剔除复核确认的抽取污染；② 修 2：名称注册表头部窗口 6000→24000 字、"
                 "标签补 公司代码/A股证券简称 且允许冒号、头部「简称+全称同头拼接」去重、新增更名披露与释义缩写两条"
                 "代码锚定来源、注册名去 ST/*ST 前缀建键；③ 修 3：3 份文本层 CID 编码损坏件走 pdfminer+GID 反查"
                 "备选通道（解析成功率上升）"},
    ]

    # ---------------- 修 1：抽取污染登记与敏感性（修正前/复核/修正后三套读数） ----------------
    r3_pollution = {
        "issue": "R3「6 位代码 + 2–10 字中文」把表格/正文片段误当短名（独立复核 CONFIRMED_WITH_ISSUES 实测 14 例，"
                 "经相邻代码误挂为同行）",
        "examples": ["601606 属于铁路→600893 / 钢丝→603969 / 太原重→600072 / 属于专用设备制造业→601106 / "
                     "属于通用设备→600765 / 生产销售日用塑胶制品→002641",
                     "300724 半导体集成电路制造设→002371 / 单晶硅晶体生长炉→300029 / 太阳能光伏专用装备→002006 / "
                     "晶体生长→300316",
                     "300741 食用香精→002568 / 香精→603020", "300664 城市水环境综合治→300388",
                     "002943 控股公司奥瑞德→600666（R2 起头词污染）"],
        "fix": "R3 名称形态约束（3–10 汉字、非动词/连接词/科目词起头、无业务/产品/工序描述词）"
               "+ 与 R1/R2 交叉校验（R3 名不得是本文档 R1/R2 候选名的真片段）+ R2 同受起头词约束",
        "residual": "「太原重」（太原重工截断）形态上无法与真名区分，仍保留 → 已知残余（相邻代码 600072 本身是该件"
                    "同行，pair 落在同行集内但名-码配对错位）；601606 的**码-名错位**（表格列错行）属另一缺陷类，"
                    "本轮未改（登记为已知限制）；**剔除污染名的同时会连带剔除其相邻代码对**（如 300741 华宝股份的"
                    "两条 R3 名-码对：食用香精→002568、香精→603020，其中 603020 是其在网邻居）→ 该件因此由"
                    "「已覆盖」转为「抽到名单但无可映射同行」，与 3 份 CID 恢复件新增的覆盖相抵 → 覆盖率总数"
                    "持平（62/105）",
        "sensitivity": {
            "修正前（v4 交付口径）": {"pair_total": 253, "pair_edge": 137, "overlap_share": 0.541502,
                                     "lift": 25.7681, "diag_pair_total": 245, "diag_pair_edge": 137,
                                     "diag_overlap_share": 0.559184, "diag_lift": 26.6095},
            "复核口径（人工剔 14 名，独立复核 B）": {"pair_total": 246, "pair_edge": 136,
                                                     "overlap_share": 0.552846, "lift": 26.3079,
                                                     "diag_pair_total": 238, "diag_pair_edge": 136,
                                                     "diag_overlap_share": 0.571429, "diag_lift": 27.1922},
            "修正后（本轮规则口径）": {"pair_total": pair_tot, "pair_edge": pair_hit,
                                       "overlap_share": share, "lift": lift,
                                       "diag_pair_total": pair_tot_in, "diag_pair_edge": pair_hit_in,
                                       "diag_overlap_share": share_in, "diag_lift": lift_in},
        },
    }
    # ---------------- 修 4：下界/方向性说明（覆盖率为下界；lift 方向未证，已知偏置指向低估） ----------------
    direction_note = {
        "coverage_lower_bound": "覆盖率为**下界**：分母（试点年 IPO 家数）固定，分子只随抽取/映射成功而增加；"
                                "未定位（含无文本层扫描件）与未抽到的同行只会让覆盖率更低。",
        "lift_direction_unproven": "重合率/lift 的**方向未证**：漏抽同行会同时减少分子（漏掉的同行对）与分母"
                                   "（自报同行对总数），净方向取决于**漏抽同行的落网率**相对全体自报同行落网率的"
                                   "高低，不能先验断言「重合率为下界」。",
        "probe": "方向性检查（本轮，3 份污染/抽漏件）：宽松通道（注册表精确名 + 词边界）补出被漏同行后，"
                 "其网络边落网率与严格通道保留集对照见 anchor_readout.direction_check；"
                 "已知两处偏置（① 2 份无文本层扫描件的同行全漏；② R3 形态约束剔掉的名-码对中含真边）"
                 "方向不一致 → 头条 lift 视为**方向未证**，不宣称下界。",
    }

    # ---------------- 修 3：提取通道计数（按通道类别归并，缓存命中与首次解码同类） ----------------
    channel_counts = {}
    for r in per_firm:
        ch = r.get("channel", "?")
        if ch.startswith("pdfminer+cid_gid"):
            ch = "pdfminer+cid_gid（CID 编码损坏件备选通道）"
        elif "备选通道失败" in ch:
            ch = "cid_recover_failed"
        channel_counts[ch] = channel_counts.get(ch, 0) + 1
    # ---------------- 修 4 方向性检查（内联·廉价部分）：被剔名-码对的落网率 vs 保留集 ----------------
    rej = {}
    for x in rejected:
        if x.get("peer") and x["peer"] in firm and x["focal"] in firm:
            fid, j = firm[x["focal"]]["firm_id"], firm[x["peer"]]["firm_id"]
            if fid != j and fid in nodes and j in nodes:
                rej.setdefault((fid, j), x["base"])
    rej_hit = sum(1 for (i, j) in rej if (min(i, j), max(i, j)) in edges)

    def direction_check_cases(pb):
        out = []
        for k, v in ((pb or {}).get("cases") or {}).items():
            if v.get("missed_rate") is not None:
                out.append((k, v))
        return out

    # 宽松通道探针（artifacts/tmp/tier2_prosp_fix/f4_direction.json，由 f4_direction_probe2.py 产出）
    probe_path = os.path.join(ROOT, "artifacts", "tmp", "tier2_prosp_fix", "f4_direction.json")
    probe = None
    try:
        with open(probe_path, encoding="utf-8") as f:
            probe = json.load(f)
    except Exception:
        probe = None
    direction_check = {
        "note": "方向性检查（修 4）：核对「被剔/被漏同行的网络落网率」——若与保留集同量级或更高，"
                "说明抽取遗漏/污染剔除会**改变**（而非仅压低）重合率，故 lift 方向未证。",
        "rejected_pairs_in_network": len(rej), "rejected_pairs_edge": rej_hit,
        "rejected_pairs_edge_rate": (rej_hit / len(rej)) if rej else None,
        "kept_pairs_edge_rate_diag": share_in,
        "kept_pairs_in_network": pair_tot_in, "kept_pairs_edge": pair_hit_in,
        "rejected_examples": sorted({f"{x['focal']}:{x['base']}→{x['peer']}（{x['why']}）"
                                     for x in rejected})[:10],
        "loose_channel_probe": {
            "path": "artifacts/tmp/tier2_prosp_fix/f4_direction.json",
            "method": "宽松通道 = 窗口内注册表精确短名（≥3 汉字）+ 词边界 + 剔自身名；与严格通道保留集比对得"
                      "「被漏同行」，再看其网络边落网率",
            "base_density": (probe or {}).get("base_density"),
            "cases": {k: {"strict_edge/total": v.get("strict_edge/total"),
                          "strict_rate": v.get("strict_rate"),
                          "missed_edge/total": v.get("missed_edge/total"),
                          "missed_rate": v.get("missed_rate"),
                          "missed_names": sorted((v.get("missed_names") or {}).keys())}
                      for k, v in ((probe or {}).get("cases") or {}).items()} if probe else None,
            "reading": "被漏同行的落网率远高于基线密度 "
                       + (f"{(probe or {}).get('base_density', float('nan')):.4f}" if probe else "0.0210")
                       + "，但**相对保留集重合率方向不一**"
                       + ("（"
                          + "；".join(f"{k} = {v['missed_rate']:.4f}"
                                      for k, v in direction_check_cases(probe) if v.get("missed_rate") is not None)
                          + f"，保留集诊断口径 = {share_in:.4f}）" if probe else "")
                       + " → 与独立复核两处探针（均指向低估）合并后，结论 = **方向未证**，不宣称下界",
        },
        "no_text_layer_docs": "300760/300748 为无文本层扫描件（同行全漏，落网率不可测）→ 已知偏置之一",
    }

    result = {
        "report": "第二梯队 #9 路线 A 试点（招股书「同行业可比公司」自报锚，锚 1 / B″11-7）",
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "parameter": "prospectus_peer_anchor_pilot（registry 1.5.5，P0 先登记后执行）",
        "scope": {
            "pilot_year": year, "universe": "data/processed/firm_id_map.csv（5,543 只 A 股）",
            "cohort_n": len(cohort), "sample_scope": "全量（不抽样）",
            "network_version_rule": f"试点年 + {args.text_lag} → 文本年 {text_year}（该 focal 上市后第一个"
                                    f"其为网络节点的版本；冻结样本筛选 ma_sample_filter「上市不足 2 年」）",
            "network_version": net_ver, "available_at": f"{text_year + 1}-04-30",
            "download": {"dir": rel(datadir), "n_downloaded": n_downloaded,
                         "status_counts": year_counts,
                         "log": rel(os.path.join(datadir, "download_log.csv")),
                         "summary": rel(os.path.join(datadir, "download_summary.json"))},
            "keyword_set": list(KWS), "window": [WIN_PRE, WIN_POST],
            "extraction_rules": ["R1_full", "R2_name_code", "R3_code_name", "R4_enum", "R5_table_row"],
            "mapping_modes": ["exact_short", "exact_full", "suffix_norm", "adjacent_code"],
            "name_dict": {"firm_universe": len(firm), "short_keys": len(short), "full_keys": len(full),
                          "full_name_build": full_info},
        },
        "rates": {
            "parse_success_rate": parse_rate, "n_located": n_located, "n_downloaded": n_downloaded,
            "mapping_failure_rate": map_fail_rate, "n_unique_names": n_names,
            "n_unmapped_names": len(all_unmapped),
            "coverage_rate": coverage_rate, "n_covered": len(covered), "cohort_n": len(cohort),
            "coverage_rate_in_network": coverage_in_net, "n_network_nodes": n_net_nodes,
            "thresholds": {"parse_min": 0.80, "map_fail_max": 0.20, "coverage_min": 0.50},
            "stop_triggered": bool(stop), "stop_reasons": stop,
            "coverage_breakdown": cov_breakdown,
            "mapping_mode_totals": mode_totals,
            "peers_per_firm_mean": (sum(peers_per_firm) / len(peers_per_firm)) if peers_per_firm else None,
            "parse_iterations": parse_iterations,
            "failure_decomposition": failure_decomposition,
            "extraction_channels": channel_counts,
        },
        "anchor_readout": {
            "definition": "lift = P(自报同行 ∈ 网络对手 | 自报同行) ÷ P(随机 universe 公司 ∈ 网络对手)"
                          "（分母 = 该网络年无向对密度 d = E/(N(N−1)/2)）",
            "overlap_share_raw": share, "pair_total": pair_tot, "pair_edge": pair_hit,
            "base_density": base_density, "lift": lift,
            "diagnostic_node_restricted": {"overlap_share_raw": share_in, "pair_total": pair_tot_in,
                                           "pair_edge": pair_hit_in, "lift": lift_in,
                                           "note": "诊断变体：只统计两端都在该网络年节点集内的自报同行对"},
            "per_firm_overlap_stats": ov_stats,
            "network": {"text_year": text_year, "version": net_ver, "N_nodes": N,
                        "E_undirected": E},
            "direction_check": direction_check,
        },
        "extraction_pollution": r3_pollution,
        "direction_note": direction_note,
        "failure_cases_sample": fail_cases,
        "unmapped_top30": [{"name": k, "n_firms": v} for k, v in unmapped_top],
        "known_limitations": [
            "**覆盖率为下界**（分母固定、分子只增）；**重合率/lift 方向未证**——漏抽同行同时减少分子与分母，"
            "净方向取决于漏抽同行的落网率（方向性检查见 anchor_readout.direction_check 与 §2.2），"
            "不得写成「重合率为下界」。",
            "2 份招股书（300760 迈瑞医疗、300748 金力永磁）为**无文本层扫描件**（620/413 页仅 2,480/948 字），"
            "pymupdf 与 pdfminer 均无文本层 → 未进入解析分母；全量需补 OCR 通道。",
            "3 份招股书（300737 科顺股份、603059 倍加洁、603301 振德医疗）为**文本层 CID 编码损坏**"
            "（无 ToUnicode CMap，pymupdf 读出 chr(cid) 乱码、「公司/发行人/万元」计数 = 0）；本轮经 pdfminer.six "
            "+ SimSun GID 反查备选通道恢复（确定性字符还原、非 OCR），已进入解析/抽取/映射口径。",
            "单年（2018）、单网络版本（pm_ma_ewcos_teacher_v1_2020）试点；lift 不能外推为全期结论。",
            "项目 firm 名单为 2026 当前简称，招股书用 2018 当时简称；已用年报头部「简称+代码」（标签/顺序/冒号变体、"
            "去 ST 前缀）、更名披露与释义缩写四条代码锚定来源补历史简称，残余漂移仍计入映射失败。",
            "已知残余（本轮登记）：① 「太原重」（太原重工截断）形态上与真名不可分，仍经相邻代码映射；"
            "② 601606 等表格列错行造成的**码-名错位**（名与码各自都在同行集内、但配对错位）未在本轮修复。",
            "锚只作外部一致性读数，**不进入面板/回归、不改主口径**；未完成全量前「外部锚」条件仍未满足。",
        ],
        "per_firm": per_firm,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    os.makedirs(os.path.dirname(detail_csv), exist_ok=True)
    with open(detail_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "firm_id", "name", "ipo_date", "board", "status", "n_kw_hits",
                    "first_hit_page", "n_unique_names", "n_mapped", "n_unmapped", "network_node",
                    "n_pairs", "n_pairs_edge", "overlap_share", "peers"])
        for r in detail:
            w.writerow([r.get("code"), r.get("firm_id"), r.get("name"), r.get("ipo_date"),
                        r.get("board"), r.get("status"), r.get("n_kw_hits", ""),
                        r.get("first_hit_page", ""), r.get("n_unique_names", ""), r.get("n_mapped", ""),
                        r.get("n_unmapped", ""), r.get("network_node", ""), r.get("n_pairs", ""),
                        r.get("n_pairs_edge", ""), r.get("overlap_share", ""),
                        "|".join(r.get("peers", []) or [])])
    write_md(result, out_md)
    print(f"[P-6] 报告 → {rel(out)} / {rel(out_md)}；明细 → {rel(detail_csv)}", flush=True)
    print(f"[P-6] 三项率 = 解析 {parse_rate:.4f}（{n_located}/{n_downloaded}）/ 映射失败 "
          f"{map_fail_rate:.4f}（{len(all_unmapped)}/{n_names}）/ 覆盖率 {coverage_rate:.4f}"
          f"（{len(covered)}/{len(cohort)}）", flush=True)
    print(f"[P-6] 锚读数 = 重合率 {share:.6f}（{pair_hit}/{pair_tot}）vs 基线 {base_density:.6f} "
          f"→ lift {lift:.4f}（诊断·限在网 {lift_in:.4f}）", flush=True)
    print(f"[P-6] 停机 = {'触发: ' + '; '.join(stop) if stop else '未触发'}", flush=True)
    with open(out.replace(".json", "_run.log"), "w", encoding="utf-8") as f:
        f.write(json.dumps({k: result[k] for k in ("run_at", "rates", "anchor_readout", "scope")},
                           ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
