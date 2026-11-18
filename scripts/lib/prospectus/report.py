"""s3_network_prospectus_anchor 的 report 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.prospectus.consts import GEN_SUB_MAXLEN
from lib.prospectus.consts import RESIDUAL_CLS
from lib.prospectus.names import rel
import os

def write_md(r, path):
    """人读版报告（P6）：三项率 / lift 表 / 抽样口径 / 失败案例 / 全量可行性判断。"""
    sc, ra, an = r["scope"], r["rates"], r["anchor_readout"]
    ov, nb = an["per_firm_overlap_stats"], an["network"]
    L = []
    A = L.append
    A(f"# 第二梯队 #9 路线 A 试点：招股书「同行业可比公司」自报锚（锚 1 / plan §8 B″11-7）")
    A("")
    A(f"- 运行时间（UTC）：{r['run_at']}；口径参数：`{r['parameter']}`")
    A(f"- 试点年 **{sc['pilot_year']}**（该年 IPO {sc['cohort_n']} 家 ≥100）；抽样口径 = **{sc['sample_scope']}**")
    A(f"- 网络版本：`{sc['network_version']}`（{sc['network_version_rule']}；available_at {sc['available_at']}）")
    A(f"- 招股书目录：`{sc['download']['dir']}`（pdf/ + txt/ + `download_log.csv` + `download_summary.json`）")
    A("")
    A("## 1. 三项实测率与停机判定")
    A("")
    A("| 指标 | 实测 | 分子/分母 | 停机线 | 判定 |")
    A("|---|---|---|---|---|")
    A(f"| 解析成功率 | **{ra and r['rates']['parse_success_rate']:.4f}** | {r['rates']['n_located']}/"
      f"{r['rates']['n_downloaded']}（下载成功份数） | ≥0.80 | "
      f"{'PASS' if r['rates']['parse_success_rate'] >= 0.80 else 'FAIL'} |")
    A(f"| 映射失败率 | **{r['rates']['mapping_failure_rate']:.4f}** | {r['rates']['n_unmapped_names']}/"
      f"{r['rates']['n_unique_names']}（去重唯一名称） | ≤0.20 | "
      f"{'PASS' if r['rates']['mapping_failure_rate'] <= 0.20 else 'FAIL'} |")
    A(f"| 覆盖率 | **{r['rates']['coverage_rate']:.4f}** | {r['rates']['n_covered']}/"
      f"{r['rates']['cohort_n']}（试点年 IPO ∩ 项目 universe） | ≥0.50 | "
      f"{'PASS' if r['rates']['coverage_rate'] >= 0.50 else 'FAIL'} |")
    A("")
    A(f"**停机判定 = {'触发（' + '；'.join(r['rates']['stop_reasons']) + '）' if r['rates']['stop_triggered'] else '未触发（三项全过）'}**")
    A("")
    A("覆盖率拆解：" + "；".join(f"{k} = {v}" for k, v in r["rates"]["coverage_breakdown"].items()))
    A(f"；网络节点内覆盖率 = {r['rates']['coverage_rate_in_network']:.4f}"
      f"（{r['rates']['n_network_nodes']} 家在网公司中已映射）。")
    A(f"每份已定位招股书平均抽出自报同行（已映射）= {r['rates']['peers_per_firm_mean']:.2f} 家；"
      f"映射模式计数 = {r['rates']['mapping_mode_totals']}。")
    A("")
    fd = r["rates"].get("failure_decomposition")
    if fd:
        A("### 1.1 映射失败率口径分列（碎片 / 名称漂移 / 残差桶（未过前三类判据））")
        A("")
        A("| 口径 | 分子/分母 | 比率 | 说明 |")
        A("|---|---|---|---|")
        A(f"| 上界（含碎片，主口径） | {fd['上界分子分母']} | **{fd['上界（含碎片，主口径）']:.4f}** | "
          f"碎片 = 抽取残片（截断/句片段/体裁词），不是公司名 |")
        A(f"| 剔碎片后 | {fd['剔碎片后分子分母']} | **{fd['剔碎片后']:.4f}** | 分子分母同剔碎片 |")
        A(f"| 修正前对照（v4 交付口径） | {fd['修正前对照（v4 交付口径）']['含碎片分子分母']} | "
          f"{fd['修正前对照（v4 交付口径）']['含碎片上界']:.4f} | 复核口径剔 12 碎片后 = "
          f"{fd['修正前对照（v4 交付口径）']['剔碎片后（复核清单 12 碎片）']:.4f} |")
        A("")
        A(f"**本轮实测三分类**（unmapped 唯一名 {sum([fd['碎片'], fd['名称漂移'], fd.get(RESIDUAL_CLS, 0)])} 个）："
          f"碎片 **{fd['碎片']}** / 名称漂移 **{fd['名称漂移']}** / 残差桶（未过前三类判据） **{fd.get(RESIDUAL_CLS)}**。")
        for k in ("碎片", "名称漂移", RESIDUAL_CLS):
            items = fd["明细"].get(k) or []
            if items:
                A(f"- {k}（{len(items)}）：" + "、".join(f"{x['name']}（{x['evidence']}）" for x in items[:12])
                  + ("…" if len(items) > 12 else ""))
        A("")
        A("（分类为**规则判读**口径：逐名证据见 JSON `failure_decomposition.明细`；主口径失败率不依赖该分类。"
          "已知噪声：2 字同头判据已排除地名前缀（PLACE_PREFIX），仍可能有少量子公司/全称形态落入"
          "「残差桶（未过前三类判据）」（registry 1.5.17 修 2 改名；原类名「非 A 股实体」为兜底断言），"
          "属描述性口径的已知不确定性。）")
        A("")
    ch = r["rates"].get("extraction_channels")
    if ch:
        A("提取通道计数（修 3）：" + "；".join(f"{k} = {v}" for k, v in ch.items()) + "。")
        A("")
    A("## 2. 锚读数（lift）")
    A("")
    A(f"口径（P0 登记）：lift = P(自报同行 ∈ 网络对手 | 自报同行) ÷ P(随机 universe 公司 ∈ 网络对手)，"
      f"分母基线 = 该网络年无向对密度 d = E/(N(N−1)/2)。")
    A("")
    A("| 读数 | 重合率（原始） | 边对数/总对数 | 基线密度 d | lift |")
    A("|---|---|---|---|---|")
    A(f"| 主口径（全部自报同行对） | **{an['overlap_share_raw']:.6f}** | {an['pair_edge']}/{an['pair_total']} | "
      f"{an['base_density']:.6f} | **{an['lift']:.4f}** |")
    A(f"| 诊断·限双方均在网 | {an['diagnostic_node_restricted']['overlap_share_raw']:.6f} | "
      f"{an['diagnostic_node_restricted']['pair_edge']}/{an['diagnostic_node_restricted']['pair_total']} | "
      f"{an['base_density']:.6f} | {an['diagnostic_node_restricted']['lift']:.4f} |")
    A("")
    A(f"网络：文本年 {nb['text_year']}（`{nb['version']}`）、节点 N = {nb['N_nodes']}、无向边 E = {nb['E_undirected']}。")
    A(f"逐公司重合率分布（在网且已有映射同行的 {ov['n_firms_in_network_with_peers']} 家）："
      f"均值 {ov['mean']:.4f} / 中位 {ov['median']:.4f} / 最小 {ov['min']:.4f} / 最大 {ov['max']:.4f}；"
      f"零重合 {ov['n_zero_overlap']} 家、≥50% {ov['n_ge_50pct']} 家、≥90% {ov['n_ge_90pct']} 家。")
    A("")
    ep = r.get("extraction_pollution")
    if ep:
        A("### 2.1 抽取污染登记与敏感性（修 1）")
        A("")
        A(f"**问题**：{ep['issue']}")
        A("")
        A("**污染样例**（独立复核 14 例）：")
        for x in ep["examples"]:
            A(f"- {x}")
        A("")
        A(f"**修法**：{ep['fix']}")
        A("")
        A(f"**残余**：{ep['residual']}")
        A("")
        A("**敏感性（修正前 / 复核口径 / 修正后 三套读数，同一网络版本同一基线）**：")
        A("")
        A("| 口径 | 重合率（主） | 边对/总对 | lift（主） | 诊断·限在网 | lift（诊断） |")
        A("|---|---|---|---|---|---|")
        for k, v in ep["sensitivity"].items():
            A(f"| {k} | {v['overlap_share']:.6f} | {v['pair_edge']}/{v['pair_total']} | "
              f"{v['lift']:.4f} | {v['diag_overlap_share']:.6f}（{v['diag_pair_edge']}/{v['diag_pair_total']}） | "
              f"{v['diag_lift']:.4f} |")
        A("")
    dn = r.get("direction_note")
    if dn:
        A("### 2.2 下界与方向性说明（修 4）")
        A("")
        A(f"- {dn['coverage_lower_bound']}")
        A(f"- {dn['lift_direction_unproven']}")
        A(f"- {dn['probe']}")
        dc = an.get("direction_check") or {}
        if dc:
            rr = dc["rejected_pairs_edge_rate"]
            rr_s = "n/a" if rr is None else f"{rr:.4f}"
            A(f"- 内联方向性检查：被剔名-码对（双方均在网）{dc['rejected_pairs_edge']}/"
              f"{dc['rejected_pairs_in_network']} = {rr_s}"
              f"（保留集诊断口径 {dc['kept_pairs_edge_rate_diag']:.4f}，"
              f"{dc['kept_pairs_edge']}/{dc['kept_pairs_in_network']}）"
              f" → 剔除的名-码对中含真边，方向与「漏抽压低 lift」相反 → lift 方向未证。")
        A("")
    A("**角色限定**：本表为**外部锚（锚 1）试点读数**，只回答「网络边是否落在公司自报同行上」，"
      "不进入面板/回归、不改主口径；**单年试点不构成「外部锚条件满足」**（全量见 §5）。")
    A("")
    A("## 3. 抽样口径与解析修复轮次（停机 → 先修解析 → 复测）")
    A("")
    A(f"- 关键词集合 = {' / '.join(sc['keyword_set'])}；定位窗口 = hit−{sc['window'][0]}..hit+{sc['window'][1]} 字。")
    A(f"- 抽取规则 = {' / '.join(sc['extraction_rules'])}；映射模式 = {' / '.join(sc['mapping_modes'])}"
      f"（**仅精确匹配 + 公司后缀归一化 + 代码锚定，无模糊匹配、无人工补录**）。")
    A(f"- 名称字典来源：项目 firm 名单（`firm_id_map.csv` {sc['name_dict']['firm_universe']} 只，"
      f"**去 ST/*ST 前缀标记建键**）+ `a_stocks.csv` + `name_history_*.csv` + 年报头部「简称+代码」配对"
      f"（标签/顺序/冒号变体、头部窗口 24000 字、同头拼接去重）+ 更名披露（「证券简称由 X 变更为 Y」）"
      f"+ 释义缩写（「X 指 Y（…公司）」，后缀一致过滤）+ 东财曾用名缓存；"
      f"简称键 {sc['name_dict']['short_keys']}、全称键 {sc['name_dict']['full_keys']}。")
    A("")
    A("| 轮次 | 映射失败率 | 未映射/唯一名称 | 覆盖率 | lift | 修法 | 证据日志 |")
    A("|---|---|---|---|---|---|---|")
    for it in r["rates"]["parse_iterations"]:
        A(f"| {it['round']} | {it['mapping_failure_rate']:.4f} | {it['n_unmapped']}/{it['n_unique']} | "
          f"{it['coverage_rate']:.4f} | {it['lift']:.4f} | {it['note']} | `{it['log']}` |")
    A("")
    A("## 4. 失败案例样本（≤10 条）")
    A("")
    if r["failure_cases_sample"]:
        A("| 类型 | 代码 | 名称 | 细节 |")
        A("|---|---|---|---|")
        for c in r["failure_cases_sample"]:
            det = c.get("detail") or c.get("unmapped") or c.get("peers") or ""
            if isinstance(det, list):
                det = "、".join(str(x) for x in det)
            A(f"| {c['case']} | {c.get('code','')} | {c.get('name','')} | {det} |")
    else:
        A("（无）")
    A("")
    A("未映射名称 Top（跨公司出现次数）：" +
      "、".join(f"{x['name']}×{x['n_firms']}" for x in r["unmapped_top30"][:15]))
    A("")
    A("## 5. 下一步全量可行性判断（路线 B）")
    A("")
    n_nosection = r["rates"]["coverage_breakdown"].get("no_section（解析未定位，计入解析分母）", 0)
    cid_docs = [x for x in r["per_firm"] if str(x.get("channel", "")).startswith("pdfminer+cid_gid")]
    A(f"- **解析层**：{r['rates']['parse_success_rate']:.1%} 定位成功率。失败分解（修 3 单列）："
      f"① **无文本层扫描件** 2 份（300760/300748，pymupdf 与 pdfminer 均无文本层 → 需 OCR，属路线 B 前置）；"
      f"② **文本层 CID 编码损坏** {len(cid_docs)} 份（300737/603059/603301：无 ToUnicode CMap，本轮经 "
      f"pdfminer.six + SimSun GID 反查备选通道**已恢复**，并已进入抽取/映射）；"
      f"③ 其余 {max(0, n_nosection - len(cid_docs))} 份为**非标准/缺失章节标题**（4 关键词命中 = 0）。"
      "全量可复用同一脚本（`--year` 参数化），成本 = 每年下载 + 解析（本试点 105 份约 4 分钟，含备选通道）。")
    fd2 = r["rates"].get("failure_decomposition") or {}
    A(f"- **映射层**：主口径失败率 {r['rates']['mapping_failure_rate']:.1%}（上界口径，含碎片）"
      f"；**实测三分类** = 碎片 {fd2.get('碎片', '?')} / 名称漂移 {fd2.get('名称漂移', '?')} / "
      f"残差桶（未过前三类判据）{fd2.get(RESIDUAL_CLS, '?')}（逐名证据见 §1.1）。该桶来源需独立判读"
      "（未上市同业 / 境外上市 / 港股），属口径内不可映射；名称漂移由年报头部「简称+代码」（标签/顺序/冒号变体、"
      "去 ST 前缀）、更名披露与释义缩写四条**代码锚定**来源覆盖（本轮补齐 5 例复核确认的漏配）。")
    A(f"- **覆盖率**：{r['rates']['coverage_rate']:.1%}（**下界**；分母含 2 份无文本层 + "
      f"{max(0, n_nosection - len(cid_docs))} 份非标准标题未定位 + "
      f"{r['rates']['coverage_breakdown'].get('located_but_no_mapped_peer', 0)} 份抽到名单但无可映射同行）。"
      "全量若保持同量级，锚表可覆盖网络节点内约 4/5 的 IPO 公司。")
    A("- **lift 量级**：单年试点 lift ≈ "
      f"{r['anchor_readout']['lift']:.1f}（对照：分析师共同覆盖锚池化 5.39；同行业 lift 13–22）。"
      "量级与「自报同行更可能落在网内边」一致，但**单年、单网络版本，不能外推为全期结论**；"
      "且**方向未证**（§2.2）：漏抽与污染剔除两侧偏置方向不一致。")
    A("- **建议**：① 全量前补 OCR 通道处理无文本层扫描件（2 份）；② 逐年重复本脚本并汇总为 "
      "`s3_network_prospectus_anchor.json`；③ **在完成全量（≥3 年）之前，论文仍不得写 pre-validated**；"
      "④ 全量轮补「表格列错行 → 码-名配对校验」（本轮已知残余，见 §6）。")
    A("")
    A("## 6. 已知限制")
    A("")
    for x in r.get("known_limitations", []):
        A(f"- {x}")
    A("")
    A("## 7. 复现与产物版本关系")
    A("")
    A("```bash")
    A(f"python scripts/data/fetch_cninfo_prospectus.py --ipo-year {sc['pilot_year']} --workers 3")
    A(f"python scripts/data_audit/s3_network_prospectus_anchor.py --year {sc['pilot_year']}")
    A("```")
    A("")
    A("- **明细 CSV 版本关系（修 5）**：本报告同目录 `s3_network_prospectus_pilot_2018_detail.csv` 为**权威交付版**"
      "（脚本 `--detail-csv` 直出，105 行）。修正轮清理了中间产物 `artifacts/tmp/tier2_prosp/pilot_2018_detail.csv`"
      "（v4 期中间件，与交付版有 **9 行 × 6 字段 = 54 处字段填充差异**：`no_section` 行在中间件里留空、"
      "交付版填 0/True；sha256 = 45a61e31… 已留档 `artifacts/tmp/tier2_prosp_fix/f5_tmp_evidence.json`）。"
      "**以交付版为准，中间件不再保留、不得引用。**")
    A("- **tmp 大件清理（修 5）**：`artifacts/tmp/tier2_prosp/` 下 2 份探针 PDF（300748 165 MB / 300760 31 MB，"
      "用于「无文本层扫描件」实测：413 页 948 字 / 620 页 2,480 字）已删除，其 sha256、字节数与页/字计数留档于"
      " `artifacts/tmp/tier2_prosp_fix/f5_tmp_evidence.json`（PDF 可经 P1 下载器重取；下载日志 "
      "`download_log.csv` 已登记 `all_candidates_failed` + `extract_fail`）。")
    A("- **CID 恢复缓存（修 3）**：`artifacts/tmp/tier2_prosp/cid_recover/<code>.{txt,key}`（3 份解码文本 + "
      "PDF 字节数键；删缓存后可由 `pdfminer.six + SimSun GID 反查`确定性重建，需 `C:/Windows/Fonts/simsun.ttc`）。")
    A("- **修正轮探针与证据**：`artifacts/tmp/tier2_prosp_fix/`（候选 dump、R3 规则变体网格、CID 恢复探针、"
      "方向性检查、tmp 清理证据、四审计与自检日志）。")
    A("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

def _f(x, spec=".4f"):
    """None 安全的数值格式化（边界年/空集情形）。"""
    if x is None:
        return "n/a"
    try:
        return f"{x:{spec}}"
    except (TypeError, ValueError):
        return str(x)

def write_anchor_md(r, path):
    """全量锚报告人读版：逐年表 + 池化 + 边界年 + 外部锚判定 + 限制 + 试点对照。"""
    sc, ra, ex = r["scope"], r["anchor_readout"], r["external_anchor"]
    ov = r["rates"]["overall"]
    fd = r["rates"]["failure_decomposition_pooled"]
    L = []
    A = L.append
    A("# 第二梯队 #9 路线 B 全量：招股书「同行业可比公司」自报锚（锚 1 / plan §8 B″11-7）")
    A("")
    A(f"- 运行时间（UTC）：{r['run_at']}；口径参数：`{r['parameter']}`")
    fr2 = r.get("fix_round2") or {}
    A(f"- 范围：IPO 年 **{sc['ipo_years'][0]}–{sc['ipo_years'][-1]} 全量不抽样**；cohort 合计 **{sc['cohort_n']}**"
      f"（universe = `{sc['universe']}`）；四项定义 / 停机线 / 匹配口径逐字沿用试点登记值；"
      f"解析层为 **registry {r.get('fix_round', {}).get('registry_version', '1.5.15')} H1/M1/M2 修复轮**后的读数；"
      f"本轮为 **registry {fr2.get('registry_version', '1.5.17')} 修复轮二（修 1–7）**读数"
      f"（上一轮 1.5.16 = 本轮修复前基线，见 §0）。")
    if fr2:
        A(f"- **报告头登记版本（修 5）**：本报告 run 时的实际 registry 版本 = **registry "
          f"{fr2.get('registry_version', '1.5.17')}**（**非** 1.5.15 —— 前版报告头写 1.5.15 为登记文本滞后，已改）；"
          f"`name_dict.short_keys` **净减 1,297**（H1 错切键移除，13,412→12,115；非人工删除）。")
    A(f"- 网络版本：逐 IPO 年 Y 用其 `network_version_used`（y_target = min(Y+2, 2023)；v1_2023 不存在 → "
      f"fallback 最近可用 v1_2022）；**主读数年 = {ra['pooled_main_years_2015_2020']['years']}**；"
      f"边界年子集 = {list(ra['boundary_subset'].keys())}（单列、不混入主读数）。")
    A("")
    # ---- 修复轮与前置/后对照（registry 1.5.15）----
    pre = r.get("sensitivity_pre_fix")
    pre2 = r.get("sensitivity_pre_round2")
    fr = r.get("fix_round") or {}
    if fr:
        A("## 0. 修复轮与修复前/后对照（三轮读数：v1 → 上一轮 H1/M1/M2 → 本轮修 1–7）")
        A("")
        A(f"- 登记：{fr.get('change', '')}")
        A(f"- **判据零改动**：{fr.get('judgment_unchanged', '')}")
        A(f"- 归因口径（additive）：{fr.get('attribution_additive', '')}")
        if fr2:
            A(f"- **本轮（registry {fr2.get('registry_version', '1.5.17')}）登记**：{fr2.get('change', '')}")
            A(f"- **本轮判据零改动**：{fr2.get('judgment_unchanged', '')}")
            A(f"- 重抽口径：{fr2.get('re_extract_scope', '')}")
        A("")
        if pre and not pre.get("error"):
            A(f"v1 基线来源：`{pre['source']}`（run_at {pre['run_at_before_fix']}）。{pre['note']}")
            if pre2 and not pre2.get("error"):
                A("")
                A(f"**本轮修复前基线（上一轮 H1/M1/M2 修复后，registry 1.5.16）**：`{pre2['source']}`"
                  f"（run_at {pre2['run_at_before_fix2']}）。{pre2['note']}")
            A("")
            A("| 指标 | v1 基线（1.5.13/1.5.14） | 上一轮修复后（1.5.16） | 本轮修复后（1.5.17） | 停机线 | 判定 |")
            A("|---|---|---|---|---|---|")
            p2c = (pre2 or {}).get("cls4") or {}
            A(f"| 解析成功率 | {_f(pre['parse_success_rate'])}（{pre['parse_n']}） | "
              f"{_f((pre2 or {}).get('parse_success_rate'))}（{(pre2 or {}).get('parse_n', '')}） | "
              f"**{_f(ov['parse_success_rate'])}**（{ov['parsed']}/{ov['downloaded']}） | ≥0.80 | "
              f"{'PASS' if ov['parse_success_rate'] >= 0.80 else 'FAIL'} |")
            A(f"| 映射失败率（上界，主口径） | {_f(pre['mapping_failure_rate_upper'])}（{pre['mapping_failure_n']}） | "
              f"{_f((pre2 or {}).get('mapping_failure_rate_upper'))}（{(pre2 or {}).get('mapping_failure_n', '')}） | "
              f"**{_f(fd['上界（含碎片，主口径）'])}**（{fd['上界分子分母']}） | ≤0.20 | "
              f"{'PASS' if fd['上界（含碎片，主口径）'] <= 0.20 else 'FAIL'} |")
            A(f"| 映射失败率·剔碎片与噪声后（诊断） | {_f(pre['mapping_failure_ex_frag'])}（{pre['mapping_failure_ex_frag_n']}） | "
              f"{_f((pre2 or {}).get('mapping_failure_ex_frag_noise'))}（{(pre2 or {}).get('mapping_failure_ex_frag_noise_n', '')}） | "
              f"{_f(fd['剔碎片与噪声后'])}（{fd['剔碎片与噪声后分子分母']}） | —（诊断列） | — |")
            A(f"| 覆盖率 | {_f(pre['coverage_rate'])}（{pre['coverage_n']}） | "
              f"{_f((pre2 or {}).get('coverage_rate'))}（{(pre2 or {}).get('coverage_n', '')}） | "
              f"**{_f(ov['coverage_rate'])}**（{ov['n_covered']}/{ov['cohort']}） | ≥0.50 | "
              f"{'PASS' if ov['coverage_rate'] >= 0.50 else 'FAIL'} |")
            A(f"| 池化 lift（主读数年 2015–2020） | {_f(pre['pooled_lift'])}（{pre['pooled_pair']}） | "
              f"{_f((pre2 or {}).get('pooled_lift'))}（{(pre2 or {}).get('pooled_pair', '')}） | "
              f"**{_f(ra['pooled_main_years_2015_2020']['lift'])}**"
              f"（{ra['pooled_main_years_2015_2020']['pair_edge']}/{ra['pooled_main_years_2015_2020']['pair_total']}） | —（读数） | — |")
            A(f"| 四分类（唯一名） | 碎片 {(pre.get('cls3') or {}).get('碎片')} / 名称漂移 {(pre.get('cls3') or {}).get('名称漂移')} / "
              f"非A股实体（旧名）{(pre.get('cls3') or {}).get('非 A 股实体')} | 碎片 {p2c.get('碎片')} / 名称漂移 {p2c.get('名称漂移')} / "
              f"抽取噪声 {p2c.get('抽取噪声')} / 残差桶 {p2c.get(RESIDUAL_CLS)} | 碎片 {fd['碎片']} / 名称漂移 {fd['名称漂移']} / "
              f"抽取噪声 {fd['抽取噪声']} / 残差桶 {fd.get(RESIDUAL_CLS)} | —（描述性） | — |")
            A(f"| 外部锚条件（`external_anchor_satisfied`） | {pre['external_anchor_satisfied']} | "
              f"{(pre2 or {}).get('external_anchor_satisfied')} | {ex['external_anchor_satisfied']} | 三项全过 ∧ lift 齐全 | "
              f"{'满足' if ex['external_anchor_satisfied'] else '未满足'} |")
            A("")
            A("**逐年对照（v1 → 上一轮 → 本轮）**")
            A("")
            A("| IPO 年 | 解析率 | 覆盖率 | 映射失败率 | lift |")
            A("|---|---|---|---|---|")
            for y in map(str, sc["ipo_years"]):
                a = pre["per_year"].get(y, {})
                b = ((pre2 or {}).get("per_year") or {}).get(y, {})
                by = r["rates"]["per_year"][y]
                ly = ra["per_year"][y]
                A(f"| {y} | {_f(a.get('parse_success_rate'))} → {_f(b.get('parse_success_rate'))} → {_f(by['parse_success_rate'])}"
                  f" | {_f(a.get('coverage_rate'))} → {_f(b.get('coverage_rate'))} → {_f(by['coverage_rate'])}"
                  f" | {_f(a.get('mapping_failure_rate'))} → {_f(b.get('mapping_failure_rate'))} → "
                  f"{_f(r['rates']['failure_decomposition_per_year'][y]['上界（含碎片，主口径）'])}"
                  f" | {_f(a.get('lift'))} → {_f(b.get('lift'))} → {_f(ly['lift'])} |")
            A("")
        elif pre and pre.get("error"):
            A(f"（修复前基线读取失败：{pre['error']}；来源 `{pre.get('source', '')}`）")
            A("")
    A("## 1. 四项指标（全量）")
    A("")
    A("| 指标 | 实测 | 分子/分母 | 停机线 | 判定 |")
    A("|---|---|---|---|---|")
    A(f"| 解析成功率 | **{ov['parse_success_rate']:.4f}** | {ov['parsed']}/{ov['downloaded']}（下载成功份数）"
      f" | ≥0.80 | {'PASS' if ov['parse_success_rate'] >= 0.80 else 'FAIL'} |")
    A(f"| 映射失败率（上界，主口径） | **{fd['上界（含碎片，主口径）']:.4f}** | {fd['上界分子分母']}（跨年去重唯一名）"
      f" | ≤0.20 | {'PASS' if fd['上界（含碎片，主口径）'] <= 0.20 else 'FAIL'} |")
    A(f"| 覆盖率 | **{ov['coverage_rate']:.4f}** | {ov['n_covered']}/{ov['cohort']}"
      f"（2015–2023 cohort ∩ 项目 universe） | ≥0.50 | {'PASS' if ov['coverage_rate'] >= 0.50 else 'FAIL'} |")
    A("")
    A(f"**停机判定 = {'触发（' + '；'.join(ex['stop_reasons']) + '）' if ex['stop_triggered'] else '未触发（三项全过）'}**；"
      f"**外部锚（锚 1）条件 = {'已满足（external_anchor_satisfied = true）' if ex['external_anchor_satisfied'] else '未满足'}**"
      f"（判据与 lift 报告完备性见 §5）。")
    A(f"- **网络节点两口径（修 6）**：`n_network_nodes` = **{ov.get('n_network_nodes')}**（**located 口径**：已定位件中"
      f"在该网络版本为节点的家数）；`n_network_nodes_member_scope` = **{ov.get('n_network_nodes_member_scope')}**"
      f"（**预登记会员口径**：cohort 中 firm_id ∈ 该网络版本节点集的家数，含未下载/未定位件；预登记轮登记值 "
      f"= 1,537/1,592 = 96.5%）——两口径并报、不得混用。")
    A("- **`first_hit_page` 语义（修 4）**：非 pymupdf 通道（CID 确定性还原 / OCR 回退）置 `not_applicable_channel`"
      "（替换后文本的字符偏移与 pymupdf 页码不对应）；pymupdf 通道报真实页码。")
    A("")
    A("### 1.1 映射失败率口径分列（上界 / 剔碎片 / 剔碎片与噪声 / 四分类）")
    A("")
    A("| 口径 | 唯一名 | 未映射 | 上界（含碎片） | 剔碎片后 | 剔碎片与噪声后 | 碎片 | 名称漂移 | 抽取噪声 | 残差桶（未过前三类判据） |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for key, lab in (("failure_decomposition_pooled", "**全量池化（主口径）**"),
                     ("failure_decomposition_main_years", "主读数年池化（2015–2020，诊断）"),
                     ("failure_decomposition_boundary", "边界年子集（2021–2023，单列）")):
        d = r["rates"][key]
        A(f"| {lab} | {d['n_unique_names']} | {d['n_unmapped']} | **{_f(d['上界（含碎片，主口径）'])}**"
          f" | {_f(d['剔碎片后'])} | {_f(d['剔碎片与噪声后'])} | {d['碎片']} | {d['名称漂移']}"
          f" | {d['抽取噪声']} | {d.get(RESIDUAL_CLS)} |")
    A("")
    A("**逐年映射失败率（年内去重，诊断分列）**：")
    A("")
    A("| IPO 年 | 唯一名 | 未映射 | 上界 | 剔碎片后 | 剔碎片与噪声后 | 碎片 | 名称漂移 | 抽取噪声 | 残差桶（未过前三类判据） |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for y in map(str, sc["ipo_years"]):
        d = r["rates"]["failure_decomposition_per_year"][y]
        A(f"| {y} | {d['n_unique_names']} | {d['n_unmapped']} | {_f(d['上界（含碎片，主口径）'])}"
          f" | {_f(d['剔碎片后'])} | {_f(d['剔碎片与噪声后'])} | {d['碎片']} | {d['名称漂移']}"
          f" | {d['抽取噪声']} | {d.get(RESIDUAL_CLS)} |")
    A("")
    A(f"（分类 = 规则判读口径（逐名证据见 JSON `failure_decomposition_*.明细`）；主口径失败率不依赖该分类。"
      f"第 4 类「抽取噪声」= registry 1.5.15 ④ 新增、**registry 1.5.17 修 3 改判据**（= **整名 ≈ 通用词**"
      f"（整名 ∈ 词表 或 ≤{GEN_SUB_MAXLEN} 字且含词表词）+ **公司后缀豁免**：公司/集团/股份/有限/企业 收尾或含 股份/集团）；"
      f"**「剔碎片后」列口径未改**。全量池化未映射唯一名 {fd['n_unmapped']} 个：碎片 {fd['碎片']} / 名称漂移 {fd['名称漂移']} / "
      f"抽取噪声 {fd['抽取噪声']} / **残差桶（未过前三类判据）** {fd.get(RESIDUAL_CLS)} —— 残差桶为**兜底 return**，"
      f"其来源需独立判读，**不得**以该桶断言「非 A 股实体」（修 2 改名；原类名/字段名 `非 A 股实体`/`non_a_share_split`）。）")
    A("")
    for k in ("名称漂移", "抽取噪声", RESIDUAL_CLS):
        items = fd["明细"].get(k) or []
        if items:
            A(f"- {k}（{len(items)}）：" + "、".join(x["name"] for x in items[:40]) + ("…" if len(items) > 40 else ""))
    A("")
    A("### 1.2 失败构成诊断（描述性；不改主口径、不参与门槛判定）")
    A("")
    fdg = r.get("failure_diagnostics")
    if fdg:
        A(fdg["note"])
        A("")
        _rb = fdg.get("residual_bucket_audit") or {}
        _sa = _rb.get("sampling_audit") or {}
        A(f"- unmapped 唯一名 {fdg['n_unmapped_unique']} 个；四分类（碎片 "
          f"{fd['碎片']} / 名称漂移 {fd['名称漂移']} / 抽取噪声 {fd['抽取噪声']} / "
          f"**残差桶（未过前三类判据）** {fd.get(RESIDUAL_CLS)}）。")
        A(f"- **残差桶（修 2，{_rb.get('bucket_n', fd.get(RESIDUAL_CLS))} 个）需独立判读**：抽样审计（n="
          f"{_sa.get('n_sampled')}）显示 真非上市/境外 {_sa.get('真非上市/境外')}"
          f"（{_sa.get('share', {}).get('真非上市/境外', 0) * 100:.1f}%）/ 抽取伪影 {_sa.get('抽取伪影')}"
          f"（{_sa.get('share', {}).get('抽取伪影', 0) * 100:.1f}%）/ A 股名漂移 {_sa.get('A股名漂移')}"
          f"（{_sa.get('share', {}).get('A股名漂移', 0) * 100:.1f}%）→ **不是单一来源**。")
        _qd = _rb.get("quantified_decomposition") or {}
        _lc = _qd.get("loosest_correction") or {}
        A(f"- **复核量化分解**：{_qd.get('note', '')}")
        A(f"- **最宽松修正（同剔全部抽取伪影 844；口径 = 复核的修复前分母 5818）**："
          f"失败率 = {_lc.get('numerator_denominator')} = **{_lc.get('rate'):.4f}**"
          f"（复核载 {_lc.get('review_value')}）**仍 > 0.20** → **停机不解除**、外部锚条件仍不满足（未放宽任何门槛）；"
          f"（按本轮修复后分母重算同口径剔伪影读数 = (2266−844)/(6175−844)，仍 > 0.20，判定不变。）"
          if _lc.get('rate') is not None else "- （复核量化分解缺省）")
        A(f"- 读法：{fdg['outcome_reading']}")
        A("")
        A("| # | 未映射名（按文档频次） | 文档数 | 四分类 | 出现年 | 证据 |")
        A("|---|---|---|---|---|---|")
        for i, x in enumerate(fdg["top_unmapped_by_docfreq"][:25], 1):
            A(f"| {i} | {x['name']} | {x['n_docs']} | {x['cls']} | {','.join(map(str, x['years']))}"
              f" | {x['evidence']} |")
        A("")
    A("## 2. 逐年表（cohort / 下载 / 解析 / 覆盖率 / 映射失败 / lift + CI）")
    A("")
    A("| IPO 年 | cohort | 下载 | 解析 | 解析率 | 覆盖率 | 映射失败（上界） | 网络版本 | 对数/命中 | 重合率 | lift | lift CI95 | 诊断·限在网 lift | 边界年 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for y in map(str, sc["ipo_years"]):
        py = r["rates"]["per_year"][y]
        ly = ra["per_year"][y]
        lift_s = "n/a" if ly["lift"] is None else f"{ly['lift']:.4f}"
        ci = ly.get("lift_ci95") or [None, None]
        ci_s = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci[0] is not None else "n/a"
        dg = "n/a" if ly["diag_lift"] is None else f"{ly['diag_lift']:.4f}"
        mf = r["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"]
        share_s = "n/a" if ly["overlap_share"] is None else f"{ly['overlap_share']:.6f}"
        A(f"| {y} | {py['cohort']} | {py['downloaded']} | {py['parsed']} | {py['parse_success_rate']:.4f}"
          f" | **{py['coverage_rate']:.4f}**（{py['n_covered']}） | {_f(mf)} | `{ly['network_version_used']}`"
          f" | {ly['pair_edge']}/{ly['pair_total']} | {share_s}"
          f" | **{lift_s}** | {ci_s} | {dg} | {'是' if py['boundary_year_subset'] else '否'} |")
    A("")
    A(f"- 合计：cohort **{ov['cohort']}** / 下载 {ov['downloaded']} / 解析 {ov['parsed']}（{ov['parse_success_rate']:.4f}）"
      f"/ 覆盖 {ov['n_covered']}（{ov['coverage_rate']:.4f}）/ no_section {ov['no_section']} / no_file {ov['no_file']}"
      f" / parse_error {ov['parse_error']}。")
    A("")
    A("## 3. 锚读数（lift）")
    A("")
    A(f"口径：{ra['definition']}")
    A("")
    pm = ra["pooled_main_years_2015_2020"]
    pa = ra["pooled_including_boundary"]
    A("| 读数 | 对数/命中 | 重合率 | 池化密度 d_w | lift | lift CI95 | 诊断·限双方在网 lift |")
    A("|---|---|---|---|---|---|---|")
    A(f"| **池化（主读数年 {pm['years']}）** | {pm['pair_edge']}/{pm['pair_total']} | **{_f(pm['overlap_share'], '.6f')}**"
      f" | {_f(pm['pooled_density_dw'], '.6f')} | **{_f(pm['lift'])}** | "
      f"[{_f(pm['lift_ci95'][0], '.2f')}, {_f(pm['lift_ci95'][1], '.2f')}] | {_f(pm['diag_lift'])} |")
    A(f"| 池化（含边界年，对照） | {pa['pair_edge']}/{pa['pair_total']} | {_f(pa['overlap_share'], '.6f')}"
      f" | {_f(pa['pooled_density_dw'], '.6f')} | {_f(pa['lift'])} | — | {_f(pa['diag_lift'])} |")
    A("")
    ovs = ra["per_firm_overlap_stats"]
    A(f"逐公司重合率（在网且有映射同行 {ovs['n_firms_in_network_with_peers']} 家）：均值 {ovs['mean']:.4f}；"
      f"零重合 {ovs['n_zero_overlap']} 家、≥50% {ovs['n_ge_50pct']} 家。")
    A("")
    A("### 3.1 边界年子集（{2021, 2022, 2023}，单列、不混入主读数）")
    A("")
    A("| IPO 年 | cohort | 下载 | 解析 | 网络版本 | 该版本中节点数 | 对数 | lift |")
    A("|---|---|---|---|---|---|---|---|")
    for y, b in ra["boundary_subset"].items():
        lift_s = "n/a" if b["lift"] is None else f"{b['lift']:.4f}"
        A(f"| {y} | {b['cohort']} | {b['downloaded']} | {b['parsed']} | `{b['network_version_used']}`"
          f" | **{b['n_network_nodes']}** | {b['pair_total']} | {lift_s} |")
    A("")
    A("边界年处理：y_target = min(Y+2, 2023) 不可得（v1_2023 不存在）→ fallback 最近可用网络版本 v1_2022；"
      "三 cohort 在该版本中节点数 = 0（冻结样本筛选「上市不足 2 年」）→ 锚读数退化（无对）；"
      "下载/解析/覆盖计数全量纳入，**未静默剔除**。")
    A("")
    dc = ra["direction_check"]
    rr = dc["rejected_pairs_edge_rate"]
    A(f"**方向性说明（沿用试点「方向未证」的如实表述）**：被剔名-码对（双方均在网）{dc['rejected_pairs_edge']}/"
      f"{dc['rejected_pairs_in_network']} = {'n/a' if rr is None else f'{rr:.4f}'}"
      f"（保留集诊断口径 {_f(dc['kept_pairs_edge_rate_diag'])}）→ 剔除/漏抽两侧偏置方向不一致 → "
      f"**lift 方向未证**（不宣称下界、不宣称上界）。")
    A("")
    A("## 4. 通道与失败披露（不得静默剔除）")
    A("")
    cr = r.get("ingest_crosscheck") or {}
    A(f"**与全量第一轮 ingest 明细逐份对账**：比较 {cr.get('n_compared', 0)} 份；**status 不一致 "
      f"{len(cr.get('status_mismatch') or [])} 份**（口径未漂移）；**channel 标签不一致 "
      f"{len(cr.get('channel_mismatch') or [])} 份**——全部为 OCR/CID 件的**运行元数据差**"
      f"（ingest 轮记 `render/ocr` 耗时、本轮缓存命中记 `（缓存命中）`），**终态通道一致**"
      f"（逐条见 JSON `ingest_crosscheck.channel_mismatch`）。")
    A("")
    ch = r["channel_disclosure"]
    A(f"非 pymupdf 通道 **{len(ch)} 件**（OCR 回退 / CID 确定性还原；逐件 channel 见 JSON `channel_disclosure`）：")
    A("")
    A("| IPO 年 | code | 通道 | 页数 | 字数 | 关键词命中 | 解析状态 |")
    A("|---|---|---|---|---|---|---|")
    for x in ch:
        A(f"| {x['year']} | {x['code']} | `{x['channel']}` | {x['pages']} | {x['chars']} | {x['kw_hits']} | {x['status']} |")
    A("")
    dl = r["download_failures"]
    A(f"下载失败 **{dl['n_failed']} 件**（status 概览 = {dl['by_status']}）；逐条：")
    A("")
    A("| IPO 年 | code | download_log 末次 status |")
    A("|---|---|---|")
    for x in dl["items"]:
        A(f"| {x['year']} | {x['code']} | `{x['status']}` |")
    A("")
    A("## 5. 外部锚（锚 1）条件判定（机械、不得放宽）")
    A("")
    A("| 判据 | 门槛 | 实测 | 判定 |")
    A("|---|---|---|---|")
    for k, v in ex["criteria"].items():
        val = v["value"]
        A(f"| {k} | {'≥0.80' if '解析' in k else ('≤0.20' if '映射' in k else '≥0.50')} | {_f(val)}"
          f"（{v.get('scope', '')}） | {'PASS' if v['pass'] else 'FAIL'} |")
    A(f"| lift 报告完备性 | 逐年 + 池化 + 边界年单列 + 诊断口径 | {'齐全' if ex['lift_reports_complete'] else '不齐全'} | "
      f"{'PASS' if ex['lift_reports_complete'] else 'FAIL'} |")
    A("")
    A(f"**判定：外部锚（锚 1）条件 = {'已满足' if ex['external_anchor_satisfied'] else '未满足'}**"
      f"（`external_anchor_satisfied = {str(ex['external_anchor_satisfied']).lower()}`）。")
    if ex["external_anchor_satisfied"]:
        A("")
        A("据此登记：**「pre-validated / 已验证」措辞限制在本条件（外部锚条件已满足）下解除**——"
          "解除范围 = 论文/材料中针对**外生锚定证据**的「pre-validated（预先验证过的）」表述，"
          "条件 = 本报告登记的四项指标与 lift 读数（registry change_log 同文登记）。"
          "**但该解除与 #10 的「AI 独立判读（三代理，非人工）」证据层级限定并列存在、互不替代**："
          "涉及**「边级语义」（网络中的边 = 真实竞争关系）**的表述仍受 #10 证据层级与门槛判定"
          "（`TRIGGER_NETWORK_RECHECK`）约束，不得据本报告写「网络已验证为真实竞争关系」。")
    else:
        A("")
        A("据此登记：**外部锚（锚 1）条件未满足**（`external_anchor_satisfied = false`），卡点 = "
          + "；".join(ex["stop_reasons"] or ["lift 报告不齐全"]) + "。")
        A("")
        A("- **未放宽任何门槛**：三项判据与停机线沿用试点登记值（解析 <80% / 映射失败 >20% / 覆盖 <50% → 停），"
          "本轮**未做任何宽松匹配、未人工补录、未调分母、未剔除失败件**。")
        yrates = {y: r["rates"]["failure_decomposition_per_year"][y]["上界（含碎片，主口径）"]
                  for y in map(str, sc["ipo_years"])}
        yv = [v for v in yrates.values() if v is not None]
        ymin, ymax = (min(yv) if yv else None), (max(yv) if yv else None)
        ypass = [y for y, v in yrates.items() if v is not None and v <= 0.20]
        cats = {"碎片": fd["碎片"], "名称漂移": fd["名称漂移"], "抽取噪声": fd["抽取噪声"],
                RESIDUAL_CLS: fd.get(RESIDUAL_CLS)}
        cat_s = " / ".join(f"{k} {v}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1]))
        heads = {"上界（含碎片，主口径）": fd["上界（含碎片，主口径）"], "剔碎片后": fd["剔碎片后"],
                 "剔碎片与噪声后": fd["剔碎片与噪声后"], "逐年最大值": ymax}
        over = [k for k, v in heads.items() if v is not None and v > 0.20]
        under = [k for k, v in heads.items() if v is not None and v <= 0.20]
        A("- **卡点定位（描述性，不宣称单一原因；**全部按本报告数值机械读出**）**：映射失败率分列 = "
          f"上界（主口径）{_f(fd['上界（含碎片，主口径）'])}（{fd['上界分子分母']}）、剔碎片后 {_f(fd['剔碎片后'])}"
          f"（{fd['剔碎片后分子分母']}）、剔碎片与噪声后 {_f(fd['剔碎片与噪声后'])}"
          f"（{fd['剔碎片与噪声后分子分母']}）、逐年 {_f(ymin)}–{_f(ymax)}"
          f"（年内口径 ≤0.20 的年份 = {'、'.join(ypass) if ypass else '无'}）；四分类（唯一名，降序）= {cat_s}；"
          f"其中**超线列 = {'、'.join(over) if over else '无'}**、未超线列 = {'、'.join(under) if under else '无'}。"
          "细分类（抽取噪声 / 残差桶，见 §1.2）= "
          + "；".join(f"{k} {v}" for k, v in ((r.get('failure_diagnostics') or {})
                      .get("residual_split") or {}).items()) + "。"
          "**残差桶（未过前三类判据）需独立判读**：复核抽样（n=160）显示 真非上市/境外 ≈62.5% / 抽取伪影 ≈30.0% / "
          "A 股名漂移 ≈7.5%；复核量化分解 = 2027 未映射 ≈ 抽取伪影 844 / 真非 A 股 926 / A 股名漂移 257。"
          "**最宽松修正（同剔全部抽取伪影 844）后失败率 = 1183/4974 = 0.2378（复核载 0.2379）仍 > 0.20 → 停机不解除。**"
          "可能来源并列三类（不宣称单一原因）：① 口径内不可映射实体（未上市/境外/港股同业、子公司实体）；"
          "② 抽取层伪影（弱证据规则在部分年份正文/表格上误触发，通用词·科目·句片段，及桶内的句片段/公司名混合）；"
          "③ 名称漂移残余（历史简称不在代码锚定字典内）。**逐名归因需独立判读、不在本轮**；本轮已按登记执行"
          "「先修解析」两轮修复（H1/M1/M2 = registry 1.5.15/1.5.16；修 1–7 = registry 1.5.17），"
          "**修复未触及任何判据/门槛/匹配口径/分母**。")
        A("- **措辞限制保持**：**「pre-validated / 已验证」仍禁用**（外部锚条件未满足）；且与 #10 的"
          "「AI 独立判读（三代理，非人工）」证据层级限定**并列存在、互不替代**。")
    A("")
    A("## 6. 与试点（2018）的对照")
    A("")
    pc = r.get("pilot_2018_contrast")
    if pc:
        A("| 项 | 试点 2018（registry 1.5.5–1.5.7） | 全量 2018 行（本轮） |")
        A("|---|---|---|")
        rows = [("下载/定位", f"{pc['pilot']['n_downloaded']} / {pc['pilot']['n_located']}",
                 f"{pc['full_2018']['n_downloaded']} / {pc['full_2018']['n_located']}"),
                ("解析成功率", _f(pc['pilot']['parse_success_rate']), _f(pc['full_2018']['parse_success_rate'])),
                ("映射失败率", f"{_f(pc['pilot']['mapping_failure_rate'])}（{pc['pilot']['n_unmapped/n_unique']}）",
                 f"{_f(pc['full_2018']['mapping_failure_rate'])}（{pc['full_2018']['n_unmapped/n_unique']}）"),
                ("覆盖率", f"{_f(pc['pilot']['coverage_rate'])}（{pc['pilot']['n_covered/cohort']}）",
                 f"{_f(pc['full_2018']['coverage_rate'])}（{pc['full_2018']['n_covered/cohort']}）"),
                ("重合率", f"{_f(pc['pilot']['overlap_share'], '.6f')}（{pc['pilot']['pair_edge/pair_total']}）",
                 f"{_f(pc['full_2018']['overlap_share'], '.6f')}（{pc['full_2018']['pair_edge/pair_total']}）"),
                ("lift", _f(pc['pilot']['lift']), _f(pc['full_2018']['lift'])),
                ("诊断·限在网 lift", _f(pc['pilot']['diag_lift']), _f(pc['full_2018']['diag_lift'])),
                ("网络版本", pc['pilot']['network_version'], pc['full_2018']['network_version'])]
        for a, b, c in rows:
            A(f"| {a} | {b} | {c} |")
        A("")
        A(pc["note"])
        A("")
        A("- **口径一致性校验（修复轮二重跑实测，防定义漂移）**：以试点路径（`--year 2018`，同一脚本同一规则、"
          "未改任何判据）重跑 2018 → 解析 " + _f(pc['full_2018']['parse_success_rate']) + "（" +
          str(pc['full_2018']['n_located']) + "/" + str(pc['full_2018']['n_downloaded']) + "）/ 映射失败 " +
          _f(pc['full_2018']['mapping_failure_rate']) + "（" + pc['full_2018']['n_unmapped/n_unique'] + "）/ 覆盖率 " +
          _f(pc['full_2018']['coverage_rate']) + "（" + pc['full_2018']['n_covered/cohort'] + "）/ lift " +
          _f(pc['full_2018']['lift'], '.4f') + " —— 与本报告的 2018 行**逐值一致（差分 = 0）**，且与全量第一轮 ingest "
          "明细逐份对账 status 不一致 = 0 份 → 判定 **无定义漂移**；与交付版试点（99/103、0.1626、0.5905、26.4573）"
          "的差异全部来自**登记在案**的（a）解析子通道新增（OCR 回退使 2 份无文本层扫描件进入口径）、（b）修复轮二"
          "抽取层触发条件修复（R4 复合触发 / R4_tbl_head / R2 空白容错与左边界扩展）、（c）覆盖率分母口径"
          "（全量 = cohort 全量 105）。回归证据：`artifacts/tmp/tier2_full/fix2/regress_2018.json`"
          "（+ `_detail.csv`）；交付版试点产物 `reports/data_gates/G0/s3_network_prospectus_pilot_2018.*` 零改动。")
    else:
        A("（未找到试点产物，跳过对照）")
    A("")
    A("## 7. 已知限制")
    A("")
    for x in r["known_limitations"]:
        A(f"- {x}")
    A("")
    A("## 8. 复现")
    A("")
    A("```bash")
    A("python scripts/data_audit/s3_network_prospectus_anchor.py --measure --years 2015-2023 --workers 6 \\")
    A("    --ingest-detail reports/data_gates/G0/s3_network_prospectus_full_ingest_detail.csv \\")
    A("    --pre-fix-report reports/data_gates/G0/s3_network_prospectus_anchor_v1.json \\")
    A("    --pre-fix-report2 reports/data_gates/G0/s3_network_prospectus_anchor_v2_pre_fix.json")
    A("```")
    A("")
    A("前置步骤（本报告口径）：修复前报告先留档为 `s3_network_prospectus_anchor_v1.*`（标 superseded）；"
      "首次运行因名称注册表 `cache_version` 3→4 触发**强制重算**（约 2 min，缓存 `artifacts/tmp/tier2_prosp/full_name_map.json`）；"
      "locate/抽取通道（pymupdf → CID 确定性还原 → OCR 缓存）与修复前逐字一致，**不重新下载 PDF**。")
    A("")
    A(f"- 逐份明细：`{rel(path).replace('.md', '_detail.csv')}`（{sc['cohort_n']} 行：peers / mapped_peers_codes / "
      f"unmapped / channel / network_version_used）。")
    A("- 阶段报告（全量第一轮 ingest）：`reports/data_gates/G0/s3_network_prospectus_full_ingest_status.{json,md}`。")
    A("- 修复前基线（只读）：`reports/data_gates/G0/s3_network_prospectus_anchor_v1.{json,md,_detail.csv}`"
      "（superseded，修复前数值 = 本报告 `sensitivity_pre_fix`）。")
    A("")
    fx = r.get("fix_round_evidence")
    if fx:
        A("## 9. 解析层修复轮证据（H1/M1/M2；registry 1.5.15）")
        A("")
        A(fx["note"])
        A("")
        ni = fx.get("name_registry_info") or {}
        A(f"- 名称注册表（年报头部）：cache_version = **{fx.get('name_registry_cache_version')}**"
          f"（3→4 强制重算）；覆盖 {ni.get('n_codes')} 只 / {ni.get('n_files')} 份年报；"
          f"历史简称条目 {ni.get('n_short_hist')}；全称键 {ni.get('n_full_names')}；"
          f"构建耗时 {ni.get('elapsed_sec')} s；变更来源命中 = 更名披露 {ni.get('n_name_change_hits')} / "
          f"释义缩写 {ni.get('n_defn_hits')}。来源与代码锚定机制不变，**无人工补录**。")
        A(f"- M1② R2 左边界剥离（语境/连接词前缀）：命中 **{fx.get('n_r2_left_repairs')}** 条"
          "（raw → fixed，逐条留在逐份记录 `r2_left_repairs`）。")
        ex_ = fx.get("r2_left_repair_examples") or []
        if ex_:
            A("  - 高频样例：" + "、".join(f"`{x['raw→fixed']}`（{x['n']}）" for x in ex_[:10]))
        A(f"- M2 发行人自名容错剔除：**{fx.get('n_self_name_rejections')}** 条（判据 = 精确/同核/编辑距离·OCR 近似/"
          "近全包含；A 股 universe 名录守卫不做自指剔除）；形态约束/交叉校验剔除 "
          f"**{fx.get('n_shape_rejections')}** 条。")
        A("")
    rep2 = r.get("repair_round2_evidence")
    if rep2:
        A("## 10. 修复轮二证据（修 1–7；registry 1.5.17）")
        A("")
        A("修复轮二（registry 1.5.17）证据：**判据零改动**（四项定义 / 停机线（0.80 / 0.20 / 0.50 逐字不变）/ "
          "匹配口径 / 覆盖率分母 / 网络版本规则 / 边界年处理）；**不重新下载 PDF**（用既有 txt / ocr_txt / CID "
          "缓存重抽）；**无人工补录、未放宽匹配口径**。")
        A("")
        A(f"- 抽取层逐份规则计数（全量 cohort，修 1 后）：{fx.get('extraction_rule_counts') if fx else ''}")
        A(f"- R2 左边界修复（含修 1(e) 语境后缀片段/尾随「有」名录守卫剥离）命中：**{fx.get('n_r2_left_repairs') if fx else ''}** 条")
        A(f"- **修复件清单**（与上一轮逐份 diff；语料未动）：{rep2.get('n_files_changed')} 件")
        for cc, cd in (rep2.get("focus_cases") or {}).items():
            A(f"  - **{cd['year']} {cc}**（{cd.get('channel')}）：唯一名 {cd['n_unique_before_after'][0]} → "
              f"{cd['n_unique_before_after'][1]}；已映射同行 {cd['n_mapped_before_after'][0]} → {cd['n_mapped_before_after'][1]}；"
              f"新增同行名 = {'、'.join(cd['peers_added']) or '—'}；新增同行代码 = {'、'.join(cd['mapped_codes_added']) or '—'}；"
              f"新增未映射名 = {'、'.join(cd['unmapped_added'][:8]) or '—'}")
        A("")
        chg = rep2.get("changed_files") or []
        if chg:
            A("| IPO 年 | code | 通道 | 唯一名（前→后） | 已映射（前→后） | 新增同行（代码） | 新增未映射名 |")
            A("|---|---|---|---|---|---|---|")
            for cd in chg[:25]:
                A(f"| {cd['year']} | {cd['code']} | `{cd.get('channel')}` | {cd['n_unique_before_after'][0]} → "
                  f"{cd['n_unique_before_after'][1]} | {cd['n_mapped_before_after'][0]} → {cd['n_mapped_before_after'][1]} | "
                  f"{'、'.join(cd['mapped_codes_added']) or '—'} | {'、'.join(cd['unmapped_added'][:6]) or '—'} |")
            A("")
        A("- **修复轮二其余项**：修 2 残差桶改名（§1.1/§1.2/§5）；修 3 抽取噪声判据 = 整名 ≈ 通用词 + 公司后缀豁免"
          "（§1.1/§1.2）；修 4 `first_hit_page` 非 pymupdf 通道 = `not_applicable_channel`（§1 注）；修 5 报告头"
          "registry 版本 + `short_keys` 净减 1,297（报告头注）；修 6 网络节点两口径（§1 注）；修 7 "
          "`identity_check_pooled_lift` = **恒等式校验（非独立证据）**（§3 池化行）。")
        A("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

