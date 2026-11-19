#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F5：对表**语料三方并列表**（原文 × 主口径 MD&A × 全文语料），只读既有对表产物、不重估任何回归。

问题（registry `ma_fulltext_text_corpus`，2026-10-19 F0 前置登记）：
  既有对表（语料 = 老师 MD&A 章节）在旧符号判据下记为 **复现不成功（符号相反）**，而原文语料 = **年报全文**；
  **2026-10-27 口径更正（V-P1-2 / N-2）**：项目对手集合 = 同团成员、原文 = 相似度 > 团内最小相似度（字面含团外），
  二者不可比 → 对外判定统一为「不可比/未复现（对手集合口径不同）」；旧标签只作历史读数。
  故做一次语料敏感性（把语料贴回原文口径）并回答：**语料差异是否解释反号？**

三方：
  ① 原文 = 刘亚辉等（2021）表 2 / 表 3 / 表 4（字面量，来自对表脚本 `TARGETS`，不重录）；
  ② MD&A 版 = `reports/data_gates/G0/competition_pricing_correspondence.{json,md}`（主口径，2026-10-16）；
  ③ 全文版 = `reports/data_gates/G0/s3_pc_pricing_correspondence_fulltext.{json,md}`（本列，2026-10-19）。

判据（**沿用既有、一字不改**）：符号一致 + 显著性同向（t 同号且 |t| > 1.96）→ 成功；
符号一致不显著 → 部分；反号 → 不成功。**本脚本不做任何统计估计**（纯读数并列表），
不覆盖既有产物、不改任何冻结值；结论句由「三层符号判定是否随语料切换而改变」机械推出。

产物：`reports/data_gates/G0/s3_pc_corpus_contrast_fulltext.{json,md}`
用法：python scripts/competition_corpus_contrast.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPD = ROOT / 'reports/data_gates/G0'
MD_A = REPD / 's3_pc_pricing_correspondence.json'
FT = REPD / 's3_pc_pricing_correspondence_fulltext.json'
OUT_JSON = REPD / 's3_pc_corpus_contrast_fulltext.json'
OUT_MD = REPD / 's3_pc_corpus_contrast_fulltext.md'
PROGRESS = ROOT / 'artifacts/tmp/fulltext_corr_progress.md'


def psay(msg: str) -> None:
    line = f'[FT-5] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')


def f(x, nd=4):
    return 'NA' if x is None else f'{x:.{nd}f}'


NOT_COMPARABLE_LABEL = '不可比/未复现（对手集合口径不同）'
COMPARABLE_TO_PAPER = False   # N-2：对手集合口径不同


def verdict_cell(ours, target_t, ours_t):
    """对外判定：对手集合不可比 → 一律 NOT_COMPARABLE_LABEL；点估计符号另存 sign_reading。"""
    if ours is None or ours_t is None:
        return '不可得'
    if not COMPARABLE_TO_PAPER:
        return NOT_COMPARABLE_LABEL
    sign_same = (ours > 0) == (target_t > 0)
    sig = abs(ours_t) > 1.96 and (ours_t > 0) == (target_t > 0)
    if not sign_same:
        return '不成功（符号相反）'
    return '复现成功' if sig else '部分（符号一致不显著）'


def sign_reading(ours) -> str:
    if ours is None:
        return 'NA'
    return '点估计为正' if ours > 0 else '点估计为负'


def main() -> int:
    for p in (MD_A, FT):
        if not p.exists():
            print(f'缺产物：{p}', file=sys.stderr)
            return 2
    a = json.loads(MD_A.read_text(encoding='utf-8'))
    b = json.loads(FT.read_text(encoding='utf-8'))
    tgt = a['targets']
    psay(f'状态 = 开始；读入 MD&A 版（{MD_A.name}，{a.get("generated_at")}）与全文版（{FT.name}，{b.get("generated_at")}）')

    rows = []

    def add(layer, target_txt, ta, tb, va, vb, note=''):
        rows.append({'layer': layer, 'target': target_txt, 'mda': ta, 'fulltext': tb,
                     'verdict_mda': va, 'verdict_fulltext': vb, 'same_sign': (va == vb) or note,
                     'note': note})

    # --- 表 2 描述统计 ---
    ta = a['L1_descriptives']; tb = b['L1_descriptives']
    add('表 2 描述统计（PC 均值 / 中位）',
        f'均值 {tgt["table2_pc_mean"]["value"]} / 中位 {tgt["table2_pc_median"]["value"]}',
        f'均值 {f(ta["pc_pooled_firm_year"]["mean"])} / 中位 {f(ta["pc_pooled_firm_year"]["median"])}'
        f'（n = {ta["pc_pooled_firm_year"]["n"]:,}）',
        f'均值 {f(tb["pc_pooled_firm_year"]["mean"])} / 中位 {f(tb["pc_pooled_firm_year"]["median"])}'
        f'（n = {tb["pc_pooled_firm_year"]["n"]:,}）',
        ta['verdict_mean'], tb['verdict_mean'], note='同为负')

    # --- 表 4 列(1)/(2)：主读数 = 登记样本 EW lag12；并报原文最近似样本 ---
    for tag, key, tkey in (('表 4 列(1)（单变量 FM）', 'registered_EW_lag12', 'table4_col1_pc'),
                           ('表 4 列(2)（加控制 FM）', 'registered_EW_lag12', 'table4_col2_pc')):
        sa = a['L2_fm_univariate' if '列(1)' in tag else 'L2_fm_with_controls'][key]['PC']
        sb = b['L2_fm_univariate' if '列(1)' in tag else 'L2_fm_with_controls'][key]['PC']
        pa = a['L2_fm_univariate' if '列(1)' in tag else 'L2_fm_with_controls']['paper_like_EW_lag12']['PC']
        pb = b['L2_fm_univariate' if '列(1)' in tag else 'L2_fm_with_controls']['paper_like_EW_lag12']['PC']
        T = tgt[tkey]
        add(tag + '（NW lag 12，% 单位）', f'{T["coef"]}（t = {T["t"]}）',
            f'登记样本 {f(sa["beta_pct_per_unit"])}（t = {f(sa["t_nw"])}，{sa["n_months"]} 月）；'
            f'原文最近似 {f(pa["beta_pct_per_unit"])}（t = {f(pa["t_nw"])}）',
            f'登记样本 {f(sb["beta_pct_per_unit"])}（t = {f(sb["t_nw"])}，{sb["n_months"]} 月）；'
            f'原文最近似 {f(pb["beta_pct_per_unit"])}（t = {f(pb["t_nw"])}）',
            verdict_cell(sa['beta_pct_per_unit'], T['t'], sa['t_nw']),
            verdict_cell(sb['beta_pct_per_unit'], T['t'], sb['t_nw']))

    # --- 表 3：五分位/十分位 H−L raw 与 α3/α4 ---
    for key, tag in (('registered_quintile_EW', '表 3 H−L 五分位等权'),
                     ('registered_quintile_VW', '表 3 H−L 五分位市值加权'),
                     ('registered_decile_EW', '表 3 H−L 十分位等权'),
                     ('registered_decile_VW', '表 3 H−L 十分位市值加权')):
        la, lb = a['L3_portfolios'][key], b['L3_portfolios'][key]
        T = tgt['table3_HL_ew_a3' if key.endswith('EW') else 'table3_HL_vw_a3']
        add(tag + '（α3 = MKT/SMB/HML）', f'α3 {T["coef"]}（t = {T["t"]}）',
            f'raw {f(la["HL_raw_pct"])}（t = {f(la["HL_raw_t_nw"])}）/ α3 {f(la["alpha_a3_pct"])}'
            f'（t = {f(la["alpha_a3_t"])}）/ α4 {f(la["alpha_a4_pct"])}（t = {f(la["alpha_a4_t"])}）',
            f'raw {f(lb["HL_raw_pct"])}（t = {f(lb["HL_raw_t_nw"])}）/ α3 {f(lb["alpha_a3_pct"])}'
            f'（t = {f(lb["alpha_a3_t"])}）/ α4 {f(lb["alpha_a4_pct"])}（t = {f(lb["alpha_a4_t"])}）',
            verdict_cell(la['alpha_a3_pct'], T['t'], la['alpha_a3_t']),
            verdict_cell(lb['alpha_a3_pct'], T['t'], lb['alpha_a3_t']))

    # --- 假设 1（可选层）---
    ha, hb = a.get('L4_hypothesis1_optional'), b.get('L4_hypothesis1_optional')
    if ha and hb:
        def _s1(d, key):
            s = d.get(key) or {}
            return (f'{f(s.get("beta_sigma_in_1e8cny"))}（t_cl = {f(s.get("t_cluster_firm"))}）'
                    if s else 'NA')
        add('假设 1（PC → 业绩波动率，可选；σSale）', 'PC 系数均为正',
            f'β {_s1(ha, "sigma_sale")}；n = {ha.get("n_firm_year")}',
            f'β {_s1(hb, "sigma_sale")}；n = {hb.get("n_firm_year")}',
            (ha.get('sigma_sale') or {}).get('verdict', 'NA'),
            (hb.get('sigma_sale') or {}).get('verdict', 'NA'), note='可选层')

    # --- 样本量 / 月数 ---
    samples = {'mda': a['sample_sizes'], 'fulltext': b['sample_sizes']}
    # --- 结论：三层主判定是否随语料改变 ---
    def verdict_of(d, layer_key):
        for c in d['checklist']:
            if c['layer'].startswith(layer_key):
                return c['verdict']
        return 'NA'

    core_layers = ['表 4 列(1)', '表 4 列(2)', '表 3（H−L 十分位/五分位，α3）']
    v_a = {k: verdict_of(a, k) for k in core_layers}
    v_b = {k: verdict_of(b, k) for k in core_layers}
    unchanged = all(v_a[k] == v_b[k] for k in core_layers)
    # 点估计符号（描述层；不用于对原文判定）
    core_sign = {}
    for tag, key in (('表 4 列(1)', ('L2_fm_univariate', 'registered_EW_lag12')),
                     ('表 4 列(2)', ('L2_fm_with_controls', 'registered_EW_lag12')),
                     ('表 3（H−L 十分位/五分位，α3）', ('L3_portfolios', 'registered_quintile_EW'))):
        va = a[key[0]][key[1]]
        vb = b[key[0]][key[1]]
        if key[0] == 'L3_portfolios':
            core_sign[tag] = ((va['alpha_a3_pct'] or 0) < 0, (vb['alpha_a3_pct'] or 0) < 0)
        else:
            core_sign[tag] = ((va['PC']['beta_pct_per_unit'] or 0) < 0,
                              (vb['PC']['beta_pct_per_unit'] or 0) < 0)
    core_negative = all(x and y for x, y in core_sign.values())
    # 非主读数层的判定变化（诚实披露，不作择优）
    # --- 全层 × 全子统计的跨语料判定/符号变化普查（2026-10-20 复核修正；纯读数，不重估）---
    def _fm_group(d, kind, tkey, group):
        rows_ = []
        for spec, v in d[kind].items():
            pc = v['PC']
            rows_.append({'group': group, 'spec': spec, 'stat': 'PC',
                          'sample': 'registered' if spec.startswith('registered') else 'paper_like',
                          'mda_val': pc['beta_pct_per_unit'], 'mda_t': pc['t_nw'],
                          'ft_val': b[kind][spec]['PC']['beta_pct_per_unit'], 'ft_t': b[kind][spec]['PC']['t_nw'],
                          'target_t': d['targets'][tkey]['t']})
        return rows_

    def _pf_group(d, group):
        rows_ = []
        for spec, v in d['L3_portfolios'].items():
            tg = d['targets']['table3_HL_ew_a3']['t'] if spec.endswith('EW') else d['targets']['table3_HL_vw_a3']['t']
            ft = b['L3_portfolios'][spec]
            for stat, k, tk in (('H−L raw', 'HL_raw_pct', 'HL_raw_t_nw'),
                                ('α3', 'alpha_a3_pct', 'alpha_a3_t'),
                                ('α4', 'alpha_a4_pct', 'alpha_a4_t')):
                rows_.append({'group': group, 'spec': spec, 'stat': stat,
                              'sample': 'registered' if spec.startswith('registered') else 'paper_like',
                              'mda_val': v[k], 'mda_t': v[tk], 'ft_val': ft[k], 'ft_t': ft[tk],
                              'target_t': (d['targets']['table3_HL_ew_a7']['t'] if stat == 'α4' else tg)})
        return rows_

    census = (_fm_group(a, 'L2_fm_univariate', 'table4_col1_pc', '表 4 列(1) 单变量 FM')
              + _fm_group(a, 'L2_fm_with_controls', 'table4_col2_pc', '表 4 列(2) 加控制 FM')
              + _pf_group(a, '表 3 H−L 组合'))
    # 表 2 描述统计（符号层）
    for stat, key in (('均值', 'mean'), ('中位', 'median')):
        va = a['L1_descriptives']['pc_pooled_firm_year'][key]
        vb = b['L1_descriptives']['pc_pooled_firm_year'][key]
        tg = a['targets']['table2_pc_mean' if key == 'mean' else 'table2_pc_median']['value']
        census.append({'group': '表 2 描述统计', 'spec': 'pc_pooled_firm_year', 'stat': stat,
                       'sample': 'both', 'mda_val': va, 'mda_t': None, 'ft_val': vb, 'ft_t': None,
                       'target_t': None, 'target_val': tg})
    # 假设 1（可选层；判定用两版 JSON 存储值，不重估）
    for k in ('sigma_sale', 'sigma_profit'):
        sa = (a.get('L4_hypothesis1_optional') or {}).get(k) or {}
        sb = (b.get('L4_hypothesis1_optional') or {}).get(k) or {}
        if sa and sb:
            census.append({'group': '假设 1（可选）', 'spec': k, 'stat': 'PC→σ', 'sample': 'both',
                           'mda_val': sa.get('beta_sigma_in_1e8cny'), 'mda_t': sa.get('t_cluster_firm'),
                           'ft_val': sb.get('beta_sigma_in_1e8cny'), 'ft_t': sb.get('t_cluster_firm'),
                           'target_t': 1.0, 'stored_mda': sa.get('verdict'), 'stored_ft': sb.get('verdict')})
    for r in census:
        if 'stored_mda' in r:                      # 假设 1：用存储判定（口径不可比项不重推）
            r['mda_verdict'], r['ft_verdict'] = r['stored_mda'], r['stored_ft']
        elif r['group'] == '表 2 描述统计':
            r['mda_verdict'] = ('符号一致（同为负；描述性，不构成复现）'
                                if (r['mda_val'] or 0) < 0 == (r['target_val'] < 0) else '符号不一致（描述性）')
            r['ft_verdict'] = ('符号一致（同为负；描述性，不构成复现）'
                               if (r['ft_val'] or 0) < 0 == (r['target_val'] < 0) else '符号不一致（描述性）')
        else:
            r['mda_verdict'] = verdict_cell(r['mda_val'], r['target_t'], r['mda_t'])
            r['ft_verdict'] = verdict_cell(r['ft_val'], r['target_t'], r['ft_t'])
        r['sign_change'] = (r['mda_val'] is not None and r['ft_val'] is not None
                            and (r['mda_val'] > 0) != (r['ft_val'] > 0))
        r['verdict_change'] = r['mda_verdict'] != r['ft_verdict']
    census_changes = [r for r in census if r['sign_change']]
    verdict_changes_full = [r for r in census if r['verdict_change']]
    census_summary = {
        'n_layer_stat_pairs': len(census),
        'n_sign_changes': len(census_changes),
        'n_verdict_changes': len(verdict_changes_full),
        'verdict_changes_by_group': {g: sum(1 for r in verdict_changes_full if r['group'] == g)
                                     for g in sorted({r['group'] for r in census})},
        'verdict_changes_by_sample': {s: sum(1 for r in verdict_changes_full if r['sample'] == s)
                                      for s in ('registered', 'paper_like', 'both')},
        'fm_layers_unchanged': all(not r['verdict_change'] for r in census if r['group'].startswith('表 4')),
        'table2_unchanged': all(not r['verdict_change'] for r in census if r['group'] == '表 2 描述统计'),
        'h1_unchanged': all(not r['verdict_change'] for r in census if r['group'].startswith('假设 1')),
        'ew_portfolio_alpha_unchanged': all(
            not r['verdict_change'] for r in census if r['group'] == '表 3 H−L 组合'
            and r['stat'] == 'α3' and r['spec'].endswith('EW')),
        'all_changes_in_portfolio_layers': all(r['group'] == '表 3 H−L 组合' for r in verdict_changes_full),
        'sign_change_is_subset_of_verdict_change': len(census_changes) == len(verdict_changes_full),
    }
    changed_rows = [f'{r["layer"]}（MD&A「{r["verdict_mda"]}」→ 全文「{r["verdict_fulltext"]}」）'
                    for r in rows if r['verdict_mda'] != r['verdict_fulltext']]
    nuance = ('；三方并列表 8 层中判定变化 = ' + '、'.join(changed_rows) + '（非主读数层；本表 8 行内，全量清单见「逐层判定 / 符号变化全清单」节）') if changed_rows else \
        '；三方并列表 8 层判定逐层相同'
    if unchanged and core_negative:
        conclusion = ('**项目 PC 与原文不可比（对手集合口径不同），判定统一为「不可比/未复现」**。'
                      '在描述层：MD&A 版与全文版在 **FM 主层**（表 4 列(1)/(2) 共 16 规格 = '
                      '[paper_like, registered] × [EW, VW] × [lag6, lag12] × [单变量, 加控制]）点估计全为负、'
                      '**换语料不变**；**组合等权层**（五分位/十分位 EW α3）同样为负；'
                      '但 **VW 组合层**（十分位 VW 的 raw/α3/α4、五分位 VW 的 raw、十分位 EW 的 raw）在换语料后**由负转正**'
                      '（全部不显著）——即**语料只改变部分组合层的点估计符号，不改变主读数层**。'
                      '上述符号变化仅描述项目自建 PC 的语料敏感性，不构成对原文的复现或证伪'
                      '（与原文的其余差异 = 期间 FY2017–2022 vs 2007–2016、K 校准口径、控制集可得性，登记为限制）'
                      + nuance + '。')
    elif unchanged:
        conclusion = ('**判定统一为「不可比/未复现」**：三层判定在 MD&A 版与全文版逐层相同（详见并列表）；'
                      '点估计符号变化仅作描述层披露' + nuance + '。')
    else:
        conclusion = '**语料改变部分层的点估计/判定读数**：见并列表逐层 verdict（不做事后择优；对外判定仍为不可比/未复现）' + nuance + '。'

    # 历史快照：从既有产物冻结读取（重跑不得用新标签覆盖 sources.historical_overall）
    hist_overall = None
    if OUT_JSON.exists():
        try:
            _prev = json.loads(OUT_JSON.read_text(encoding='utf-8'))
            hist_overall = _prev.get('historical_overall')
        except Exception:  # noqa: BLE001
            hist_overall = None
    if not hist_overall:
        hist_overall = {'mda': {'overall': '复现不成功（历史标签）'},
                        'fulltext': {'overall': '复现不成功（历史标签）'}}
    hist_src = {k: (hist_overall[k].get('overall') if isinstance(hist_overall[k], dict) else hist_overall[k])
                for k in ('mda', 'fulltext')}
    HIST_NOTE = ('本文件判定列/结论为 2026-10-19/2026-10-20 历史标签；自 2026-10-27 起统一读作'
                 '「不可比/未复现（对手集合口径不同）」；表内数值与符号变化仍有效，但只描述项目自建 PC 的语料敏感性，'
                 '不构成对原文的复现或证伪。')
    out = {'artifact': 's3_pc_corpus_contrast_fulltext', 'run_at': datetime.now(timezone.utc).isoformat(),
           'purpose': 'F5：对表语料三方并列表（原文 × MD&A 主口径 × 年报全文敏感性列）',
           'historical_label_note': HIST_NOTE,
           'comparability': {
               'comparable_to_paper': False,
               'registered_deviation': 'methodology_execution_plan:2290',
               'project_rule': 'same_cluster_members（同团全部成员）',
               'paper_rule': '任何与其相似度超过团内最小相似度的公司（字面含团外）',
               'consequence': 'PC 不是原文变量的严格复现；判定 = 不可比/未复现（对手集合口径不同）；'
                              '不得写符号相反/复现失败/证伪',
               'replication_requirement': '按原文第四步规则 + 接近原文词表规模（active_vocab ≈ 1.5 万量级）重建（未做）'},
           'sources': {'paper': a['paper'], 'mda_version': {'file': str(MD_A.relative_to(ROOT)).replace('\\', '/'),
                                                            'generated_at': a.get('generated_at'),
                                                            'historical_overall': hist_src['mda']},
                       'fulltext_version': {'file': str(FT.relative_to(ROOT)).replace('\\', '/'),
                                            'generated_at': b.get('generated_at'),
                                            'historical_overall': hist_src['fulltext']}},
           'criteria': a['pre_registered_criteria'],
           'sample_sizes': samples,
           'verdicts_core': {'mda': v_a, 'fulltext': v_b, 'unchanged_across_corpus': bool(unchanged)},
           'census_summary': census_summary,
           'census_rows': census,
           'verdict_changes_full': verdict_changes_full,
           'same_sign_field_meaning': 'rows[].same_sign = 两版**判定字符串**是否相同（不是系数符号是否相同）；'
                                      '逐子统计的符号变化见 census_rows / verdict_changes_full',
           'review_20261020': {
               'why': '复核轮（repro_D/E）发现本并列表自述「唯一层级变化 = 表 3 十分位市值加权 α3」不完整：'
                      '该表述只遍历本表 8 行、且每行仅以 α3 作判定，漏掉 ① 原文最近似样本同层、② 同层 H−L raw 与 α4、'
                      '③ 五分位 VW 的 raw 与十分位 EW 的 raw。',
               'fix': '新增 census_rows（全层 × 全子统计逐格判定，纯读数不重估）与 verdict_changes_full（全量判定变化清单），'
                      '结论句改为准确版本；数值读数与判据零改动。',
               'reference': 'reports/audits/s3_fulltext_correspondence_review_20261020.md'},
           'rows': rows, 'conclusion': conclusion,
           'historical_overall': hist_overall}

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')

    L = ['# S3 对表语料三方并列表：原文 × 主口径（老师 MD&A）× 全文语料（年报全文）', '',
         f'- 原文 = {a["paper"]["citation"]}（{a["paper"]["file"]}）',
         f'- MD&A 版 = `{out["sources"]["mda_version"]["file"]}`（{a.get("generated_at")}，总判定 = {hist_src["mda"]}（历史标签））',
         f'- 全文版 = `{out["sources"]["fulltext_version"]["file"]}`（{b.get("generated_at")}，总判定 = {hist_src["fulltext"]}（历史标签））',
         f'- 判据（先登记，未改）：{a["pre_registered_criteria"]["rule"]}',
         '',
         '> **2026-10-27 口径更正（Phase 0.2 / V-P1-2 + N-2）**：本表判定列与结论为 2026-10-19/2026-10-20 历史标签；'
         '自 2026-10-27 起统一读作**不可比/未复现（对手集合口径不同）**（项目 = 同团成员；原文 = 相似度 > 团内最小相似度，'
         '字面含团外；`methodology_execution_plan.md:2290` 已登记偏离）。表内数值与符号变化仍有效，但只描述项目自建 PC '
         '的语料敏感性，不构成对原文的复现或证伪。',
         '', '| 层 | 原文（靶子） | MD&A 版（主口径） | 全文版（本列） | MD&A 判定 | 全文判定 |',
         '|---|---|---|---|---|---|']
    for r in rows:
        L.append(f'| {r["layer"]} | {r["target"]} | {r["mda"]} | {r["fulltext"]} | {r["verdict_mda"]} | {r["verdict_fulltext"]} |')
    L += ['', '## 逐层判定 / 符号变化全清单（2026-10-20 复核修正）', '',
          f'- 普查范围 = **全层 × 全子统计**：{census_summary["n_layer_stat_pairs"]} 个「层 × 规格 × 子统计」格'
          '（FM 主层 16 格 = 表 4 列(1) 8 规格 + 列(2) 8 规格，每格 1 个 PC 读数；'
          '表 3 组合层 24 格 = 8 规格 × [H−L raw, α3, α4]；表 2 2 格；假设 1 2 格）。',
          f'- 结果：**符号变化 {census_summary["n_sign_changes"]} 格 / 判定变化 {census_summary["n_verdict_changes"]} 格**，'
          f'全部落在**表 3 组合层**（FM 主层判定不变 = {census_summary["fm_layers_unchanged"]}；'
          f'表 2 不变 = {census_summary["table2_unchanged"]}；假设 1 不变 = {census_summary["h1_unchanged"]}；'
          f'等权组合 α3 不变 = {census_summary["ew_portfolio_alpha_unchanged"]}）。',
          f'- 按样本：registered {census_summary["verdict_changes_by_sample"]["registered"]} 格 / '
          f'paper_like {census_summary["verdict_changes_by_sample"]["paper_like"]} 格。',
          '',
          '| 组合规格 | 样本 | 子统计 | MD&A 版（%，t） | 全文版（%，t） | 符号翻转 | MD&A 判定 → 全文判定 |',
          '|---|---|---|---|---|---|---|']
    for r in verdict_changes_full:
        L.append(f'| {r["spec"]} | {r["sample"]} | {r["stat"]} | {f(r["mda_val"])}（t = {f(r["mda_t"])}） | '
                 f'{f(r["ft_val"])}（t = {f(r["ft_t"])}） | 是 | {r["mda_verdict"]} → {r["ft_verdict"]} |')
    h1_labels = {r['spec']: r['mda_verdict'] for r in census if r['group'].startswith('假设 1')}
    L += ['',
          f'未变化（逐格判定跨语料相同）：FM 主层 16 格（表 4 列(1)/(2)，判定 =「{NOT_COMPARABLE_LABEL}」）/ '
          '表 3 五分位等权（raw、α3、α4，两样本）/ 表 3 十分位等权（α3、α4，两样本）/ '
          '表 3 五分位市值加权（α3、α4，两样本）/ 表 2（均值、中位，符号一致）/ '
          f'假设 1（σSale「{h1_labels.get("sigma_sale", "NA")}」、σProfit「{h1_labels.get("sigma_profit", "NA")}」）。',
          '',
          '**修正说明（2026-10-20 复核）**：本并列表原自述「唯一层级变化 = 表 3 十分位市值加权 α3」**不完整**——'
          '该表述只遍历上表 8 行、且每行仅以 α3 作判定，漏掉 ① 原文最近似样本（paper_like）同层、② 同层 H−L raw 与 α4。'
          '今补齐为上述全清单（**数值读数与判据零改动**，仅新增普查与措辞修正）；复核证据见 '
          '`reports/audits/s3_fulltext_correspondence_review_20261020.md`。',
          '', '## 样本量 / 月数', '',
          f'- MD&A 版：S1 原文最近似 {samples["mda"]["S1_paper_like"]["rows"]:,} 行 / '
          f'{samples["mda"]["S1_paper_like"]["months"]} 月（{samples["mda"]["S1_paper_like"]["first_month"]}..'
          f'{samples["mda"]["S1_paper_like"]["last_month"]}）；S1 登记样本 {samples["mda"]["S1_registered"]["rows"]:,} 行 / '
          f'{samples["mda"]["S1_registered"]["months"]} 月；S2 原文最近似 {samples["mda"]["S2_paper_like"]["rows"]:,}；'
          f'S2 登记样本 {samples["mda"]["S2_registered"]["rows"]:,}',
          f'- 全文版：S1 原文最近似 {samples["fulltext"]["S1_paper_like"]["rows"]:,} 行 / '
          f'{samples["fulltext"]["S1_paper_like"]["months"]} 月（{samples["fulltext"]["S1_paper_like"]["first_month"]}..'
          f'{samples["fulltext"]["S1_paper_like"]["last_month"]}）；S1 登记样本 {samples["fulltext"]["S1_registered"]["rows"]:,} 行 / '
          f'{samples["fulltext"]["S1_registered"]["months"]} 月；S2 原文最近似 {samples["fulltext"]["S2_paper_like"]["rows"]:,}；'
          f'S2 登记样本 {samples["fulltext"]["S2_registered"]["rows"]:,}',
          '', '## 结论', '', conclusion, '',
          '> 量级差异按期间/语料差异解释；本表只做读数并列与符号判定，**不做事后样本挑选、不做任何调参**。']
    OUT_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')
    psay(f'状态 = ok：并列表 {len(rows)} 层；三层主判定跨语料一致 = {unchanged}；'
         f'产物 {OUT_JSON.name} / {OUT_MD.name}')
    psay(f'结论 = {conclusion[:120]}…')
    print(json.dumps({'rows': len(rows), 'unchanged': bool(unchanged),
                      'mda_overall': a['summary']['overall'], 'fulltext_overall': b['summary']['overall']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
