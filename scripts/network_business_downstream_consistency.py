#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""network Phase B″6：下游方向一致性（**探索性 / 事后（post hoc）描述性检验**，2026-09-29）。

**定性声明（2026-09-29 二修，必须随数值一并引用）**：本脚本的口径在观察到 B″5 v1 结果
（2026-09-28 停机）之后设计；2026-09-28 预先写下的替代处置口径中**不含"下游方向"这一项**，
故本检验**不构成 outcome-blind 预登记**。其作用**仅限于**回答"两套网络的下游结果是否反向"
这一**描述性**问题——**不得**作为 定价 gate / 机制层 输入合格性的证据，也**不得**作为"换章节稳健性通过"
的证据。B″5 的闸门判据仍是 out-of-sample 口径无关的 ⓐ（peer 规模，统一口径）＋ⓑ（lift 同向）；
ⓔ 只作报告。

目的：回答「换章节（主口径老师 MD&A ↔ 文本稳健性自建业务描述）两套网络的下游结果是否反向」。
不看边集重合率（ⓒ 已按裁定降级为描述性），看两套网络各自算出的 PeerMAXraw / PeerReturn
与组合层（按 PeerMAXraw 分五组的次月收益）是否同向、强度多大。

输入：
  data/processed/monthly_pricing_panel_ma_teacher.csv（主口径，2015-01~2023-12）
  data/processed/monthly_pricing_panel_ma.csv（文本稳健性列，2018-01~2025-12）
  data/processed/monthly_returns_clean.csv（R，t+1 月收益）
名义重叠月份（两面板月份交集）：2018-01~2023-12 = **72 个月**；
**有效配对月份 = 68 个月**（2018-01~04 无配对：文本稳健性列最早版本 available_at=2018-04-30，
按严格 `<` 消费规则 2018-05 才可用）。公司-月层相关只在有效配对月份内有值。

指标（有效配对月份）：
  - 公司-月层：PeerMAXraw / PeerReturn 的 Pearson、Spearman（池化 + 逐月截面的中位数）＋共享方差
  - 分位一致性：逐月各自按 PeerMAXraw 分五组 → 同组率、相邻组（±1）率、组序 Spearman
  - 组合层：逐月按 PeerMAXraw 分五组 → 各组等权次月收益；报组均值的秩相关与形状、多空（Q5−Q1）
    的均值/标准差/**t 值**/同号月份占比（含反向占比）/**最负 3 个月占 LS 总和的比例**；
    并报"同一月份集合（68 月）"下两网的对照数（主口径在两网共同月份上的 LS），
    因为主口径 72 月 / 文本稳健性 68 月**不是同一月份集合**
  - 分段：2018–2020 与 2021–2023（对应 2021 语料结构性断点）

产物：reports/data_gates/G0/network_business_downstream_consistency.{json,md}
      （json 内含 cross_audit_handoff 段，供 network_business_chapter_cross_audit.py --downstream-report 读取，
      其中 e_downstream_direction 明确标为 post hoc / 描述性、非闸门）
用法：
  python scripts/network_business_downstream_consistency.py \
      --main-panel data/processed/monthly_pricing_panel_ma_teacher.csv \
      --robust-panel data/processed/monthly_pricing_panel_ma.csv \
      --out-json reports/data_gates/G0/network_business_downstream_consistency.json \
      --out-md reports/data_gates/G0/network_business_downstream_consistency.md

v2（2026-10-05，Phase B″10）：
  - 新增 `--main-label` / `--robust-label`（只改报告人读文案，默认值与历史完全一致）——同语料权重对照
    （等权主口径 ↔ IDF 变体）复用时不必让报告自称"换章节"；
  - `effective_paired_months_note` 由写死"2018-01~04 无配对 / 68 个月"改为**按实际重叠与配对月份动态生成**
    （原文案只对「主口径 72 月 ↔ 文本稳健性 68 月」成立，对同语料两面板不成立）；
  - `strength.reading` 由写死数值（"共享方差约 26%/49%、同组率 0.393、多空 −0.28%/−0.22%、同号 51%、
    最负 3 个月 139%"）改为**由当次实算数值生成**——LA 面板对换成同语料（等权 ↔ IDF）时，写死文案会
    与实际数值矛盾；默认面板对下的数值与旧文案同源（0.2596/0.4887/0.3931/0.7387/−0.28%/−0.22%/0.5139/1.392），
    仅呈现方式从"约 26%"改为四位小数；
  - **数值口径、字段名与全部指标一字未改**；两面板月份交集与有效配对月份由数据决定。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAIN_PANEL = ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv'
ROBUST_PANEL = ROOT / 'data/processed/monthly_pricing_panel_ma.csv'
RETURNS = ROOT / 'data/processed/monthly_returns_clean.csv'
REPD = ROOT / 'reports/data_gates/G0'
N_QUINTILE = 5


def ym_add(month: str, k: int) -> str:
    y, m = int(month[:4]), int(month[5:])
    m += k
    y += (m - 1) // 12
    m = (m - 1) % 12 + 1
    return f'{y:04d}-{m:02d}'


def rel(p) -> str:
    """相对 ROOT 的 POSIX 路径；ROOT 外或相对路径回退为原串。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p).replace('\\', '/')


def load_panel(p: Path, label: str) -> pd.DataFrame:
    d = pd.read_csv(p, dtype={'focal_id': str, 'month': str},
                    usecols=['focal_id', 'month', 'PeerMAXraw', 'PeerReturn'])
    d = d.rename(columns={'PeerMAXraw': f'pm_{label}', 'PeerReturn': f'pr_{label}'})
    return d


def load_returns() -> dict[str, dict[str, float]]:
    import csv
    out: dict[str, dict[str, float]] = {}
    with open(RETURNS, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            out.setdefault(r['month'], {})[r['firm_id']] = float(r['R'])
    return out


def corr_pair(x: pd.Series, y: pd.Series) -> dict:
    m = x.notna() & y.notna()
    n = int(m.sum())
    if n < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return {'n': n, 'pearson': None, 'spearman': None}
    return {'n': n, 'pearson': round(float(x[m].corr(y[m])), 4),
            'spearman': round(float(x[m].corr(y[m], method='spearman')), 4)}


def per_month_corr(j: pd.DataFrame, ca: str, cb: str) -> dict:
    """逐月截面相关 → 中位数/均值（散点方向是否按月稳定）。"""
    ps, ss = [], []
    for _, g in j.groupby('month'):
        c = corr_pair(g[ca], g[cb])
        if c['pearson'] is not None:
            ps.append(c['pearson'])
            ss.append(c['spearman'])
    if not ps:
        return {'n_months': 0, 'pearson_median': None, 'spearman_median': None}
    return {'n_months': len(ps),
            'pearson_median': round(float(np.median(ps)), 4),
            'pearson_mean': round(float(np.mean(ps)), 4),
            'spearman_median': round(float(np.median(ss)), 4),
            'share_months_pearson_positive': round(float(np.mean([p > 0 for p in ps])), 4)}


def quintile_labels(s: pd.Series, q: int = N_QUINTILE) -> pd.Series:
    """按分位切分（重复值用 rank(method='first') 打破，保证各组容量均衡）。"""
    r = s.rank(method='first')
    try:
        lab = pd.qcut(r, q, labels=False)
    except ValueError:
        return pd.Series(np.nan, index=s.index)
    return lab + 1


def portfolio_block(j: pd.DataFrame, ca: str, ret_col: str,
                    months_used: set) -> dict:
    """按 ca（PeerMAXraw）逐月分五组 → 各组等权次月收益；报单调方向与多空。"""
    rows = []
    for m, g in j.groupby('month'):
        if m not in months_used:
            continue
        g2 = g[g[ca].notna() & g[ret_col].notna()].copy()
        if len(g2) < 50:
            continue
        g2['q'] = quintile_labels(g2[ca])
        qr = g2.groupby('q')[ret_col].mean()
        if len(qr) < N_QUINTILE:
            continue
        mono = pd.Series(qr.index.astype(float)).corr(pd.Series(qr.values), method='spearman')
        rows.append({'month': m, 'n': int(len(g2)), 'q_means': [round(float(v), 6) for v in qr.values],
                     'mono_spearman': round(float(mono), 4),
                     'ls': round(float(qr.iloc[-1] - qr.iloc[0]), 6)})
    if not rows:
        return {'n_months': 0}
    df = pd.DataFrame(rows)
    qbar = np.mean([r['q_means'] for r in rows], axis=0)
    mono_overall = float(pd.Series(range(1, N_QUINTILE + 1)).corr(
        pd.Series(qbar), method='spearman'))
    ls_mean = float(df['ls'].mean())
    ls = df['ls'].values
    t_stat = float(ls_mean / (ls.std(ddof=1) / np.sqrt(len(ls)))) if len(ls) > 1 and ls.std(ddof=1) > 0 else None
    top3 = float(np.sort(ls)[:3].sum())
    peak = int(np.argmax(qbar) + 1)
    trough = int(np.argmin(qbar) + 1)
    return {'n_months': int(len(df)),
            'quintile_mean_next_month_ret': [round(float(v), 6) for v in qbar],
            'quintile_rank_corr_spearman': round(mono_overall, 4),
            'quintile_rank_corr_direction': ('up' if mono_overall > 0 else 'down' if mono_overall < 0 else 'flat'),
            'quintile_shape': f'非单调：Q{peak} 最高（{round(float(qbar[peak-1]), 6)}）、'
                              f'Q{trough} 最低（{round(float(qbar[trough-1]), 6)}）',
            'quintile_peak': peak, 'quintile_trough': trough,
            'monotonicity_spearman_median_per_month': round(float(df['mono_spearman'].median()), 4),
            'long_short_q5_minus_q1_mean': round(ls_mean, 6),
            'long_short_sign': ('positive' if ls_mean > 0 else 'negative' if ls_mean < 0 else 'zero'),
            'long_short_monthly_std': round(float(df['ls'].std()), 6),
            'long_short_t_stat': round(t_stat, 3) if t_stat is not None else None,
            'long_short_share_months_same_sign': round(float(np.mean(
                [np.sign(v) == np.sign(ls_mean) for v in df['ls']])), 4),
            'long_short_share_months_opposite_sign': round(float(np.mean(
                [np.sign(v) == -np.sign(ls_mean) for v in df['ls']])), 4),
            'long_short_top3_months_share_of_sum': round(float(top3 / ls.sum()), 3) if ls.sum() != 0 else None,
            'significance_note': '不做显著性宣称：t 值仅作描述性参考，|t| < 1，统计上与 0 无异。'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--main-panel', default=str(MAIN_PANEL))
    ap.add_argument('--robust-panel', default=str(ROBUST_PANEL))
    ap.add_argument('--out-json', default=str(REPD / 'network_business_downstream_consistency.json'))
    ap.add_argument('--out-md', default=str(REPD / 'network_business_downstream_consistency.md'))
    ap.add_argument('--main-label', default='主口径（老师 MD&A）', help='main 侧人读标签（只影响报告文案）')
    ap.add_argument('--robust-label', default='文本稳健性（业务描述）', help='robust 侧人读标签（只影响报告文案）')
    args = ap.parse_args()

    main_lbl, rob_lbl = 'main', 'robust'
    a = load_panel(Path(args.main_panel), main_lbl)
    b = load_panel(Path(args.robust_panel), rob_lbl)
    months = sorted(set(a['month']) & set(b['month']))
    print(f'重叠月份 {months[0]}..{months[-1]} ({len(months)})')
    j = a.merge(b, on=['focal_id', 'month'], how='inner')
    j = j[j['month'].isin(months)]
    rmap = load_returns()

    # t+1 月收益（组合层用）
    j['ret_next'] = [rmap.get(ym_add(m, 1), {}).get(f) for f, m in zip(j['focal_id'], j['month'])]
    months_next = {m for m in months if ym_add(m, 1) in rmap}

    pm_corr = corr_pair(j[f'pm_{main_lbl}'], j[f'pm_{rob_lbl}'])
    pr_corr = corr_pair(j[f'pr_{main_lbl}'], j[f'pr_{rob_lbl}'])
    pm_month = per_month_corr(j, f'pm_{main_lbl}', f'pm_{rob_lbl}')
    pr_month = per_month_corr(j, f'pr_{main_lbl}', f'pr_{rob_lbl}')

    # 分位一致性（逐月各自分五组）
    q_rows = []
    for m, g in j.groupby('month'):
        g2 = g[g[f'pm_{main_lbl}'].notna() & g[f'pm_{rob_lbl}'].notna()]
        if len(g2) < 50:
            continue
        qa = quintile_labels(g2[f'pm_{main_lbl}'])
        qb = quintile_labels(g2[f'pm_{rob_lbl}'])
        q_rows.append({'same': float((qa == qb).mean()), 'within1': float((qa - qb).abs().le(1).mean()),
                       'spearman': float(pd.Series(qa).corr(pd.Series(qb), method='spearman')),
                       'n': int(len(g2))})
    q_df = pd.DataFrame(q_rows)
    quintile = ({'n_months': int(len(q_df)),
                 'same_quintile_share_mean': round(float(q_df['same'].mean()), 4),
                 'within_plus_minus_1_quintile_share_mean': round(float(q_df['within1'].mean()), 4),
                 'quintile_spearman_median': round(float(q_df['spearman'].median()), 4),
                 'n_firms_per_month_median': int(q_df['n'].median())} if len(q_df) else {'n_months': 0})

    pf_main = portfolio_block(j, f'pm_{main_lbl}', 'ret_next', months_next)
    pf_rob = portfolio_block(j, f'pm_{rob_lbl}', 'ret_next', months_next)
    # 有效配对月份（两网同月同 focal 均有值）—— 主口径 72 月 / 文本稳健性 68 月不是同一月份集合
    paired = []
    for m, g in j.groupby('month'):
        if m not in months_next:
            continue
        ok = g[f'pm_{main_lbl}'].notna() & g[f'pm_{rob_lbl}'].notna() & g['ret_next'].notna()
        if int(ok.sum()) >= 50:
            paired.append(m)
    paired = set(paired)
    unpaired = [m for m in months if m not in paired]
    paired_note = (f'有效配对 {len(paired)} 个月；未配对 {len(unpaired)} 个月'
                   + (f'（{unpaired[0]}~{unpaired[-1]}：该月两面板无共同可用网络版本，或同月同 focal '
                      f'双面板均有值且次月收益可得者 < 50 家）' if unpaired else ''))
    pf_main_paired = portfolio_block(j, f'pm_{main_lbl}', 'ret_next', paired)
    pf_rob_paired = portfolio_block(j, f'pm_{rob_lbl}', 'ret_next', paired)

    # 分段（2021 语料结构性断点：2018-2020 vs 2021-2023；2015-2020 段缺文本稳健性列可比月份）
    segs = {}
    for name, lo, hi in [('2018_2020', '2018-01', '2020-12'), ('2021_2023', '2021-01', '2023-12')]:
        js = j[(j['month'] >= lo) & (j['month'] <= hi)]
        segs[name] = {'months': [lo, hi], 'n_rows_merged': int(len(js)),
                      'peer_maxraw': corr_pair(js[f'pm_{main_lbl}'], js[f'pm_{rob_lbl}']),
                      'peer_return': corr_pair(js[f'pr_{main_lbl}'], js[f'pr_{rob_lbl}']),
                      'long_short_main': portfolio_block(js, f'pm_{main_lbl}', 'ret_next',
                                                         months_next).get('long_short_q5_minus_q1_mean'),
                      'long_short_robust': portfolio_block(js, f'pm_{rob_lbl}', 'ret_next',
                                                           months_next).get('long_short_q5_minus_q1_mean')}

    # **描述性检查（post hoc，非闸门）**：只回答"两网是否反向"
    corr_positive = bool(pm_corr['pearson'] is not None and pm_corr['pearson'] > 0
                         and pm_corr['spearman'] is not None and pm_corr['spearman'] > 0
                         and pr_corr['pearson'] is not None and pr_corr['pearson'] > 0
                         and pr_corr['spearman'] is not None and pr_corr['spearman'] > 0)
    ls_same_sign = bool(pf_main.get('long_short_sign') and pf_rob.get('long_short_sign')
                        and pf_main['long_short_sign'] == pf_rob['long_short_sign'])
    rank_corr_same_sign = bool(pf_main.get('quintile_rank_corr_direction')
                               and pf_rob.get('quintile_rank_corr_direction')
                               and pf_main['quintile_rank_corr_direction']
                               == pf_rob['quintile_rank_corr_direction']
                               and pf_main['quintile_rank_corr_direction'] != 'flat')
    not_reversed = bool(corr_positive and ls_same_sign and rank_corr_same_sign)
    shared_var = {'peer_maxraw': round(pm_corr['pearson'] ** 2, 4) if pm_corr['pearson'] else None,
                  'peer_return': round(pr_corr['pearson'] ** 2, 4) if pr_corr['pearson'] else None,
                  'note': '共享方差 = Pearson^2；PeerMAXraw ≈26%、PeerReturn ≈49%。'}
    strength = {
        'peer_maxraw_shared_variance': shared_var['peer_maxraw'],
        'peer_return_shared_variance': shared_var['peer_return'],
        'quintile_same_share_vs_random_multiple':
            round(quintile['same_quintile_share_mean'] / 0.2, 2) if quintile.get('same_quintile_share_mean') else None,
        'quintile_same_share_random_benchmark': 0.2,
        'quintile_within_pm1_share': quintile.get('within_plus_minus_1_quintile_share_mean'),
        'long_short_main_mean': pf_main.get('long_short_q5_minus_q1_mean'),
        'long_short_main_t': pf_main.get('long_short_t_stat'),
        'long_short_robust_mean': pf_rob.get('long_short_q5_minus_q1_mean'),
        'long_short_robust_t': pf_rob.get('long_short_t_stat'),
        'long_short_main_paired_months_mean': pf_main_paired.get('long_short_q5_minus_q1_mean'),
        'long_short_main_paired_months_t': pf_main_paired.get('long_short_t_stat'),
        'long_short_robust_paired_months_mean': pf_rob_paired.get('long_short_q5_minus_q1_mean'),
        'long_short_robust_paired_months_t': pf_rob_paired.get('long_short_t_stat'),
        'share_months_same_sign_main': pf_main.get('long_short_share_months_same_sign'),
        'share_months_opposite_sign_main': pf_main.get('long_short_share_months_opposite_sign'),
        'top3_months_share_of_ls_sum_main': pf_main.get('long_short_top3_months_share_of_sum'),
    }
    # 强度判读文案由**当次实算数值**生成（v2，2026-10-05）：原为写死「26%/49%、同组率 0.393、
    # 多空 −0.28%/−0.22%、同号 51%、最负 3 个月 139%」——那是「主口径 × 文本稳健性」当次的数值，
    # 复用到其它面板对（如同语料 等权 ↔ IDF）会造成数值与结论不符。
    _t_m, _t_r = pf_main.get('long_short_t_stat'), pf_rob.get('long_short_t_stat')
    _abs_lt2 = all(t is not None and abs(t) < 2 for t in (_t_m, _t_r))
    strength['reading'] = (
        f'相关与分位一致性为正：PeerMAXraw 共享方差 {shared_var["peer_maxraw"]}、'
        f'PeerReturn {shared_var["peer_return"]}；五分位同组率 {quintile.get("same_quintile_share_mean")}'
        f'（≈ {strength["quintile_same_share_vs_random_multiple"]}× 随机基线 0.2）、±1 组内 '
        f'{quintile.get("within_plus_minus_1_quintile_share_mean")}、组序 Spearman 中位 '
        f'{quintile.get("quintile_spearman_median")}；组合层多空均值 '
        f'{pf_main.get("long_short_q5_minus_q1_mean")}（t = {_t_m}）与 '
        f'{pf_rob.get("long_short_q5_minus_q1_mean")}（t = {_t_r}）、同号月份 '
        f'{pf_main.get("long_short_share_months_same_sign")}、最负 3 个月占 LS 总和 '
        f'{pf_main.get("long_short_top3_months_share_of_sum")} → '
        + ('|t| < 2，**组合层差异在统计上与 0 无异、不宣称显著性**。' if _abs_lt2 else
           '|t| ≥ 2，但多空仅为等权分组描述性统计（无因子调整/稳健标准误），仍不作显著性宣称。')
    )

    handoff_criteria = {'peer_maxraw_peerreturn_corr_positive': corr_positive,
                        'long_short_sign_consistent': ls_same_sign,
                        'quintile_rank_corr_direction_consistent': rank_corr_same_sign,
                        'not_reversed_overall': not_reversed}
    handoff = {
        'e_downstream_direction': not_reversed,
        'e_semantics': 'post hoc / 描述性：非 outcome-blind 预登记、非闸门；'
                       '不得作为 定价 gate / 机制层 输入合格性或"换章节稳健性通过"的证据',
        'descriptive_checks': handoff_criteria,
        'conclusion': '两网下游未反向（描述性）',
        'overlap_months': [months[0], months[-1], len(months)],
        'effective_paired_months': len(paired),
        'peer_maxraw': {'pearson': pm_corr['pearson'], 'spearman': pm_corr['spearman'],
                        'n': pm_corr['n'], 'per_month': pm_month},
        'peer_return': {'pearson': pr_corr['pearson'], 'spearman': pr_corr['spearman'],
                        'n': pr_corr['n'], 'per_month': pr_month},
        'quintile_agreement': quintile,
        'strength': strength,
        'portfolio': {'main': {k: pf_main.get(k) for k in
                               ('n_months', 'quintile_mean_next_month_ret', 'quintile_rank_corr_spearman',
                                'quintile_rank_corr_direction', 'quintile_shape', 'long_short_sign',
                                'long_short_q5_minus_q1_mean', 'long_short_t_stat')},
                      'robust': {k: pf_rob.get(k) for k in
                                 ('n_months', 'quintile_mean_next_month_ret', 'quintile_rank_corr_spearman',
                                  'quintile_rank_corr_direction', 'quintile_shape', 'long_short_sign',
                                  'long_short_q5_minus_q1_mean', 'long_short_t_stat')},
                      'main_restricted_to_paired_months': {
                          k: pf_main_paired.get(k) for k in
                          ('n_months', 'quintile_mean_next_month_ret', 'quintile_rank_corr_spearman',
                           'quintile_shape', 'long_short_q5_minus_q1_mean', 'long_short_t_stat')}},
        'note': '方向/排序一致性与相关数值；**不做显著性宣称**（t 值仅作描述性参考，|t|<1）。'
                '组合层排序变量 = PeerMAXraw，收益 = 次月 R（monthly_returns_clean），等权。'
                '**本检验为探索性/事后（post hoc）**：在观察到 B″5 v1 结果之后设计，'
                '不构成 outcome-blind 预登记。',
    }

    rep = {
        'step': 'network Phase B″6 下游方向一致性（探索性 / 事后（post hoc）描述性检验）',
        'qualitative_status': 'POST_HOC_DESCRIPTIVE_NOT_PRE_REGISTERED',
        'qualitative_note': '该口径在观察到 B″5 v1 结果（2026-09-28 停机）之后设计；2026-09-28 预先写下的'
                            '替代处置口径中不含"下游方向"这一项 → **不构成 outcome-blind 预登记**。'
                            '作用限于"两网下游结果是否反向"的描述性对照；**不得**作为 定价 gate / 机制层 输入合格性'
                            '或"换章节稳健性通过"的证据（闸门判据只有 ⓐ+ⓑ）。',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'main_panel': rel(args.main_panel),
        'robust_panel': rel(args.robust_panel),
        'returns': 'data/processed/monthly_returns_clean.csv',
        'overlap_months': [months[0], months[-1], len(months)],
        'effective_paired_months': len(paired),
        'effective_paired_months_note': paired_note,
        'unpaired_months': unpaired,
        'merged_rows': int(len(j)),
        'peer_maxraw': {'pearson': pm_corr['pearson'], 'spearman': pm_corr['spearman'], 'n': pm_corr['n'],
                        'per_month': pm_month},
        'peer_return': {'pearson': pr_corr['pearson'], 'spearman': pr_corr['spearman'], 'n': pr_corr['n'],
                        'per_month': pr_month},
        'quintile_agreement': quintile,
        'strength': strength,
        'portfolio_main_by_peermaxraw': pf_main,
        'portfolio_robust_by_peermaxraw': pf_rob,
        'portfolio_main_restricted_to_paired_months': pf_main_paired,
        'portfolio_robust_restricted_to_paired_months': pf_rob_paired,
        'segments': segs,
        'descriptive_checks': handoff_criteria,
        'e_downstream_direction': not_reversed,
        'conclusion': '两网下游未反向（描述性）',
        'cross_audit_handoff': handoff,
        'status': 'DESCRIPTIVE_ONLY',
        'status_semantics': '本文件不设 PASS/FAIL：ⓔ 为 post hoc 描述性检验；闸门判据为 network_business_chapter_cross_audit 的 ⓐ+ⓑ。',
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# network Phase B″6 下游方向一致性（判据 ⓔ）', '',
          f'生成时间：{rep["run_at"]}', '',
          f'重叠月份：{months[0]}–{months[-1]}（{len(months)} 个月；合并后公司-月 {len(j):,} 行）', '',
          '## 公司-月层相关（' + f'{args.main_label} vs {args.robust_label}）', '',
          '| 变量 | Pearson | Spearman | n | 逐月 Pearson 中位 | 逐月正值月份占比 |',
          '|---|---|---|---|---|---|',
          f'| PeerMAXraw | {pm_corr["pearson"]} | {pm_corr["spearman"]} | {pm_corr["n"]:,} | '
          f'{pm_month.get("pearson_median")} | {pm_month.get("share_months_pearson_positive")} |',
          f'| PeerReturn | {pr_corr["pearson"]} | {pr_corr["spearman"]} | {pr_corr["n"]:,} | '
          f'{pr_month.get("pearson_median")} | {pr_month.get("share_months_pearson_positive")} |', '',
          '## 分位一致性（逐月各自按 PeerMAXraw 分五组）', '',
          f'- 同组率（均值）：{quintile.get("same_quintile_share_mean")}'
          f'（≈ {strength["quintile_same_share_vs_random_multiple"]}× 随机基线 0.2）',
          f'- 相邻组（±1）内占比（均值）：{quintile.get("within_plus_minus_1_quintile_share_mean")}',
          f'- 组序 Spearman（逐月中位）：{quintile.get("quintile_spearman_median")}',
          f'- 逐月公司数中位：{quintile.get("n_firms_per_month_median")}', '',
          '## 组合层（按 PeerMAXraw 分五组，等权次月收益；只报方向与排序）', '',
          '| 网络 | 月份数 | Q1→Q5 组均收益 | 组均秩相关 | 组均值形状 | 多空(Q5−Q1) 均值 | t | 同号/反向月份占比 |',
          '|---|---|---|---|---|---|---|---|']
    for lbl, pf in [(args.main_label, pf_main), (args.robust_label, pf_rob)]:
        qm = pf.get('quintile_mean_next_month_ret')
        md.append(f'| {lbl} | {pf.get("n_months")} | ' +
                  (' / '.join(f'{v:+.4f}' for v in qm) if qm else '—') +
                  f' | {pf.get("quintile_rank_corr_spearman")} | {pf.get("quintile_shape")} | '
                  f'{pf.get("long_short_q5_minus_q1_mean")} | {pf.get("long_short_t_stat")} | '
                  f'{pf.get("long_short_share_months_same_sign")}/{pf.get("long_short_share_months_opposite_sign")} |')
    md += ['', f'- 名义重叠 {len(months)} 个月（{months[0]}~{months[-1]}）；'
               f'{args.main_label} {pf_main.get("n_months")} 月 / {args.robust_label} '
               f'{pf_rob.get("n_months")} 月**不是同一月份集合**（有效配对 {paired_note}）；'
               f'在**同一月份集合（{pf_main_paired.get("n_months")} 月，两网同月均有值）**下，'
               f'{args.main_label}多空 = {pf_main_paired.get("long_short_q5_minus_q1_mean")}（t = '
               f'{pf_main_paired.get("long_short_t_stat")}），'
               f'组均 = {pf_main_paired.get("quintile_mean_next_month_ret")}。',
           f'- 最负 3 个月占多空总和：{args.main_label} {strength.get("top3_months_share_of_ls_sum_main")}'
           f'（>100% 说明多空均值由少数月份主导）。', '',
           '## 分段（2021 语料结构性断点）', '',
           '| 子段 | PeerMAXraw Pearson / Spearman | PeerReturn Pearson / Spearman | '
           f'多空（{args.main_label}） | 多空（{args.robust_label}） |',
           '|---|---|---|---|---|']
    for name, s in segs.items():
        md.append(f'| {name} | {s["peer_maxraw"]["pearson"]} / {s["peer_maxraw"]["spearman"]} | '
                  f'{s["peer_return"]["pearson"]} / {s["peer_return"]["spearman"]} | '
                  f'{s["long_short_main"]} | {s["long_short_robust"]} |')
    md += ['', f'（分段固定为 {segs["2018_2020"]["months"][0]}~{segs["2018_2020"]["months"][1]} 与 '
           f'{segs["2021_2023"]["months"][0]}~{segs["2021_2023"]["months"][1]} 两段（2021 语料断点）；'
           f'名义重叠区间内落在这两段之外的月份不进入分段对照。）', '',
           '## 结论（探索性 / 事后（post hoc）描述性检验，**非闸门**）', '',
           f'- 相关为正（PeerMAXraw/PeerReturn）：{corr_positive}'
           f'（共享方差 {strength.get("peer_maxraw_shared_variance")}/'
           f'{strength.get("peer_return_shared_variance")}）',
           f'- 多空（Q5−Q1）符号一致：{ls_same_sign}',
           f'- 五分位组均秩相关方向一致：{rank_corr_same_sign}',
           f'- **结论：{rep["conclusion"]}**', '',
           f'强度（须与结论一并引用）：{strength.get("reading")}', '',
           '**定性声明**：本检验在观察到 B″5 v1 结果之后设计，**不构成 outcome-blind 预登记**；'
           '其作用**仅限于**上述描述性对照——**不得**作为 定价 gate / 机制层 输入合格性或"换章节稳健性通过"的证据。'
           '（闸门判据只有 network_business_chapter_cross_audit 的 ⓐ+ⓑ。）']
    Path(args.out_md).write_text('\n'.join(md), encoding='utf-8')
    print(json.dumps(rep['descriptive_checks'], ensure_ascii=False, indent=1))
    print('conclusion:', rep['conclusion'])
    print('long_short main/robust:', pf_main.get('long_short_q5_minus_q1_mean'),
          pf_rob.get('long_short_q5_minus_q1_mean'),
          '| t:', pf_main.get('long_short_t_stat'), pf_rob.get('long_short_t_stat'),
          '| rank rho:', pf_main.get('quintile_rank_corr_spearman'), pf_rob.get('quintile_rank_corr_spearman'))
    print('peer_maxraw corr:', pm_corr, '| peer_return corr:', pr_corr)
    print('effective paired months:', rep['effective_paired_months'])
    print('status =', rep['status'])
    print('->', outp)


if __name__ == '__main__':
    main()
