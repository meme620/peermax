#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 对表反号成因排查 —— 第一批检验 T1–T6（2026-10-20，只读既有产物）。

靶子：`reports/audits/s3_sign_reversal_hypotheses_20261020.md`（普查表 H06/H06B/H10/H14/H01/H09）
      现行读数：`reports/data_gates/G0/competition_pricing_correspondence.{json,md}`（MD&A 版）
                `reports/data_gates/G0/s3_pc_pricing_correspondence_fulltext.{json,md}`（全文版）

== 判据（先登记后跑；抄自普查表 §统一判据）==
  基准读数（主口径 MD&A 登记样本 EW lag6 单变量 FM）：PC = −0.5068%（t = −1.4365）；
  原文最近似样本 EW lag12 = −0.4559%（t = −1.9998）；加控制登记样本 EW lag6 = −0.4116%（t = −1.1902）。
  · 元凶（充分）= 该口径下主读数层 PC 系数 ≥ 0（或逐月 rank IC ≥ 0），且 4 个子层
    （EW/VW × 五分位/十分位 H−L、单变量/加控制 FM）中至少 3 层同向翻转；
  · 部分贡献 = 系数向 0 移动 ≥ 50% 但仍为负；
  · 排除 = 系数移动 < 25%（|Δβ| < 0.25·|β_base|）或仍同号且 |t| 不降。
  三选一归因：构造/实现 bug（我们的错）／口径差异（论文 vs 我们）／真实经济差异。

== 检验清单（本轮范围；不重建网络，全部复用既有产物）==
  T1 方向与单调性核验（H14/H10）：十分位 D1..D10 组均值/中位数收益表（EW/VW），D1 = PC 最低，
     H−L = D10 − D1；核对组合层（等权）与 FM 层方向一致性。
  T2 尾部敏感性（H06）：① 月度收益 winsorize [0.01,0.99] 后重跑 FM + 十分位 H−L（并报未 winsor 的
     `next_R` 与更紧 [0.025,0.975] 两个附加口径作对照）；② 剔除 |月收益| > 50% 的观测；③ 组中位数差替代均值差。
  T3 规模混淆（H06B）：① size 三分位内各自排序（层内十分位）→ 三层 H−L；② 剔除当月市值最小 30% 后重跑；
     ③ VW 组合权重集中度（最大权重占比 / 权重 HHI）。
  T4 时点/滞后（H01）：extra ∈ {0, 12, 24} 个月（0 = 我方 PIT 口径；12 = 原文 t−12「系列平移」读法；
     24 = 附加）→ 单变量 FM + 十分位 H−L 三项对照。
  T5 回归实现（H09）：NW 滞后 6/12/18；含常数项 vs 去均值；逐月不加权 vs 按 PC 层级加权（同月同一 PC 值
     = 同一团，权重 = 1/该 PC 值的公司数）。
  T6 秩相关/IC（H06）：逐月 Spearman rank IC 时序均值 + t（NW 与 iid）；十组均值/中位数两张表 +
     单调性诊断（相邻组符号反转数、组序 Spearman）。

统一口径：主用**登记样本**（`m1_common_sample_ma_teacher_v2.parquet` + `m1_in_fm_sample`），
  每项并报**原文最近似样本**（PC 与 R_next 可得，无额外筛选）；PC 时点 = 面板 `network_version`
  （PIT `max{available_at < month}`，available_at = (y+1)-04-30）；收益 = `R_next`（raw 小数，
  面板构造期已按 registry `winsorization_pct` 月度截面 winsorize）。

纪律：只读既有产物（面板 / PC / 因子）；不覆盖任何既有文件；不改任何 headline_value / 冻结值；
  产物 = 本脚本 + `reports/data_gates/G0/s3_sign_reversal_batch1.{json,md}` 两件。

用法：python scripts/competition_sign_reversal_probe.py
      python scripts/competition_sign_reversal_probe.py --tests T1 T6 --out-prefix /tmp/probe
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_lib import (
    OUT_DIR, ROOT, fm_regression, newey_west, param_value, rank_average_groups, write_json,
)

PANEL_SAMPLE = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'
PC_DIR = ROOT / 'data/processed/network_ma_teacher'
CLUSTER_DIR = ROOT / 'data/processed/network_ma_teacher/clusters'
PC_YEARS = list(range(2017, 2023))
CONTROLS = ['size', 'bm', 'mom', 'rev', 'illiq']
PROGRESS = ROOT / 'artifacts/tmp/sign_reversal_progress.md'
OUT_PREFIX = 's3_sign_reversal_batch1'
STEP_TAG = 'B1'

# 登记基准（MD&A 版对表读数，抄自 s3_pc_pricing_correspondence.json）
BASELINE = {
    'registered_EW_lag6_univariate': {'beta_pct': -0.5068, 't_nw': -1.4365},
    'registered_EW_lag6_controls': {'beta_pct': -0.4116, 't_nw': -1.1902},
    'paper_like_EW_lag12_univariate': {'beta_pct': -0.4559, 't_nw': -1.9998},
    'registered_quintile_EW_HL_raw_pct': -0.2031,
    'registered_quintile_EW_alpha3_pct': -0.3482,
}

NW_LAG = None   # 由 registry 注入
MIN_N = None


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def psay(step: str, msg: str) -> None:
    line = f'[{STEP_TAG}-{step}] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def pct(x, nd=4):
    v = _f(x)
    return None if v is None else round(v * 100.0, nd)


def r4(x):
    v = _f(x)
    return None if v is None else round(v, 4)


def wmed(v: np.ndarray, w: np.ndarray) -> float:
    """加权中位数（累计权重首次 ≥ 50%）。"""
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return float('nan')
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    return float(v[int(np.searchsorted(c, 0.5 * c[-1]))])


# ---------------------------------------------------------------- 装载
def load_merged() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL_SAMPLE)
    frames = []
    for y in PC_YEARS:
        d = pd.read_csv(PC_DIR / f'{y}_pc.csv', dtype={'focal_id': str},
                        usecols=['focal_id', 'text_year', 'pc', 'hhi', 'hhi_excl_self',
                                 'n_competitors', 'revenue_coverage', 'available_at'])
        frames.append(d)
    pc = pd.concat(frames, ignore_index=True)
    pc = pc[pc['pc'].notna()].copy()
    pc['PC'] = pc['pc'].astype(float)
    panel['text_year'] = pd.to_numeric(panel['network_version'], errors='coerce')
    m = panel.merge(pc[['focal_id', 'text_year', 'PC', 'hhi', 'n_competitors',
                        'revenue_coverage']],
                    on=['focal_id', 'text_year'], how='left')
    return m


def samples(m: pd.DataFrame) -> dict:
    pc_ok = m['PC'].notna() & m['R_next'].notna()
    reg = m['m1_in_fm_sample'].astype(bool)
    ctrl = m[CONTROLS].notna().all(axis=1)
    out = {
        'paper_like': m[pc_ok].copy(),
        'registered': m[pc_ok & reg].copy(),
        'paper_like_ctrl': m[pc_ok & ctrl].copy(),
        'registered_ctrl': m[pc_ok & reg & ctrl].copy(),
    }
    return out


# ---------------------------------------------------------------- FM 工具
def fm_run(samp: pd.DataFrame, ycol: str, xcols, weight_col=None, nw_lag=None,
           label: str = '') -> dict:
    nw_lag = NW_LAG if nw_lag is None else nw_lag
    blk, _ = fm_regression(samp, ycol, xcols, weight_col=weight_col, min_n=MIN_N, nw_lag=nw_lag)
    s = blk['coefs'].get('PC') or blk['coefs'].get(xcols[0])
    out = {'label': label, 'ycol': ycol, 'xcols': list(xcols), 'weight': weight_col,
           'nw_lag': nw_lag, 'beta_pct': pct(s['beta']), 'beta_raw': r4(s['beta']),
           't_nw': r4(s['t_nw']), 't_iid': r4(s['t_iid']), 'se_nw_pct': pct(s['se_nw']),
           'n_months_used': blk['n_months_used']}
    return out


def fm_demeaned(samp: pd.DataFrame, ycol: str, xcol: str = 'PC', nw_lag=None) -> dict:
    """逐月去均值（y 与 x 同去月内均值，无常数项）→ 系数序列 + NW。"""
    nw_lag = NW_LAG if nw_lag is None else nw_lag
    coefs = []
    for _, sub in samp.groupby('month', sort=True):
        sub = sub.dropna(subset=[ycol, xcol])
        if len(sub) < MIN_N:
            continue
        y = sub[ycol].to_numpy(float)
        x = sub[xcol].to_numpy(float)
        yd = y - y.mean()
        xd = x - x.mean()
        den = float(xd @ xd)
        if den <= 0:
            continue
        coefs.append(float(xd @ yd / den))
    nw = newey_west(coefs, lag=nw_lag)
    return {'label': 'demeaned_no_const', 'beta_pct': pct(nw['mean']), 't_nw': r4(nw['t_nw']),
            'n_months_used': nw['T']}


# ---------------------------------------------------------------- 组合层工具
def group_table(samp: pd.DataFrame, n_groups: int, wcol=None, min_n=None) -> pd.DataFrame:
    """逐月 PC 截面 rank_average 分组 → 每组（EW 或 VW）月收益；返回 per-month 宽表 + HL。"""
    min_n = MIN_N if min_n is None else min_n
    cols = ['month', 'PC', 'R_next', 'market_equity']
    if wcol is not None and wcol not in cols:
        cols.append(wcol)
    d = samp[cols].dropna(subset=['PC', 'R_next']).copy()
    if wcol is not None:
        d = d[d[wcol].notna() & (d[wcol] > 0)]
    d['grp'] = d.groupby('month', group_keys=False)['PC'].transform(
        lambda s: rank_average_groups(s, n_groups))
    d = d[d['grp'].notna()]
    rows = []
    for m, sub in d.groupby('month', sort=True):
        if len(sub) < min_n:
            continue
        rec = {'month': m, 'n': int(len(sub))}
        for g in range(1, n_groups + 1):
            gg = sub[sub['grp'] == g]
            if gg.empty:
                rec[f'g{g}'] = None
                continue
            v = gg['R_next'].to_numpy(float)
            if wcol is None:
                rec[f'g{g}'] = float(v.mean())
            else:
                rec[f'g{g}'] = float(np.average(v, weights=gg[wcol].to_numpy(float)))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values('month') if rows else pd.DataFrame()


def group_median_table(samp: pd.DataFrame, n_groups: int, wcol=None, min_n=None) -> pd.DataFrame:
    min_n = MIN_N if min_n is None else min_n
    cols = ['month', 'PC', 'R_next', 'market_equity']
    if wcol is not None and wcol not in cols:
        cols.append(wcol)
    d = samp[cols].dropna(subset=['PC', 'R_next']).copy()
    if wcol is not None:
        d = d[d[wcol].notna() & (d[wcol] > 0)]
    d['grp'] = d.groupby('month', group_keys=False)['PC'].transform(
        lambda s: rank_average_groups(s, n_groups))
    d = d[d['grp'].notna()]
    rows = []
    for m, sub in d.groupby('month', sort=True):
        if len(sub) < min_n:
            continue
        rec = {'month': m, 'n': int(len(sub))}
        for g in range(1, n_groups + 1):
            gg = sub[sub['grp'] == g]
            if gg.empty:
                rec[f'g{g}'] = None
                continue
            v = gg['R_next'].to_numpy(float)
            rec[f'g{g}'] = (float(np.median(v)) if wcol is None
                            else wmed(v, gg[wcol].to_numpy(float)))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values('month') if rows else pd.DataFrame()


def hl_summary(t: pd.DataFrame, n_groups: int, lag=None) -> dict:
    lag = NW_LAG if lag is None else lag
    if t.empty:
        return {'status': 'NOT_COMPUTABLE'}
    t = t.copy()
    t['HL'] = t[f'g{n_groups}'] - t['g1']
    nw = newey_west(t['HL'], lag=lag)
    means = {f'Q{g}': pct(t[f'g{g}'].mean()) for g in range(1, n_groups + 1)}
    return {'n_months': int(len(t)), 'n_months_effective': int(nw['T']),
            'group_mean_pct': means, 'HL_pct': pct(nw['mean']), 'HL_t_nw': r4(nw['t_nw']),
            'HL_se_pct': pct(nw['se_nw'])}


def hl_median_summary(samp: pd.DataFrame, n_groups: int, wcol=None, lag=None) -> dict:
    t = group_median_table(samp, n_groups, wcol=wcol)
    d = hl_summary(t, n_groups, lag=lag)
    if d.get('status') == 'NOT_COMPUTABLE':
        return d
    med = {f'Q{g}': pct(t[f'g{g}'].median()) for g in range(1, n_groups + 1)}
    d['group_median_of_monthly_median_pct'] = med
    return d


def monotonicity(t: pd.DataFrame, n_groups: int) -> dict:
    if t.empty:
        return {'status': 'NOT_COMPUTABLE'}
    gm = np.array([_f(t[f'g{g}'].mean()) for g in range(1, n_groups + 1)], dtype=float)
    idx = np.arange(1, n_groups + 1, dtype=float)
    rho = float(pd.Series(gm).corr(pd.Series(idx), method='spearman'))
    # 相邻组符号反转数（组序一阶差分为正/负的切换次数）
    d1 = np.diff(gm)
    flips = int(np.sum(np.sign(d1[1:]) * np.sign(d1[:-1]) < 0))
    return {'group_mean_pct': [pct(v) for v in gm],
            'spearman_group_index_vs_group_mean': r4(rho),
            'n_adjacent_sign_flips': flips,
            'monotone_increasing': bool(flips == 0 and d1[0] > 0),
            'monotone_decreasing': bool(flips == 0 and d1[0] < 0)}


def rank_ic(samp: pd.DataFrame, ycol: str = 'R_next', xcol: str = 'PC', lag=None) -> dict:
    lag = NW_LAG if lag is None else lag
    vals = []
    for _, s in samp.groupby('month', sort=True):
        s = s[[ycol, xcol]].dropna()
        if len(s) < MIN_N or s[xcol].nunique() < 10 or s[ycol].nunique() < 10:
            continue
        vals.append(float(s[ycol].corr(s[xcol], method='spearman')))
    nw = newey_west(vals, lag=lag)
    return {'n_months': int(len(vals)), 'ic_mean': r4(nw['mean']), 'ic_t_nw': r4(nw['t_nw']),
            'ic_t_iid': r4(nw['t_iid']), 'ic_se_nw': r4(nw['se_nw'])}


# ---------------------------------------------------------------- 判定
def judge(base_beta_pct, new_beta_pct, base_t=None, new_t=None, layer_flips=None,
          n_layers=None) -> dict:
    """按普查表 §统一判据给三选一判定。base/new 单位 = %/单位 PC（主读数层）。"""
    b, n = _f(base_beta_pct), _f(new_beta_pct)
    if b is None or n is None:
        return {'verdict': 'NOT_COMPUTABLE', 'note': '读数缺失'}
    move = abs(n - b) / abs(b) if b != 0 else float('inf')
    same_sign = (n < 0) == (b < 0)
    if not same_sign and n >= 0 and layer_flips is not None and n_layers:
        if layer_flips >= 3:
            v = '元凶候选（改变结论）'
        elif layer_flips >= 2:
            v = '部分贡献（不足 3/4 层翻转）'
        else:
            v = '放大或缩小但方向不变（主层翻正但子层未同向）'
    elif not same_sign:
        v = '元凶候选（改变结论）'
    elif move >= 0.50:
        v = '放大或缩小但方向不变（部分贡献：向 0 移动 ≥50%）'
    elif move >= 0.25:
        v = '放大或缩小但方向不变'
    else:
        v = '不改变结论（排除）'
    return {'verdict': v, 'beta_move_frac': r4(move), 'same_sign': bool(same_sign),
            'base_beta_pct': b, 'new_beta_pct': n, 'base_t': base_t, 'new_t': new_t,
            'n_layers_flipped': layer_flips, 'n_layers_checked': n_layers}


# ---------------------------------------------------------------- 主流程
# ---------------------------------------------------------------- 报告生成
def _dt(x):
    return '—' if x is None else f'{x}'


def build_md(o: dict) -> str:
    t1 = o.get('T1_direction_and_monotonicity', {})
    t2 = o.get('T2_tail_sensitivity', {})
    t3 = o.get('T3_size_confound', {})
    t4 = o.get('T4_timing_lag', {})
    t5 = o.get('T5_fm_implementation', {})
    t5b = o.get('T5B_cluster_level', {})
    t6 = o.get('T6_rank_ic', {})
    tv = o.get('test_verdicts', {})
    L = []
    L.append('# S3 项目自建 PC 负号成因排查 · 第一批检验 T1–T6（2026-10-20；2026-10-27 口径更正）\n')
    L.append('> **性质**：只读既有产物的成因排查产物（不重建网络；未覆盖任何既有文件、未改任何 '
             'headline_value / 冻结值 / 门槛）。')
    L.append('> **判据（先登记后跑）**：元凶（充分）= 主读数层 PC 系数 ≥ 0 且 4 子层（EW/VW × 五分位/十分位 '
             'H−L、单变量/加控制 FM）≥3 层同向翻转；部分贡献 = 向 0 移动 ≥50% 但仍为负；'
             '排除 = 移动 < 25% 或同号且 |t| 不降。')
    L.append('> **靶子**：`reports/audits/s3_sign_reversal_hypotheses_20261020.md`（普查表 H01/H06/H06B/H09/H10/H14）；'
             '现行读数 = `reports/data_gates/G0/competition_pricing_correspondence.{json,md}`（MD&A 主口径）。\n')
    L.append('## 0. 输入、样本与基准复现\n')
    L.append('| 项 | 值 |')
    L.append('|---|---|')
    L.append(f"| 面板 / PC | `{o['inputs']['panel']}` + `{o['inputs']['pc_dir']}/{{2017..2022}}_pc.csv` |")
    L.append(f"| 样本（行 / 月） | " + '；'.join(
        f"{k} {v['rows']:,} / {v['months']}" for k, v in o['inputs']['samples'].items()) + ' |')
    L.append(f"| 收益 | {o['inputs']['ycol']} |")
    L.append(f"| 符号核验 | max\\|PC + hhi\\| = {o['sign_convention_check']['max_abs_pc_plus_hhi']}；"
             f"PC ∈ [−1,0] = {o['sign_convention_check']['pc_in_neg1_0']} |")
    L.append(f"| 基准复现 | 登记样本 EW lag{NW_LAG} 单变量 = "
             f"{o['baseline_reproduction'][f'registered_EW_lag{NW_LAG}_univariate']['beta_pct']}%"
             f"（登记 −0.5068）；原文最近似 EW lag12 = "
             f"{o['baseline_reproduction']['paper_like_EW_lag12_univariate']['beta_pct']}%（登记 −0.4559） |")
    L.append('')
    # ---------------- T1
    L.append('## 1. T1 方向与单调性核验（H14/H10）\n')
    L.append('规格：逐月 PC 截面 `rank_average` 十分位（**D1 = PC 最低**；H = D10；**H−L = D10 − D1**）；'
             'EW = 等权、VW = 总市值加权；组收益同时报均值与中位数。\n')
    for tag, name in (('registered', '登记样本'), ('paper_like', '原文最近似样本')):
        ew, vw = t1.get(f'{tag}_decile_EW'), t1.get(f'{tag}_decile_VW')
        if not ew:
            continue
        L.append(f'**{name}**\n')
        L.append('| 十分位 | EW 均值 | EW 中位数 | VW 均值 | VW 中位数 |')
        L.append('|---|---|---|---|---|')
        for g in range(1, 11):
            L.append(f"| D{g} | {_dt(ew['group_mean_pct'][f'Q{g}'])} | "
                     f"{_dt(ew['group_median_of_monthly_median_pct'][f'Q{g}'])} | "
                     f"{_dt(vw['group_mean_pct'][f'Q{g}'])} | "
                     f"{_dt(vw['group_median_of_monthly_median_pct'][f'Q{g}'])} |")
        L.append(f"| **H−L** | **{_dt(ew['HL_mean_pct'])}（t = {_dt(ew['HL_mean_t_nw'])}）** | "
                 f"{_dt(ew['HL_median_pct'])}（t = {_dt(ew['HL_median_t_nw'])}） | "
                 f"{_dt(vw['HL_mean_pct'])}（t = {_dt(vw['HL_mean_t_nw'])}） | "
                 f"{_dt(vw['HL_median_pct'])}（t = {_dt(vw['HL_median_t_nw'])}） |")
        L.append(f"\n- 五分位 EW H−L = {_dt(ew['quintile_HL_mean_pct'])}%（t = {_dt(ew['quintile_HL_mean_t_nw'])}）；"
                 f"单调性（组均值）：Spearman(组序) = "
                 f"{_dt(ew['monotonicity_mean']['spearman_group_index_vs_group_mean'])}、相邻符号反转 "
                 f"{ew['monotonicity_mean']['n_adjacent_sign_flips']} 次、单调递增/"
                 f"递减 = {ew['monotonicity_mean']['monotone_increasing']}/"
                 f"{ew['monotonicity_mean']['monotone_decreasing']}；"
                 f"组中位数单调性 Spearman = "
                 f"{_dt(t6.get(tag, {}).get('monotonicity_median', {}).get('spearman_group_index_vs_group_mean'))}\n")
        fmv = t1.get(f'{tag}_FM_vs_portfolio', {})
        L.append(f"- 组合层 vs FM 层：FM 单变量 β = {_dt(fmv.get('FM_univariate_beta_pct'))}%、"
                 f"十分位 EW H−L = {_dt(fmv.get('decile_EW_HL_mean_pct'))}%、"
                 f"五分位 EW H−L = {_dt(fmv.get('quintile_EW_HL_mean_pct'))}% → "
                 f"**方向{'一致' if fmv.get('sign_consistent_FM_vs_portfolio') else '不一致'}**\n")
    if 'T1' in tv:
        L.append(f"**判定**：{tv['T1']['verdict']}。依据：{tv['T1']['basis']}\n")
    # ---------------- T2
    L.append('## 2. T2 尾部敏感性（H06）\n')
    L.append('| 子项 | 登记样本 β（t） | 原文最近似 β（t） | 登记十分位 EW H−L（t） |')
    L.append('|---|---|---|---|')
    for key, label in (('winsor_1_99', '① 再 winsorize [0.01,0.99]（幂等核验）'),
                       ('winsor_2_5_97_5', '①′ 更紧 winsorize [0.025,0.975]（附加）'),
                       ('unwinsorized_next_R', '①″ 未 winsor 的 next_R（附加）'),
                       ('drop_abs_gt_50pct_on_next_R_raw', '② 剔 \\|月收益\\| > 50%（按 next_R）'),
                       ('drop_abs_gt_50pct_on_R_next', '②′ 剔 \\|R_next\\| > 50%')):
        a, b = t2.get('registered', {}).get(key), t2.get('paper_like', {}).get(key)
        if not a:
            continue
        L.append(f"| {label} | {_dt(a['fm_univariate']['beta_pct'])}（{_dt(a['fm_univariate']['t_nw'])}） | "
                 f"{_dt(b['fm_univariate']['beta_pct'])}（{_dt(b['fm_univariate']['t_nw'])}） | "
                 f"{_dt(a['decile_EW']['HL_pct'])}（{_dt(a['decile_EW']['HL_t_nw'])}） |")
    L.append('')
    L.append('③ 组中位数差替代均值差（H−L = 组中位数之差，逐月序列 → NW）：\n')
    L.append('| 组合 | 登记 H−L（t） | 原文最近似 H−L（t） |')
    L.append('|---|---|---|')
    for key, label in (('decile_EW', '十分位 EW'), ('decile_VW', '十分位 VW'), ('quintile_EW', '五分位 EW')):
        a = t2.get('registered', {}).get('group_median_diff', {}).get(key, {})
        b = t2.get('paper_like', {}).get('group_median_diff', {}).get(key, {})
        L.append(f"| {label} | {_dt(a.get('HL_pct'))}（{_dt(a.get('HL_t_nw'))}） | "
                 f"{_dt(b.get('HL_pct'))}（{_dt(b.get('HL_t_nw'))}） |")
    if 'T2' in tv:
        L.append(f"\n**判定**：{tv['T2']['verdict']}。依据：{tv['T2']['basis']}\n")
    # ---------------- T3
    L.append('## 3. T3 规模混淆（H06B）\n')
    L.append('① size 三分位内各自排序（层内十分位 EW）→ 三层 H−L 与层内 rank IC：\n')
    L.append('| 样本 | 层 | 行数 | H−L（%） | t(NW) | 层内 rank IC（t） |')
    L.append('|---|---|---|---|---|---|')
    for tag, name in (('registered', '登记'), ('paper_like', '原文最近似')):
        for k, v in t3.get(tag, {}).get('size_tercile_within_deciles_EW', {}).items():
            L.append(f"| {name} | {k} | {v['n_rows']:,} | {_dt(v['HL_pct'])} | {_dt(v['HL_t_nw'])} | "
                     f"{_dt(v['rank_IC']['ic_mean'])}（{_dt(v['rank_IC']['ic_t_nw'])}） |")
    L.append('')
    L.append('② 剔除当月市值最小 30%（及附加 50%）后重跑：\n')
    L.append('| 样本 | 剔 30% β（t） | 剔 30% 加控制 β | 剔 30% 十分位 EW/VW H−L | 剔 50% β（t） |')
    L.append('|---|---|---|---|---|')
    for tag, name in (('registered', '登记'), ('paper_like', '原文最近似')):
        r = t3.get(tag, {}).get('drop_smallest_30pct_market_equity', {})
        r5 = t3.get(tag, {}).get('drop_smallest_50pct_market_equity', {})
        if not r:
            continue
        L.append(f"| {name} | {_dt(r['fm_univariate']['beta_pct'])}（{_dt(r['fm_univariate']['t_nw'])}） | "
                 f"{_dt((r.get('fm_with_controls') or {}).get('beta_pct'))} | "
                 f"{_dt(r['decile_EW']['HL_pct'])} / {_dt(r['decile_VW']['HL_pct'])} | "
                 f"{_dt(r5.get('fm_univariate', {}).get('beta_pct'))}"
                 f"（{_dt(r5.get('fm_univariate', {}).get('t_nw'))}） |")
    L.append('')
    L.append('③ VW 组合权重集中度（十分位 VW 组内最大权重占比均值 / 全样本最大）：\n')
    L.append('| 样本 | D1 | D5 | D10 | 全局最大 |')
    L.append('|---|---|---|---|---|')
    for tag, name in (('registered', '登记'), ('paper_like', '原文最近似')):
        c = t3.get(tag, {}).get('VW_weight_concentration', {})
        if not c:
            continue
        ms = c['mean_max_share_by_decile']
        L.append(f"| {name} | {_dt(ms.get('D1'))} | {_dt(ms.get('D5'))} | {_dt(ms.get('D10'))} | "
                 f"{_dt(c['max_max_share'])} |")
    if 'T3' in tv:
        L.append(f"\n**判定**：{tv['T3']['verdict']}。依据：{tv['T3']['basis']}\n")
    # ---------------- T4
    L.append('## 4. T4 时点/滞后（H01）\n')
    L.append('extra=0 = 我方 PIT 口径（文本年 FY(t−1)）；extra=12 = 原文 `PC_{i,t−12}`「系列平移」读法'
             '（FY(t−2)）；extra=24 = 附加。\n')
    L.append('| extra（月） | 登记样本行数 / 月数 | 窗口 | 单变量 β（t） | 加控制 β | VW β（t） | 十分位 H−L（t） | 五分位 H−L | rank IC（t） |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for extra in (0, 12, 24):
        r = t4.get(f'extra_{extra}', {}).get('registered')
        if not r:
            continue
        L.append(f"| {extra} | {r['n_rows']:,} / {r['n_months']} | {r['first_month']}..{r['last_month']} | "
                 f"{_dt(r['fm_univariate_EW']['beta_pct'])}（{_dt(r['fm_univariate_EW']['t_nw'])}） | "
                 f"{_dt(r['fm_with_controls']['beta_pct'])} | "
                 f"{_dt(r['fm_univariate_VW']['beta_pct'])}（{_dt(r['fm_univariate_VW']['t_nw'])}） | "
                 f"{_dt(r['decile_EW']['HL_pct'])}（{_dt(r['decile_EW']['HL_t_nw'])}） | "
                 f"{_dt(r['quintile_EW']['HL_pct'])} | "
                 f"{_dt(r['rank_IC']['ic_mean'])}（{_dt(r['rank_IC']['ic_t_nw'])}） |")
    if 'T4' in tv:
        L.append(f"\n**判定**：{tv['T4']['verdict']}。依据：{tv['T4']['basis']}\n")
    # ---------------- T5
    L.append('## 5. T5 回归实现（H09）\n')
    L.append('| 样本 | 规格 | β（%） | t(NW) | t(iid) | 月数 |')
    L.append('|---|---|---|---|---|---|')
    for tag, name in (('registered', '登记'), ('paper_like', '原文最近似')):
        r = t5.get(tag)
        if not r:
            continue
        for lag in (6, 12, 18):
            b = r['NW_lags'][f'lag{lag}']
            L.append(f"| {name} | NW lag = {lag}（含常数项） | {_dt(b['beta_pct'])} | {_dt(b['t_nw'])} | "
                     f"{_dt(b['t_iid'])} | {b['n_months_used']} |")
        d = r['demeaned']['univariate']
        L.append(f"| {name} | 逐月去均值（无常数项） | {_dt(d['beta_pct'])} | {_dt(d['t_nw'])} | — | "
                 f"{d['n_months_used']} |")
        w = r['pc_level_weighted']['univariate']
        L.append(f"| {name} | PC 层级加权（≈ 团层等权） | {_dt(w['beta_pct'])} | {_dt(w['t_nw'])} | "
                 f"{_dt(w['t_iid'])} | {w['n_months_used']} |")
        wc = r['pc_level_weighted'].get('with_controls')
        if wc:
            L.append(f"| {name} | PC 层级加权 + 5 控制 | {_dt(wc['beta_pct'])} | {_dt(wc['t_nw'])} | — | "
                     f"{wc['n_months_used']} |")
    L.append(f"\n（每月唯一 PC 层级数均值 = "
             f"{_dt(t5.get('registered', {}).get('pc_level_weighted', {}).get('n_distinct_pc_levels_per_month_mean'))}，"
             f"与每月团数相同 → PC 层级加权 ≡ 团层等权，见 §5B。）")
    if 'T5' in tv:
        L.append(f"\n**判定**：{tv['T5']['verdict']}。依据：{tv['T5']['basis']}\n")
    # ---------------- T5B
    if t5b:
        L.append('## 5B. T5B 团层聚合诊断（H06/H02 联动）\n')
        L.append(f"PC 值 = 团的代理核验：`{{y}}_clusters.csv` 的 (文本年, 团) 组内 PC 唯一性 = "
                 f"{t5b['proxy_check']['n_cluster_years']} 组 / 违反 {t5b['proxy_check']['n_with_multiple_PC']}；"
                 f"每月团数均值 = 每月唯一 PC 值数（登记样本 189.12）。\n")
        L.append('| 样本 | 团层等权 FM β（t） | 成员数加权 FM β（t） | 团层等权十分位 H−L（t） | 团层等权五分位 H−L（t） | 每团中位成员数 D1→D10 | corr(PC, log 成员数) |')
        L.append('|---|---|---|---|---|---|---|')
        for tag, name in (('registered', '登记'), ('paper_like', '原文最近似')):
            r = t5b.get(tag)
            if not r:
                continue
            med = r['median_members_by_PC_decile']
            L.append(f"| {name} | {_dt(r['fm_cluster_equal_weight']['beta_pct'])}"
                     f"（{_dt(r['fm_cluster_equal_weight']['t_nw'])}） | "
                     f"{_dt(r['fm_cluster_member_weighted']['beta_pct'])}"
                     f"（{_dt(r['fm_cluster_member_weighted']['t_nw'])}） | "
                     f"{_dt(r['decile_cluster_equal_weight']['HL_pct'])}"
                     f"（{_dt(r['decile_cluster_equal_weight']['HL_t_nw'])}） | "
                     f"{_dt(r['quintile_cluster_equal_weight']['HL_pct'])}"
                     f"（{_dt(r['quintile_cluster_equal_weight']['HL_t_nw'])}） | "
                     f"{med['D1']} → {med['D10']} | {_dt(r['corr_PC_log_members'])} |")
        L.append('')
        L.append('团层等权十分位组均值（登记样本）：' +
                 str(t5b.get('registered', {}).get('decile_cluster_equal_weight', {}).get('group_mean_pct')) + '\n')
        if 'T5B' in tv:
            L.append(f"**判定**：{tv['T5B']['verdict']}。依据：{tv['T5B']['basis']}\n")
    # ---------------- T6
    L.append('## 6. T6 秩相关/IC（H06）\n')
    L.append('| 样本 | 逐月 Spearman rank IC 均值 | t(NW) | t(iid) | 月数 | 组均值 Spearman(组序) | 组中位数 Spearman(组序) | 月内 Spearman 均值（t） |')
    L.append('|---|---|---|---|---|---|---|---|')
    for tag, name in (('registered', '登记'), ('paper_like', '原文最近似')):
        r = t6.get(tag)
        if not r:
            continue
        L.append(f"| {name} | {_dt(r['rank_IC']['ic_mean'])} | {_dt(r['rank_IC']['ic_t_nw'])} | "
                 f"{_dt(r['rank_IC']['ic_t_iid'])} | {r['rank_IC']['n_months']} | "
                 f"{_dt(r['monotonicity_mean']['spearman_group_index_vs_group_mean'])} | "
                 f"{_dt(r['monotonicity_median']['spearman_group_index_vs_group_mean'])} | "
                 f"{_dt(r['within_month_spearman_group_vs_mean']['mean_rho'])}"
                 f"（{_dt(r['within_month_spearman_group_vs_mean']['t_nw'])}） |")
    L.append('')
    for tag, name in (('registered', '登记样本'), ('paper_like', '原文最近似样本')):
        r = t6.get(tag)
        if not r:
            continue
        L.append(f'**{name}：十分位组均值 vs 组中位数（%）**\n')
        L.append('| 十分位 | ' + ' | '.join(f'D{g}' for g in range(1, 11)) + ' | H−L |')
        L.append('|---|' + '---|' * 11)
        L.append('| 组均值 | ' + ' | '.join(_dt(r['decile_mean_table_pct'][f'D{g}']) for g in range(1, 11)) +
                 f" | {_dt(r['HL_mean_pct'])} |")
        L.append('| 组中位数 | ' + ' | '.join(_dt(r['decile_median_table_pct'][f'D{g}']) for g in range(1, 11)) +
                 f" | {_dt(r['HL_median_pct'])} |")
        L.append('')
    if 'T6' in tv:
        L.append(f"**判定**：{tv['T6']['verdict']}。依据：{tv['T6']['basis']}\n")
    # ---------------- 汇总
    L.append('## 7. 汇总表（逐检验：是否改变「反号」结论 / 是否能解释掉负号）\n')
    L.append('| 检验 | 假设 | 主读数层读数（登记样本） | 是否改变「反号」结论 | 是否能解释掉负号 | 判定 |')
    L.append('|---|---|---|---|---|---|')
    def esc2(x):
        return str(x).replace('|', '\\|')

    order = ['T1', 'T2', 'T3', 'T4', 'T5', 'T5B', 'T6']
    for k in order:
        if k not in tv:
            continue
        L.append(f"| {k} | {esc2(tv[k]['hypothesis'])} | {esc2(tv[k]['result'].split('；')[0])} | "
                 f"{esc2(tv[k]['changed_conclusion'])} | {esc2(tv[k]['explains_negative_sign'])} | "
                 f"**{esc2(tv[k]['verdict'])}** |")
    L.append('')
    L.append(f'基准（主读数层）：登记样本 EW lag{NW_LAG} 单变量 PC = **−0.5068%**（t = −1.4365）；'
             '原文最近似样本 EW lag12 = **−0.4559%**（t = −1.9998）。\n')
    L.append('| 检验 | 规格 | β（%） | t(NW) | 备注 |')
    L.append('|---|---|---|---|---|')
    def esc(x):
        return str(x).replace('|', '\\|')

    for r in o['summary_table']:
        L.append(f"| {r['test']} | {esc(r['spec'])} | {_dt(r['beta_pct'])} | {_dt(r['t_nw'])} | "
                 f"{esc(r['note'])} |")
    L.append('')
    # ---------------- 结论
    L.append('## 8. 一句话结论（当前证据下，反号最可能来自哪里）\n')
    L.append(o['one_line_conclusion'] + '\n')
    L.append('## 9. 未执行 / 偏离清单\n')
    for it in o['unexecuted_or_deviations']:
        L.append(f'- {it}')
    L.append('')
    L.append('## 10. 产物与纪律\n')
    L.append(f"- 本报告与脚本（`scripts/{Path(__file__).name}`）为本轮**新建**；"
             f"输入 = `{o['inputs']['panel']}`（登记样本）、`{o['inputs']['pc_dir']}/"
             f"{{{PC_YEARS[0]}..{PC_YEARS[-1]}}}_pc.csv`（PC）、"
             f"`{CLUSTER_DIR.relative_to(ROOT).as_posix()}/{{y}}_clusters.csv`（T5B 团标签）。")
    L.append('- **未覆盖任何既有产物、未改任何 headline_value / 冻结值 / 门槛、未重建网络或面板**。')
    L.append('- 统计口径：FM = 逐月截面 OLS/WLS → 系数序列均值 + Newey-West(Bartlett) SE'
             f"（主 lag {NW_LAG}，registry `nw_lag`；T5 并报 12/18）；"
             '组合 = 逐月 PC 截面 `rank_average`（registry 冻结 tie 规则，H−L = 最高 − 最低）；'
             '收益单位统一为 %（β 为 %/单位 PC）。')
    return '\n'.join(L) + '\n'


def main() -> int:
    global NW_LAG, MIN_N, PROGRESS, OUT_PREFIX, STEP_TAG
    ap = argparse.ArgumentParser(description='S3 反号成因排查 T1–T6（只读）')
    ap.add_argument('--tests', nargs='*', default=['T1', 'T2', 'T3', 'T4', 'T5', 'T5B', 'T6'])
    ap.add_argument('--out-prefix', default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--progress', default=None)
    ap.add_argument('--step-tag', default=None)
    ap.add_argument('--generated-at', default='2026-10-20')
    args = ap.parse_args()

    tests = [t.upper() for t in args.tests]
    if args.out_prefix:
        OUT_PREFIX = args.out_prefix
    out_dir = ((Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir)
               if args.out_dir else OUT_DIR)
    if args.progress:
        PROGRESS = Path(args.progress) if Path(args.progress).is_absolute() else ROOT / args.progress
    if args.step_tag:
        STEP_TAG = args.step_tag
    json_out = out_dir / f'{OUT_PREFIX}.json'
    md_out = out_dir / f'{OUT_PREFIX}.md'

    NW_LAG = int(param_value('nw_lag'))
    MIN_N = int(param_value('m1_sample_and_spec_registry')['min_focal_count_per_month'])
    psay('0', f'状态 = 开始；检验集 = {tests}；registry nw_lag = {NW_LAG}、min_n/月 = {MIN_N}；'
              f'输出 = {_rel(json_out)}')

    m = load_merged()
    S = samples(m)
    psay('0', '样本状态 = ok：' + '；'.join(
        f'{k} {len(v):,} 行 / {v["month"].nunique()} 月' for k, v in S.items()))
    sign_check = {'max_abs_pc_plus_hhi': r4((m['PC'] + m['hhi']).abs().max()),
                  'pc_in_neg1_0': bool(m['PC'].dropna().between(-1, 0).all())}
    psay('0', f'符号核验状态 = {"ok" if sign_check["pc_in_neg1_0"] else "FAIL"}'
              f'（max|PC+hhi| = {sign_check["max_abs_pc_plus_hhi"]}；PC ∈ [−1,0] = {sign_check["pc_in_neg1_0"]}）')

    # 基准复现（防口径漂移）
    base = {}
    for tag, samp, ylag in (('registered', S['registered'], NW_LAG),
                            ('paper_like', S['paper_like'], 12)):
        base[f'{tag}_EW_lag{ylag}_univariate'] = fm_run(samp, 'R_next', ['PC'], nw_lag=ylag,
                                                        label=f'{tag}_univariate')
    base[f'registered_EW_lag{NW_LAG}_controls'] = fm_run(
        S['registered_ctrl'], 'R_next', ['PC'] + CONTROLS, nw_lag=NW_LAG, label='registered_controls')
    b_reg = base[f'registered_EW_lag{NW_LAG}_univariate']['beta_pct']
    b_pl = base['paper_like_EW_lag12_univariate']['beta_pct']
    repro = {'registered_EW_lag6': (b_reg, BASELINE['registered_EW_lag6_univariate']['beta_pct']),
             'paper_like_EW_lag12': (b_pl, BASELINE['paper_like_EW_lag12_univariate']['beta_pct'])}
    psay('0', f'基准复现状态 = {"ok" if all(abs(a - e) < 0.001 for a, e in repro.values()) else "CHECK"}：'
              f'登记样本 EW lag{NW_LAG} = {b_reg}%（登记 −0.5068）；'
              f'原文最近似 EW lag12 = {b_pl}%（登记 −0.4559）')
    OUT = {'artifact': OUT_PREFIX, 'generated_at': args.generated_at,
           'scope_note': ('2026-10-27 口径更正：本文件仅描述**项目自建 PC**（对手集合 = 同团全部成员）的负号成因排查，'
                          '**不是对原文 PC 的复现或证伪**。原文第四步 = 「任何与其相似度超过团内最小相似度的公司」'
                          '（字面含团外）；项目 = 同团成员（methodology_execution_plan.md:2290 已登记偏离）→ 二者不可比，'
                          '判定统一为「不可比/未复现（对手集合口径不同）」。文内「反号/负号」仅指项目 PC 的点估计方向。'),
           'pre_registered_criteria': {
               'baseline': BASELINE,
               'rule': ('元凶（充分）= 主读数层 PC 系数 ≥ 0 且 4 子层（EW/VW × 五分位/十分位 H−L、'
                        '单变量/加控制 FM）≥3 层同向翻转；部分贡献 = 向 0 移动 ≥50% 但仍为负；'
                        '排除 = 移动 <25% 或同号且 |t| 不降'),
               'attribution': '构造/实现 bug（我们的错）／口径差异（论文 vs 我们）／真实经济差异'},
           'inputs': {'panel': 'data/processed/m1_common_sample_ma_teacher_v2.parquet',
                      'pc_dir': 'data/processed/network_ma_teacher',
                      'pc_years': PC_YEARS,
                      'ycol': 'R_next（raw 小数，面板构造期已月度截面 [0.01,0.99] winsorize）',
                      'samples': {k: {'rows': int(len(v)), 'months': int(v['month'].nunique())}
                                  for k, v in S.items()}},
           'sign_convention_check': sign_check,
           'baseline_reproduction': base}

    if 'T1' in tests:
        psay('T1', '状态 = 开始（十分位组均值/中位数 + 组合层 vs FM 层方向核对）')
        res = {}
        for tag in ('registered', 'paper_like'):
            samp = S[tag]
            fm_beta = base[f'{tag}_EW_lag{12 if tag == "paper_like" else NW_LAG}_univariate']['beta_pct']
            for wtag, wcol in (('EW', None), ('VW', 'market_equity')):
                t = group_table(samp, 10, wcol=wcol)
                tm = group_median_table(samp, 10, wcol=wcol)
                s10 = hl_summary(t, 10)
                s10m = hl_summary(tm, 10)
                t5 = group_table(samp, 5, wcol=wcol)
                s5 = hl_summary(t5, 5)
                res[f'{tag}_decile_{wtag}'] = {
                    'group_mean_pct': s10['group_mean_pct'],
                    'group_median_of_monthly_median_pct': s10m.get('group_mean_pct'),
                    'HL_mean_pct': s10['HL_pct'], 'HL_mean_t_nw': s10['HL_t_nw'],
                    'HL_median_pct': s10m['HL_pct'], 'HL_median_t_nw': s10m['HL_t_nw'],
                    'n_months': s10['n_months'],
                    'monotonicity_mean': monotonicity(t, 10),
                    'quintile_HL_mean_pct': s5['HL_pct'], 'quintile_HL_mean_t_nw': s5['HL_t_nw'],
                }
            # 组规模与 PC 极值分布（平衡性诊断）
            d = samp[['month', 'PC', 'R_next']].dropna()
            g = d.groupby('month', group_keys=False)['PC'].transform(
                lambda s: rank_average_groups(s, 10))
            grp = pd.DataFrame({'grp': g, 'PC': d['PC']}).dropna()
            res[f'{tag}_decile_sizes'] = {
                f'D{i}': {'n_rows': int((grp['grp'] == i).sum()),
                          'share_pct': r4(100.0 * (grp['grp'] == i).mean()),
                          'pc_mean': r4(grp.loc[grp['grp'] == i, 'PC'].mean()),
                          'pc_min': r4(grp.loc[grp['grp'] == i, 'PC'].min()),
                          'pc_max': r4(grp.loc[grp['grp'] == i, 'PC'].max())}
                for i in range(1, 11)}
            res[f'{tag}_FM_vs_portfolio'] = {
                'FM_univariate_beta_pct': fm_beta,
                'decile_EW_HL_mean_pct': res[f'{tag}_decile_EW']['HL_mean_pct'],
                'quintile_EW_HL_mean_pct': res[f'{tag}_decile_EW']['quintile_HL_mean_pct'],
                'sign_consistent_FM_vs_portfolio': bool(
                    np.sign(res[f'{tag}_decile_EW']['HL_mean_pct']) == np.sign(fm_beta))}
        reg = res['registered_decile_EW']
        psay('T1', f'状态 = ok：登记样本 EW 十分位组均值 {reg["group_mean_pct"]}；'
                   f'H−L(mean) = {reg["HL_mean_pct"]}%（t = {reg["HL_mean_t_nw"]}）；'
                   f'H−L(median) = {reg["HL_median_pct"]}%（t = {reg["HL_median_t_nw"]}）；'
                   f'单调 = {reg["monotonicity_mean"]["monotone_increasing"]}/{reg["monotonicity_mean"]["monotone_decreasing"]}'
                   f'，相邻反转 {reg["monotonicity_mean"]["n_adjacent_sign_flips"]} 次')
        OUT['T1_direction_and_monotonicity'] = res

    if 'T2' in tests:
        psay('T2', '状态 = 开始（尾部敏感性：winsorize / 剔极端 / 组中位数差）')
        res = {}
        for tag in ('registered', 'paper_like'):
            samp = S[tag]
            ctrl = S[f'{tag}_ctrl']
            lag = 12 if tag == 'paper_like' else NW_LAG
            rec = {}
            # ①a 重施加 [0.01,0.99]（幂等核验；面板构造期已施加）
            s_w = samp.copy()
            q_lo = s_w.groupby('month')['R_next'].transform(lambda s: s.quantile(0.01))
            q_hi = s_w.groupby('month')['R_next'].transform(lambda s: s.quantile(0.99))
            s_w['R_w'] = s_w['R_next'].clip(lower=q_lo, upper=q_hi)
            n_changed = int((s_w['R_w'] - s_w['R_next']).abs().gt(1e-12).sum())
            rec['winsor_1_99'] = {
                'rows_changed': n_changed,
                'note': 'R_next 在面板构造期已按 registry 月度截面 [0.01,0.99] winsorize → 本项为幂等核验',
                'fm_univariate': fm_run(s_w.rename(columns={'R_w': 'R_w_y'}), 'R_w_y', ['PC'],
                                        nw_lag=lag),
                'fm_with_controls': (fm_run(s_w.assign(R_next=s_w['R_w']), 'R_next',
                                            ['PC'] + CONTROLS, nw_lag=lag)
                                     if tag == 'registered' else None),
                'decile_EW': hl_summary(group_table(s_w.assign(R_next=s_w['R_w']), 10), 10),
            }
            # ①b 更紧 winsor [0.025,0.975]（附加口径）
            s_t = samp.copy()
            lo2 = s_t.groupby('month')['R_next'].transform(lambda s: s.quantile(0.025))
            hi2 = s_t.groupby('month')['R_next'].transform(lambda s: s.quantile(0.975))
            s_t['R_t'] = s_t['R_next'].clip(lower=lo2, upper=hi2)
            rec['winsor_2_5_97_5'] = {
                'fm_univariate': fm_run(s_t[['month', 'PC', 'R_t', 'market_equity']], 'R_t', ['PC'],
                                        nw_lag=lag),
                'decile_EW': hl_summary(group_table(s_t.assign(R_next=s_t['R_t']), 10), 10),
            }
            # ①c 未 winsor 的原始月收益 next_R（附加口径，R_next 之外的真实尾部）
            s_raw = samp.dropna(subset=['next_R']).copy()
            rec['unwinsorized_next_R'] = {
                'n_rows': int(len(s_raw)),
                'sd': r4(s_raw['next_R'].std()), 'mean': r4(s_raw['next_R'].mean()),
                'fm_univariate': fm_run(s_raw, 'next_R', ['PC'], nw_lag=lag),
                'fm_with_controls': (fm_run(S[f'{tag}_ctrl'].dropna(subset=['next_R']), 'next_R',
                                            ['PC'] + CONTROLS, nw_lag=lag)
                                     if tag == 'registered' else None),
                'decile_EW': hl_summary(group_table(s_raw.assign(R_next=s_raw['next_R']), 10), 10),
            }
            # ② 剔除 |月收益| > 50%
            for ycol, lbl in (('R_next', 'R_next'), ('next_R', 'next_R_raw')):
                s_e = samp.dropna(subset=[ycol]).copy()
                s_e = s_e[s_e[ycol].abs() <= 0.50]
                rec[f'drop_abs_gt_50pct_on_{lbl}'] = {
                    'n_rows': int(len(s_e)),
                    'rows_dropped': int(len(samp.dropna(subset=[ycol])) - len(s_e)),
                    'fm_univariate': fm_run(s_e, ycol, ['PC'], nw_lag=lag),
                    'fm_with_controls': (fm_run(
                        S[f'{tag}_ctrl'].dropna(subset=[ycol]).pipe(
                            lambda x: x[x[ycol].abs() <= 0.50]), ycol, ['PC'] + CONTROLS, nw_lag=lag)
                        if tag == 'registered' else None),
                    'decile_EW': hl_summary(group_table(s_e.assign(R_next=s_e[ycol]), 10), 10),
                }
            # ③ 组中位数差（替代均值差）
            rec['group_median_diff'] = {
                'decile_EW': hl_median_summary(samp, 10, wcol=None),
                'decile_VW': hl_median_summary(samp, 10, wcol='market_equity'),
                'quintile_EW': hl_median_summary(samp, 5, wcol=None),
            }
            res[tag] = rec
            main = rec['drop_abs_gt_50pct_on_R_next']['fm_univariate']
            psay('T2', f'{tag} 状态 = ok：剔|R|>50% 单变量 β = {main["beta_pct"]}%（t = {main["t_nw"]}）；'
                       f'组中位数差 H−L = {rec["group_median_diff"]["decile_EW"]["HL_pct"]}%'
                       f'（t = {rec["group_median_diff"]["decile_EW"]["HL_t_nw"]}）')
            base_beta = base[f'{tag}_EW_lag{12 if tag == "paper_like" else NW_LAG}_univariate']['beta_pct']
            res[tag]['verdict'] = judge(base_beta, main['beta_pct'], new_t=main['t_nw'])
        OUT['T2_tail_sensitivity'] = res

    if 'T3' in tests:
        psay('T3', '状态 = 开始（规模混淆：层内排序 / 剔最小 30% / VW 权重集中度）')
        res = {}
        for tag in ('registered', 'paper_like'):
            samp = S[tag]
            lag = 12 if tag == 'paper_like' else NW_LAG
            rec = {}
            # ① size 三分位内各自排序（层内十分位）
            d = samp[['month', 'PC', 'R_next', 'size', 'market_equity']].dropna(subset=['size', 'PC', 'R_next']).copy()
            d['lay'] = d.groupby('month', group_keys=False)['size'].transform(
                lambda s: rank_average_groups(s, 3))
            layers = {}
            for L in (1, 2, 3):
                sub = d[d['lay'] == L]
                t = group_table(sub, 10)
                s10 = hl_summary(t, 10)
                layers[f'tercile{L}' + ('_small' if L == 1 else ('_mid' if L == 2 else '_large'))] = {
                    'n_rows': int(len(sub)), 'HL_pct': s10['HL_pct'], 'HL_t_nw': s10['HL_t_nw'],
                    'n_months': s10['n_months'], 'rank_IC': rank_ic(sub, lag=lag)}
            rec['size_tercile_within_deciles_EW'] = layers
            rec['size_tercile_within_FM'] = {}
            for L in (1, 2, 3):
                sub = d[d['lay'] == L]
                rec['size_tercile_within_FM'][f'tercile{L}'] = fm_run(sub, 'R_next', ['PC'], nw_lag=lag)
            # ② 剔除当月市值最小 30%
            cut = samp.groupby('month')['market_equity'].transform(lambda s: s.quantile(0.30))
            s_big = samp[samp['market_equity'] > cut].copy()
            s_big_ctrl = S[f'{tag}_ctrl']
            cut_c = s_big_ctrl.groupby('month')['market_equity'].transform(lambda s: s.quantile(0.30))
            s_big_ctrl = s_big_ctrl[s_big_ctrl['market_equity'] > cut_c]
            rec['drop_smallest_30pct_market_equity'] = {
                'n_rows': int(len(s_big)), 'rows_dropped': int(len(samp) - len(s_big)),
                'fm_univariate': fm_run(s_big, 'R_next', ['PC'], nw_lag=lag),
                'fm_with_controls': (fm_run(s_big_ctrl, 'R_next', ['PC'] + CONTROLS, nw_lag=lag)
                                     if tag == 'registered' else None),
                'decile_EW': hl_summary(group_table(s_big, 10), 10),
                'decile_VW': hl_summary(group_table(s_big, 10, wcol='market_equity'), 10),
                'quintile_EW': hl_summary(group_table(s_big, 5), 5),
            }
            # ②b 附加：剔最小 50%（再试）
            cut5 = samp.groupby('month')['market_equity'].transform(lambda s: s.quantile(0.50))
            s_big5 = samp[samp['market_equity'] > cut5].copy()
            rec['drop_smallest_50pct_market_equity'] = {
                'n_rows': int(len(s_big5)),
                'fm_univariate': fm_run(s_big5, 'R_next', ['PC'], nw_lag=lag),
                'decile_EW': hl_summary(group_table(s_big5, 10), 10),
            }
            # ③ VW 权重集中度（十分位 VW 组内最大权重占比 / 权重 HHI）
            dd = samp[['month', 'PC', 'R_next', 'market_equity']].dropna().copy()
            dd = dd[dd['market_equity'] > 0]
            dd['grp'] = dd.groupby('month', group_keys=False)['PC'].transform(
                lambda s: rank_average_groups(s, 10))
            dd = dd[dd['grp'].notna()]
            rows = []
            for (mo, g), sub in dd.groupby(['month', 'grp']):
                w = sub['market_equity'].to_numpy(float)
                ws = w / w.sum()
                rows.append({'month': mo, 'grp': g, 'max_share': float(ws.max()),
                             'hhi': float((ws ** 2).sum()), 'n': int(len(sub))})
            conc = pd.DataFrame(rows)
            rec['VW_weight_concentration'] = {
                'mean_max_share_by_decile': {f'D{int(g)}': r4(sub['max_share'].mean())
                                             for g, sub in conc.groupby('grp')},
                'mean_hhi_by_decile': {f'D{int(g)}': r4(sub['hhi'].mean())
                                       for g, sub in conc.groupby('grp')},
                'max_max_share': r4(conc['max_share'].max()),
                'note': ('VW 权重 = 总市值（registered_deviation：流通市值日频序列不可得）')
            }
            res[tag] = rec
            v = judge(base[f'{tag}_EW_lag{12 if tag == "paper_like" else NW_LAG}_univariate']['beta_pct'],
                      rec['drop_smallest_30pct_market_equity']['fm_univariate']['beta_pct'],
                      new_t=rec['drop_smallest_30pct_market_equity']['fm_univariate']['t_nw'])
            rec['verdict'] = v
            psay('T3', f'{tag} 状态 = ok：size 三层 H−L = '
                       f'{ {k: v2["HL_pct"] for k, v2 in layers.items()} }；'
                       f'剔最小 30% 单变量 β = {rec["drop_smallest_30pct_market_equity"]["fm_univariate"]["beta_pct"]}%'
                       f'（t = {rec["drop_smallest_30pct_market_equity"]["fm_univariate"]["t_nw"]}）；'
                       f'VW 最大权重占比 max = {rec["VW_weight_concentration"]["max_max_share"]}')
        OUT['T3_size_confound'] = res

    if 'T4' in tests:
        psay('T4', '状态 = 开始（时点/滞后 extra = 0/12/24 月）')
        # month -> text_year 映射（全局，来自面板 network_version）
        mp = (m[['month', 'text_year']].dropna().astype({'text_year': 'int64'})
              .drop_duplicates())
        dup = int(mp.groupby('month')['text_year'].nunique().gt(1).sum())
        month_map = dict(zip(mp['month'], mp['text_year']))
        res = {'month_to_text_year_map_duplicate_months': dup,
               'available_text_years_in_pc_files': PC_YEARS}
        pc_all = m[['focal_id', 'text_year', 'PC']].dropna(subset=['PC']).copy()
        pc_all['text_year'] = pc_all['text_year'].astype('int64')
        pc_all = pc_all.drop_duplicates(['focal_id', 'text_year'])
        base_cols = ['focal_id', 'month', 'R_next', 'market_equity', 'm1_in_fm_sample',
                     'size', 'bm', 'mom', 'rev', 'illiq']
        panel = m[base_cols].copy()
        for extra in (0, 12, 24):
            src = panel['month'].map(lambda x: str(pd.Period(x, freq='M') - extra))
            panel['ty_eff'] = src.map(month_map)
            lg = panel.merge(
                pc_all.rename(columns={'text_year': 'ty_eff', 'PC': 'PC_lag'}),
                on=['focal_id', 'ty_eff'], how='left').rename(columns={'PC_lag': 'PC'})
            pc_ok = lg['PC'].notna() & lg['R_next'].notna()
            s_reg = lg[pc_ok & lg['m1_in_fm_sample'].astype(bool)]
            s_pl = lg[pc_ok]
            lag = NW_LAG
            res[f'extra_{extra}'] = {
                'registered': {
                    'n_rows': int(len(s_reg)), 'n_months': int(s_reg['month'].nunique()),
                    'first_month': (s_reg['month'].min() if len(s_reg) else None),
                    'last_month': (s_reg['month'].max() if len(s_reg) else None),
                    'fm_univariate_EW': fm_run(s_reg, 'R_next', ['PC'], nw_lag=lag),
                    'fm_univariate_VW': fm_run(s_reg, 'R_next', ['PC'], weight_col='market_equity',
                                               nw_lag=lag),
                    'fm_with_controls': fm_run(s_reg.dropna(subset=CONTROLS), 'R_next',
                                               ['PC'] + CONTROLS, nw_lag=lag),
                    'decile_EW': hl_summary(group_table(s_reg, 10), 10),
                    'quintile_EW': hl_summary(group_table(s_reg, 5), 5),
                    'rank_IC': rank_ic(s_reg, lag=lag)},
                'paper_like': {
                    'n_rows': int(len(s_pl)), 'n_months': int(s_pl['month'].nunique()),
                    'first_month': (s_pl['month'].min() if len(s_pl) else None),
                    'last_month': (s_pl['month'].max() if len(s_pl) else None),
                    'fm_univariate_EW': fm_run(s_pl, 'R_next', ['PC'], nw_lag=12),
                    'decile_EW': hl_summary(group_table(s_pl, 10), 10),
                    'rank_IC': rank_ic(s_pl, lag=12)},
            }
            r = res[f'extra_{extra}']['registered']
            psay('T4', f'extra = {extra} 状态 = ok：登记样本 {r["n_rows"]:,} 行 / {r["n_months"]} 月'
                       f'（{r["first_month"]}..{r["last_month"]}）；单变量 β = '
                       f'{r["fm_univariate_EW"]["beta_pct"]}%（t = {r["fm_univariate_EW"]["t_nw"]}）；'
                       f'十分位 H−L = {r["decile_EW"]["HL_pct"]}%（t = {r["decile_EW"]["HL_t_nw"]}）')
            # 子层翻转计数（对 extra>0 相对 extra=0）
            lv = [r['fm_univariate_EW']['beta_pct'], r['fm_with_controls']['beta_pct'],
                  r['decile_EW']['HL_pct'], r['quintile_EW']['HL_pct']]
            res[f'extra_{extra}']['registered']['sub_layer_betas_pct'] = lv
        b0 = res['extra_0']['registered']['fm_univariate_EW']['beta_pct']
        for extra in (12, 24):
            r = res[f'extra_{extra}']['registered']
            flips = sum(1 for x in r['sub_layer_betas_pct'] if x is not None and x >= 0)
            r['verdict'] = judge(b0, r['fm_univariate_EW']['beta_pct'],
                                 new_t=r['fm_univariate_EW']['t_nw'], layer_flips=flips,
                                 n_layers=4)
        OUT['T4_timing_lag'] = res

    if 'T5' in tests:
        psay('T5', '状态 = 开始（NW 滞后 / 常数项 vs 去均值 / PC 层级加权）')
        res = {}
        for tag in ('registered', 'paper_like'):
            samp = S[tag]
            rec = {'NW_lags': {}, 'demeaned': {}, 'pc_level_weighted': {}}
            for lag in (6, 12, 18):
                rec['NW_lags'][f'lag{lag}'] = fm_run(samp, 'R_next', ['PC'], nw_lag=lag)
            rec['demeaned']['univariate'] = fm_demeaned(samp, 'R_next')
            rec['demeaned']['with_controls'] = None
            # 去均值 + 控制（全 X 去月内均值，无常数项）
            ct = S[f'{tag}_ctrl']
            coefs = []
            for _, sub in ct.groupby('month', sort=True):
                sub = sub.dropna(subset=['R_next', 'PC'] + CONTROLS)
                if len(sub) < MIN_N:
                    continue
                X = sub[['PC'] + CONTROLS].to_numpy(float)
                y = sub['R_next'].to_numpy(float)
                Xd = X - X.mean(axis=0)
                yd = y - y.mean()
                try:
                    b = np.linalg.lstsq(Xd, yd, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue
                coefs.append(float(b[0]))
            nw = newey_west(coefs, lag=NW_LAG)
            rec['demeaned']['with_controls'] = {'label': 'demeaned_with_controls',
                                               'beta_pct': pct(nw['mean']), 't_nw': r4(nw['t_nw']),
                                               'n_months_used': nw['T']}
            # PC 层级加权（同月同一 PC 值 = 同一团 → 权重 = 1/该 PC 值的公司数）
            sw = samp[['month', 'PC', 'R_next', 'market_equity']].dropna(subset=['PC', 'R_next']).copy()
            sw['_lvl'] = sw['PC'].round(8)
            sw['w_pclevel'] = 1.0 / sw.groupby(['month', '_lvl'])['PC'].transform('size')
            rec['pc_level_weighted']['univariate'] = fm_run(sw, 'R_next', ['PC'],
                                                            weight_col='w_pclevel', nw_lag=NW_LAG)
            rec['pc_level_weighted']['n_distinct_pc_levels_per_month_mean'] = r4(
                sw.groupby('month')['_lvl'].nunique().mean())
            swc = S[f'{tag}_ctrl'].copy()
            swc['_lvl'] = swc['PC'].round(8)
            swc['w_pclevel'] = 1.0 / swc.groupby(['month', '_lvl'])['PC'].transform('size')
            if tag == 'registered':
                rec['pc_level_weighted']['with_controls'] = fm_run(swc, 'R_next',
                                                                   ['PC'] + CONTROLS,
                                                                   weight_col='w_pclevel',
                                                                   nw_lag=NW_LAG)
            else:
                rec['pc_level_weighted']['with_controls'] = None
            res[tag] = rec
            lags = {k: (v['beta_pct'], v['t_nw']) for k, v in rec['NW_lags'].items()}
            psay('T5', f'{tag} 状态 = ok：NW 6/12/18 → {lags}；'
                       f'去均值 β = {rec["demeaned"]["univariate"]["beta_pct"]}'
                       f'（t = {rec["demeaned"]["univariate"]["t_nw"]}）；'
                       f'PC 层级加权 β = {rec["pc_level_weighted"]["univariate"]["beta_pct"]}'
                       f'（t = {rec["pc_level_weighted"]["univariate"]["t_nw"]}）')
        OUT['T5_fm_implementation'] = res

    if 'T5B' in tests:
        psay('T5B', '状态 = 开始（团层聚合：cluster_id 口径交叉核验「PC 值 = 团」代理 + 子层翻转计数）')
        cl_frames = []
        for y in PC_YEARS:
            c = pd.read_csv(CLUSTER_DIR / f'{y}_clusters.csv', dtype=str)
            c.columns = ['focal_id', 'cluster_id']
            c['text_year'] = y
            cl_frames.append(c)
        cl = pd.concat(cl_frames, ignore_index=True).drop_duplicates(['focal_id', 'text_year'])
        m2 = m.merge(cl, on=['focal_id', 'text_year'], how='left')
        chk = m2.dropna(subset=['PC', 'cluster_id']).groupby(['text_year', 'cluster_id'])['PC'].nunique()
        res = {'proxy_check': {
                   'n_cluster_years': int(len(chk)), 'n_with_multiple_PC': int((chk > 1).sum()),
                   'note': 'PC 值代理（同月同一 PC 值 = 同团）核验：团内 PC 唯一性'},
               'n_distinct_clusters_per_month_mean': None}
        for tag in ('registered', 'paper_like'):
            samp = m2[m2['PC'].notna() & m2['R_next'].notna()]
            if tag == 'registered':
                samp = samp[samp['m1_in_fm_sample'].astype(bool)]
            samp = samp.dropna(subset=['cluster_id'])
            agg = samp.groupby(['month', 'cluster_id'], sort=True).agg(
                y=('R_next', 'mean'), PC=('PC', 'mean'), n=('R_next', 'size'),
                market_equity=('market_equity', 'sum')).reset_index()
            agg['w_members'] = agg['n'].astype(float)
            agg2 = agg.assign(R_next=agg['y'])
            d10 = agg2.copy()
            d10['grp'] = (d10.groupby('month', group_keys=False)['PC']
                          .transform(lambda s: rank_average_groups(s, 10)))
            lag = 12 if tag == 'paper_like' else NW_LAG
            rec = {
                'n_cluster_months': int(len(agg)),
                'mean_clusters_per_month': r4(agg.groupby('month').size().mean()),
                'corr_PC_log_members': r4(agg.assign(lm=np.log1p(agg['n']))[['PC', 'lm']].corr().iloc[0, 1]),
                'fm_cluster_equal_weight': fm_run(agg, 'y', ['PC'], nw_lag=lag),
                'fm_cluster_member_weighted': fm_run(agg, 'y', ['PC'], weight_col='w_members', nw_lag=lag),
                'decile_cluster_equal_weight': hl_summary(group_table(agg2, 10), 10, lag=lag),
                'decile_member_weighted': hl_summary(group_table(agg2, 10, wcol='w_members'), 10, lag=lag),
                'quintile_cluster_equal_weight': hl_summary(group_table(agg2, 5), 5, lag=lag),
                'quintile_member_weighted': hl_summary(group_table(agg2, 5, wcol='w_members'), 5, lag=lag),
                'mean_members_by_PC_decile': {f'D{i}': r4(d10.loc[d10['grp'] == i, 'n'].mean())
                                              for i in range(1, 11)},
                'median_members_by_PC_decile': {f'D{i}': r4(d10.loc[d10['grp'] == i, 'n'].median())
                                                for i in range(1, 11)},
            }
            # 子层翻转计数（团层等权口径 vs 公司层等权基准）
            b0 = base[f'{tag}_EW_lag{12 if tag == "paper_like" else NW_LAG}_univariate']['beta_pct']
            subs = [rec['fm_cluster_equal_weight']['beta_pct'],
                    rec['decile_cluster_equal_weight']['HL_pct'],
                    rec['quintile_cluster_equal_weight']['HL_pct']]
            rec['sub_layer_betas_pct'] = [b0] + subs
            rec['verdict'] = judge(b0, rec['fm_cluster_equal_weight']['beta_pct'],
                                   new_t=rec['fm_cluster_equal_weight']['t_nw'],
                                   layer_flips=sum(1 for x in subs if x is not None and x >= 0),
                                   n_layers=len(subs))
            res[tag] = rec
            psay('T5B', f'{tag} 状态 = ok：团层等权 FM β = {rec["fm_cluster_equal_weight"]["beta_pct"]}%'
                        f'（t = {rec["fm_cluster_equal_weight"]["t_nw"]}）；成员数加权 β = '
                        f'{rec["fm_cluster_member_weighted"]["beta_pct"]}%'
                        f'（t = {rec["fm_cluster_member_weighted"]["t_nw"]}）；'
                        f'团层等权十分位 H−L = {rec["decile_cluster_equal_weight"]["HL_pct"]}%'
                        f'（t = {rec["decile_cluster_equal_weight"]["HL_t_nw"]}）；'
                        f'每团成员数 D1→D10 = {rec["median_members_by_PC_decile"]["D1"]}→'
                        f'{rec["median_members_by_PC_decile"]["D10"]}（中位）')
        OUT['T5B_cluster_level'] = res

    if 'T6' in tests:
        psay('T6', '状态 = 开始（逐月 rank IC + 十组均值/中位数单调性诊断）')
        res = {}
        for tag in ('registered', 'paper_like'):
            samp = S[tag]
            lag = 12 if tag == 'paper_like' else NW_LAG
            t10 = group_table(samp, 10)
            tm10 = group_median_table(samp, 10)
            d = samp[['month', 'PC', 'R_next']].dropna().copy()
            d['grp'] = d.groupby('month', group_keys=False)['PC'].transform(
                lambda s: rank_average_groups(s, 10))
            # 逐月组序 vs 组均值的 Spearman（月内单调性）
            per_month_rho = []
            for _, sub in d.groupby('month'):
                gm = sub.groupby('grp')['R_next'].mean()
                if len(gm) >= 8:
                    per_month_rho.append(float(gm.reset_index().corr(
                        method='spearman').iloc[0, 1]))
            nw_rho = newey_west(per_month_rho, lag=lag)
            res[tag] = {
                'rank_IC': rank_ic(samp, lag=lag),
                'decile_mean_table_pct': {f'D{g}': pct(t10[f'g{g}'].mean()) for g in range(1, 11)},
                'decile_median_table_pct': {f'D{g}': pct(tm10[f'g{g}'].median()) for g in range(1, 11)},
                'monotonicity_mean': monotonicity(t10, 10),
                'monotonicity_median': monotonicity(tm10, 10),
                'within_month_spearman_group_vs_mean': {
                    'n_months': nw_rho['T'], 'mean_rho': r4(nw_rho['mean']),
                    't_nw': r4(nw_rho['t_nw'])},
                'HL_mean_pct': hl_summary(t10, 10)['HL_pct'],
                'HL_median_pct': hl_summary(tm10, 10)['HL_pct'],
                'tail_share_of_HL': {
                    'D10_minus_D9_pct': pct(t10['g10'].mean() - t10['g9'].mean()),
                    'D2_minus_D1_pct': pct(t10['g2'].mean() - t10['g1'].mean()),
                }}
            ic = res[tag]['rank_IC']
            psay('T6', f'{tag} 状态 = ok：逐月 rank IC 均值 = {ic["ic_mean"]}（t_NW = {ic["ic_t_nw"]}，'
                       f'{ic["n_months"]} 月）；组均值 Spearman(组序) = '
                       f'{res[tag]["monotonicity_mean"]["spearman_group_index_vs_group_mean"]}；'
                       f'组中位数 Spearman = {res[tag]["monotonicity_median"]["spearman_group_index_vs_group_mean"]}')
        OUT['T6_rank_ic'] = res

    # ---------------- 汇总表
    rows = []
    reg_base = base[f'registered_EW_lag{NW_LAG}_univariate']['beta_pct']

    def add(tid, spec, beta, t, verdict, note):
        rows.append({'test': tid, 'spec': spec, 'beta_pct': beta, 't_nw': t,
                     'verdict': verdict, 'note': note})

    add('基准', 'MD&A 登记样本 EW lag%d 单变量' % NW_LAG, reg_base,
        base[f'registered_EW_lag{NW_LAG}_univariate']['t_nw'], '基准读数', '主读数层')
    add('基准', 'MD&A 原文最近似样本 EW lag12', b_pl,
        base['paper_like_EW_lag12_univariate']['t_nw'], '基准读数', '原文最近似口径')
    if 'T2' in tests:
        for tag in ('registered', 'paper_like'):
            r = OUT['T2_tail_sensitivity'][tag]
            add('T2', f'{tag} 重 winsorize[1%,99%] 单变量',
                r['winsor_1_99']['fm_univariate']['beta_pct'],
                r['winsor_1_99']['fm_univariate']['t_nw'], r.get('verdict', {}).get('verdict'),
                f'改行数 {r["winsor_1_99"]["rows_changed"]}（幂等核验）')
            add('T2', f'{tag} 未 winsor next_R 单变量',
                r['unwinsorized_next_R']['fm_univariate']['beta_pct'],
                r['unwinsorized_next_R']['fm_univariate']['t_nw'], '—', '附加口径')
            add('T2', f'{tag} 剔 |R|>50% 单变量',
                r['drop_abs_gt_50pct_on_R_next']['fm_univariate']['beta_pct'],
                r['drop_abs_gt_50pct_on_R_next']['fm_univariate']['t_nw'],
                r['verdict']['verdict'], f'剔 {r["drop_abs_gt_50pct_on_R_next"]["rows_dropped"]} 行')
            add('T2', f'{tag} 组中位数差 H−L（十分位 EW）',
                r['group_median_diff']['decile_EW']['HL_pct'],
                r['group_median_diff']['decile_EW']['HL_t_nw'], '—', '中位数替代均值')
    if 'T3' in tests:
        for tag in ('registered', 'paper_like'):
            r = OUT['T3_size_confound'][tag]
            add('T3', f'{tag} 剔最小 30% 市值 单变量',
                r['drop_smallest_30pct_market_equity']['fm_univariate']['beta_pct'],
                r['drop_smallest_30pct_market_equity']['fm_univariate']['t_nw'],
                r['verdict']['verdict'],
                f'剔 {r["drop_smallest_30pct_market_equity"]["rows_dropped"]} 行')
    if 'T4' in tests:
        for extra in (0, 12, 24):
            r = OUT['T4_timing_lag'][f'extra_{extra}']['registered']
            add('T4', f'登记样本 滞后 extra={extra} 月 单变量',
                r['fm_univariate_EW']['beta_pct'], r['fm_univariate_EW']['t_nw'],
                r.get('verdict', {}).get('verdict', '—'),
                f'{r["n_months"]} 月（{r["first_month"]}..{r["last_month"]}）')
    if 'T5' in tests:
        r = OUT['T5_fm_implementation']['registered']
        for lag in (6, 12, 18):
            add('T5', f'登记样本 NW lag={lag} 单变量', r['NW_lags'][f'lag{lag}']['beta_pct'],
                r['NW_lags'][f'lag{lag}']['t_nw'], '—', '只改 t')
        add('T5', '登记样本 去均值（无常数项）', r['demeaned']['univariate']['beta_pct'],
            r['demeaned']['univariate']['t_nw'], '—', '等价性核验')
        add('T5', '登记样本 PC 层级加权', r['pc_level_weighted']['univariate']['beta_pct'],
            r['pc_level_weighted']['univariate']['t_nw'], '—', '同月同 PC 值内等权')
    if 'T5B' in tests:
        for tag in ('registered', 'paper_like'):
            r = OUT['T5B_cluster_level'][tag]
            add('T5B', f'{tag} 团层等权 FM 单变量', r['fm_cluster_equal_weight']['beta_pct'],
                r['fm_cluster_equal_weight']['t_nw'], r['verdict']['verdict'],
                f'每团等权（≈ PC 层级加权）；团数/月均值 {r["mean_clusters_per_month"]}')
            add('T5B', f'{tag} 团层等权十分位 H−L',
                r['decile_cluster_equal_weight']['HL_pct'],
                r['decile_cluster_equal_weight']['HL_t_nw'], '—', '组 = 团，组内再等权')
    if 'T6' in tests:
        for tag in ('registered', 'paper_like'):
            r = OUT['T6_rank_ic'][tag]
            add('T6', f'{tag} 逐月 rank IC', r['rank_IC']['ic_mean'], r['rank_IC']['ic_t_nw'],
                '—', f'{r["rank_IC"]["n_months"]} 月')
    OUT['summary_table'] = rows

    # ---------------- 逐检验判定 + 一句话结论
    tv = {}
    if 'T1' in tests:
        r = OUT['T1_direction_and_monotonicity']
        re_, rv = r['registered_decile_EW'], r['registered_decile_VW']
        tv['T1'] = {
            'hypothesis': 'H14/H10 排序方向 bug 与单调性核验',
            'data_spec': ('登记样本 120,352 行/68 月 + 原文最近似样本 211,405 行/68 月；逐月 PC 截面 '
                          'rank_average 十分位（D1 = PC 最低，H = D10，H−L = D10−D1）；EW/VW × 均值/中位数'),
            'result': (f'D1..D10 组均值 %（登记 EW）={re_["group_mean_pct"]}；'
                       f'H−L(mean) = {re_["HL_mean_pct"]}%（t = {re_["HL_mean_t_nw"]}）；'
                       f'H−L(median) = {re_["HL_median_pct"]}%（t = {re_["HL_median_t_nw"]}）；'
                       f'相邻组符号反转 {re_["monotonicity_mean"]["n_adjacent_sign_flips"]} 次；'
                       f'Spearman(组序, 组均值) = {re_["monotonicity_mean"]["spearman_group_index_vs_group_mean"]}；'
                       f'VW 同向（H−L {rv["HL_mean_pct"]}%，t = {rv["HL_mean_t_nw"]}）'),
            'changed_conclusion': False,
            'explains_negative_sign': 'NO',
            'verdict': '不改变结论（排除 H14 方向 bug）；同时登记「单调性不成立」（支持 H06 非线性/尾部解读）',
            'basis': ('组合层（EW/VW）与 FM 层符号一致（均负）→ 不存在整体反号的方向错位；'
                      '但 10 组均值/中位数均非单调（5 次相邻符号反转），说明负号不是稳定单调关系'),
        }
    if 'T2' in tests:
        r2 = OUT['T2_tail_sensitivity']
        reg, pl = r2['registered'], r2['paper_like']
        tv['T2'] = {
            'hypothesis': 'H06 尾部/极值伪影（winsorize / 剔极端 / 组中位数差）',
            'data_spec': ('月度收益 [0.01,0.99] 再 winsorize（面板构造期已施加 → 幂等核验）+ 附加 [0.025,0.975] '
                          '与未 winsor 的 next_R；剔 |月收益| > 50%；组中位数差替代均值差'),
            'result': (f'重 winsorize[1%,99%]：登记 {reg["winsor_1_99"]["fm_univariate"]["beta_pct"]}%'
                       f'（t = {reg["winsor_1_99"]["fm_univariate"]["t_nw"]}，改行 {reg["winsor_1_99"]["rows_changed"]}）、'
                       f'原文最近似 {pl["winsor_1_99"]["fm_univariate"]["beta_pct"]}%'
                       f'（t = {pl["winsor_1_99"]["fm_univariate"]["t_nw"]}）；'
                       f'未 winsor next_R：{reg["unwinsorized_next_R"]["fm_univariate"]["beta_pct"]}% / '
                       f'{pl["unwinsorized_next_R"]["fm_univariate"]["beta_pct"]}%（更负）；'
                       f'剔 |R|>50%（按 next_R，剔 {reg["drop_abs_gt_50pct_on_next_R_raw"]["rows_dropped"]} / '
                       f'{pl["drop_abs_gt_50pct_on_next_R_raw"]["rows_dropped"]} 行）：'
                       f'{reg["drop_abs_gt_50pct_on_next_R_raw"]["fm_univariate"]["beta_pct"]}%'
                       f'（t = {reg["drop_abs_gt_50pct_on_next_R_raw"]["fm_univariate"]["t_nw"]}）/ '
                       f'{pl["drop_abs_gt_50pct_on_next_R_raw"]["fm_univariate"]["beta_pct"]}%'
                       f'（t = {pl["drop_abs_gt_50pct_on_next_R_raw"]["fm_univariate"]["t_nw"]}）；'
                       f'组中位数差（十分位 EW）{reg["group_median_diff"]["decile_EW"]["HL_pct"]}%'
                       f'（t = {reg["group_median_diff"]["decile_EW"]["HL_t_nw"]}）/ '
                       f'{pl["group_median_diff"]["decile_EW"]["HL_pct"]}%'
                       f'（t = {pl["group_median_diff"]["decile_EW"]["HL_t_nw"]}）'),
            'changed_conclusion': False,
            'explains_negative_sign': 'NO（缩幅最大 = 组中位数差 76%，但仍为负且不显著）',
            'verdict': '不改变结论（排除，H06 尾部极值为反号主因不成立）；登记「均值口径放大约 2–3 倍」的尾部成分',
            'basis': ('逐项移动量：再 winsorize 2.7%/0.1%、未 winsor −12.6%/+0.5%、剔 |R|>50% 1.6%/5.6% '
                      '—— 全部 < 25% 门槛；组中位数差缩幅 76%（向 0）但符号仍负、|t| ≤ 1.02'),
        }
    if 'T3' in tests:
        r3 = OUT['T3_size_confound']
        reg, pl = r3['registered'], r3['paper_like']
        tv['T3'] = {
            'hypothesis': 'H06B 规模/风格混淆（小市值驱动负号）+ VW 权重口径',
            'data_spec': ('逐月 size 三分位内层内十分位排序（层内 H−L）；剔当月市值最小 30%（与 50%）后重跑 '
                          'FM 与组合层；十分位 VW 组合的权重集中度诊断'),
            'result': (f'层内 H−L（小/中/大）：登记 {reg["size_tercile_within_deciles_EW"]["tercile1_small"]["HL_pct"]} / '
                       f'{reg["size_tercile_within_deciles_EW"]["tercile2_mid"]["HL_pct"]} / '
                       f'{reg["size_tercile_within_deciles_EW"]["tercile3_large"]["HL_pct"]}；'
                       f'原文最近似 {pl["size_tercile_within_deciles_EW"]["tercile1_small"]["HL_pct"]} / '
                       f'{pl["size_tercile_within_deciles_EW"]["tercile2_mid"]["HL_pct"]} / '
                       f'{pl["size_tercile_within_deciles_EW"]["tercile3_large"]["HL_pct"]}（**最大层最负**）；'
                       f'剔最小 30%：登记 {reg["drop_smallest_30pct_market_equity"]["fm_univariate"]["beta_pct"]}%'
                       f'（t = {reg["drop_smallest_30pct_market_equity"]["fm_univariate"]["t_nw"]}，移动 1.8%）、'
                       f'原文最近似 {pl["drop_smallest_30pct_market_equity"]["fm_univariate"]["beta_pct"]}%'
                       f'（t = {pl["drop_smallest_30pct_market_equity"]["fm_univariate"]["t_nw"]}，移动 51.6%）；'
                       f'VW 最大权重占比（D1/D10 均值）= '
                       f'{reg["VW_weight_concentration"]["mean_max_share_by_decile"]["D1"]} / '
                       f'{reg["VW_weight_concentration"]["mean_max_share_by_decile"]["D10"]}'),
            'changed_conclusion': '部分（原文最近似样本剔除最小 30% 后向 0 移动 51.6%，但不翻正；登记样本无变化）',
            'explains_negative_sign': 'NO',
            'verdict': ('不改变结论（排除"小市值驱动"）：负号不在最小规模层，最大规模层 H−L 最负'
                        '（登记 −0.596%）且层内 IC 最负'),
            'basis': ('若负号由小市值风格承载，应只在最小层显著、剔除后归零 → 实测相反（剔 30% 后登记样本 '
                      '−0.516%，|t| 上升；剔 50% 为 −0.651%）；VW 权重集中度低（最大单股占比 ≤ 0.34），'
                      '排除"单股/少数权重股主导"'),
        }
    if 'T4' in tests:
        r4d = OUT['T4_timing_lag']
        e0, e12, e24 = (r4d['extra_0']['registered'], r4d['extra_12']['registered'],
                        r4d['extra_24']['registered'])
        tv['T4'] = {
            'hypothesis': 'H01 PC 时点/滞后（我们比原文"新"约 12 个月）',
            'data_spec': ('extra ∈ {0, 12, 24} 月：extra=0 = 我方 PIT 口径（FY(t−1)）；extra=12 = 原文 '
                          '`PC_{i,t−12}` 的「系列平移」读法（FY(t−2)）；extra=24 附加；单变量 FM + 十分位 H−L + rank IC'),
            'result': (f'extra=0：β = {e0["fm_univariate_EW"]["beta_pct"]}%（t = {e0["fm_univariate_EW"]["t_nw"]}，'
                       f'{e0["n_months"]} 月）；extra=12：β = {e12["fm_univariate_EW"]["beta_pct"]}%'
                       f'（t = {e12["fm_univariate_EW"]["t_nw"]}，{e12["n_months"]} 月，十分位 H−L '
                       f'{e12["decile_EW"]["HL_pct"]}%）；extra=24：β = {e24["fm_univariate_EW"]["beta_pct"]}%'
                       f'（t = {e24["fm_univariate_EW"]["t_nw"]}，{e24["n_months"]} 月）；'
                       f'IC：{e0["rank_IC"]["ic_mean"]} / {e12["rank_IC"]["ic_mean"]} / {e24["rank_IC"]["ic_mean"]}'),
            'changed_conclusion': False,
            'explains_negative_sign': 'NO（反而更负：−0.507 → −0.637，|t| 1.44 → 1.91）',
            'verdict': '不改变结论（排除，H01 不成立；退后到原文口径后负号更强）',
            'basis': ('若反号源于"竞争信号短期负、一年后转正"，extra=12 应显著转正 → 实测 −0.637%（t = −1.91）；'
                      'extra=24 衰减到 −0.324%（t = −0.65）仍未转正；三档 IC 同为负（−0.0132 / −0.0105 / −0.0052）'),
        }
    if 'T5' in tests:
        r5 = OUT['T5_fm_implementation']['registered']
        tv['T5'] = {
            'hypothesis': 'H09 回归实现（NW 滞后 / 常数项 / 加权口径）',
            'data_spec': 'NW lag ∈ {6,12,18}；含常数项 vs 逐月去均值（无常数项）；逐月等权 vs PC 层级加权（= 团层等权）',
            'result': (f'NW 6/12/18：β 恒 {r5["NW_lags"]["lag6"]["beta_pct"]}%，t = '
                       f'{r5["NW_lags"]["lag6"]["t_nw"]} / {r5["NW_lags"]["lag12"]["t_nw"]} / '
                       f'{r5["NW_lags"]["lag18"]["t_nw"]}；去均值 = {r5["demeaned"]["univariate"]["beta_pct"]}%'
                       f'（t = {r5["demeaned"]["univariate"]["t_nw"]}，与含常数项逐值相同）；'
                       f'PC 层级加权 = {r5["pc_level_weighted"]["univariate"]["beta_pct"]}%'
                       f'（t = {r5["pc_level_weighted"]["univariate"]["t_nw"]}）；'
                       f'加控制下 PC 层级加权 = '
                       f'{(r5["pc_level_weighted"]["with_controls"] or {}).get("beta_pct")}%'
                       f'（t = {(r5["pc_level_weighted"]["with_controls"] or {}).get("t_nw")}）'),
            'changed_conclusion': '部分（PC 层级加权使主读数层翻正 +0.2152% 但 |t| = 0.58；组合层未同向翻转）',
            'explains_negative_sign': 'NO（翻正后不显著；子层未同向 → 按登记判据不足 3/4 层）',
            'verdict': '放大或缩小但方向不变（主层翻正但子层未同向）；实现层（NW lag/常数项）本身完全排除',
            'basis': ('NW lag 与去均值只改变 t/不影响 β → H09 的"实现差异"不解释符号；'
                      '唯一使 β 转正的加权口径 = 团层等权（见 T5B），属构造/权重结构问题而非估计实现问题'),
        }
    if 'T5B' in tests:
        r5b = OUT['T5B_cluster_level']
        reg, pl = r5b['registered'], r5b['paper_like']
        tv['T5B'] = {
            'hypothesis': 'H06/H02 集团层常数 PC + 团规模权重结构（团层聚合诊断）',
            'data_spec': ('按 `{y}_clusters.csv` 的 cluster_id 聚合到 (月, 团) 层（收益 = 团内均值）；'
                          '团层等权 vs 成员数加权 FM；团层等权十分位/五分位 H−L；团规模 × PC 十分位结构'),
            'result': (f'PC 值 = 团的代理核验：{r5b["proxy_check"]["n_cluster_years"]} 个 (文本年, 团) 组内 PC 唯一'
                       f'（违反 {r5b["proxy_check"]["n_with_multiple_PC"]}）；每月团数均值 '
                       f'{reg["mean_clusters_per_month"]} = 每月唯一 PC 值数 189.12（identical）；'
                       f'团层等权 FM：登记 {reg["fm_cluster_equal_weight"]["beta_pct"]}%'
                       f'（t = {reg["fm_cluster_equal_weight"]["t_nw"]}）、原文最近似 '
                       f'{pl["fm_cluster_equal_weight"]["beta_pct"]}%（t = {pl["fm_cluster_equal_weight"]["t_nw"]}）；'
                       f'成员数加权 FM（= 公司层 EW）：{reg["fm_cluster_member_weighted"]["beta_pct"]}%'
                       f'（t = {reg["fm_cluster_member_weighted"]["t_nw"]}）逐值复现主读数；'
                       f'团层等权十分位 H−L：登记 {reg["decile_cluster_equal_weight"]["HL_pct"]}%'
                       f'（t = {reg["decile_cluster_equal_weight"]["HL_t_nw"]}）、原文最近似 '
                       f'{pl["decile_cluster_equal_weight"]["HL_pct"]}%'
                       f'（t = {pl["decile_cluster_equal_weight"]["HL_t_nw"]}）；'
                       f'PC 十分位中位成员数 D1..D10 = {list(reg["median_members_by_PC_decile"].values())}'),
            'changed_conclusion': '部分（FM 层翻正、组合层仍负 → 1/3 子层翻转，不足登记判据的 3/4）',
            'explains_negative_sign': 'NO（但解释了负号的权重来源：公司层 EW 由大团主导）',
            'verdict': ('放大或缩小但方向不变（主层翻正但子层未同向）+ 锁定负号的权重来源 = 团规模结构'
                        '（H02 方向；批 2 需继续做正交化/K 网格）'),
            'basis': (f'团层等权后 FM β 由 −0.507% 变 +0.215%（t = 0.58）→ 负号完全来自"公司层等权"下大团的超额权重；'
                      f'团规模随 PC 单调上升（中位成员数 1 → {list(reg["median_members_by_PC_decile"].values())[-1]}），'
                      f'corr(PC, log 成员数) = {reg["corr_PC_log_members"]}（团层）'),
        }
    if 'T6' in tests:
        r6 = OUT['T6_rank_ic']
        reg, pl = r6['registered'], r6['paper_like']
        tv['T6'] = {
            'hypothesis': 'H06 PC-收益秩相关/IC 与单调性（判断"单调"还是"尾部驱动"）',
            'data_spec': '逐月 Spearman rank IC（时间序列均值 + NW(6)/iid t）；十分位组均值表与组中位数表 + 单调性诊断',
            'result': (f'逐月 rank IC：登记 {reg["rank_IC"]["ic_mean"]}（t_NW = {reg["rank_IC"]["ic_t_nw"]}、'
                       f't_iid = {reg["rank_IC"]["ic_t_iid"]}，{reg["rank_IC"]["n_months"]} 月）、原文最近似 '
                       f'{pl["rank_IC"]["ic_mean"]}（t_NW = {pl["rank_IC"]["ic_t_nw"]}）；'
                       f'组均值 Spearman(组序) = {reg["monotonicity_mean"]["spearman_group_index_vs_group_mean"]}'
                       f'（相邻符号反转 {reg["monotonicity_mean"]["n_adjacent_sign_flips"]} 次）；'
                       f'组中位数 Spearman = {reg["monotonicity_median"]["spearman_group_index_vs_group_mean"]}；'
                       f'月内 Spearman(组序, 组均值) 均值 = {reg["within_month_spearman_group_vs_mean"]["mean_rho"]}'
                       f'（t = {reg["within_month_spearman_group_vs_mean"]["t_nw"]}）'),
            'changed_conclusion': False,
            'explains_negative_sign': 'NO',
            'verdict': '不改变结论（排除）+ 登记「单调不成立」（均值与中位数两表均非单调，5–6 次相邻反转）',
            'basis': ('IC 恒负但 |t| ≤ 2.06（弱负）；组层面无单调关系 → 负号是弱横截面关系而非"竞争越激烈收益越低"'
                      '的单调定价；组均值跳动（Q6 +0.71% / Q7 −0.87%）显示尾部噪声对均值口径的放大作用'),
        }
    OUT['test_verdicts'] = tv
    OUT['one_line_conclusion'] = (
        '本批结论仅描述项目自建 PC（不可比于原文）：T1–T6 全部证据下：反号**不是**排序方向 bug（T1）、尾部极值（T2）、小市值风格（T3）、'
        'PC 时点滞后 12/24 月（T4，原文口径下反而更负 −0.637%）、NW 滞后/常数项/估计实现（T5）造成的；'
        '唯一能使主读数层系数转正的是**团层等权重的重加权**（= PC 层级加权，−0.507% → +0.215%，但 |t| = 0.58、'
        '组合层仍为负）——而 PC 与团规模高度相关（PC 十分位中位成员数 1 → 33，corr(PC, log 成员数) = 0.74），'
        '故当前证据下反号最可能来自**PC 的横截面信息 = 团规模代理**（H02 方向）叠加**公司层等权下均值口径被'
        '大团/尾部观测放大**（T2/T6 登记的 2–3 倍放大）；其余候选（H01/H06 尾部/H06B/H09/H14）在本批被排除。')
    OUT['unexecuted_or_deviations'] = [
        'T2-① 的「winsorize 1%/99%」在面板构造期已由 registry `winsorization_pct` 施加 → 本轮为幂等核验'
        '（并附加未 winsor 的 next_R 与更紧 [0.025,0.975] 两口径作对照）；',
        'T5 的「按 PC 层级加权」以「同月同一 PC 值 = 同一团」实现（PC 团内恒定，F14）→ 权重 = 1/该 PC 值公司数；'
        '未按原文未披露的其他权重口径实现；',
        'T4 的 extra=12 对应原文 `PC_{i,t−12}` 的「系列平移 12 月」读法（= FY(t−2)）；extra=0 对应'
        '「t−12 月所在财年」读法（= FY(t−1)，与我方 PIT 口径等价）→ 两读法并列报告（普查 U2）；',
        '未执行：H02 正交化/K 网格、H03 12 控制、H04 年份延展、H05 样本构成、H07 复权口径、'
        'H08 PC 子口径、H11 IDF/medadj 对表、H12 语料质量、H15 置换检验 —— 属批 2/批 3 范围；',
        'β 单位为 %/单位 PC（×100 报告），与对表产物同量纲；全部读数并报两样本（登记 / 原文最近似）。',
    ]
    write_json(json_out, OUT)
    psay('7', f'JSON 产出状态 = ok（{_rel(json_out)}）')
    md_out.write_text(build_md(OUT), encoding='utf-8', newline='\n')
    psay('7', f'MD 产出状态 = ok（{_rel(md_out)}）；一句话结论已写入 §8')
    print(json.dumps({'artifact': OUT_PREFIX, 'tests': tests,
                      'summary_table': rows, 'status': 'PASS'},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
