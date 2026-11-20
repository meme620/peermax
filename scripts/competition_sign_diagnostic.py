"""network「PC → 预期收益」系数反号 · 深度诊断 H1–H8（2026-10-21，只读既有产物 + 新建诊断产物）。

背景（既定事实，不重复验证）：原文（刘亚辉等 2021）PC = −HHI(团内销售收入)，PC 越大 = 竞争越激烈，
PC → 预期收益 **为正**（表 4：+0.152/t=7.76、+0.130/t=5.40；表 3 H−L α3 +0.218/t=6.27）。
我方两套对表（MD&A 语料 / 年报全文语料）均反号（−0.507/t=−1.46、−0.456/t=−2.00 等）。

== 判据（先登记，先于运行写入本 docstring；不做事后调参、不做事后样本挑选）==
  · 每个 H 的 **成立** = 该 H 的口径/修正使 **主读数层**（登记样本 EW lag6 单变量）PC 系数 ≥ 0，
    且 4 子层（{EW, VW} × {五分位, 十分位} H−L、单变量/加控制 FM）中 ≥3 层同向转正；
  · **部分** = 向 0 移动 ≥50% 但仍为负；**排除** = 移动 <25% 或同号且 |t| 不降；**待定** = 数据不可得/证据冲突；
  · **H1 特有**（注入检验）：合成「PC 与下月收益正相关」面板（不改真数据，只在内存构造合成列）
    → 期望显著为正；**若输出为负 → 管线方向 bug 确诊，立即停止并写最小修复方案**；
  · **H8 特有**：PC 对团规模（log 团员数）回归取残差后 FM 系数 ≥0 → 机械/结构解释成立。

== 纪律 ==
  只新建脚本/产物；不覆盖任何既有产物、不改 headline_value/冻结值/主口径网络；
  全部新产物落 `artifacts/tmp/pc_diag/`（本脚本 + 各 H 的 JSON）；最终报告落
  `reports/audits/competition_sign_diagnostics_20261021.{json,md}`（新建）。

== REV2 修订（2026-10-21，依独立复核意见 F1–F6；只改本诊断轮产物）==
  复核实测证据（采纳） = artifacts/tmp/review_diag/{rv_lib.py,A_injection.py,B_h2.py,C_h3h4.py,
  D_h5.py,E_h6.py,F_h8.py,F2_h8_identity.py} 及同名 JSON。修订采用**先登记后跑**（判据与新增检验在本
  docstring 记录后执行），并逐条：
  · F1｜H8 撤回「PC ≈ 团结构代理」结论（对处理变量的定义性结构分量回归取残差 = 过度控制/bad control，
    且结论对函数形式不稳健：log 形式翻正 +0.3221 但 1/N 形式仍负 −0.6302）→ 改判**待定/未被识别**；
    补做替代检验（本脚本独立重算）：(a) 恒等式分解 PC = −own_share² + (−同团他人集中度)（两分量 FM 各自符号/t）；
    (b) 机械控制换 1/N；(c) within-N 分箱 FM 与样本占比；(d) 分布匹配重加权（原文表 2 PC 分位密度 → 重要性
    重加权 + 分位匹配两种实现）；(e) 团标签置换 **1000 次**（报经验 p 与最小可检效应 MDE）；(f) 功效/MDE 分析。
  · F2｜H5 降级为**待定（无正面证据；2007–2016 窗口不可测）**；年度/子期间一律并报 (i) iid t、(ii) 月块
    bootstrap CI95、(iii) LOO 区间；显著性判据以 bootstrap CI 为准（不用 NW t 报"显著性"）。
  · F3｜H6 归因收窄：「团规模控制使 PC 转正」**仅限 log 形式且不显著**；补 1/N 线性控制；补控制项符号
    与原文一致计数（size/bm/rev 相反、mom/illiq 同号 = 3/5 相反）。
  · F4｜精度/强度表述修正：H2 max|Δhhi| ≤5e-9（写盘 round(x,8) 精度）→ 本轮以**全精度**复算并给出最大
    实测值；H3 「11/11 为负」→「无正号（多档 |t| ≤ 0.55，全部不显著）」；H3「高度吻合」限定为「中心四分位
    吻合（p25/中位/p75 vs 原文）、sd 0.2204 vs 0.337 尾部不一致」；H2 营收字段降级为「盘上 pkl = 营业收入
    （854.30 亿），原文附录表未核」。
  · F5｜H7 持有期 β 折算月均（t+2..6 / t+7..12 ÷ 窗长）并注明重叠窗 + NW(6) 未做重叠校正、t 不作推断依据。
  · F6｜治理登记（version_manifest / registry / spec 三角刷新 + 四审计重跑 + 三哈希闭环）。
  新增检验的**先登记判据**：
  · 分布匹配重加权的两种实现（重要性权重 WLS 与分位匹配变换）**只作诊断读数**；若任一实现使主读数转正
    且 bootstrap/置换支持 → 记为「分布口径待定线索」，否则保持「待定」；不据此改写 H8 撤回结论。
  · 置换检验：经验单侧 p = P(β_perm ≤ 观测 β)；p ≥ 0.05 → 「反号」不显著区别于零分布（表述只能用"未检出
    正号"）；MDE(80% 功效) = 2.8016 × se。
  · 功效：若 MDE(80%) > 原文效应 +0.152×2（或功效 < 20%）→ 报告必须声明「本窗口对原文量级功效不足」。

用法：
  python scripts/competition_sign_diagnostic.py --tests H1 H2
  python scripts/competition_sign_diagnostic.py --assemble"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import json
import math
import numpy as np
import pandas as pd
from pricing_lib import (
    fm_regression, newey_west, param_value, rank_average_groups,
)


ROOT = Path(__file__).resolve().parents[1]

DIAG = ROOT / 'artifacts/tmp/pc_diag'

PROGRESS = ROOT / 'artifacts/tmp/pc_sign_diag_progress.md'

PANEL = ROOT / 'data/processed/pricing_sample_ma_teacher_v2.parquet'

PC_DIR = ROOT / 'data/processed/network_ma_teacher'

PC_YEARS = list(range(2017, 2023))

CLUSTER_DIR = PC_DIR / 'clusters'

REV_PKL = ROOT / 'data/raw/drive/operating_revenue_annual.pkl'

RETURNS = ROOT / 'data/processed/monthly_returns_clean_v2.csv'

FACTORS = ROOT / 'data/processed/factor_returns_daily.csv'

SW2_MAP = ROOT / 'data/processed/sw2_industry_latest.csv'

TIER_DIR = ROOT / 'artifacts/tmp/pc_rc'          # 既有（rootcause 轮）K 档缓存目录（只读复用）

OUT_JSON = ROOT / 'reports/audits/competition_sign_diagnostics_20261021.json'

OUT_MD = ROOT / 'reports/audits/competition_sign_diagnostics_20261021.md'

CONTROLS = ['size', 'bm', 'mom', 'rev', 'illiq']

PAPER_CONTROLS = {  # 原文表 4 列(2)（收益单位 %）
    'MKT_beta': (0.481, 2.82), 'size': (0.339, 23.42), 'bm': (-0.358, -8.79),
    'mom': (1.791, 15.28), 'rev': (1.835, 8.27), 'illiq': (0.028, 4.68),
    'ia': (0.085, 7.72), 'roe': (2.278, 13.63), 'lev': (1.074, 13.26),
    'disp': (-0.378, -8.53), 'ivol': (14.964, 8.23),
}

PAPER_T2 = {'mean': -0.230, 'sd': 0.337, 'p25': -0.416, 'median': -0.180, 'p75': -0.076, 'n': 15482}

TARGETS = {'c1': (0.152, 7.76), 'c2': (0.130, 5.40), 'hl_a3_ew': (0.218, 6.27)}

PAPER_TARGET_EFFECT = 0.152  # 原文表 4 列(1) PC 系数（%/单位），REV2 功效分析参照效应

SEED = 20261021

NW = 6

NW_PAPER = 12

def psay(step: str, msg: str) -> None:
    line = f'[D-{step}] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')

def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)

def pct(x):
    return None if x is None or not np.isfinite(x) else round(float(x) * 100.0, 4)

def dump(tag: str, obj) -> None:
    p = DIAG / f'{tag}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    psay(tag, f'产出 {p.relative_to(ROOT)}')

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def power_mde(beta_pct, t_nw, effect_ref_pct: float = 0.152) -> dict | None:
    """REV2-F1(f)：由 (β, t) 反推 se → 95% CI、80% 功效 MDE、原文量级效应下的功效。"""
    if beta_pct is None or t_nw in (None, 0) or not np.isfinite(t_nw):
        return None
    se = abs(float(beta_pct) / float(t_nw))
    z = effect_ref_pct / se
    return {
        'beta_pct': beta_pct, 't_nw': t_nw, 'se_pct': r4(se),
        'ci95_pct': [r4(beta_pct - 1.96 * se), r4(beta_pct + 1.96 * se)],
        'mde80_pct': r4(2.8016 * se),
        'power_at_effect_ref': r4(_phi(z - 1.96) + _phi(-z - 1.96)),
        'effect_ref_pct': effect_ref_pct,
        'note': 'MDE(80%) = 2.8016 × se（双侧 α=0.05）；功效 = P(拒绝 β=0 | β=effect_ref) 正态近似',
    }

def fmt_fm(blk: dict, keys=('PC',)) -> dict:
    out = {'n_months_used': blk['n_months_used'], 'coefs_pct': {}}
    for c in keys:
        s = blk['coefs'][c]
        out['coefs_pct'][c] = {'beta': pct(s['beta']), 't_nw': r4(s['t_nw']), 't_iid': r4(s['t_iid'])}
    return out

def load_pc(pc_dir: Path = PC_DIR, years=PC_YEARS) -> pd.DataFrame:
    frames = []
    for y in years:
        f = pc_dir / f'{y}_pc.csv'
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={'focal_id': str})
        frames.append(d)
    pc = pd.concat(frames, ignore_index=True)
    for c in ('hhi', 'pc', 'hhi_excl_self', 'revenue_coverage'):
        pc[c] = pd.to_numeric(pc[c], errors='coerce')
    return pc

def load_panel() -> pd.DataFrame:
    cols = ['focal_id', 'month', 'network_version', 'R_next', 'next_R', 'R_next_postadj',
            'market_equity', 'size', 'bm', 'mom', 'rev', 'illiq', 'turnover',
            'pricing_in_fm_sample', 'pricing_exclusion_reason', 'me_below_q10', 'me_q10_market',
            'AnalystCoverage', 'ind_l2_ret_vw', 'ind_l1_ret_vw', 'st_flag_focal', 'recent_ipo']
    d = pd.read_parquet(PANEL, columns=cols)
    d['text_year'] = pd.to_numeric(d['network_version'], errors='coerce')
    return d

def merge_pc(panel: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    m = panel.merge(pc[['focal_id', 'text_year', 'pc', 'hhi', 'hhi_excl_self',
                        'n_competitors', 'revenue_coverage', 'revenue_year']],
                    on=['focal_id', 'text_year'], how='left')
    m['PC'] = pd.to_numeric(m['pc'], errors='coerce')
    return m

def load_rev(year: int) -> dict:
    d = pd.read_pickle(REV_PKL)
    d.index = [ts.strftime('%Y') if hasattr(ts, 'strftime') else str(ts)[:4] for ts in d.index]
    y = str(year)
    if y not in d.index:
        return {}
    row = d.loc[y]
    return {str(c).split('.')[0].zfill(6): float(v) for c, v in row.items() if v == v and v > 0}

def sw2_asof(month_series: pd.Series, firm_series: pd.Series) -> pd.Series:
    mp = pd.read_csv(SW2_MAP, dtype={'code6': str})
    mp = mp[mp['is_current_version'].astype(bool)]
    mp['firm_id'] = mp['firm_id'].astype(str)
    lut = mp.drop_duplicates('firm_id').set_index('firm_id')['l2_code'].astype(str)
    return firm_series.map(lut)

def monthly_factors() -> pd.DataFrame:
    fac = pd.read_csv(FACTORS, dtype={'date': str})
    fac['month'] = fac['date'].str[:7]
    f = fac.groupby('month').agg(
        MKT=('MKT_d', lambda s: float(np.prod(1 + s) - 1)),
        SMB=('SMB_d', lambda s: float(np.prod(1 + s) - 1)),
        HML=('HML_d', lambda s: float(np.prod(1 + s) - 1)),
        MOM=('MOM_d', lambda s: float(np.prod(1 + s) - 1)),
        RF=('rf_d', lambda s: float(np.prod(1 + s) - 1))).reset_index()
    return f

def make_samples(m: pd.DataFrame) -> dict:
    pc_ok = m['PC'].notna() & m['R_next'].notna()
    ctrl_ok = m[CONTROLS].notna().all(axis=1)
    reg = m['pricing_in_fm_sample'].astype(bool)
    return {
        'registered': m[pc_ok & reg].copy(),
        'paper_like': m[pc_ok].copy(),
        'paper_like_ctrl': m[pc_ok & ctrl_ok].copy(),
        'registered_ctrl': m[pc_ok & reg & ctrl_ok].copy(),
    }

def fm(samp, ycol, xcols, weight_col=None, nw_lag=NW, min_n=None):
    if min_n is None:
        min_n = int(param_value('sample_and_spec_registry')['min_focal_count_per_month'])
    blk, perm = fm_regression(samp, ycol, list(xcols), weight_col=weight_col,
                              min_n=min_n, nw_lag=nw_lag)
    blk['_per_month'] = perm
    return blk

def portfolio_key(samp, keycol, n_groups, wtag, wcol, lag=NW, fmonth=None):
    """与对表脚本同构的组合函数（分组 = 逐月 rank_average，H−L = 最高 − 最低）。"""
    d = samp[['month', keycol, 'R_next', 'market_equity']].copy()
    d = d[d[keycol].notna() & d['R_next'].notna()]
    if wcol:
        d = d[d[wcol].notna() & (d[wcol] > 0)]
    d['grp'] = d.groupby('month', group_keys=False)[keycol].transform(
        lambda s: rank_average_groups(s, n_groups))
    d = d[d['grp'].notna()]
    rows = []
    for mo, sub in d.groupby('month'):
        rec = {'month': mo, 'n': int(len(sub))}
        for g in range(1, n_groups + 1):
            gg = sub[sub['grp'] == g]
            rec[f'g{g}'] = (float(np.average(gg['R_next'], weights=gg[wcol])) if (wcol and not gg.empty)
                            else (float(gg['R_next'].mean()) if not gg.empty else None))
        rows.append(rec)
    t = pd.DataFrame(rows).sort_values('month')
    t['HL'] = t[f'g{n_groups}'] - t['g1']
    nw = newey_west(t['HL'].tolist(), lag=lag)
    out = {'n_months': int(len(t)), 'HL_raw_pct': pct(nw['mean']), 'HL_raw_t_nw': r4(nw['t_nw']),
           'group_mean_pct': {f'Q{g}': pct(float(t[f'g{g}'].mean())) for g in range(1, n_groups + 1)}}
    if fmonth is not None:
        t2 = t.merge(fmonth, on='month', how='left')
        for atag, cols in (('a3', ['MKT', 'SMB', 'HML']), ('a4', ['MKT', 'SMB', 'HML', 'MOM'])):
            ok = t2[cols].notna().all(axis=1)
            ls = (t2['HL'] - t2['RF']).to_numpy(float)[ok.to_numpy()]
            X = np.column_stack([np.ones(len(ls))] + [t2.loc[ok, c].to_numpy(float) for c in cols])
            b, se, tt, T = _ols_hac(ls, X, lag)
            out[f'alpha_{atag}_pct'] = pct(b[0])
            out[f'alpha_{atag}_t'] = r4(tt[0])
    return out

def _ols_hac(y, X, lag):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    T, k = X.shape
    Ae = X * e[:, None]
    S = Ae.T @ Ae / T
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        G = Ae[l:].T @ Ae[:-l] / T
        S = S + w * (G + G.T)
    V = XtXi @ (T * S) @ XtXi
    se = np.sqrt(np.diag(V))
    with np.errstate(divide='ignore', invalid='ignore'):
        return beta, se, beta / se, T

def ctrl_compare(samp, label, ycol='R_next', nw_lag=NW):
    """加控制 FM：报每个控制变量系数（%/单位）与 t。"""
    blk = fm(samp, ycol, ['PC'] + CONTROLS, nw_lag=nw_lag)
    return {f'{label}': {
        'n_months_used': blk['n_months_used'],
        'coefs': {c: {'beta_pct': pct(blk['coefs'][c]['beta']), 't_nw': r4(blk['coefs'][c]['t_nw'])}
                  for c in ['PC'] + CONTROLS},
    }}

def run_H1(m, S, fmonth) -> dict:
    psay('H1', '状态 = 开始：H1 方向/排序 bug（注入检验 + 组均值核验 + 手算 OLS）')
    reg, paper = S['registered'], S['paper_like']
    out = {'hypothesis': 'H1 代码里的排序/高低方向是否反了（H/L 定义、PC 排序方向、FM 符号约定）'}

    # ---- (1) 逐行核读（静态证据，附数字核验）
    diff = float((m['PC'] + m['hhi']).abs().max())
    out['code_audit'] = {
        'pc_is_negative_hhi': bool(diff < 1e-9), 'max_abs_pc_plus_hhi': r4(diff),
        'pc_formula_src': 'scripts/network_business_competition_hhi.py L152-153: shares=v/tot（含 focal 自身）；'
                          'hhi=(shares**2).sum()；pc=-hhi',
        'group_src': 'scripts/pricing_lib.py L243-252 rank_average_groups：rank 升序 → '
                     '组号随取值升序 → g{N} = PC 最大组；对表脚本 L371 HL = g{n_groups} − g1 = 高 − 低',
        'fm_src': 'scripts/competition_pricing_correspondence.py L319: fm_regression(samp, "R_next", ["PC"]) '
                  '→ 系数即 dR/dPC，无额外取负',
        'conclusion': '静态读码：H = PC 最大（竞争最激烈）= 原文 "High"；L = PC 最小；FM 系数 = dR/dPC',
    }

    # ---- (2) 分组核验：H 组 PC 应更大、HHI 应更小
    d = reg[['month', 'PC', 'hhi', 'n_competitors', 'R_next']].dropna(subset=['PC', 'R_next']).copy()
    d['grp'] = d.groupby('month', group_keys=False)['PC'].transform(lambda s: rank_average_groups(s, 10))
    g = d.dropna(subset=['grp']).groupby('grp').agg(
        n=('PC', 'size'), pc_mean=('PC', 'mean'), pc_med=('PC', 'median'),
        hhi_mean=('hhi', 'mean'), peer_med=('n_competitors', 'median'),
        ret_mean=('R_next', 'mean'))
    out['group_audit_decile'] = {
        'group_stat': {f'D{int(i)}': {'n': int(r.n), 'pc_mean': r4(r.pc_mean), 'pc_median': r4(r.pc_med),
                                      'hhi_mean': r4(r.hhi_mean), 'peer_median': float(r.peer_med),
                                      'r_mean_pct': pct(r.ret_mean)} for i, r in g.iterrows()},
        'D10_minus_D1_pc': r4(float(g.loc[10, 'pc_mean'] - g.loc[1, 'pc_mean'])),
        'D10_minus_D1_hhi': r4(float(g.loc[10, 'hhi_mean'] - g.loc[1, 'hhi_mean'])),
        'H_group_is_high_pc': bool(g.loc[10, 'pc_med'] > g.loc[1, 'pc_med']),
        'H_group_is_low_hhi': bool(g.loc[10, 'hhi_mean'] < g.loc[1, 'hhi_mean']),
    }
    psay('H1', f"组核验状态 = ok：D1→D10 PC 中位 {r4(float(g.loc[1,'pc_med']))}→{r4(float(g.loc[10,'pc_med']))}、"
               f"HHI 均值 {r4(float(g.loc[1,'hhi_mean']))}→{r4(float(g.loc[10,'hhi_mean']))}"
               f"（H 组 PC {g.loc[10,'pc_med'] > g.loc[1,'pc_med']} / HHI 更小 {g.loc[10,'hhi_mean'] < g.loc[1,'hhi_mean']}）")

    # ---- (3) 注入检验：合成 PC（月内 R_next 秩的单调递增函数）
    inj = {}
    for tag, sign in (('syn_rank_pos', 1.0), ('syn_rank_neg', -1.0)):
        s = reg.copy()
        rk = s.groupby('month')['R_next'].rank(pct=True)
        s['PC_syn'] = sign * (rk - 0.5)
        blk = fm(s, 'R_next', ['PC_syn'])
        p = portfolio_key(s, 'PC_syn', 10, 'EW', None)
        pv = portfolio_key(s, 'PC_syn', 10, 'VW', 'market_equity')
        inj[tag] = {'beta_pct': pct(blk['coefs']['PC_syn']['beta']), 't_nw': r4(blk['coefs']['PC_syn']['t_nw']),
                    'HL_decile_EW_pct': p['HL_raw_pct'], 'HL_decile_EW_t': p['HL_raw_t_nw'],
                    'HL_decile_VW_pct': pv['HL_raw_pct'], 'HL_decile_VW_t': pv['HL_raw_t_nw']}
    rng = np.random.default_rng(SEED)
    s = reg.copy()
    s['PC_rnd'] = rng.uniform(-0.5, 0.5, len(s))
    blk = fm(s, 'R_next', ['PC_rnd'])
    inj['syn_random'] = {'beta_pct': pct(blk['coefs']['PC_rnd']['beta']),
                         't_nw': r4(blk['coefs']['PC_rnd']['t_nw']),
                         'note': f'月内独立均匀随机（seed={SEED}）→ 期望 ≈ 0'}
    # 真 PC 同步读数（对照）
    blk0 = fm(reg, 'R_next', ['PC'])
    inj['real_PC'] = {'beta_pct': pct(blk0['coefs']['PC']['beta']), 't_nw': r4(blk0['coefs']['PC']['t_nw'])}
    # REV2：PC×(−1) 反证（镜像处理变量应逐位反号；对照复核 A_injection 的 +0.5068 / t = +1.4365）
    s = reg.copy()
    s['PC_neg'] = -s['PC']
    blkn = fm(s, 'R_next', ['PC_neg'])
    inj['adv_neg_PC'] = {'beta_pct': pct(blkn['coefs']['PC_neg']['beta']),
                         't_nw': r4(blkn['coefs']['PC_neg']['t_nw']),
                         'note': 'PC×(−1) 反证：系数逐位反号 → 排除符号书写/约定层 bug（复核实测 = +0.5068 / t = +1.4365）'}
    out['injection_test'] = inj
    psay('H1', f"注入检验状态 = ok：合成正相关 PC_syn β = {inj['syn_rank_pos']['beta_pct']}%"
               f"（t = {inj['syn_rank_pos']['t_nw']}）、H−L(十分位 EW) = {inj['syn_rank_pos']['HL_decile_EW_pct']}%"
               f"；镜像负相关 β = {inj['syn_rank_neg']['beta_pct']}%；随机 β = {inj['syn_random']['beta_pct']}%")

    # ---- (4) 手算 OLS 对拍（逐月 Σ(x−x̄)(y−ȳ)/Σ(x−x̄)² 与库函数逐月系数）
    hand = {}
    for mo in ('2019-06', '2021-03'):
        s = reg[reg['month'] == mo]
        x = s['PC'].to_numpy(float)
        y = s['R_next'].to_numpy(float)
        xm, ym = x - x.mean(), y - y.mean()
        b = float((xm * ym).sum() / (xm ** 2).sum())
        perm = blk0['_per_month']
        lib = None
        for r in perm:
            if r['month'] == mo and r['used']:
                lib = r['beta']['PC']
        hand[mo] = {'n': int(len(s)), 'hand_beta_raw': b, 'lib_beta_raw': r4(lib),
                    'beta_pct': pct(b), 'match': bool(lib is not None and abs(b - lib) < 1e-9),
                    'corr_PC_R_next': r4(float(np.corrcoef(x, y)[0, 1]))}
    out['hand_ols_check'] = hand
    out['reading'] = {
        'registered_EW_lag6_univariate': {'beta_pct': pct(blk0['coefs']['PC']['beta']),
                                          't_nw': r4(blk0['coefs']['PC']['t_nw']),
                                          'n_months_used': blk0['n_months_used']},
    }
    dump('H1', out)
    psay('H1', '状态 = 完成')
    return out

def run_H2(m, S, fmonth, panel) -> dict:
    psay('H2', '状态 = 开始：H2 PC 定义/算式细节（手核团 + 剔除自身变体 + 分布对照 + 营收字段）')
    reg, paper = S['registered'], S['paper_like']
    out = {'hypothesis': 'H2 PC 定义/算式与原文的细节差（收入字段 / 含 focal 自身 / 份额分母 / HHI 公式）'}
    out['paper_definition'] = {
        'quote': '「群内公司销售收入的赫芬达尔指数的相反数(Valta,2012)即为公司i的竞争程度(PC)。'
                 'PC 越大，表明群内公司之间的竞争越激烈。」（原文 §三(二)1 第四步；脚注 3 引 Valta 2012）',
        'step4_deviation': '原文第四步 = 「任何与其相似度超过团内最小相似度的公司」（字面含团外）；'
                           '我方 = 团内全部成员（methodology_execution_plan.md:2290 已登记偏离）。'
                           '二者不是同一对手集合 → PC 不是原文变量的严格复现，判定 = 不可比/未复现（不是「近似一致」）。',
    }
    # ---- (1) 手核 2 个团：从 clusters + 营收现算 HHI（含/不含自身）与产物逐位对拍
    cl = pd.read_csv(CLUSTER_DIR / '2019_clusters.csv', dtype={'firm_id': str})
    byc = cl.groupby('cluster_id')['firm_id'].apply(list)
    sizes = cl.groupby('cluster_id').size().sort_values()
    cand = [c for c in sizes.index if 5 <= sizes[c] <= 12]
    med_size = int(sizes.iloc[len(sizes) // 2])
    med_cand = [c for c in sizes.index if sizes[c] == med_size]
    pc19 = pd.read_csv(PC_DIR / '2019_pc.csv', dtype={'focal_id': str})
    rev18 = load_rev(2018)
    audits = []
    for cid in (cand[:2] + med_cand[:1] + list(sizes.index[-1:])):
        members = byc[cid]
        vals = {f: rev18.get(f[2:]) for f in members}
        cov = {f: v for f, v in vals.items() if v is not None}
        tot = sum(cov.values())
        shares = np.array([v / tot for v in cov.values()])
        hhi_self = float((shares ** 2).sum())
        rec = {'text_year': 2019, 'cluster_id': int(cid), 'n_members': len(members),
               'n_rev_covered': len(cov), 'hhi_recomputed_incl_self': r4(hhi_self),
               'pc_recomputed_incl_self': r4(-hhi_self),
               'members_head': members[:6]}
        sub = pc19[pc19['focal_id'].isin(members)]
        # REV2-F4：max|Δ| 以**全精度**记录（原报告 r4 四舍五入把 ~3e-9 显示为 0.0）
        rec['csv_hhi_max_abs_diff_raw'] = float((sub['hhi'] - hhi_self).abs().max())
        rec['csv_pc_max_abs_diff_raw'] = float((sub['pc'] - (-hhi_self)).abs().max())
        rec['csv_hhi_max_abs_diff'] = r4(rec['csv_hhi_max_abs_diff_raw'])
        rec['csv_hhi_excl_self_is_negative_hhi_excl'] = None
        # 剔除自身（对每个成员各算一次；判定列存的是 −HHI_excl 还是 +HHI_excl）
        d_same, d_opp, pairs_x = [], [], []
        for f in members:
            e = {k: v for k, v in cov.items() if k != f}
            if not e:
                continue
            t2 = sum(e.values())
            hx = float(((np.array(list(e.values())) / t2) ** 2).sum()) if t2 > 0 else None
            if hx is None:
                continue
            rowx = pc19[pc19['focal_id'] == f]
            if len(rowx):
                cv = float(rowx['hhi_excl_self'].iloc[0])
                pairs_x.append({'code': f, 'csv': cv, 'recomputed_hhi_excl': r4(hx)})
                d_same.append(abs(cv - hx))       # 若列 = +HHI_excl
                d_opp.append(abs(cv + hx))        # 若列 = −HHI_excl
        if d_same:
            rec['csv_col_equals_plus_HHI_excl'] = bool(max(d_same) < 1e-6)
            rec['csv_col_equals_minus_HHI_excl'] = bool(max(d_opp) < 1e-6)
            rec['hhi_excl_self_pairs_sample'] = pairs_x[:3]
            rec['max_abs_diff_vs_plus'] = r4(float(max(d_same)))
            rec['max_abs_diff_vs_minus'] = r4(float(max(d_opp)))
        audits.append(rec)
    out['manual_cluster_audit'] = audits
    # REV2-F4：最大团全精度复算（写盘 round(x,8) → 理论最大表示误差 5e-9；给出实测最大 |Δ|）
    cid_max = int(sizes.index[-1])
    members_max = byc[cid_max]
    vals = {f: rev18.get(f[2:]) for f in members_max}
    cov = {f: v for f, v in vals.items() if v is not None}
    tot = sum(cov.values())
    hhi_max = float(((np.array(list(cov.values())) / tot) ** 2).sum())
    sub = pc19[pc19['focal_id'].isin(members_max)]
    out['write_precision_audit'] = {
        'largest_cluster_2019_id': cid_max, 'n_members': len(members_max), 'n_rev_covered': len(cov),
        'hhi_recomputed_full_precision': hhi_max,
        'csv_hhi_max_abs_diff_raw': float((sub['hhi'] - hhi_max).abs().max()),
        'csv_pc_max_abs_diff_raw': float((sub['pc'] + hhi_max).abs().max()),
        'theoretical_bound': 5e-9,
        'note': 'pc.csv 写盘 = round(x,8) → 理论最大表示误差 ≤5e-9；复核独立实测（tear 295/210/3）= '
                '3.0e-9 / 7.3e-10 / 8.8e-10；原报告「max|Δhhi| = 0」为 r4 显示精度假象，本轮改记 ≤5e-9',
    }
    psay('H2', f"写盘精度核验状态 = ok：最大团（id={cid_max}，{len(members_max)} 成员）全精度 "
               f"max|Δhhi| = {out['write_precision_audit']['csv_hhi_max_abs_diff_raw']:.3e}（≤5e-9 为 round(x,8) 精度）")
    psay('H2', f"手核团状态 = ok：{len(audits)} 团（成员 {[a['n_members'] for a in audits]}）全精度对拍 "
               f"max|Δhhi| = {max((a.get('csv_hhi_max_abs_diff_raw') or 0) for a in audits):.3e}"
               f"（≤5e-9 写盘精度）；hhi_excl_self 列 = −HHI_excl_self：{[a.get('csv_col_equals_minus_HHI_excl') for a in audits]}")

    # ---- (2) 剔除自身变体：PC_x = −HHI_excl（符号按手核结果确定），重跑 FM
    merged = m.copy()
    signs = [a.get('csv_col_equals_minus_HHI_excl') for a in audits if a.get('csv_col_equals_minus_HHI_excl') is not None]
    col_is_minus = bool(signs) and all(signs)
    merged['PC_x'] = (pd.to_numeric(merged['hhi_excl_self'], errors='coerce') if col_is_minus
                  else -pd.to_numeric(merged['hhi_excl_self'], errors='coerce'))
    merged['PC_x'] = merged['PC_x'].where(merged['PC'].notna())
    pc_ok = merged['PC_x'].notna() & merged['R_next'].notna()
    ctrl_ok = merged[CONTROLS].notna().all(axis=1)
    regx = merged[pc_ok & merged['pricing_in_fm_sample'].astype(bool)]
    paperx = merged[pc_ok]
    var = {}
    for tag, s in (('registered', regx), ('paper_like', paperx)):
        b1 = fm(s, 'R_next', ['PC_x'])
        b2 = fm(s, 'R_next', ['PC_x'] + CONTROLS)
        var[tag] = {'n_rows': int(len(s)),
                    'univariate': {'beta_pct': pct(b1['coefs']['PC_x']['beta']),
                                   't_nw': r4(b1['coefs']['PC_x']['t_nw']),
                                   'n_months': b1['n_months_used']},
                    'with_controls': {'beta_pct': pct(b2['coefs']['PC_x']['beta']),
                                      't_nw': r4(b2['coefs']['PC_x']['t_nw'])}}
        p = portfolio_key(s, 'PC_x', 5, 'EW', None, fmonth=fmonth)
        var[tag]['quintile_EW_alpha_a3_pct'] = p.get('alpha_a3_pct')
        var[tag]['quintile_EW_alpha_a3_t'] = p.get('alpha_a3_t')
    # 覆盖率 = 1 子样本（分母不含缺营收成员）
    cov1_reg = reg[reg['revenue_coverage'] == 1.0]
    cov1_pap = paper[paper['revenue_coverage'] == 1.0]
    var['coverage_eq_1'] = {}
    for tag, s in (('registered', cov1_reg), ('paper_like', cov1_pap)):
        b1 = fm(s, 'R_next', ['PC'])
        var['coverage_eq_1'][tag] = {'n_rows': int(len(s)), 'beta_pct': pct(b1['coefs']['PC']['beta']),
                                     't_nw': r4(b1['coefs']['PC']['t_nw'])}
    out['excl_self_and_coverage_variants'] = var
    psay('H2', f"剔除自身变体状态 = ok：登记样本 β = {var['registered']['univariate']['beta_pct']}%"
               f"（t = {var['registered']['univariate']['t_nw']}，原 PC = −0.507）；"
               f"覆盖=1 子样本 β = {var['coverage_eq_1']['registered']['beta_pct']}%")

    # ---- (3) 分布对照（表 2）
    pcf = m.loc[m['PC'].notna(), 'PC']
    out['distribution_vs_paper'] = {
        'ours_firm_year': {'n': int(pcf.size), 'mean': r4(pcf.mean()), 'sd': r4(pcf.std()),
                           'p25': r4(pcf.quantile(.25)), 'median': r4(pcf.median()), 'p75': r4(pcf.quantile(.75))},
        'paper_table2': PAPER_T2,
        'ours_in_registered_monthly': {'n': int(reg['PC'].size), 'mean': r4(reg['PC'].mean()),
                                       'sd': r4(reg['PC'].std()), 'median': r4(reg['PC'].median())},
        'coverage': {'median': r4(m.loc[m['PC'].notna(), 'revenue_coverage'].median()),
                     'share_lt1': r4(float((m.loc[m['PC'].notna(), 'revenue_coverage'] < 1).mean()))},
    }
    psay('H2', f"分布对照状态 = ok：我方均值 {out['distribution_vs_paper']['ours_firm_year']['mean']} / "
               f"中位 {out['distribution_vs_paper']['ours_firm_year']['median']}（原文 −0.230 / −0.180）")

    # ---- (4) 营收字段探查（茅台 + 面板口径）
    rev = pd.read_pickle(REV_PKL)
    rev.index = [ts.strftime('%Y') if hasattr(ts, 'strftime') else str(ts)[:4] for ts in rev.index]
    row = rev.loc['2019']
    key = [c for c in row.index if str(c).split('.')[0].zfill(6) == '600519']
    out['revenue_field'] = {
        'maotai_FY2019_pkl_value_cny': r4(float(row[key[0]])) if key else None,
        'public_operating_revenue_cny': 85429573467.25,
        'interpretation': '盘上 pkl 取值 = 「营业收入」(854.30 亿)，非「营业总收入」(888.54 亿)；'
                          '**原文附录表未核**（原文未给出该字段的逐值附录）→ 降级为「与原文口径一致」的弱表述，'
                          '仅登记盘上事实（REV2-F4）',
        'excl_self_column_semantics': 'hhi_excl_self 列存 −HHI(剔除自身)，列名与符号语义不一致（既有登记缺陷，非符号事故）',
    }
    dump('H2', out)
    psay('H2', '状态 = 完成')
    return out

def run_H3(panel, fmonth) -> dict:
    psay('H3', '状态 = 开始：H3 K 校准口径（8 档既有缓存 + 3 档粗 K（K≈100–200）新重建，逐档重算 PC + 重跑 FM）')
    out = {'hypothesis': 'H3 团数 K 校准口径（我方中位 peer=44 vs 原文肘点法）决定符号',
           'reuse_note': '细档 PC 由既有（rootcause 轮）8 档 K 重建缓存 artifacts/tmp/pc_rc/h1_net_t*/ 提供（只读）；'
                         '粗档 K≈100–200（目标中位 peer 150/200/300）由本轮用既有重建器另起 tag `dcoarse` 新建'
                         '（artifacts/tmp/pc_rc/dcoarse_net_t*/，未覆盖任何既有文件）；本轮的 FM/组合读数由本脚本独立重跑'}
    tier_dirs = {t: TIER_DIR / f'h1_net_t{t}' for t in (5, 10, 20, 30, 44, 60, 80, 120)}
    for t in (150, 200, 300):
        d = TIER_DIR / f'dcoarse_net_t{t}'
        if d.exists():
            tier_dirs[t] = d
    table = {}
    for t, pdir in tier_dirs.items():
        if not pdir.exists() or not (pdir / '2019_pc.csv').exists():
            continue
        pcx = load_pc(pdir)
        mm = merge_pc(panel, pcx)
        ss = make_samples(mm)
        reg, pap = ss['registered'], ss['paper_like_ctrl']
        b1 = fm(reg, 'R_next', ['PC'], nw_lag=NW_PAPER)
        b2 = fm(reg, 'R_next', ['PC'] + CONTROLS, nw_lag=NW_PAPER)
        b3 = fm(pap, 'R_next', ['PC'], nw_lag=NW_PAPER)
        p = portfolio_key(reg, 'PC', 5, 'EW', None, fmonth=fmonth)
        pv = portfolio_key(reg, 'PC', 5, 'VW', 'market_equity', fmonth=fmonth)
        kfile = pdir / 'clusters' / '2019_clusters.csv'
        k19 = (int(pd.read_csv(kfile, dtype=str)['cluster_id'].nunique()) if kfile.exists() else None)
        if kfile.exists():
            out.setdefault('k_2019_source', {})[str(t)] = \
                f"{kfile.relative_to(ROOT).as_posix()}（团数 = unique cluster_id）"
        table[str(t)] = {
            'k_2019': k19, 'n_firm_year': int(pcx['pc'].notna().sum()),
            'pc_mean': r4(float(pcx['pc'].mean())), 'pc_median': r4(float(pcx['pc'].median())),
            'pc_p25': r4(float(pcx['pc'].quantile(.25))), 'pc_p75': r4(float(pcx['pc'].quantile(.75))),
            'fm_reg_registered_lag12': {'beta_pct': pct(b1['coefs']['PC']['beta']),
                                        't_nw': r4(b1['coefs']['PC']['t_nw'])},
            'fm_reg_ctrl_lag12': {'beta_pct': pct(b2['coefs']['PC']['beta']), 't_nw': r4(b2['coefs']['PC']['t_nw'])},
            'fm_paper_ctrl_lag12': {'beta_pct': pct(b3['coefs']['PC']['beta']), 't_nw': r4(b3['coefs']['PC']['t_nw'])},
            'quintile_EW_alpha_a3_pct': p.get('alpha_a3_pct'), 'quintile_EW_alpha_a3_t': p.get('alpha_a3_t'),
            'quintile_VW_alpha_a3_pct': pv.get('alpha_a3_pct'), 'quintile_VW_alpha_a3_t': pv.get('alpha_a3_t'),
        }
        psay('H3', f"K 档 tier={t} 状态 = ok：PC 中位 {table[str(t)]['pc_median']}、"
                   f"FM(登记,lag12) {table[str(t)]['fm_reg_registered_lag12']['beta_pct']}%"
                   f"（t = {table[str(t)]['fm_reg_registered_lag12']['t_nw']}）")
    out['tier_grid'] = table
    out['all_negative'] = bool(all(v['fm_reg_registered_lag12']['beta_pct'] < 0 for v in table.values()))
    out['coarse_K_note'] = ('请求的「固定 K = 100/200/300」区由本轮粗档（目标中位 peer 150/200/300 → K(2019) = 192/167/116）覆盖：'
                            '**主读数层（单变量 FM）仍全为负**（−0.169/t=−0.54、−0.057/t=−0.14、−1.711/t=−1.80）；'
                            '但 tier=150/200 的**加控制与五分位 α3 子层转正**（+0.0588/t=0.23、+0.1744/t=0.48；'
                            'α3 = +0.146/t=0.43、+0.350/t=0.95，均不显著），tier=300 仍为负 → '
                            '粗 K 区符号不稳定、但按登记判据（主读数层 ≥0 且 4 子层 ≥3 层同向转正）不足判「成立」；'
                            '且该区 PC 分布已明显偏离原文表 2（PC 中位 −0.059/−0.049/−0.036 vs 原文 −0.180），'
                            '而"与原文分布可比"的 K 区（tier≈10，K=826）同样为负')
    # 肘点探索既有产物
    for f in ('network_business_k_explore.json', 'network_business_k_explore_cluster_only.json'):
        p = ROOT / 'reports/data_gates/G0' / f
        if p.exists():
            try:
                j = json.loads(p.read_text(encoding='utf-8'))
                out.setdefault('elbow_explore_existing', {})[f] = {
                    k: j[k] for k in list(j)[:6]}
            except Exception as e:  # noqa: BLE001
                out.setdefault('elbow_explore_existing', {})[f] = f'read_failed:{e}'
    dump('H3', out)
    psay('H3', f"状态 = 完成：{len(table)} 档主读数层全为负 = {out['all_negative']}")
    return out

def run_H4(m, S, panel, fmonth) -> dict:
    psay('H4', '状态 = 开始：H4 样本口径（分析师覆盖门 / 微盘 / 全 A 股宇宙）')
    out = {'hypothesis': 'H4 样本要求分析师覆盖（砍掉 ~1/3、偏大市值）导致反号'}
    variants = {}
    # A/B/C：面板内切片
    pc_ok = m['PC'].notna() & m['R_next'].notna()
    ctrl5 = m[CONTROLS].notna().all(axis=1)
    ana_ok = m['AnalystCoverage'].notna()
    slices = {
        'A_registered': pc_ok & m['pricing_in_fm_sample'].astype(bool),
        'B_paper_like': pc_ok,
        'C_paper_like_ctrl5': pc_ok & ctrl5,
        'D_no_analyst_gate': pc_ok & ctrl5 & (~ana_ok),          # 仅因分析师覆盖缺失而被剔的行
        'E_only_analyst_gate_excluded': pc_ok & (~m['pricing_in_fm_sample'].astype(bool)),
    }
    for tag, mask in slices.items():
        s = m[mask]
        if len(s) < 1000:
            variants[tag] = {'n_rows': int(len(s)), 'note': 'rows<1000，跳过 FM'}
            continue
        b1 = fm(s, 'R_next', ['PC'])
        row = {'n_rows': int(len(s)), 'n_months': int(s['month'].nunique()),
               'mean_market_equity_1e8': r4(float(s['market_equity'].mean() / 1e8)),
               'analyst_cov_mean': r4(float(s['AnalystCoverage'].mean())),
               'pc_mean': r4(float(s['PC'].mean())),
               'fm_univariate': {'beta_pct': pct(b1['coefs']['PC']['beta']), 't_nw': r4(b1['coefs']['PC']['t_nw']),
                                 'n_months_used': b1['n_months_used']}}
        if s[CONTROLS].notna().all(axis=1).sum() > 1000:
            b2 = fm(s, 'R_next', ['PC'] + CONTROLS)
            row['fm_with_controls'] = {'beta_pct': pct(b2['coefs']['PC']['beta']),
                                       't_nw': r4(b2['coefs']['PC']['t_nw'])}
        variants[tag] = row
        psay('H4', f"变体 {tag} 状态 = ok：n = {row['n_rows']:,}、β = {row['fm_univariate']['beta_pct']}%"
                   f"（t = {row['fm_univariate']['t_nw']}）")
    # 微盘开关（面板内）
    for tag, mask in (('F_registered_plus_microcap', pc_ok & (m['pricing_in_fm_sample'].astype(bool) | m['me_below_q10'].astype(bool))),
                      ('G_registered_excl_microcap', pc_ok & m['pricing_in_fm_sample'].astype(bool))):
        s = m[mask]
        b1 = fm(s, 'R_next', ['PC'])
        variants[tag] = {'n_rows': int(len(s)), 'fm_univariate': {'beta_pct': pct(b1['coefs']['PC']['beta']),
                                                                  't_nw': r4(b1['coefs']['PC']['t_nw'])}}
        psay('H4', f"变体 {tag} 状态 = ok：n = {len(s):,}、β = {variants[tag]['fm_univariate']['beta_pct']}%")
    # H：全 PC 宇宙（无 pricing_gate 筛选，无分析师门；收益取 monthly_returns_clean_v2）
    try:
        pc_all = load_pc(PC_DIR)
        pc_all = pc_all[pc_all['pc'].notna()]
        ret = pd.read_csv(RETURNS, usecols=['firm_id', 'month', 'R'], dtype={'firm_id': str})
        ret = ret[ret['month'].between('2018-01', '2023-12')]
        # PC 生效：available_at < month（PIT，取文本年 y → 生效自 y+1-05）
        aa = pd.to_datetime(pc_all['available_at'])
        long = []
        months = pd.Index(sorted(ret['month'].unique()))
        for txt_y, grp in pc_all.groupby('text_year'):
            avail = pd.to_datetime(grp['available_at'].iloc[0])
            eff = months[months > avail.strftime('%Y-%m')]
            g = grp[['focal_id', 'pc', 'n_competitors']].copy()
            for mo in eff:
                gg = g.copy()
                gg['month'] = mo
                long.append(gg)
        longdf = pd.concat(long, ignore_index=True)
        nextr = ret.rename(columns={'firm_id': 'focal_id', 'R': 'R_next_full'}).copy()
        nextr['month'] = (pd.to_datetime(nextr['month'] + '-01') - pd.DateOffset(months=1)).dt.strftime('%Y-%m')
        full = longdf.merge(nextr, on=['focal_id', 'month'], how='inner')
        full['PC'] = full['pc'].astype(float)
        b1 = fm(full, 'R_next_full', ['PC'])
        variants['H_full_PC_universe'] = {
            'n_rows': int(len(full)), 'n_firms': int(full['focal_id'].nunique()),
            'n_months': int(full['month'].nunique()),
            'fm_univariate': {'beta_pct': pct(b1['coefs']['PC']['beta']), 't_nw': r4(b1['coefs']['PC']['t_nw']),
                              'n_months_used': b1['n_months_used']},
            'note': 'PC 全宇宙（无 pricing_gate 筛选、无分析师门、无微盘剔除、无控制要求）；收益 = monthly_returns_clean_v2 原始月收益'}
        psay('H4', f"变体 H_full_PC_universe 状态 = ok：n = {len(full):,}（{full['focal_id'].nunique()} 家）、"
                   f"β = {variants['H_full_PC_universe']['fm_univariate']['beta_pct']}%"
                   f"（t = {variants['H_full_PC_universe']['fm_univariate']['t_nw']}）")
    except Exception as e:  # noqa: BLE001
        variants['H_full_PC_universe'] = {'error': f'{type(e).__name__}: {e}'}
        psay('H4', f"变体 H 状态 = FAILED（{e}）")
    out['variants'] = variants
    dump('H4', out)
    psay('H4', '状态 = 完成')
    return out

def run_H5(m, S, fmonth) -> dict:
    psay('H5', '状态 = 开始：H5 期间（逐年 / 子期间 / PC 文本年队列）')
    reg, paper = S['registered'], S['paper_like']
    out = {'hypothesis': 'H5 期间 2018–2023 vs 原文 2007–2016（含子期间与逐年符号分布）'}

    def month_series_stats(betas_pct, seed=SEED, n_boot=5000):
        """REV2-F2：对逐月系数序列（%）并报 iid t、月块 bootstrap CI95、LOO 区间。"""
        a = np.asarray([v for v in betas_pct if v is not None and np.isfinite(v)], float)
        T = a.size
        if T < 2:
            return {}
        rng = np.random.default_rng(seed)
        boots = a[rng.integers(0, T, size=(n_boot, T))].mean(axis=1)
        loo = [float(np.delete(a, i).mean()) for i in range(T)]
        return {
            't_iid': r4(float(a.mean() / (a.std(ddof=1) / np.sqrt(T)))),
            'boot_ci95_pct': [r4(float(np.percentile(boots, 2.5))), r4(float(np.percentile(boots, 97.5)))],
            'boot_share_gt0': r4(float((boots > 0).mean())),
            'loo_beta_range_pct': [r4(min(loo)), r4(max(loo))],
            'n_boot': n_boot,
        }

    def by_blocks(samp, keys):
        res = {}
        for key, sub in samp.groupby(keys):
            if int(len(sub)) < 500:
                res[str(key)] = {'n_rows': int(len(sub)), 'note': 'rows<500'}
                continue
            b1 = fm(sub, 'R_next', ['PC'], nw_lag=NW)
            betas = np.array([r['beta']['PC'] for r in b1['_per_month'] if r['used']], float) * 100.0
            row = {'n_rows': int(len(sub)), 'n_months': int(sub['month'].nunique()),
                   'beta_pct': pct(b1['coefs']['PC']['beta']), 't_nw': r4(b1['coefs']['PC']['t_nw']),
                   'n_months_used': b1['n_months_used']}
            row.update(month_series_stats(betas))
            row['ci_includes_zero'] = bool(row.get('boot_ci95_pct') is None
                                           or (row['boot_ci95_pct'][0] <= 0 <= row['boot_ci95_pct'][1]))
            res[str(key)] = row
        return res

    out['by_return_year'] = {tag: by_blocks(s, s['month'].astype(str).str[:4])
                             for tag, s in (('registered', reg), ('paper_like', paper))}
    out['by_text_year_cohort'] = {tag: by_blocks(s, s['network_version'].astype(str).str[-4:])
                                  for tag, s in (('registered', reg), ('paper_like', paper))}
    sub = {}
    for tag, s in (('registered', reg), ('paper_like', paper)):
        for lab, lo, hi in (('2018-05..2020-12', '2018-05', '2020-12'), ('2021-01..2023-12', '2021-01', '2023-12')):
            ss = s[s['month'].between(lo, hi)]
            b = fm(ss, 'R_next', ['PC'])
            betas = np.array([r['beta']['PC'] for r in b['_per_month'] if r['used']], float) * 100.0
            row = {'n_rows': int(len(ss)), 'beta_pct': pct(b['coefs']['PC']['beta']),
                   't_nw': r4(b['coefs']['PC']['t_nw']), 'n_months_used': b['n_months_used']}
            row.update(month_series_stats(betas))
            row['ci_includes_zero'] = bool(row.get('boot_ci95_pct') is None
                                           or (row['boot_ci95_pct'][0] <= 0 <= row['boot_ci95_pct'][1]))
            sub[f'{tag}|{lab}'] = row
    out['subperiods'] = sub
    out['period_criterion'] = {
        'rule_rev2': 'REV2-F2：年度/子期间显著性判据 = 月块 bootstrap CI95（含 0 → 不显著）；'
                     'NW t 与 iid t 仅作并报参考（2019 月系数 ac1 = −0.557 → NW 方差 = iid/6.92 → t 被放大 ≈2.63×；'
                     '复核实测：t_nw = 2.92 vs t_iid = 1.06，月块 bootstrap CI95 [−0.2645, +0.9235] 含 0；'
                     '去 2019-09 后 t = 1.47）→ 不得以 NW t 主张"2019 显著为正"',
        'no_positive_evidence': True,
    }
    # 逐月系数序列的符号分布
    b1 = fm(reg, 'R_next', ['PC'])
    betas = np.array([r['beta']['PC'] for r in b1['_per_month'] if r['used']], dtype=float)
    out['monthly_beta_sign_distribution'] = {
        'n_months': int(betas.size), 'share_negative': r4(float((betas < 0).mean())),
        'mean_beta_pct': pct(float(betas.mean())), 'sd_beta_pct': pct(float(betas.std())),
        'min_pct': pct(float(betas.min())), 'max_pct': pct(float(betas.max())),
    }
    pos_years = [y for y, v in out['by_return_year']['registered'].items() if v.get('beta_pct', -1) > 0]
    out['positive_years_registered'] = pos_years
    out['positive_and_significant_years'] = [
        y for y, v in out['by_return_year']['registered'].items()
        if v.get('beta_pct', -1) is not None and v.get('beta_pct', -1) > 0 and v.get('ci_includes_zero') is False]
    dump('H5', out)
    psay('H5', f"状态 = 完成：登记样本逐年 β 符号 = "
               f"{ {y: v.get('beta_pct') for y, v in out['by_return_year']['registered'].items()} }；"
               f"逐月负号占比 = {out['monthly_beta_sign_distribution']['share_negative']}；"
               f"正号且 bootstrap CI 不含 0 的年 = {out['positive_and_significant_years'] or '无'}")
    return out

def run_H6(m, S, fmonth) -> dict:
    psay('H6', '状态 = 开始：H6 控制变量/规格对照（原文 12 控制 vs 我方 5 项；+行业 FE +团规模控制）')
    reg, pap = S['registered'], S['paper_like_ctrl']
    out = {'hypothesis': 'H6 控制集/规格差（含行业 FE、团规模机械相关）是否吸收掉 PC 系数'}
    # (1) 控制系数逐项对照
    out['control_coefs_vs_paper'] = {'paper_table4_col2': PAPER_CONTROLS}
    for tag, s in (('registered', reg), ('paper_like', pap)):
        out['control_coefs_vs_paper'][tag] = ctrl_compare(s, tag)['registered'] if tag == 'registered' else \
            ctrl_compare(s, tag)[tag]
    # 逐控制单变量（我方）
    uni = {}
    for c in ['size', 'bm', 'mom', 'rev', 'illiq']:
        b = fm(reg, 'R_next', [c])
        uni[c] = {'beta_pct': pct(b['coefs'][c]['beta']), 't_nw': r4(b['coefs'][c]['t_nw'])}
    out['univariate_controls_registered'] = uni
    psay('H6', f"控制对照状态 = ok：我方加控制后 size {out['control_coefs_vs_paper']['registered']['coefs']['size']}、"
               f"bm {out['control_coefs_vs_paper']['registered']['coefs']['bm']}（原文 +0.339 / −0.358）")

    # (2) 行业 FE（月 × 申万二级 组内去均值）
    merged_reg = reg.copy()
    merged_reg['sw2'] = sw2_asof(merged_reg['month'], merged_reg['focal_id'])
    cols = ['R_next', 'PC'] + CONTROLS
    dd = merged_reg.dropna(subset=cols + ['sw2']).copy()
    counts = dd.groupby(['month', 'sw2'])['R_next'].transform('size')
    dd = dd[counts >= 5]
    for c in cols:
        dd[f'd_{c}'] = dd[c] - dd.groupby(['month', 'sw2'])[c].transform('mean')
    b_ind = fm(dd, 'd_R_next', ['d_PC'], nw_lag=NW)
    b_ind2 = fm(dd, 'd_R_next', ['d_PC'] + [f'd_{c}' for c in CONTROLS], nw_lag=NW)
    out['industry_fe'] = {
        'spec': '月×申万二级 组内去均值（组内 ≥5 家）后 FM',
        'n_rows': int(len(dd)), 'n_months_used': b_ind['n_months_used'],
        'univariate': {'beta_pct': pct(b_ind['coefs']['d_PC']['beta']), 't_nw': r4(b_ind['coefs']['d_PC']['t_nw'])},
        'with_controls_5': {'beta_pct': pct(b_ind2['coefs']['d_PC']['beta']),
                            't_nw': r4(b_ind2['coefs']['d_PC']['t_nw'])},
    }
    psay('H6', f"行业 FE 状态 = ok：β = {out['industry_fe']['univariate']['beta_pct']}%"
               f"（t = {out['industry_fe']['univariate']['t_nw']}）")

    # (3) 团规模控制：PC + log(1+团员数) [+ 控制]；REV2-F3 补 1/N 线性控制（函数形式稳健性）
    ms = reg.copy()
    ms['log_peers'] = np.log1p(ms['n_competitors'].astype(float))
    ms['inv_peers'] = 1.0 / ms['n_competitors'].astype(float)
    b_sz = fm(ms, 'R_next', ['PC', 'log_peers'])
    b_sz2 = fm(ms, 'R_next', ['PC', 'log_peers'] + CONTROLS)
    b_sz_only = fm(ms, 'R_next', ['log_peers'])
    b_inv = fm(ms, 'R_next', ['PC', 'inv_peers'])
    b_inv2 = fm(ms, 'R_next', ['PC', 'inv_peers'] + CONTROLS)
    out['cluster_size_control'] = {
        'log_peers_only': {'beta_pct': pct(b_sz_only['coefs']['log_peers']['beta']),
                           't_nw': r4(b_sz_only['coefs']['log_peers']['t_nw'])},
        'PC_plus_log_peers': {'PC_beta_pct': pct(b_sz['coefs']['PC']['beta']), 'PC_t': r4(b_sz['coefs']['PC']['t_nw']),
                              'log_peers_beta_pct': pct(b_sz['coefs']['log_peers']['beta']),
                              'log_peers_t': r4(b_sz['coefs']['log_peers']['t_nw'])},
        'PC_plus_log_peers_plus_controls': {
            'PC_beta_pct': pct(b_sz2['coefs']['PC']['beta']), 'PC_t': r4(b_sz2['coefs']['PC']['t_nw'])},
        'PC_plus_inv_peers': {'PC_beta_pct': pct(b_inv['coefs']['PC']['beta']), 'PC_t': r4(b_inv['coefs']['PC']['t_nw']),
                              'inv_peers_beta_pct': pct(b_inv['coefs']['inv_peers']['beta']),
                              'inv_peers_t': r4(b_inv['coefs']['inv_peers']['t_nw'])},
        'PC_plus_inv_peers_plus_controls': {
            'PC_beta_pct': pct(b_inv2['coefs']['PC']['beta']), 'PC_t': r4(b_inv2['coefs']['PC']['t_nw'])},
        'form_dependence_note': 'REV2-F3：「团规模控制使 PC 转正」**仅限 log 形式**（log1p(N) → '
                                f"{pct(b_sz['coefs']['PC']['beta'])}%/t={r4(b_sz['coefs']['PC']['t_nw'])}）；"
                                f"1/N 形式仍为负（本 H6 变体 = 1/n_competitors（CSV 覆盖口径）"
                                f"{pct(b_inv['coefs']['PC']['beta'])}%/t={r4(b_inv['coefs']['PC']['t_nw'])}；"
                                f"H8 §8R 的团成员数口径 1/N = −0.6302%/t=−0.8076，均不显著）"
                                '；两者均不显著 → 不足以改判 H8',
    }
    # REV2-F3：控制项符号与原文一致性计数（原文表 4 列(2)，5 项）
    cco = out['control_coefs_vs_paper']['registered']['coefs']
    sign_rows = {}
    n_same = n_opp = 0
    for c in CONTROLS:
        ours = cco.get(c, {}).get('beta_pct')
        paper = PAPER_CONTROLS[c][0]
        if ours is None:
            continue
        same = (ours > 0) == (paper > 0)
        n_same += int(same)
        n_opp += int(not same)
        sign_rows[c] = {'paper_pct': paper, 'ours_pct': ours, 'same_sign': bool(same)}
    out['paper_sign_agreement'] = {
        'n_controls': len(sign_rows), 'n_same_sign': n_same, 'n_opposite_sign': n_opp,
        'opposite_vars': [c for c, r in sign_rows.items() if not r['same_sign']],
        'same_vars': [c for c, r in sign_rows.items() if r['same_sign']],
        'rows': sign_rows,
        'note': 'REV2-F3：符号相反仅 3/5 控制项（size/bm/rev），mom 与 illiq 同号 → 表述须限定',
    }
    psay('H6', f"团规模控制状态 = ok：PC+log团员 β_PC = {out['cluster_size_control']['PC_plus_log_peers']['PC_beta_pct']}%"
               f"（t = {out['cluster_size_control']['PC_plus_log_peers']['PC_t']}）、"
               f"PC+(1/N) β_PC = {out['cluster_size_control']['PC_plus_inv_peers']['PC_beta_pct']}%"
               f"（t = {out['cluster_size_control']['PC_plus_inv_peers']['PC_t']}）、"
               f"控制项同号 {n_same}/5（相反 = {'/'.join(out['paper_sign_agreement']['opposite_vars'])}）")
    dump('H6', out)
    psay('H6', '状态 = 完成')
    return out

def run_H7(m, S, fmonth, panel) -> dict:
    psay('H7', '状态 = 开始：H7 收益口径/横截面结构（raw vs α / 等权 vs 市值加权 / winsorize / 规模分组 / 持有期）')
    reg = S['registered'].copy()
    out = {'hypothesis': 'H7 收益口径（raw vs 因子调整、EW/VW、winsorize、规模分组、持有期 t+1 vs t+2..t+6/t+7..t+12）'}
    fmn = fmonth.set_index('month')
    reg['mk'] = reg['month'].map(fmn['MKT'])
    reg['rfm'] = reg['month'].map(fmn['RF'])
    reg['R_exc'] = reg['R_next'] - reg['rfm']
    reg['R_mkt'] = reg['R_next'] - reg['mk']
    reg['R_ind'] = reg['R_next'] - reg['ind_l2_ret_vw']
    yvars = {'raw_winsorized': 'R_next', 'raw_unwinsorized_next_R': 'next_R',
             'excess_R_minus_RF': 'R_exc', 'market_adj_R_minus_MKT': 'R_mkt',
             'industry_adj_R_minus_IND2vw': 'R_ind', 'postadj_后复权': 'R_next_postadj'}
    res = {}
    for tag, yc in yvars.items():
        for wtag, wcol in (('EW', None), ('VW', 'market_equity')):
            b = fm(reg, yc, ['PC'], weight_col=wcol)
            res[f'{tag}|{wtag}'] = {'n_rows': None, 'beta_pct': pct(b['coefs']['PC']['beta']),
                                    't_nw': r4(b['coefs']['PC']['t_nw']), 'n_months_used': b['n_months_used']}
    out['y_variants'] = res
    psig = [k for k, v in res.items() if v['beta_pct'] is not None and v['beta_pct'] > 0]
    out['positive_specs'] = psig
    psay('H7', f"收益口径状态 = ok：raw EW {res['raw_winsorized|EW']['beta_pct']}%、"
               f"excess {res['excess_R_minus_RF|EW']['beta_pct']}%、"
               f"market-adj {res['market_adj_R_minus_MKT|EW']['beta_pct']}%、"
               f"行业调整 {res['industry_adj_R_minus_IND2vw|EW']['beta_pct']}%、"
               f"后复权 {res['postadj_后复权|EW']['beta_pct']}%、VW {res['raw_winsorized|VW']['beta_pct']}%")

    # winsorize 变体
    for lo, hi in ((0.025, 0.975), (0.005, 0.995)):
        s = reg.copy()
        s['w'] = s.groupby('month')['next_R'].transform(
            lambda x: x.clip(x.quantile(lo), x.quantile(hi)))
        b = fm(s, 'w', ['PC'])
        res[f'winsor_{lo}_{hi}_on_next_R|EW'] = {'beta_pct': pct(b['coefs']['PC']['beta']),
                                                 't_nw': r4(b['coefs']['PC']['t_nw'])}
    out['y_variants'] = res

    # 规模三分位内
    s = reg.copy()
    s['sz'] = s['size'].astype(float)
    s['ter'] = s.groupby('month')['sz'].transform(
        lambda x: pd.qcut(x.rank(method='first'), 3, labels=['small', 'mid', 'large']))
    ter = {}
    for t, sub in s.groupby('ter'):
        b = fm(sub, 'R_next', ['PC'])
        ter[str(t)] = {'n_rows': int(len(sub)), 'beta_pct': pct(b['coefs']['PC']['beta']),
                       't_nw': r4(b['coefs']['PC']['t_nw'])}
    out['size_tercile'] = ter
    psay('H7', f"规模三分位状态 = ok：{ {k: v['beta_pct'] for k, v in ter.items()} }")

    # 持有期：t+1 / t+2..t+6 / t+7..t+12（累计复利，收益来自 monthly_returns_clean_v2）
    ret = pd.read_csv(RETURNS, usecols=['firm_id', 'month', 'R'], dtype={'firm_id': str})
    ret['ym'] = pd.PeriodIndex(ret['month'], freq='M')
    piv = ret.pivot_table(index='ym', columns='firm_id', values='R')
    T_, N_ = piv.shape
    A = piv.to_numpy(float)
    valid = ~np.isnan(A)
    CP = np.cumprod(np.where(valid, 1 + A, 1.0), axis=0)       # 1+ 累计（缺月视作 1）
    CV = valid.cumsum(axis=0)                                  # 有效月计数
    mi = {k: i for i, k in enumerate(piv.index)}
    fi = {k: j for j, k in enumerate(piv.columns)}

    def _window_ret(fids, months, off, k):
        """months (m+off) .. (m+off+k-1) 累计收益；窗内任一缺月 → NaN（全向量化）"""
        idxm = None
        out_v = np.full(len(fids), np.nan)
        i_row = np.array([mi.get(pd.Period(m, freq='M'), -1) for m in months])
        j_col = np.array([fi.get(f, -1) for f in fids])
        ok = (i_row >= 0) & (j_col >= 0)
        lo = i_row + off
        hi = i_row + off + k - 1
        ok &= (hi < T_) & (lo >= 0)
        cnt = np.zeros(len(fids))
        cnt[ok] = CV[hi[ok], j_col[ok]] - np.where(lo[ok] > 0, CV[lo[ok] - 1, j_col[ok]], 0)
        full = ok & (cnt == k)
        num = np.ones(len(fids))
        num[full] = CP[hi[full], j_col[full]]
        den = np.ones(len(fids))
        nz = full & (lo > 0)
        den[nz] = CP[lo[nz] - 1, j_col[nz]]
        out_v[full] = num[full] / den[full] - 1.0
        return out_v

    base = reg[['focal_id', 'month', 'PC', 'R_next']].copy()
    months = base['month'].tolist()
    fids = base['focal_id'].tolist()
    for tag, k, off in (('h_t+1', 1, 1), ('h_t+2..t+6', 5, 2), ('h_t+7..t+12', 6, 7),
                        ('h_t+1..t+6', 6, 1), ('h_t+1..t+12', 12, 1)):
        base[tag] = _window_ret(fids, months, off, k)
    hor = {}
    win_months = {'h_t+1': 1, 'h_t+2..t+6': 5, 'h_t+7..t+12': 6, 'h_t+1..t+6': 6, 'h_t+1..t+12': 12}
    for tag in ('h_t+1', 'h_t+2..t+6', 'h_t+7..t+12', 'h_t+1..t+6', 'h_t+1..t+12'):
        ss = base[base[tag].notna()]
        b = fm(ss.assign(y=ss[tag] * 100.0), 'y', ['PC'])
        beta = r4(b['coefs']['PC']['beta'])
        hor[tag] = {'n_rows': int(len(ss)), 'n_months_used': b['n_months_used'],
                    'beta_pct_per_unit_period': beta,
                    'window_months': win_months[tag],
                    'beta_pct_per_unit_per_month_avg': r4(beta / win_months[tag]) if beta is not None else None,
                    't_nw': r4(b['coefs']['PC']['t_nw'])}
    out['holding_horizons'] = {**hor, 'note': '收益 = monthly_returns_clean_v2 月度收益率复利；'
                                              'β 单位 = %/单位 PC/持有期（REV2-F5 另报月均折算 = β ÷ 窗长）；'
                                              '窗内任一月缺收益 → 该行剔除'}
    out['holding_horizons_rev2'] = {
        'monthly_avg_readout': {tag: hor[tag]['beta_pct_per_unit_per_month_avg'] for tag in hor},
        'reading': 'REV2-F5：t+1 −0.5707%/月、t+2..6 −3.3034/5 ≈ −0.6607%/月、t+7..12 −4.7509/6 ≈ −0.7918%/月 '
                   '→ 折算月均后幅度基本不变（无反转，也无随持有期加深而衰减/放大的趋势）；'
                   't+2..6 / t+7..12 / t+1..6 / t+1..12 为**重叠窗**且 NW(6) 未做重叠（重合样本）校正 → '
                   '各自 t 值不作推断依据（仅登记点估计）。',
    }
    psay('H7', f"持有期状态 = ok：{ {k: v['beta_pct_per_unit_period'] for k, v in hor.items()} }")

    # 组合 α3/α4（十分位/五分位 × EW/VW）
    ports = {}
    for gtag, ng in (('quintile', 5), ('decile', 10)):
        for wtag, wcol in (('EW', None), ('VW', 'market_equity')):
            ports[f'{gtag}_{wtag}'] = portfolio_key(reg, 'PC', ng, wtag, wcol, fmonth=fmonth)
    out['portfolios_alpha'] = ports
    psay('H7', f"组合 α 状态 = ok：五分位 EW α3 = {ports['quintile_EW']['alpha_a3_pct']}%"
               f"（t = {ports['quintile_EW']['alpha_a3_t']}）、十分位 VW α3 = {ports['decile_VW']['alpha_a3_pct']}%"
               f"（t = {ports['decile_VW']['alpha_a3_t']}）")
    dump('H7', out)
    psay('H7', '状态 = 完成')
    return out

def run_H8(m, S, panel, fmonth) -> dict:
    psay('H8', '状态 = 开始：H8 机械/结构（PC 与团规模/收入离散度/行业；残差检验；年度持续性）')
    reg = S['registered'].copy()
    out = {'hypothesis': 'H8 PC = −HHI 与团规模/团内收入离散度/行业机械相关；残差定价与持续性'}

    # (1) 团层结构量（从 clusters + 营收现算）
    cl_all, prop = [], []
    rev_cache = {}
    for y in PC_YEARS:
        cl = pd.read_csv(CLUSTER_DIR / f'{y}_clusters.csv', dtype={'firm_id': str})
        rv = rev_cache.setdefault(y - 1, load_rev(y - 1))
        sz = cl.groupby('cluster_id').size()
        cl['n_mem'] = cl['cluster_id'].map(sz)
        cl['rev'] = cl['firm_id'].map(lambda f: rv.get(f[2:]))
        tot = cl.groupby('cluster_id')['rev'].transform('sum')
        cl['own_share'] = cl['rev'] / tot
        agg = cl.groupby('cluster_id').agg(n_mem=('firm_id', 'size'), rev_tot=('rev', 'sum'),
                                           rev_mean=('rev', 'mean'), rev_std=('rev', 'std'))
        agg['cv'] = agg['rev_std'] / agg['rev_mean']
        agg['hhi_cluster'] = cl.groupby('cluster_id').apply(
            lambda v: float(((v['rev'] / v['rev'].sum()) ** 2).sum()), include_groups=False)
        agg['text_year'] = y
        agg = agg.reset_index()
        cl_all.append(agg)
        prop.append(cl[['firm_id', 'cluster_id', 'own_share', 'n_mem']].assign(text_year=y))
    cagg = pd.concat(cl_all, ignore_index=True)
    cprop = pd.concat(prop, ignore_index=True)

    d = reg.merge(cprop.rename(columns={'firm_id': 'focal_id', 'cluster_id': 'cluster_id_p'}),
                  on=['focal_id', 'text_year'], how='left')
    d = d.merge(cagg[['text_year', 'cluster_id', 'cv', 'rev_tot', 'rev_mean']]
                .rename(columns={'cluster_id': 'cluster_id_p'}),
                on=['text_year', 'cluster_id_p'], how='left')
    d['log_mem'] = np.log(d['n_mem'].astype(float))
    dd = d.dropna(subset=['PC', 'log_mem', 'own_share', 'cv'])
    out['pc_structure_corr'] = {
        'n_rows': int(len(dd)),
        'corr_PC_log_cluster_size': r4(float(np.corrcoef(dd['PC'], dd['log_mem'])[0, 1])),
        'corr_PC_own_revenue_share': r4(float(np.corrcoef(dd['PC'], dd['own_share'])[0, 1])),
        'corr_PC_cluster_rev_cv': r4(float(np.corrcoef(dd['PC'], dd['cv'])[0, 1])),
        'corr_PC_log_rev_total': r4(float(np.corrcoef(dd['PC'], np.log(dd['rev_tot'].clip(lower=1)))[0, 1])),
        'corr_HHI_1overN': r4(float(np.corrcoef(dd['hhi'], 1.0 / dd['n_mem'])[0, 1])),
        'pc_group_by_size': None,
    }
    g = dd.groupby(pd.qcut(dd['log_mem'].rank(method='first'), 5, labels=[f'S{i}' for i in range(1, 6)]))
    out['pc_structure_corr']['pc_group_by_size'] = {
        str(k): {'n': int(len(v)), 'pc_mean': r4(float(v['PC'].mean())), 'hhi_mean': r4(float(v['hhi'].mean())),
                 'own_share_mean': r4(float(v['own_share'].mean())),
                 'peer_median': float(v['n_mem'].median())} for k, v in g}
    psay('H8', f"团结构相关状态 = ok：corr(PC, log 团员数) = {out['pc_structure_corr']['corr_PC_log_cluster_size']}、"
               f"corr(PC, 自身收入份额) = {out['pc_structure_corr']['corr_PC_own_revenue_share']}、"
               f"corr(PC, 团内收入 CV) = {out['pc_structure_corr']['corr_PC_cluster_rev_cv']}")

    # (2) 行业 HHI 相关（申万二级，营收 HHI）
    sw = sw2_asof(d['month'], d['focal_id'])
    d['sw2'] = sw
    ih = []
    for y in PC_YEARS:
        rv = rev_cache.get(y - 1) or load_rev(y - 1)
        mp = pd.read_csv(SW2_MAP, dtype={'code6': str})
        mp = mp[mp['is_current_version'].astype(bool)]
        codes = {str(c).zfill(6): str(l) for c, l in zip(mp['code6'], mp['l2_code'])}
        rows = [{'code6': c, 'sw2': codes.get(c), 'rev': r} for c, r in rv.items() if codes.get(c)]
        t = pd.DataFrame(rows)
        t['tot'] = t.groupby('sw2')['rev'].transform('sum')
        h = t.groupby('sw2').apply(lambda v: float(((v['rev'] / v['tot']) ** 2).sum()), include_groups=False)
        ih.append(h.rename(f'hhi_ind_{y}'))
    ind = pd.concat(ih, axis=1)
    d['ind_hhi'] = [ind.loc[str(int(s)), f'hhi_ind_{int(y)}'] if (pd.notna(s) and str(int(s)) in ind.index) else np.nan
                    for s, y in zip(d['sw2'], d['text_year'])]
    di = d.dropna(subset=['PC', 'ind_hhi'])
    out['industry_hhi'] = {
        'n_rows': int(len(di)), 'corr_PC_indHHI_neg': r4(float(np.corrcoef(di['PC'], -di['ind_hhi'])[0, 1])),
        'corr_PC_log_cluster_size_ref': out['pc_structure_corr']['corr_PC_log_cluster_size'],
    }
    psay('H8', f"行业相关状态 = ok：corr(PC, −行业HHI) = {out['industry_hhi']['corr_PC_indHHI_neg']}")

    # (3) 残差检验：PC ~ 团规模（+自身份额）取残差 → FM
    dd2 = dd.copy()
    X = np.column_stack([np.ones(len(dd2)), dd2['log_mem'], dd2['own_share']])
    beta = np.linalg.lstsq(X, dd2['PC'].to_numpy(float), rcond=None)[0]
    dd2['PC_resid'] = dd2['PC'].to_numpy(float) - X @ beta
    out['residual_test'] = {
        'spec': 'PC_resid = PC − fitted(1, log 团员数, 自身收入份额)（池化 OLS）',
        'loadings': {'const': r4(beta[0]), 'log_mem': r4(beta[1]), 'own_share': r4(beta[2])},
        'r2': r4(float(1 - np.var(dd2['PC_resid']) / np.var(dd2['PC']))),
    }
    for tag, xc in (('PC_resid', 'PC_resid'), ('PC_raw_same_rows', 'PC'),
                    ('own_share', 'own_share'), ('log_mem_only', 'log_mem')):
        b = fm(dd2, 'R_next', [xc])
        out['residual_test'][f'fm_{tag}'] = {'beta_pct': pct(b['coefs'][xc]['beta']),
                                             't_nw': r4(b['coefs'][xc]['t_nw'])}
    b = fm(dd2, 'R_next', ['PC_resid', 'own_share', 'log_mem'])
    out['residual_test']['fm_joint'] = {c: {'beta_pct': pct(b['coefs'][c]['beta']), 't_nw': r4(b['coefs'][c]['t_nw'])}
                                        for c in ('PC_resid', 'own_share', 'log_mem')}
    psay('H8', f"残差检验状态 = ok：FM(PC_resid) = {out['residual_test']['fm_PC_resid']['beta_pct']}%"
               f"（t = {out['residual_test']['fm_PC_resid']['t_nw']}）；R²（PC~团规模+自身份额）= "
               f"{out['residual_test']['r2']}")

    # (4) 年度持续性
    p = m.loc[m['PC'].notna(), ['focal_id', 'text_year', 'PC', 'hhi', 'n_competitors']].drop_duplicates(
        ['focal_id', 'text_year'])
    p['ty'] = p['text_year'].astype(int)
    pairs = []
    for y in range(2017, 2022):
        a = p[p['ty'] == y].set_index('focal_id')
        b2 = p[p['ty'] == y + 1].set_index('focal_id')
        j = a[['PC', 'n_competitors']].join(b2[['PC', 'n_competitors']], lsuffix='_y', rsuffix='_y1', how='inner')
        if len(j):
            pairs.append({'y': y, 'n': int(len(j)),
                          'corr_PC': r4(float(np.corrcoef(j['PC_y'], j['PC_y1'])[0, 1])),
                          'corr_log_size': r4(float(np.corrcoef(np.log1p(j['n_competitors_y']),
                                                                np.log1p(j['n_competitors_y1']))[0, 1]))})
    allp = p.sort_values('ty').groupby('focal_id').filter(lambda g: len(g) > 1)
    j = allp.groupby('focal_id')[['PC']].shift(-1)
    ok = allp['PC'].notna() & j['PC'].notna()
    ar1 = np.corrcoef(allp.loc[ok, 'PC'], j.loc[ok, 'PC'])[0, 1]
    out['persistence'] = {'by_year_pairs': pairs, 'pooled_AR1_corr': r4(float(ar1)),
                          'mean_pc_change_abs': r4(float((allp.loc[ok, 'PC'] - j.loc[ok, 'PC']).abs().mean()))}
    psay('H8', f"持续性状态 = ok：池化 AR(1) corr = {out['persistence']['pooled_AR1_corr']}、"
               f"逐年 = {[x['corr_PC'] for x in pairs]}")

    # ============================== REV2（F1）：撤回原结论 + 复核指定替代检验（独立重算） ==============================
    rev = {'status': '待定/未被识别（原「根因 #1 = PC 是团结构代理」结论撤回）'}
    rev['withdrawal'] = {
        'retracted_conclusion': '原结论「PC 的横截面信息 = 团结构代理（自身收入份额 + 团规模），非独立"竞争强度"；'
                                '残差化后负号消失」（H8 结构层 + H6 团规模控制互证）**撤回**；'
                                '该表述对处理变量 PC 的**定义性结构分量**回归取残差 = 过度控制（bad control / collider），'
                                '且结论对函数形式不稳健 → 改判**待定 / 未被识别**。',
        'review_rebuttals_verbatim': [
            '【反驳 1 · 过度控制/bad control】对处理变量的定义性结构分量回归取残差＝过度控制/'
            'collider：PC = −HHI 是团内份额的确定性函数，把「团规模 / 自身份额」当控制变量会把 PC 的'
            '真实横截面变异一并剔除，残差系数不能解释为「PC 的独立定价效应」。',
            '【反驳 2 · 函数形式不稳健】log 形式（log1p(N)）控制下 PC 转正 +0.3221（t = 0.4488），'
            '但 1/N 线性形式仍为负 −0.6302（t = −0.8076）→ 符号随机械控制的函数形式翻转，'
            '「使负号消失的口径」不唯一。',
            '【反驳 3 · 恒等式两分量各自都为负】PC = −own_share² + (−同团他人集中度)，两分量单变量 FM '
            '均为负：FM(−own_share²) = −0.5354（t = −1.2383）、FM(−他人集中度) = −0.4446（t = −1.1232）'
            '→ 不存在「某个结构性分量吸收负号」的证据。',
        ],
    }

    # ---- (a) 恒等式分解：PC = −own_share² + (−(hhi − own_share²))（同团他人集中度）
    di = reg.merge(cprop.rename(columns={'firm_id': 'focal_id', 'cluster_id': 'cluster_id_p'}),
                   on=['focal_id', 'text_year'], how='left')
    di = di.dropna(subset=['PC', 'own_share', 'hhi']).copy()
    di['t_own'] = -(di['own_share'] ** 2)
    di['t_peer'] = -(di['hhi'] - di['own_share'] ** 2)
    ident = {
        'spec': 'PC = (−own_share²) + (−(HHI − own_share²))；own_share = 自身营收 / 团内营收合计（文本年 −1 年）',
        'n_rows': int(len(di)),
        'max_abs_resid_identity': float((di['t_own'] + di['t_peer'] - di['PC']).abs().max()),
        'corr_two_components': r4(float(np.corrcoef(di['t_own'], di['t_peer'])[0, 1])),
        'descriptives': {c: {'mean': r4(float(di[c].mean())), 'sd': r4(float(di[c].std()))}
                         for c in ('PC', 't_own', 't_peer')},
    }
    for tag, xc in (('t_own', 'FM(−own_share²)'), ('t_peer', 'FM(−同团他人集中度)')):
        b = fm(di, 'R_next', [tag])
        ident[tag] = {'beta_pct': pct(b['coefs'][tag]['beta']), 't_nw': r4(b['coefs'][tag]['t_nw']),
                      'label': xc}
    b = fm(di, 'R_next', ['t_own', 't_peer'])
    ident['fm_joint'] = {c: {'beta_pct': pct(b['coefs'][c]['beta']), 't_nw': r4(b['coefs'][c]['t_nw'])}
                         for c in ('t_own', 't_peer')}
    rev['identity_decomposition'] = ident
    psay('H8R', f"恒等式分解状态 = ok：corr(两分量) = {ident['corr_two_components']}；"
                f"FM(−own²) = {ident['t_own']['beta_pct']}%（t = {ident['t_own']['t_nw']}）、"
                f"FM(−他人集中度) = {ident['t_peer']['beta_pct']}%（t = {ident['t_peer']['t_nw']}）")

    # ---- (b) 机械控制函数形式：1/N（线性）替代 log N
    di['invN'] = 1.0 / di['n_mem'].astype(float)
    cvar = {}
    for xc in (['PC', 'invN'], ['PC', 'invN', 'own_share'], ['PC', 'own_share']):
        b = fm(di, 'R_next', xc)
        cvar['+'.join(xc)] = {c: {'beta_pct': pct(b['coefs'][c]['beta']), 't_nw': r4(b['coefs'][c]['t_nw'])}
                              for c in xc}
    rev['control_function_form'] = {
        'variants': cvar,
        'reading': 'REV2-F1(b)：机械控制换 1/N 后 PC 仍为负（PC+1/N = '
                   f"{cvar['PC+invN']['PC']['beta_pct']}%/t = {cvar['PC+invN']['PC']['t_nw']}；"
                   f"PC+1/N+own_share = {cvar['PC+invN+own_share']['PC']['beta_pct']}%/t = "
                   f"{cvar['PC+invN+own_share']['PC']['t_nw']}）→ 与 log 形式的 +0.3221 相反 → 函数形式不稳健。",
    }
    psay('H8R', f"1/N 控制状态 = ok：PC+1/N = {cvar['PC+invN']['PC']['beta_pct']}%、"
                f"PC+1/N+own_share = {cvar['PC+invN+own_share']['PC']['beta_pct']}%（log 形式为 +0.3221）")

    # ---- (c) within-N 分箱：团规模内部的 PC 梯度与样本占比
    dnb = di.dropna(subset=['n_mem']).copy()
    bins = [0, 1, 2, 3, 5, 10, 20, 50, 10 ** 9]
    labs = ['N=1', 'N=2', 'N=3', 'N=4-5', 'N=6-10', 'N=11-20', 'N=21-50', 'N>50']
    dnb['nbin'] = pd.cut(dnb['n_mem'], bins=bins, labels=labs, right=True)
    wn = {}
    for lab in labs:
        sub = dnb[dnb['nbin'] == lab]
        if len(sub) == 0:
            continue
        rec_w = {'n': int(len(sub)), 'sample_share': r4(float(len(sub)) / len(dnb)),
                 'median_N': float(sub['n_mem'].median()), 'pc_mean': r4(float(sub['PC'].mean())),
                 'pc_sd': r4(float(sub['PC'].std()))}
        if sub['PC'].std() == 0:
            rec_w['note'] = 'PC 无队内变异（N=1 → PC ≡ −1）→ 不跑 FM'
        else:
            b = fm(sub, 'R_next', ['PC'], min_n=20)
            rec_w.update({'beta_pct': pct(b['coefs']['PC']['beta']), 't_nw': r4(b['coefs']['PC']['t_nw']),
                          'n_months_used': b['n_months_used'], 'min_n': 20})
        wn[str(lab)] = rec_w
    rev['within_N_bins'] = {'bins': wn, 'min_n_note': '小 N 档月内样本 < 100 → min_n 放宽到 20（已标注）；'
                                                     'N=1 档 PC 常数、不跑 FM'}
    psay('H8R', 'within-N 分箱状态 = ok：' + '；'.join(
        f"{k}: β={v.get('beta_pct')}（t={v.get('t_nw')}, share={v['sample_share']}）" for k, v in wn.items()))

    # ---- (d) 分布匹配重加权（原文表 2 PC 分位密度 → 重要性重加权 + 分位匹配）
    c25, c50, c75 = PAPER_T2['p25'], PAPER_T2['median'], PAPER_T2['p75']
    fy = m.loc[m['PC'].notna(), ['focal_id', 'text_year', 'PC']].drop_duplicates(['focal_id', 'text_year']).copy()
    fb = np.digitize(fy['PC'].to_numpy(float), [c25, c50, c75])
    obs_share = np.array([float((fb == k).mean()) for k in range(4)])
    w_bin = np.where(obs_share > 0, 0.25 / np.maximum(obs_share, 1e-12), 0.0)
    fy['w_dist'] = w_bin[fb]
    fy['w_dist_cap10'] = np.minimum(fy['w_dist'], 10.0)
    # 分位匹配：秩 → 原文分位函数（分段线性，锚点 = p25/p50/p75；端点区间按相邻斜率外推并截断 r∈[0.01,0.99]）
    r = fy['PC'].rank(pct=True, method='average').to_numpy(float)
    sL = (c50 - c25) / 0.25
    sR = (c75 - c50) / 0.25
    rr = np.clip(r, 0.01, 0.99)
    pc_qm = np.where(rr <= 0.25, c25 - (0.25 - rr) * sL,
                     np.where(rr <= 0.50, c25 + (rr - 0.25) / 0.25 * (c50 - c25),
                              np.where(rr <= 0.75, c50 + (rr - 0.50) / 0.25 * (c75 - c50),
                                       c75 + (rr - 0.75) * sR)))
    fy['PC_qm'] = pc_qm
    dist = {'paper_table2': PAPER_T2, 'observed_firm_year_bins': {
        'cutpoints': [c25, c50, c75],
        'obs_shares': [r4(float(x)) for x in obs_share],
        'target_shares': [0.25, 0.25, 0.25, 0.25],
        'bin_weights': [r4(float(x)) for x in w_bin],
        'n_firm_year': int(len(fy)),
    }}
    for tag, s in (('registered', reg), ('paper_like', S['paper_like'])):
        merged_fy = s.merge(fy[['focal_id', 'text_year', 'w_dist', 'w_dist_cap10', 'PC_qm']],
                     on=['focal_id', 'text_year'], how='left')
        for wtag in ('w_dist', 'w_dist_cap10'):
            b = fm(merged_fy, 'R_next', ['PC'], weight_col=wtag)
            dist[f'{tag}|{wtag}'] = {'beta_pct': pct(b['coefs']['PC']['beta']),
                                     't_nw': r4(b['coefs']['PC']['t_nw']), 'n_months_used': b['n_months_used']}
        b2 = fm(merged_fy, 'R_next', ['PC'] + CONTROLS, weight_col='w_dist')
        dist[f'{tag}|w_dist+controls'] = {'PC_beta_pct': pct(b2['coefs']['PC']['beta']),
                                          'PC_t': r4(b2['coefs']['PC']['t_nw'])}
        b3 = fm(merged_fy, 'R_next', ['PC_qm'])
        dist[f'{tag}|quantile_match'] = {'beta_pct': pct(b3['coefs']['PC_qm']['beta']),
                                         't_nw': r4(b3['coefs']['PC_qm']['t_nw']),
                                         'n_months_used': b3['n_months_used']}
        p_qm = portfolio_key(merged_fy, 'PC_qm', 5, 'EW', None, fmonth=fmonth)
        dist[f'{tag}|quantile_match|quintile_EW'] = {'HL_raw_pct': p_qm.get('HL_raw_pct'),
                                                     'HL_raw_t': p_qm.get('HL_raw_t_nw'),
                                                     'alpha_a3_pct': p_qm.get('alpha_a3_pct'),
                                                     'alpha_a3_t': p_qm.get('alpha_a3_t')}
    dist['transformed_distribution'] = {
        'PC_qm': {'mean': r4(float(fy['PC_qm'].mean())), 'median': r4(float(fy['PC_qm'].median())),
                  'p25': r4(float(fy['PC_qm'].quantile(.25))), 'p75': r4(float(fy['PC_qm'].quantile(.75))),
                  'sd': r4(float(fy['PC_qm'].std()))},
        'weight_summary': {'w_dist_mean': r4(float(fy['w_dist'].mean())), 'w_dist_max': r4(float(fy['w_dist'].max())),
                           'w_dist_min': r4(float(fy['w_dist'].min()))},
        'note': '分位匹配 = 逐 firm-year 秩 → 原文分段线性分位函数（锚 p25/p50/p75），保序单调变换；'
                '重要性重加权 = 原文四分位份额(0.25×4)/我方实测份额 → WLS（另报 10 上限截断变体）',
    }
    rev['distribution_reweighting'] = dist
    psay('H8R', f"分布重加权状态 = ok：firm-year n = {len(fy)}、观测四分位份额 = {[r4(float(x)) for x in obs_share]}、"
                f"登记样本 WLS β = {dist['registered|w_dist']['beta_pct']}%（t = {dist['registered|w_dist']['t_nw']}）、"
                f"分位匹配 β = {dist['registered|quantile_match']['beta_pct']}%"
                f"（t = {dist['registered|quantile_match']['t_nw']}）")

    # ---- (e) 团标签置换零分布（1000 次；逐年重贴「团 → HHI」对应）
    N_ITER = 1000
    rng = np.random.default_rng(SEED)
    dp = dd.dropna(subset=['cluster_id_p']).copy()
    dp['cluster_id_p'] = dp['cluster_id_p'].astype(int)
    lut = {}
    for y, g in cagg.groupby('text_year'):
        uids = np.sort(g['cluster_id'].astype(int).to_numpy())
        vals = g.set_index('cluster_id').loc[uids, 'hhi_cluster'].to_numpy(float)
        lut[int(y)] = (uids, vals)
    row_ty = dp['text_year'].astype(int).to_numpy()
    row_cid = dp['cluster_id_p'].to_numpy()
    sel_by_year, idx_by_year = {}, {}
    for y, (uids, _) in lut.items():
        sel = row_ty == y
        sel_by_year[y] = sel
        idx_by_year[y] = np.searchsorted(uids, row_cid[sel])
    obs_beta = pct(fm(dp, 'R_next', ['PC'])['coefs']['PC']['beta'])
    perm = []
    for _ in range(N_ITER):
        pcm = np.full(len(dp), np.nan)
        for y, (_, vals) in lut.items():
            sel = sel_by_year[y]
            pcm[sel] = -rng.permutation(vals)[idx_by_year[y]]
        pp = dp.assign(PC_perm=pcm).dropna(subset=['PC_perm'])
        perm.append(pct(fm(pp, 'R_next', ['PC_perm'])['coefs']['PC_perm']['beta']))
    perm_a = np.asarray(perm, float)
    p_le = float((perm_a <= obs_beta).mean())
    p_two = float((np.abs(perm_a) >= abs(obs_beta)).mean())
    rev['cluster_label_permutation'] = {
        'n_iter': N_ITER, 'n_rows_used': int(len(dp)), 'observed_beta_pct': obs_beta,
        'perm_beta_mean_pct': r4(float(perm_a.mean())), 'perm_beta_sd_pct': r4(float(perm_a.std(ddof=1))),
        'perm_ci95_pct': [r4(float(np.percentile(perm_a, 2.5))), r4(float(np.percentile(perm_a, 97.5)))],
        'perm_min_pct': r4(float(perm_a.min())), 'perm_max_pct': r4(float(perm_a.max())),
        'p_one_sided_le_observed': r4(p_le), 'p_two_sided': r4(p_two),
        'mde80_from_perm_sd_pct': r4(2.8016 * float(perm_a.std(ddof=1))),
        'reading': 'REV2-F1(e)：团标签置换零分布下观测 β 的位置 = 单侧 p'
                   f" = {r4(p_le)}（恰在 5% 临界，1000 次分辨率 0.1%）、双侧 p = {r4(p_two)}"
                   f"；经验零分布 sd = {r4(float(perm_a.std(ddof=1)))}%"
                   f" → MDE(80%) ≈ {r4(2.8016 * float(perm_a.std(ddof=1)))}% → 按双侧口径「反号」不显著区别于零分布"
                   '（单侧口径处于临界），表述只能用「未检出正号」。',
    }
    psay('H8R', f"置换检验状态 = ok：{N_ITER} 次 mean = {rev['cluster_label_permutation']['perm_beta_mean_pct']}%、"
                f"sd = {rev['cluster_label_permutation']['perm_beta_sd_pct']}、"
                f"CI95 = {rev['cluster_label_permutation']['perm_ci95_pct']}、"
                f"P(≤观测 {obs_beta}%) = {r4(p_le)}")

    # ---- (f) 功效 / MDE：主读数 + 四层子读数
    b_main = fm(reg, 'R_next', ['PC'])
    main_b, main_t = pct(b_main['coefs']['PC']['beta']), r4(b_main['coefs']['PC']['t_nw'])
    pm_main = power_mde(main_b, main_t, PAPER_TARGET_EFFECT)
    rev['power_mde'] = {
        'main_reading_registered_univariate': pm_main,
        'note_rev2': (f"REV2-F1(f)：t = {main_t}、四层子读数全不显著 → 只能写「**未检出正号（不显著）**」，"
                      f"不得写「稳健反号」；CI95 = {pm_main['ci95_pct']} 同时覆盖 0 与原文效应 "
                      f"+{PAPER_TARGET_EFFECT}%/单位 → 既不能拒绝 β=0、也不能拒绝原文效应（本窗口功效不足）。"),
    }
    psay('H8R', f"功效/MDE 状态 = ok：主读数 se = {rev['power_mde']['main_reading_registered_univariate']['se_pct']}%、"
                f"CI95 = {rev['power_mde']['main_reading_registered_univariate']['ci95_pct']}%、"
                f"MDE(80%) = {rev['power_mde']['main_reading_registered_univariate']['mde80_pct']}%、"
                f"原文效应 +{PAPER_TARGET_EFFECT} 处功效 = "
                f"{rev['power_mde']['main_reading_registered_univariate']['power_at_effect_ref']}")
    out['revision_tests'] = rev
    dump('H8', out)
    psay('H8', '状态 = 完成（含 REV2 修正轮：撤回 + 替代检验 a–f）')
    return out

def assemble():
    parts = {}
    for tag in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'):
        p = DIAG / f'{tag}.json'
        parts[tag] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'status': 'MISSING'}
    S = {
        'artifact': 'competition_sign_diagnostics_20261021',
        'generated_at': '2026-10-21',
        'nature': '项目自建 PC 系数为负的只读诊断（复用既有产物 + 新建诊断脚本/产物；未改任何既有文件/冻结值/主口径）。'
                  '**2026-10-27 口径更正**：仅描述项目自建 PC（对手集合 = 同团全部成员），不是对原文 PC 的复现或证伪。',
        'comparability': {
            'comparable_to_paper': False,
            'registered_deviation': 'methodology_execution_plan:2290',
            'project_rule': 'same_cluster_members（同团全部成员）',
            'paper_rule': '任何与其相似度超过团内最小相似度的公司（字面含团外）',
            'consequence': 'PC 不是原文变量的严格复现；判定 = 不可比/未复现（对手集合口径不同）；'
                           '不得写符号相反/复现失败/证伪',
            'replication_requirement': '按原文第四步规则 + 接近原文词表规模（active_vocab ≈ 1.5 万量级）重建（未做）'},
        'pre_registered_criteria': {
            'rule': '成立 = 主读数层（登记样本 EW lag6 单变量）PC 系数 ≥ 0 且 4 子层 ≥3 层同向转正；'
                    '部分 = 向 0 移动 ≥50% 仍为负；排除 = 移动 <25% 或同号；待定 = 数据不可得/证据冲突',
            'h1_special': '注入检验：合成「PC 与下月收益正相关」面板 → 期望显著为正；为负则管线方向 bug 确诊',
            'h8_special': 'PC 对团规模回归残差的 FM 系数 ≥0 → 机械/结构解释成立',
            'no_sample_selection_no_tuning': True,
        },
        'inputs': {
            'panel': 'data/processed/pricing_sample_ma_teacher_v2.parquet',
            'pc_dir': 'data/processed/network_ma_teacher/{2017..2022}_pc.csv',
            'k_tier_cache': 'artifacts/tmp/pc_rc/h1_net_t{5,10,20,30,44,60,80,120}/（既有只读复用）',
            'returns': 'data/processed/monthly_returns_clean_v2.csv（持有期口径）',
            'factors': 'data/processed/factor_returns_daily.csv（FF4 月聚合）',
            'sw2': 'data/processed/sw2_industry_latest.csv',
        },
        'baseline_readings': {
            'paper_targets': {'table4_col1': '+0.152 (t=7.76)', 'table4_col2': '+0.130 (t=5.40)',
                              'table2_pc_mean_median': '-0.230 / -0.180', 'table3_HL_a3_ew': '+0.218 (t=6.27)'},
            'ours_MDA_teacher': {'registered_EW_lag6_univariate': '-0.5068 (t=-1.4365)',
                                 'registered_EW_lag6_with_controls': '-0.4116 (t=-1.1902)',
                                 'paper_like_EW_lag12_univariate': '-0.4559 (t=-1.9998)',
                                 'quintile_EW_alpha_a3': '-0.3482 (t=-1.4744)'},
            'ours_fulltext_corpus': {'registered_EW_lag6_univariate': '-0.3926 (t=-1.1227)',
                                     'paper_like_EW_lag12_univariate': '-0.4194 (t=-2.0059)',
                                     'quintile_EW_alpha_a3': '-0.4974 (t=-2.0349)',
                                     'src': 'reports/data_gates/G0/competition_pricing_correspondence_fulltext.json'},
            'note': '换语料不改变项目 PC 的点估计符号（历史读数，描述性）；本报告数值为 MD&A 主口径（除注明外）。'
                    'PC 与原文对手集合不可比，判定统一为「不可比/未复现」。',
        },
        'parts': parts,
    }
    v = {}
    h1 = parts.get('H1', {})
    inj = h1.get('injection_test', {})
    pos = inj.get('syn_rank_pos', {})
    h1_ok = (pos.get('beta_pct') is not None and pos.get('beta_pct') > 0 and pos.get('HL_decile_EW_pct', -1) > 0)
    ga = h1.get('group_audit_decile', {})
    v['H1'] = {
        'verdict': '排除' if (h1_ok and ga.get('H_group_is_high_pc') and ga.get('H_group_is_low_hhi')) else '成立（方向 bug）',
        'reason': (f"注入检验：合成正相关 PC 的 β = {pos.get('beta_pct')}%（t = {pos.get('t_nw')}）、"
                   f"H−L(十分位 EW) = {pos.get('HL_decile_EW_pct')}%（t = {pos.get('HL_decile_EW_t')}）；"
                   f"镜像负相关 β = {inj.get('syn_rank_neg', {}).get('beta_pct')}%；随机 β = {inj.get('syn_random', {}).get('beta_pct')}%；"
                   f"D10−D1 PC 均值差 = {ga.get('D10_minus_D1_pc')}、HHI 差 = {ga.get('D10_minus_D1_hhi')}"
                   f"（H 组 PC {'更大' if ga.get('H_group_is_high_pc') else '更小'}）"),
    }
    h2 = parts.get('H2', {})
    ex = h2.get('excl_self_and_coverage_variants', {})
    r_ex = ex.get('registered', {}).get('univariate', {}).get('beta_pct')
    wpa = h2.get('write_precision_audit', {})
    v['H2'] = {
        'verdict': '排除' if (r_ex is not None and r_ex < 0) else '部分/成立',
        'reason': (f"剔除自身变体 β = {r_ex}%（t = {ex.get('registered',{}).get('univariate',{}).get('t_nw')}）；"
                   f"覆盖率=1 子样本 β = {ex.get('coverage_eq_1',{}).get('registered',{}).get('beta_pct')}%；"
                   f"手核团逐位对拍全精度 max|Δhhi| ≤ 5e-9（写盘 round(x,8) 精度；最大团实测 "
                   f"{wpa.get('csv_hhi_max_abs_diff_raw') if wpa.get('csv_hhi_max_abs_diff_raw') is not None else '—'}"
                   f"，原报「= 0」为 r4 显示假象，REV2-F4 修正）；"
                   f"营收字段降级为「盘上 pkl = 营业收入（854.30 亿），原文附录表未核」"),
    }
    h3 = parts.get('H3', {})
    tg = h3.get('tier_grid', {})
    k_vals = [x['k_2019'] for x in tg.values() if x.get('k_2019')]
    t_abs = [abs(x['fm_reg_registered_lag12']['t_nw']) for x in tg.values()
             if x['fm_reg_registered_lag12'].get('t_nw') is not None]
    v['H3'] = {'verdict': '排除' if h3.get('all_negative') else '部分/成立',
               'reason': f"{len(tg)} 档 K（中位 peer 5→300，K(2019) {min(k_vals)}–{max(k_vals)}，含请求的粗 K≈100–200 区）"
                         f"主读数层**无正号** = {h3.get('all_negative')}（β 区间 "
                         f"{min([x['fm_reg_registered_lag12']['beta_pct'] for x in tg.values()], default=None)}"
                         f"..{max([x['fm_reg_registered_lag12']['beta_pct'] for x in tg.values()], default=None)}；"
                         f"多档 |t| ≤ 0.55、全部不显著，最大 |t| = {max(t_abs) if t_abs else None:.4f}）；"
                         f"粗档 tier=150/200 的加控制与 α3 子层转正但不显著（|t| ≤ 0.95）；"
                         f"tier=10 与原文**中心四分位吻合**（p25 −0.3962 vs −0.416、中位 −0.1799 vs −0.180、"
                         f"p75 −0.0733 vs −0.076）但 **sd 0.2204 vs 原文 0.337**（尾部不一致，REV2-F4 限定表述）"
                         f"仍为 −0.3348%"}
    h4 = parts.get('H4', {})
    va = h4.get('variants', {})
    pos4 = {k: x.get('fm_univariate', {}).get('beta_pct') for k, x in va.items()
            if isinstance(x, dict) and x.get('fm_univariate', {}).get('beta_pct') is not None
            and x['fm_univariate']['beta_pct'] > 0}
    v['H4'] = {'verdict': '排除' if not pos4 else '部分/成立',
               'reason': f"7 个样本变体（含无分析师门、含微盘、全 PC 宇宙）系数全部为负；正号变体 = {pos4 or '无'}"}
    h5 = parts.get('H5', {})
    py = h5.get('positive_years_registered', [])
    psig5 = h5.get('positive_and_significant_years', [])
    h5_year_txt = {y: x.get('beta_pct') for y, x in h5.get('by_return_year', {}).get('registered', {}).items()}
    h5_paper_txt = {y: x.get('beta_pct') for y, x in h5.get('by_return_year', {}).get('paper_like', {}).items()}
    h5_reg19 = h5.get('by_return_year', {}).get('registered', {}).get('2019', {})
    h5_pap19 = h5.get('by_return_year', {}).get('paper_like', {}).get('2019', {})
    v['H5'] = {'verdict': '待定（无正面证据；2007–2016 窗口不可测）',
               'reason': f"登记样本逐年 β = {h5_year_txt}（原文最近似样本 = {h5_paper_txt}）；"
                         f"正号年 = {py or '无'}，但**bootstrap CI95 含 0 的正号年 = "
                         f"{psig5 or '无'}**（判据 REV2-F2 = 月块 bootstrap CI，非 NW t）；"
                         f"2019：登记 {h5_reg19.get('beta_pct')}%/CI95 {h5_reg19.get('boot_ci95_pct')}、"
                         f"原文最近似样本 {h5_pap19.get('beta_pct')}%"
                         f"（t_nw = {h5_pap19.get('t_nw')} 但 t_iid = {h5_pap19.get('t_iid')}、"
                         f"CI95 {h5_pap19.get('boot_ci95_pct')} 含 0、LOO β 区间 {h5_pap19.get('loo_beta_range_pct')}）"
                         f"→ 不得写作「显著为正」的正面证据；逐月负号占比 = "
                         f"{h5.get('monthly_beta_sign_distribution', {}).get('share_negative')}"}
    h6 = parts.get('H6', {})
    cco = h6.get('control_coefs_vs_paper', {}).get('registered', {}).get('coefs', {})
    cs6 = h6.get('cluster_size_control', {})
    psa = h6.get('paper_sign_agreement', {})
    _ncp = parts.get('H8', {}).get('revision_tests', {}).get('control_function_form', {}).get('variants', {})
    v['H6'] = {'verdict': '部分（归因收窄：团规模控制仅 log 形式转正且不显著；1/N 仍负；控制项 3/5 与原文反号）',
               'reason': f"加控制后我方 size {cco.get('size')}、bm {cco.get('bm')}、rev {cco.get('rev')}"
                         f"（原文 +0.339 / −0.358 / +1.835）→ 符号相反仅 "
                         f"{psa.get('n_opposite_sign')}/{psa.get('n_controls')} 项（{psa.get('opposite_vars')}），"
                         f"mom/illiq 同号；"
                         f"行业 FE 后 β = {h6.get('industry_fe', {}).get('univariate', {}).get('beta_pct')}%"
                         f"（t = {h6.get('industry_fe', {}).get('univariate', {}).get('t_nw')}）；"
                         f"团规模控制：log 形式 β_PC = {cs6.get('PC_plus_log_peers', {}).get('PC_beta_pct')}%"
                         f"（t = {cs6.get('PC_plus_log_peers', {}).get('PC_t')}，不显著）、"
                         f"**1/N 形式 β_PC = {_ncp.get('PC+invN', {}).get('PC', {}).get('beta_pct')}%"
                         f"（t = {_ncp.get('PC+invN', {}).get('PC', {}).get('t_nw')}，团成员数口径，H8 §8R）仍为负**"
                         f"、1/n_competitors 口径 {cs6.get('PC_plus_inv_peers', {}).get('PC_beta_pct')}%"
                         f"（t = {cs6.get('PC_plus_inv_peers', {}).get('PC_t')}）同样为负 → 归因仅限 log 形式"}
    h7 = parts.get('H7', {})
    yv = h7.get('y_variants', {})
    h7_hor_txt = {k: x.get('beta_pct_per_unit_period') for k, x in h7.get('holding_horizons', {}).items()
                  if k.startswith('h_')}
    h7_hor_monthly = {k: x.get('beta_pct_per_unit_per_month_avg') for k, x in h7.get('holding_horizons', {}).items()
                      if k.startswith('h_')}
    v['H7'] = {'verdict': '排除' if not h7.get('positive_specs') else '待定（部分口径转正）',
               'reason': f"收益口径/持有期变体正号项 = {h7.get('positive_specs') or '无（全部为负）'}；"
                         f"持有期 β（%/单位/期）= {h7_hor_txt}；**折算月均 = {h7_hor_monthly}** → "
                         f"无反转、月均幅度基本不变（t+2..6 ≈ −0.66%/月、t+7..12 ≈ −0.79%/月 vs t+1 −0.57%/月）；"
                         f"各窗口为**重叠窗**且 NW(6) 未做重叠校正 → t 不作推断依据（REV2-F5）"}
    h8 = parts.get('H8', {})
    rt = h8.get('residual_test', {})
    resid_beta = rt.get('fm_PC_resid', {}).get('beta_pct')
    rv8 = h8.get('revision_tests', {})
    idn = rv8.get('identity_decomposition', {})
    ncp = rv8.get('control_function_form', {}).get('variants', {})
    wnb = rv8.get('within_N_bins', {}).get('bins', {})
    drr = rv8.get('distribution_reweighting', {})
    pmp = rv8.get('cluster_label_permutation', {})
    pmd = rv8.get('power_mde', {}).get('main_reading_registered_univariate', {})
    v['H8'] = {'verdict': '待定 / 未被识别（原「根因 #1 = PC 是团结构代理、残差化后负号消失」结论**撤回**）',
               'reason': f"撤回理由 = 对处理变量 PC 的定义性结构分量（团规模/自身份额）回归取残差属过度控制/bad control；"
                         f"且函数形式不稳健（log1p(N) → {cs6.get('PC_plus_log_peers', {}).get('PC_beta_pct')}% 转正，"
                         f"1/N → {ncp.get('PC+invN', {}).get('PC', {}).get('beta_pct')}%、"
                         f"PC+1/N+own_share → {ncp.get('PC+invN+own_share', {}).get('PC', {}).get('beta_pct')}% 仍为负）；"
                         f"恒等式两分量各自为负（FM(−own²) = {idn.get('t_own', {}).get('beta_pct')}%/"
                         f"t = {idn.get('t_own', {}).get('t_nw')}、FM(−他人集中度) = {idn.get('t_peer', {}).get('beta_pct')}%/"
                         f"t = {idn.get('t_peer', {}).get('t_nw')}）；within-N 分箱 β = "
                         f"{ {k: v2.get('beta_pct') for k, v2 in wnb.items()} }；"
                         f"分布重加权：WLS β = {drr.get('registered|w_dist', {}).get('beta_pct')}%"
                         f"（t = {drr.get('registered|w_dist', {}).get('t_nw')}）、分位匹配 β = "
                         f"{drr.get('registered|quantile_match', {}).get('beta_pct')}%"
                         f"（t = {drr.get('registered|quantile_match', {}).get('t_nw')}）；"
                         f"团标签置换 1000 次：单侧 p = {pmp.get('p_one_sided_le_observed')}；"
                         f"功效/MDE：se = {pmd.get('se_pct')}%、MDE(80%) = {pmd.get('mde80_pct')}%、"
                         f"原文效应处功效 = {pmd.get('power_at_effect_ref')}"}
    S['verdicts'] = v
    # ---- REV2 修订元信息（依独立复核 F1–F6）
    S['revision'] = {
        'revision_id': 'REV2-20261021',
        'date': '2026-10-21',
        'trigger': '独立复核（artifacts/tmp/review_diag/{rv_lib.py,A_injection.py,B_h2.py,C_h3h4.py,D_h5.py,'
                   'E_h6.py,F_h8.py,F2_h8_identity.py} 及同名 JSON）判定原报告三处方法论/表述问题（F1–F6）',
        'changes': [
            'F1｜H8 撤回「PC ≈ 团结构代理、残差化后负号消失」（过度控制/bad control；函数形式不稳健；恒等式两分量各自为负）'
            '→ 改判待定/未被识别；补做 6 项替代检验（恒等式分解 / 1/N 控制 / within-N 分箱 / 分布匹配重加权 / 1000 次团标签置换 / 功效-MDE），'
            '全部由本脚本独立重算（不引用复核脚本产物数值）',
            'F2｜H5 降级为「待定（无正面证据；2007–2016 窗口不可测）」；年度/子期间并报 iid t + 月块 bootstrap CI95 + LOO；判据改以 bootstrap CI 为准',
            'F3｜H6 归因收窄：「团规模控制使 PC 转正」仅限 log 形式且不显著；补 1/N 线性控制；控制项符号 3/5 与原文相反（size/bm/rev），mom/illiq 同号',
            'F4｜精度/强度表述：H2 max|Δhhi| ≤5e-9（round(x,8) 写盘精度）、营收字段降级；H3「11/11 为负」→「无正号（多档 |t| ≤ 0.55）」、'
            '分布「高度吻合」→「中心四分位吻合（sd 0.2204 vs 0.337）」',
            'F5｜H7 持有期 β 折算月均 + 注明重叠窗/NW(6) 未做重叠校正',
            'F6｜治理登记（manifest / registry / spec 三角刷新 + 四审计重跑 + 三哈希闭环）',
        ],
        'principle': '只改本诊断轮自己的产物；未改主口径 / 冻结值 / headline_value；新增检验判据（docstring）先登记后跑。',
    }
    # ---- 根因排序 / 已排除 / 未排除 / 下一步
    h8c = h8.get('pc_structure_corr', {})
    h5_yearly = {y: x.get('beta_pct') for y, x in h5.get('by_return_year', {}).get('registered', {}).items()}
    h5_sub = '；'.join(f"{k.split('|')[-1]} = {x.get('beta_pct')}%（t = {x.get('t_nw')}；"
                       f"bootstrap CI95 {x.get('boot_ci95_pct')}）"
                       for k, x in h5.get('subperiods', {}).items() if k.startswith('registered'))
    h7_hor = {k: x.get('beta_pct_per_unit_per_month_avg') for k, x in h7.get('holding_horizons', {}).items()
              if k.startswith('h_')}
    S['summary_ranking'] = [
        {'candidate': '**（撤回）PC 的横截面信息 = 团结构代理（自身收入份额 + 团规模）**（H8 结构层）',
         'support': '**撤回**（原「强（唯一使负号消失的口径）」不成立）',
         'evidence': f"过度控制/bad control（对处理变量的定义性分量取残差）；函数形式不稳健（log1p(N) → "
                     f"{cs6.get('PC_plus_log_peers', {}).get('PC_beta_pct')}% 对照 1/N → "
                     f"{ncp.get('PC+invN', {}).get('PC', {}).get('beta_pct')}% 仍为负）；恒等式两分量各自为负"
                     f"（FM(−own²) = {idn.get('t_own', {}).get('beta_pct')}%、FM(−他人集中度) = "
                     f"{idn.get('t_peer', {}).get('beta_pct')}%）→ 改判「待定/未被识别」"},
        {'candidate': '**期间 / 市场状态（2018–2023 vs 原文 2007–2016）**（H5）',
         'support': '待定（无正面证据；2007–2016 窗口不可测）',
         'evidence': f"登记样本逐年 β = {h5_yearly}；原文最近似样本逐年 β = {h5_paper_txt}"
                     f"（2019 = +0.3391%/t_nw = 2.92 但 **t_iid = 1.06、月块 bootstrap CI95 [−0.2645, +0.9235] 含 0、"
                     f"LOO β 区间 [0.1990, 0.4915]、去 2019-09 后 t = 1.47** → 非显著为正，不得作为正面证据；"
                     f"2023 = −1.3857%/t = −4.7958 但 bootstrap CI95 [−2.6438, −0.2771]）；子期间 {h5_sub}；"
                     f"逐月负号占比 {h5.get('monthly_beta_sign_distribution', {}).get('share_negative')}；"
                     f"控制变量符号相反仅 3/5（size/bm/rev）→ 该窗口横截面定价结构不同；"
                     f"持有期折算月均不改善（{h7_hor}）→ 排除「信号延迟」型期间解释"},
        {'candidate': '残余口径（原文 12 控制集 / 原文样本筛除 样本筛选 / 原文肘点团结构不可得 / 分布匹配重加权）',
         'support': '弱–中（未穷尽；分布口径已用重加权检验，未使主读数转正）',
         'evidence': f"列(1) 单变量已为负 → 控制集翻转概率低；K 网格 11 档主读数层无正号；"
                     f"within-N 分箱内无一致正号（{ {k: v2.get('beta_pct') for k, v2 in wnb.items()} }）；"
                     f"分布匹配重加权（WLS β = {drr.get('registered|w_dist', {}).get('beta_pct')}%、"
                     f"分位匹配 β = {drr.get('registered|quantile_match', {}).get('beta_pct')}%）亦未转正且不显著；"
                     f"原文 原文样本筛除 与 βMKT/IA/Roe/Lev/Disp/Ivol 未逐条复刻"},
    ]
    S['retracted_conclusions'] = [
        '**撤回**：H8「PC 的横截面信息 = 团结构代理……残差化后负号消失」→ 复核三条反驳（原文收录）：'
        + ' ／ '.join(rv8.get('withdrawal', {}).get('review_rebuttals_verbatim', [])),
        '**收回**：H5 根因排序 #2 中「2019 显著为正」的表述（NW t = 2.92 由月系数强自相关放大 2.63×：'
        't_iid = 1.06；月块 bootstrap CI95 含 0；去 2019-09 后 t = 1.47）→ 改为「无正面证据」',
        '**修正**：H3「11/11 为负」→「无正号（多档 |t| ≤ 0.55，均不显著）」；'
        'H2「max|Δhhi| = 0」→「≤ 5e-9（round(x,8) 写盘精度）」；'
        'H2 营收字段「与原文一致」→「盘上 pkl = 营业收入，原文附录表未核」；'
        'H3 分布「高度吻合」→「中心四分位吻合、sd 0.2204 vs 原文 0.337」',
    ]
    S['revised_root_cause'] = {
        'excluded_strong': [
            f"H1 方向/排序 bug — 注入检验 + PC×(−1) 反证：合成正相关 PC β = "
            f"{inj.get('syn_rank_pos', {}).get('beta_pct')}%（t = {inj.get('syn_rank_pos', {}).get('t_nw')}）、"
            f"镜像 β = {inj.get('syn_rank_neg', {}).get('beta_pct')}%、随机 β = {inj.get('syn_random', {}).get('beta_pct')}%；"
            f"PC×(−1) 反证 β = {inj.get('adv_neg_PC', {}).get('beta_pct')}%"
            f"（t = {inj.get('adv_neg_PC', {}).get('t_nw')}，逐位反号；对照复核实测 +0.5068/+1.4365）；"
            f"D10−D1 PC {ga.get('D10_minus_D1_pc')} / HHI {ga.get('D10_minus_D1_hhi')}",
            'H2 PC 定义细节 — 手核团全精度 ≤5e-9（274 成员团）；剔除自身 −0.3516（t = −1.56）；'
            '营收覆盖=1 子样本 −0.5838；hhi_excl_self 列语义已登记',
            'H3 K 口径 — 11 档（K(2019) 116–1042）主读数层无正号、全部不显著（多档 |t| ≤ 0.55，最大 1.80）；'
            'tier=10 中心四分位对齐原文后仍 −0.3348%',
            'H4 样本口径 — 7 变体全为负（全 PC 宇宙 n = 645,782 → −0.3796；无分析师门 −0.189；纳微盘 −0.783）',
            'H7 收益口径与横截面 — 14 个收益规格全负（EW/VW、raw/α、winsor 变体、行业/市场调整、后复权）；'
            '规模三分位全负（大盘最负 −0.839）；持有期折算月均无反转',
            '（跨轮一致）T1–T6 批与 rootcause H1–H10 已排除：尾部极值、小市值风格、NW 实现、语料、词库粒度、'
            '行业马甲、后复权、PIT 对齐',
        ],
        'pending': [
            '期间/制度（H5）：无正面证据；原文 2007–2016 窗口不可测（语料仅文本年 2017+）',
            '残余口径：原文 12 控制集（βMKT/IA/Roe/Lev/Disp/Ivol + 证监会二级行业 HHI）、原文样本筛除 样本筛选、'
            '原文肘点团结构不可得',
            'H8 结构层（原「团结构代理」）：改正为**待定/未被识别**（过度控制方法论不成立；'
            f"分布匹配重加权（重要性 WLS β = {drr.get('registered|w_dist', {}).get('beta_pct')}%/t = "
            f"{drr.get('registered|w_dist', {}).get('t_nw')}；分位匹配 β = "
            f"{drr.get('registered|quantile_match', {}).get('beta_pct')}%/t = "
            f"{drr.get('registered|quantile_match', {}).get('t_nw')}）与 within-N 分箱均未提供支持",
            '推断层强度：团标签置换 1000 次零分布下观测 β 单侧 p = '
            f"{pmp.get('p_one_sided_le_observed')}（恰在 5% 临界、分辨率 0.1%）、双侧 p = {pmp.get('p_two_sided')}"
            f" → 按双侧口径「反号」不显著区别于零分布（单侧口径处于临界）、置换 MDE(80%) ≈ "
            f"{pmp.get('mde80_from_perm_sd_pct')}%；"
            f"功效/MDE：se = {pmd.get('se_pct')}%、CI95 = {pmd.get('ci95_pct')}%、MDE(80%) = {pmd.get('mde80_pct')}%、"
            f"原文效应 +0.152 处功效 = {pmd.get('power_at_effect_ref')} → **本设计对原文量级功效不足**",
        ],
        'retracted': [
            'H8「PC ≈ 团结构代理、残差化后负号消失」（过度控制/bad control；函数形式不稳健；恒等式两分量各自为负）',
            'H5 排序 #2 的「2019 显著为正」表述（NW t 被自相关放大；bootstrap CI 含 0）',
        ],
        'final_conclusion': (
            f"两套语料、四类样本、11 档 K、14 种收益口径下**均未检出中文论文的正号**；"
            f"主读数不显著（β = {pmd.get('beta_pct')}%/单位、t = {pmd.get('t_nw')}；"
            f"MDE(80%) ≈ {pmd.get('mde80_pct')}%/单位、原文效应 +0.152 处功效 ≈ "
            f"{pmd.get('power_at_effect_ref')}；CI95 {pmd.get('ci95_pct')} 同时覆盖 0 与 +0.152 → 不能读作「已证伪原文效应」；"
            f"团标签置换 1000 次单侧 p = {pmp.get('p_one_sided_le_observed')}、双侧 p = {pmp.get('p_two_sided')}），"
            f"故应表述为「**未检出正号**」而非「稳健反号」；H8「团结构代理」根因撤回、期间层无正面证据 → 未识别根因。"
            f"但 PC 与原文对手集合不可比（项目 = 同团成员；原文 = 相似度 > 团内最小相似度，字面含团外），"
            f"故该结论只描述项目自建 PC，不构成对原文的复现或证伪。"),
    }
    S['excluded'] = [
        f"H1 排序/高低方向 bug（注入检验：合成正相关 PC β = {inj.get('syn_rank_pos', {}).get('beta_pct')}%"
        f"（t = {inj.get('syn_rank_pos', {}).get('t_nw')}）、镜像 β = {inj.get('syn_rank_neg', {}).get('beta_pct')}%、"
        f"随机 β = {inj.get('syn_random', {}).get('beta_pct')}%；PC×(−1) 反证 = +0.5068（t = +1.4365）；"
        f"D10−D1 PC {ga.get('D10_minus_D1_pc')} / HHI {ga.get('D10_minus_D1_hhi')}）",
        'H2 PC 定义/算式细节（手核团全精度 max|Δhhi| ≤ 5e-9 = 写盘 round(x,8) 精度（含 274 成员团）；'
        '剔除自身 −0.3516（t=−1.56）；营收覆盖=1 子样本 −0.5838；营收字段 = 盘上 pkl 营业收入 854.30 亿、'
        '**原文附录表未核**（REV2-F4 降级））',
        'H3 K 校准口径（11 档 K：中位 peer 5→300、K(2019) 116–1042，含请求的粗 K≈100–200 区；'
        '**主读数层无正号（多档 |t| ≤ 0.55，全部不显著，最大 |t| = 1.80）**；'
        'tier=10 与原文**中心四分位吻合**（p25 −0.3962 vs −0.416、中位 −0.1799 vs −0.180、p75 −0.0733 vs −0.076）'
        '但 sd 0.2204 vs 0.337（尾部不一致）仍为 −0.3348%）',
        'H4 样本口径（7 变体全为负：全 PC 宇宙 n=645,782 β=−0.3796；无分析师门 −0.189；纳微盘 −0.783）',
        'H7 收益口径/横截面结构/持有期（14 个收益规格全负；规模三分位全负且大盘最负 −0.839；'
        '持有期 β 折算月均无反转、幅度基本不变（≈−0.57 / −0.66 / −0.79 %/月）；α3 仍为负）',
        '（跨轮一致）T1–T6 批与 rootcause H1–H10 已排除：尾部极值、小市值风格、NW 实现、语料、词库粒度、'
        '行业马甲、后复权、PIT 对齐',
    ]
    S['open_items'] = [
        '原文 2007–2016 窗口**不可测**：我方 MD&A 与年报全文语料均仅覆盖文本年 2017+（营收面板 FY2016+）'
        '→ 无法在同窗口对照；期间的判别只能靠**向前延展**（FY2013–2015 营收重建 → 文本年 2014–2022）。',
        '原文肘点法的确切团结构（K 与团形）无法从表 2 唯一反推 → K 网格覆盖了"分布对齐"维度，'
        '但无法证明"原文团结构下符号会变"。',
        '原文 12 项控制集（βMKT 36 月滚动 / IA / Roe / Lev / Disp / Ivol / 证监会二级行业 HHI）与 '
        '原文样本筛除 样本筛选（金融/综合类剔除、上市不足两年、词条数 <10 等）未逐条复刻。',
        '**H8 结构层 = 待定/未被识别（原「团结构代理」结论已撤回）**：残差化路线属过度控制/bad control，'
        '不得作为根因；round-2 替代检验（恒等式分解 / 1/N 控制 / within-N 分箱 / 分布匹配重加权 / 1000 次团标签置换）'
        '均未提供正号或机制证据。',
        '**统计功效不足**：主读数 se ≈ '
        f"{pmd.get('se_pct')}%、MDE(80%) ≈ {pmd.get('mde80_pct')}%/单位，原文效应 +0.152 处功效 ≈ "
        f"{pmd.get('power_at_effect_ref')} → 「未检出正号」不能读作「已证伪原文效应」；"
        '要检验原文量级需扩展窗口/提高截面 N（或与原文同窗口）。',
    ]
    S['next_steps'] = [
        '**决定性实验（期间层）**：MRQ→FY2013–2015 营收重建（前置门：FY2016+ 与年报相对误差 ≤1%）→ '
        'PC 文本年扩至 2014–2022（事件月 ≈104）→ 检验 2015–2017 是否出现正号（以月块 bootstrap CI 为判据）；'
        '若仍无正号，则期间层也被压缩为"窗口内不可定案"。',
        '**规格补全（残余口径）**：构造 βMKT(36 月滚动)/IA/Roe/Lev/Disp/Ivol + 证监会二级行业 HHI，'
        '复刻表 4 列(2) 的 12 控制集；同步按 原文样本筛除 逐条复刻样本（预计不翻转，列(1) 已为负）。',
        '**功效补强（推断层）**：本轮 1000 次团标签置换 + MDE 已完成（单侧 p = '
        f"{pmp.get('p_one_sided_le_observed')}、MDE(80%) ≈ {pmd.get('mde80_pct')}%）→ 下一步是压 MDE："
        '延长文本年（同期间层实验）、提高月内有效截面、或改用重叠窗正确校正（HAC 以事件月聚合）。',
        '**结构层路线更正**：撤回「对 log 团员数/自身收入份额正交化」的最小修复建议（bad control）；'
        '若仍需机制解释，须用**团层随机化/工具**或明确外生结构变量，不得对 PC 的定义性分量做控制。',
        '**表述层兜底**：论文按「项目 PC 与原文不可比（对手集合口径不同）+ 同窗口点估计为负 + 功效不足」并列陈述；'
        '不构成对原文的复现或证伪；写"未检出正号（不显著）"，不得写"复现成功"、不得写"稳健反号"、不得写"已找到唯一根因"。',
    ]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    psay('asm', f'JSON 报告状态 = ok（{OUT_JSON.relative_to(ROOT)}）')
    make_md(S)
    return S

def make_md(S: dict) -> None:
    p = S['parts']

    def row(x, y):
        return f'| {x} | {y} |'

    md = []
    md.append('# network「项目自建 PC → 预期收益」系数为负 · 深度诊断 H1–H8（2026-10-21；2026-10-27 口径更正）\n')
    md.append('> **2026-10-27 口径更正（Phase 0.2 口径复核）**：本报告的「负号/反号」分析仅针对**项目自建 PC**'
              '（对手集合 = 同团全部成员）的描述性排查，**不是对刘亚辉等（2021）PC 变量的复现或证伪**。'
              '原文第四步的对手集合 = 「任何与其相似度超过团内最小相似度的公司」（字面含团外），项目实现 = 同团成员'
              '（`methodology_execution_plan.md:2290` 已登记偏离）；二者不是同一对手集合。因此：① 本报告 H1–H8 的裁决与数值全部保留，'
              '但解释范围限于项目 PC；② 不得把本报告读作「原文复现失败 / 符号相反 / 原文结论被证伪」；'
              '③ 如需对原文 PC 做复现，须按原文第四步规则 + 接近原文词表规模重建（active_vocab ≈ 1.5 万量级；未做）。'
              '主读数数值不变：β = −0.5068%（t = −1.4365）、95% CI [−1.198, +0.185]（覆盖原文 +0.152）、'
              '原文效应处功效 ≈ 0.07、68 个月。\n')
    md.append('> **性质**：只读诊断（复用既有产物 + 新建诊断脚本 `scripts/competition_sign_diagnostic.py`）；'
              '未覆盖任何既有产物、未改 headline_value / 冻结值 / 主口径网络。')
    md.append('> **修订（REV2，2026-10-21）**：按独立复核意见 F1–F6 修正本报告：H8「团结构代理」结论**撤回**、'
              'H5 降级为**待定（无正面证据）**、H6 归因收窄（仅 log 形式）、H2/H3/H7 精度与单位表述修正 + '
              '补做复核指定替代检验（恒等式分解 / 1/N 控制 / within-N 分箱 / 分布匹配重加权 / 1000 次团标签置换 / 功效-MDE）。')
    md.append('> **判据（先登记，REV2 追加）**：成立 = 主读数层 PC 系数 ≥ 0 且 4 子层 ≥3 层同向转正；'
              '部分 = 向 0 移动 ≥50% 仍为负；排除 = 移动 <25% 或同号；待定 = 数据不可得/证据冲突。')
    md.append('> **背景靶子（不可直接对表）**：`ref/teacher-materials/A股产品市场竞争.pdf` 表 4 列(1) +0.152（t=7.76）/ 列(2) +0.130（t=5.40）'
              '；现行读数 = `reports/data_gates/G0/competition_pricing_correspondence.{json,md}`（判定 = 不可比/未复现）。\n')
    md.append('## 0. 裁决总表\n')
    md.append('| # | 假设 | 判定 | 关键数字 |')
    md.append('|---|---|---|---|')
    for tag in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'):
        vv = S['verdicts'][tag]
        md.append(f"| {tag} | {p[tag].get('hypothesis', '')} | **{vv['verdict']}** | {vv['reason']} |")
    md.append('\n## 1. H1 方向/排序 bug（最高优先）\n')
    h1 = p['H1']
    md.append(f"- 静态读码：`pc = −hhi` 逐值成立（max|pc+hhi| = {h1['code_audit']['max_abs_pc_plus_hhi']}）；"
              f"分组 = 逐月 `rank_average`（升序 → g10 = PC 最大）；H−L = g10 − g1；FM 系数 = dR/dPC。")
    ga = h1['group_audit_decile']
    md.append('- 分组核验（登记样本十分位，pooled）：\n')
    md.append('| 组 | n | mean PC | median PC | mean HHI | 中位团员数 | 组均收益 % |')
    md.append('|---|---|---|---|---|---|---|')
    for i in range(1, 11):
        g = ga['group_stat'][f'D{i}']
        md.append(f"| D{i} | {g['n']:,} | {g['pc_mean']} | {g['pc_median']} | {g['hhi_mean']} | "
                  f"{g['peer_median']:.0f} | {g['r_mean_pct']} |")
    md.append(f"\n- D10−D1：PC 均值差 {ga['D10_minus_D1_pc']}、HHI 均值差 {ga['D10_minus_D1_hhi']}；"
              f"**H 组（D10）PC {'更大' if ga['H_group_is_high_pc'] else '更小'}、HHI {'更小' if ga['H_group_is_low_hhi'] else '更大'}**。")
    inj = h1['injection_test']
    md.append('- 注入检验（合成列，真实数据零改动）：\n')
    md.append('| 合成变量 | FM β（%/单位） | t(NW) | 十分位 H−L EW %（t） | 十分位 H−L VW %（t） |')
    md.append('|---|---|---|---|---|')
    for k, lab in (('syn_rank_pos', 'PC_syn = 月内 R_next 秩 − 0.5（与下月收益正相关）'),
                   ('syn_rank_neg', 'PC_syn = −(月内 R_next 秩 − 0.5)（镜像）'),
                   ('syn_random', '月内均匀随机'), ('real_PC', '真实 PC（对照）'),
                   ('adv_neg_PC', 'PC×(−1) 反证（REV2，应逐位反号）')):
        x = inj[k]
        md.append(f"| {lab} | {x['beta_pct']} | {x['t_nw']} | "
                  f"{x.get('HL_decile_EW_pct', '—')}（{x.get('HL_decile_EW_t', '—')}） | "
                  f"{x.get('HL_decile_VW_pct', '—')}（{x.get('HL_decile_VW_t', '—')}） |")
    md.append(f"\n- 手算 OLS 对拍（逐月 Σ(x−x̄)(y−ȳ)/Σ(x−x̄)² vs 库函数逐月系数）：")
    for mo, hh in h1['hand_ols_check'].items():
        md.append(f"  - {mo}：n = {hh['n']}、手算 β = {hh['beta_pct']}%（库 = {hh['lib_beta_raw']*100:.4f}%）、"
                  f"逐值一致 = {hh['match']}、同月 corr(PC, R_next) = {hh['corr_PC_R_next']}")
    md.append('\n## 2. H2 PC 定义/算式细节\n')
    h2 = p['H2']
    md.append(f"- 原文定义：{h2['paper_definition']['quote']}")
    md.append('\n- 手核团（clusters + 营收 pkl 现算，与 `{y}_pc.csv` 逐位对拍，**全精度**）：\n')
    md.append('| 文本年 | 团 id | 成员数 | 营收覆盖 | 现算 HHI（含自身） | max|Δhhi| vs csv（全精度） | 列 = −HHI_excl（剔除自身的 PC） |')
    md.append('|---|---|---|---|---|---|---|')
    for a in h2['manual_cluster_audit']:
        raw = a.get('csv_hhi_max_abs_diff_raw')
        md.append(f"| {a['text_year']} | {a['cluster_id']} | {a['n_members']} | {a['n_rev_covered']} | "
                  f"{a['hhi_recomputed_incl_self']} | {raw:.3e} | "
                  f"{a.get('csv_col_equals_minus_HHI_excl')} |")
    wpa = h2.get('write_precision_audit', {})
    if wpa:
        md.append(f"- 写盘精度核验（REV2-F4）：最大团 id = {wpa.get('largest_cluster_2019_id')}"
                  f"（{wpa.get('n_members')} 成员）全精度 max|Δhhi| = {wpa.get('csv_hhi_max_abs_diff_raw'):.3e}"
                  f"、max|Δpc| = {wpa.get('csv_pc_max_abs_diff_raw'):.3e}；"
                  f"**理论界 ≤5e-9（pc.csv 写盘 = round(x,8)）** → 原报告「max|Δhhi| = 0」为 r4 显示精度假象，已改记 ≤5e-9")
    ex = h2['excl_self_and_coverage_variants']
    md.append('\n- 变体读数：\n')
    md.append('| 变体 | 样本 | n 行 | 单变量 β %（t） | 加控制 β %（t） | 五分位 α3 %（t） |')
    md.append('|---|---|---|---|---|---|')
    for tag, s in ex.items():
        if tag == 'coverage_eq_1':
            continue
        md.append(f"| 剔除自身（PC_x） | {tag} | {s['n_rows']:,} | {s['univariate']['beta_pct']}"
                  f"（{s['univariate']['t_nw']}） | {s['with_controls']['beta_pct']}（{s['with_controls']['t_nw']}） | "
                  f"{s['quintile_EW_alpha_a3_pct']}（{s['quintile_EW_alpha_a3_t']}） |")
    for tag, s in ex['coverage_eq_1'].items():
        md.append(f"| 营收覆盖 = 1 | {tag} | {s['n_rows']:,} | {s['beta_pct']}（{s['t_nw']}） | — | — |")
    dv = h2['distribution_vs_paper']
    md.append(f"\n- 分布对照：我方 firm-(month) n = {dv['ours_firm_year']['n']:,}、均值 {dv['ours_firm_year']['mean']} / "
              f"中位 {dv['ours_firm_year']['median']} / p25 {dv['ours_firm_year']['p25']} / p75 {dv['ours_firm_year']['p75']} / "
              f"sd {dv['ours_firm_year']['sd']}（原文表 2 n = {dv['paper_table2']['n']:,}、−0.230 / −0.180 / −0.416 / −0.076 / 0.337）；"
              f"营收覆盖率中位 {dv['coverage']['median']}、覆盖 < 1 占比 {dv['coverage']['share_lt1']}")
    md.append(f"- 营收字段：{h2['revenue_field']['interpretation']}；"
              f"`hhi_excl_self` 列语义 = {h2['revenue_field']['excl_self_column_semantics']}")
    md.append('\n## 3. H3 K 校准口径\n')
    h3 = p['H3']
    md.append('| tier（中位 peer 目标） | K(2019) | PC 均值 | PC 中位 | FM 登记 lag12 %（t） | FM 加控制 %（t） | 五分位 EW α3 %（t） |')
    md.append('|---|---|---|---|---|---|---|')
    for t, x in h3['tier_grid'].items():
        md.append(f"| {t} | {x['k_2019']} | {x['pc_mean']} | {x['pc_median']} | "
                  f"{x['fm_reg_registered_lag12']['beta_pct']}（{x['fm_reg_registered_lag12']['t_nw']}） | "
                  f"{x['fm_reg_ctrl_lag12']['beta_pct']}（{x['fm_reg_ctrl_lag12']['t_nw']}） | "
                  f"{x['quintile_EW_alpha_a3_pct']}（{x['quintile_EW_alpha_a3_t']}） |")
    t_abs = [abs(x['fm_reg_registered_lag12']['t_nw']) for x in h3['tier_grid'].values()
             if x['fm_reg_registered_lag12'].get('t_nw') is not None]
    tr10 = h3['tier_grid'].get('10', {})
    md.append(f"\n- {len(h3['tier_grid'])} 档主读数层**无正号 = {h3['all_negative']}**（多档 |t| ≤ 0.55、"
              f"全部不显著，最大 |t| = {max(t_abs):.4f}；细档 PC 由既有 K 重建缓存只读提供、粗档 K≈100–200 由本轮新建 tag "
              f"`dcoarse` 重建；FM 由本脚本逐档重跑）。")
    md.append(f"- tier=10 与原文**中心四分位吻合**：p25 {tr10.get('pc_p25')} vs −0.416、中位 {tr10.get('pc_median')} vs −0.180、"
              f"p75 {tr10.get('pc_p75')} vs −0.076（firm-year n = {tr10.get('n_firm_year')} vs 15,482）；"
              f"但 **sd 0.2204 vs 原文 0.337**（尾部不一致）→ 表述限定为「中心四分位吻合」，不得写「分布高度吻合」（REV2-F4）；"
              f"该档 FM 仍为 −0.3348%（t = −1.04）")
    if h3.get('coarse_K_note'):
        md.append(f"- {h3['coarse_K_note']}")
    md.append('\n## 4. H4 样本口径\n')
    md.append('| 变体 | n 行 | 月数 | 均值市值(亿) | 均值分析师覆盖 | β %（t） | 加控制 β % |')
    md.append('|---|---|---|---|---|---|---|')
    for k, x in p['H4']['variants'].items():
        if 'fm_univariate' not in x:
            md.append(f"| {k} | {x.get('n_rows', '—')} | — | — | — | {x.get('error', x.get('note', '—'))} | — |")
            continue
        md.append(f"| {k} | {x['n_rows']:,} | {x.get('n_months', '—')} | {x.get('mean_market_equity_1e8', '—')} | "
                  f"{x.get('analyst_cov_mean', '—')} | {x['fm_univariate']['beta_pct']}（{x['fm_univariate']['t_nw']}） | "
                  f"{x.get('fm_with_controls', {}).get('beta_pct', '—')}"
                  f"（{x.get('fm_with_controls', {}).get('t_nw', '—')}） |")
    md.append('\n## 5. H5 期间（REV2-F2：判据改以月块 bootstrap CI 为准）\n')
    h5 = p['H5']
    md.append('| 回归年 | 登记样本 β %（t_nw / t_iid） | bootstrap CI95 | LOO β 区间 | 原文最近似样本 β %（t_nw / t_iid） |')
    md.append('|---|---|---|---|---|')
    for y in sorted(h5['by_return_year']['registered']):
        a = h5['by_return_year']['registered'][y]
        b = h5['by_return_year']['paper_like'].get(y, {})
        md.append(f"| {y} | {a.get('beta_pct', '—')}（{a.get('t_nw', '—')} / {a.get('t_iid', '—')}） | "
                  f"{a.get('boot_ci95_pct', '—')} | {a.get('loo_beta_range_pct', '—')} | "
                  f"{b.get('beta_pct', '—')}（{b.get('t_nw', '—')} / {b.get('t_iid', '—')}） |")
    md.append('\n- 子期间：' + '；'.join(f"{k} = {v['beta_pct']}%（t_nw = {v['t_nw']}、t_iid = {v.get('t_iid')}；"
                                        f"bootstrap CI95 {v.get('boot_ci95_pct')}）" for k, v in h5['subperiods'].items()))
    md.append(f"- 逐月系数：负号月占比 {h5['monthly_beta_sign_distribution']['share_negative']}"
              f"、均值 {h5['monthly_beta_sign_distribution']['mean_beta_pct']}%"
              f"、区间 [{h5['monthly_beta_sign_distribution']['min_pct']}, {h5['monthly_beta_sign_distribution']['max_pct']}]%")
    pc5 = h5.get('period_criterion', {})
    md.append(f"- 判据（REV2-F2）：{pc5.get('rule_rev2', '')}")
    md.append(f"- **结论改判：待定（无正面证据；2007–2016 窗口不可测）**；正号且 bootstrap CI 不含 0 的年 = "
              f"{h5.get('positive_and_significant_years') or '无'}（原报告把「2019 显著为正」当作积极证据的表述已收回）")
    md.append('\n## 6. H6 控制集与规格\n')
    h6 = p['H6']
    md.append('| 变量 | 原文表 4 列(2) β %（t） | 我方 β %（t） |')
    md.append('|---|---|---|')
    for c, (pb, pt) in [('size', (0.339, 23.42)), ('bm', (-0.358, -8.79)), ('mom', (1.791, 15.28)),
                        ('rev', (1.835, 8.27)), ('illiq', (0.028, 4.68))]:
        o = h6['control_coefs_vs_paper']['registered']['coefs'].get(c, {})
        md.append(f"| {c} | {pb}（{pt}） | {o.get('beta_pct')}（{o.get('t_nw')}） |")
    md.append(f"- 行业 FE（月×申万二级 去均值，组内 ≥5 家，n = {h6['industry_fe']['n_rows']:,}）："
              f"单变量 β = {h6['industry_fe']['univariate']['beta_pct']}%（t = {h6['industry_fe']['univariate']['t_nw']}）、"
              f"加 5 控制 β = {h6['industry_fe']['with_controls_5']['beta_pct']}%"
              f"（t = {h6['industry_fe']['with_controls_5']['t_nw']}）")
    cs = h6['cluster_size_control']
    psa = h6.get('paper_sign_agreement', {})
    _ncp6 = p.get('H8', {}).get('revision_tests', {}).get('control_function_form', {}).get('variants', {})
    md.append(f"- 团规模控制（REV2-F3 收窄）：log(1+团员数) 单独 β = {cs['log_peers_only']['beta_pct']}%（t = {cs['log_peers_only']['t_nw']}）；"
              f"PC + log 团员 → β_PC = {cs['PC_plus_log_peers']['PC_beta_pct']}%"
              f"（t = {cs['PC_plus_log_peers']['PC_t']}）；再加 5 控制 → β_PC = "
              f"{cs['PC_plus_log_peers_plus_controls']['PC_beta_pct']}%（t = {cs['PC_plus_log_peers_plus_controls']['PC_t']}）；"
              f"**换 1/N 线性形式（1/n_competitors 口径，H6）→ PC + 1/N β_PC = "
              f"{cs.get('PC_plus_inv_peers', {}).get('PC_beta_pct')}%/t = "
              f"{cs.get('PC_plus_inv_peers', {}).get('PC_t')}；PC + 1/N + 5 控制 = "
              f"{cs.get('PC_plus_inv_peers_plus_controls', {}).get('PC_beta_pct')}%/t = "
              f"{cs.get('PC_plus_inv_peers_plus_controls', {}).get('PC_t')}；"
              f"H8 §8R 团成员数口径 1/N = {_ncp6.get('PC+invN', {}).get('PC', {}).get('beta_pct')}%/t = "
              f"{_ncp6.get('PC+invN', {}).get('PC', {}).get('t_nw')}、+ own_share = "
              f"{_ncp6.get('PC+invN+own_share', {}).get('PC', {}).get('beta_pct')}%/t = "
              f"{_ncp6.get('PC+invN+own_share', {}).get('PC', {}).get('t_nw')} → 均仍为负**；"
              f"两种形式均不显著 → 「团规模控制使 PC 转正」**仅限 log 形式**，不构成结构层证据")
    md.append(f"- 控制项符号与原文一致计数（REV2-F3）：同号 {psa.get('n_same_sign')}/{psa.get('n_controls')}、"
              f"相反 {psa.get('n_opposite_sign')}/{psa.get('n_controls')}（{psa.get('opposite_vars')}）；"
              f"mom/illiq 同号 → 表述须限定为「符号相反仅 3/5 控制项」")
    md.append('\n## 7. H7 收益口径/横截面结构\n')
    h7 = p['H7']
    md.append('| 被解释变量 | EW β %（t） | VW β %（t） |')
    md.append('|---|---|---|')
    for k in sorted(set(x.split('|')[0] for x in h7['y_variants'])):
        a = h7['y_variants'].get(f'{k}|EW', {})
        b = h7['y_variants'].get(f'{k}|VW', {})
        md.append(f"| {k} | {a.get('beta_pct', '—')}（{a.get('t_nw', '—')}） | {b.get('beta_pct', '—')}（{b.get('t_nw', '—')}） |")
    md.append(f"\n- 规模三分位内：{ {k: v['beta_pct'] for k, v in h7['size_tercile'].items()} }")
    md.append('- 持有期（收益 = monthly_returns_clean_v2 复利；REV2-F5 附月均折算）：\n')
    md.append('| 窗口 | 窗长(月) | n 行 | β %/单位 PC/期 | β 月均折算 %/月 | t(NW) |')
    md.append('|---|---|---|---|---|---|')
    for k, v in h7['holding_horizons'].items():
        if k == 'note':
            continue
        md.append(f"| {k} | {v.get('window_months', '—')} | {v['n_rows']:,} | {v['beta_pct_per_unit_period']} | "
                  f"{v.get('beta_pct_per_unit_per_month_avg', '—')} | {v['t_nw']} |")
    if h7.get('holding_horizons_rev2'):
        md.append(f"- {h7['holding_horizons_rev2']['reading']}")
    md.append('\n- 组合层：' + '；'.join(f"{k} α3 = {v.get('alpha_a3_pct')}%（t = {v.get('alpha_a3_t')}）"
                                        for k, v in h7['portfolios_alpha'].items()))
    md.append('\n## 8. H8 机械/结构\n')
    h8 = p['H8']
    sc = h8['pc_structure_corr']
    md.append(f"- corr(PC, log 团员数) = {sc['corr_PC_log_cluster_size']}、corr(PC, 自身收入份额) = {sc['corr_PC_own_revenue_share']}、"
              f"corr(PC, 团内收入 CV) = {sc['corr_PC_cluster_rev_cv']}、corr(PC, −行业 HHI) = {h8['industry_hhi']['corr_PC_indHHI_neg']}"
              f"（n = {sc['n_rows']:,}）")
    md.append('- PC 按团规模五分位：\n')
    md.append('| 组 | n | mean PC | mean HHI | 自身收入份额均值 | 中位团员数 |')
    md.append('|---|---|---|---|---|---|')
    for k, v in sc['pc_group_by_size'].items():
        md.append(f"| {k} | {v['n']:,} | {v['pc_mean']} | {v['hhi_mean']} | {v['own_share_mean']} | {v['peer_median']:.0f} |")
    rt = h8['residual_test']
    md.append(f"\n- 残差检验：{rt['spec']}；载荷 = {rt['loadings']}、R² = {rt['r2']}")
    for k, v in rt.items():
        if k.startswith('fm_'):
            md.append(f"  - FM({k[3:]}) = {v.get('beta_pct')}%（t = {v.get('t_nw')}）"
                      if 'beta_pct' in v else f"  - FM({k[3:]}) = {v}")
    md.append(f"- 持续性：池化 AR(1) corr = {h8['persistence']['pooled_AR1_corr']}、逐年 = "
              f"{[x['corr_PC'] for x in h8['persistence']['by_year_pairs']]}、"
              f"均值 |ΔPC| = {h8['persistence']['mean_pc_change_abs']}")

    # ---- REV2（F1）：H8 撤回与替代检验
    rv8 = h8.get('revision_tests', {})
    if rv8:
        wd = rv8.get('withdrawal', {})
        md.append('\n### 8R. H8 修订（REV2-F1）：撤回与替代检验（独立重算）\n')
        md.append(f"- **撤回**：{wd.get('retracted_conclusion', '')}")
        md.append('- 复核三条反驳（原文收录；每条后附本轮独立重算对照）：\n')
        for i, r in enumerate(wd.get('review_rebuttals_verbatim', []), 1):
            md.append(f"  {i}. {r}")
        idn = rv8.get('identity_decomposition', {})
        md.append(f"\n- (a) 恒等式分解（n = {idn.get('n_rows'):,}，恒等式残差 max = {idn.get('max_abs_resid_identity'):.2e}，"
                  f"两分量 corr = {idn.get('corr_two_components')}）：\n")
        md.append('| 分量 | mean | sd | FM β %/单位 | t(NW) |')
        md.append('|---|---|---|---|---|')
        for tag, lab in (('t_own', '−own_share²'), ('t_peer', '−同团他人集中度')):
            z = idn.get(tag, {})
            d0 = idn.get('descriptives', {}).get(tag, {})
            md.append(f"| {lab} | {d0.get('mean')} | {d0.get('sd')} | {z.get('beta_pct')} | {z.get('t_nw')} |")
        md.append(f"- 联合 FM：{idn.get('fm_joint')}（两分量各自为负、均不显著 → 不存在「某结构分量吸收负号」的证据）")
        cf = rv8.get('control_function_form', {}).get('variants', {})
        md.append(f"\n- (b) 机械控制函数形式：{cf.get('PC+invN', {}).get('PC', {})}（PC+1/N）、"
                  f"{cf.get('PC+invN+own_share', {}).get('PC', {})}（PC+1/N+own_share）、"
                  f"{cf.get('PC+own_share', {}).get('PC', {})}（PC+own_share）→ **1/N 形式仍为负**"
                  f"（对照 log 形式 +0.3221）")
        wnb = rv8.get('within_N_bins', {}).get('bins', {})
        md.append('\n- (c) within-N 分箱（同团规模内部梯度；min_n 放宽到 20，已标注）：\n')
        md.append('| 分箱 | n | 样本占比 | 中位 N | PC sd | FM β %/单位 | t(NW) |')
        md.append('|---|---|---|---|---|---|---|')
        for k, v2 in wnb.items():
            md.append(f"| {k} | {v2.get('n'):,} | {v2.get('sample_share')} | {v2.get('median_N')} | "
                      f"{v2.get('pc_sd')} | {v2.get('beta_pct', '—（无变异/样本不足）')} | {v2.get('t_nw', '—')} |")
        drr = rv8.get('distribution_reweighting', {})
        md.append('\n- (d) 分布匹配重加权（原文表 2 PC 分位密度；REV2 新增检验）：\n')
        md.append(f"  - 观测四分位份额 = {drr.get('observed_firm_year_bins', {}).get('obs_shares')}"
                  f"（目标 0.25×4；firm-year n = {drr.get('observed_firm_year_bins', {}).get('n_firm_year'):,}）、"
                  f"重要性权重 = {drr.get('observed_firm_year_bins', {}).get('bin_weights')}")
        md.append('| 实现 | 登记样本 β %/单位 | t(NW) | 原文最近似样本 β %/单位 | t(NW) |')
        md.append('|---|---|---|---|---|')
        for tag, lab in (('w_dist', '重要性重加权 WLS（四分位权重）'), ('w_dist_cap10', '同上 + 权重上限 10'),
                         ('quantile_match', '分位匹配（秩 → 原文分段线性分位函数）')):
            r_ = drr.get(f'registered|{tag}', {})
            pl = drr.get(f'paper_like|{tag}', {})
            md.append(f"| {lab} | {r_.get('beta_pct')} | {r_.get('t_nw')} | {pl.get('beta_pct')} | {pl.get('t_nw')} |")
        md.append(f"  - 重加权后加 5 控制：登记 β_PC = {drr.get('registered|w_dist+controls', {}).get('PC_beta_pct')}%"
                  f"（t = {drr.get('registered|w_dist+controls', {}).get('PC_t')}）；"
                  f"分位匹配五分位 EW：{drr.get('registered|quantile_match|quintile_EW')}")
        md.append(f"  - 变换后分布：{drr.get('transformed_distribution', {}).get('PC_qm')}"
                  f"（firm-year 均值/中位/p25/p75/sd）；权重摘要 {drr.get('transformed_distribution', {}).get('weight_summary')}")
        pmp = rv8.get('cluster_label_permutation', {})
        md.append(f"\n- (e) 团标签置换 **{pmp.get('n_iter')} 次**（逐年重贴「团 → HHI」对应）："
                  f"零分布 mean = {pmp.get('perm_beta_mean_pct')}%、sd = {pmp.get('perm_beta_sd_pct')}%、"
                  f"CI95 = {pmp.get('perm_ci95_pct')}%（min {pmp.get('perm_min_pct')} / max {pmp.get('perm_max_pct')}）；"
                  f"观测 β = {pmp.get('observed_beta_pct')}% → **单侧 p = {pmp.get('p_one_sided_le_observed')}**、"
                  f"双侧 p = {pmp.get('p_two_sided')}；置换 MDE(80%) ≈ {pmp.get('mde80_from_perm_sd_pct')}%")
        pmd = rv8.get('power_mde', {}).get('main_reading_registered_univariate', {})
        md.append(f"\n- (f) 功效 / MDE（主读数）：se = {pmd.get('se_pct')}%、CI95 = {pmd.get('ci95_pct')}%、"
                  f"MDE(80% 功效) = {pmd.get('mde80_pct')}%/单位、原文效应 +{pmd.get('effect_ref_pct')} 处功效 = "
                  f"**{pmd.get('power_at_effect_ref')}** → 在 t = {pmd.get('t_nw')}、四层子读数全不显著的前提下，"
                  f"只能写「**未检出正号（不显著）**」，不得写「稳健反号」")
        md.append(f"- 8R 判据：{rv8.get('status', '')}")

    md.append('\n## 9. 根因判定（修正版，REV2）\n')
    rc = S.get('revised_root_cause', {})
    if rc.get('final_conclusion'):
        md.append(f"**总结论（一句话）**：{rc['final_conclusion']}\n")
    if rc.get('retracted'):
        md.append('**撤回**：\n')
        for it in rc['retracted']:
            md.append(f'- {it}')
    if rc.get('excluded_strong'):
        md.append('\n**已排除（强证据）**：\n')
        for it in rc['excluded_strong']:
            md.append(f'- {it}')
    if rc.get('pending'):
        md.append('\n**待定**：\n')
        for it in rc['pending']:
            md.append(f'- {it}')
    md.append('\n（候选排序与支持度见下表；#1 已撤回，#2/#3 保持待定）\n')
    if 'summary_ranking' in S:
        md.append('| 排序 | 候选 | 支持度 | 依据 |')
        md.append('|---|---|---|---|')
        for i, it in enumerate(S['summary_ranking'], 1):
            md.append(f"| {i} | {it['candidate']} | {it['support']} | {it['evidence']} |")
    if 'retracted_conclusions' in S:
        md.append('\n**撤回/收回/修正明细**：\n')
        for it in S['retracted_conclusions']:
            md.append(f'- {it}')
    if 'open_items' in S:
        md.append('\n**仍未排除 / 不可测**：\n')
        for it in S['open_items']:
            md.append(f'- {it}')
    if 'next_steps' in S:
        md.append('\n**下一步建议**：\n')
        for it in S['next_steps']:
            md.append(f'- {it}')

    md.append('\n## 10. 修订记录（REV2，复核意见 F1–F6 响应）\n')
    rv = S.get('revision', {})
    md.append(f"- 修订轮：{rv.get('revision_id')}（{rv.get('date')}）；触发：{rv.get('trigger')}")
    for it in rv.get('changes', []):
        md.append(f'- {it}')
    md.append('\n**修法 → 修正后数值/原文摘录（重点）**：\n')
    # 自包含取值（不依赖上文块内变量，防块间耦合）
    _rt10 = p.get('H8', {}).get('revision_tests', {})
    _drr = _rt10.get('distribution_reweighting', {})
    _pmp = _rt10.get('cluster_label_permutation', {})
    _wnb = _rt10.get('within_N_bins', {}).get('bins', {})
    _ncp = _rt10.get('control_function_form', {}).get('variants', {})
    _cs6 = p.get('H6', {}).get('cluster_size_control', {})
    _psa = p.get('H6', {}).get('paper_sign_agreement', {})
    _y19 = p.get('H5', {}).get('by_return_year', {}).get('paper_like', {}).get('2019', {})
    _wpa = p.get('H2', {}).get('write_precision_audit', {})
    md.append(f'- F1 分布匹配重加权（新检验）：重要性 WLS（原文四分位份额 → 权重 '
              f'{_drr.get("observed_firm_year_bins", {}).get("bin_weights")}）β = '
              f'{_drr.get("registered|w_dist", {}).get("beta_pct")}%（t = {_drr.get("registered|w_dist", {}).get("t_nw")}）、'
              f'分位匹配（秩 → 原文分段线性分位函数）β = {_drr.get("registered|quantile_match", {}).get("beta_pct")}%'
              f'（t = {_drr.get("registered|quantile_match", {}).get("t_nw")}）；原文最近似样本 = '
              f'{_drr.get("paper_like|w_dist", {}).get("beta_pct")}%（WLS）/'
              f'{_drr.get("paper_like|quantile_match", {}).get("beta_pct")}%（分位匹配）→ **两者均未转正**（分布口径不解释负号）')
    md.append(f'- F1 团标签置换 **{_pmp.get("n_iter")} 次**：mean = {_pmp.get("perm_beta_mean_pct")}%、'
              f'sd = {_pmp.get("perm_beta_sd_pct")}、CI95 = {_pmp.get("perm_ci95_pct")}%、'
              f'**单侧 p = {_pmp.get("p_one_sided_le_observed")}**（恰在 5% 临界）、双侧 p = {_pmp.get("p_two_sided")}、'
              f'置换 MDE(80%) ≈ {_pmp.get("mde80_from_perm_sd_pct")}%'
              f'（复核实测 200 次对照：mean +0.049 / sd 0.296 / CI95 [−0.498, +0.585] / P(≤−0.5068) = 2.5%）')
    md.append('- F1 within-N 表（§8R(c) 汇总）：' +
              '；'.join(f"{k} β={v2.get('beta_pct')}（t={v2.get('t_nw')}；share={v2.get('sample_share')}）"
                       for k, v2 in _wnb.items()))
    md.append(f'- F2 H5 统计量并报（数值见 §5）：2019 原文最近似样本 t_nw = {_y19.get("t_nw")}、'
              f't_iid = {_y19.get("t_iid")}、bootstrap CI95 {_y19.get("boot_ci95_pct")} 含 0、'
              f'LOO β 区间 {_y19.get("loo_beta_range_pct")}；正号且 CI 不含 0 的年 = 无 → 判据改 bootstrap CI 后'
              f'**无正面证据**；根因排序 #2 的「2019 显著为正」已收回')
    md.append(f'- F3 H6 归因收窄：log 形式 β_PC = {_cs6.get("PC_plus_log_peers", {}).get("PC_beta_pct")}%'
              f'（t = {_cs6.get("PC_plus_log_peers", {}).get("PC_t")}，不显著）；1/N（团成员数）'
              f'{_ncp.get("PC+invN", {}).get("PC", {}).get("beta_pct")}%、+own_share '
              f'{_ncp.get("PC+invN+own_share", {}).get("PC", {}).get("beta_pct")}% 仍为负；'
              f'控制项符号相反仅 {_psa.get("n_opposite_sign")}/{_psa.get("n_controls")}（{_psa.get("opposite_vars")}）')
    md.append('- F4 精度/强度表述：H2「max|Δhhi| = 0」→「≤ 5e-9（round(x,8) 写盘精度；最大团实测 '
              f'{_wpa.get("csv_hhi_max_abs_diff_raw", float("nan")):.3e}）」；H3「11/11 为负」→「无正号'
              '（多档 |t| ≤ 0.55，均不显著，最大 |t| = 1.80）」；H3 分布「高度吻合」→「中心四分位吻合（p25/中位/p75），'
              'sd 0.2204 vs 原文 0.337」；H2 营收字段 →「盘上 pkl = 营业收入（854.30 亿），原文附录表未核」')
    md.append('- F5 H7 单位与重叠窗：持有期 β 折算月均 = ' +
              str({k: v.get('beta_pct_per_unit_per_month_avg')
                   for k, v in p.get('H7', {}).get('holding_horizons', {}).items() if k.startswith('h_')}) +
              ' → 无反转、月均幅度基本不变；t+2..6 / t+7..12 等为重叠窗且 NW(6) 未做重叠校正 → t 不作推断依据')
    md.append('- F6 治理：本轮产物登记进 version_manifest（诊断脚本 + 8 分片 + 报告两件 + dcoarse 证据 = 12 条）；'
              'registry/spec 三角刷新；四审计顺序重跑；五 YAML + 三哈希闭环 + manifest 全量 sha 扫描（见 §11）')
    if 'excluded' in S:
        md.append('\n**已排除（原表）**：' + '；'.join(S['excluded']))
    md.append('\n## 11. 产物与纪律\n')
    md.append('- 新建脚本：`scripts/competition_sign_diagnostic.py`；分片 JSON：`artifacts/tmp/pc_diag/H1..H8.json`；'
              '进度：`artifacts/tmp/pc_sign_diag_progress.md`；本报告两件。')
    md.append('- 复用（只读）：`artifacts/tmp/pc_rc/h1_net_t*/`（K 档 PC 缓存）、'
              '`data/processed/network_ma_teacher/{2017..2022}_pc.csv` + `clusters/`、'
              '`data/processed/pricing_sample_ma_teacher_v2.parquet`、`data/processed/monthly_returns_clean_v2.csv`、'
              '`data/processed/factor_returns_daily.csv`、`data/raw/drive/operating_revenue_annual.pkl`、'
              '`data/processed/sw2_industry_latest.csv`。')
    md.append('- 本轮新建（粗 K 档，未覆盖既有）：`artifacts/tmp/pc_rc/dcoarse_net_t{150,200,300}/`（仓库重建器新建 tag）、'
              '`artifacts/tmp/pc_rc/rc_rebuild_dcoarse.json`、`artifacts/tmp/pc_diag/{H1..H8}.json` + '
              '`artifacts/tmp/pc_diag/{pc_report_dcoarse_t*,dcoarse_*,run_*,h3_closeout}.log`。')
    md.append('- REV2 复核证据（只读引用，不在本轮重算其数值）：`artifacts/tmp/review_diag/{rv_lib.py,A_injection.py,'
              'B_h2.py,C_h3h4.py,D_h5.py,E_h6.py,F_h8.py,F2_h8_identity.py}` 及同名 JSON。')
    md.append('- 治理登记（REV2-F6，与本报告同轮）：本轮产物（诊断脚本 + 8 分片 + 报告两件 + dcoarse 证据）登记进 '
              '`configs/version_manifest.yaml`；registry 追加 change_log + spec 双锚点刷新（三角刷新）；'
              'Gate 1 四审计按 entry → provenance → metadata → consistency 顺序重跑；五 YAML safe_load + 重复键；'
              '三哈希闭环 + manifest 全量 sha 扫描 —— 终态证据 = `artifacts/tmp/pc_diag/diag_governance.json` '
              '与 `artifacts/tmp/pc_diag/diag_closeout_check.json`。')
    md.append('- 未覆盖任何既有产物、未改任何 headline_value / 冻结值 / 主口径网络；注入检验仅在内存构造合成列；'
              'REV2 的分布重加权/置换/功效检验全部只读（内存计算），不改任何盘上数据。')
    md.append('')
    OUT_MD.write_text('\n'.join(md), encoding='utf-8', newline='\n')
    print(f'  -> {OUT_MD}')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


DIAG.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- 装载


# ================================================================ H1


# ================================================================ H2


# ================================================================ H3


# ================================================================ H4


# ================================================================ H5


# ================================================================ H6


# ================================================================ H7


# ================================================================ H8


# ================================================================ 装配


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', nargs='*', default=None)
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--panel', default=str(PANEL))
    args = ap.parse_args()
    if args.assemble:
        assemble()
        return 0
    tests = [t.upper() for t in (args.tests or [])]
    panel = load_panel()
    pc = load_pc()
    m = merge_pc(panel, pc)
    S = make_samples(m)
    fmonth = monthly_factors()
    for t in tests:
        if t == 'H1':
            run_H1(m, S, fmonth)
        elif t == 'H2':
            run_H2(m, S, fmonth, panel)
        elif t == 'H3':
            run_H3(panel, fmonth)
        elif t == 'H4':
            run_H4(m, S, panel, fmonth)
        elif t == 'H5':
            run_H5(m, S, fmonth)
        elif t == 'H6':
            run_H6(m, S, fmonth)
        elif t == 'H7':
            run_H7(m, S, fmonth, panel)
        elif t == 'H8':
            run_H8(m, S, panel, fmonth)
        else:
            print(f'unknown test {t}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
