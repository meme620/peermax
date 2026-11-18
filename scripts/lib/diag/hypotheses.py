"""s3_pc_sign_diag_h1h8 的 hypotheses 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.diag.consts import CLUSTER_DIR
from lib.diag.consts import CONTROLS
from lib.diag.consts import NW
from lib.diag.consts import NW_PAPER
from lib.diag.consts import PAPER_CONTROLS
from lib.diag.consts import PAPER_T2
from lib.diag.consts import PAPER_TARGET_EFFECT
from lib.diag.consts import PC_DIR
from lib.diag.consts import PC_YEARS
from lib.diag.consts import RETURNS
from lib.diag.consts import REV_PKL
from lib.diag.consts import ROOT
from lib.diag.consts import SEED
from lib.diag.consts import SW2_MAP
from lib.diag.consts import TIER_DIR
from lib.diag.data import load_pc
from lib.diag.data import load_rev
from lib.diag.data import merge_pc
from lib.diag.data import sw2_asof
from lib.diag.stats import ctrl_compare
from lib.diag.stats import fm
from lib.diag.stats import make_samples
from lib.diag.stats import portfolio_key
from lib.diag.util import dump
from lib.diag.util import pct
from lib.diag.util import power_mde
from lib.diag.util import psay
from lib.diag.util import r4
from s3_m1_lib import (  # noqa: E402
    fm_regression, newey_west, param_value, rank_average_groups,
)
import json
import numpy as np
import pandas as pd

def run_H1(m, S, fmonth) -> dict:
    psay('H1', '状态 = 开始：H1 方向/排序 bug（注入检验 + 组均值核验 + 手算 OLS）')
    reg, paper = S['registered'], S['paper_like']
    out = {'hypothesis': 'H1 代码里的排序/高低方向是否反了（H/L 定义、PC 排序方向、FM 符号约定）'}

    # ---- (1) 逐行核读（静态证据，附数字核验）
    diff = float((m['PC'] + m['hhi']).abs().max())
    out['code_audit'] = {
        'pc_is_negative_hhi': bool(diff < 1e-9), 'max_abs_pc_plus_hhi': r4(diff),
        'pc_formula_src': 'scripts/data_audit/s3_ma_pc_hhi.py L152-153: shares=v/tot（含 focal 自身）；'
                          'hhi=(shares**2).sum()；pc=-hhi',
        'group_src': 'scripts/data_audit/s3_m1_lib.py L243-252 rank_average_groups：rank 升序 → '
                     '组号随取值升序 → g{N} = PC 最大组；对表脚本 L371 HL = g{n_groups} − g1 = 高 − 低',
        'fm_src': 'scripts/data_audit/s3_pc_pricing_correspondence.py L319: fm_regression(samp, "R_next", ["PC"]) '
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
    m2 = m.copy()
    signs = [a.get('csv_col_equals_minus_HHI_excl') for a in audits if a.get('csv_col_equals_minus_HHI_excl') is not None]
    col_is_minus = bool(signs) and all(signs)
    m2['PC_x'] = (pd.to_numeric(m2['hhi_excl_self'], errors='coerce') if col_is_minus
                  else -pd.to_numeric(m2['hhi_excl_self'], errors='coerce'))
    m2['PC_x'] = m2['PC_x'].where(m2['PC'].notna())
    pc_ok = m2['PC_x'].notna() & m2['R_next'].notna()
    ctrl_ok = m2[CONTROLS].notna().all(axis=1)
    regx = m2[pc_ok & m2['m1_in_fm_sample'].astype(bool)]
    paperx = m2[pc_ok]
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
    for f in ('s3_ma_K_explore.json', 's3_ma_K_explore2.json'):
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
        'A_registered': pc_ok & m['m1_in_fm_sample'].astype(bool),
        'B_paper_like': pc_ok,
        'C_paper_like_ctrl5': pc_ok & ctrl5,
        'D_no_analyst_gate': pc_ok & ctrl5 & (~ana_ok),          # 仅因分析师覆盖缺失而被剔的行
        'E_only_analyst_gate_excluded': pc_ok & (~m['m1_in_fm_sample'].astype(bool)),
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
    for tag, mask in (('F_registered_plus_microcap', pc_ok & (m['m1_in_fm_sample'].astype(bool) | m['me_below_q10'].astype(bool))),
                      ('G_registered_excl_microcap', pc_ok & m['m1_in_fm_sample'].astype(bool))):
        s = m[mask]
        b1 = fm(s, 'R_next', ['PC'])
        variants[tag] = {'n_rows': int(len(s)), 'fm_univariate': {'beta_pct': pct(b1['coefs']['PC']['beta']),
                                                                  't_nw': r4(b1['coefs']['PC']['t_nw'])}}
        psay('H4', f"变体 {tag} 状态 = ok：n = {len(s):,}、β = {variants[tag]['fm_univariate']['beta_pct']}%")
    # H：全 PC 宇宙（无 M1 筛选，无分析师门；收益取 monthly_returns_clean_v2）
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
            'note': 'PC 全宇宙（无 M1 筛选、无分析师门、无微盘剔除、无控制要求）；收益 = monthly_returns_clean_v2 原始月收益'}
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
    m2 = reg.copy()
    m2['sw2'] = sw2_asof(m2['month'], m2['focal_id'])
    cols = ['R_next', 'PC'] + CONTROLS
    dd = m2.dropna(subset=cols + ['sw2']).copy()
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
        s2 = s.merge(fy[['focal_id', 'text_year', 'w_dist', 'w_dist_cap10', 'PC_qm']],
                     on=['focal_id', 'text_year'], how='left')
        for wtag in ('w_dist', 'w_dist_cap10'):
            b = fm(s2, 'R_next', ['PC'], weight_col=wtag)
            dist[f'{tag}|{wtag}'] = {'beta_pct': pct(b['coefs']['PC']['beta']),
                                     't_nw': r4(b['coefs']['PC']['t_nw']), 'n_months_used': b['n_months_used']}
        b2 = fm(s2, 'R_next', ['PC'] + CONTROLS, weight_col='w_dist')
        dist[f'{tag}|w_dist+controls'] = {'PC_beta_pct': pct(b2['coefs']['PC']['beta']),
                                          'PC_t': r4(b2['coefs']['PC']['t_nw'])}
        b3 = fm(s2, 'R_next', ['PC_qm'])
        dist[f'{tag}|quantile_match'] = {'beta_pct': pct(b3['coefs']['PC_qm']['beta']),
                                         't_nw': r4(b3['coefs']['PC_qm']['t_nw']),
                                         'n_months_used': b3['n_months_used']}
        p_qm = portfolio_key(s2, 'PC_qm', 5, 'EW', None, fmonth=fmonth)
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

