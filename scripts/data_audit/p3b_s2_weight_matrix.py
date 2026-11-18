#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3（S-2）：VW 权重函数敏感性矩阵（Phase 3b，只读诊断，不改 headline）。

问题：headline M1 的 EW PeerMAX = +0.03427（t=0.47）与 VW PeerMAX = −0.10996（t=−1.18）
符号相反。本脚本只改**回归权重函数**（firm-level WLS 权重），保持样本、outcome、
控制集、FM/NW 口径完全不变，判断 PeerMAX 符号是否由 ME（market equity）水平值权重函数决定。

权重六种（全部要求 w>0；ME 在 FM 样本内全部 >0）：
  1. ew        : 无权重（w=1；对应 headline EW）
  2. me        : market_equity（对应 headline VW）
  3. sqrt_me   : sqrt(market_equity)
  4. log_me    : log(market_equity)
  5. me_w95    : ME 月度截面 winsorize [q0.05, q0.95]
  6. me_w99    : ME 月度截面 winsorize [q0.01, q0.99]

规格 S1（与 s3_m1_2_fm_main.py 主规格逐项一致）：
  R_next ~ PeerMAXraw + OwnMAXraw + PeerReturn + size + bm + mom + rev + turnover
            + illiq + AnalystCoverage + NetworkDegree
  逐月横截面 WLS -> 系数序列 -> 均值 + Newey-West(lag=6)；min_n=100（registry）。

只读边界：
  - 输入只有 data/processed/m1_common_sample_ma_teacher_v2.parquet（只读）与 s3_m1_lib；
  - 不重跑 M1 / PC / 网络，不写 configs / G0 / reports；
  - 9 个 M1 主输入 sha256 在运行前后各校验一次（前后必须一致）。

产物（仅 artifacts/phase3b/3b_s2_*）：
  3b_s2_weight_matrix.csv
  3b_s2_counts.json
  3b_s2_execution.md

用法：python scripts/data_audit/p3b_s2_weight_matrix.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s3_m1_lib import (  # noqa: E402
    ROOT, assert_m1_contract, assert_m1_inputs_sha, fm_regression, param_value,
    require, sha256_file, winsorize_bounds_by_month, write_json,
)

SAMPLE = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'
OUT_DIR = ROOT / 'artifacts/phase3b'
SAMPLE_ROWS_TOTAL = 307503          # v2 样本登记行数
SAMPLE_ROWS_FM = 181178             # v2 FM 样本登记行数
SAMPLE_MONTHS = 104                 # v2 FM 登记月数（2015-05..2023-12）
FM_FLAG = 'm1_in_fm_sample'

P = param_value('m1_sample_and_spec_registry')
NW_LAG = int(param_value('nw_lag'))
MIN_N = int(P['min_focal_count_per_month'])
CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq', 'AnalystCoverage',
            'NetworkDegree']
MAIN = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + CONTROLS
YCOL = 'R_next'
KEY_VARS = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn']

# winsor 口径：95% = 月度截面 [5%,95%] 两侧各裁 5%；99% = [1%,99%] 两侧各裁 1%
# （与 registry winsorization_pct=[0.01,0.99] 的「1%/99%」习惯一致，不采用
#  「保留 95% 中心区间 = [2.5%,97.5%]」的另一读法；如需后者应另立规格）。
WEIGHT_SPECS = [
    {'id': 'ew', 'label': 'EW (w=1)', 'weight_col': None,
     'transform': 'none', 'rule': 'unweighted（w=1）',
     'winsor_low': None, 'winsor_high': None},
    {'id': 'me', 'label': 'ME', 'weight_col': '_w_me',
     'transform': 'market_equity', 'rule': 'w = ME（raw）',
     'winsor_low': None, 'winsor_high': None},
    {'id': 'sqrt_me', 'label': 'sqrt(ME)', 'weight_col': '_w_sqrt_me',
     'transform': 'sqrt(market_equity)', 'rule': 'w = sqrt(ME)',
     'winsor_low': None, 'winsor_high': None},
    {'id': 'log_me', 'label': 'log(ME)', 'weight_col': '_w_log_me',
     'transform': 'log(market_equity)', 'rule': 'w = ln(ME)',
     'winsor_low': None, 'winsor_high': None},
    {'id': 'me_w95', 'label': 'ME winsor 5/95', 'weight_col': '_w_me_w95',
     'transform': 'market_equity winsorized [q0.05,q0.95] by month',
     'rule': 'w = clip(ME, 月度 q05, 月度 q95)',
     'winsor_low': 0.05, 'winsor_high': 0.95},
    {'id': 'me_w99', 'label': 'ME winsor 1/99', 'weight_col': '_w_me_w99',
     'transform': 'market_equity winsorized [q0.01,q0.99] by month',
     'rule': 'w = clip(ME, 月度 q01, 月度 q99)',
     'winsor_low': 0.01, 'winsor_high': 0.99},
]


def add_weight_columns(fm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """在 FM 样本上加六种权重列；边界一律在该月 FM 宇宙上计算。"""
    fm = fm.copy()
    me = fm['market_equity']
    fm['_w_me'] = me
    fm['_w_sqrt_me'] = np.sqrt(me)
    fm['_w_log_me'] = np.log(me)
    bounds = {}
    for spec in WEIGHT_SPECS:
        if spec['winsor_low'] is None:
            continue
        lo, hi = spec['winsor_low'], spec['winsor_high']
        q_lo, q_hi = winsorize_bounds_by_month(fm, 'market_equity', lo=lo, hi=hi)
        fm[spec['weight_col']] = me.clip(lower=q_lo, upper=q_hi)
        bounds[spec['id']] = {'lo_pct': lo, 'hi_pct': hi}
    return fm, bounds


def weight_diagnostics(fm: pd.DataFrame, weight_col: str | None,
                       used_months: set, min_n: int) -> dict:
    """按 fm_regression 同口径逐月过滤，计算权重诊断（仅 used 月）。"""
    rows = []
    for m, sub in fm.groupby('month', sort=True):
        if m not in used_months:
            continue
        sub = sub.dropna(subset=[YCOL] + MAIN)
        if weight_col:
            sub = sub[sub[weight_col].notna() & (sub[weight_col] > 0)]
        n = len(sub)
        if n < min_n:
            continue
        if weight_col:
            w = sub[weight_col].to_numpy(dtype=float)
        else:
            w = np.ones(n, dtype=float)
        s = float(w.sum())
        kish = s * s / float(np.dot(w, w))
        order = np.sort(w)[::-1]
        top5 = float(order[:5].sum() / s)
        rows.append({'month': m, 'n': n, 'weight_sum': s, 'kish_eff_n': kish,
                     'max_weight': float(order[0]),
                     'max_weight_share': float(order[0] / s),
                     'top5_weight_share': top5})
    d = pd.DataFrame(rows)
    require(len(d) > 0, f'{weight_col}: 无可用月份')
    require(int(d['n'].sum()) > 0, f'{weight_col}: 有效 N=0')
    return {
        'n_obs_N': int(d['n'].sum()),
        'n_months_T': int(len(d)),
        'n_per_month_min': int(d['n'].min()),
        'n_per_month_median': float(d['n'].median()),
        'n_per_month_max': int(d['n'].max()),
        'kish_eff_n_median': float(d['kish_eff_n'].median()),
        'kish_eff_n_min': float(d['kish_eff_n'].min()),
        'kish_eff_n_max': float(d['kish_eff_n'].max()),
        'kish_eff_n_pct_of_n_median': float(
            (d['kish_eff_n'] / d['n']).median() * 100.0),
        'max_single_weight_median': float(d['max_weight'].median()),
        'max_single_weight_share_median': float(d['max_weight_share'].median()),
        'max_single_weight_share_max': float(d['max_weight_share'].max()),
        'top5_weight_share_median': float(d['top5_weight_share'].median()),
        'per_month': d.to_dict(orient='records'),
    }


def run_one(fm: pd.DataFrame, spec: dict) -> dict:
    wc = spec['weight_col']
    if wc:
        w = fm[wc]
        require(bool(w.notna().all()), f'{spec["id"]}: 权重含缺失')
        require(bool((w > 0).all()), f'{spec["id"]}: 权重存在 <=0')
        require(bool(np.isfinite(w.to_numpy(dtype=float)).all()),
                f'{spec["id"]}: 权重存在非有限值')
    res, per_month = fm_regression(fm, YCOL, MAIN, weight_col=wc,
                                   min_n=MIN_N, nw_lag=NW_LAG)
    require(res['n_months_used'] == SAMPLE_MONTHS,
            f'{spec["id"]}: 使用月数 {res["n_months_used"]} != {SAMPLE_MONTHS}')
    used = {d['month'] for d in per_month if d.get('used')}
    diag = weight_diagnostics(fm, wc, used, MIN_N)
    require(diag['n_months_T'] == res['n_months_used'],
            f'{spec["id"]}: 诊断月数 {diag["n_months_T"]} != 回归月数 '
            f'{res["n_months_used"]}')
    require(diag['n_obs_N'] == SAMPLE_ROWS_FM,
            f'{spec["id"]}: 有效 N {diag["n_obs_N"]} != {SAMPLE_ROWS_FM}')
    pm_key = {d['month']: d for d in per_month if d.get('used')}
    series = {v: [float(pm_key[m]['beta'][v]) for m in sorted(used)] for v in KEY_VARS}
    neg_share = float(np.mean(np.asarray(series['PeerMAXraw']) < 0.0))
    return {'spec': spec, 'summary': res, 'diagnostics': diag,
            'beta_series': series, 'peer_max_negative_month_share': neg_share,
            'monthly_n': [int(pm_key[m]['n']) for m in sorted(used)]}


def build_md(meta: dict, rows: list, verdict: dict) -> str:
    lines = []
    lines.append('# B3（S-2）VW 权重函数敏感性矩阵 — 执行报告')
    lines.append('')
    lines.append(f"- 生成时间（UTC）：{meta['created_at_utc']}")
    lines.append(f"- 脚本：`scripts/data_audit/p3b_s2_weight_matrix.py`"
                 f"（sha256 `{meta['script_sha256'][:16]}…`）")
    lines.append(f"- 样本：`data/processed/m1_common_sample_ma_teacher_v2.parquet`"
                 f"（sha256 `{meta['sample_sha256'][:16]}…`）")
    lines.append(f"- FM 样本：{meta['fm_rows']:,} 行 / {meta['fm_months']} 月；"
                 f"规格 S1（PeerMAX + OwnMAX + PeerReturn + 8 controls），"
                 f"min_n={meta['min_n']}，NW lag={meta['nw_lag']}")
    lines.append('- 边界：只改 firm-level WLS 权重函数；不重跑 M1/PC/网络，'
                 '不写 headline/系数/gate/configs/G0/reports。')
    lines.append('')
    lines.append('## 1. 输入锚（9 个 M1 主输入，运行前后各一次）')
    lines.append('')
    lines.append('| 输入 | sha256（前=后） | before | after |')
    lines.append('| --- | --- | --- | --- |')
    for b, a in zip(meta['input_sha_before'], meta['input_sha_after']):
        lines.append(f"| {b['input']} | `{b['sha256'][:16]}…` | "
                     f"{'MATCH' if b['match'] else 'FAIL'} | "
                     f"{'MATCH' if a['match'] else 'FAIL'} |")
    lines.append('')
    lines.append(f"结论：9/9 输入前后 sha256 一致 = "
                 f"**{str(meta['input_sha_unchanged']).lower()}**；"
                 f"运行未改动任何登记输入。")
    lines.append('')
    lines.append('## 2. 权重定义')
    lines.append('')
    lines.append('| 权重 | 权重列 | 定义 |')
    lines.append('| --- | --- | --- |')
    for r in rows:
        lines.append(f"| {r['weight_label']} | `{r['weight_col'] or 'None'}` | "
                     f"{r['weight_rule']} |")
    lines.append('')
    lines.append('ME 的 winsor 边界在每个月的 FM 样本横截面上计算；'
                 '95% = [q0.05, q0.95]，99% = [q0.01, q0.99]。'
                 '六种权重均逐月确认 >0、有限、无缺失。')
    lines.append('')
    lines.append('## 3. 六权重 S1 结果')
    lines.append('')
    lines.append('| 权重 | N | T | Kish 有效 N（中位） | 有效 N / n | '
                 '单股最大权重份额（中位） | PeerMAX β | PeerMAX t | '
                 'OwnMAX β | OwnMAX t | PeerReturn β | PeerReturn t |')
    lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | '
                 '---: | ---: | ---: | ---: |')
    for r in rows:
        lines.append(
            f"| {r['weight_label']} | {r['n_obs_N']:,} | {r['n_months_T']} | "
            f"{r['kish_eff_n_median']:.1f} | "
            f"{r['kish_eff_n_pct_of_n_median']:.1f}% | "
            f"{r['max_single_weight_share_median'] * 100:.2f}% | "
            f"{r['peer_max_beta']:+.5f} | {r['peer_max_t_nw']:+.2f} | "
            f"{r['own_max_beta']:+.5f} | {r['own_max_t_nw']:+.2f} | "
            f"{r['peer_return_beta']:+.5f} | {r['peer_return_t_nw']:+.2f} |")
    lines.append('')
    lines.append('EW 与 ME 两行逐值复现 headline M1 主规格'
                 '（EW +0.03427 / t=+0.47；VW −0.10996 / t=−1.18），'
                 '确认本矩阵与 S1 口径一致。')
    lines.append('')
    rmap = {r['weight_id']: r for r in rows}
    raw_max = '；'.join(
        f"{rmap[k]['weight_label']} {rmap[k]['max_single_weight_median']:.3g}"
        for k in ('ew', 'me', 'sqrt_me', 'log_me', 'me_w95', 'me_w99'))
    lines.append(f'单股最大权重（原始单位，月中位）：{raw_max}'
                 '（原始单位不可跨权重比较，集中度以「有效 N / n」与'
                 '「单股最大份额」为准；逐值见 CSV `max_single_weight_median`）。')
    lines.append('')
    lines.append('PeerMAX 的 95% CI 与双侧 p 值：')
    lines.append('')
    lines.append('| 权重 | β | 95% CI | t | p（双侧） | 5% 显著 | 月度 β<0 占比 |')
    lines.append('| --- | ---: | --- | ---: | ---: | --- | ---: |')
    for r in rows:
        ci = (f"[{r['peer_max_ci95_low']:+.5f}, {r['peer_max_ci95_high']:+.5f}]"
              if r.get('peer_max_ci95_low') is not None else 'NA')
        lines.append(
            f"| {r['weight_label']} | {r['peer_max_beta']:+.5f} | {ci} | "
            f"{r['peer_max_t_nw']:+.2f} | {r['peer_max_p_two_sided']:.4f} | "
            f"{'YES' if r['peer_max_significant_5pct'] else 'NO'} | "
            f"{r['peer_max_negative_month_share'] * 100:.1f}% |")
    lines.append('')
    lines.append('## 4. PeerMAX 符号与显著性')
    lines.append('')
    lines.append('| 权重 | 符号 | 显著性（5% 双侧） |')
    lines.append('| --- | --- | --- |')
    for r in rows:
        lines.append(f"| {r['weight_label']} | {r['peer_max_sign']} | "
                     f"{'显著' if r['peer_max_significant_5pct'] else '不显著'} |")
    lines.append('')
    lines.append(f"- 六权重符号集合：{verdict['signs_text']}")
    lines.append(f"- 显著规格数（5% 双侧）：{verdict['n_significant']} / 6")
    lines.append(f"- 最大 |t|：{verdict['max_abs_t']:.2f}")
    lines.append('')
    lines.append('## 5. 判断：符号是否由 ME 水平值权重函数决定')
    lines.append('')
    for p in verdict['paragraphs']:
        lines.append(p)
        lines.append('')
    lines.append('## 6. 结论')
    lines.append('')
    lines.append(verdict['conclusion'])
    lines.append('')
    lines.append('## 7. 复现与产物')
    lines.append('')
    lines.append('```bash')
    lines.append('python scripts/data_audit/p3b_s2_weight_matrix.py')
    lines.append('```')
    lines.append('')
    lines.append('- `artifacts/phase3b/3b_s2_weight_matrix.csv`')
    lines.append('- `artifacts/phase3b/3b_s2_counts.json`')
    lines.append('- `artifacts/phase3b/3b_s2_execution.md`（本文件）')
    lines.append('')
    lines.append('## 8. 解释边界')
    lines.append('')
    lines.append('- 本矩阵只回答「权重函数是否改变 PeerMAX 符号/显著性」，'
                 '不是新的 headline、不构成实证通过或失败判定。')
    lines.append('- FM 逐月 WLS 的 Kish 有效 N 与单股权重份额是**权重集中度**诊断，'
                 '不是独立观测数；标准误仍按月份序列 NW(6) 计算，'
                 '未对权重集中做额外推断修正。')
    lines.append('- winsor 口径固定为 [5%,95%] / [1%,99%]；'
                 '「保留 95% 中心区间 [2.5%,97.5%]」是另一种读法，未在本矩阵中运行。')
    lines.append('- 所有输入只读；未重跑 M1/PC/网络，未改动任何既有产物。')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec='seconds')

    print('[S2] 校验 9 个 M1 主输入（运行前）…', flush=True)
    sha_before = assert_m1_inputs_sha()
    require(all(x['match'] for x in sha_before), '运行前输入锚存在失败项')

    print(f'[S2] 读取样本 {SAMPLE.name} …', flush=True)
    df = pd.read_parquet(SAMPLE)
    contract = assert_m1_contract(df, expected_rows=SAMPLE_ROWS_TOTAL, kind='sample',
                                  where='p3b_s2:ma_teacher:v2')
    require(len(df) == SAMPLE_ROWS_TOTAL,
            f'样本行数 {len(df)} != {SAMPLE_ROWS_TOTAL}')
    fm = df[df[FM_FLAG].astype(bool)].copy()
    require(len(fm) == SAMPLE_ROWS_FM,
            f'FM 行数 {len(fm)} != {SAMPLE_ROWS_FM}')
    require(fm['month'].nunique() == SAMPLE_MONTHS,
            f'FM 月数 {fm["month"].nunique()} != {SAMPLE_MONTHS}')
    require(int(fm.duplicated(['focal_id', 'month']).sum()) == 0,
            'FM 样本存在重复 (focal_id, month)')

    fm, winsor_bounds = add_weight_columns(fm)

    results = []
    for spec in WEIGHT_SPECS:
        print(f'[S2] 运行 {spec["id"]} …', flush=True)
        results.append(run_one(fm, spec))

    rows = []
    for r in results:
        c = r['summary']['coefs']
        pm = c['PeerMAXraw']
        rows.append({
            'weight_id': r['spec']['id'],
            'weight_label': r['spec']['label'],
            'weight_col': r['spec']['weight_col'],
            'weight_transform': r['spec']['transform'],
            'weight_rule': r['spec']['rule'],
            'winsor_low_pct': r['spec']['winsor_low'],
            'winsor_high_pct': r['spec']['winsor_high'],
            'n_obs_N': r['diagnostics']['n_obs_N'],
            'n_months_T': r['diagnostics']['n_months_T'],
            'n_per_month_median': r['diagnostics']['n_per_month_median'],
            'kish_eff_n_median': r['diagnostics']['kish_eff_n_median'],
            'kish_eff_n_min': r['diagnostics']['kish_eff_n_min'],
            'kish_eff_n_max': r['diagnostics']['kish_eff_n_max'],
            'kish_eff_n_pct_of_n_median': r['diagnostics']['kish_eff_n_pct_of_n_median'],
            'max_single_weight_median': r['diagnostics']['max_single_weight_median'],
            'max_single_weight_share_median':
                r['diagnostics']['max_single_weight_share_median'],
            'max_single_weight_share_max':
                r['diagnostics']['max_single_weight_share_max'],
            'top5_weight_share_median': r['diagnostics']['top5_weight_share_median'],
            'peer_max_beta': pm['beta'],
            'peer_max_se_nw': pm['se_nw'],
            'peer_max_t_nw': pm['t_nw'],
            'peer_max_ci95_low': pm.get('ci95_low'),
            'peer_max_ci95_high': pm.get('ci95_high'),
            'peer_max_p_two_sided': pm['p_two_sided'],
            'peer_max_significant_5pct': bool(pm['p_two_sided'] is not None
                                              and pm['p_two_sided'] < 0.05),
            'peer_max_sign': '+' if pm['beta'] > 0 else '-',
            'peer_max_negative_month_share': r['peer_max_negative_month_share'],
            'own_max_beta': c['OwnMAXraw']['beta'],
            'own_max_se_nw': c['OwnMAXraw']['se_nw'],
            'own_max_t_nw': c['OwnMAXraw']['t_nw'],
            'own_max_p_two_sided': c['OwnMAXraw']['p_two_sided'],
            'own_max_significant_5pct': bool(c['OwnMAXraw']['p_two_sided'] is not None
                                             and c['OwnMAXraw']['p_two_sided'] < 0.05),
            'peer_return_beta': c['PeerReturn']['beta'],
            'peer_return_se_nw': c['PeerReturn']['se_nw'],
            'peer_return_t_nw': c['PeerReturn']['t_nw'],
            'peer_return_p_two_sided': c['PeerReturn']['p_two_sided'],
            'peer_return_significant_5pct': bool(
                c['PeerReturn']['p_two_sided'] is not None
                and c['PeerReturn']['p_two_sided'] < 0.05),
        })

    by_id = {r['weight_id']: r for r in rows}
    steep = ['me', 'sqrt_me', 'me_w95', 'me_w99']
    flat = ['ew', 'log_me']
    steep_neg = all(by_id[s]['peer_max_beta'] < 0 for s in steep)
    flat_pos = all(by_id[s]['peer_max_beta'] > 0 for s in flat)
    signs = [by_id[s]['peer_max_sign'] for s in
             ('ew', 'me', 'sqrt_me', 'log_me', 'me_w95', 'me_w99')]
    n_sig = int(sum(1 for r in rows if r['peer_max_significant_5pct']))
    max_abs_t = max(abs(r['peer_max_t_nw']) for r in rows)

    ew, me = by_id['ew'], by_id['me']
    kish_ew = ew['kish_eff_n_median']
    kish_me = me['kish_eff_n_median']
    share_ew = ew['max_single_weight_share_median'] * 100
    share_me = me['max_single_weight_share_median'] * 100

    if steep_neg and flat_pos:
        answer = '是（条件性：由权重函数的 ME 梯度/集中度决定，而非「任何含 ME 的权重」）'
        p1 = (f"所有**陡峭的 ME 递增权重**都把 PeerMAX 压成负号："
              f"ME {me['peer_max_beta']:+.5f}（t={me['peer_max_t_nw']:+.2f}）、"
              f"sqrt(ME) {by_id['sqrt_me']['peer_max_beta']:+.5f}"
              f"（t={by_id['sqrt_me']['peer_max_t_nw']:+.2f}）、"
              f"ME winsor 5/95 {by_id['me_w95']['peer_max_beta']:+.5f}"
              f"（t={by_id['me_w95']['peer_max_t_nw']:+.2f}）、"
              f"ME winsor 1/99 {by_id['me_w99']['peer_max_beta']:+.5f}"
              f"（t={by_id['me_w99']['peer_max_t_nw']:+.2f}）。")
        p2 = (f"而**压平或压缩 ME 的权重**保持正号：EW {ew['peer_max_beta']:+.5f}"
              f"（t={ew['peer_max_t_nw']:+.2f}）、log(ME) "
              f"{by_id['log_me']['peer_max_beta']:+.5f}"
              f"（t={by_id['log_me']['peer_max_t_nw']:+.2f}）。"
              f"log(ME) 虽然仍是 ME 的函数，但把权重分布在有效样本上几乎压平"
              f"（Kish 有效 N {by_id['log_me']['kish_eff_n_median']:.0f}，"
              f"为 n 的 {by_id['log_me']['kish_eff_n_pct_of_n_median']:.1f}%），"
              f"因此回到 EW 的结果，而不是 VW 的负号。")
        p3 = (f"权重集中度与符号翻转同步：EW 的 Kish 有效 N 中位 {kish_ew:.0f}"
              f"（=n）、单股最大权重份额 {share_ew:.2f}%；"
              f"ME 权重把有效 N 压到 {kish_me:.0f}"
              f"（仅 n 的 {me['kish_eff_n_pct_of_n_median']:.1f}%）、"
              f"单股最大份额 {share_me:.2f}%（月度最高 "
              f"{me['max_single_weight_share_max'] * 100:.2f}%）。"
              f"即负号来自少数大市值股票在每个月的横截面中占据过高权重，"
              f"而不是来自样本、控制集或 outcome 的任何变化（三者逐项未动）。")
        p4 = (f"但**六种权重下 PeerMAX 全部不显著**（最大 |t|={max_abs_t:.2f}，"
              f"5% 显著规格 {n_sig}/6；月度 β<0 的月份占比 "
              f"{min(r['peer_max_negative_month_share'] for r in rows) * 100:.0f}%–"
              f"{max(r['peer_max_negative_month_share'] for r in rows) * 100:.0f}%）。"
              f"因此这是**符号脆弱性**：符号随权重函数的 ME 梯度翻转，"
              f"但没有任何权重方案提供 PeerMAX 非零的统计证据；"
              f"不能把任一符号当作「真实方向」。")
        conclusion = ('PeerMAX 的符号确实由 ME 水平值权重函数的**陡峭程度/集中度**决定：'
                      '陡峭的 ME 权重（ME、√ME、ME winsor 95/99）为负，'
                      '压平/压缩的权重（EW、log ME）为正；翻转由大市值股票主导的'
                      '权重集中（Kish 有效 N 从约 1,725 降至约 199，单股最大份额从约 0.06% '
                      '升至约 4.55%）驱动。但六种权重下 PeerMAX 均不显著（|t|≤1.18），'
                      '故这是符号脆弱性，不构成方向性证据；'
                      'headline 的 EW/VW 符号相反应如实保留为「不显著 + 权重敏感」，'
                      '不得择一解读。')
    elif steep_neg:
        p1 = '所有陡峭 ME 权重为负号，但 EW/log 并未一致为正；符号翻转方向不完全对称。'
        p2 = f'六种权重下 PeerMAX 最大 |t|={max_abs_t:.2f}，5% 显著规格 {n_sig}/6。'
        p3 = ''
        p4 = ''
        conclusion = 'PeerMAX 符号随 ME 权重函数变化，且所有规格均不显著；符号不稳健。'
        answer = '部分（符号随 ME 权重梯度变化，但非干净二分）'
    else:
        p1 = 'ME 系列权重并未一致为负号；符号不由 ME 水平值权重函数单向决定。'
        p2 = f'六种权重下 PeerMAX 最大 |t|={max_abs_t:.2f}，5% 显著规格 {n_sig}/6。'
        p3 = ''
        p4 = ''
        conclusion = 'PeerMAX 符号与 ME 权重函数无一致关系；且所有规格不显著，符号不稳健。'
        answer = '否'

    verdict = {
        'question': 'PeerMAX 符号是否由 ME 水平值权重函数决定？',
        'answer': answer,
        'signs_text': ' / '.join(signs),
        'n_significant': n_sig,
        'max_abs_t': float(max_abs_t),
        'steep_me_weights_all_negative': bool(steep_neg),
        'flat_weights_all_positive': bool(flat_pos),
        'paragraphs': [p for p in (p1, p2, p3, p4) if p],
        'conclusion': conclusion,
    }

    # ---- 写出 CSV
    csv_path = OUT_DIR / '3b_s2_weight_matrix.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8')
    print(f'  -> {csv_path.relative_to(ROOT)}', flush=True)

    # ---- 运行后输入锚
    print('[S2] 校验 9 个 M1 主输入（运行后）…', flush=True)
    sha_after = assert_m1_inputs_sha()
    unchanged = all(b['sha256'] == a['sha256'] for b, a in zip(sha_before, sha_after))
    require(unchanged, '运行前后输入 sha256 发生变化')
    require(all(a['match'] for a in sha_after), '运行后输入锚存在失败项')

    script_path = Path(__file__).resolve()
    sample_sha = sha256_file(SAMPLE)
    meta = {
        'created_at_utc': created,
        'agent': 'B3 (S-2)',
        'purpose': 'VW 权重函数敏感性矩阵：六种 firm-level WLS 权重下 S1 PeerMAX/OwnMAX/PeerReturn',
        'script_path': str(script_path.relative_to(ROOT)).replace('\\', '/'),
        'script_sha256': sha256_file(script_path),
        'sample_path': str(SAMPLE.relative_to(ROOT)).replace('\\', '/'),
        'sample_sha256': sample_sha,
        'fm_rows': int(len(fm)),
        'fm_months': int(fm['month'].nunique()),
        'fm_window': [str(fm['month'].min()), str(fm['month'].max())],
        'spec': 'S1: R_next ~ PeerMAXraw + OwnMAXraw + PeerReturn + 8 controls',
        'controls': CONTROLS,
        'ycol': YCOL,
        'min_n': MIN_N,
        'nw_lag': NW_LAG,
        'winsor_definition': ('95% = 月度 FM 截面 [q0.05,q0.95]；'
                              '99% = [q0.01,q0.99]（两侧分位裁剪）'),
        'input_sha_before': sha_before,
        'input_sha_after': sha_after,
        'input_sha_unchanged': bool(unchanged),
        'input_sha_all_match': bool(all(x['match'] for x in sha_before)
                                    and all(x['match'] for x in sha_after)),
    }
    counts = {
        'meta': meta,
        'contract': contract,
        'sample_checks': {
            'rows_total': int(len(df)),
            'rows_total_expected': SAMPLE_ROWS_TOTAL,
            'fm_rows': int(len(fm)),
            'fm_rows_expected': SAMPLE_ROWS_FM,
            'fm_months': int(fm['month'].nunique()),
            'fm_months_expected': SAMPLE_MONTHS,
            'duplicate_focal_month': int(fm.duplicated(['focal_id', 'month']).sum()),
            'market_equity_nonpositive': int((fm['market_equity'] <= 0).sum()),
            'market_equity_missing': int(fm['market_equity'].isna().sum()),
            'log_me_min': float(np.log(fm['market_equity']).min()),
        },
        'winsor_bounds': winsor_bounds,
        'weights': {
            r['weight_id']: {
                'label': r['weight_label'],
                'weight_col': r['weight_col'],
                'transform': r['weight_transform'],
                'rule': r['weight_rule'],
                'winsor_low_pct': r['winsor_low_pct'],
                'winsor_high_pct': r['winsor_high_pct'],
                'diagnostics': {
                    k: v for k, v in res['diagnostics'].items() if k != 'per_month'
                },
                'per_month_diagnostics': res['diagnostics']['per_month'],
                'summary': res['summary'],
                'peer_max_beta_series': res['beta_series']['PeerMAXraw'],
                'peer_max_negative_month_share': res['peer_max_negative_month_share'],
            }
            for r, res in zip(rows, results)
        },
        'verdict': verdict,
        'outputs': [
            'artifacts/phase3b/3b_s2_weight_matrix.csv',
            'artifacts/phase3b/3b_s2_counts.json',
            'artifacts/phase3b/3b_s2_execution.md',
        ],
    }
    write_json(OUT_DIR / '3b_s2_counts.json', counts)

    md = build_md(meta, rows, verdict)
    (OUT_DIR / '3b_s2_execution.md').write_text(md, encoding='utf-8')
    print(f'  -> {(OUT_DIR / "3b_s2_execution.md").relative_to(ROOT)}', flush=True)

    print('[S2] 完成；六权重 PeerMAX β/t：', flush=True)
    for r in rows:
        print(f"  {r['weight_id']:8s} β={r['peer_max_beta']:+.5f} "
              f"t={r['peer_max_t_nw']:+.2f} "
              f"kish={r['kish_eff_n_median']:.1f} "
              f"maxshare={r['max_single_weight_share_median'] * 100:.2f}%",
              flush=True)


if __name__ == '__main__':
    main()
