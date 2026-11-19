#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 第一梯队 ③：同密度零模型（`s3_network_density_null.json`）。

**只读复算**：不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；只新建
`reports/data_gates/G0/s3_network_density_null.json`。

口径真源：
  - `reports/audits/s3_remaining_workplan_20261011.md` 第一梯队 3
  - 主仓库 methodology_execution_plan §8 B″11-2
  - `reports/audits/s3_network_open_items_plan_20261006.md` §3-②(a)（判据三件套 (a)）
  - registry：`rewiring_reps`（R = 1000，借用）、`random_seed`（20260727）

零模型：R = `rewiring_reps` 个「同 N 同 E 随机简单图」（seed = `random_seed`）——在当年网络节点上
均匀无放回抽边（无自环），报 realized lift 的经验分位与 `p = (1 + #{null_lift ≥ realized})/(1 + R)`。

**实现说明（统计量与图抽样等价）**：realized 统计量 `lift = same_sw2_share / base` 只通过
「随机图中同二级 pair 数」依赖随机图——在 U = C(N,2) 个无序对中均匀无放回抽 m 条时，同二级 pair 数
**精确服从 Hypergeometric(U, K_same, m)**（K_same = Σ_k n_k(n_k−1)/2）。故用 `rng.hypergeometric`
抽样 = 对「同 N 同 E 随机简单图」零模型统计量的**精确**同分布抽样（非近似、非放宽）。脚本另做一次
**逐 replicate 真实抽样**的抽查（--graph-check-reps）以验证该等价性。

**口径**：U / K_same / m 与 `network_pair_density.py` 的 same_sw2_share / pair_density **同一口径**
（只计有申万映射的节点；m = 可映射边数/2 = 可映射无序对数）。另报「全节点节点集」的 spec 字面变体作
诊断（其 null 均值不等于 1，因为 realized share 的样本池是可映射边）。

**证据力与参数借用声明（必须随产物）**：本检验只排除「同密度随机图」，**证据力弱**；R 借用
`rewiring_reps`，其 scope 原属 degree/industry-preserving 重连——如项目要求严格须**另登记独立 rep 数**。
本零模型与 proposal p7/p12 的 degree- and industry-preserving rewiring（M1 安慰剂）是**两个不同对象**。

验收：realized lift > null p95；null 均值 ≈ 1（内部一致性）。
停机：null 均值偏离 1 超 5% → 先修零模型再报（不得改判据）。

用法：python scripts/network_density_null.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import network_business_chapter_cross_audit as CA  # noqa: E402
import network_pair_density as PD  # noqa: E402  （同一口径：pair_density / same_sw2_share / lift）

REPD = ROOT / 'reports/data_gates/G0'
TEACH_DIR = ROOT / 'data/processed/network_ma_teacher'
REGISTRY = ROOT / 'configs/parameter_registry.yaml'
YEARS = list(range(2014, 2023))              # 文本年 → 网络年 2015–2023（主口径 9 年）
NULL_MEAN_TOL = 0.05                         # null 均值偏离 1 的上限（5%）

PARAM_BORROW_NOTE = (
    '**证据力与参数借用声明**：本检验只排除「同密度随机图」，**证据力弱**（不保 degree、不保行业、'
    '不保团结构）；R 借用 registry `rewiring_reps`（= 1000），其 scope 原属 degree- and '
    'industry-preserving 重连——**如项目要求严格须另登记独立 rep 数**。本零模型与 proposal p7/p12 的 '
    'degree- and industry-preserving rewiring（M1 安慰剂）是**两个不同对象**，不得混写。'
)
# 解析等价实现说明（含复核建议的两条注记，2026-10-11）：Hypergeom 精确分布 + rng.hypergeometric 抽样
# + 200-replicate 真实图抽查三件套；**复制级数值不可与图抽样实现逐值对齐（RNG 流不同）**；
# **null 均值 ≈1 是代数恒等、不作为一致性证据**。
ANALYTIC_EQUIVALENCE_NOTE = (
    '**解析等价实现说明**：本零模型用三件套实现——① Hypergeom(U, K_same, m) **精确分布**（解析）；'
    '② R = {reps} 的 `rng.hypergeometric` 抽样（seed = {seed}）；③ `graph_sampling_check` 的 '
    '200-replicate **真实图抽样**抽查（逐 replicate 均匀无放回抽 m 条无序对）用于核对等价性。'
    '**复核注记（2026-10-11）**：（a）**复制级（replicate-level）数值不可与图抽样实现逐值对齐'
    '（RNG 流不同）**——两者只保证同分布，逐 replicate 的 draw 不一一对应（均值/标准差/分位在 '
    'Monte Carlo 误差内一致即为通过）；（b）**null 均值 ≈ 1 是代数恒等**（E[share_null] = K_same/U '
    '= pair_density ⟹ E[null_lift] = 1），**不作为一致性证据**——它只校验实现未违反该恒等，不具备'
    '拒绝零模型的证据力。'
)
# 年份轴消歧（L-3；全部逐键一律网络年；graph_sampling_check 的抽样年并报文本年）
YEAR_AXIS_NOTE = (
    '**年份轴（L-3 消歧）**：`per_year[].network_year` = **网络年（文本年 y−1）**、'
    '`per_year[].text_year` = **文本年**；`pooled.years` 与全部逐年/池化读数的年份一律为**网络年**；'
    '`graph_sampling_check` 的抽样年 = 文本年 2014 = 网络年 2015（两字段均已并报）。'
    '本文件不出现无轴名的裸年份。'
)


def read_registry(value_name: str, key: str = 'headline_value'):
    """从 registry 文本中读参数（避免 yaml 依赖）：返回 (值, 上下文行)。"""
    txt = REGISTRY.read_text(encoding='utf-8')
    pat = re.compile(r'^\s*- name: ' + re.escape(value_name) + r'\s*$', re.M)
    m = pat.search(txt)
    if not m:
        raise KeyError(f'registry 未找到参数 {value_name}')
    tail = txt[m.end():]
    nxt = re.search(r'^\s*- name: ', tail, re.M)
    block = tail[:nxt.start()] if nxt else tail
    km = re.search(r'^\s*' + re.escape(key) + r':\s*(\S+)', block, re.M)
    if not km:
        raise KeyError(f'{value_name} 缺 {key}')
    raw = km.group(1).strip().strip('"')
    try:
        val = int(raw)
    except ValueError:
        val = raw
    return val, km.group(0).strip()


def null_block(u_ngood: int, u_nbad: int, m: int, rng: np.random.Generator, reps: int) -> dict:
    """精确同分布抽样：同二级 pair 数 ~ Hypergeometric(U, K_same, m)。"""
    draws = rng.hypergeometric(u_ngood, u_nbad, m, size=reps)
    return {'draws': draws, 'share': draws / m, 'n_same_pairs': draws}


def stats(x: np.ndarray) -> dict:
    return {'mean': float(x.mean()), 'std': float(x.std(ddof=1)) if len(x) > 1 else None,
            'p05': float(np.quantile(x, 0.05)), 'p50': float(np.quantile(x, 0.5)),
            'p95': float(np.quantile(x, 0.95)), 'p99': float(np.quantile(x, 0.99)),
            'max': float(x.max()), 'min': float(x.min())}


def graph_sampling_check(nodes_codes: list, N: int, m: int, reps: int, seed: int) -> dict:
    """逐 replicate **真实**抽样（均匀无放回抽无序对）→ 与 Hypergeometric 等价性核对。"""
    codes = np.asarray(nodes_codes)
    rng = np.random.default_rng(seed + 991)
    U = N * (N - 1) // 2
    # 无序对索引 t → (i,j) 解码：offset[i] = i*N − i(i+1)/2
    off_i = np.arange(N, dtype=np.int64)
    off = off_i * N - off_i * (off_i + 1) // 2
    shares = np.empty(reps)
    for r in range(reps):
        t = rng.choice(U, size=m, replace=False)
        i = np.searchsorted(off, t, side='right') - 1
        j = i + 1 + (t - off[i])
        shares[r] = float((codes[i] == codes[j]).mean())
    return {'reps': reps, 'share_mean': float(shares.mean()), 'share_std': float(shares.std(ddof=1)),
            'share_p95': float(np.quantile(shares, 0.95)), 'share_min': float(shares.min()),
            'share_max': float(shares.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--net-dir', default=str(TEACH_DIR))
    ap.add_argument('--years', type=int, nargs='*', default=YEARS)
    ap.add_argument('--reps', type=int, default=None, help='缺省 = registry rewiring_reps')
    ap.add_argument('--seed', type=int, default=None, help='缺省 = registry random_seed')
    ap.add_argument('--graph-check-reps', type=int, default=200,
                    help='逐 replicate 真实抽样的抽查次数（0 = 跳过）')
    ap.add_argument('--graph-check-year', type=int, default=None, help='抽查年份（缺省 = 首个有数据年）')
    ap.add_argument('--out-json', default=str(REPD / 's3_network_density_null.json'))
    args = ap.parse_args()

    reps = args.reps
    seed = args.seed
    reg_reps, reg_reps_line = read_registry('rewiring_reps')
    reg_seed, reg_seed_line = read_registry('random_seed')
    reps = reps if reps is not None else reg_reps
    seed = seed if seed is not None else reg_seed
    print(f'[T1-③] R = {reps}（registry rewiring_reps = {reg_reps}，借用）; seed = {seed} '
          f'（registry random_seed = {reg_seed}）', flush=True)

    net_dir = Path(args.net_dir)
    sw2 = CA.load_sw2()
    rng = np.random.default_rng(seed)
    per_year, pooled_draws, pooled_m = [], [], []
    first_year_data = None
    for y in args.years:
        if not (net_dir / f'{y}_similarity.csv').exists():
            per_year.append({'text_year': y, 'status': 'missing'})
            continue
        sup = CA.read_support(net_dir, y)
        edges = CA.read_edges(net_dir, y)
        blk = PD.pair_density_block(sup, edges, sw2)
        base, share = blk['pair_density'], blk['same_sw2_share']
        lift = blk['lift']
        N, K, U = blk['n_nodes_mapped'], blk['same_class_pairs_undirected'], blk['universe_pairs_undirected']
        Em = blk['n_edges_mapped']
        m = Em // 2
        Na, Ea = blk['n_nodes_total'], blk['n_edges']
        # 口径核对：边表按对称（双向）存 → E = 2 × 无向对数（无序对、无自环）
        fa_ = edges['focal_id'].map(lambda f: CA.l2_of(f, sw2)).to_numpy()
        fb_ = edges['source_id'].map(lambda f: CA.l2_of(f, sw2)).to_numpy()
        fids, sids = edges['focal_id'].to_numpy(), edges['source_id'].to_numpy()
        lo = np.where(fids < sids, fids, sids)
        hi = np.where(fids < sids, sids, fids)
        both = ~pd.isna(fa_) & ~pd.isna(fb_)
        uniq_all = int(pd.Series(list(zip(lo, hi))).nunique())
        uniq_mapped = int(pd.Series(list(zip(lo[both], hi[both]))).nunique())
        symmetry_ok = bool(Em == 2 * uniq_mapped and Ea == 2 * uniq_all)
        nb = null_block(int(K), int(U - K), int(m), rng, reps)
        null_lift = nb['share'] / base
        # spec 字面变体：全节点节点集（含缺映射公司）
        Ua, ma = Na * (Na - 1) // 2, Ea // 2
        nb_a = null_block(int(K), int(Ua - K), int(ma), np.random.default_rng(seed), reps)
        null_lift_a = nb_a['share'] / base
        p = (1 + int(np.sum(null_lift >= lift))) / (1 + reps)
        pct = 100.0 * (1 + int(np.sum(null_lift < lift))) / (1 + reps)
        st = stats(null_lift)
        st_a = stats(null_lift_a)
        row = {'text_year': y, 'network_year': y + 1, 'status': 'ok',
               'n_nodes_total': Na, 'n_nodes_mapped': N, 'n_edges': Ea, 'n_edges_mapped': Em,
               'n_unordered_pairs_all_edges': uniq_all,
               'n_unordered_pairs_mapped_edges': uniq_mapped,
               'symmetry_check_passed': symmetry_ok,
               'symmetry_check_rule': '对称边表：E = 2 × 无向对数（无序对、无自环）；'
                                      '可映射侧 Em = 2 × 可映射无向对数',
               'same_class_pairs_undirected': int(K), 'universe_pairs_undirected': U,
               'null_draws_m': int(m), 'pair_density_base': base, 'same_sw2_share': share,
               'realized_lift': lift,
               'null_lift': {**st, 'n_greater_or_equal': int(np.sum(null_lift >= lift))},
               'null_share': stats(nb['share']),
               'p_value': p, 'empirical_percentile_of_realized': pct,
               'realized_gt_null_p95': bool(lift > st['p95']),
               'null_mean_within_5pct_of_1': bool(abs(st['mean'] - 1.0) <= NULL_MEAN_TOL),
               'spec_literal_variant_all_nodes': {
                   'note': 'spec 字面（节点集 = 全部入网公司含缺映射者；m = 全部有向边/2）→ null 均值 '
                           '≠ 1（因 realized share 的样本池为可映射边）；仅作诊断，不参与验收',
                   'universe_pairs_undirected': Ua, 'null_draws_m': int(ma),
                   'null_lift_mean': st_a['mean'], 'null_lift_p95': st_a['p95'],
                   'p_value': (1 + int(np.sum(null_lift_a >= lift))) / (1 + reps),
                   'realized_gt_null_p95': bool(lift > st_a['p95'])},
               }
        per_year.append(row)
        pooled_draws.append(nb['draws'])
        pooled_m.append(m)
        if first_year_data is None:
            codes = [c for c in (CA.l2_of(f, sw2) for f in sup['focal_id']) if c is not None]
            first_year_data = (y, codes, N, m)
        print(f'[T1-③] 网络年 {y+1}: lift={lift:.3f} null_mean={st["mean"]:.4f} '
              f'null_p95={st["p95"]:.4f} p={p:.5f} 分位={pct:.1f}% 成立={row["realized_gt_null_p95"]}',
              flush=True)

    ok = [r for r in per_year if r.get('status') == 'ok']
    # 池化：各年独立 replicate 求和（同一 seed 流）→ 池化 null share
    pool = {'years': [r['network_year'] for r in ok]}
    if ok:
        D = np.vstack(pooled_draws)                        # (Y, R)
        pooled_share_null = D.sum(axis=0) / float(sum(pooled_m))
        pooled_base = sum(r['same_class_pairs_undirected'] for r in ok) / \
            sum(r['universe_pairs_undirected'] for r in ok)
        pooled_share_real = sum(r['same_sw2_share'] * r['n_edges_mapped'] for r in ok) / \
            sum(r['n_edges_mapped'] for r in ok)
        pooled_lift = pooled_share_real / pooled_base
        pl = pooled_share_null / pooled_base
        pool.update({'pooled_pair_density_base': pooled_base, 'pooled_same_sw2_share': pooled_share_real,
                     'pooled_realized_lift': pooled_lift, 'null_lift': stats(pl),
                     'p_value': (1 + int(np.sum(pl >= pooled_lift))) / (1 + reps),
                     'empirical_percentile_of_realized': 100.0 * (1 + int(np.sum(pl < pooled_lift))) / (1 + reps),
                     'realized_gt_null_p95': bool(pooled_lift > stats(pl)['p95']),
                     'null_mean_within_5pct_of_1': bool(abs(stats(pl)['mean'] - 1.0) <= NULL_MEAN_TOL),
                     'note': ('池化 null = 各**网络年（文本年 y−1）**独立 replicate 之和 / Σm'
                              '（同样本边际基线池化）')})

    # 真实抽样抽查（与 Hypergeometric 等价性核对）
    gcheck = {}
    if args.graph_check_reps and first_year_data is not None:
        y, codes, N, m = first_year_data
        gc = graph_sampling_check(codes, N, m, args.graph_check_reps, seed)
        ref = next(r for r in ok if r['text_year'] == y)
        gcheck = {'text_year': y, 'network_year': y + 1, **gc,
                  'hypergeometric_reference': {'mean': ref['null_share']['mean'],
                                               'std': ref['null_share']['std'],
                                               'p95': ref['null_share']['p95'], 'reps': reps},
                  'equivalence_note': ('逐 replicate 真实抽样（均匀无放回抽 m 条无序对）与 '
                                       'Hypergeometric 抽样同分布；两者均值/标准差应一致（差异仅为 '
                                       'Monte Carlo 误差），此处用于验证解析抽样的正确性。'
                                       '**复核注记（2026-10-11）**：**复制级数值不可与图抽样实现逐值'
                                       '对齐（RNG 流不同）**——本抽查与 R=1000 解析抽样各自独立的 RNG 流，'
                                       '只保证同分布、不保证逐 replicate 相等；'
                                       '**null 均值 ≈1 是代数恒等、不作为一致性证据**。'),
                  'consistent': bool(abs(gc['share_mean'] - ref['null_share']['mean'])
                                     <= 4 * gc['share_std'] / max(1.0, np.sqrt(args.graph_check_reps)))}

    checks = {'all_years_realized_gt_null_p95': all(r['realized_gt_null_p95'] for r in ok),
              'all_years_null_mean_within_5pct': all(r['null_mean_within_5pct_of_1'] for r in ok),
              'pooled_realized_gt_null_p95': bool(pool.get('realized_gt_null_p95')),
              'pooled_null_mean_within_5pct': bool(pool.get('null_mean_within_5pct_of_1')),
              'symmetry_check_all': all(r['symmetry_check_passed'] for r in ok)}
    stops = []
    if not checks['all_years_null_mean_within_5pct'] or not checks['pooled_null_mean_within_5pct']:
        stops.append('null 均值偏离 1 超 5% → 先修零模型再报（不得改判据）')
    if not checks['all_years_realized_gt_null_p95']:
        stops.append('存在年份 realized lift ≤ null p95')

    rep = {
        'step': 'S3 第一梯队 ③ 同密度零模型（只排除「同密度随机图」）',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'net_dir': str(net_dir.relative_to(ROOT)).replace('\\', '/'),
        'nature': '只读复算：不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；产物为新建文件',
        'null_model_definition': (
            '同 N 同 E 随机简单图：在当年网络节点集上均匀无放回抽 m 条无序对（无自环）得到随机图；'
            'N = 可映射节点数、E = 可映射有向边数 = 2m（与 pair_density / same_sw2_share 同口径）；'
            '统计量 null_lift = share_null / pair_density。'),
        'sampler_note': ('同二级 pair 数在「均匀无放回抽 m 条无序对」下**精确服从** Hypergeometric'
                         '(U, K_same, m)（U = C(N,2)、K_same = Σ_k n_k(n_k−1)/2）→ 用 rng.hypergeometric '
                         '实现 = 对「同密度随机图」零模型统计量的精确同分布抽样；已另做逐 replicate 真实'
                         '抽样抽查核对（graph_sampling_check）。'
                         + ANALYTIC_EQUIVALENCE_NOTE.format(reps=reps, seed=seed)),
        'p_value_definition': 'p = (1 + #{null_lift ≥ realized_lift}) / (1 + R)',
        'R': reps, 'R_source': f'registry `rewiring_reps`（借用；{reg_reps_line}）',
        'seed': seed, 'seed_source': f'registry `random_seed`（{reg_seed_line}）',
        'parameter_borrow_and_power_note': PARAM_BORROW_NOTE,
        'scope_note': ('本检验只给「网络结构信号非随机同密度图」的**弱**证据；degree/行业/团结构一律不保；'
                       '不得据此宣称外部验证或机制证据'),
        'per_year': per_year, 'pooled': pool, 'graph_sampling_check': gcheck,
        'acceptance': {'realized_gt_null_p95': 'required', 'null_mean_approx_1_tol': NULL_MEAN_TOL,
                       'checks': checks, 'stop_conditions_triggered': stops,
                       'year_axis': 'network_year（网络年；= 文本年 + 1）'},
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'year_axis_note': YEAR_AXIS_NOTE,
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[T1-③] status={rep["status"]} checks={checks}', flush=True)
    print(f'[T1-③] p 值 = ' + ', '.join(f'{r["network_year"]}:{r["p_value"]:.5f}' for r in ok)
          + f'；池化 p = {pool.get("p_value"):.5f}', flush=True)
    print(f'[T1-③] null 均值 = ' + ', '.join(f'{r["network_year"]}:{r["null_lift"]["mean"]:.4f}' for r in ok), flush=True)
    print(f'[T1-③] realized lift = ' + ', '.join(f'{r["network_year"]}:{r["realized_lift"]:.3f}' for r in ok), flush=True)
    print(f'[T1-③] 真实抽样抽查：{gcheck.get("consistent")} -> {outp}', flush=True)


if __name__ == '__main__':
    main()
