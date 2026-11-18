#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 第一梯队 ⑥ / B″11-7 锚 2：分析师共同覆盖重叠度（`s3_network_analyst_overlap.json`）。

**只读复算**：不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；只新建
`reports/data_gates/G0/s3_network_analyst_overlap.json`。

口径真源：
  - `reports/audits/s3_remaining_workplan_20261011.md` 第一梯队 6
  - `docs/methodology/methodology_execution_plan.md` §8 B″11-7（**二修：v3**）
  - `reports/audits/s3_network_open_items_plan_20261006.md` §3-②(c)
  - S3A 9H：`reports/data_gates/G0/s3a_9h_absorb.json`（CF 方向参照）

输入：
  - 主口径产品网络：`data/processed/network_ma_teacher/{text_year}_{similarity,support}.csv`
  - 分析师月度边表：`data/processed/analyst/edges/edges_coverage_win12_fy12_**v3**_{month}.csv`
    （**必须 v3**：v2 仅 97 个月末（2017-12 起）= 67% 覆盖 < 80% 停机线，开工即停）
  - 覆盖集（权威）：`data/processed/analyst/firm_analyst_coverage.csv`（按 `network_cutoff` 取月）

月份映射：网络年 y 的产品网络 `available_at = y-04-30`（构网脚本口径）→ 取分析师 **y-04-30** 月末件。

读数（两侧同样本边际基线 lift，逐年）：
  - `analyst_link_share` = 产品网络边中两端**同月末共享 ≥1 分析师**的比例；
  - 反向 `product_link_share` = 分析师边中两端在产品网络内有边的比例；
  - 样本 = 产品网络节点 ∩ 该月末分析师覆盖公司（**两侧同一 N**，U = N(N−1)/2）；
  - 基线 = 对方网络在同样本内的无序对密度（`analyst_pair_density` / `product_pair_density`）；
  - `lift = share / 对方网络密度`（>1 = 重叠高于同样本随机配对）。

验收：两向 lift >1 且与 S3A 9H 的 CF 方向一致。停机：月份映射无法覆盖 ≥80% 的产品网络年 → 停。

**角色定位（必须随产物，二修口径）**：本步（分析师共同覆盖重叠）**只作替代网络一致性**
（proposal §6.3 替代网络），**不等价于 HP 式自报锚**——首选自报锚为招股书「同行业可比公司」
（需新数据 CNINFO 招股说明书 PDF；见第二梯队 #9 / B″11-7 锚 1，**未做**）。
**自报锚未做之前「外部锚」条件未满足**（报告 §5-3）。

用法：
  python scripts/network_analyst_overlap.py
  python scripts/network_analyst_overlap.py --years 2014 2015
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import network_business_chapter_cross_audit as CA  # noqa: E402  （read_support / read_edges 统一口径）

REPD = ROOT / 'reports/data_gates/G0'
TEACH_DIR = ROOT / 'data/processed/network_ma_teacher'
ANALYST_EDGES = ROOT / 'data/processed/analyst/edges'
ANALYST_COVERAGE = ROOT / 'data/processed/analyst/firm_analyst_coverage.csv'
S3A_9H = REPD / 's3a_9h_absorb.json'
TEXT_YEARS = list(range(2014, 2023))          # 文本年 → 网络年 2015–2023（主口径 9 年）
ANALYST_VERSION = 'v3'
COVERAGE_MIN = 0.80                           # 月份映射覆盖率停机线（≥80%）

ROLE_NOTE = (
    '**角色定位（二修口径，必须随产物）**：本步（分析师共同覆盖重叠）**只作替代网络一致性**'
    '（proposal §6.3 替代网络），**不等价于 HP 式自报锚**——首选自报锚为招股书「同行业可比公司」'
    '（需新数据 CNINFO 招股说明书 PDF；见 workplan 第二梯队 #9 / plan §8 B″11-7 锚 1，**本轮未做**）。'
    '**自报锚未做之前「外部锚」条件未满足**（报告 §5-3）；本文件不得被引用为「外部锚已满足」。'
)


def code6_of(firm: str) -> str | None:
    """产品网络 firm_id（如 `sz000020`）→ 6 位代码（与分析师表 `firm_id` int 同口径）。"""
    if not isinstance(firm, str) or len(firm) < 7:
        return None
    return firm[2:].zfill(6)


def unordered(a: str, b: str) -> tuple:
    return (a, b) if a <= b else (b, a)


def load_covered_months(months: list[str]) -> dict:
    """从覆盖表一次性取指定月末的覆盖公司集（权威口径；缺月记 None）。"""
    want = set(months)
    out: dict[str, set] = {m: set() for m in months}
    seen = set()
    for ch in pd.read_csv(ANALYST_COVERAGE, usecols=['network_cutoff', 'firm_id', 'coverage_flag'],
                          dtype={'network_cutoff': str, 'firm_id': str}, chunksize=1_000_000):
        ch = ch[ch['network_cutoff'].isin(want) & (ch['coverage_flag'] == 'covered')]
        for m, g in ch.groupby('network_cutoff'):
            out[m].update(g['firm_id'].str.zfill(6))
            seen.add(m)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--net-dir', default=str(TEACH_DIR))
    ap.add_argument('--text-years', type=int, nargs='*', default=TEXT_YEARS)
    ap.add_argument('--out-json', default=str(REPD / 's3_network_analyst_overlap.json'))
    args = ap.parse_args()

    net_dir = Path(args.net_dir)
    months = [f'{ty + 1}-04-30' for ty in args.text_years]      # 网络年 y ↔ y-04-30
    missing_files = [m for m in months if not (ANALYST_EDGES / f'edges_coverage_win12_fy12_{ANALYST_VERSION}_{m}.csv').exists()]
    covered = load_covered_months(months) if not missing_files else {}
    covered_sets = covered

    per_year, rows = [], []
    for ty, m in zip(args.text_years, months):
        y = ty + 1
        fp_an = ANALYST_EDGES / f'edges_coverage_win12_fy12_{ANALYST_VERSION}_{m}.csv'
        if not (net_dir / f'{ty}_support.csv').exists() or not fp_an.exists():
            per_year.append({'text_year': ty, 'network_year': y, 'month_end': m,
                             'status': 'missing', 'analyst_file': fp_an.name,
                             'product_files_present': bool((net_dir / f'{ty}_support.csv').exists())})
            continue
        sup = CA.read_support(net_dir, ty)
        edg = CA.read_edges(net_dir, ty)
        prod_nodes = {c for c in (code6_of(f) for f in sup['focal_id']) if c is not None}
        prod_pairs_all = {unordered(code6_of(a), code6_of(b))
                          for a, b in zip(edg['focal_id'], edg['source_id'])}
        prod_pairs_all = {p for p in prod_pairs_all if p[0] and p[1] and p[0] != p[1]}
        an = pd.read_csv(fp_an, dtype={'focal_id': str, 'source_id': str},
                         usecols=['focal_id', 'source_id', 'common_analyst_count'])
        an['focal_id'] = an['focal_id'].str.zfill(6)
        an['source_id'] = an['source_id'].str.zfill(6)
        an = an[an['common_analyst_count'] >= 1]
        an_pairs_all = {unordered(a, b) for a, b in zip(an['focal_id'], an['source_id']) if a != b}
        an_nodes_file = set(an['focal_id']) | set(an['source_id'])
        cov = covered_sets.get(m, set())
        if not cov:                                   # 覆盖表缺月 → 回退边表节点集（并记降级）
            cov = an_nodes_file
        sample = prod_nodes & cov
        U = len(sample) * (len(sample) - 1) / 2
        prod_pairs = {p for p in prod_pairs_all if p[0] in sample and p[1] in sample}
        an_pairs = {p for p in an_pairs_all if p[0] in sample and p[1] in sample}
        inter = prod_pairs & an_pairs
        n_prod, n_an, n_inter = len(prod_pairs), len(an_pairs), len(inter)
        analyst_link_share = n_inter / n_prod if n_prod else None
        product_link_share = n_inter / n_an if n_an else None
        analyst_pair_density = n_an / U if U else None
        product_pair_density = n_prod / U if U else None
        row = {'text_year': ty, 'network_year': y, 'status': 'ok', 'month_end': m,
               'analyst_file': fp_an.name, 'analyst_version': ANALYST_VERSION,
               'n_nodes_product': len(prod_nodes), 'n_nodes_analyst_covered': len(cov),
               'n_nodes_analyst_in_edge_file': len(an_nodes_file),
               'analyst_covered_equals_edge_file_nodes': bool(cov == an_nodes_file),
               'n_nodes_sample': len(sample), 'universe_pairs_undirected': U,
               'n_product_pairs_same_sample': n_prod, 'n_analyst_pairs_same_sample': n_an,
               'n_overlap_pairs': n_inter,
               'analyst_link_share': analyst_link_share,
               'product_link_share': product_link_share,
               'analyst_pair_density': analyst_pair_density,
               'product_pair_density': product_pair_density,
               'lift_analyst_link': (analyst_link_share / analyst_pair_density
                                     if analyst_link_share and analyst_pair_density else None),
               'lift_product_link': (product_link_share / product_pair_density
                                     if product_link_share and product_pair_density else None),
               'product_nodes_in_coverage_share': round(len(prod_nodes & cov) / max(1, len(prod_nodes)), 4),
               'analyst_covered_in_product_share': round(len(cov & prod_nodes) / max(1, len(cov)), 4)}
        per_year.append(row)
        rows.append(row)
        print(f'[T1B-⑥] 网络年 {y}（月末 {m}）：N={len(sample)} 产品对={n_prod} 分析师对={n_an} '
              f'重叠={n_inter} analyst_link_share={analyst_link_share:.4f} '
              f'product_link_share={product_link_share:.4f} lift={row["lift_analyst_link"]:.3f}/'
              f'{row["lift_product_link"]:.3f}', flush=True)

    ok = [r for r in rows if r['status'] == 'ok']
    n_years_total = len(args.text_years)
    coverage_rate = (len(ok) / n_years_total) if n_years_total else 0.0
    # 池化（pair 层：Σ 交 / Σ 分母；密度层：Σ 对 / ΣU）
    s_inter = sum(r['n_overlap_pairs'] for r in ok)
    s_prod = sum(r['n_product_pairs_same_sample'] for r in ok)
    s_an = sum(r['n_analyst_pairs_same_sample'] for r in ok)
    s_U = sum(r['universe_pairs_undirected'] for r in ok)
    pooled = {'years': [r['network_year'] for r in ok],
              'n_overlap_pairs': s_inter, 'n_product_pairs_same_sample': s_prod,
              'n_analyst_pairs_same_sample': s_an, 'universe_pairs_undirected_sum': s_U,
              'analyst_link_share': s_inter / s_prod if s_prod else None,
              'product_link_share': s_inter / s_an if s_an else None,
              'analyst_pair_density': s_an / s_U if s_U else None,
              'product_pair_density': s_prod / s_U if s_U else None,
              'lift_analyst_link': (s_inter / s_prod) / (s_an / s_U) if s_prod and s_an and s_U else None,
              'lift_product_link': (s_inter / s_an) / (s_prod / s_U) if s_prod and s_an and s_U else None,
              'note': '池化 = pair 层 Σ 交 / Σ 分母（逐年样本并集口径不适用，故为各年计数求和）'}
    # S3A 9H CF 参照（方向一致性判据）
    ref = {}
    if S3A_9H.exists():
        d = json.loads(S3A_9H.read_text(encoding='utf-8'))
        cf = (d.get('col3_cf_ind') or d.get('col1_cf_only') or {}).get('cf', {})
        ref = {'source': str(S3A_9H.relative_to(ROOT)).replace('\\', '/'),
               'spec_id': d.get('spec_id'), 'network_version': d.get('network_version'),
               'cf_beta': cf.get('beta'), 'cf_t_1way': cf.get('t_iid'),
               'cf_t_mean_monthly_ref': cf.get('t_mean_monthly_ref'), 'cf_t_nw1': cf.get('t_nw1'),
               'cf_direction': ('positive' if (cf.get('beta') or 0) > 0 else
                                'negative' if cf.get('beta') is not None else None),
               'reading': ('9H：分析师共同覆盖网络度（CF）正向预测下月收益（β>0、t≈6-7）→ 分析师网络'
                           '承载收益相关信息；本步两向 lift >1 = 产品网络与该替代网络的共覆盖结构'
                           '**非随机同向**，与 CF 方向一致（只作一致性，不作机制/外部锚证据）')}
    checks = {
        'coverage_ge_80pct': bool(coverage_rate >= COVERAGE_MIN),
        'all_years_lift_analyst_link_gt_1': bool(ok) and all((r['lift_analyst_link'] or 0) > 1 for r in ok),
        'all_years_lift_product_link_gt_1': bool(ok) and all((r['lift_product_link'] or 0) > 1 for r in ok),
        'pooled_lift_analyst_link_gt_1': bool(pooled['lift_analyst_link'] and pooled['lift_analyst_link'] > 1),
        'pooled_lift_product_link_gt_1': bool(pooled['lift_product_link'] and pooled['lift_product_link'] > 1),
        's3a_9h_cf_direction_consistent': bool(ref.get('cf_direction') == 'positive'
                                              and pooled['lift_analyst_link'] and pooled['lift_analyst_link'] > 1),
    }
    stops = []
    if not checks['coverage_ge_80pct']:
        stops.append(f'月份映射覆盖率 {coverage_rate:.1%} < 80% → 停（先修月份映射/版本，不得放宽匹配）')
    if not (checks['all_years_lift_analyst_link_gt_1'] and checks['all_years_lift_product_link_gt_1']):
        stops.append('存在年份两向 lift ≤1 → 记入 observations（不得据此宣称一致性成立）')

    rep = {
        'step': 'S3 第一梯队 ⑥ / B″11-7 锚 2：分析师共同覆盖重叠度（替代网络一致性）',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'nature': ('只读复算：不重建网络、不碰面板/回归、不改任何 headline_value / 冻结参数；'
                   '产物为新建文件'),
        'inputs': {'product_network': str(net_dir.relative_to(ROOT)).replace('\\', '/'),
                   'analyst_edges_dir': str(ANALYST_EDGES.relative_to(ROOT)).replace('\\', '/'),
                   'analyst_edge_pattern': f'edges_coverage_win12_fy12_{ANALYST_VERSION}_{{month}}.csv',
                   'analyst_version_required': ANALYST_VERSION,
                   'analyst_version_reason': ('v2 仅 97 个月末（2017-12 起）= 产品网络年覆盖 6/9 = 67% '
                                              '< 80% 停机线 → 二修改 v3（241 个月末 2005-12–2025-12）'),
                   'coverage_table': str(ANALYST_COVERAGE.relative_to(ROOT)).replace('\\', '/'),
                   's3a_9h_reference': str(S3A_9H.relative_to(ROOT)).replace('\\', '/')},
        'definitions': {
            'month_mapping': '网络年 y 的产品网络 available_at = y-04-30（构网脚本口径）→ 取分析师 y-04-30 '
                             '月末件（241 个月末内必存在）',
            'analyst_link_share': '产品网络边（无序对、样本内）中两端同月末共享 ≥1 分析师'
                                  '（common_analyst_count ≥1）的比例',
            'product_link_share': '反向：分析师边（无序对、样本内）中两端在产品网络内也有边的比例',
            'same_sample': '样本 = 产品网络节点（support 表 focal）∩ 该月末分析师覆盖公司；两侧同一 N、'
                           'U = N(N−1)/2（无序对、无自环）',
            'baseline': '同样本边际基线 = 对方网络在样本内的无序对密度（analyst_pair_density / '
                        'product_pair_density）',
            'lift': 'lift = share / 对方网络密度（>1 = 重叠高于同样本随机配对）',
            'lift_identity': ('**两向 lift 代数上恒等**：lift_analyst_link = (inter/prod)/(an/U) = '
                              'inter·U/(prod·an) = (inter/an)/(prod/U) = lift_product_link（交叉积比）'
                              '→ 本文件两向 lift 逐值相同属**恒等式**，非重复计算；两向 **share** 不同'
                              '（analyst_link_share = inter/prod vs product_link_share = inter/an），'
                              '两者并报才有信息（同样的重叠量在两个分母下的含义不同）')},
        'role_positioning': ROLE_NOTE,
        'per_year': per_year,
        'pooled': pooled,
        's3a_9h_reference': ref,
        'coverage_check': {'rule': f'月份映射覆盖率 ≥ {COVERAGE_MIN:.0%}（否则停）',
                           'n_years_total': n_years_total, 'n_years_mapped': len(ok),
                           'coverage_rate': round(coverage_rate, 4),
                           'missing_analyst_files': missing_files,
                           'pass': bool(coverage_rate >= COVERAGE_MIN)},
        'acceptance': {'two_way_lift_gt_1': 'required',
                       'direction_consistent_with_s3a_9h_cf': 'required',
                       'checks': checks, 'stop_conditions_triggered': stops},
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'scope_note': ('本读数只给「产品网络与分析师共同覆盖网络在同样本下重叠高于随机」的一致性证据；'
                       '**不等于外部锚验证**、不构成机制证据；不得据此宣称 pre-validated。'),
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[T1B-⑥] status={rep["status"]} checks={checks} -> {outp}', flush=True)
    print(f'[T1B-⑥] 池化：analyst_link_share={pooled["analyst_link_share"]:.4f} '
          f'product_link_share={pooled["product_link_share"]:.4f} '
          f'lift={pooled["lift_analyst_link"]:.3f}/{pooled["lift_product_link"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
