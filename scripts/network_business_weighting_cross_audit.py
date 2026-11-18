#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 Phase B″10：**同语料权重对照审计（等权 0/1 ↔ IDF 加权）**，纯描述性、**非闸门**。

目的：把「同一份老师 MD&A、同一词表、同一筛选、同一聚类算法，只把词向量元素由等权 1.0 换成
smooth IDF」这件事量化为可登记的对照结论，并与主口径逐项并排：① 同申万二级边占比；
② lift（同样本边际基线）；③ peer 规模（中位，统一「全部公司含零对手」口径，目标 44）；
④ 下游方向（PeerMAXraw / PeerReturn 相关、五分位同组率、多空符号与 t 值）。

**定性**：本对照**不是闸门**，不设 PASS/FAIL，不改任何冻结 headline_value，**本次不升主口径**
（`ma_vector_weighting` 的 headline_value 仍为等权）。它回答"权重选择是否改变结构/下游结论方向"。

实现 = 薄 wrapper：直接复用 `network_business_chapter_cross_audit.py` 的纯函数（`read_edges/read_support/
load_sw2/peer_stats/sw2_lift/isolation/cluster_consistency/density_diagnostics/corpus_richness`），
只是把两侧标签改成「等权(EW) / IDF」并新增**强不变量核对**（两侧 `n_nodes` 与 `filter_stats`
逐年必须逐值相等——命中集合与样本筛选与权重无关）。被复用的换章节脚本本身一字未改。

输入：
  data/processed/network_ma_teacher/（主口径 EW，`pm_ma_ewcos_teacher_v1_{y}`，文本年 2014–2022）
  data/processed/network_ma_teacher_idf/（IDF 变体，`pm_ma_idf_teacher_v1_{y}`，同语料同参数）
  data/processed/sw2_industry_latest.csv（同申万二级基线）
  可选 --downstream-report reports/data_gates/G0/s3_ma_downstream_consistency_idf.json（下游方向块）

产物：--out-json / --out-md（一律显式给出；`reports/data_gates/G0/network_business_weighting_cross_audit.{json,md}`）
用法：
  python scripts/network_business_weighting_cross_audit.py \
      --ew-dir data/processed/network_ma_teacher --idf-dir data/processed/network_ma_teacher_idf \
      --years 2014 2015 2016 2017 2018 2019 2020 2021 2022 \
      --downstream-report reports/data_gates/G0/s3_ma_downstream_consistency_idf.json \
      --out-json reports/data_gates/G0/s3_ma_weighting_cross_audit.json \
      --out-md reports/data_gates/G0/s3_ma_weighting_cross_audit.md
（`--self-dir` / `--teacher-dir` 为同一参数的历史兼容别名，语义 = 等权侧 / IDF 侧。）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import network_business_chapter_cross_audit as CA  # noqa: E402  （薄 wrapper：复用其纯函数与口径常量）

REPD = ROOT / 'reports/data_gates/G0'
EW_DIR = ROOT / 'data/processed/network_ma_teacher'
IDF_DIR = ROOT / 'data/processed/network_ma_teacher_idf'
YEARS = list(range(2014, 2023))
NOT_A_GATE_NOTE = (
    '本对照**不是闸门**：无 PASS/FAIL 判定、不改任何冻结 headline_value、**本次不升主口径**'
    '（ma_vector_weighting 的 headline_value 仍为等权 0/1）。两侧网络除**向量元素权重**外参数完全相同'
    '——同语料（老师 MD&A 导出）、同词表（product_lexicon_v2_teacher）、同筛选（J/S + 上市不足 2 年 + '
    'min_product_terms=10）、同聚类（average + 逐年 K 校准到中位 peer=44）、同对手集合与行归一化权重。'
)

# ---- 2026-10-05 审查修订：结论口径、结构差异实况、已知限制（只加字段，不改任何数值）----
CONCLUSION_DESCRIPTIVE = (
    '换权重（等权 0/1 → IDF）后**行业集中的方向一致且更强，但结构层差异显著**：同申万二级边占比与同样本边际基线 '
    'lift 逐年更高，peer 规模（统一口径）落在同一水平（池化 44 vs 45）；但共同节点集上边 Jaccard 仅 0.358–0.584、'
    'K 系统性下降 17.4%–37.1%、peer 均值下降（2014 年 48.51 vs 65.90，中位数被「目标 44 校准」掩盖）→ '
    '**不得**表述为「结构基本不变」或「结构层同向」；**下游方向未反向（描述性）**。'
)
FORBIDDEN_READINGS = [
    '「结构基本不变」/「结构层同向」——与边 Jaccard 0.358–0.584、K 下降 17.4%–37.1%、peer 均值下降矛盾',
    '「IDF 的 K 命中精度低于等权」——实测两侧命中精度相当（IDF exact 4 / closest 5、Σ|peer中位−44| = 9、'
    '最大偏离 3；等权 exact 3 / closest 6、Σ|偏离| = 9、最大偏离 2）；IDF 与等权的实质差异是 **K 结构差异大**，'
    '不是命中精度',
]
KNOWN_LIMITATIONS = [
    {'id': 'idf_chain_float32_level_reproducibility',
     'severity': 'registered_non_blocking',
     'mechanism': '等权 0/1 向量行内元素全为 1（列序不影响求和）；IDF 向量非零元取值随词条变化，'
                  '「词条 → 列号」赋值顺序取决于 Python 字符串集合迭代顺序（PYTHONHASHSEED 随机化）→ '
                  'float32 稀疏矩阵乘法累加序不同 → similarity / weight 可在 float32 末位不同',
     'measured_20261005': {'text_year': 2017, 'hash_seeds': [1, 2],
                           'edge_set_identical': True, 'n_edges': 206164,
                           'similarity_max_abs_diff_seed1_vs_seed2': 1.192e-07,
                           'similarity_mean_abs_diff': 2.538e-09,
                           'max_abs_diff_vs_production_artifact': [1.192e-07, 1.490e-07],
                           'cluster_labels_sha256_equal_across_seeds_and_production': True,
                           'K': 198, 'peer_median': 44.0, 'K_mode': 'exact',
                           'filter_stats_equal_across_runs': True,
                           'ew_chain_byte_reproducible': True,
                           'float32_eps': 1.1920928955078125e-07,
                           'reviewer_measured_max_abs_diff_other_sample': 5.96e-07},
     'evidence': 'artifacts/repro/phase_bpp_snapshot/idf_seed_reproducibility_spotcheck_20261005.json',
     'scope_note': '仅文本年 2017 抽检双种子；其余 8 年未逐年全检；等权（主口径）链另有 self 2021 + teacher 2014 '
                   '全链逐字节回归 PASS',
     'impact': '不影响任何已登记数值的等级（四项对照与九项验收在两套运行下均通过）；但 IDF 链**不保证 sha256 级'
               '逐字节复现**，引用 IDF 产物做比对应用数值容差（≤1e-6）而非 sha256'},
    {'id': 'idf_only_no_tf',
     'severity': 'registered_non_blocking',
     'detail': '本变体为 **IDF-only（不含 TF）**；文献（刘亚辉、牟爽、尹玉刚 2021）第(3)列稳健性为 '
               '**TF-IDF（含 TF）**，本变体对应其 **IDF 分量**，TF 部分未复现（差异已披露）'},
]


def rel(p) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p).replace('\\', '/')


def build_report(d: Path, y: int) -> dict:
    p = d / f'{y}_build_report.json'
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))


def structure_difference_facts(per_year: list, ew_freeze: Path, idf_freeze: Path) -> dict:
    """权重造成的**结构差异实况**（2026-10-05 审查修订；只描述既有产物，不重算任何数值）。"""
    k_ew = {r['text_year']: r['corpus_richness_ew'].get('K_clusters') for r in per_year}
    k_idf = {r['text_year']: r['corpus_richness_idf'].get('K_clusters') for r in per_year}
    chg = {y: round((k_idf[y] / k_ew[y] - 1) * 100, 1) for y in k_ew if k_ew[y] and k_idf[y]}
    jac = {r['text_year']: r['edge_jaccard_common_nodes'] for r in per_year}
    pm = {r['text_year']: [r['peer_mean_ew'], r['peer_mean_idf']] for r in per_year}
    out = {'edge_jaccard_common_nodes_by_year': jac,
           'edge_jaccard_range': [min(jac.values()), max(jac.values())] if jac else None,
           'K_by_year_ew': k_ew, 'K_by_year_idf': k_idf,
           'K_change_pct_by_year': chg,
           'K_change_pct_range': [min(chg.values()), max(chg.values())] if chg else None,
           'peer_mean_by_year_ew_idf': pm,
           'peer_median_masked_by_calibration': 'peer 中位数被「逐年校准到 44」掩盖；peer 均值不被掩盖（如 2014 年）'}
    def prec(fp, key):
        if not fp.exists():
            return {'available': False}
        d = json.loads(fp.read_text(encoding='utf-8')).get('years', {})
        modes = {}
        dev = 0.0
        mx = 0.0
        for y, v in d.items():
            modes[v.get('K_mode')] = modes.get(v.get('K_mode'), 0) + 1
            dv = abs(float(v.get('peer_median', 44)) - 44.0)
            dev += dv
            mx = max(mx, dv)
        return {'available': True, 'report': rel(fp), 'K_mode_counts': modes,
                'sum_abs_peer_median_dev_from_44': dev, 'max_abs_dev': mx}
    out['K_calibration_precision'] = {'ew': prec(ew_freeze, 'ew'), 'idf': prec(idf_freeze, 'idf'),
                                     'note': '两侧命中精度相当 → **不得**表述为「IDF 的 K 精度更低」'}
    out['reading'] = ('权重改变使聚类划分与边集显著变化（边 Jaccard ' 
                      + (f'{out["edge_jaccard_range"][0]}–{out["edge_jaccard_range"][1]}' if out['edge_jaccard_range'] else 'NA')
                      + '、K 系统性下降 ' + (f'{abs(out["K_change_pct_range"][1]):.1f}%–{abs(out["K_change_pct_range"][0]):.1f}%'
                                             if out['K_change_pct_range'] else 'NA')
                      + '、peer 均值下降但因「目标 44 校准」被中位数掩盖），而行业集中的**方向**一致且更强、'
                        '下游方向未反向 → 结论只能写成「方向一致且更强 + 结构层差异显著」。')
    return out


def invariant_block(ew_dir: Path, idf_dir: Path, years: list) -> dict:
    """强不变量：命中集合与样本筛选与权重无关 → n_nodes / filter_stats 逐年逐值相等。"""
    rows, ok_all = [], True
    keys = ('docs', 'drop_menlei', 'drop_ipo', 'drop_min', 'firms_kept', 'active_vocab', 'mean_terms')
    for y in years:
        a, b = build_report(ew_dir, y), build_report(idf_dir, y)
        if not a or not b:
            rows.append({'text_year': y, 'status': 'missing'})
            ok_all = False
            continue
        fs_a, fs_b = a.get('filter_stats', {}), b.get('filter_stats', {})
        diff = {k: [fs_a.get(k), fs_b.get(k)] for k in keys if fs_a.get(k) != fs_b.get(k)}
        eq = bool(a.get('n_nodes') == b.get('n_nodes') and not diff)
        ok_all = ok_all and eq
        rows.append({'text_year': y, 'status': 'ok',
                     'n_nodes_ew': a.get('n_nodes'), 'n_nodes_idf': b.get('n_nodes'),
                     'filter_stats_ew': {k: fs_a.get(k) for k in keys},
                     'filter_stats_idf': {k: fs_b.get(k) for k in keys},
                     'diff': diff, 'equal': eq})
    return {'rule': '权重不改变命中集合与样本筛选：n_nodes 与 filter_stats（docs/drop_menlei/drop_ipo/'
                    'drop_min/firms_kept/active_vocab/mean_terms）逐年必须逐值相等；只有相似度分布、'
                    'K_y、边权可变化',
            'per_year': rows, 'all_equal': ok_all}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ew-dir', '--self-dir', dest='ew_dir', default=str(EW_DIR),
                    help='等权（主口径）网络目录；--self-dir 为兼容别名')
    ap.add_argument('--idf-dir', '--teacher-dir', dest='idf_dir', default=str(IDF_DIR),
                    help='IDF 变体网络目录；--teacher-dir 为兼容别名')
    ap.add_argument('--years', type=int, nargs='*', default=YEARS, help='文本年（缺省 2014-2022）')
    ap.add_argument('--downstream-report', default=str(REPD / 's3_ma_downstream_consistency_idf.json'),
                    help='④ 下游方向对照报告（network_business_downstream_consistency.py --out-json；缺失则记 pending）')
    ap.add_argument('--ew-freeze', default=str(REPD / 's3_b3_ma_param_freeze_teacher.json'),
                    help='等权侧 K 冻结报告（仅用于 K_mode/命中精度实况，缺失则该项记 unavailable）')
    ap.add_argument('--idf-freeze', default=str(REPD / 's3_b3_ma_param_freeze_teacher_idf.json'),
                    help='IDF 侧 K 冻结报告（同上）')
    ap.add_argument('--out-json', default=str(REPD / 's3_ma_weighting_cross_audit.json'))
    ap.add_argument('--out-md', default=str(REPD / 's3_ma_weighting_cross_audit.md'))
    args = ap.parse_args()

    ew_dir = Path(args.ew_dir).resolve()
    idf_dir = Path(args.idf_dir).resolve()
    sw2 = CA.load_sw2()

    per_year, pool_ew, pool_idf = [], [], []
    for y in [int(v) for v in args.years]:
        if not (ew_dir / f'{y}_similarity.csv').exists() or not (idf_dir / f'{y}_similarity.csv').exists():
            per_year.append({'text_year': y, 'status': 'missing'})
            continue
        a = CA.read_edges(ew_dir, y)          # 等权（主口径）
        b = CA.read_edges(idf_dir, y)         # IDF 变体
        sa, sb = CA.read_support(ew_dir, y), CA.read_support(idf_dir, y)
        na = set(a.focal_id) | set(a.source_id)
        nb = set(b.focal_id) | set(b.source_id)
        common = na & nb

        pa, pb = CA.peer_stats(sa, a), CA.peer_stats(sb, b)
        ca = a.groupby('focal_id').size(); cb = b.groupby('focal_id').size()
        pool_ew.extend([(y, f, int(ca.get(f, 0))) for f in sa['focal_id']])
        pool_idf.extend([(y, f, int(cb.get(f, 0))) for f in sb['focal_id']])

        la = CA.sw2_lift(a, sw2, set(sa['focal_id']))
        lb = CA.sw2_lift(b, sw2, set(sb['focal_id']))
        A = set(map(tuple, a.loc[a.focal_id.isin(common) & a.source_id.isin(common),
                                 ['focal_id', 'source_id']].values))
        B = set(map(tuple, b.loc[b.focal_id.isin(common) & b.source_id.isin(common),
                                 ['focal_id', 'source_id']].values))
        jac = len(A & B) / max(1, len(A | B))
        ka = CA.cluster_consistency(ew_dir, y, a)
        kb = CA.cluster_consistency(idf_dir, y, b)
        ia, ib = CA.isolation(a, sa), CA.isolation(b, sb)

        rel_pm = abs(pb['peer_median'] - pa['peer_median']) / pa['peer_median'] if pa['peer_median'] else None
        per_year.append({
            'text_year': y, 'network_year': y + 1, 'status': 'ok',
            'nodes_ew': ia['nodes_total'], 'nodes_idf': ib['nodes_total'],
            'nodes_common': len(common),
            'edges_ew': pa['edges'], 'edges_idf': pb['edges'],
            'isolation_rate_ew': ia['isolation_rate'], 'isolation_rate_idf': ib['isolation_rate'],
            'peer_median_ew': pa['peer_median'], 'peer_median_idf': pb['peer_median'],
            'peer_median_edges_only_ew': pa['peer_median_edges_only'],
            'peer_median_edges_only_idf': pb['peer_median_edges_only'],
            'peer_mean_ew': pa['peer_mean'], 'peer_mean_idf': pb['peer_mean'],
            'peer_p90_ew': pa['peer_p90'], 'peer_p90_idf': pb['peer_p90'],
            'isolated_share_ew': pa['isolated_share'], 'isolated_share_idf': pb['isolated_share'],
            'peer_median_rel_diff': round(rel_pm, 4) if rel_pm is not None else None,
            'same_sw2_share_ew': la['same_sw2_share'], 'same_sw2_share_idf': lb['same_sw2_share'],
            'sw2_lift_ew': la['lift'], 'sw2_lift_idf': lb['lift'],
            'sw2_base_canonical_ew': la['base_rate_canonical'],
            'sw2_base_canonical_idf': lb['base_rate_canonical'],
            'sw2_lift_legacy_baseline_ew': la['lift_edge_endpoint_baseline_legacy'],
            'sw2_lift_legacy_baseline_idf': lb['lift_edge_endpoint_baseline_legacy'],
            'edge_jaccard_common_nodes': round(jac, 4),
            'cluster_consistency_ew': ka['consistency'], 'cluster_consistency_idf': kb['consistency'],
            'corpus_richness_ew': CA.corpus_richness(ew_dir, y),
            'corpus_richness_idf': CA.corpus_richness(idf_dir, y),
            'density_diagnostics': CA.density_diagnostics(a, b, common),
        })

    ok = [r for r in per_year if r.get('status') == 'ok']
    pe = [r[2] for r in pool_ew]
    pi = [r[2] for r in pool_idf]
    pooled = {'n_focal_year_ew': len(pool_ew), 'n_focal_year_idf': len(pool_idf),
              'peer_median_ew': float(np.median(pe)) if pe else None,
              'peer_median_idf': float(np.median(pi)) if pi else None}
    pooled['peer_median_rel_diff'] = (
        round(abs(pooled['peer_median_idf'] - pooled['peer_median_ew'])
              / pooled['peer_median_ew'], 4) if pooled['peer_median_ew'] else None)

    inv = invariant_block(ew_dir, idf_dir, [r['text_year'] for r in ok])

    # ④ 下游方向（读 network_business_downstream_consistency.py 的报告；缺失 → pending）
    dp = Path(args.downstream_report)
    e_block = {'status': 'pending', 'gating': False, 'report': rel(dp) if dp.exists() else None}
    if dp.exists():
        dd = json.loads(dp.read_text(encoding='utf-8'))
        h = dd.get('cross_audit_handoff', {})
        e_block = {'status': 'available', 'gating': False, 'report': rel(dp), 'run_at': dd.get('run_at'),
                   'overlap_months': dd.get('overlap_months'),
                   'effective_paired_months': dd.get('effective_paired_months'),
                   'effective_paired_months_note': dd.get('effective_paired_months_note'),
                   'peer_maxraw': {'pearson': (dd.get('peer_maxraw') or {}).get('pearson'),
                                   'spearman': (dd.get('peer_maxraw') or {}).get('spearman'),
                                   'n': (dd.get('peer_maxraw') or {}).get('n')},
                   'peer_return': {'pearson': (dd.get('peer_return') or {}).get('pearson'),
                                   'spearman': (dd.get('peer_return') or {}).get('spearman'),
                                   'n': (dd.get('peer_return') or {}).get('n')},
                   'quintile_agreement': dd.get('quintile_agreement'),
                   'portfolio_ew_by_peermaxraw': dd.get('portfolio_main_by_peermaxraw'),
                   'portfolio_idf_by_peermaxraw': dd.get('portfolio_robust_by_peermaxraw'),
                   'segments': dd.get('segments'),
                   'strength': dd.get('strength'),
                   'conclusion': dd.get('conclusion'),
                   'note': h.get('e_semantics') or dd.get('status_semantics')}

    rep = {
        'step': 'S3 Phase B″10 同语料权重对照审计（老师 MD&A：等权 0/1 ↔ IDF 加权）',
        'audit_version': 'v2 (2026-10-05, 审查修订：结论口径收紧 + 结构差异实况 + 已知限制；数值未改)',
        'descriptive_only': True, 'gating': False,
        'note': NOT_A_GATE_NOTE,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'years_text': [r['text_year'] for r in ok], 'years_network': [r['network_year'] for r in ok],
        'ew_dir': rel(ew_dir), 'idf_dir': rel(idf_dir),
        'peer_metric_definition': CA.PEER_MEDIAN_DEF,
        'lift_definition': CA.LIFT_DEF,
        'idf_rule': ('idf_{t,y} = ln((1+N_y)/(1+df_{t,y})) + 1（smooth；与 M-B 侧 sklearn '
                     'TfidfVectorizer 默认 smooth_idf=True 可比）；df_{t,y} = '
                     'product_lexicon_v2_teacher_df_by_year.csv 的 doc_count_{t,y}（该文本年 ok 文档中'
                     '命中该词的文档数），N_y = n_docs_{y}（该文本年 ok 文档总数，实测 2620–4931）；'
                     '纯式对照列 ln(N_y/df_{t,y}) 仅可作说明，不作为实现口径；dtype 保持 float32；'
                     '**只 IDF 加权、不含 TF**（与 M-B 的 TF-IDF 不同）'),
        'invariant_check': inv,
        'conclusion_descriptive': CONCLUSION_DESCRIPTIVE,
        'forbidden_readings': FORBIDDEN_READINGS,
        'structure_difference_facts': structure_difference_facts(
            ok, Path(args.ew_freeze), Path(args.idf_freeze)),
        'known_limitations': KNOWN_LIMITATIONS,
        'per_year': per_year, 'pooled': pooled, 'downstream_direction': e_block,
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# S3 Phase B″10 同语料权重对照（等权 0/1 ↔ IDF 加权，**描述性、非闸门**）', '',
          f'生成时间：{rep["run_at"]}　文本年：{rep["years_text"]}（网络年 {rep["years_network"]}）', '',
          f'- 等权侧（主口径）：`{rep["ew_dir"]}`（`pm_ma_ewcos_teacher_v1_{{y}}`）',
          f'- IDF 侧（对照变体）：`{rep["idf_dir"]}`（`pm_ma_idf_teacher_v1_{{y}}`）', '',
          '**定性**：' + NOT_A_GATE_NOTE, '',
          '## 强不变量核对（权重不改变命中集合与样本筛选）', '',
          f'规则：{inv["rule"]}', '',
          '| 文本年 | n_nodes(等权/IDF) | filter_stats 差异项 | 相等 |', '|---|---|---|---|']
    for r in inv['per_year']:
        if r.get('status') != 'ok':
            md.append(f'| {r["text_year"]} | — | — | 缺文件 |')
            continue
        md.append(f'| {r["text_year"]} | {r["n_nodes_ew"]}/{r["n_nodes_idf"]} | '
                  f'{r["diff"] if r["diff"] else "无"} | {"✔" if r["equal"] else "✘"} |')
    md += ['', f'**逐年全部相等：{inv["all_equal"]}**', '',
           '## ① 同申万二级边占比　② lift（同样本边际基线）　③ peer 规模（统一口径，目标 44）', '',
           '| 文本年 | 网络年 | 边数(等权/IDF) | same_sw2(等权/IDF) | lift(等权/IDF) | '
           'peer中位(等权/IDF) | 隔离率(等权/IDF) | 边Jaccard(描述) |', '|---|---|---|---|---|---|---|---|']
    for r in ok:
        md.append(f'| {r["text_year"]} | {r["network_year"]} | {r["edges_ew"]:,}/{r["edges_idf"]:,} | '
                  f'{r["same_sw2_share_ew"]}/{r["same_sw2_share_idf"]} | '
                  f'{r["sw2_lift_ew"]}/{r["sw2_lift_idf"]} | '
                  f'{r["peer_median_ew"]:.0f}/{r["peer_median_idf"]:.0f} | '
                  f'{r["isolation_rate_ew"]:.3f}/{r["isolation_rate_idf"]:.3f} | '
                  f'{r["edge_jaccard_common_nodes"]} |')
    md += ['', f'池化 peer 中位（统一口径，全部入网公司含零对手）：等权 {pooled["peer_median_ew"]:.0f} / '
               f'IDF {pooled["peer_median_idf"]:.0f}（n_firm-year = {pooled["n_focal_year_ew"]}/'
               f'{pooled["n_focal_year_idf"]}），相对差 = {pooled["peer_median_rel_diff"]}。'
               '注：**逐年** K 校准目标均为 44（年际口径 = 逐年中位，非 pooled）。', '',
           '## ④ 下游方向（PeerMAXraw / PeerReturn；来自 network_business_downstream_consistency.py）', '']
    if e_block['status'] != 'available':
        md += ['待产出：`reports/data_gates/G0/s3_ma_downstream_consistency_idf.json`。']
    else:
        pm, pr = e_block['peer_maxraw'], e_block['peer_return']
        q = e_block['quintile_agreement'] or {}
        pe_pf = e_block['portfolio_ew_by_peermaxraw'] or {}
        pi_pf = e_block['portfolio_idf_by_peermaxraw'] or {}
        md += [f'报告：`{e_block["report"]}`（run_at {e_block.get("run_at")}）',
               f'- 重叠月份 {e_block.get("overlap_months")}；有效配对 '
               f'{e_block.get("effective_paired_months")} 个月',
               f'- PeerMAXraw：Pearson {pm.get("pearson")} / Spearman {pm.get("spearman")}（n={pm.get("n"):,}）',
               f'- PeerReturn：Pearson {pr.get("pearson")} / Spearman {pr.get("spearman")}（n={pr.get("n"):,}）',
               f'- 五分位同组率 {q.get("same_quintile_share_mean")}（±1 组 '
               f'{q.get("within_plus_minus_1_quintile_share_mean")}；组序 Spearman 中位 '
               f'{q.get("quintile_spearman_median")}）',
               f'- 多空（Q5−Q1，按 PeerMAXraw 分组、等权次月收益）：等权 {pe_pf.get("long_short_q5_minus_q1_mean")}'
               f'（t={pe_pf.get("long_short_t_stat")}）；IDF {pi_pf.get("long_short_q5_minus_q1_mean")}'
               f'（t={pi_pf.get("long_short_t_stat")}）',
               f'- 组均秩相关：等权 {pe_pf.get("quintile_rank_corr_spearman")}'
               f'（{pe_pf.get("quintile_rank_corr_direction")}）；IDF '
               f'{pi_pf.get("quintile_rank_corr_spearman")}（{pi_pf.get("quintile_rank_corr_direction")}）',
               f'- 定性声明：{e_block.get("note")}']
    sdf = rep['structure_difference_facts']
    md += ['', '## 结论（描述性、非闸门；2026-10-05 审查修订后口径）', '',
           f'- {CONCLUSION_DESCRIPTIVE}',
           f'- 结构差异实况：边 Jaccard（共同节点）{sdf["edge_jaccard_range"]}'
           f'（逐年 {sdf["edge_jaccard_common_nodes_by_year"]}）；'
           f'K 等权 {list(sdf["K_by_year_ew"].values())} → IDF {list(sdf["K_by_year_idf"].values())}'
           f'（变化 {sdf["K_change_pct_range"]}%）；peer 均值（等权/IDF）'
           f'{sdf["peer_mean_by_year_ew_idf"]}——{sdf["peer_median_masked_by_calibration"]}',
           f'- K 命中精度（**不得**写成「IDF 更低」）：{sdf["K_calibration_precision"]["note"]}；'
           f'等权 {sdf["K_calibration_precision"]["ew"].get("K_mode_counts")}'
           f'（Σ|偏离| = {sdf["K_calibration_precision"]["ew"].get("sum_abs_peer_median_dev_from_44")}、'
           f'最大 {sdf["K_calibration_precision"]["ew"].get("max_abs_dev")}）；'
           f'IDF {sdf["K_calibration_precision"]["idf"].get("K_mode_counts")}'
           f'（Σ|偏离| = {sdf["K_calibration_precision"]["idf"].get("sum_abs_peer_median_dev_from_44")}、'
           f'最大 {sdf["K_calibration_precision"]["idf"].get("max_abs_dev")}）',
           '- **必须避免的表述**：' + '；'.join(FORBIDDEN_READINGS),
           '', '## 已知限制（登记项，非缺陷）', '']
    for lim in KNOWN_LIMITATIONS:
        if lim['id'] == 'idf_chain_float32_level_reproducibility':
            m = lim['measured_20261005']
            md += [f'- **{lim["id"]}**（{lim["severity"]}）：{lim["mechanism"]}。',
                   f'  - 实测（文本年 {m["text_year"]}，PYTHONHASHSEED={m["hash_seeds"]}）：边集（focal, source 对）'
                   f'**完全一致**（{m["n_edges"]:,} 条）；similarity max|Δ| = '
                   f'{m["similarity_max_abs_diff_seed1_vs_seed2"]:.2e}（均值 {m["similarity_mean_abs_diff"]:.2e}）；'
                   f'对**已落盘产物** max|Δ| = {m["max_abs_diff_vs_production_artifact"]}；聚类标签 sha256 '
                   f'**两侧与产物一致**（K = {m["K"]}、peer 中位 = {m["peer_median"]}、`filter_stats` 相等）；'
                   f'等权链逐字节可复现 = {m["ew_chain_byte_reproducible"]}；float32 eps = {m["float32_eps"]:.2e}'
                   f'（审查方在另一样本测得 {m["reviewer_measured_max_abs_diff_other_sample"]:.2e}，同量级）。',
                   f'  - 范围：{lim["scope_note"]}。证据 `{lim["evidence"]}`。',
                   f'  - 影响：{lim["impact"]}。']
        else:
            md += [f'- **{lim["id"]}**（{lim["severity"]}）：{lim["detail"]}。']

    Path(args.out_md).write_text('\n'.join(md), encoding='utf-8')
    print('invariant all_equal =', inv['all_equal'])
    print('->', outp)


if __name__ == '__main__':
    main()