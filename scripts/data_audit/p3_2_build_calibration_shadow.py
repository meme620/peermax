#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3.2：L-2 / L-6 校准 shadow 变体（逐年 target / 逐年 df / 组合 / latest 对照）。

范围（`artifacts/phase3/3.0_plan.md` §2.2 + 项目 3.2 任务书）：
  - 只写 shadow：`data/processed/phase3_shadow/lexicon_p3ydf_{y}.csv`、
    `data/processed/network_ma_teacher_p3{yt,ydf,yt_ydf,yt_latest}/*`、`artifacts/phase3/3.2_*`。
  - **不改** headline 边表/样本/网络/M1/PC/gate/configs；不重跑 headline M1/PC；不写 G0；
    不碰 3.1 产物（`artifacts/phase3/3.1_*`、`network_ma_teacher_p3asof/`）——只读依赖。

四个变体（两轴：词表 df × K 目标映射；矩阵见 `VARIANTS`）：

| 变体 | 词表 | K 目标 | 隔离的效应 |
|---|---|---|---|
| `p3yt`        | headline（9 年上确界 df≤0.25） | 逐年 as-of 目标 38/40/44/49/50/52/68/57/64 | L-2：K 目标「逐年 vs 池化 44」 |
| `p3ydf`       | 逐年 df≤0.25（9 个 shadow 词表） | 池化 44（= headline 目标） | L-6：词表 df「逐年 vs 9 年上确界」 |
| `p3yt_ydf`    | 逐年 df | 逐年 as-of 目标 | 组合（两轴交互） |
| `p3yt_latest` | headline | 逐年 latest 目标 26/27/30/38/37/46/52/58/62 | 目标映射（latest vs as-of 逐年目标） |

**口径限定（不得改写）**：
  1. as-of = **申万分类生效日口径**（`effective_from <= (文本年+1)-04-30`），**不是严格 PIT**（见 3.1 §0）。
  2. **L2 映射不进入网络构造**：`build_vectors` / 聚类 / 边表只吃（语料 × 词表 × K 目标）；`l2_code` 只出现在
     (a) 逐年 K 目标的来源（as-of vs latest）、(b) lift 分式的行业标签。故 `p3yt` 与 `p3yt_latest` 的网络
     差异 = 目标向量差异（不是「标签进入网络」）；标签口径对网络的**构造性影响 = 0**（可证伪点：
     `test_l2_labels_do_not_enter_network_construction`）。
  3. 逐年词表 df：`df_ratio_y = doc_count_{y} / n_docs_{y}`（`product_lexicon_v2_teacher_df_by_year.csv`），
     未出现在该表的词条 df=0（保留）；阈值 0.25 与 headline 同。**不改匹配器**（`--lexicon` 原路径）。
  4. 逐年目标的可达性（**两级 K 选择**）：
     一级（plan §2.2 处方）= `s3_b3_ma_freeze.calibrate_K` 的二分 + 网格命中（`K_mode ∈ {exact, closest}`）。
     该二分假设「中位 peer 对 K 单调不增」，且**终局候选只有 h−l ≤ 2 个 K** → 目标落在阶梯跳点上时
     `closest` 会偏出容差（实测 8 个 (变体,年) 的 |dev| > 2）。
     二级（本脚本）= 当一级 |dev| > 2 时，在「freeze 选定 K ∪ 参照网络 K」±40 的**扩展阶梯**上重选 K
     （|dev| 最小优先、其次与 freeze K 最近；未命中则再扩窗），落地于 `--steps rebuild`
     （内存 patch `calibrate_K`，聚类算法与全部参数不变）。两级结果逐值留
     `3.2_k_selection.csv` / `3.2_k_ladder_{headline,per_year}.csv`。
     3.1 的 `3.1_k_ladder.csv` 窗口按池化目标 49 建（9 年里仅 2018 的目标 50 落在窗口内）→ 覆盖不到逐年
     目标，故本脚本另建扩展阶梯。

步骤（`--steps`，缺省 all = preflight,lexicons,networks,ladder,select,rebuild,compare,postflight）：
  preflight   9 个 M1 输入锚 + 额外只读输入 + headline 网络全文件 sha 台账（`3.2_anchor_pre.json`）
  lexicons    逐年词表（`phase3_shadow/lexicon_p3ydf_{y}.csv`）+ 计数/命中质量 + `3.2_df_removed_terms.csv`
  networks    4 变体 × 9 年：复用 `s3_b3_ma_freeze.py` / `s3_ma_build_edges.py`（**不改磁盘脚本**，
              运行时注入 shadow `ver_prefix` / K 规则标签 / shadow 默认输出路径）；可增量、可续跑
              （已完成年跳过；`--force` 强制重建；`3.2_progress_{variant}.json` 逐落盘）
  ladder      逐年 K-中位 peer 扩展阶梯（headline 词表 + 逐年词表两套）+ 目标可达性 + **二级 K 选择**
              （一级 freeze 自校准 |dev| > 2 → 扩展阶梯重选；落 `3.2_k_selection.csv`）
  select      汇总两套词表的可达性 → 合并 `3.2_k_selection.csv`（从阶梯表复算，不重扫）
  rebuild     落地二级 K 选择（内存 patch `calibrate_K`，只重建 K 有变的年）
  compare     变体 × headline × p3asof 对比表（边数/Jaccard/密度/peer 中位与 p90/lift 两口径/楔子/K/
              isolated）+ 验收硬门 + 词表/目标效应分解；`3.2_comparison.csv` / `3.2_wedge.csv` /
              `3.2_freeze_summary.csv` / `3.2_output_sha.json` / `3.2_counts.json`
  postflight  再次核验锚与 headline 网络 sha（`3.2_anchor_post.json`）；CHANGED → 停机

用法：
  python scripts/data_audit/p3_2_build_calibration_shadow.py --steps all
  python scripts/data_audit/p3_2_build_calibration_shadow.py --steps lexicons,compare
  python scripts/data_audit/p3_2_build_calibration_shadow.py --steps networks --variants p3yt --years 2014 2015
"""
from __future__ import annotations

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import json_default as _json_default
from lib.shadow.p3_2_build_calibration_shadow import ALLOWED_ROOTS, ANCHOR_POST, ANCHOR_PRE, ART, BANNED_ROOTS, COMPARISON_CSV, COUNTS_JSON, DF_BY_YEAR, DF_MAX, DF_REMOVED_CSV, EDGE_COLS, EXTRA_INPUTS, FREEZE_SUMMARY_CSV, HEADLINE_EDGES, HEADLINE_K, HEADLINE_NET, HEADLINE_NODES, HEADLINE_VERSION_PREFIX, K_ASOF_PER_YEAR, K_ASOF_POOLED, K_HEADLINE_POOLED, K_LATEST_PER_YEAR, K_SELECTION_CSV, K_TARGETS_3_1, K_TOLERANCE, LADDER_3_1, LADDER_MAX_EXTEND, LADDER_WINDOW, LEXICON_BY_YEAR_CSV, OUTPUT_SHA, P3ASOF_NET, P3ASOF_VERSION_PREFIX, READONLY_INPUTS, REF_NETS, REGISTERED_HIT_MASS_SHARE, REGISTERED_HIT_MASS_TOL, REGISTERED_RESCUED_WITH_HITS, REP_VARIANT_BY_MODE, ROOT, SHADOW_DIR, SW2_MAP, TEACHER_LEXICON, TEACHER_PANEL, VARIANTS, VARIANT_ORDER, WEDGE_CSV, YEARS, _PRISTINE, __all__, _atomic_write_text, _best_choice, _call_main, _freeze_K, _variant_modules, anchors_snapshot, assert_output_path_guard, build_code_map, build_variant_year, build_yearly_lexicons, check_edge_contract, compare_rows, density, dk, edge_keys, effect_decomposition, freeze_summary_rows, jaccard_from_keys, lexicon_for, load_json_or, load_k_targets_3_1, log, net_dir, net_lift, net_prefix, networks_summary_from_disk, patch_corpora, progress_path, read_rep, read_support_stats, selection_from_ladder_csv, step_compare, step_ladder, step_networks, step_postflight, step_preflight, step_rebuild, step_select, step_sha_ledger, target_for, variant_pair_jaccard, version_spec_check, write_csv, write_json, write_text, year_built
# --- end scripts/lib bootstrap ---


import argparse
import csv
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(ROOT / 'scripts' / 'data_audit'))

import p3_1_build_asof_shadow as p31  # noqa: E402  （as-of 映射 / lift / 锚台账：3.1 交接件，只读复用）
from s3_m1_lib import assert_m1_inputs_sha, require, sha256_file  # noqa: E402
from s4_step2_peermax_formal import (  # noqa: E402
    NETWORK_AVAILABLE_AT_RULE, load_versions, select_version,
)

# ---------------------------------------------------------------- 路径与常量


# 3.1 交接的 K 目标（`3.1_k_targets.json` 逐值复核；容差 0）
# headline 登记读数（只读对账；`3.0_plan.md` §1.1 / 3.1 §5）
# 逐年 df 词表落盘（plan §2.2 R1 登记值：2014 多剔 560 词 = 0.66% 命中词 / 7.26% 命中质量；
# 2015 = 819 词 / 0.99% / 12.08%）
# 3.1 扩展阶梯窗口（±K）

# 变体登记（唯一真源）

# 只读输入（sha 台账；headline 词表 / df_by_year / sw2 映射不得被写）


# ---------------------------------------------------------------- 输出路径守卫














# ---------------------------------------------------------------- 锚台账（复用 3.1 语义）


# ---------------------------------------------------------------- 变体登记 / 目标










# ---------------------------------------------------------------- 逐年词表（L-6）


# ---------------------------------------------------------------- 网络构建（复用现脚本）


















# ---------------------------------------------------------------- 扩展 K 阶梯（目标可达性 + 二级 K 选择）




















# ---------------------------------------------------------------- 读数 / 对比 / 验收
































# ---------------------------------------------------------------- steps 入口




def main():
    ap = argparse.ArgumentParser(description='Phase 3.2 L-2/L-6 校准 shadow 变体（逐年 target / 逐年 df）')
    ap.add_argument('--steps', default='all',
                    help='all 或逗号分隔子集：preflight,lexicons,networks,ladder,rebuild,compare,postflight')
    ap.add_argument('--variants', nargs='*', default=None, choices=VARIANT_ORDER,
                    help='缺省全 4 个变体')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='文本年（缺省全 9 年）')
    ap.add_argument('--ladder-lexicon', nargs='*', default=['headline', 'per_year'],
                    choices=['headline', 'per_year'], help='扩展阶梯所用词表（缺省两套）')
    ap.add_argument('--force', action='store_true', help='networks：强制重建已完成年')
    ap.add_argument('--force-lexicons', action='store_true', help='lexicons：强制重写词表文件')
    args = ap.parse_args()
    steps = (['preflight', 'lexicons', 'networks', 'ladder', 'select', 'rebuild', 'compare', 'postflight']
             if args.steps == 'all' else [s.strip() for s in args.steps.split(',') if s.strip()])
    years = args.years if args.years else YEARS
    variants = args.variants if args.variants else VARIANT_ORDER
    t_all = time.time()
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    for v in VARIANT_ORDER:
        VARIANTS[v]['dir'].mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    counts = {'run_at': now, 'steps': steps, 'years': years, 'variants': variants,
              'shadow_version_tag': 'p3shadow_v1_p3yt|p3ydf|p3yt_ydf|p3yt_latest',
              'run_history': []}
    prev = load_json_or(COUNTS_JSON, None)
    if isinstance(prev, dict):
        hist = prev.get('run_history') if isinstance(prev.get('run_history'), list) else []
        hist = hist + [{'run_at': prev.get('run_at'), 'steps': prev.get('steps'),
                        'elapsed_sec_total': prev.get('elapsed_sec_total')}]
        prev = {k: v for k, v in prev.items() if k != 'run_history'}
        prev.update({'run_at': now, 'steps': steps, 'years': years, 'variants': variants,
                     'run_history': hist})
        counts = prev
    counts['variant_registry'] = {v: {k: (str(x.relative_to(ROOT)).replace('\\', '/')
                                          if isinstance(x, Path) else x)
                                      for k, x in VARIANTS[v].items()} for v in VARIANT_ORDER}
    counts['k_targets_3_1'] = {'asof_per_year': K_ASOF_PER_YEAR, 'latest_per_year': K_LATEST_PER_YEAR,
                               'asof_pooled': K_ASOF_POOLED, 'latest_pooled_headline': K_HEADLINE_POOLED,
                               'source': str(K_TARGETS_3_1.relative_to(ROOT)).replace('\\', '/')}

    if 'preflight' in steps:
        counts['anchor_pre'] = step_preflight()
    if 'lexicons' in steps:
        counts['lexicons'] = build_yearly_lexicons(years, force=args.force_lexicons)
    if 'networks' in steps or 'compare' in steps:
        tgt3_1 = load_k_targets_3_1()
        counts['k_targets_3_1']['verified_against_3_1_json'] = True
        counts['k_targets_3_1']['3_1_targets'] = {
            'asof': {str(p['text_year']): p['median_peer'] for p in tgt3_1['asof']['per_year']},
            'latest': {str(p['text_year']): p['median_peer'] for p in tgt3_1['latest']['per_year']},
        }
    if 'networks' in steps:
        counts['networks'] = step_networks(variants, years, force=args.force)
    if 'ladder' in steps:
        lad = counts.get('k_ladder') if isinstance(counts.get('k_ladder'), dict) else {}
        for m in args.ladder_lexicon:
            lad[m] = step_ladder([int(y) for y in years], m)
        counts['k_ladder'] = lad
    if 'select' in steps:
        sel_res = step_select([int(y) for y in years], args.ladder_lexicon)
        counts['k_ladder'] = {**(counts.get('k_ladder') or {}), **sel_res['k_ladder']}
        counts['k_selection'] = sel_res['k_selection']
        counts['k_selection_summary'] = {
            'csv': sel_res['k_selection_csv'], 'n_pairs': len(sel_res['k_selection']),
            'n_needs_rebuild': sel_res['n_needs_rebuild'],
            'n_dev_over_tolerance': sum(1 for x in sel_res['k_selection']
                                        if float(x['dev_selected']) > K_TOLERANCE),
            'tolerance': K_TOLERANCE,
            'note': ('|dev| > tolerance 的 (变体,年) 均为「目标在扩展阶梯窗口内不可达」'
                     '（跳点/平台）；`target_reachable_in_window=False` 与 '
                     '`best_attainable_dev_in_window` 逐条留档'),
        }
    if 'rebuild' in steps:
        sel_rows = counts.get('k_selection') or []
        if not sel_rows and K_SELECTION_CSV.is_file():
            sel_rows = pd.read_csv(K_SELECTION_CSV).to_dict('records')
        require(bool(sel_rows), 'rebuild：缺 K 选择表（先跑 --steps select）')
        counts['rebuild'] = step_rebuild([int(y) for y in years], variants, sel_rows)
    if 'compare' in steps:
        res, keys, detail = step_compare([int(y) for y in years], variants=variants)
        counts['comparison_rows'] = res['rows']
        counts['networks_from_disk'] = networks_summary_from_disk([int(y) for y in years], variants)
        counts['acceptance_flags'] = res['flags']
        counts['effect_decomposition'] = res['effect_decomposition']
        counts['compare_meta'] = {'comparison_csv': res['comparison_csv'],
                                  'wedge_csv': res['wedge_csv'],
                                  'freeze_summary_csv': res['freeze_summary_csv'],
                                  'n_rows': res['n_rows'], 'elapsed_sec': res['elapsed_sec']}
        counts['flag_counts'] = {}
        for f in res['flags']:
            for x in f['flags']:
                counts['flag_counts'][x] = counts['flag_counts'].get(x, 0) + 1
        counts['output_sha_rerun'] = step_sha_ledger([int(y) for y in years], variants)
    if 'postflight' in steps:
        counts['postflight'] = step_postflight(counts)

    counts['elapsed_sec_total'] = round(time.time() - t_all, 1)
    counts['run_at'] = now
    write_json(COUNTS_JSON, counts)
    log(f'完成：总耗时 {counts["elapsed_sec_total"]}s → {COUNTS_JSON}')


if __name__ == '__main__':
    main()
