#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3.1：申万 **as-of 分类口径** shadow —— as-of 映射 / 8 个行业基线 / K 目标 / N1 网络。

范围（`artifacts/phase3/3.0_plan.md` §2.1 + 项目已定决策）：
  - 只写 shadow：`data/processed/phase3_shadow/*`、`data/processed/network_ma_teacher_p3asof/*`、
    `artifacts/phase3/3.1_*`。**不改** headline 边表/样本/M1/PC/gate/configs；不重跑 headline M1/PC。
  - 本脚本不重建窄词表（3.3 决策）、不建 M-B shadow（留 Phase 4）、不做逐年 latest 目标变体（N5 → 3.2）。

as-of 定义（**口径限定，不得写成严格 PIT**）：
  as-of 标签 = 由「**分类生效日**」(`sw2_industry_map.csv.effective_from`) 取的当期申万分类：
    月频基线：`effective_from <= 月首日`，取 `max(effective_from)`；
    K/τ 目标：`effective_from <= (文本年+1)-04-30`（与网络 `available_at` 规则一致）。
  `sw2_industry_map.last_updated` 是**快照批量刷新字段**（与 `effective_from` 中位滞后 1,343 天、
  最早仅到 2015、22.8% 的行在 180 天内刷新）→ 记录可得日口径不可行，本脚本**不用** `last_updated`
  作 PIT 门。因此本口径是「分类生效日口径」，**不是严格 PIT**。
  已知断点：申万 2021 版自 **2021-07-30** 生效 → L2 组数 2014–2020 ≈102–104、2021–2022 = 131；
  两版码表拼接不做事后映射（不做无据的跨版本码表映射）。

步骤（`--steps`，缺省 all = map,baselines,targets,network,accept）：
  preflight   9 个 M1 输入锚 + 额外只读输入 + headline 网络全文件 sha 台账（`3.1_anchor_pre.json`）
  map         as-of 映射（`phase3_shadow/sw2_asof_map_p3shadow.parquet`）+ 边界/缺失/2021 切换审计
  baselines   8 个行业基线（LOO 同算法，仅换连接键）+ 与 headline 对比
              （`phase3_shadow/ind_baselines_p3asof.parquet`、`3.1_baseline_diff*.csv`）
  targets     as-of 与 latest 两套池化/逐年 K 目标（`3.1_k_targets.json`）
  network     N1 `p3asof` shadow 网络：复用 `s3_b3_ma_freeze.py` / `s3_ma_build_edges.py`
              （**不改磁盘脚本**；运行时把 `CORPORA['teacher']['ver_prefix']` 注入
              `pm_ma_ewcos_teacher_p3asof_v1`，并把 freeze 的默认 out_dir/report 也指向 shadow）
  accept      逐年边数/K/peer/isolated/lift（latest + as-of 两套）、vs headline 的 Jaccard/密度
  postflight  再次核验锚与 headline 网络 sha（`3.1_anchor_post.json`）

用法：
  python scripts/data_audit/p3_1_build_asof_shadow.py                       # 全链
  python scripts/data_audit/p3_1_build_asof_shadow.py --steps map,baselines
  python scripts/data_audit/p3_1_build_asof_shadow.py --steps network --years 2014 2015
"""
from __future__ import annotations

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import json_default as _json_default
from lib.shadow.p3_1_build_asof_shadow import ALLOWED_ROOTS, ANCHOR_POST, ANCHOR_PRE, ART, ASOF_BASELINES, ASOF_MAP, ASOF_RULE, BANNED_ROOTS, BASELINE_DIFF_CSV, BASELINE_DIFF_YEAR_CSV, COUNTS_JSON, EDGE_COLS, EXTRA_INPUTS, HEADLINE_EDGES, HEADLINE_IND_CACHE, HEADLINE_K, HEADLINE_NET, HEADLINE_NODES, HEADLINE_VERSION_PREFIX, IND_COLS, K_ASOF_PER_YEAR, K_ASOF_POOLED, K_LATEST_PER_YEAR, K_LATEST_POOLED, K_TARGETS_JSON, MAP_AUDIT_JSON, NETWORK_COMPARE_CSV, NET_DIR, OUTPUT_SHA, ROOT, SAMPLE_ASOF_BASELINES, SAMPLE_PARQUET, SHADOW_DIR, SHADOW_VERSION_PREFIX, SHADOW_VERSION_TAG, SW2_LATEST, SW2_MAP, TEACHER_LEXICON, TEACHER_PANEL, VINTAGE_SWITCH_DATE, YEARS, _atomic_write_text, _call_main, _variant_modules, acceptance, anchors_snapshot, asof_labels_for_key, asof_labels_for_months, assert_output_path_guard, build_industry_baselines_labelled, build_network_year, cluster_lift, compare_anchors, density, freeze_report_value, headline_latest_l2_map, jaccard, l2_map_for_key, load_src_month_safe, load_sw2_map, log, loo_handcheck, patch_corpora, read_clusters, read_edges, sha_ledger_headline_net, step_accept, step_baselines, step_ladder, step_map, step_network, step_sha_ledger, step_targets, sw2_k_target, write_csv, write_json, write_parquet
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

from s3_m1_lib import (  # noqa: E402
    SAMPLE_WINDOW_END, SAMPLE_WINDOW_START, assert_m1_inputs_sha, load_me, load_sw2,
    months_between, require, sha256_file,
)
from s4_step2_peermax_formal import (  # noqa: E402
    NETWORK_AVAILABLE_AT_RULE, load_versions, select_version,
)

# ---------------------------------------------------------------- 路径与常量


# K 目标（as-of L2 口径；`3.0_plan.md` §1.2 E2 预登记值，容差 0）
# headline 登记读数（只读对账；`3.0_plan.md` §1.1）

# 只读输入（sha 台账用；sw2_industry_latest 已在 9 锚内）


# ---------------------------------------------------------------- 输出路径守卫














# ---------------------------------------------------------------- 锚台账






# ---------------------------------------------------------------- as-of 映射










# ---------------------------------------------------------------- 行业基线（LOO）




# ---------------------------------------------------------------- K 目标




# ---------------------------------------------------------------- 网络构建（复用现脚本）








# ---------------------------------------------------------------- 网络读数 / 对比














# ---------------------------------------------------------------- steps
















# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Phase 3.1 as-of 行业 + K 目标 + p3asof shadow 网络')
    ap.add_argument('--steps', default='all',
                    help='all 或逗号分隔子集：preflight,map,baselines,targets,network,accept,ladder,postflight')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='network 文本年（缺省全 9 年）')
    args = ap.parse_args()
    steps = (['preflight', 'map', 'baselines', 'targets', 'network', 'accept', 'postflight']
             if args.steps == 'all' else [s.strip() for s in args.steps.split(',') if s.strip()])
    years = args.years if args.years else YEARS
    t_all = time.time()
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    counts = {'run_at': now, 'steps': steps,
              'shadow_version_tag': SHADOW_VERSION_TAG, 'years': years, 'run_history': []}
    # 增量运行合并：本次未执行的 step 保留上次结果（不丢审计轨迹），并记录 run_history
    if COUNTS_JSON.is_file():
        try:
            prev = json.loads(COUNTS_JSON.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            prev = None
        if isinstance(prev, dict):
            hist = prev.get('run_history') if isinstance(prev.get('run_history'), list) else []
            hist = hist + [{'run_at': prev.get('run_at'), 'steps': prev.get('steps'),
                            'elapsed_sec_total': prev.get('elapsed_sec_total')}]
            prev = {k: v for k, v in prev.items() if k != 'run_history'}
            prev.update({'run_at': now, 'steps': steps, 'years': years, 'run_history': hist})
            counts = prev

    if 'preflight' in steps:
        counts['anchor_pre'] = anchors_snapshot('pre')
        write_json(ANCHOR_PRE, counts['anchor_pre'])
        log(f'preflight：9 锚 + {len(counts["anchor_pre"]["headline_network_sha"])} 个 headline '
            f'网络文件 sha 已记录')

    months = months_between(SAMPLE_WINDOW_START, SAMPLE_WINDOW_END)
    m = load_sw2_map()
    if 'map' in steps:                      # 未执行的 step 保留上次结果（不写 None 覆盖）
        counts['map_audit'] = step_map(months, load_src_month_safe())

    if 'targets' in steps:
        teach = pd.read_csv(TEACHER_PANEL, usecols=['firm_id', 'year', 'status'])
        panel_teach = teach[teach['status'] == 'ok'].copy()
        panel_teach['firm_id'] = panel_teach['firm_id'].astype(str)
        counts['k_targets'] = step_targets(panel_teach)

    if 'baselines' in steps:
        counts['baselines'] = step_baselines(months)
    if 'network' in steps:
        net_res = step_network([int(y) for y in years],
                               (counts.get('k_targets') or {}).get('asof', {}).get(
                                   'median_peer_pooled_rounded', K_ASOF_POOLED))
        prev_net = counts.get('network') or {}
        if isinstance(prev_net.get('years'), dict):        # 增量运行：保留其它年份的读数
            merged = dict(prev_net['years'])
            merged.update(net_res['years'])
            net_res['years'] = merged
            net_res['years_rebuilt_this_run'] = [str(y) for y in years]
        counts['network'] = net_res
    if 'accept' in steps:
        rows, detail = step_accept(m, [int(y) for y in years])
        counts['network_compare'] = rows
        counts['lift_detail'] = {str(k): v for k, v in detail.items()}
        counts['output_sha_rerun'] = step_sha_ledger()
    if 'ladder' in steps:
        counts['k_ladder'] = step_ladder([int(y) for y in years])

    if 'postflight' in steps:
        post = anchors_snapshot('post')
        write_json(ANCHOR_POST, post)
        pre = json.loads(ANCHOR_PRE.read_text(encoding='utf-8')) if ANCHOR_PRE.is_file() else None
        cmp_ = compare_anchors(pre, post) if pre else {'status': 'NO_PRE'}
        counts['anchor_compare'] = cmp_
        log(f'postflight：锚对比 {cmp_["status"]}（diff={cmp_["n_diff"]}）')
        if cmp_['status'] != 'MATCH':
            write_json(COUNTS_JSON, counts)
            print('[HARD-GATE] headline/输入锚发生变化；停机（禁止重算覆盖）', file=sys.stderr, flush=True)
            raise SystemExit(2)

    counts['elapsed_sec_total'] = round(time.time() - t_all, 1)
    write_json(COUNTS_JSON, counts)
    log(f'完成：总耗时 {counts["elapsed_sec_total"]}s → {COUNTS_JSON}')


if __name__ == '__main__':
    main()
