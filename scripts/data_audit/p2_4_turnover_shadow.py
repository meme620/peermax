#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.4（审计 L-8）shadow：turnover 月末值 vs 月度均值 + shadow 控制集回归矩阵。

背景（`artifacts/phase2/2.0_plan.md` §2.4；`reports/audits/peermax_audit_item_review_20261027.md` L-8）：
  - headline 主回归 `turnover = load_turnover_drive()` = Drive 日换手率 `.resample('ME').last()`，
    即「月末最后一个有数据的交易日」快照；
  - `firm_chars_clean.turnover` 是月均值但文件旧（2014–2016 覆盖率 0），不可直接使用；
  - 本脚本在 shadow 层构造 `turnover_last` / `turnover_mean` / `n_days`（只读 pkl），
    并在**只读** M1 headline 样本上跑 4 个规格的 FM 矩阵（a 登记全控制 / b 均值替换 /
    c 无 turnover / d 无 turnover+无 rev），其余控制、outcome、估计器与 headline 完全一致。

口径与护栏：
  - 只读：`data/raw/drive/daily_turnover_rate_20260630.pkl`、M1 样本 parquet、G0 定价表 json；
  - 不重跑 M1/PC/网络；不写 9 个 M1 输入 / headline 面板 / 样本 / configs / 正式 reports；
    结果只写 `data/processed/phase2_shadow/`、`artifacts/phase2/`；
  - `turnover_last` 规格（a）**直接使用样本自身的 `turnover` 列**（winsorized last/100），
    不重新计算 last，避免 winsor 差异；`turnover_mean` 在 panel 宇宙（样本 parquet 全 307,503 行）
    上按 registry `winsorization_pct` 月度截面 winsor，与 headline `turnover` 的施加方式同构；
  - `load_turnover_drive()` / `_drive_monthly_wide()` 保持不动（`how='mean'` 在该函数里落到
    `.last()`，属已知陷阱，本脚本不复用该分支，mean 由本脚本显式计算）。

产出：
  - `data/processed/phase2_shadow/turnover_p2shadow.csv`（firm_id/month/turnover_last/turnover_mean/n_days）
  - `data/processed/phase2_shadow/turnover_p2shadow.meta.json`
  - `artifacts/phase2/2.4_counts.json`（计数、差异分位、四规格系数表、9 锚前后、对比）
  - `artifacts/phase2/2.4_turnover_diff.csv`（按年差异表）
  - `artifacts/phase2/2.4_control_matrix.csv`（四规格 × EW/VW 系数表，供 2.6 直接引用）

用法：
  python scripts/data_audit/p2_4_turnover_shadow.py
"""
from __future__ import annotations

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import atomic_write_text as _atomic_write_text
from lib.shadow.p2_4_turnover_shadow import CONTROLS, COUNTS_JSON, DIAG_SPEC_DEFS, DIFF_CSV, FM_WINDOW_END, FM_WINDOW_START, HEADLINE_GUARD_FILES, LAST_COL_RAW, MATRIX_CSV, MEAN_COL, MEAN_COL_FM, MEAN_COL_RAW, META_JSON, OUT_CSV, P2SHADOW_VERSION, PRICING_TABLE_V2, SAMPLE_PARQUET, SHADOW_DIR, SPEC_DEFS, TRIO, WINDOW_END, WINDOW_START, _to_long, _to_long_asis, assert_output_path_guard, audit_l8_block, build_control_frame, build_shadow_frame, headline_comparison, load_turnover_wide, matrix_rows, monthly_frames, parse_args, plan_replication, run_spec_matrix, window_diff_stats, write_diff_csv, write_json, write_matrix_csv, write_shadow_csv, yearly_diff_stats
# --- end scripts/lib bootstrap ---


import argparse
import json
import os
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s3_m1_lib import (  # noqa: E402
    ROOT, TURNOVER_PKL, assert_m1_inputs_sha, drive_code_to_firm_id, fm_regression,
    param_value, require, sha256_file, winsor_bounds,
)



# headline 关键文件（本脚本必须不写、且 sha 前后不变）

# shadow turnover CSV 窗口：2014-01..2024-01（与 2.0_plan §2.4 R1 的「匹配 446,301 行」窗口一致；
# 覆盖 M1 事件月 2015-01..2023-12 与 outcome 月 2015-05..2023-12）




# ---------------------------------------------------------------- 基础 IO










# ---------------------------------------------------------------- 差异统计




# ---------------------------------------------------------------- 控制集（样本只读）


# ---------------------------------------------------------------- FM 矩阵






# ---------------------------------------------------------------- 产物写出










# ---------------------------------------------------------------- 主流程


def main(argv=None):
    args = parse_args(argv)
    t_generated = datetime.now(timezone.utc).isoformat(timespec='seconds')
    print('[2.4] preflight：9 个 M1 输入锚（只读）', flush=True)
    anchors_before = assert_m1_inputs_sha()
    headline_sha_before = {k: sha256_file(p) for k, p in HEADLINE_GUARD_FILES.items()}

    print('[2.4] 读取 Drive turnover pkl 并聚合 last/mean/n_days', flush=True)
    pkl_path = Path(args.turnover_pkl)
    wide = load_turnover_wide(pkl_path)
    shadow = build_shadow_frame(wide)
    stats_full = window_diff_stats(_to_long_asis(wide))
    stats_plan_window = window_diff_stats(shadow)
    stats_fm_window = window_diff_stats(shadow, FM_WINDOW_START, FM_WINDOW_END)
    yearly = yearly_diff_stats(shadow, WINDOW_START, WINDOW_END)
    lo, hi = winsor_bounds()
    require(0.70 <= stats_plan_window['corr'] <= 0.85,
            f"plan 窗口 corr={stats_plan_window['corr']:.4f} 超出 [0.70,0.85]（停机条件）")
    require(stats_plan_window['mean_abs_diff_pp'] <= 2.0,
            f"plan 窗口平均绝对差 {stats_plan_window['mean_abs_diff_pp']:.4f}pp > 2pp（停机条件）")
    print(f"  rows={len(shadow):,} both={stats_plan_window['n_both']:,} "
          f"corr={stats_plan_window['corr']:.4f} mad={stats_plan_window['mean_abs_diff_pp']:.4f}pp",
          flush=True)

    out_info = write_shadow_csv(shadow, Path(args.out))
    sha = out_info['sha256']
    meta = {
        'p2shadow_version': P2SHADOW_VERSION,
        'artifact': Path(args.out).name,
        'generated_at_utc': t_generated,
        'script': 'scripts/data_audit/p2_4_turnover_shadow.py',
        'script_sha256': sha256_file(Path(__file__)),
        'output_sha256': sha,
        'row_count': int(len(shadow)),
        'row_count_last_mean_nonnull': int(stats_plan_window['n_both']),
        'columns': list(shadow.columns),
        'dtypes': {c: str(t) for c, t in shadow.dtypes.items()},
        'window': {'start': WINDOW_START, 'end': WINDOW_END,
                   'months': int(shadow['month'].nunique()),
                   'semantics': 'pkl 全 code × 窗口月格子；n_days==0 的格子 last/mean 为空（不填 0）'},
        'unit': 'percent（Drive 原始口径；回归使用 /100 并在样本 parquet 宇宙做月度 '
                f'[{lo},{hi}] winsor）',
        'source': {
            'file': str(pkl_path.relative_to(ROOT)),
            'sha256': sha256_file(pkl_path),
            'shape': [int(wide.shape[0]), int(wide.shape[1])],
            'index_range': [str(wide.index.min().date()), str(wide.index.max().date())],
            'read_only': True,
        },
        'turnover_last_semantics': ('当月最后一个非空日频换手率（pkl 对停牌日记 0，故整月停牌月可取值 0；'
                                    '与 headline load_turnover_drive() 的 .last() 同义，但本文件不做 /100、'
                                    '不做 winsor）'),
        'turnover_mean_semantics': '当月非空日频换手率的算术平均（skipna；停牌日的 0 计入分母）',
        'n_days_semantics': '当月非空日频观测数（含 0 值停牌日）；n_days==0 表示该月无任何日频数据',
        'created_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    write_json(Path(args.meta_out), meta)

    print('[2.4] 只读 M1 样本并构造控制矩阵', flush=True)
    fm, diag = build_control_frame(Path(args.sample), shadow)
    print(f"  FM rows={len(fm):,} mean 覆盖={diag['fm_turnover_mean_raw_coverage']:.4f} "
          f"last 桥 max|diff|={diag['last_bridge_max_abs_diff_vs_sample_turnover']:.2e}", flush=True)
    require(diag['fm_turnover_mean_raw_coverage'] >= 0.99, 'FM 样本 turnover_mean 覆盖 <99%（停机条件）')
    require(diag['last_bridge_max_abs_diff_vs_sample_turnover'] <= 1e-12,
            'panel-winsor(raw last) 与样本 turnover 列不一致（stop：不得重新计算 last）')

    print('[2.4] 四规格 FM 矩阵（EW/VW，min_n=100，NW lag=6）', flush=True)
    matrix = run_spec_matrix(fm)
    diag_defs = {k: {'label': v[0],
                     'x': TRIO + [v[1] if c == 'turnover' else c for c in CONTROLS],
                     'turnover_col': v[1], 'turnover_variant': 'mean-diagnostic'}
                 for k, v in DIAG_SPEC_DEFS.items()}
    diag_matrix = run_spec_matrix(fm, diag_defs)
    cmp_headline = headline_comparison(matrix)
    require(cmp_headline['all_match'], '规格 a 未能复现 headline 主表（停机条件）')
    for sid, blk in matrix['specs'].items():
        e = blk['ew']['coefs']
        print(f"  {sid}: EW PeerMAX {e['PeerMAXraw']['beta']:+.6f} "
              f"(t={e['PeerMAXraw']['t_nw']:+.3f}) | OwnMAX {e['OwnMAXraw']['beta']:+.6f} "
              f"(t={e['OwnMAXraw']['t_nw']:+.3f}) | T={blk['ew']['T']} n={blk['ew']['n_frame']:,}",
              flush=True)
    for sid, blk in diag_matrix['specs'].items():
        e = blk['ew']['coefs']
        print(f"  [{sid}] EW PeerMAX {e['PeerMAXraw']['beta']:+.6f} "
              f"(t={e['PeerMAXraw']['t_nw']:+.3f}) | OwnMAX {e['OwnMAXraw']['beta']:+.6f} "
              f"(t={e['OwnMAXraw']['t_nw']:+.3f})", flush=True)

    diff_info = write_diff_csv(yearly, Path(args.diff_out))
    matrix_info = write_matrix_csv(matrix, Path(args.matrix_out))

    print('[2.4] postflight：9 个 M1 输入锚与 headline 文件 sha 复核', flush=True)
    anchors_after = assert_m1_inputs_sha()
    headline_sha_after = {k: sha256_file(p) for k, p in HEADLINE_GUARD_FILES.items()}
    anchors_match = bool([a['sha256'] for a in anchors_before]
                         == [a['sha256'] for a in anchors_after])
    headline_unchanged = headline_sha_before == headline_sha_after
    require(anchors_match, '9 锚前后 sha 不一致（停机条件）')
    require(headline_unchanged, 'headline 文件 sha 发生变化（停机条件）')

    counts = {
        'p2shadow_version': P2SHADOW_VERSION,
        'generated_at_utc': t_generated,
        'script': 'scripts/data_audit/p2_4_turnover_shadow.py',
        'script_sha256': sha256_file(Path(__file__)),
        'scope': ('只读 pkl/样本/headline 定价表；只写 data/processed/phase2_shadow/turnover_p2shadow* '
                  '与 artifacts/phase2/2.4_*；不重跑 M1/PC/网络，不改 headline/configs/正式 reports'),
        'outputs': {'shadow_turnover_csv': out_info,
                    'meta_json': {'file': str(Path(args.meta_out).relative_to(ROOT))},
                    'counts_json': {'file': str(Path(args.counts_out).relative_to(ROOT))},
                    'diff_csv': diff_info, 'matrix_csv': matrix_info},
        'inputs': {
            'turnover_pkl': {'file': str(pkl_path.relative_to(ROOT)),
                             'sha256': meta['source']['sha256'],
                             'shape': meta['source']['shape'],
                             'index_range': meta['source']['index_range'], 'read_only': True},
            'm1_common_sample_ma_teacher_v2': {
                'file': str(Path(args.sample).relative_to(ROOT)),
                'sha256_before': headline_sha_before['m1_common_sample_ma_teacher_v2.parquet'],
                'sha256_after': headline_sha_after['m1_common_sample_ma_teacher_v2.parquet'],
                'rows': int(diag['sample_rows']), 'fm_rows': int(diag['fm_rows']),
                'read_only': True},
            'headline_pricing_table_v2': {'file': str(PRICING_TABLE_V2.relative_to(ROOT)),
                                          'sha256': sha256_file(PRICING_TABLE_V2),
                                          'read_only': True},
        },
        'turnover_diff': {
            'full_pkl_all_months': stats_full,
            'plan_window_2014_01_2024_01': stats_plan_window,
            'fm_window_2015_05_2023_12': stats_fm_window,
            'yearly_file': diff_info['file'],
            'definition': ('corr = Pearson(last, mean)（匹配行，原始 % 口径）；'
                           'mean_abs_diff_pp = |last-mean| 均值（1pp = 1 个百分点）；'
                           'signed = last - mean'),
        },
        'fm_sample_coverage': diag,
        'spec_matrix': matrix,
        'diagnostic_specs': diag_matrix,
        'vs_headline': {'spec_a_matches_pricing_table_v2': cmp_headline,
                        'note': ('规格 a 使用样本自身的 winsorized turnover 列（只读），'
                                 '与 headline 主表逐值一致即证明回归管线未漂移')},
        'audit_l8_replication': audit_l8_block(matrix, cmp_headline),
        'plan_2_0_replication': plan_replication(stats_plan_window, matrix, diag_matrix,
                                                 cmp_headline, diag),
        'headline_anchors_before': anchors_before,
        'headline_anchors_after': anchors_after,
        'headline_anchors_match': anchors_match,
        'headline_files': {
            k: {'sha256_before': headline_sha_before[k], 'sha256_after': headline_sha_after[k],
                'unchanged': headline_sha_before[k] == headline_sha_after[k]}
            for k in HEADLINE_GUARD_FILES},
        'headline_files_modified': not headline_unchanged,
        'm1_rerun': False,
        'configs_modified': False,
        'notes': [
            ('`s3_m1_lib._drive_monthly_wide(pkl, "mean")` 实际落到 `.last()`（该分支只区分 sum/其它），'
             '故 2.0_plan §2.4「复用 _drive_monthly_wide(...,"mean")」不可行；本脚本显式实现月度均值。'),
            ('2.0_plan §2.4「现状」的 mean 行（OwnMAX +0.00583/t=0.30、PeerMAX +0.05377/t=0.74）'
             '= 未 winsor 的 Drive 均值变体（diagnostic b3，逐值复现）；主口径 b（registry [0.01,0.99] '
             '月度 winsor，与 headline `turnover` 施加方式同构）实测 OwnMAX +0.01188/t=0.62、'
             'PeerMAX +0.05694/t=0.80，与审计 L-8 判定表逐值一致。两变体方向相同、均不显著。'),
            ('审计 L-8 的「平均绝对差 0.954pp / 中位差 0.410pp」= M1 v2 面板口径（306,673 个双非空格，'
             '2015-01..2023-12，原始 % 单位、未 winsor）的 mean/median |Δ|，复现 0.9541/0.4101；'
             '审计该行跨口径混用（水平 2.746/2.892 与 corr 0.779 属 pkl 口径；面板水平 2.1457/2.2394、'
             '面板 corr 0.7524）；plan 口径 1.2572/0.4843 亦正确，两者不冲突。'
             '另有面板 winsor 变体（全 pkl 全格子 1,759,812 格，对 last/mean 各自按当月全部格子 skipna 的 '
             '[0.01,0.99] 分位截尾，2000-01..2026-06，双非空 862,230 行）= 1.1753/0.5194，供口径对照。'),
            ('`firm_chars_clean.csv` 的 turnover 为旧文件（2015–2016 覆盖 0），本脚本未使用；'
             '主口径为 Drive 重算。'),
            ('Phase 3 待修（排期）：`s3_m1_lib._drive_monthly_wide` 的 `how != "sum"` 全部落 `.last()`'
             '（含 `how="mean"`）；当前 headline 未受影响（生产调用仅 "last"/"sum"）。'),
        ],
    }
    write_json(Path(args.counts_out), counts)
    print(f'[2.4] DONE：shadow CSV {out_info["rows"]:,} 行 sha={sha[:12]}…；'
          f'counts/diff/matrix 已落盘', flush=True)
    return 0








if __name__ == '__main__':
    sys.exit(main())
