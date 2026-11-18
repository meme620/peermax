#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.5（审计 L-3）shadow：rev 同月 / 滞后(t-1) / 移除 控制集矩阵。

背景（`artifacts/phase2/2.0_plan.md` §2.5；`reports/audits/peermax_audit_item_review_20261027.md` L-3）：
  - headline 控制集含 `rev` = **事件月当月收益** `R_t`（`build_firm_chars.py:108-111`），与
    `OwnMAXraw` 同月同源（月度截面 corr 均值 0.425），是第二条疑似机械吸收 MAX 的控制；
  - `turnover`（月末值）已在 2.4 验证；本脚本在**只读** M1 headline 样本上跑
    rev 三口径（same / lag1 / removed）× turnover 三口径（headline last / Drive 均值 / removed）
    的 FM 矩阵（EW/VW），其余控制、outcome、估计器与 headline 一致。

rev 口径（重要）：
  - `rev_same` = 样本自带 `rev` 列（headline winsorized `R_t`，只读；与
    `winsorize_by_month(v2.R, panel 宇宙)` 逐值一致，脚本硬门校验）；
  - `rev_lag1` = **合规滞后**：`monthly_returns_clean_v2.R` 的 `t-1` 月值（裸值）。构造硬门
    要求来源月严格小于当前月（无前视/无同月），缺失保持 NaN 不填 0；
  - `rev_lag1_win` = 敏感性：裸 lag1 在**当前月面板宇宙**按 registry [0.01,0.99] winsor
    （与 headline `rev` 的月度截面 winsor 同构；样本行集合不变）。次变体（诊断）用样本
    winsorized `rev` 沿 `month-1` 平移（源月 winsor 值，FM 少 1,428 行）。

**关键更正（2.0_plan §2.5「现状」引用的 rev_lag1 数字是前视构造）**：
  2.0_plan 引用的 `rev_lag1` 三组数字（corr -0.0343；规格 OwnMAX -0.01341/t=-4.02、
  PeerMAX +0.00927/t=1.21；+turnover_mean 的 -0.01728/t=-4.91），逐值可由 **R_{t+1}**
  （下一自然月收益，即 outcome `R_next` 的源）复现；该列在规格里的自身系数 = **+0.895
  (t=+107.5)** ≈ 1，且 corr(R_next, winsor(R_{t+1}))=0.985 —— 即 2.0_plan 的
  `rev_lag1` 实际是**被解释变量本身**（前视）。合规的 t-1 滞后数字完全不同（见 `b`/`d`）。
  本脚本把前视变体仅作 provenance 诊断（`plan_rev_lag1_lookahead_diagnostic`），
  **不进入规格矩阵/CSV**，并在 counts 中给出逐项复现证据。

产出（只写 artifacts；不写 data/、不写 G0）：
  - `artifacts/phase2/2.5_counts.json`（计数、相关、13 规格系数表、VIF、headline/2.4 对比、
    9 锚前后、前视诊断）
  - `artifacts/phase2/2.5_control_matrix.csv`（规格 × EW/VW × 项 的 beta/t/T/n）

用法：
  python scripts/data_audit/p2_5_rev_shadow.py
"""
from __future__ import annotations

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import atomic_write_text as _atomic_write_text
from lib.shadow.p2_5_rev_shadow import CHARS, CONTROLS, COUNTS_24, COUNTS_JSON, FM_WINDOW_END, FM_WINDOW_START, FORWARD_DIAG_SPEC_DEFS, HEADLINE_GUARD_FILES, MATRIX_24, MATRIX_CSV, P2SHADOW_VERSION, PRICING_TABLE_V2, RETURNS_V1, REV_LAG1, REV_LAG1_FWD, REV_LAG1_WIN, REV_SAME, SAMPLE_PARQUET, SHADOW_TURNOVER_CSV, SHADOW_TURNOVER_META, SPEC_DEFS, TRIO, TURN_LAST, TURN_LAST_RAW, TURN_MEAN_NOWIN, TURN_MEAN_RAW, TURN_MEAN_WIN, _bt, _lag_merge, _near, absorption_block, anchor_block, assert_output_path_guard, attach_forward_diagnostic, attach_lagged_column, build_control_frame, build_xcols, compare_2_4, headline_comparison, matrix_rows, monthly_corr_stats, monthly_vif, parse_args, plan_lookahead_diagnostic, run_spec_matrix, shift_month_series, spec_xcols, write_json, write_matrix_csv
# --- end scripts/lib bootstrap ---


import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s3_m1_lib import (  # noqa: E402
    RETURNS_V2, ROOT, assert_m1_inputs_sha, fm_regression, param_value, require,
    sha256_file, winsor_bounds, winsorize_by_month,
)



# headline 关键文件（本脚本必须不写、且 sha 前后不变）





# ---------------------------------------------------------------- 基础 IO / 月键












# ---------------------------------------------------------------- 相关 / VIF




# ---------------------------------------------------------------- 规格矩阵



# 前视诊断规格（provenance，只为复现 2.0_plan 引用数字；不写矩阵 CSV、不得被引用为结论）




















# ---------------------------------------------------------------- 输出






def main(argv=None) -> int:
    args = parse_args(argv)
    t_generated = datetime.now(timezone.utc).isoformat(timespec='seconds')
    print('[2.5] preflight：9 个 M1 输入锚（只读）', flush=True)
    anchors_before = assert_m1_inputs_sha()
    headline_sha_before = {k: sha256_file(p) for k, p in HEADLINE_GUARD_FILES.items()}
    extra_inputs = {
        'monthly_returns_clean.csv': RETURNS_V1,
        'shadow_turnover_2_4_csv': Path(args.turnover_shadow),
        'shadow_turnover_2_4_meta': SHADOW_TURNOVER_META,
        'headline_pricing_table_v2': PRICING_TABLE_V2,
        'matrix_2_4_csv': MATRIX_24,
        'counts_2_4_json': COUNTS_24,
    }
    extra_sha_before = {k: (sha256_file(p) if p.is_file() else None)
                        for k, p in extra_inputs.items()}

    print('[2.5] 构造控制矩阵（样本只读 + v2 收益 lag + 2.4 turnover shadow）', flush=True)
    fm, diag = build_control_frame(Path(args.sample), Path(args.returns),
                                   Path(args.turnover_shadow))
    print(f"  panel={diag['sample_rows']:,} FM={diag['fm_rows']:,} "
          f"合规 lag1 FM 覆盖={diag['rev_lag1']['fm_lag_coverage']:.4f} "
          f"rev 同月桥 max|d|={diag['rev_same_panel_bridge']['max_abs_diff_vs_sample_rev']:.2e}",
          flush=True)
    require(diag['fm_rows'] == 181178, f"FM 行数 {diag['fm_rows']} != 181178（停机条件）")
    require(diag['fm_months'] == 104, f"FM 月数 {diag['fm_months']} != 104（停机条件）")
    require(diag['rev_lag1']['fm_lag_coverage'] >= 0.999,
            f"FM rev_lag1 覆盖 {diag['rev_lag1']['fm_lag_coverage']:.6f} < 0.999")
    require(diag['turnover']['fm_mean_win_coverage'] >= 0.99,
            'FM turnover_mean winsor 覆盖 <99%（停机条件）')
    require(diag['turnover']['last_bridge_max_abs_diff_vs_sample_col'] <= 1e-12,
            'panel-winsor(last)/100 与样本 turnover 列不一致（不得重算 last）')
    require((fm['month'] >= FM_WINDOW_START).all() and (fm['month'] <= FM_WINDOW_END).all(),
            'FM 窗口越界')

    print('[2.5] 相关（逐月截面 + pooled）', flush=True)
    corr = {
        'rev_own': monthly_corr_stats(fm, REV_SAME, 'OwnMAXraw'),
        'lag1_own': monthly_corr_stats(fm, REV_LAG1, 'OwnMAXraw'),
        'rev_lag1': monthly_corr_stats(fm, REV_SAME, REV_LAG1),
        'rev_peer': monthly_corr_stats(fm, REV_SAME, 'PeerMAXraw'),
        'lag1_peer': monthly_corr_stats(fm, REV_LAG1, 'PeerMAXraw'),
        'lag1win_own': monthly_corr_stats(fm, REV_LAG1_WIN, 'OwnMAXraw'),
        'lag1fwd_own': monthly_corr_stats(fm, REV_LAG1_FWD, 'OwnMAXraw'),
    }
    for k in ('rev_own', 'lag1_own', 'rev_lag1', 'lag1fwd_own'):
        print(f"  corr {k}: monthly_mean={corr[k]['monthly_mean']:+.4f} "
              f"median={corr[k]['monthly_median']:+.4f} pooled={corr[k]['pooled_corr']:+.4f}",
              flush=True)
    require(0.40 <= corr['rev_own']['monthly_mean'] <= 0.46, 'corr(rev, OwnMAX) 偏离 0.42–0.43')
    require(0.08 <= corr['lag1_own']['monthly_mean'] <= 0.16,
            f"合规 corr(rev_lag1, OwnMAX)={corr['lag1_own']['monthly_mean']:.4f} 偏离 0.08–0.16")

    print('[2.5] 13 规格 × EW/VW FM 矩阵（min_n=100，NW lag=6）', flush=True)
    matrix = run_spec_matrix(fm)
    for sid, blk in matrix['specs'].items():
        e = blk['ew']['coefs']
        print(f"  {sid}: EW PeerMAX {e['PeerMAXraw']['beta']:+.5f} "
              f"(t={e['PeerMAXraw']['t_nw']:+.3f}) | OwnMAX {e['OwnMAXraw']['beta']:+.5f} "
              f"(t={e['OwnMAXraw']['t_nw']:+.3f}) | T={blk['ew']['T']} "
              f"n={blk['ew']['n_rows_used']:,}", flush=True)
    for sid, blk in matrix['specs'].items():
        require(blk['ew']['T'] == 104 and blk['vw']['T'] == 104, f'{sid}: T != 104（停机条件）')
        require(blk['ew']['n_rows_used'] == 181178 and blk['vw']['n_rows_used'] == 181178,
                f'{sid}: n != 181178（停机条件）')
        require(blk['rev_variant'] != REV_LAG1_FWD and REV_LAG1_FWD not in blk['x'],
                f'{sid}: 前视列进入规格矩阵（停机条件）')

    cmp_headline = headline_comparison(matrix)
    require(cmp_headline['all_match'], '规格 a 未能复现 headline 主表（停机条件）')
    cmp_24 = compare_2_4(matrix)
    require(cmp_24.get('all_match', True), '2.5 与 2.4 同控制集规格系数不一致（停机条件）')

    print('[2.5] 前视诊断：复现 2.0_plan §2.5 引用的 rev_lag1 数字（不入矩阵 CSV）', flush=True)
    fwd_matrix = run_spec_matrix(fm, FORWARD_DIAG_SPEC_DEFS)
    fwd_diag = plan_lookahead_diagnostic(fm, fwd_matrix, corr['lag1fwd_own'],
                                         diag['outcome_identity_evidence'])
    print(f"  前视变体对 plan 引用逐值复现 all_match={fwd_diag['all_items_match']}；"
          f"合规 lag1 corr={corr['lag1_own']['monthly_mean']:+.4f} vs plan 引 -0.0343",
          flush=True)
    require(fwd_diag['all_items_match'],
            '未能复现 plan §2.5 引用的 rev_lag1 数字（前视诊断失败，需人工复核）')

    print('[2.5] VIF（R6：rev_lag1 与 mom 共线检查，EW 截面）', flush=True)

    def _vif_block(sid):
        x = spec_xcols(SPEC_DEFS[sid])
        return {'x': x, 'vif': monthly_vif(fm, x)}
    vif = {
        'spec_a_controls': _vif_block('a_registered_rev_turnover'),
        'spec_b_controls': _vif_block('b_rev_lag1_turnover_last'),
        'spec_d_controls': _vif_block('d_rev_lag1_no_turnover'),
        'corr_lag1_mom_monthly_mean': monthly_corr_stats(fm, REV_LAG1, 'mom')['monthly_mean'],
    }
    hints = vif['spec_b_controls']['vif']
    print(f"  spec b VIF: rev_lag1 mean={hints[REV_LAG1]['mean']:.3f} | mom mean={hints['mom']['mean']:.3f} "
          f"| turnover mean={hints['turnover']['mean']:.3f} | "
          f"corr(lag1, mom)={vif['corr_lag1_mom_monthly_mean']:+.4f}", flush=True)

    anchors = anchor_block(matrix, corr)
    require(anchors['all_items_match'], '合规锚点对账失败（停机条件）')
    out_info = write_matrix_csv(matrix, Path(args.matrix_out))

    print('[2.5] postflight：9 个 M1 输入锚与 headline 文件 sha 复核', flush=True)
    anchors_after = assert_m1_inputs_sha()
    headline_sha_after = {k: sha256_file(p) for k, p in HEADLINE_GUARD_FILES.items()}
    extra_sha_after = {k: (sha256_file(p) if p.is_file() else None)
                       for k, p in extra_inputs.items()}
    anchors_match = bool([a['sha256'] for a in anchors_before]
                         == [a['sha256'] for a in anchors_after])
    headline_unchanged = headline_sha_before == headline_sha_after
    extra_unchanged = extra_sha_before == extra_sha_after
    require(anchors_match, '9 锚前后 sha 不一致（停机条件）')
    require(headline_unchanged, 'headline 文件 sha 发生变化（停机条件）')
    require(extra_unchanged, '只读输入文件 sha 发生变化（停机条件）')

    counts = {
        'p2shadow_version': P2SHADOW_VERSION,
        'generated_at_utc': t_generated,
        'script': 'scripts/data_audit/p2_5_rev_shadow.py',
        'script_sha256': sha256_file(Path(__file__)),
        'scope': ('只读样本/月收益/2.4 shadow turnover/headline 定价表；只写 artifacts/phase2/2.5_counts.json '
                  '与 2.5_control_matrix.csv；不重跑 M1/PC/网络，不写 data/、G0、configs、正式 reports'),
        'outputs': {
            'counts_json': {'file': str(Path(args.counts_out).relative_to(ROOT))},
            'matrix_csv': out_info,
        },
        'inputs': {
            'm1_common_sample_ma_teacher_v2': {
                'file': str(SAMPLE_PARQUET.relative_to(ROOT)),
                'sha256_before': headline_sha_before['m1_common_sample_ma_teacher_v2.parquet'],
                'sha256_after': headline_sha_after['m1_common_sample_ma_teacher_v2.parquet'],
                'rows': int(diag['sample_rows']), 'fm_rows': int(diag['fm_rows']),
                'fm_months': int(diag['fm_months']), 'read_only': True},
            'monthly_returns_clean_v2': {
                'file': str(RETURNS_V2.relative_to(ROOT)),
                'sha256_before': headline_sha_before['monthly_returns_clean_v2.csv'],
                'sha256_after': headline_sha_after['monthly_returns_clean_v2.csv'],
                'read_only': True},
            'monthly_returns_clean': {'file': str(RETURNS_V1.relative_to(ROOT)),
                                      'sha256': extra_sha_before['monthly_returns_clean.csv'],
                                      'read_only': True},
            'firm_chars_clean': {'file': str(CHARS.relative_to(ROOT)),
                                 'sha256_before': headline_sha_before['firm_chars_clean.csv'],
                                 'sha256_after': headline_sha_after['firm_chars_clean.csv'],
                                 'read_only': True},
            'shadow_turnover_2_4_csv': {
                'file': str(Path(args.turnover_shadow).relative_to(ROOT)),
                'sha256': extra_sha_before['shadow_turnover_2_4_csv'],
                'read_only': True},
            'headline_pricing_table_v2': {
                'file': str(PRICING_TABLE_V2.relative_to(ROOT)),
                'sha256': extra_sha_before['headline_pricing_table_v2'], 'read_only': True},
            'registry_params': {'winsorization_pct': list(diag['winsorization_pct']),
                                'nw_lag': matrix['nw_lag'], 'min_n': matrix['min_n']},
        },
        'rev_construction': {
            'rev_same': '样本 rev 列（headline winsorized R_t，只读）；面板桥 max|d|=0',
            'rev_lag1': diag['rev_lag1'],
            'rev_lag1_win': diag['rev_lag1_win'],
            'rev_lag1_win_src_diagnostic': diag['rev_lag1_win_src_diagnostic'],
            'rev_lag1_fwd_lookahead': diag['rev_lag1_fwd_lookahead'],
            'v1_vs_v2_and_chars_consistency': ('月度收益 v1/v2 与 firm_chars_clean.rev 在样本键上同值'
                                               '（≤5e-16，只读复核）；样本 rev = winsor(v2.R) 逐值一致'),
        },
        'turnover_construction': diag['turnover'],
        'correlations': {
            'definition': ('逐月截面 Pearson corr（双方非空、月内 n>=30 且两列有方差），'
                           '汇总 mean/median；pooled 为 FM 全样本相关；per_month 为逐月明细'),
            'rev_own': corr['rev_own'], 'lag1_own': corr['lag1_own'], 'rev_lag1': corr['rev_lag1'],
            'rev_peer': corr['rev_peer'], 'lag1_peer': corr['lag1_peer'],
            'lag1win_own': corr['lag1win_own'], 'lag1fwd_own': corr['lag1fwd_own'],
        },
        'spec_matrix': matrix,
        'vif': vif,
        'vs_headline': {
            'spec_a_matches_pricing_table_v2': cmp_headline,
            'spec_2_5_vs_2_4_matrix': cmp_24,
            'absorption_decomposition': absorption_block(matrix),
            'note': ('规格 a 使用样本自身的 winsorized rev/turnover 列（只读），与 headline 主表逐值一致'
                     '即证明回归管线未漂移；与 2.4 同控制集规格逐值一致（≤1e-8）'),
        },
        'compliant_anchor_replication': anchors,
        'plan_rev_lag1_lookahead_diagnostic': fwd_diag,
        'headline_anchors_before': anchors_before,
        'headline_anchors_after': anchors_after,
        'headline_anchors_match': anchors_match,
        'headline_files': {
            k: {'sha256_before': headline_sha_before[k], 'sha256_after': headline_sha_after[k],
                'unchanged': headline_sha_before[k] == headline_sha_after[k]}
            for k in HEADLINE_GUARD_FILES},
        'extra_input_files': {
            k: {'sha256_before': extra_sha_before[k], 'sha256_after': extra_sha_after[k],
                'unchanged': extra_sha_before[k] == extra_sha_after[k]}
            for k in extra_inputs},
        'headline_files_modified': not headline_unchanged,
        'm1_rerun': False,
        'configs_modified': False,
        'notes': [
            ('**关键更正**：2.0_plan §2.5「现状」与 §2 摘要引用的 rev_lag1 三组数字'
             '（corr -0.0343；-0.01341/t=-4.02；-0.01728/t=-4.91）全部由 **R_{t+1}**（下一自然月收益，'
             '≈ outcome R_next）逐值复现；该列在规格内自身系数 +0.895（t=+107.5）。即 plan 的「rev_lag1」'
             '实际是前视/被解释变量，命中 plan 自己的停机条件。合规 t-1 滞后下：corr(OwnMAX) 月度均值 '
             '+0.1192；规格 b（+turnover）OwnMAX -0.01512（t=-0.71）、PeerMAX +0.04391（t=0.59）；'
             '规格 d（无 turnover）OwnMAX -0.13353（t=-4.91）。2.6 memo 不得引用 plan 的 rev_lag1 数字。'),
            ('吸收分解（EW，对比「rev 与 turnover 皆移除」= -0.13586）：turnover 单独移除后 '
             'OwnMAX -0.01133（吸收 ≈91.7%）；同月 rev 单独移除后 -0.10550（≈22.3%）；合规 lag1 单独移除后 '
             '-0.13353（≈1.7%）。含 turnover 时 rev 的额外边际很小（c→b→a：-0.0113→-0.0151→-0.0028），'
             '即 **turnover 是主吸收源，同月 rev 是次级**，滞后 rev 不吸收。'),
            ('规格 f 与 e 控制集完全相同（任务矩阵要求的参考行 = 审计 no_rev_no_turnover 锚点），'
             '系数逐值一致，保留两行以对齐任务矩阵与审计锚点。'),
            ('VIF：rev_lag1 与 mom 的月度 corr 接近 0（mom 用 t-12..t-2，lag1 只用 t-1，月份集合不重叠），'
             'R6 共线风险实测不成立；VIF 明细见 vif 块（EW 口径）。'),
            ('前视护栏：合规 lag1 来源月 = 当前月 -1（构造硬门）；无 fillna(0)；FM 内 lag1 覆盖 100%'
             '（panel 904 行因该月无收益记录自然留空）。前视诊断列仅存在于 counts 的 provenance 块，'
             '脚本内置硬门禁止其进入任何规格（含 CSV）。'),
            ('2.5 不改控制集、不改 headline；「同月 rev/turnover 吸收 OwnMAX」不等于「MAX 折价不存在」，'
             '结论只能写：同月 rev/turnover 吸收后 OwnMAX 接近 0，PeerMAX 在所有规格仍未检出。'),
        ],
    }
    write_json(Path(args.counts_out), counts)
    print(f'[2.5] DONE：13 规格矩阵落盘 {out_info["rows"]} 行 '
          f'sha={out_info["sha256"][:12]}…；锚点 all_match={anchors["all_items_match"]}；'
          f'plan 前视复现 all_match={fwd_diag["all_items_match"]}', flush=True)
    return 0




if __name__ == '__main__':
    sys.exit(main())
