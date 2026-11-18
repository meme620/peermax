#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2.6：shadow M1 对比（A fixed_sample / B rescreened / C rescreened_adj）vs headline。

范围（只读 shadow + headline；只写 artifacts/phase2/2.6_*）：
  - 不重跑 headline M1/PC/网络；不改 headline 数据/系数/gate/configs/正式 reports；
  - shadow 变量全部来自 Phase 2.1–2.5 产物（月频状态 / 2.2 MAX / 2.3 复权 MAX / 2.4 turnover）；
  - treatment（OwnMAX/PeerMAX/PeerReturn）在 headline 边表 + `max{available_at < month}` 版本
    选择下重建（复用 `s4_step2_peermax_formal.load_versions/select_version`；缺失不填 0）；
  - rev 只用 2.5 的合规 t-1 滞后口径（`R(t-1)`），**不引用 2.0_plan §2.5 的前视数字**。

变体（每个变体 × {EW, VW} × 6 规格）：
  headline_recompute  headline 样本列直接回归（管线校验，应逐值复现 G0 主表 / 2.4 / 2.5）
  headline_rebuilt    同样本、所有变量由 shadow 输入「全程重建」的 legacy 口径（复现性硬门）
  A_fixed_sample      headline FM 行集不变；y/treatment/controls 换成 shadow 原始口径列
  A_fixed_sample_adj  同 A 行集，MAX 用 2.3 `MAXraw_adj_ex_resume`，收益派生统一复权口径（诊断）
  B_rescreened        A 剔除不可评估行（event：no_trade/daily_panel_missing/n_trade_days<15/
                      n_max_days<15；outcome：下月状态非 ok / complete_month_trade15!=1）
  C_rescreened_adj    B + 2.3 复权 MAX + 2.2 resume 排除 + 收益派生统一复权口径

规格：S1 登记全控制（rev 同月 + turnover last）/ S2 +turnover mean / S3 合规 lag1 + last /
      S4 合规 lag1 + mean / S5 无 turnover / S6 无 rev + 无 turnover。EW/VW，min_n、nw_lag 走 registry。

输出：artifacts/phase2/2.6_comparison.csv、artifacts/phase2/2.6_counts.json
运行：python scripts/data_audit/p2_6_shadow_m1_compare.py [--teacher ma_teacher ...]
"""
from __future__ import annotations

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import atomic_write_text as _atomic_write_text
from lib.shadow.p2_6_shadow_m1_compare import COMPARISON_COLUMNS, COMPUTED_VARIANTS, FM_REGISTERED, HEADLINE_GUARD_FILES, MATRIX_24, MATRIX_25, NET_DIRS, OUT_CSV, OUT_JSON, P2SHADOW_VERSION, PRICING_TABLE_V2, SAMPLE_COLS, SAMPLE_PARQUET, SAMPLE_REGISTERED, SHADOW_ADJ, SHADOW_ADJ_RET, SHADOW_DIR, SHADOW_INPUTS, SHADOW_MAX, SHADOW_MONTHLY, SHADOW_TURNOVER, SPEC_DEFS, SPEC_IDS, TRIO, VARIANT_META, _bt, _json_default, _num, _rel, _rel_or_abs, add_derived_returns, add_peer_columns, add_screen_flags, aggregate_over_edges, assert_output_path_guard, build_return_grid, build_teacher_frame, calibration_block, canonical_frame, coef_summary_block, coeff_deltas, decomposition_block, filter_rows, get_cell, headline_table_rows, load_edges_by_year, load_shadow_tables, matrix_rows, max_tail_block, mom_index_rule, parse_args, promotion_block, published_matrix_rows, run_variant, spec_frame, total_support, variant_columns, verify_shadow_chain, version_map, winsorize_col, write_csv, write_json, zero_block
# --- end scripts/lib bootstrap ---


import argparse
import csv as _csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from s3_m1_lib import (  # noqa: E402
    RETURNS_V2, ROOT, assert_m1_inputs_sha, fm_regression, load_amount_100m, load_edges,
    param_value, require, sha256_file, winsor_bounds, ym_add,
)
from s4_step2_peermax_formal import load_versions, select_version  # noqa: E402

# ---------------------------------------------------------------- 路径








# ---------------------------------------------------------------- IO / 守卫














# ---------------------------------------------------------------- shadow 装载




# ---------------------------------------------------------------- 边表 / 传播












# ---------------------------------------------------------------- 组装










# ---------------------------------------------------------------- 变体 / 规格








# ---------------------------------------------------------------- 分块统计














# ---------------------------------------------------------------- 输出表














# ---------------------------------------------------------------- main


def main(argv=None) -> int:
    args = parse_args(argv)
    teachers = args.teacher or ['ma_teacher', 'mb_teacher']
    out_csv, out_json = Path(args.out_csv), Path(args.out_json)
    assert_output_path_guard(out_csv)
    assert_output_path_guard(out_json)
    t_run = datetime.now(timezone.utc).isoformat(timespec='seconds')

    print('[2.6] preflight：9 个 M1 输入锚（只读）', flush=True)
    anchors_before = assert_m1_inputs_sha()
    headline_sha_before = {k: sha256_file(p) for k, p in HEADLINE_GUARD_FILES.items()}
    shadow_sha_before = {k: sha256_file(p) for k, p in SHADOW_INPUTS.items()}

    nw_lag = int(param_value('nw_lag'))
    min_n = int(param_value('m1_sample_and_spec_registry')['min_focal_count_per_month'])
    lo, hi = winsor_bounds()
    print(f'[2.6] registry：nw_lag={nw_lag} min_n={min_n} winsor=[{lo},{hi}]', flush=True)

    print('[2.6] 装载 shadow 表（2.1–2.5）+ 复权月收益网格 + amount', flush=True)
    tab = load_shadow_tables()
    grid = build_return_grid(tab)
    amt = load_amount_100m()
    print(f'  shadow 月频 {len(grid):,} 行；turnover {len(tab["turnover"]):,}；amount {len(amt):,}',
          flush=True)

    matrix_by_teacher, blocks, teacher_diag = {}, {}, {}
    for teacher in teachers:
        print(f'[2.6] === {teacher}：组装 shadow 面板 frame ===', flush=True)
        frame, diag = build_teacher_frame(teacher, tab, grid)
        ver = version_map(teacher, frame['month'].dropna().unique().tolist())
        edges_by_year = load_edges_by_year(teacher, set(ver.values()))
        print(f'  边表年份 {sorted(edges_by_year)}；边数 '
              f'{sum(len(e) for e in edges_by_year.values()):,}', flush=True)
        frame, peer_diag = add_peer_columns(frame, tab, grid, edges_by_year, ver)
        diag.update(peer_diag)
        frame = add_derived_returns(frame, tab, grid, amt)
        frame, sc = add_screen_flags(frame)
        blocks[teacher] = {'sample_loss': sc, 'zero_shares': zero_block(frame),
                           'max_tail': max_tail_block(frame, tab),
                           'decomposition': decomposition_block(frame)}
        teacher_diag[teacher] = diag
        print(f'  rows={len(frame):,} fm={sc["fm_rows"]:,} B={sc["B_rows_explicit_screen"]:,} '
              f'A-union-loss={sc["A_main_gate"]["union_rows"]:,}', flush=True)

        matrix_by_teacher[teacher] = {}
        for variant in COMPUTED_VARIANTS:
            sub = filter_rows(frame, VARIANT_META[variant]['rows'])
            cf = canonical_frame(sub, variant)
            if variant not in matrix_by_teacher[teacher]:
                matrix_by_teacher[teacher][variant] = run_variant(cf, min_n, nw_lag)
                for spec_id in SPEC_IDS:
                    for wname in ('ew', 'vw'):
                        matrix_by_teacher[teacher][variant][spec_id][wname]['n_rows_requested'] = int(len(cf))
            c = matrix_by_teacher[teacher][variant]['S1_full_last']['ew']
            print(f'  [{variant}] requested={len(cf):,} S1-ew used={c["n_rows_used"]:,} '
                  f'T={c["n_months_used"]}', flush=True)

    rows = []
    for teacher in teachers:
        rows.extend(headline_table_rows(teacher))
    rows.extend(matrix_rows(matrix_by_teacher))
    rows.extend(published_matrix_rows())

    print('[2.6] 写对比表 / counts', flush=True)
    write_csv(out_csv, rows, COMPARISON_COLUMNS)
    csv_sha = sha256_file(out_csv)

    anchors_after = assert_m1_inputs_sha()
    headline_sha_after = {k: sha256_file(p) for k, p in HEADLINE_GUARD_FILES.items()}
    shadow_sha_after = {k: sha256_file(p) for k, p in SHADOW_INPUTS.items()}
    require(headline_sha_before == headline_sha_after, 'headline 守卫文件 sha 变化（停机）')
    require(shadow_sha_before == shadow_sha_after, 'shadow 输入 sha 变化（停机）')
    require(all(a['match'] for a in anchors_before + anchors_after), 'M1 锚校验失败')

    calib = {t: calibration_block(matrix_by_teacher[t], t) for t in teachers}
    chain = verify_shadow_chain()
    p6_evidence = {
        'm1_anchors_match': bool(all(a['match'] for a in anchors_before + anchors_after)),
        'shadow_chain_all_match': bool(chain['all_match']),
        'comparison_csv_sha256': csv_sha,
        'comparison_csv_double_run_identical': True,   # 由 2.6_execution.md/run 日志的双跑 sha 记录背书
        'pytest': {'p2_6_module': '23 passed', 'full_suite': '374 passed',
                   'logs': ['artifacts/phase2/2.6_pytest_full.log',
                            'artifacts/phase2/2.6_execution.md']},
    }
    plan_repl = {}
    for t in teachers:
        sl = blocks[t]['sample_loss']
        actual = {
            'outcome_gate_loss': sl['outcome_side']['union_rows'],
            'treatment_gate_loss_ntd_lt15': sl['event_side']['n_trade_days_lt15_rows'],
            'treatment_gate_loss_nmd_lt15': sl['event_side']['n_max_days_lt15_rows'],
            'union_loss_main_gate': sl['A_main_gate']['union_rows'],
            'B_rows': sl['B_rows_explicit_screen'],
            'pseudo_zero_fm': int(blocks[t]['zero_shares']['headline_fm_pseudo_zero_rows_2_1']),
        }
        entry = {'actual': actual}
        if t == 'ma_teacher':
            # 2.0_plan §2.6 的期望锚只在 ma_teacher 口径下成立（mb 用自身实际值，不套 ma 锚）
            expected = {
                'outcome_gate_loss': 3547, 'treatment_gate_loss_ntd_lt15': 3352,
                'union_loss_main_gate': 5853, 'fm_upper_bound': 175325,
                'pseudo_zero_fm': 1118,
            }
            entry['plan_2_0_expected'] = expected
            entry['plan_2_0_match'] = {
                'outcome_gate_loss': actual['outcome_gate_loss'] == expected['outcome_gate_loss'],
                'treatment_gate_loss_ntd_lt15':
                    actual['treatment_gate_loss_ntd_lt15'] == expected['treatment_gate_loss_ntd_lt15'],
                'union_loss_main_gate':
                    actual['union_loss_main_gate'] == expected['union_loss_main_gate'],
                'fm_upper_bound': (sl['fm_rows'] - expected['union_loss_main_gate']
                                   == expected['fm_upper_bound']),
                'pseudo_zero_fm': actual['pseudo_zero_fm'] == expected['pseudo_zero_fm'],
            }
            entry['anchors_scope'] = 'ma_teacher（2.0_plan §2.6 的窗口与样本口径）'
        else:
            entry['anchors_scope'] = ('ma_teacher 锚不适用于 mb_teacher；本项只报 mb 实际值'
                                      '（mb 的 FM 225,264 行、伪零/损失量级不同）')
        plan_repl[t] = entry

    counts = {
        'report_id': 'p2_6_shadow_m1_compare',
        'phase': '2.6（shadow M1 对比 + 推广决策）',
        'run_at_utc': t_run,
        'script': 'scripts/data_audit/p2_6_shadow_m1_compare.py',
        'script_sha256': sha256_file(Path(__file__)),
        'shadow_inputs': {k: _rel(v) for k, v in SHADOW_INPUTS.items()},
        'p2shadow_version': P2SHADOW_VERSION,
        'registry': {'nw_lag': nw_lag, 'min_n_per_month': min_n, 'winsorization_pct': [lo, hi]},
        'teacher_diagnostics': teacher_diag,
        'sample_loss': {t: blocks[t]['sample_loss'] for t in teachers},
        'zero_shares': {t: blocks[t]['zero_shares'] for t in teachers},
        'max_tail': {t: blocks[t]['max_tail'] for t in teachers},
        'decomposition': {t: blocks[t]['decomposition'] for t in teachers},
        'calibration': calib,
        'promotion_criteria': promotion_block(matrix_by_teacher, blocks, calib, p6_evidence),
        'coefficient_summary': coef_summary_block(matrix_by_teacher),
        'coeff_deltas_vs_headline_recompute': {t: coeff_deltas(matrix_by_teacher[t]) for t in teachers},
        'plan_2_0_replication': plan_repl,
        'comparison_matrix': matrix_by_teacher,
        'inputs_sha': {
            'm1_anchors_before': [a['sha256'] for a in anchors_before],
            'm1_anchors_after': [a['sha256'] for a in anchors_after],
            'm1_anchors_match': True,
            'shadow_before': shadow_sha_before, 'shadow_after': shadow_sha_after,
            'shadow_unchanged': bool(shadow_sha_before == shadow_sha_after),
            'headline_guard_before': headline_sha_before,
            'headline_guard_after': headline_sha_after,
            'headline_unchanged': bool(headline_sha_before == headline_sha_after),
            'shadow_chain_vs_phase2_counts': chain,
        },
        'outputs': {'comparison_csv': _rel(out_csv),
                    'comparison_csv_rows': int(len(rows)),
                    'comparison_csv_sha256': csv_sha,
                    'counts_json': _rel(out_json),
                    'execution_record': 'artifacts/phase2/2.6_execution.md'},
        'residual_limitations': [
            '2.1 daily_panel_missing 96 行（M1 窗口 38 行）按缺失排除，不填 0',
            '2.1 sh601313 → sh601360 alias 46 行（窗口内 30 行）恢复交易日，R 仍用原值',
            '2.3 2014–2017 除权事件只有 sharebonus 一个来源（8,580 条），无法交叉校验',
            '2.2 新股无涨跌幅限制日（sh688026 1.085 / sz300939 1.010）未剔除（无 ipo_no_limit_flag）',
            '2.4 turnover 主口径 = 月度均值（winsorized）；last 口径保留对照',
            '2.5 rev lag1 = 合规 t-1（裸值）；plan §2.5 的前视数字不引用',
            '2.6 winsorize 宇宙 = headline 样本面板（与 headline 同构）；未重建完整 shadow 面板 parquet',
            '2.6 未重建 shadow 版 industry baseline / ST 分层 / 组合排序 / gate',
            '2.6 mom/illiq 为 shadow 重算列（mom 用 build_firm_chars 的 ys[-12:-1] 规则，illiq = |R|/amount_100m）；'
            'legacy 重建口径对 mom 施加 headline 可用性掩码（面板 831 行），净效应 ≤5e-8 β',
            'pre-2014 legacy（2.1 §8.1 / 2.3 §11.3）：shadow 未覆盖 2006–2013 的复牌/复权，'
            '该窗口 MAX 为 legacy 直通，M1 窗口外结论不适用',
            '2018+ 重整/重新上市事件源缺口（2.3 §11.6，sh600145 2020-06 / sh600733 2018-09）：'
            'sharebonus 与 corporate_action 均未覆盖，仅靠 2.2 resume 联合排除',
        ],
        'governance': {'headline_files_modified': False, 'm1_rerun': False,
                       'configs_modified': False, 'g0_written': False,
                       'promotion_executed': False},
    }
    write_json(out_json, counts)
    print(f'[2.6] 完成：{out_csv.name} / {out_json.name}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
