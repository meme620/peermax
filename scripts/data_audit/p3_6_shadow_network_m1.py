#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3.6：shadow 网络 M1 对比 + 推广决策（默认 HOLD）。

范围（3.0_plan §2.6 + 3.6 任务书）：
  - 只读 headline 样本 / headline 网络 / 5 个 shadow 网络 / s3_m1_lib；
  - 在**同一固定样本**（headline FM 181,178 行、104 月）上重算 shadow 网络 treatment；
  - 逐变体 × 规格（S1 登记全控制 / S5 无 turnover / S6 无 rev+无 turnover）× EW/VW 跑 FM；
  - 网络层读数：边 Jaccard、边权相关、treatment 相关、覆盖/样本损失；
  - 只写 `artifacts/phase3/3.6_*` 与 `data/processed/phase3_shadow/p3_6_shadow_treat_*.parquet`。

不重跑 headline M1/PC/网络，不写 G0/reports/configs，不自动推广。

用法：
  python scripts/data_audit/p3_6_shadow_network_m1.py
"""
from __future__ import annotations
# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.shadow.p3_6_shadow_network_m1 import ALL_IDS, ANCHOR_POST, ANCHOR_PRE, ART, COMPARE_3_2, COMPARISON_CSV, CONTROLS, COUNTS_JSON, FM_MONTHS, FM_ROWS, FM_VARIANTS, HEADLINE_EW_BETA, HEADLINE_EW_T, HEADLINE_NET, HEADLINE_VW_BETA, HEADLINE_VW_T, LIMIT_UNKNOWN, MIN_N, NW_LAG, OUTPUT_SHA, Q4_JACCARD_MIN, ROOT, RUN_LOG, S1, S5, S6, SAMPLE_PATH, SAMPLE_ROWS, SHADOW_DATA, SHADOW_IDS, SPECS, TEXT_YEAR_MONTHS, VARIANTS, WINSOR, YEARS, _TEE, _json_default, _load_variant_versions, _net_files, _sha_map, _summarise_fm, _year_contract, aggregate_network, assert_output_path_guard, attach_treatment, check_sample_mapping, compare_anchors, coverage_stats, enrich_fm_rows, evaluate_q_status, load_values, log, map_months, pooled_network_metrics, run_fm_matrix, snapshot_anchors, treatment_correlations, write_csv, write_json, write_parquet
# --- end scripts/lib bootstrap ---


import argparse
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
    assert_m1_inputs_sha, fm_regression, param_value, require, sha256_file,
    winsorize_by_month,
)
from s4_step2_peermax_formal import (  # noqa: E402
    NETWORK_AVAILABLE_AT_RULE, load_versions, select_version,
)







# ------------------------------------------------------------------ 输出路径守卫


class _Tee:
    def __init__(self, path: Path):
        path = assert_output_path_guard(path)
        self.fh = open(path, 'a', encoding='utf-8', newline='\n')

    def write(self, s: str):
        sys.__stdout__.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        sys.__stdout__.flush()
        self.fh.flush()














# ------------------------------------------------------------------ 锚 / sha 台账








# ------------------------------------------------------------------ 网络装载 / 月份映射






# ------------------------------------------------------------------ 聚合








# ------------------------------------------------------------------ 固定样本 treatment




# ------------------------------------------------------------------ FM






# ------------------------------------------------------------------ 相关矩阵


# ------------------------------------------------------------------ Q 判据


# ------------------------------------------------------------------ 主流程
def main() -> int:
    global _TEE
    ap = argparse.ArgumentParser(description='Phase 3.6 shadow 网络 M1 对比')
    ap.add_argument('--skip-anchor-pre', action='store_true',
                    help='调试用：跳过 pre 锚快照（正式运行不要用）')
    args = ap.parse_args()
    t0 = time.time()
    _TEE = _Tee(RUN_LOG)
    log('Phase 3.6 shadow 网络 M1 对比开始')
    log(f'registry: min_n={MIN_N} nw_lag={NW_LAG} winsor={WINSOR}')

    pre = None if args.skip_anchor_pre else snapshot_anchors('pre')
    if pre is not None:
        write_json(ANCHOR_PRE, pre)

    sample = pd.read_parquet(SAMPLE_PATH)
    require(len(sample) == SAMPLE_ROWS, f'headline 样本行数 {len(sample)} != {SAMPLE_ROWS}')
    require(int(sample['m1_in_fm_sample'].sum()) == FM_ROWS,
            f'FM 行数 {int(sample["m1_in_fm_sample"].sum())} != {FM_ROWS}')
    require(sample.duplicated(['focal_id', 'month']).sum() == 0, 'headline 样本键重复')
    fm_months = sorted(sample.loc[sample['m1_in_fm_sample'].astype(bool), 'month'].unique())
    require(len(fm_months) == FM_MONTHS, f'FM 月数 {len(fm_months)} != {FM_MONTHS}')
    log(f'headline 样本 {len(sample):,} 行；FM {FM_ROWS:,} 行 / {FM_MONTHS} 月')

    srcm = pd.read_csv(ROOT / 'data/processed/source_month_argmax_panel.csv',
                       usecols=['source_id', 'month', 'MAXraw', 'qualification_status'],
                       dtype={'source_id': str, 'month': str})
    rets = pd.read_csv(ROOT / 'data/processed/monthly_returns_clean_v2.csv',
                       usecols=['firm_id', 'month', 'R'],
                       dtype={'firm_id': str, 'month': str}).dropna(subset=['R'])
    maxv, retv = load_values(srcm, rets)
    log(f'值表：MAX {len(maxv):,} 行（qual != {LIMIT_UNKNOWN}）/ 月收益 {len(retv):,} 行')

    vinfo = {v['id']: _load_variant_versions(v) for v in VARIANTS}
    headline_vers = vinfo['headline']['vers']
    mapping = map_months(headline_vers, fm_months)
    for vid in SHADOW_IDS:
        require(map_months(vinfo[vid]['vers'], fm_months) == mapping,
                f'{vid}: 月份→文本年映射与 headline 不一致')
    map_check = check_sample_mapping(sample, mapping)
    require(map_check['n_mismatch'] == 0, f"样本 network_version 映射不一致: {map_check}")
    months_by_ty = {ty: [] for ty in YEARS}
    for m, ty in mapping.items():
        months_by_ty[ty].append(m)
    for ty in months_by_ty:
        months_by_ty[ty] = sorted(months_by_ty[ty])
    log(f'月份映射：{len(mapping)} 月 → 文本年 {sorted(set(mapping.values()))}；'
        f'逐年月数 {[len(months_by_ty[t]) for t in YEARS]}')

    # headline 元数据（n_nodes）
    headline_meta = {}
    for ty, rec in headline_vers.items():
        doc = json.loads((HEADLINE_NET / f'{ty}_build_report.json').read_text(encoding='utf-8'))
        headline_meta[str(ty)] = {'n_nodes': int(doc['n_nodes']),
                                  'n_edges': int(doc['n_edges'])}

    treatments, net_metrics, contracts = {}, {}, {}
    for v in VARIANTS:
        vid = v['id']
        log(f'--- 聚合 {vid} ---')
        raw, metrics, chk = aggregate_network(v, vinfo[vid]['vers'], months_by_ty,
                                              maxv, retv, headline_meta)
        net_metrics[vid] = pooled_network_metrics(metrics)
        contracts[vid] = chk
        treat = attach_treatment(sample, raw)
        treatments[vid] = treat
        cov = coverage_stats(sample, treat)
        net_metrics[vid]['coverage'] = cov
        if vid != 'headline':
            fm_out = sample.loc[sample['m1_in_fm_sample'].astype(bool),
                                ['focal_id', 'month']].copy()
            fm_out = fm_out.merge(
                treat[['focal_id', 'month', 'PeerMAXraw', 'PeerReturn', 'n_sources',
                       'coverage_ratio_max', 'coverage_ratio_ret']],
                on=['focal_id', 'month'], how='left')
            fm_out.insert(0, 'variant', vid)
            fm_out['peermax_missing'] = fm_out['PeerMAXraw'].isna()
            fm_out['peerreturn_missing'] = fm_out['PeerReturn'].isna()
            write_parquet(SHADOW_DATA / f'p3_6_shadow_treat_{vid}.parquet', fm_out)
        log(f"{vid} 覆盖：fixed PM {cov['fixed_peermax_nonnull']:,} / FM PM "
            f"{cov['fm_peermax_nonnull']:,}（缺 {cov['fm_peermax_missing']:,}）；"
            f"Jaccard 中位 {net_metrics[vid]['jaccard_median_yearly']:.4f}")

    # headline 复算保真门（同一聚合器）
    ht = treatments['headline']
    for col, tol in (('PeerMAXraw', 1e-7), ('PeerReturn', 1e-7),
                     ('coverage_ratio_max', 2e-6), ('coverage_ratio_ret', 2e-6)):
        d = float((ht[col] - sample[col]).abs().max())
        require(d <= tol, f'headline 复算 {col} max|Δ|={d} > {tol}')
        log(f'headline 复算 {col}: max|Δ|={d:.3e}（tol {tol:g}）')
    ns_diff = int((ht['n_sources'].to_numpy() != sample['n_sources'].to_numpy()).sum())
    require(ns_diff == 0, f'headline 复算 n_sources 不一致 {ns_diff} 行')
    mask_diff = int((ht['PeerMAXraw'].notna().to_numpy()
                     != sample['PeerMAXraw'].notna().to_numpy()).sum())
    require(mask_diff == 0, f'headline 复算 PeerMAX 缺失模式不一致 {mask_diff} 行')
    recompute_validation = {
        'peer_max_max_abs_diff': float((ht['PeerMAXraw'] - sample['PeerMAXraw']).abs().max()),
        'peer_return_max_abs_diff': float((ht['PeerReturn'] - sample['PeerReturn']).abs().max()),
        'coverage_ratio_max_max_abs_diff': float(
            (ht['coverage_ratio_max'] - sample['coverage_ratio_max']).abs().max()),
        'coverage_ratio_ret_max_abs_diff': float(
            (ht['coverage_ratio_ret'] - sample['coverage_ratio_ret']).abs().max()),
        'n_sources_mismatch_rows': ns_diff,
        'peermax_missing_pattern_mismatch_rows': mask_diff,
        'tolerances': {'peermax': 1e-7, 'peerreturn': 1e-7,
                       'coverage': 1e-6},
        'status': 'PASS',
    }

    # 相关
    corr = treatment_correlations(sample, treatments)
    log('treatment 相关矩阵完成')

    # FM
    rows = run_fm_matrix(sample, treatments)
    fm_df = enrich_fm_rows(rows)
    # 附加网络/覆盖列
    for vid in FM_VARIANTS:
        vid_id = vid['id']
        meta_id = 'headline' if vid_id == 'headline_recompute' else vid_id
        nm = net_metrics[meta_id]
        cov = nm['coverage']
        mask = fm_df['variant'] == vid_id
        fm_df.loc[mask, 'edge_jaccard_median'] = nm['jaccard_median_yearly']
        fm_df.loc[mask, 'edge_jaccard_min'] = nm['jaccard_min_yearly']
        fm_df.loc[mask, 'edge_jaccard_pooled'] = nm['jaccard_pooled']
        fm_df.loc[mask, 'edge_weight_corr_pooled'] = nm['edge_weight_corr_pooled']
        fm_df.loc[mask, 'treatment_corr_peermax_vs_headline'] = (
            corr['PeerMAXraw']['corr'][meta_id]['headline'])
        fm_df.loc[mask, 'treatment_corr_peerreturn_vs_headline'] = (
            corr['PeerReturn']['corr'][meta_id]['headline'])
        fm_df.loc[mask, 'fixed_peermax_nonnull'] = cov['fixed_peermax_nonnull']
        fm_df.loc[mask, 'fm_peermax_nonnull'] = cov['fm_peermax_nonnull']
        fm_df.loc[mask, 'fm_peermax_lost_vs_headline'] = cov['fm_peermax_lost_vs_headline']
        fm_df.loc[mask, 'fm_peerreturn_lost_vs_headline'] = cov['fm_peerreturn_lost_vs_headline']
        fm_df.loc[mask, 'n_sources_median'] = cov['fm_n_sources_median']
        fm_df.loc[mask, 'coverage_ratio_max_median'] = cov['fm_coverage_ratio_max_median']
        fm_df.loc[mask, 'coverage_ratio_ret_median'] = cov['fm_coverage_ratio_ret_median']
    # headline_recompute 对照行
    rec = fm_df[fm_df['variant'] == 'headline'][['spec', 'weight', 'beta_peermax',
                                                  't_peermax', 'n', 'T']]
    rec = rec.rename(columns={'beta_peermax': 'beta_recompute',
                              't_peermax': 't_recompute',
                              'n': 'n_recompute', 'T': 'T_recompute'})
    fm_df = fm_df.merge(rec, on=['spec', 'weight'], how='left')
    fm_df = fm_df.drop(columns=['beta_recompute', 't_recompute',
                                'n_recompute', 'T_recompute'])
    cols = ['variant', 'variant_label', 'spec', 'spec_label', 'weight',
            'beta_peermax', 'se_peermax', 't_peermax', 'p2_peermax',
            'ci_low_peermax', 'ci_high_peermax',
            'beta_ownmax', 'se_ownmax', 't_ownmax',
            'beta_peerreturn', 'se_peerreturn', 't_peerreturn',
            'T', 'n', 'n_fm_rows', 'n_missing_peermax_fm',
            'beta_headline', 't_headline', 'delta_beta_vs_headline',
            'delta_t_vs_headline',
            'edge_jaccard_median', 'edge_jaccard_min', 'edge_jaccard_pooled',
            'edge_weight_corr_pooled',
            'treatment_corr_peermax_vs_headline',
            'treatment_corr_peerreturn_vs_headline',
            'fixed_peermax_nonnull', 'fm_peermax_nonnull',
            'fm_peermax_lost_vs_headline', 'fm_peerreturn_lost_vs_headline',
            'n_sources_median', 'coverage_ratio_max_median',
            'coverage_ratio_ret_median']
    fm_df = fm_df[cols]
    write_csv(COMPARISON_CSV, fm_df)
    log(f'对比表写入 {COMPARISON_CSV.name}：{len(fm_df)} 行')

    # 决策
    post = snapshot_anchors('post')
    write_json(ANCHOR_POST, post)
    anchor_cmp = compare_anchors(pre, post) if pre is not None else {
        'status': 'SKIPPED', 'n_diff': None, 'diff': []}
    q = evaluate_q_status(fm_df, {}, net_metrics, contracts, anchor_cmp)
    proposed = [
        {'param': 'ma_k_target_teacher', 'current': 44,
         'proposed': 'per-year as-of targets [38,40,44,49,50,52,68,57,64]',
         'variants': ['p3yt', 'p3yt_ydf'],
         'impact': '网络 K、边数、PeerMAX/PeerReturn 及其全部下游',
         'requires': 'Phase 4 变更单 + registry/manifest/methodology + Gate1 重跑'},
        {'param': 'ma_lexicon_df_max', 'current': 0.25,
         'proposed': 'per-year df ratio from product_lexicon_v2_teacher_df_by_year.csv '
                     '(headline = 9-year supremum)',
         'variants': ['p3ydf', 'p3yt_ydf'],
         'impact': '词表/相似度/边集与密度（χ 词表 560/819 词、7.26%/12.08% 命中质量）',
         'requires': 'Phase 4 变更单 + 9 年词表与网络重建 + Gate1 重跑'},
        {'param': 'sw2_industry_source', 'current': 'latest static (SwClass2021)',
         'proposed': 'effective_from <= as-of 时点（分类生效日口径，非严格 PIT）',
         'variants': ['p3asof', 'p3yt', 'p3yt_ydf'],
         'impact': 'K 目标/行业基线/A12 行业列；2021-07-30 vintage 断点（L2 组数 104→131）',
         'requires': 'Phase 4 变更单 + R1 阈值口径裁定（样本覆盖 97.27% vs ≥99.9%）'},
        {'param': 'version_manifest.network_versions',
         'proposed': '新增 5 个 shadow 目录条目（前缀与 available_at=(y+1)-04-30）',
         'variants': SHADOW_IDS,
         'impact': '网络版本白名单；现有两处 Phase-1 护栏的 shadow 白名单需回滚',
         'requires': 'Phase 4 变更单 + manifest 版本 bump'},
        {'param': 'm1_contracts', 'current': '1.0',
         'proposed': 'bump 到覆盖 shadow treatment 源的合同版本',
         'variants': SHADOW_IDS,
         'impact': '样本/面板合同与审计',
         'requires': 'Phase 4 变更单'},
        {'param': 'methodology', 'proposed': '§3 网络构造 / §5 推断段落增补 shadow 口径与 '
                                            'as-of「分类生效日」限定',
         'variants': SHADOW_IDS,
         'impact': '对外表述与可复现性',
         'requires': 'Phase 4 文档变更'},
        {'param': 'rewiring_reps', 'current': 1000, 'observed': 200,
         'proposed': '不推广推断层前必须补跑到 1000（3.4 R=200，仅提精度不改局部重连语义）',
         'variants': SHADOW_IDS,
         'impact': 'N-8 零分布分位/p 值精度',
         'requires': '夜间批跑 + 3.4 回填'},
    ]
    counts = {
        'phase': '3.6',
        'script': str(Path(__file__).relative_to(ROOT)).replace('\\', '/'),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_sec': round(time.time() - t0, 1),
        'version_tag': 'p3shadow_v1_network_m1_compare',
        'constants': {'sample_rows': SAMPLE_ROWS, 'fm_rows': FM_ROWS,
                      'fm_months': FM_MONTHS, 'min_n': MIN_N, 'nw_lag': NW_LAG,
                      'winsor': list(WINSOR),
                      'specs': {sid: label for sid, label, _ in SPECS},
                      'controls': CONTROLS,
                      'headline_registered': {'ew_beta': HEADLINE_EW_BETA,
                                              'ew_t': HEADLINE_EW_T,
                                              'vw_beta': HEADLINE_VW_BETA,
                                              'vw_t': HEADLINE_VW_T}},
        'variants': {v['id']: {'label': v['label'], 'dir': str(v['dir'].relative_to(ROOT)),
                               'prefix': v['prefix'],
                               'registered_in_manifest': bool(v.get('registered'))}
                     for v in VARIANTS},
        'month_mapping': {'by_text_year': {str(ty): months_by_ty[ty] for ty in YEARS},
                          'n_months': len(mapping),
                          'sample_check': map_check},
        'headline_recompute_validation': recompute_validation,
        'anchors': {'pre': pre, 'post': post, 'compare': anchor_cmp},
        'coverage': {vid: net_metrics[vid]['coverage'] for vid in ALL_IDS},
        'network_metrics': {vid: net_metrics[vid] for vid in ALL_IDS},
        'edge_contracts': {vid: contracts[vid] for vid in ALL_IDS},
        'treatment_correlations': corr,
        'fm_rows': fm_df.to_dict('records'),
        'q_status': q,
        'decision': {'value': 'HOLD',
                     'reason': 'Q4/Q5/Q8 未满足 + 默认 HOLD；任一 shadow 网络均未通过全部推广判据；'
                               '口径分叉的选择权交回项目，需显式书面批准（Phase 4 变更单）。'},
        'proposed_registry_diff': proposed,
    }
    write_json(COUNTS_JSON, counts)

    # 输出 sha + 重跑比对
    prev = {}
    if OUTPUT_SHA.is_file():
        try:
            prev = json.loads(OUTPUT_SHA.read_text(encoding='utf-8')).get('files', {})
        except Exception:
            prev = {}
    files = [COMPARISON_CSV] + [SHADOW_DATA / f'p3_6_shadow_treat_{vid}.parquet'
                                for vid in SHADOW_IDS]
    cur = {str(p.relative_to(ROOT)).replace('\\', '/'): sha256_file(p) for p in files}
    diff = sorted(k for k in set(prev) | set(cur) if prev.get(k) != cur.get(k))
    rerun = {'status': ('NO_PREVIOUS' if not prev else
                        'IDENTICAL' if not diff else 'DIFFERENT'),
             'n_compared': len(cur), 'n_diff': len(diff), 'diff': diff}
    write_json(OUTPUT_SHA, {'phase': '3.6', 'run_at': datetime.now(timezone.utc).isoformat(),
                            'files': cur, 'rerun_check': rerun})
    counts['rerun_check'] = rerun
    counts['output_sha'] = cur
    if rerun['status'] == 'IDENTICAL':
        counts['q_status']['Q6'] = {
            'status': 'PASS',
            'note': '确定性产物重跑逐字节一致；pytest 全量结果见 '
                    'artifacts/phase3/3.6_execution.md。'}
        counts['q_status']['_overall'] = {
            'all_pass': all(v.get('status') == 'PASS'
                            for k, v in counts['q_status'].items()
                            if not k.startswith('_')),
            'decision': 'HOLD'}
    # 重新写 counts 以纳入 rerun_check（counts 不参与 sha 台账）
    write_json(COUNTS_JSON, counts)

    log(f"决策：HOLD；Q 状态 "
        f"{ {k: v.get('status') for k, v in q.items() if not k.startswith('_')} }")
    log(f"重跑比对：{rerun['status']}（n_diff={rerun['n_diff']}）")
    log(f'完成：{COMPARISON_CSV.name} / {COUNTS_JSON.name}；elapsed='
        f'{counts["elapsed_sec"]}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
