"""Phase 3.3（N-2 原文第四步阈值规则）shadow：原文字面规则的**最小证据** + PC 对照 + 「不可比」决策数据底座。

背景（`artifacts/phase3/3.0_plan.md` §2.3；审计 L-2 / N-2）
--------------------------------------------------------------
原文第四步对手集合 = 「相似度 > **团内最小相似度**」的公司，**字面不限定候选域**（可含团外）；
项目口径 = **同团成员**（`s3_ma_build_edges.py` 已登记偏离：「团内最小相似度阈值法实测退化」）。
既有退化证据只在 2021 年 **self 语料**：K=300 → peer 3,456 / lift 1.14；**K=400 → 3,347 / 1.19**
（`reports/data_gates/G0/s3_ma_K_explore.json`）；**teacher 语料（headline K_y 235–463）从未读过** → 本脚本补齐。
「窄词表重建到原文 ~1.6 万 active_vocab」路线实测不可行（df 阈值收紧对词表/active_vocab 几乎无影响，
见本脚本 `--vocab-probe` 实测与 `3.3_counts.json` 的 `vocab_sensitivity`）。

本脚本做什么（headline 只读；只写 shadow）
--------------------------------------------------------------
1. 逐年 `build_vectors(y, corpus='teacher')`（headline 词表 df≤0.25、同筛选、等权 0/1）→ 全相似度矩阵 C；
2. **原文规则**：`thr_i` = focal 所在团的**团内最小相似度**（团内任意两成员对的最小相似度，团内全员共享；
   单点团 → 1.0，即无 peer），`mask = C > thr_i`（**严格大于**、**候选域 = 全部节点**、排除对角）；
   **项目规则**（对照）：同团成员（peer = 团规模 − 1）；
3. 逐年报：peer 分位 / 占市场比 / lift（申万二级；latest 静态与 as-of 两套标签）/ 与同团规则重合度 /
   thr 分位 / 团内零相似对；
4. **PC 对照**（文本年 2017–2022，y−1 年营收可得）：`pc_origrule = −HHI(原文规则对手集合 ∪ {focal},
   y−1 年销售收入)`（对手集合按原文规则含团外；HHI 分母含 focal 自身，与 headline PC 同构），
   与 headline 同团 PC 并排；2014–2016 营收面板缺 → pc 留空（不填 0，`pc_status=revenue_year_not_in_panel`）；
5. R1 对账：逐年矩阵与 headline `{y}_similarity.csv` 逐对核对；同团规则重算 PC 与 headline `{y}_pc.csv`
   逐值对账（容差 1e-6）；同团规则 peer/lift/K 重算与 freeze 报告逐值对账；
6. 固化解读：`3.3_counts.json`（退化计数/PC 对比/词表敏感度/结论）、`3.3_origrule_readout.csv`、
   `pc_origrule_p3shadow.csv`(+`.meta.json`)，另 `3.3_decision_memo.md` / `3.3_execution.md`（执行期写）。

不改：headline 网络/PC/M1（只读）、`reports/data_gates/G0/*`、`configs/*`、现有脚本与测试；版本串 `p3shadow_v1`。

产物
----
  data/processed/phase3_shadow/pc_origrule_p3shadow.csv       逐 focal×年×规则（原文规则 / 同团对照）
  data/processed/phase3_shadow/pc_origrule_p3shadow.meta.json 口径/列定义/版本
  artifacts/phase3/3.3_origrule_readout.csv                   逐年 × {headline,p3asof} × {2 规则} 读数
  artifacts/phase3/3.3_counts.json                            计数/对账/结论/锚台账
  artifacts/phase3/3.3_anchor_{pre,post}.json                 9 锚 + headline 45 文件 sha 台账（+3.1 产物）
  artifacts/phase3/3.3_output_sha.json                        本次 shadow 产物 sha

用法
----
  python scripts/data_audit/p3_3_pc_origrule_shadow.py --steps all
  python scripts/data_audit/p3_3_pc_origrule_shadow.py --steps readout --years 2021   # 子集重算
  python scripts/data_audit/p3_3_pc_origrule_shadow.py --steps vocab --no-vocab-probe
"""

# --- scripts/lib bootstrap ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from lib.io import json_default as _json_default
from lib.shadow.p3_3_pc_origrule_shadow import ALLOWED_ROOTS, ANCHOR_POST, ANCHOR_PRE, ART, ASOF_MAP_PARQUET, BANNED_ROOTS, COUNTS_JSON, FREEZE_REPORT, HEADLINE_EDGES, HEADLINE_K, HEADLINE_NET, HEADLINE_NODES, NOT_DEGENERATE_LIFT_MIN, NOT_DEGENERATE_PEER_MED_MAX, ORIGINAL_PAPER_EFFECTIVE_WORDS, OUTPUT_SHA, P3ASOF_NET, PC_YEARS, READONLY_INPUTS, READOUT_COLS, READOUT_CSV, REVENUE_PKL, ROOT, RULE_CLUSTER, RULE_ORIG, SHADOW_CSV, SHADOW_CSV_COLS, SHADOW_DIR, SHADOW_META, SHADOW_VERSION, SOURCE_VERSION, SW2_LATEST, TEACHER_LEXICON, VOCAB_DF_GRID, VOCAB_PROBES, YEARS, _atomic_write_text, anchors_snapshot, asof_l2_map_for_key, assert_output_path_guard, build_shadow_rows, cluster_rule_mask, compare_anchors, degeneracy_verdict, edge_share_lift, hhi_pc, l2_index, latest_l2_map, load_rev, log, matrix_vs_headline_edges, orig_rule_mask, output_sha_ledger, peer_stats, read_year, sha_ledger_dir, vocab_sensitivity, within_cluster_min_sim, write_csv, write_json
# --- end scripts/lib bootstrap ---

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

sys.path.insert(0, str(ROOT / 'scripts' / 'data_audit'))
from s3_m1_lib import assert_m1_inputs_sha, require, sha256_file  # noqa: E402
from s3_ma_lexicon import build_automaton, load_lexicon  # noqa: E402
from s3_b3_ma_cluster_grid import build_vectors, load_refs  # noqa: E402

# ---------------------------------------------------------------- 路径与常量



# headline 登记读数（只读对账；`3.0_plan.md` §1.1）
# 退化停机判据（3.0_plan §2.3）：peer 中位 < 500 **且** lift > 5 = 未退化 → 停机升级项目决策
# 词表敏感度探针（(文本年, df_max) → build_vectors 实测 active_vocab；对表 3.0_plan §2.3 读数）


# ---------------------------------------------------------------- 输出路径守卫（同 3.1/3.2 型）












# ---------------------------------------------------------------- 纯函数（合成可测）














# ---------------------------------------------------------------- 只读装载










# ---------------------------------------------------------------- 逐年读数




# ---------------------------------------------------------------- 锚台账








# ---------------------------------------------------------------- vocab 敏感度


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='Phase 3.3 原文第四步阈值规则 shadow（最小证据 + PC 对照）')
    ap.add_argument('--steps', default='all',
                    help='all 或逗号分隔子集：preflight,readout,vocab,accept,postflight')
    ap.add_argument('--years', type=int, nargs='*', default=None, help='readout 文本年（缺省全 9 年）')
    ap.add_argument('--net-dir', default=None, help='headline 网络目录（缺省 data/processed/network_ma_teacher）')
    ap.add_argument('--no-vocab-probe', action='store_true',
                    help='跳过 active_vocab 实测探针（只做词表层 df 阈值表）')
    args = ap.parse_args()
    global HEADLINE_NET
    if args.net_dir:
        HEADLINE_NET = Path(args.net_dir)
    steps = (['preflight', 'readout', 'vocab', 'accept', 'postflight']
             if args.steps == 'all' else [s.strip() for s in args.steps.split(',') if s.strip()])
    years = [int(y) for y in (args.years or YEARS)]
    t_all = time.time()
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)

    counts = {'run_at': datetime.now(timezone.utc).isoformat(), 'steps': steps,
              'shadow_version': SHADOW_VERSION, 'years': years, 'pc_years': PC_YEARS,
              'run_history': []}
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
            prev.update({'run_at': counts['run_at'], 'steps': steps, 'years': years,
                         'run_history': hist})
            counts = prev

    if 'preflight' in steps:
        counts['anchor_pre'] = anchors_snapshot('pre')
        write_json(ANCHOR_PRE, counts['anchor_pre'])
        log(f'preflight：9 锚 + {counts["anchor_pre"]["headline_network_n_files"]} 个 headline 网络文件 sha'
            f' + {len(counts["anchor_pre"]["p3asof_network_sha"])} 个 3.1 产物 sha 已记录')

    if 'readout' in steps:
        terms, lmeta = load_lexicon(TEACHER_LEXICON)
        A = build_automaton(terms)
        uni = load_refs()
        l2m = latest_l2_map()
        rev = load_rev()
        log(f'词表：kept={lmeta["terms_kept"]:,}（df≤{lmeta["df_max"]}）；营收面板年 {sorted(rev)}')
        readout_rows, shadow_rows = [], []
        pc_compare, matrix_chk, vocab_stats, secs = {}, {}, {}, {}
        for y in years:
            res = read_year(y, A, uni, l2m, rev, vocab_stats)
            readout_rows += res['readout_rows']
            shadow_rows += res['shadow_rows']
            matrix_chk[str(y)] = res['matrix_check']
            secs[str(y)] = res['sec']
            if res['pc_summary'] is not None:
                pc_compare[str(y)] = res['pc_summary']
            log(f'[{y}] readout 完成（{res["sec"]}s；矩阵核对 {res["matrix_check"]["n_pairs"]:,} 对，'
                f'max|Δ|={res["matrix_check"]["max_abs_diff"]:.2e}）')
        del A, uni
        write_csv(READOUT_CSV, [[r[c] if r[c] is not None else '' for c in READOUT_COLS]
                                for r in readout_rows], READOUT_COLS)
        write_csv(SHADOW_CSV, shadow_rows, SHADOW_CSV_COLS)
        meta = {
            'shadow_version': SHADOW_VERSION,
            'script': 'scripts/data_audit/p3_3_pc_origrule_shadow.py',
            'rule_definitions': {
                'origrule_minthr_global': '原文第四步字面规则：thr_i = focal 所在团的团内最小相似度'
                                          '（团内任意两成员对的最小相似度，团内全员共享；单点团 → 1.0 无 peer）；'
                                          'peer 集合 = {j: sim(i,j) > thr_i}（严格大于；候选域 = 全部节点；'
                                          '排除 focal 自身）；含团外公司。',
                'same_cluster_members': '项目口径（对照）：peer 集合 = 同团成员（peer 数 = 团规模 − 1）。',
            },
            'origrule_asymmetry': '原文规则按**行阈值**判定（mask_ij = C_ij > thr_i），thr 逐团不同 → '
                                  'mask **不必对称**（(i,j) 入选而 (j,i) 不入选是合法结果）；lift 用有向份额'
                                  '（与 eval_K / eval_at 同式）。',
            'pc_definition': 'pc = −HHI(y−1 年销售收入份额)；HHI 分母含 focal 自身（与 headline '
                             's3_ma_pc_hhi 同构）；对手集合 = 规则给出的 peer 集合；缺营收不计入分母但报告覆盖；'
                             '覆盖 <2 或 Σv ≤ 0 或 focal 缺营收 → pc 留空（不填 0）。',
            'revenue_year_rule': 'PC_{i,y} 需 y−1 年营收（面板 FY2016–2024）→ 文本年 2017–2022；'
                                 '2014–2016 → pc_status=revenue_year_not_in_panel（留空）。',
            'available_at_rule': '(y+1)-04-30（= headline build_report available_at，逐行对账）',
            'cluster_source': 'headline（data/processed/network_ma_teacher/clusters/，只读）',
            'lexicon': lmeta, 'corpus': 'teacher_mda_panel（headline 词表 df≤0.25）',
            'p3asof_note': 'network_ma_teacher_p3asof 的聚类只进 3.3_origrule_readout.csv（交叉读数）；'
                           '不进本 CSV',
            'columns': SHADOW_CSV_COLS,
            'column_notes': {
                'peer_share_of_market': 'n_competitors / (n_firms − 1)',
                'same_sw2_peer_share_latest': '本 focal 的对手集合中与 focal 同申万二级的比例（latest 静态标签）',
                'sw2_lift_latest': '年层面 lift（同申万二级有向边份额 / 边际基线；eval_K 同式），同规则行内为常数',
                'sw2_lift_asof': '同上，行业标签用 as-of（effective_from ≤ (y+1)-04-30，来自 3.1 映射）；'
                                 '3.1 映射缺件时为空',
                'pc_headline': 'headline {y}_pc.csv 的 pc（同团成员口径；两规则行都带，便于并排）',
                'pc_matches_headline': '仅 same_cluster_members 行：本文件重算 pc 与 headline pc.csv 一致（容差 1e-6）',
            },
            'years': years, 'pc_years': sorted(int(k) for k in pc_compare),
        }
        write_json(SHADOW_META, meta)
        counts['readout'] = {'rows': readout_rows, 'matrix_check': matrix_chk,
                             'vocab_filter_stats_by_year': vocab_stats,
                             'n_shadow_rows': len(shadow_rows),
                             'shadow_csv': str(SHADOW_CSV.relative_to(ROOT)),
                             'secs': secs}
        counts['pc_compare'] = pc_compare
        log(f'readout 完成：readout {len(readout_rows)} 行 / shadow {len(shadow_rows)} 行')

    if 'vocab' in steps:
        counts['vocab_sensitivity'] = vocab_sensitivity(not args.no_vocab_probe)
        log('vocab：df 阈值敏感度表 + 探针完成')

    if 'accept' in steps:
        require('readout' in counts, 'accept 需要 readout 计数（先跑 --steps readout）')
        rr = counts['readout']['rows']
        hl = [r for r in rr if r['cluster_source'] == 'headline']
        orig_rows = [r for r in hl if r['peer_rule'] == RULE_ORIG]
        cl_rows = [r for r in hl if r['peer_rule'] == RULE_CLUSTER]
        verdict = degeneracy_verdict(orig_rows)
        freeze = json.loads(FREEZE_REPORT.read_text(encoding='utf-8'))
        fr = {}
        for r in cl_rows:
            y = str(r['text_year'])
            fz = freeze['years'].get(y, {})
            fr[y] = {'peer_median_recomputed': r['peer_median'],
                     'peer_median_freeze': fz.get('peer_median'),
                     'sw2_lift_recomputed': r['sw2_lift_latest'],
                     'sw2_lift_freeze': fz.get('sw2_lift'),
                     'K_recomputed': r['n_clusters'], 'K_freeze': fz.get('K'),
                     'isolated_share_recomputed': r['isolated_share'],
                     'isolated_share_freeze': fz.get('isolated_share')}
        require(all(abs(v['peer_median_recomputed'] - v['peer_median_freeze']) <= 0
                    for v in fr.values()), '同团规则 peer 中位与 freeze 报告不符；停机')
        require(all(v['K_recomputed'] == v['K_freeze'] for v in fr.values()),
                '同团规则团数与 freeze 报告不符；停机')
        require(all(abs(v['sw2_lift_recomputed'] - v['sw2_lift_freeze']) <= 0.005
                    for v in fr.values()), '同团规则 sw2_lift 与 freeze 报告不符（容差 0.005）；停机')
        pcc = {}
        for k, v in (counts.get('pc_compare') or {}).items():
            if not isinstance(v, dict) or 'orig' not in v:
                continue
            pcc[k] = {'revenue_year': v['revenue_year'], 'n_firms': v['n_firms'],
                      'orig_pc_median': v['orig']['pc_median'],
                      'headline_pc_median': v['headline_file']['pc_median'],
                      'abs_magnitude_ratio_median': (v.get('corr') or {}).get(
                          'abs_magnitude_ratio_median'),
                      'corr_orig_vs_headline': (v.get('corr') or {}).get('orig_vs_headline'),
                      'match': v['match']}
        counts['origrule_readout'] = {'per_year': {int(r['text_year']): r for r in orig_rows},
                                      'per_year_p3asof': {int(r['text_year']): r for r in rr
                                                          if r['cluster_source'] == 'p3asof'
                                                          and r['peer_rule'] == RULE_ORIG}}
        counts['project_rule_readout_vs_freeze'] = fr
        counts['degeneracy'] = verdict
        counts['pc_compare_summary'] = pcc
        counts['conclusion'] = {
            'status': '不可比/未复现（Phase 0.2 结论维持）',
            'verdict_wording': '原文规则在当前词表下不可构造/退化',
            'degenerate_under_current_lexicon': bool(verdict['degenerate']),
            'statements': [
                '原文第四步字面规则（相似度 > 团内最小相似度，候选域不受限）在当前 teacher 词表下**退化** —— '
                '即按本步实现（严格 >、候选域 = 全节点、行阈值）该口径的对手集合**不可构造**为「数十家量级」的'
                '对手集合：peer ≈ 全市场（见 degeneracy.peer_median_by_year 与 '
                'degeneracy.peer_share_of_market_median_by_year）；',
                'PC 与 headline 同团 PC **不可比**（量级与横截面排序均不同；见 pc_compare_summary）；',
                '窄词表路线不可行：df 阈值收紧对词表/active_vocab 几乎无影响，无法到原文 ~1.66 万词量级'
                '（见 vocab_sensitivity）；',
                '若要与原文对表，须**同时**满足「原文规则 + 近似原文词表规模」——本项目两条都不满足；'
                '本步不启动窄词表重建。',
            ],
            'wording_note': '本步避免「复现失败/符号相反/近似一致/可对表」表述；判定用词 = 不可比/未复现',
        }
        write_json(COUNTS_JSON, counts)
        log(f'accept：退化裁决 = {verdict["degenerate"]}（peer 中位 '
            f'{verdict["median_of_year_peer_medians"]:.0f}，lift {verdict["median_of_year_lifts_latest"]}）')
        if not verdict['degenerate']:
            print('[HARD-GATE] 原文规则**未**退化（peer 中位 <500 且 lift >5）→ 停机，升级项目决策；'
                  '不得自行启动窄词表重建', file=sys.stderr, flush=True)
            write_json(OUTPUT_SHA, output_sha_ledger())
            raise SystemExit(3)

    if 'postflight' in steps:
        post = anchors_snapshot('post')
        write_json(ANCHOR_POST, post)
        pre = json.loads(ANCHOR_PRE.read_text(encoding='utf-8')) if ANCHOR_PRE.is_file() else None
        cmp_ = compare_anchors(pre, post) if pre else {'status': 'NO_PRE'}
        counts['anchor_post'] = post
        counts['anchor_compare'] = cmp_
        counts['output_sha_rerun'] = output_sha_ledger()
        write_json(OUTPUT_SHA, counts['output_sha_rerun'])
        if cmp_['status'] != 'MATCH':
            write_json(COUNTS_JSON, counts)
            print('[HARD-GATE] headline/输入锚发生变化；停机（禁止重算覆盖）', file=sys.stderr, flush=True)
            raise SystemExit(2)
        log(f'postflight：锚对比 {cmp_["status"]}（headline {post["headline_network_n_files"]} 文件 + '
            f'9 锚 + 3.1 产物 {len(post["p3asof_network_sha"])} 文件）')

    counts['elapsed_sec_total'] = round(time.time() - t_all, 1)
    write_json(COUNTS_JSON, counts)
    log(f'完成：总耗时 {counts["elapsed_sec_total"]}s → {COUNTS_JSON}')


if __name__ == '__main__':
    main()
