#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S3 第一梯队 ⑤ / B″11-8：df 阈值网格 10%/25%/100%（`network_business_df_grid.{json,md}`）。

**只读复算 + 新建目录**：不重建主口径、不改任何 headline_value / 冻结参数；0.25 档（现状）的
聚类与边表**落临时目录**（`artifacts/tmp/t1b/dfgrid/regr025/`），与 `data/processed/network_ma_teacher/`
主口径产物做 **sha256 回归门**；10%/100% 两档落新目录 `data/processed/network_ma_teacher_df{10,100}/`。

口径真源：
  - `reports/audits/s3_remaining_workplan_20261011.md` 第一梯队 5
  - `docs/methodology/methodology_execution_plan.md` §8 B″11-8
  - registry：`ma_lexicon_df_max`（0.25，headline 不动）、`ma_k_target_teacher`（44）

操作（**低成本路径**）：`network_business_cluster_freeze.py` / `network_business_edges_build.py` 的 `--df-max` 按既有词表
`data/processed/product_lexicon_v2_teacher.csv` 的 `df_ratio` 列即时过滤（`df_ratio ≤ df_max`；
已复算确认 `df_ratio ≤ 0.25` ≡ `keep_df25` 在 1,355,237 行逐行一致）→ 同管线构网（K 目标 44）
→ 比 peer 分布 / same_sw2 / lift / 边 Jaccard / `mean_terms`。**不重跑词库 df 重算**（省 ~5,855 s）。

验收：① **0.25 档与主口径产物 sha256 逐字节一致（回归门）**；② 三档结构指标表齐全；③ 结论句
（以实测定稿）。停机：0.25 档不可逐字节复现 → 停，先修参数化。**范围限定**：M-B 列 df 网格不覆盖。

用法：
  python scripts/network_business_df_grid.py                     # 三档全跑（含回归门）
  python scripts/network_business_df_grid.py --levels 0.10 1.00  # 只跑新档（回归门记 skipped）
  python scripts/network_business_df_grid.py --levels 0.25       # 只跑回归门
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
import network_business_chapter_cross_audit as CA  # noqa: E402  （统一口径：peer_stats / sw2_lift / isolation）

REPD = ROOT / 'reports/data_gates/G0'
TMP = ROOT / 'artifacts/tmp/t1b/dfgrid'
PROC = ROOT / 'data/processed'
MAIN_NET = PROC / 'network_ma_teacher'
LEXICON = PROC / 'product_lexicon_v2_teacher.csv'
TEXT_YEARS = list(range(2014, 2023))          # 文本年 → 网络年 2015–2023（主口径 9 年）
K_TARGET = 44                                  # registry ma_k_target_teacher（主口径 K 目标）
FREEZE = ROOT / 'scripts/network_business_cluster_freeze.py'
EDGES = ROOT / 'scripts/network_business_edges_build.py'
# 判定阈值（**预先写明**，结论句按此生成，避免事后措辞）
TH_JACCARD_NOSIG = 0.99                        # 「无显著差异」的边 Jaccard 下限（对 0.25 档）
TH_LIFT_REL_NOSIG = 0.02                       # 「无显著差异」的 |Δlift|/lift 上限
TH_PEER_MED_DELTA = 2.0                        # 「无显著差异」的 |Δ中位 peer| 上限


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def level_paths(df_max: float, tag: str) -> dict:
    """档位 → 聚类目录 / 边表目录 / 冻结报告路径（0.25 档一律落临时目录，绝不碰主口径）。"""
    if tag == '25':
        return {'net': TMP / 'regr025' / 'net', 'clu': TMP / 'regr025' / 'clusters',
                'freeze_report': TMP / 'freeze_df25.json', 'regression': True}
    net = PROC / f'network_ma_teacher_df{tag}'
    return {'net': net, 'clu': net / 'clusters',
            'freeze_report': TMP / f'freeze_df{tag}.json', 'regression': False}


def run(cmd: list[str], log: list[str]) -> None:
    log.append('$ ' + ' '.join(str(c) for c in cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace')
    log.append(r.stdout or '')
    if r.returncode != 0:
        log.append(r.stderr or '')
        raise RuntimeError(f'命令失败（returncode={r.returncode}）：{" ".join(str(c) for c in cmd)}')
    log.append(f'[elapsed {time.time() - t0:.1f}s]')


def edge_set(net: Path, y: int) -> set:
    d = CA.read_edges(net, y)
    return set(map(tuple, d[['focal_id', 'source_id']].values))


def jaccard(a: set, b: set) -> dict:
    inter, union = len(a & b), len(a | b)
    return {'n_a': len(a), 'n_b': len(b), 'n_intersection': inter, 'n_union': union,
            'jaccard': round(inter / union, 6) if union else None,
            'containment_a_in_b': round(inter / len(a), 6) if a else None,
            'containment_b_in_a': round(inter / len(b), 6) if b else None}


def metrics(net: Path, y: int, sw2: dict, freeze_years: dict) -> dict:
    sup = CA.read_support(net, y)
    edges = CA.read_edges(net, y)
    ps = CA.peer_stats(sup, edges)
    lf = CA.sw2_lift(edges, sw2, set(sup['focal_id']))
    iso = CA.isolation(edges, sup)
    cr = CA.corpus_richness(net, y)
    fy = freeze_years.get(str(y), {})
    return {'text_year': y, 'network_year': y + 1, 'n_nodes': ps['focal_nodes_total'],
            'n_edges': ps['edges'], 'peer_median': ps['peer_median'], 'peer_mean': ps['peer_mean'],
            'peer_p90': ps['peer_p90'], 'isolated_share': ps['isolated_share'],
            'same_sw2_share': lf['same_sw2_share'], 'base_rate_canonical': lf['base_rate_canonical'],
            'lift': lf['lift'], 'n_edges_mapped': lf['n_edges_mapped'],
            'mean_terms': cr.get('mean_terms'), 'active_vocab': cr.get('active_vocab'),
            'K': fy.get('K'), 'K_mode': fy.get('K_mode'),
            'freeze_same_sw2_edge_share': fy.get('same_sw2_edge_share'),
            'freeze_sw2_lift': fy.get('sw2_lift')}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=float, nargs='*', default=[0.25, 0.10, 1.00],
                    help='df_max 档位（缺省 0.25 0.10 1.00；0.25 = 现状/回归门，产物落临时目录）')
    ap.add_argument('--out-json', default=str(REPD / 's3_ma_df_grid.json'))
    ap.add_argument('--out-md', default=str(REPD / 's3_ma_df_grid.md'))
    ap.add_argument('--years', type=int, nargs='*', default=TEXT_YEARS)
    ap.add_argument('--skip-run', action='store_true', help='跳过构网（只做指标/回归门，产物须已存在）')
    args = ap.parse_args()

    sw2 = CA.load_sw2()
    log: list[str] = []
    TMP.mkdir(parents=True, exist_ok=True)
    levels = []
    for df_max in args.levels:
        tag = f'{round(df_max * 100)}'
        p = level_paths(df_max, tag)
        print(f'[T1B-⑤] 档 df_max={df_max}（tag={tag}）→ net={p["net"]}'
              f'{" [回归档，临时目录]" if p["regression"] else ""}', flush=True)
        if not args.skip_run:
            p['net'].mkdir(parents=True, exist_ok=True)
            p['clu'].mkdir(parents=True, exist_ok=True)
            run([sys.executable, str(FREEZE), '--corpus', 'teacher', '--df-max', str(df_max),
                 '--out-dir', str(p['clu']), '--out-report', str(p['freeze_report'])], log)
            run([sys.executable, str(EDGES), '--corpus', 'teacher', '--df-max', str(df_max),
                 '--net-dir', str(p['net']), '--cluster-dir', str(p['clu'])], log)
        fr = json.loads(p['freeze_report'].read_text(encoding='utf-8')) if p['freeze_report'].exists() else {}
        levels.append({'df_max': df_max, 'tag': tag, **p, 'freeze': fr})

    # 逐档逐指标
    for L in levels:
        L['per_year'] = {y: metrics(L['net'], y, sw2, L['freeze'].get('years', {})) for y in args.years}
        L['lexicon'] = L['freeze'].get('lexicon', {})
        L['K_by_year'] = {y: L['per_year'][y]['K'] for y in args.years}
        print(f'[T1B-⑤] 档 {L["tag"]}%：terms_kept={L["lexicon"].get("terms_kept")} '
              f'K={list(L["K_by_year"].values())}', flush=True)

    # 回归门：0.25 档 vs 主口径逐文件 sha256
    regr = next((L for L in levels if L['tag'] == '25'), None)
    reg_rows, reg_pass = [], None
    if regr is not None:
        for y in args.years:
            for kind in ('similarity', 'support'):
                a = MAIN_NET / f'{y}_{kind}.csv'
                b = regr['net'] / f'{y}_{kind}.csv'
                sa = sha256(a) if a.exists() else None
                sb = sha256(b) if b.exists() else None
                reg_rows.append({'text_year': y, 'network_year': y + 1, 'file': f'{y}_{kind}.csv',
                                 'main_sha256': sa, 'regr_sha256': sb,
                                 'byte_identical': bool(sa and sb and sa == sb),
                                 'main_bytes': a.stat().st_size if a.exists() else None,
                                 'regr_bytes': b.stat().st_size if b.exists() else None})
        reg_pass = bool(reg_rows) and all(r['byte_identical'] for r in reg_rows)
        print(f'[T1B-⑤] 回归门（0.25 档 vs 主口径 sha256 逐字节）：'
              f'{"PASS" if reg_pass else "FAIL"}（{sum(1 for r in reg_rows if r["byte_identical"])}'
              f'/{len(reg_rows)} 文件一致）', flush=True)

    # 边 Jaccard（各档 vs 0.25；以及 0.10 vs 1.00）
    jac = {}
    base = next((L for L in levels if L['tag'] == '25'), None)
    for L in levels:
        if L['tag'] == '25':
            continue
        per_y = {}
        for y in args.years:
            ja = edge_set(L['net'], y)
            per_y[y] = {'vs_main_025': (jaccard(ja, edge_set(MAIN_NET, y)) if MAIN_NET.exists() else None),
                        'vs_regr_025': (jaccard(ja, edge_set(base['net'], y)) if base else None)}
        jac[L['tag']] = per_y
    if len(levels) >= 3:
        a = next((L for L in levels if L['tag'] == '10'), None)
        b = next((L for L in levels if L['tag'] == '100'), None)
        if a and b:
            jac['10_vs_100'] = {y: jaccard(edge_set(a['net'], y), edge_set(b['net'], y))
                                for y in args.years}

    # 结论（以实测定稿；阈值预先写明）
    concl, concl_basis = None, {'jaccard_nosig_min': TH_JACCARD_NOSIG,
                                'lift_rel_nosig_max': TH_LIFT_REL_NOSIG,
                                'peer_median_delta_max': TH_PEER_MED_DELTA}
    if base is not None and len(levels) >= 2:
        def pooled(L):
            ok = list(L['per_year'].values())
            share = sum(r['same_sw2_share'] * r['n_edges_mapped'] for r in ok) / sum(r['n_edges_mapped'] for r in ok)
            b = sum(r['base_rate_canonical'] for r in ok) / len(ok)
            return {'lift_pooled_share_over_mean_base': share / b, 'same_sw2_share_pooled': share}
        rows = []
        for L in levels:
            if L['tag'] == '25':
                continue
            jv = [v['vs_regr_025']['jaccard'] for v in jac.get(L['tag'], {}).values()
                  if v.get('vs_regr_025')]
            d_peer = [abs(L['per_year'][y]['peer_median'] - base['per_year'][y]['peer_median'])
                      for y in args.years]
            d_lift = [(L['per_year'][y]['lift'] - base['per_year'][y]['lift']) / base['per_year'][y]['lift']
                      for y in args.years]
            rows.append({'tag': L['tag'], 'df_max': L['df_max'],
                         'jaccard_vs_025_mean': round(sum(jv) / len(jv), 6) if jv else None,
                         'peer_median_delta_max': max(d_peer) if d_peer else None,
                         'lift_rel_delta_mean_pct': round(100 * sum(d_lift) / len(d_lift), 4) if d_lift else None,
                         'lift_pooled': pooled(L)['lift_pooled_share_over_mean_base'],
                         'no_significant_difference': bool(
                             jv and sum(jv) / len(jv) >= TH_JACCARD_NOSIG
                             and max(d_peer) <= TH_PEER_MED_DELTA
                             and abs(sum(d_lift) / len(d_lift)) <= TH_LIFT_REL_NOSIG)})
        # 实测排序（lift 池化）
        ranked = sorted(levels, key=lambda L: pooled(L)['lift_pooled_share_over_mean_base'])
        concl_basis['pooled_lift_by_level'] = {L['tag']: round(pooled(L)['lift_pooled_share_over_mean_base'], 4)
                                              for L in levels}
        concl_basis['level_comparison_vs_025'] = rows
        concl_basis['lowest_lift_level'] = ranked[0]['tag']
        parts = []
        for r in rows:
            verdict = ('与 25% 档无显著差异' if r['no_significant_difference'] else '与 25% 档存在可测差异')
            parts.append(f"{r['tag']}% 档：边 Jaccard(对 25% 档) 均值 = {r['jaccard_vs_025_mean']}、"
                         f"中位 peer 最大绝对差 = {r['peer_median_delta_max']:g}、"
                         f"lift 相对差均值 = {r['lift_rel_delta_mean_pct']:+.4f}% → {verdict}")
        concl = ('**结论（以实测定稿）**：' + '；'.join(parts) + '。'
                 f"池化 lift 三档 = " + '、'.join(f"{L['tag']}%:{pooled(L)['lift_pooled_share_over_mean_base']:.4f}"
                                                for L in levels)
                 + f"（最低档 = {ranked[0]['tag']}%）。判定阈值（预先写明）：对 25% 档边 Jaccard 均值 ≥ "
                   f"{TH_JACCARD_NOSIG}、中位 peer 绝对差 ≤ {TH_PEER_MED_DELTA:g}、|lift 相对差| ≤ "
                   f"{TH_LIFT_REL_NOSIG:.0%} 三者同时满足才记「无显著差异」。"
                   f"**headline 仍为 0.25（`ma_lexicon_df_max` 不动）**；本网格只作可比性证据。")

    # 汇总表（逐档逐年的结构指标）
    table = []
    for L in levels:
        for y in args.years:
            r = L['per_year'][y]
            table.append({'df_max': L['df_max'], 'tag': L['tag'], 'text_year': y,
                          'network_year': y + 1, **{k: r[k] for k in
                                                    ('n_nodes', 'n_edges', 'peer_median', 'peer_mean', 'peer_p90',
                                                     'isolated_share', 'same_sw2_share', 'base_rate_canonical',
                                                     'lift', 'mean_terms', 'active_vocab', 'K')}})
    rep = {
        'step': 'S3 第一梯队 ⑤ df 阈值网格 10%/25%/100%（B″11-8）',
        'run_at': datetime.now(timezone.utc).isoformat(),
        'nature': ('只读复算 + 新建目录：不重建主口径、不改任何 headline_value / 冻结参数；'
                   '0.25 档产物落临时目录（不与主口径产物冲突）'),
        'inputs': {'lexicon': str(LEXICON.relative_to(ROOT)).replace('\\', '/'),
                   'corpus': 'data/processed/teacher_mda_panel.csv（teacher 主口径语料）',
                   'main_net_dir': str(MAIN_NET.relative_to(ROOT)).replace('\\', '/'),
                   'df_filter_rule': 'df_ratio ≤ df_max（既有列；不重跑词库 df 重算）'},
        'method': {'freeze': 'network_business_cluster_freeze.py --corpus teacher --df-max …（K 目标 registry '
                             f'ma_k_target_teacher={K_TARGET}，average 链）',
                   'edges': 'network_business_edges_build.py --corpus teacher --df-max …（同管线、同阈值、同 12 位舍入）',
                   'metrics_source': 'support/similarity 表按 network_business_chapter_cross_audit 统一口径复算'},
        'levels': {L['tag']: {'df_max': L['df_max'],
                              'net_dir': str(L['net'].relative_to(ROOT)).replace('\\', '/')
                              if str(L['net']).startswith(str(ROOT)) else str(L['net']),
                              'clusters_dir': str(L['clu'].relative_to(ROOT)).replace('\\', '/')
                              if str(L['clu']).startswith(str(ROOT)) else str(L['clu']),
                              'freeze_report': str(L['freeze_report'].relative_to(ROOT)).replace('\\', '/'),
                              'lexicon_terms_kept': L['lexicon'].get('terms_kept'),
                              'lexicon_terms_total_in_file': L['lexicon'].get('terms_total_in_file'),
                              'K_by_text_year': L['K_by_year'],
                              'regression_level': L['regression']}
                   for L in levels},
        'regression_gate_sha256_vs_main': {
            'rule': '0.25 档（现状）逐文件 sha256 与 data/processed/network_ma_teacher/ 主口径产物比对，'
                    '必须逐字节一致',
            'rows': reg_rows, 'n_files': len(reg_rows),
            'n_byte_identical': sum(1 for r in reg_rows if r['byte_identical']),
            'pass': reg_pass,
            'disposition': ('PASS：0.25 档参数化路径与主口径逐字节一致（回归门通过）' if reg_pass else
                            'FAIL/未跑：0.25 档不能逐字节复现 → 按停机规则先修参数化，不得继续报网格结论')
            if reg_pass is not None else 'skipped（本轮未含 0.25 档）'},
        'edge_jaccard': jac,
        'structure_table': table,
        'conclusion': concl, 'conclusion_basis': concl_basis,
        'acceptance': {'regression_gate_sha256_pass': reg_pass,
                       'three_levels_complete': bool(len(levels) >= 3),
                       'conclusion_from_measurement': bool(concl),
                       'headline_unchanged': 'ma_lexicon_df_max = 0.25（不动）'},
        'status': ('PASS' if (reg_pass is not False and len(levels) >= 3 and concl) else
                   'PARTIAL' if concl else 'FAIL'),
        'scope_note': '**范围限定**：M-B 列 df 网格本步不覆盖（另行登记；HP 的 10/25/100% 网格为阈值链'
                      '（TNIC）做法，M-B 已有 τ 校准网格）。',
        'log_tail': log[-1] if log else None,
    }
    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')

    md = ['# S3 第一梯队 ⑤ df 阈值网格 10%/25%/100%（B″11-8）', '',
          f"- run_at：{rep['run_at']}",
          f"- 状态：**{rep['status']}**；回归门（0.25 档 vs 主口径 sha256）："
          f"**{'PASS' if reg_pass else reg_pass}**"
          f"（{rep['regression_gate_sha256_vs_main']['n_byte_identical']}/{len(reg_rows)} 文件逐字节一致）",
          f"- 词表：`{rep['inputs']['lexicon']}`（`df_ratio ≤ df_max` 即时过滤；不重跑 df 重算）",
          f"- 产物：`data/processed/network_ma_teacher_df10/`、`data/processed/network_ma_teacher_df100/`"
          f"（0.25 档 = 临时目录 `{TMP.as_posix()}/regr025/`）", '',
          '## 三档结构指标（逐年）', '',
          '| 档 df_max | 网络年 | n_nodes | n_edges | 中位 peer | 均值 peer | p90 peer | 隔离率 | same_sw2 | lift | mean_terms | K |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in table:
        md.append(f"| {r['df_max']:g} | {r['network_year']} | {r['n_nodes']} | {r['n_edges']} | "
                  f"{r['peer_median']:g} | {r['peer_mean']:g} | {r['peer_p90']:g} | {r['isolated_share']} | "
                  f"{r['same_sw2_share']:.4f} | {r['lift']:.3f} | {r['mean_terms']} | {r['K']} |")
    md += ['', '## 边 Jaccard（对 0.25 档；含 vs 主口径）', '',
           '| 档 | 网络年 | Jaccard(vs 0.25 回归档) | 包含率(A⊂B) | 包含率(B⊂A) | Jaccard(vs 主口径) |',
           '|---|---|---|---|---|---|']
    for tag, per_y in jac.items():
        if tag == '10_vs_100':
            continue
        for y, v in per_y.items():
            vr, vm = v.get('vs_regr_025'), v.get('vs_main_025')
            md.append(f"| {tag}% | {y + 1} | {vr['jaccard'] if vr else '-'} | "
                      f"{vr['containment_a_in_b'] if vr else '-'} | {vr['containment_b_in_a'] if vr else '-'} | "
                      f"{vm['jaccard'] if vm else '-'} |")
    md += ['', '## 回归门（0.25 档 vs 主口径，逐文件 sha256）', '',
           '| 网络年 | 文件 | 主口径 sha256 | 回归档 sha256 | 逐字节一致 |', '|---|---|---|---|---|']
    for r in reg_rows:
        md.append(f"| {r['network_year']} | {r['file']} | `{(r['main_sha256'] or '')[:16]}…` | "
                  f"`{(r['regr_sha256'] or '')[:16]}…` | {r['byte_identical']} |")
    md += ['', '## 结论', '', concl or '（未生成）', '',
           rep['scope_note'], '']
    outm = Path(args.out_md)
    outm.write_text('\n'.join(md), encoding='utf-8')
    print(f'[T1B-⑤] status={rep["status"]} 回归门={reg_pass} -> {outp}', flush=True)
    print(f'[T1B-⑤] {concl}', flush=True)


if __name__ == '__main__':
    main()
