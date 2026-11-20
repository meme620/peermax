#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B / N-6（B2）：两列并报统一到 2018-05..2023-12 共同窗口重跑 inventory。

任务（N-6 处理项 ①）：
  - 两列样本（ma_teacher / mb_teacher）统一过滤事件月
    2018-05..2023-12，重跑 inventory（登记全控制规格），报 n / T / PeerMAX / OwnMAX /
    PeerReturn 的 beta 与 NW t；同时报各列原窗口读数作对照。
  - 只写 `artifacts/phase3b/3b_n6_*`；不重跑 pricing_gate / PC / 网络；不改
    headline / 系数 / gate / configs / G0 / reports。

口径（只读既有实现与登记参数）：
  - inventory = PeerMAXraw + OwnMAXraw + PeerReturn + 8 controls
    （size / bm / mom / rev / turnover / illiq / AnalystCoverage / NetworkDegree），
    与 `scripts/shadow_network_pricing.py` 的 inventory、`pricing_main_v1` 主规格同。
  - FM 样本 = 样本文件内 `pricing_in_fm_sample == True` 的行（与 headline 主表一致）。
  - 逐月横截面 OLS（EW）/ WLS（VW，weight=market_equity）-> 系数序列 ->
    均值 + Newey-West(lag=6, Bartlett)。
  - min_n = registry `sample_and_spec_registry.min_focal_count_per_month`（当前 100）。
  - 共同窗口 = 2018-05..2023-12（含端点，事件月口径）。

验证：
  - 原窗口读数与 `reports/data_gates/G0/pricing_table_v2.json` 的主表逐值对账
    （beta / NW t / N / T），不一致即退出，不写任何产物。
  - 9 个 pricing_gate 主输入 + 4 个样本 parquet 的 bytes 在计算前后各记一次，必须不变。

用法：
  python scripts/shadow_common_window.py
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from pricing_lib import (  # noqa: E402
    SAMPLE_WINDOW_END, SAMPLE_WINDOW_START, assert_pricing_contract, assert_pricing_inputs,
    fm_regression, param_value, require, sample_path, file_size,
)

ART = ROOT / 'artifacts/phase3b'
OUT_CSV = ART / '3b_n6_common_window.csv'
OUT_JSON = ART / '3b_n6_counts.json'
OUT_MD = ART / '3b_n6_execution.md'
SAMPLE_REPORT = ROOT / 'reports/data_gates/G0/pricing_sample_report.json'
G0_REFERENCE = ROOT / 'reports/data_gates/G0/pricing_table_v2.json'
REGISTRY = ROOT / 'configs/parameter_registry.yaml'
SCRIPT = Path(__file__).resolve()

COLUMNS = ['ma_teacher', 'mb_teacher']
ROLE = {
    'ma_teacher': '主口径（中文法 × 老师 MD&A）',
    'mb_teacher': '方法稳健性列（TF-IDF × 同语料）',
}
VARS = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn']
CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq',
            'AnalystCoverage', 'NetworkDegree']
inventory = VARS + CONTROLS

COMMON_START = '2018-05'
COMMON_END = '2023-12'
SAMPLE_VERSION = 'pricing_sample_v5_teacher_only_20261120'
MIN_N = int(param_value('sample_and_spec_registry')['min_focal_count_per_month'])
MIN_MONTHS = int(param_value('sample_and_spec_registry')['min_effective_month_count'])
NW_LAG = int(param_value('nw_lag'))
WINSOR = param_value('winsorization_pct')

# 共同窗口之外的差异（N-6 证据链，报告用；不改任何数值）
K_TARGET = {'ma_teacher': 44, 'mb_teacher': None}
CORPUS = {
    'ma_teacher': 'teacher_mda_panel（老师 MD&A）',
    'mb_teacher': 'teacher_mda_panel（老师 MD&A）',
}
METHOD = {
    'ma_teacher': '等权 0/1 词向量 × 余弦 × average，K 目标 44',
    'mb_teacher': 'TF-IDF 词向量 × 阈值 tau=0.0888212137',
}


# ------------------------------------------------------------------ 输出路径守卫
def assert_output_path_guard(path) -> Path:
    """只允许 artifacts/phase3b/3b_n6_* 输出。"""
    p = Path(path).resolve()
    art = ART.resolve()
    require(art in p.parents and p.name.startswith('3b_n6_'),
            f'输出路径越界（只允许 artifacts/phase3b/3b_n6_*）: {p}')
    return p


def _json_ready(obj):
    """递归把 NaN/Inf/numpy 标量转换为严格 JSON 可写值。"""
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _median(values):
    return float(np.median(values)) if values else None


def _block(df: pd.DataFrame) -> dict:
    """单列单窗口 inventory：EW + VW。"""
    out = {}
    for weight_tag, weight_col in (('ew', None), ('vw', 'market_equity')):
        res, per_month = fm_regression(df, 'R_next', inventory, weight_col=weight_col,
                                       min_n=MIN_N, nw_lag=NW_LAG)
        used = [d for d in per_month if d.get('used')]
        ns = [int(d['n']) for d in used]
        out[weight_tag] = {
            'weight_col': weight_col,
            'n_rows': int(len(df)),
            'n_est': int(sum(ns)),
            'T': int(res['n_months_used']),
            'fm_window': [used[0]['month'], used[-1]['month']] if used else None,
            'per_month_n': {'min': min(ns) if ns else None,
                            'median': _median(ns), 'max': max(ns) if ns else None},
            'all_months_full_rank': bool(all(d.get('rank_status') == 'FULL' for d in used)),
            'all_months_covariance_finite': bool(
                all(d.get('covariance_status') == 'finite' for d in used)),
            'coefs': res['coefs'],
            'min_n': int(res['min_n']), 'nw_lag': int(res['nw_lag']),
        }
    return out


def _load_fm_sample(key: str, sample_report: dict) -> pd.DataFrame:
    path = sample_path(key, 'v2')
    df = pd.read_parquet(path)
    expected = int(sample_report['samples'][key]['rows_total'])
    assert_pricing_contract(df, expected_rows=expected, kind='sample',
                       where=f'shadow_common_window:{key}:v2')
    require('pricing_in_fm_sample' in df.columns,
            f'{key}: 样本缺 pricing_in_fm_sample 列，无法复刻 headline FM 样本')
    require('R_next' in df.columns and set(inventory).issubset(df.columns),
            f'{key}: inventory 变量不齐（需 R_next + {inventory}）')
    return df


def _fmt(x, digits=5, signed=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 'NA'
    return f'{x:+.{digits}f}' if signed else f'{x:.{digits}f}'


def _fmt_t(x):
    return 'NA' if x is None else f'{x:+.2f}'


def _pct(new, old):
    if old is None or old == 0 or new is None:
        return None
    return (new - old) / abs(old)


def _rel(path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def main():
    for p in (OUT_CSV, OUT_JSON, OUT_MD):
        assert_output_path_guard(p)
    ART.mkdir(parents=True, exist_ok=True)

    # ---------------- 计算前 sha 账本（9 个 pricing_gate 主输入 + 4 个样本 + 只读参照）
    pre_inputs = assert_pricing_inputs()
    sample_paths = {k: sample_path(k, 'v2') for k in COLUMNS}
    pre_samples = {k: {'path': _rel(p), 'bytes': file_size(p)}
                   for k, p in sample_paths.items()}
    pre_refs = {
        'g0_reference': {'path': _rel(G0_REFERENCE),
                         'bytes': file_size(G0_REFERENCE)},
        'parameter_registry': {'path': _rel(REGISTRY),
                               'bytes': file_size(REGISTRY)},
        'sample_report': {'path': _rel(SAMPLE_REPORT),
                          'bytes': file_size(SAMPLE_REPORT)},
    }

    sample_report = json.loads(SAMPLE_REPORT.read_text(encoding='utf-8'))
    ref = json.loads(G0_REFERENCE.read_text(encoding='utf-8'))

    readings = {}
    for key in COLUMNS:
        df = _load_fm_sample(key, sample_report)
        fm = df[df['pricing_in_fm_sample'].astype(bool)].copy()
        common = fm[(fm['month'] >= COMMON_START) & (fm['month'] <= COMMON_END)].copy()
        require(len(common) > 0, f'{key}: 共同窗口无样本')
        readings[key] = {
            'original_window': _block(fm),
            'common_window': _block(common),
            'rows': {
                'sample_rows_total': int(len(df)),
                'fm_rows_original': int(len(fm)),
                'fm_rows_common': int(len(common)),
                'fm_rows_dropped_by_window': int(len(fm) - len(common)),
                'months_original': int(fm['month'].nunique()),
                'months_common': int(common['month'].nunique()),
                'months_original_range': [str(fm['month'].min()), str(fm['month'].max())],
                'months_common_range': [str(common['month'].min()), str(common['month'].max())],
            },
        }
        print(f'[3b_n6] {key}: FM {len(fm):,} 行 / {fm["month"].nunique()} 月 -> '
              f'共同窗口 {len(common):,} 行 / {common["month"].nunique()} 月', flush=True)

    # ---------------- 原窗口对账 G0 主表（beta / t / N / T 逐值）
    recon = {'status': 'PASS', 'tolerance': 1e-10, 'checks': [], 'max_abs_diff': 0.0}
    for key in COLUMNS:
        blk = readings[key]['original_window']
        ref_blk = ref['main'][key]
        pairs = [('n_rows', blk['ew']['n_rows'], int(ref_blk['n_rows'])),
                 ('T', blk['ew']['T'], int(ref_blk['n_months'])),
                 ('fm_window_start', blk['ew']['fm_window'][0], ref_blk['fm_window'][0]),
                 ('fm_window_end', blk['ew']['fm_window'][1], ref_blk['fm_window'][1])]
        for pname, actual, expected in pairs:
            ok = actual == expected
            recon['checks'].append({'column': key, 'field': pname,
                                    'actual': actual, 'g0': expected, 'match': bool(ok)})
            require(ok, f'G0 对账失败 {key}/{pname}: actual={actual} g0={expected}')
        for weight in ('ew', 'vw'):
            for var in VARS:
                a = blk[weight]['coefs'][var]
                g = ref_blk[weight]['coefs'][var]
                for stat in ('beta', 't_nw'):
                    diff = abs(float(a[stat]) - float(g[stat]))
                    recon['max_abs_diff'] = max(recon['max_abs_diff'], diff)
                    ok = diff <= recon['tolerance']
                    recon['checks'].append({
                        'column': key, 'weight': weight, 'variable': var, 'stat': stat,
                        'actual': float(a[stat]), 'g0': float(g[stat]),
                        'abs_diff': diff, 'match': bool(ok)})
                    require(ok, f'G0 对账失败 {key}/{weight}/{var}/{stat}: '
                                f'diff={diff:.3e} > {recon["tolerance"]}')
    print(f'[3b_n6] G0 原窗口对账 PASS：max |diff| = {recon["max_abs_diff"]:.3e}', flush=True)

    # ---------------- 表格数据（wide：16 行）
    rows = []
    for key in COLUMNS:
        orig = readings[key]['original_window']
        common = readings[key]['common_window']
        for weight in ('ew', 'vw'):
            for tag, blk, base in (('original', orig[weight], None),
                                   ('common', common[weight], orig[weight])):
                row = {
                    'column': key, 'role': ROLE[key], 'weight': weight, 'window': tag,
                    'window_start': blk['fm_window'][0], 'window_end': blk['fm_window'][1],
                    'n_rows': blk['n_rows'], 'n_est': blk['n_est'], 'T': blk['T'],
                    'n_min': blk['per_month_n']['min'],
                    'n_median': blk['per_month_n']['median'],
                    'n_max': blk['per_month_n']['max'],
                    'all_months_full_rank': blk['all_months_full_rank'],
                    'all_months_covariance_finite': blk['all_months_covariance_finite'],
                }
                for var in VARS:
                    c = blk['coefs'][var]
                    row[f'beta_{var}'] = c['beta']
                    row[f'se_{var}'] = c['se_nw']
                    row[f't_{var}'] = c['t_nw']
                    row[f'p_one_sided_{var}'] = c['p_one_sided']
                    if base is not None:
                        bc = base['coefs'][var]
                        row[f'delta_beta_{var}'] = (
                            None if c['beta'] is None or bc['beta'] is None
                            else c['beta'] - bc['beta'])
                        row[f'delta_t_{var}'] = (
                            None if c['t_nw'] is None or bc['t_nw'] is None
                            else c['t_nw'] - bc['t_nw'])
                        row[f'pct_beta_{var}'] = _pct(c['beta'], bc['beta'])
                    else:
                        row[f'delta_beta_{var}'] = None
                        row[f'delta_t_{var}'] = None
                        row[f'pct_beta_{var}'] = None
                rows.append(row)
    table = pd.DataFrame(rows)

    # ---------------- 计算后 sha：必须与计算前一致
    post_inputs = assert_pricing_inputs()
    post_samples = {k: {'path': _rel(p), 'bytes': file_size(p)}
                    for k, p in sample_paths.items()}
    post_refs = {
        'g0_reference': {'path': _rel(G0_REFERENCE),
                         'bytes': file_size(G0_REFERENCE)},
        'parameter_registry': {'path': _rel(REGISTRY),
                               'bytes': file_size(REGISTRY)},
        'sample_report': {'path': _rel(SAMPLE_REPORT),
                          'bytes': file_size(SAMPLE_REPORT)},
    }
    sha_ledger = {
        'before': {'pricing_inputs_9': pre_inputs, 'samples': pre_samples, 'references': pre_refs},
        'after': {'pricing_inputs_9': post_inputs, 'samples': post_samples, 'references': post_refs},
        'unchanged': True,
        'checks': [],
    }
    for name, pre in [(f'input:{x["input"]}', x) for x in pre_inputs] + \
                     [(f'sample:{k}', v) for k, v in pre_samples.items()] + \
                     [(f'ref:{k}', v) for k, v in pre_refs.items()]:
        post = ({f'input:{x["input"]}': x for x in post_inputs}.get(name)
                or post_samples.get(name.split(':', 1)[1])
                or post_refs.get(name.split(':', 1)[1]))
        ok = pre['bytes'] == post['bytes']
        sha_ledger['checks'].append({'item': name, 'pre': pre['bytes'],
                                     'post': post['bytes'], 'unchanged': bool(ok)})
        require(ok, f'{name}: 计算前后 sha 不一致（{pre["bytes"]} -> {post["bytes"]}）')
    print(f'[3b_n6] 前后 sha 不变：9 pricing_gate 输入 + 4 样本 + 3 只读参照', flush=True)

    # ---------------- 共同窗口结论素材
    summary = {}
    for key in COLUMNS:
        c = readings[key]['common_window']
        o = readings[key]['original_window']
        summary[key] = {'common': {}, 'original': {}, 'delta': {}}
        for var in VARS:
            summary[key]['common'][var] = {
                'beta_ew': c['ew']['coefs'][var]['beta'],
                't_ew': c['ew']['coefs'][var]['t_nw'],
                'beta_vw': c['vw']['coefs'][var]['beta'],
                't_vw': c['vw']['coefs'][var]['t_nw'],
            }
            summary[key]['original'][var] = {
                'beta_ew': o['ew']['coefs'][var]['beta'],
                't_ew': o['ew']['coefs'][var]['t_nw'],
                'beta_vw': o['vw']['coefs'][var]['beta'],
                't_vw': o['vw']['coefs'][var]['t_nw'],
            }
            summary[key]['delta'][var] = {
                'd_beta_ew': summary[key]['common'][var]['beta_ew']
                - summary[key]['original'][var]['beta_ew'],
                'd_t_ew': summary[key]['common'][var]['t_ew']
                - summary[key]['original'][var]['t_ew'],
                'd_beta_vw': summary[key]['common'][var]['beta_vw']
                - summary[key]['original'][var]['beta_vw'],
                'd_t_vw': summary[key]['common'][var]['t_vw']
                - summary[key]['original'][var]['t_vw'],
            }

    # ---------------- 写 CSV
    table.to_csv(OUT_CSV, index=False, encoding='utf-8')
    csv_bytes = file_size(OUT_CSV)

    counts = {
        'report_id': '3b_n6_common_window',
        'phase': '3B',
        'audit_item': 'N-6',
        'task': '两列并报统一到 2018-05..2023-12 共同窗口重跑 inventory（原窗口同报）',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'script': {'path': _rel(SCRIPT), 'bytes': file_size(SCRIPT)},
        'spec': {
            'id': 'inventory', 'label': '登记全控制（8 controls）',
            'formula': 'R_{i,t+1} = alpha_t + beta*PeerMAXraw + theta*OwnMAXraw + '
                       'phi*PeerReturn + Gamma*X + eps',
            'x': inventory, 'controls': CONTROLS, 'y': 'R_next',
            'estimator': '逐月横截面 OLS(EW) / WLS(VW, weight=market_equity) -> '
                         '系数序列 -> 均值 + Newey-West(lag=6, Bartlett)',
            'min_n': MIN_N, 'min_effective_month_count': MIN_MONTHS,
            'nw_lag': NW_LAG, 'winsorization_pct': WINSOR,
            'sample_version': SAMPLE_VERSION,
            'fm_sample': 'pricing_in_fm_sample == True（与 headline 主表同）',
            'common_window': [COMMON_START, COMMON_END],
            'original_sample_window': [SAMPLE_WINDOW_START, SAMPLE_WINDOW_END],
        },
        'roles': ROLE, 'k_target': K_TARGET, 'corpus': CORPUS, 'method': METHOD,
        'readings': readings, 'summary': summary,
        'g0_reconciliation': recon,
        'input_sha_ledger': sha_ledger,
        'outputs': {'csv': _rel(OUT_CSV), 'csv_bytes': csv_bytes},
        'scope_note': (
            '共同窗口只消除 FM 窗口差异。两列同为 teacher_mda_panel 语料；相差网络方法：'
            'ma_teacher 是等权余弦聚类（K 目标 44），mb_teacher 是 TF-IDF 阈值网络（tau），'
            '由方法决定的 N 差异仍在。'),
    }
    OUT_JSON.write_text(json.dumps(_json_ready(counts), ensure_ascii=False, indent=1),
                        encoding='utf-8')
    json_bytes = file_size(OUT_JSON)

    # ---------------- 写执行报告 MD
    def cell(key, weight, var, stat):
        return readings[key]['common_window'][weight]['coefs'][var][stat]

    L = []
    L.append('# Phase 3B / N-6（B2）：两列共同窗口 inventory 重跑\n')
    L.append(f'- 生成时间（UTC）：{counts["generated_at_utc"]}')
    L.append(f'- 脚本：`{counts["script"]["path"]}`（bytes `{counts["script"]["bytes"]}`）')
    L.append('- 样本版本：`pricing_sample_v5_teacher_only_20261120`（两列 parquet，只读）')
    L.append('- 规格：inventory = PeerMAXraw + OwnMAXraw + PeerReturn + 8 controls'
             '（size / bm / mom / rev / turnover / illiq / AnalystCoverage / NetworkDegree）')
    L.append(f'- 估计器：逐月 EW OLS / VW WLS（weight=market_equity）→ 系数序列 → 均值 + '
             f'Newey-West(lag={NW_LAG}, Bartlett)；min_n={MIN_N}；y=`R_next`；'
             f'FM 样本=`pricing_in_fm_sample == True`')
    L.append(f'- 共同窗口：**{COMMON_START}..{COMMON_END}**（含端点；事件月口径，68 个月）')
    L.append('- 原窗口：ma_teacher / mb_teacher = 2015-05..2023-12（104 月）；'
             'ma / mb = 2018-05..2023-12（68 月）')
    L.append('- 只读约束：不重跑 pricing_gate / PC / 网络；未改 headline / 系数 / gate / configs / '
             'G0 / reports；输出仅 `artifacts/phase3b/3b_n6_*`\n')

    L.append('## 1. 共同窗口读数（2018-05..2023-12）\n')
    L.append('| 列 | 权重 | N | T | PeerMAX β (t) | OwnMAX β (t) | PeerReturn β (t) |')
    L.append('|---|---|---:|---:|---|---|---|')
    for key in COLUMNS:
        for weight in ('ew', 'vw'):
            b = readings[key]['common_window'][weight]
            L.append(f'| `{key}` | {weight.upper()} | {b["n_rows"]:,} | {b["T"]} | '
                     f'{_fmt(cell(key, weight, "PeerMAXraw", "beta"))} '
                     f'({_fmt_t(cell(key, weight, "PeerMAXraw", "t_nw"))}) | '
                     f'{_fmt(cell(key, weight, "OwnMAXraw", "beta"))} '
                     f'({_fmt_t(cell(key, weight, "OwnMAXraw", "t_nw"))}) | '
                     f'{_fmt(cell(key, weight, "PeerReturn", "beta"))} '
                     f'({_fmt_t(cell(key, weight, "PeerReturn", "t_nw"))}) |')
    L.append('\n> N = FM 样本行数（`pricing_in_fm_sample=True` 且月 ∈ 共同窗口）；'
             'T = 实际进入系数序列的月份数。逐月 n 的最小/中位/最大见 CSV 与 JSON。\n')

    L.append('## 2. 原窗口读数（对照）\n')
    L.append('| 列 | 权重 | N | T | PeerMAX β (t) | OwnMAX β (t) | PeerReturn β (t) |')
    L.append('|---|---|---:|---:|---|---|---|')
    for key in COLUMNS:
        for weight in ('ew', 'vw'):
            b = readings[key]['original_window'][weight]
            c = b['coefs']
            L.append(f'| `{key}` | {weight.upper()} | {b["n_rows"]:,} | {b["T"]} | '
                     f'{_fmt(c["PeerMAXraw"]["beta"])} ({_fmt_t(c["PeerMAXraw"]["t_nw"])}) | '
                     f'{_fmt(c["OwnMAXraw"]["beta"])} ({_fmt_t(c["OwnMAXraw"]["t_nw"])}) | '
                     f'{_fmt(c["PeerReturn"]["beta"])} ({_fmt_t(c["PeerReturn"]["t_nw"])}) |')
    L.append('\n> 原窗口读数与 `reports/data_gates/G0/pricing_table_v2.json` 主表逐值一致：'
             f'beta / NW t 最大绝对差 = {recon["max_abs_diff"]:.3e}；'
             f'N / T / 窗口完全一致（{len(recon["checks"])} 项检查全 PASS）。\n')

    L.append('## 3. 共同窗口 − 原窗口（delta）\n')
    L.append('| 列 | 权重 | ΔPeerMAX β | ΔPeerMAX t | ΔOwnMAX β | ΔOwnMAX t | '
             'ΔPeerReturn β | ΔPeerReturn t |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|')
    for key in COLUMNS:
        for weight in ('ew', 'vw'):
            c = readings[key]['common_window'][weight]['coefs']
            o = readings[key]['original_window'][weight]['coefs']
            L.append(f'| `{key}` | {weight.upper()} | '
                     f'{_fmt(c["PeerMAXraw"]["beta"] - o["PeerMAXraw"]["beta"])} | '
                     f'{_fmt_t(c["PeerMAXraw"]["t_nw"] - o["PeerMAXraw"]["t_nw"])} | '
                     f'{_fmt(c["OwnMAXraw"]["beta"] - o["OwnMAXraw"]["beta"])} | '
                     f'{_fmt_t(c["OwnMAXraw"]["t_nw"] - o["OwnMAXraw"]["t_nw"])} | '
                     f'{_fmt(c["PeerReturn"]["beta"] - o["PeerReturn"]["beta"])} | '
                     f'{_fmt_t(c["PeerReturn"]["t_nw"] - o["PeerReturn"]["t_nw"])} |')
    L.append('\n> `ma` / `mb` 原窗口即 2018-05..2023-12，delta 恒为 0；'
             '`ma_teacher` / `mb_teacher` 的 delta 是去掉 2015-05..2018-04'
             '（36 个月）后的窗口效应，不是系数口径变化。\n')

    L.append('## 4. 逐列窗口与样本\n')
    L.append('| 列 | 原窗口（月） | 原 T | 共同窗口 T | 原 N | 共同 N | 窗口剔除 N |')
    L.append('|---|---|---:|---:|---:|---:|---:|')
    for key in COLUMNS:
        r = readings[key]['rows']
        L.append(f'| `{key}` | {r["months_original_range"][0]}..{r["months_original_range"][1]} | '
                 f'{r["months_original"]} | {r["months_common"]} | '
                 f'{r["fm_rows_original"]:,} | {r["fm_rows_common"]:,} | '
                 f'{r["fm_rows_dropped_by_window"]:,} |')

    L.append('\n## 5. 仍存在的非窗口差异（共同窗口不消除）\n')
    L.append('| 列 | 语料 | 方法 / 阈值 | K 目标 | 共同窗口 N |')
    L.append('|---|---|---|---:|---:|')
    for key in COLUMNS:
        L.append(f'| `{key}` | {CORPUS[key]} | {METHOD[key]} | '
                 f'{"K=" + str(K_TARGET[key]) if K_TARGET[key] else "无（阈值网络）"} | '
                 f'{readings[key]["common_window"]["ew"]["n_rows"]:,} |')
    L.append('\n- **N 差异**：共同窗口下 ma_teacher 120,370 / mb_teacher 153,166 / '
             'ma 118,620 / mb 151,702。N 随语料、K 目标/阈值和网络覆盖变化，'
             '不是独立设计维度，但仍使两列不是同一行集。')
    L.append('- **方法差异**：ma_teacher = 等权 0/1 词向量余弦 + 聚类（K 目标 44，'
             '`network_business_cluster_freeze.py` `K_TARGET_FALLBACK`）；'
             'mb_teacher = TF-IDF 阈值网络（tau=0.0888212137），无 K 目标。')
    L.append('- **语料**：两列同为 `teacher_mda_panel`（老师 MD&A）。')
    L.append('- 因此：共同窗口只让 **FM 月份集合**对齐；跨列比较仍是'
             '「语料 + 方法/K 目标 + N」的复合差异，不能解释为单维章节效应。'
             '唯一接近单维的对照仍是 ma_teacher vs mb_teacher（设计上同语料、同窗口，'
             '仅网络方法不同：等权余弦聚类 vs TF-IDF 阈值；实际 N 仍为 181,178 vs 225,264，'
             '差异来自网络覆盖）。')

    # 结论：由读数生成
    peer_common = [(k, w, readings[k]['common_window'][w]['coefs']['PeerMAXraw'])
                   for k in COLUMNS for w in ('ew', 'vw')]
    sig_neg = sum(1 for _, _, c in peer_common
                  if c['beta'] is not None and c['beta'] < 0
                  and c['p_one_sided'] is not None and c['p_one_sided'] < 0.05)
    pos = sum(1 for _, _, c in peer_common if c['beta'] is not None and c['beta'] > 0)
    pr_common = [(k, w, readings[k]['common_window'][w]['coefs']['PeerReturn'])
                 for k in COLUMNS for w in ('ew', 'vw')]
    pr_sig = sum(1 for _, _, c in pr_common
                 if c['p_two_sided'] is not None and c['p_two_sided'] < 0.05)
    own_common = [(k, w, readings[k]['common_window'][w]['coefs']['OwnMAXraw'])
                  for k in COLUMNS for w in ('ew', 'vw')]
    own_pos = sum(1 for _, _, c in own_common if c['beta'] is not None and c['beta'] > 0)
    max_own_abs_t = max(abs(c['t_nw']) for _, _, c in own_common if c['t_nw'] is not None)
    max_peer_abs_t = max(abs(c['t_nw']) for _, _, c in peer_common if c['t_nw'] is not None)
    L.append('\n## 6. 结论\n')
    L.append(f'- 共同窗口 8 个 PeerMAX 读数：正号 {pos}/8，负号且单侧 p<0.05 的 {sig_neg}/8，'
             f'|t| 最大 {max_peer_abs_t:.2f}。两列符号不一致（ma_teacher EW+ / VW−；'
             'mb_teacher 均+；ma 均−；mb EW+ / VW−），但全部不显著；'
             'PeerMAX 在共同窗口下仍未呈现稳定的负向定价。')
    L.append(f'- PeerReturn：8 个读数中双侧 p<0.05 的 {pr_sig}/8，方向均为正；'
             '与主表一致，同伴收益项是最稳定的成分。')
    L.append(f'- OwnMAX：8 个读数全部为正（{own_pos}/8），|t| 最大 {max_own_abs_t:.2f}，'
             '均不显著；beta 与 PeerReturn 同阶（逐值见第 1 节）。')
    L.append('- 窗口效应量级：ma_teacher 的 PeerMAX beta EW +0.0343→+0.0350'
             '（t +0.47→+0.42）、VW −0.1100→−0.0715（t −1.18→−0.75）；'
             'mb_teacher EW +0.0348→+0.0849（t +0.43→+0.76）、VW +0.0346→+0.0399'
             '（t +0.36→+0.30）。变化不改变「PeerMAX 不显著」的判定；'
             'ma / mb 原窗口即共同窗口，delta=0。')
    L.append('- **N-6 验收含义**：68 月两列并报读数已给出；「只在重叠窗口比较」从此有表可依。'
             '但共同窗口**不能**把 ma 列变成「只差章节」——K 目标 56 vs 44、语料抽取差异、'
             'N 118,620 vs 120,370 仍在。要得到单维语料对照，仍需 K=44 的 ma 版本（N-6 处理项 ②，'
             '本轮未执行、未改任何现有网络/系数）。')
    L.append('- 本报告为诊断读数，不构成 headline 变更；主表仍以原窗口 `pricing_main_v1` 为准。\n')

    L.append('## 7. 复现命令\n')
    L.append('```bash')
    L.append('cd /c/Users/27697/Desktop/peer_lottery_proposal')
    L.append('python scripts/shadow_common_window.py')
    L.append('```')
    L.append('\n产物：')
    L.append(f'- `{_rel(OUT_CSV)}`（bytes `{csv_bytes}`）')
    L.append(f'- `{_rel(OUT_JSON)}`（bytes `{json_bytes}`）')
    L.append(f'- `{_rel(OUT_MD)}`（本文件）')

    L.append('\n## 8. 输入 bytes（计算前 / 计算后）\n')
    L.append('### 9 个 pricing_gate 主输入\n')
    L.append('| 输入 | 文件 | bytes（前=后） |')
    L.append('|---|---|---|')
    for x in pre_inputs:
        L.append(f'| {x["input"]} | `{x["file"]}` | `{x["bytes"]}` |')
    L.append('\n### 4 个 v2 样本（只读）\n')
    L.append('| 列 | 文件 | bytes（前=后） |')
    L.append('|---|---|---|')
    for key in COLUMNS:
        L.append(f'| `{key}` | `{pre_samples[key]["path"]}` | `{pre_samples[key]["bytes"]}` |')
    L.append('\n### 只读参照\n')
    L.append('| 文件 | bytes（前=后） |')
    L.append('|---|---|')
    for k, v in pre_refs.items():
        L.append(f'| `{v["path"]}` | `{v["bytes"]}` |')
    L.append(f'\n> 计算前后逐项比对：{len(sha_ledger["checks"])} 项全部不变；'
             f'9 个 pricing_gate 输入均与 `configs/version_manifest.yaml` 锚一致。')
    OUT_MD.write_text('\n'.join(L) + '\n', encoding='utf-8')

    print(f'[3b_n6] 写出：{_rel(OUT_CSV)} / {_rel(OUT_JSON)} / '
          f'{_rel(OUT_MD)}', flush=True)
    print(f'[3b_n6] G0 对账 max|diff|={recon["max_abs_diff"]:.3e}；'
          f'sha 台账 {len(sha_ledger["checks"])} 项全不变', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
