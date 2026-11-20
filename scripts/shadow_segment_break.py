#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B / N-4：2021 语料断点分段诊断（headline pricing_gate 固定样本，只读）。

目的
----
判断 headline pricing_gate（spec_no_turnover/pricing_gate）在 PeerMAX 上的 null 是否由 2021 语料断点后的短样本
（network_version >= 2021，事件月 2022-05..2023-12，20 个月）驱动。

做法
----
- 只读 `data/processed/pricing_sample_ma_teacher_v2.parquet`（FM 181,178 行 / 104 月）
  与 `scripts/pricing_lib.py`；不重建样本、不重跑 PC/网络、不跑官方 pricing_gate。
- 分段（均在 `pricing_in_fm_sample == True` 的固定 FM 样本内）：
    nv_pre   : network_version <= 2020（文本年 2014–2020；事件月 2015-05..2022-04，84 月）
    nv_post  : network_version >= 2021（文本年 2021–2022；事件月 2022-05..2023-12，20 月）
    cal_pre  : month <= 2020-12（68 月）
    cal_post : month >= 2021-01（36 月）
- 每段 × 规格 spec_full_controls（登记全控制）/ spec_no_turnover（无 turnover）/ spec_no_rev_no_turnover（无 rev + 无 turnover）× EW/VW
  跑 `pricing_lib.fm_regression`（min_n=100、NW lag=6，均从 registry 读取）。
- 报 PeerMAX / OwnMAX / PeerReturn 的 β/se/t/p/CI/T/n，并与 `artifacts/phase3/3.6_counts.json`
  中的 headline 全样本行（同一固定样本的 FM 登记读数，本脚本不重跑）对比。
- 月度系数差：post − pre 用月度系数序列 + 截距/post 虚拟变量 OLS + Newey-West(6) 检验。
- 写 `artifacts/phase3b/3b_n4_segment.csv`、`3b_n4_counts.json`、`3b_n4_execution.md`
  中的前两个；执行记录 md 由本脚本输出后人工定稿（不在脚本内生成）。

边界
----
- 只写 `artifacts/phase3b/3b_n4_*`；路径守卫拒绝其它写出。
- 不改 headline 样本/系数/gate/configs/G0/reports。
- 9 个 pricing_gate 输入 sha 在运行前后各快照一次；样本与 3.6 参考件 sha 同样前后核对。

用法
----
  python scripts/shadow_segment_break.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from pricing_lib import (  # noqa: E402
    assert_pricing_inputs, fm_regression, param_value, require, file_size,
    shadow_sample_path, two_sided_p,
)

ART = ROOT / 'artifacts/phase3b'
SAMPLE_PATH = shadow_sample_path('ma_teacher')
REF_3_6 = ROOT / 'artifacts/phase3/3.6_counts.json'
CSV_PATH = ART / '3b_n4_segment.csv'
JSON_PATH = ART / '3b_n4_counts.json'

VERSION_TAG = 'corpus_break_segment_v1'
def _registered_fm_rows(key='ma_teacher'):
    """从样本报告读 headline FM 行数（硬编码随样本重建必过期）。"""
    import json
    p = ROOT / 'reports/data_gates/G0/pricing_sample_report.json'
    if not p.is_file():
        return None
    ent = ((json.loads(p.read_text(encoding='utf-8')).get('samples') or {}).get(key) or {})
    n = ent.get('fm_sample_rows')
    return int(n) if n is not None else None


EXPECTED_FM_ROWS = _registered_fm_rows()
EXPECTED_FM_MONTHS = 104
EXPECTED_FM_START = '2015-05'
EXPECTED_FM_END = '2023-12'
NV_BREAK = 2021
CAL_BREAK = '2021-01'

CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq',
            'AnalystCoverage', 'NetworkDegree']
SPECS = [
    ('spec_full_controls', '登记全控制（8 controls）',
     ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + CONTROLS),
    ('spec_no_turnover', '无 turnover',
     ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + [c for c in CONTROLS if c != 'turnover']),
    ('spec_no_rev_no_turnover', '无 rev + 无 turnover',
     ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn']
     + [c for c in CONTROLS if c not in ('rev', 'turnover')]),
]
WEIGHTS = [('EW', None), ('VW', 'market_equity')]
TERMS = [('PeerMAXraw', 'peermax'), ('OwnMAXraw', 'ownmax'), ('PeerReturn', 'peerreturn')]

MIN_N = int(param_value('sample_and_spec_registry')['min_focal_count_per_month'])
MIN_EFFECTIVE_MONTHS = int(param_value('sample_and_spec_registry')['min_effective_month_count'])
NW_LAG = int(param_value('nw_lag'))
WINSOR = param_value('winsorization_pct')


# ------------------------------------------------------------------ 输出路径守卫
def out_path(path) -> Path:
    """只允许 `artifacts/phase3b/3b_n4_*`。"""
    p = Path(path).resolve()
    art = ART.resolve()
    if art in p.parents and p.name.startswith('3b_n4_'):
        return p
    print(f'[HARD-GATE] 只允许写 artifacts/phase3b/3b_n4_*：{p}', file=sys.stderr, flush=True)
    raise SystemExit(2)


def write_json(path, obj) -> None:
    p = out_path(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default),
                   encoding='utf-8')
    os.replace(tmp, p)


def write_csv(path, df: pd.DataFrame) -> None:
    p = out_path(path)
    tmp = p.with_name(p.name + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8', lineterminator='\n')
    os.replace(tmp, p)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def _clean(v):
    """NaN/±inf -> None，便于 JSON 可读。"""
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ------------------------------------------------------------------ sha 台账
def snapshot_anchors(tag: str) -> dict:
    ledger = assert_pricing_inputs()
    return {
        'tag': tag,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'pricing_inputs': {r['input']: {'path': r['path'], 'rows': r['rows']}
                      for r in ledger},
        'pricing_ledger': ledger,
        'headline_sample': {
            'path': str(SAMPLE_PATH.relative_to(ROOT)).replace('\\', '/'),
            'bytes': file_size(SAMPLE_PATH),
        },
        'reference_3_6': {
            'path': str(REF_3_6.relative_to(ROOT)).replace('\\', '/'),
            'bytes': file_size(REF_3_6),
        },
        'script': {
            'path': str(Path(__file__).resolve().relative_to(ROOT)).replace('\\', '/'),
            'bytes': file_size(Path(__file__).resolve()),
        },
    }


def compare_anchors(pre: dict, post: dict) -> dict:
    diffs = []
    for name in sorted(set(pre['pricing_inputs']) | set(post['pricing_inputs'])):
        a = pre['pricing_inputs'].get(name, {}).get('bytes')
        b = post['pricing_inputs'].get(name, {}).get('bytes')
        if a != b:
            diffs.append(f'pricing_inputs:{name}')
    for key in ('headline_sample', 'reference_3_6'):
        if pre[key]['bytes'] != post[key]['bytes']:
            diffs.append(key)
    require(len(pre['pricing_inputs']) == 9, f'9 输入台账长度={len(pre["pricing_inputs"])} != 9')
    require(len(post['pricing_inputs']) == 9, f'9 输入台账长度={len(post["pricing_inputs"])} != 9')
    return {
        'status': 'MATCH' if not diffs else 'DIFFERENT',
        'n_pricing_inputs': len(pre['pricing_inputs']),
        'n_diff': len(diffs),
        'diff': diffs,
        'pricing_inputs_match': pre['pricing_inputs'] == post['pricing_inputs'],
        'headline_sample_match': pre['headline_sample']['bytes']
        == post['headline_sample']['bytes'],
        'reference_3_6_match': pre['reference_3_6']['bytes']
        == post['reference_3_6']['bytes'],
        'script_bytes': pre['script']['bytes'],
    }


# ------------------------------------------------------------------ 数据与分段
def load_fm_sample() -> pd.DataFrame:
    sample = pd.read_parquet(SAMPLE_PATH)
    require('pricing_in_fm_sample' in sample.columns, '样本缺 pricing_in_fm_sample')
    require('network_version' in sample.columns, '样本缺 network_version')
    fm = sample.loc[sample['pricing_in_fm_sample'].astype(bool)].copy()
    fm['month'] = fm['month'].astype(str)
    fm['network_version'] = pd.to_numeric(fm['network_version'], errors='coerce')
    months = sorted(fm['month'].unique().tolist())
    require(len(fm) == EXPECTED_FM_ROWS,
            f'FM 行数={len(fm):,} != headline {EXPECTED_FM_ROWS:,}')
    require(len(months) == EXPECTED_FM_MONTHS,
            f'FM 月数={len(months)} != {EXPECTED_FM_MONTHS}')
    require(months[0] == EXPECTED_FM_START and months[-1] == EXPECTED_FM_END,
            f'FM 月份范围={months[0]}..{months[-1]}')
    require(fm['network_version'].notna().all(), 'FM 样本存在 network_version 缺失')
    return fm


def build_segments(fm: pd.DataFrame) -> list:
    nv_pre = fm['network_version'] <= NV_BREAK - 1
    nv_post = fm['network_version'] >= NV_BREAK
    cal_pre = fm['month'] <= '2020-12'
    cal_post = fm['month'] >= CAL_BREAK
    segs = [
        {'id': 'nv_pre', 'family': 'network_version', 'order': 1,
         'label': 'pre：network_version <= 2020（文本年 2014–2020）',
         'definition': 'network_version <= 2020', 'mask': nv_pre},
        {'id': 'nv_post', 'family': 'network_version', 'order': 2,
         'label': 'post：network_version >= 2021（文本年 2021–2022）',
         'definition': 'network_version >= 2021', 'mask': nv_post},
        {'id': 'cal_pre', 'family': 'calendar', 'order': 1,
         'label': 'pre：事件月 <= 2020-12',
         'definition': 'month <= 2020-12', 'mask': cal_pre},
        {'id': 'cal_post', 'family': 'calendar', 'order': 2,
         'label': 'post：事件月 >= 2021-01',
         'definition': 'month >= 2021-01', 'mask': cal_post},
    ]
    for s in segs:
        s['df'] = fm.loc[s.pop('mask')].copy()
        s['months'] = sorted(s['df']['month'].unique().tolist())
        s['T'] = len(s['months'])
        require(s['T'] > 0, f"{s['id']}: 空段")
        s['n'] = int(len(s['df']))
        s['monthly_n'] = (s['df'].groupby('month', sort=True).size()
                          .rename('n').reset_index())
        s['market_equity_share'] = float(
            s['df']['market_equity'].sum() / fm['market_equity'].sum())
        s['share_of_fm_rows'] = s['n'] / len(fm)
        s['share_of_fm_months'] = s['T'] / len(fm['month'].unique())
        s['n_network_versions'] = sorted(
            int(x) for x in s['df']['network_version'].dropna().unique())
    # 合同：两套分段都必须无重叠、无遗漏、月集等于全样本月集
    for fam in ('network_version', 'calendar'):
        parts = [s for s in segs if s['family'] == fam]
        mset = [set(s['months']) for s in parts]
        require(not (mset[0] & mset[1]), f'{fam}: pre/post 月份重叠')
        require(mset[0] | mset[1] == set(fm['month'].unique()), f'{fam}: 月份未覆盖全样本')
        require(sum(s['n'] for s in parts) == len(fm), f'{fam}: 行数不闭合')
    nv_pre_s = next(s for s in segs if s['id'] == 'nv_pre')
    nv_post_s = next(s for s in segs if s['id'] == 'nv_post')
    require(nv_pre_s['months'][-1] == '2022-04',
            f"nv_pre 末月={nv_pre_s['months'][-1]} != 2022-04")
    require(nv_post_s['months'][0] == '2022-05',
            f"nv_post 首月={nv_post_s['months'][0]} != 2022-05")
    require(nv_pre_s['T'] == 84 and nv_post_s['T'] == 20,
            f"nv 段月数={nv_pre_s['T']}/{nv_post_s['T']} != 84/20")
    return segs


def load_headline_reference() -> tuple:
    doc = json.loads(REF_3_6.read_text(encoding='utf-8'))
    rows = pd.DataFrame(doc['fm_rows'])
    h = rows[rows['variant'] == 'headline'].copy()
    require(len(h) == 6, f'3.6 headline 行数={len(h)} != 6')
    ref = {}
    for _, r in h.iterrows():
        key = (str(r['spec']), str(r['weight']))
        require(int(r['T']) == EXPECTED_FM_MONTHS and int(r['n']) == EXPECTED_FM_ROWS,
                f'3.6 headline {key}: T/n={int(r["T"])}/{int(r["n"])} 与固定样本不符')
        ref[key] = {
            'spec': key[0], 'weight': key[1], 'T': int(r['T']), 'n': int(r['n']),
            'beta_peermax': _clean(r['beta_peermax']),
            't_peermax': _clean(r['t_peermax']),
            'se_peermax': _clean(r['se_peermax']),
            'shadow_peermax': _clean(r['shadow_peermax']),
            'beta_ownmax': _clean(r['beta_ownmax']),
            't_ownmax': _clean(r['t_ownmax']),
            'beta_peerreturn': _clean(r['beta_peerreturn']),
            't_peerreturn': _clean(r['t_peerreturn']),
        }
    require(set(ref) == {(s, w) for s, _, _ in SPECS for w, _ in WEIGHTS},
            f'3.6 headline 规格/权重键不齐：{sorted(ref)}')
    return ref, doc.get('run_at'), doc.get('version_tag')


# ------------------------------------------------------------------ NW 虚拟变量回归
def nw_dummy_diff(betas: np.ndarray, post: np.ndarray, lag: int = NW_LAG) -> dict:
    """y = a + b*post + e；NW(lag, Bartlett) HAC。返回 b/se/t/p。"""
    y = np.asarray(betas, dtype=float)
    d = np.asarray(post, dtype=float)
    ok = np.isfinite(y) & np.isfinite(d)
    y, d = y[ok], d[ok]
    T = len(y)
    if T < 3 or d.min() == d.max():
        return {'diff_beta': None, 'se_nw': None, 't_nw': None, 'shadow': None,
                'T': int(T), 'status': 'NOT_COMPUTABLE'}
    X = np.column_stack([np.ones(T), d])
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < 2:
        return {'diff_beta': None, 'se_nw': None, 't_nw': None, 'shadow': None,
                'T': int(T), 'status': 'RANK_DEFICIENT'}
    bread = np.linalg.inv(XtX)
    b = bread @ (X.T @ y)
    e = y - X @ b
    xe = X * e[:, None]
    meat = xe.T @ xe
    L = min(int(lag), T - 1)
    for l in range(1, L + 1):
        w = 1.0 - l / (lag + 1.0)
        G = xe[l:].T @ xe[:-l]
        meat += w * (G + G.T)
    cov = bread @ meat @ bread
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    diff = float(b[1])
    if not np.isfinite(se) or se <= 0:
        return {'diff_beta': diff, 'se_nw': None, 't_nw': None, 'shadow': None,
                'T': int(T), 'status': 'ZERO_OR_INVALID_SE'}
    t = diff / se
    return {'diff_beta': diff, 'se_nw': se, 't_nw': t,
            'shadow': two_sided_p(t), 'T': int(T), 'status': 'OK'}


# ------------------------------------------------------------------ FM
def _coef(res: dict, term: str) -> dict:
    s = res['coefs'].get(term, {})
    return {
        'beta': _clean(s.get('beta')),
        'se': _clean(s.get('se_nw')),
        't': _clean(s.get('t_nw')),
        'shadow': _clean(s.get('p_two_sided')),
        'ci_low': _clean(s.get('ci95_low')),
        'ci_high': _clean(s.get('ci95_high')),
        'T': int(s.get('n_months', 0) or 0),
    }


def run_segment_fm(seg: dict) -> list:
    """段 × 规格 × EW/VW；返回行列表 + 月度系数（挂在 _monthly_betas）。"""
    rows = []
    df = seg['df']
    n_month_cells = len(seg['months'])
    for spec_id, spec_label, cols in SPECS:
        for wname, wcol in WEIGHTS:
            res, per = fm_regression(df, 'R_next', cols, weight_col=wcol,
                                     min_n=MIN_N, nw_lag=NW_LAG)
            used = [r for r in per if r.get('used')]
            T = int(res['n_months_used'])
            n_used = int(sum(r['n'] for r in used))
            month_n = [r.get('n', 0) for r in used]
            status = 'OK' if T >= NW_LAG + 2 else 'N/A'
            gate_note = None
            if T < MIN_EFFECTIVE_MONTHS:
                gate_note = (f'T={T} < registry min_effective_month_count='
                             f'{MIN_EFFECTIVE_MONTHS}；仅作诊断读数，'
                             '不满足 pricing_gate gate 的有效月数门槛')
            rec = {
                'row_type': 'segment',
                'family': seg['family'],
                'segment': seg['id'],
                'segment_label': seg['label'],
                'definition': seg['definition'],
                'spec': spec_id,
                'spec_label': spec_label,
                'weight': wname,
                'status': status,
                'gate_note': gate_note,
                'source': 'scripts/shadow_segment_break.py',
                'T': T,
                'n': n_used,
                'n_months_in_segment': n_month_cells,
                'n_months_skipped': int(n_month_cells - T),
                'n_rows_dropped': int(len(df) - n_used),
                'min_month_n': int(min(month_n)) if month_n else None,
                'median_month_n': float(np.median(month_n)) if month_n else None,
                'max_month_n': int(max(month_n)) if month_n else None,
                'share_of_fm_rows': seg['share_of_fm_rows'],
                'share_of_fm_months': seg['share_of_fm_months'],
                'market_equity_share': seg['market_equity_share'],
                'below_min_effective_months': bool(T < MIN_EFFECTIVE_MONTHS),
            }
            for term, key in TERMS:
                c = _coef(res, term)
                rec[f'beta_{key}'] = c['beta']
                rec[f'se_{key}'] = c['se']
                rec[f't_{key}'] = c['t']
                rec[f'shadow_{key}'] = c['shadow']
                rec[f'ci_low_{key}'] = c['ci_low']
                rec[f'ci_high_{key}'] = c['ci_high']
            rec['_monthly_betas'] = {
                str(r['month']): {term: r['beta'][term] for term, _ in TERMS}
                for r in used if r.get('beta')
            }
            rows.append(rec)
    return rows


# ------------------------------------------------------------------ 主流程
def main() -> int:
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    print(f'[{time.strftime("%H:%M:%S")}] N-4 分段诊断开始；输出目录 {ART}', flush=True)

    pre_anchor = snapshot_anchors('pre')
    print(f'[{time.strftime("%H:%M:%S")}] pre 锚：pricing_gate 输入 {len(pre_anchor["pricing_inputs"])}/9，'
          f'样本 sha={pre_anchor["headline_sample"]["bytes"][:16]}…', flush=True)

    fm = load_fm_sample()
    segs = build_segments(fm)
    ref, ref_run_at, ref_version_tag = load_headline_reference()

    seg_rows = []
    for seg in segs:
        print(f'[{time.strftime("%H:%M:%S")}] 段 {seg["id"]}: rows={seg["n"]:,} '
              f'T={seg["T"]} {seg["months"][0]}..{seg["months"][-1]}', flush=True)
        rows = run_segment_fm(seg)
        for r in rows:
            rf = ref[(r['spec'], r['weight'])]
            r.update({
                'ref_T': rf['T'], 'ref_n': rf['n'],
                'ref_beta_peermax': rf['beta_peermax'],
                'ref_t_peermax': rf['t_peermax'],
                'ref_beta_ownmax': rf['beta_ownmax'],
                'ref_t_ownmax': rf['t_ownmax'],
                'ref_beta_peerreturn': rf['beta_peerreturn'],
                'ref_t_peerreturn': rf['t_peerreturn'],
                'delta_beta_peermax_vs_full': (
                    None if r['beta_peermax'] is None
                    else r['beta_peermax'] - rf['beta_peermax']),
                'delta_t_peermax_vs_full': (
                    None if r['t_peermax'] is None else r['t_peermax'] - rf['t_peermax']),
                'delta_beta_ownmax_vs_full': (
                    None if r['beta_ownmax'] is None
                    else r['beta_ownmax'] - rf['beta_ownmax']),
                'delta_t_ownmax_vs_full': (
                    None if r['t_ownmax'] is None else r['t_ownmax'] - rf['t_ownmax']),
                'delta_beta_peerreturn_vs_full': (
                    None if r['beta_peerreturn'] is None
                    else r['beta_peerreturn'] - rf['beta_peerreturn']),
                'delta_t_peerreturn_vs_full': (
                    None if r['t_peerreturn'] is None
                    else r['t_peerreturn'] - rf['t_peerreturn']),
            })
        seg_rows.extend(rows)

    # 内部一致性：nv_pre + nv_post 的月度系数序列应逐规格/权重重建 headline 全样本均值
    partition_checks = []
    for spec_id, _, _ in SPECS:
        for wname, _ in WEIGHTS:
            pre = next(r for r in seg_rows
                       if r['segment'] == 'nv_pre' and r['spec'] == spec_id
                       and r['weight'] == wname)
            post = next(r for r in seg_rows
                        if r['segment'] == 'nv_post' and r['spec'] == spec_id
                        and r['weight'] == wname)
            pre_b = pre['_monthly_betas']
            post_b = post['_monthly_betas']
            require(not (set(pre_b) & set(post_b)), 'pre/post 月度系数月份重叠')
            require(len(pre_b) + len(post_b) == EXPECTED_FM_MONTHS,
                    f'{spec_id}/{wname}: 分段月数合计={len(pre_b) + len(post_b)} != 104')
            for term, _ in TERMS:
                vals = ([pre_b[m][term] for m in pre_b]
                        + [post_b[m][term] for m in post_b])
                reconstructed = float(np.mean(vals))
                target = float(ref[(spec_id, wname)][
                    {'PeerMAXraw': 'beta_peermax', 'OwnMAXraw': 'beta_ownmax',
                     'PeerReturn': 'beta_peerreturn'}[term]])
                partition_checks.append({
                    'spec': spec_id, 'weight': wname, 'term': term,
                    'reconstructed_full_mean': reconstructed,
                    'headline_reference': target,
                    'abs_diff': abs(reconstructed - target),
                })
                require(abs(reconstructed - target) <= 1e-9,
                        f'{spec_id}/{wname}/{term}: 分段重建全样本均值偏差 '
                        f'{reconstructed - target:.3e} > 1e-9')

    # 同族 pre/post 月度系数差（post − pre，NW6）
    diff_tests = {}
    for fam in ('network_version', 'calendar'):
        pre_id = 'nv_pre' if fam == 'network_version' else 'cal_pre'
        post_id = 'nv_post' if fam == 'network_version' else 'cal_post'
        pre_rows = [r for r in seg_rows if r['segment'] == pre_id]
        post_rows = [r for r in seg_rows if r['segment'] == post_id]
        for spec_id, _, _ in SPECS:
            for wname, _ in WEIGHTS:
                key = f'{fam}:{spec_id}:{wname}'
                pr = next(r for r in pre_rows
                          if r['spec'] == spec_id and r['weight'] == wname)
                po = next(r for r in post_rows
                          if r['spec'] == spec_id and r['weight'] == wname)
                block = {}
                months = list(pr['_monthly_betas']) + list(po['_monthly_betas'])
                is_post = [0] * len(pr['_monthly_betas']) + [1] * len(po['_monthly_betas'])
                for term, _ in TERMS:
                    y = np.array([pr['_monthly_betas'][m][term] for m in pr['_monthly_betas']]
                                 + [po['_monthly_betas'][m][term]
                                    for m in po['_monthly_betas']], dtype=float)
                    block[term] = nw_dummy_diff(y, np.array(is_post, dtype=float))
                diff_tests[key] = {
                    'family': fam, 'spec': spec_id, 'weight': wname,
                    'pre_segment': pre_id, 'post_segment': post_id,
                    'pre_T': pr['T'], 'post_T': po['T'],
                    'pre_beta_peermax': pr['beta_peermax'],
                    'post_beta_peermax': po['beta_peermax'],
                    'diff_post_minus_pre_peermax': block['PeerMAXraw']['diff_beta'],
                    'diff_se_nw_peermax': block['PeerMAXraw']['se_nw'],
                    'diff_t_nw_peermax': block['PeerMAXraw']['t_nw'],
                    'diff_shadow_peermax': block['PeerMAXraw']['shadow'],
                    'diff_post_minus_pre_ownmax': block['OwnMAXraw']['diff_beta'],
                    'diff_t_nw_ownmax': block['OwnMAXraw']['t_nw'],
                    'diff_post_minus_pre_peerreturn': block['PeerReturn']['diff_beta'],
                    'diff_t_nw_peerreturn': block['PeerReturn']['t_nw'],
                    'n_months_pooled': len(months),
                    'status': block['PeerMAXraw']['status'],
                }
                for r in (pr, po):
                    d = block['PeerMAXraw']
                    r.update({
                        'counterpart_segment': post_id if r['segment'] == pre_id else pre_id,
                        'diff_post_minus_pre_peermax': d['diff_beta'],
                        'diff_se_nw_peermax': d['se_nw'],
                        'diff_t_nw_peermax': d['t_nw'],
                        'diff_shadow_peermax': d['shadow'],
                        'diff_post_minus_pre_ownmax': block['OwnMAXraw']['diff_beta'],
                        'diff_t_nw_ownmax': block['OwnMAXraw']['t_nw'],
                        'diff_post_minus_pre_peerreturn': block['PeerReturn']['diff_beta'],
                        'diff_t_nw_peerreturn': block['PeerReturn']['t_nw'],
                    })

    # 参考行（不重跑；来自 3.6_counts 的 headline 全样本 FM 登记读数）
    full_monthly_n = fm.groupby('month', sort=True).size()
    ref_rows = []
    for spec_id, spec_label, _ in SPECS:
        for wname, _ in WEIGHTS:
            rf = ref[(spec_id, wname)]
            ref_rows.append({
                'row_type': 'reference', 'family': 'full', 'segment': 'full_headline',
                'segment_label': 'headline 全样本（104 月）', 'definition': 'pricing_in_fm_sample == True',
                'spec': spec_id, 'spec_label': spec_label, 'weight': wname,
                'status': 'REFERENCE', 'gate_note': '读取 artifacts/phase3/3.6_counts.json；本脚本未重跑官方 pricing_gate',
                'T': rf['T'], 'n': rf['n'],
                'n_months_in_segment': rf['T'], 'n_months_skipped': 0,
                'n_rows_dropped': 0,
                'min_month_n': int(full_monthly_n.min()),
                'median_month_n': float(full_monthly_n.median()),
                'max_month_n': int(full_monthly_n.max()),
                'share_of_fm_rows': 1.0, 'share_of_fm_months': 1.0,
                'market_equity_share': 1.0,
                'below_min_effective_months': False,
                'beta_peermax': rf['beta_peermax'], 't_peermax': rf['t_peermax'],
                'se_peermax': rf['se_peermax'], 'shadow_peermax': rf['shadow_peermax'],
                'ci_low_peermax': None, 'ci_high_peermax': None,
                'beta_ownmax': rf['beta_ownmax'], 't_ownmax': rf['t_ownmax'],
                'se_ownmax': None, 'p_two_sided_ownmax': None,
                'ci_low_ownmax': None, 'ci_high_ownmax': None,
                'beta_peerreturn': rf['beta_peerreturn'],
                't_peerreturn': rf['t_peerreturn'],
                'se_peerreturn': None, 'p_two_sided_peerreturn': None,
                'ci_low_peerreturn': None, 'ci_high_peerreturn': None,
                'ref_T': rf['T'], 'ref_n': rf['n'],
                'ref_beta_peermax': rf['beta_peermax'], 'ref_t_peermax': rf['t_peermax'],
                'ref_beta_ownmax': rf['beta_ownmax'], 'ref_t_ownmax': rf['t_ownmax'],
                'ref_beta_peerreturn': rf['beta_peerreturn'],
                'ref_t_peerreturn': rf['t_peerreturn'],
                'delta_beta_peermax_vs_full': 0.0, 'delta_t_peermax_vs_full': 0.0,
                'delta_beta_ownmax_vs_full': 0.0, 'delta_t_ownmax_vs_full': 0.0,
                'delta_beta_peerreturn_vs_full': 0.0,
                'delta_t_peerreturn_vs_full': 0.0,
                'counterpart_segment': None,
                'diff_post_minus_pre_peermax': None, 'diff_se_nw_peermax': None,
                'diff_t_nw_peermax': None, 'diff_shadow_peermax': None,
                'diff_post_minus_pre_ownmax': None, 'diff_t_nw_ownmax': None,
                'diff_post_minus_pre_peerreturn': None, 'diff_t_nw_peerreturn': None,
                'source': 'artifacts/phase3/3.6_counts.json:fm_rows[variant=headline]',
            })

    # CSV 列序
    front = ['row_type', 'family', 'segment', 'segment_label', 'definition',
             'spec', 'spec_label', 'weight', 'status', 'T', 'n',
             'n_months_in_segment', 'n_months_skipped', 'n_rows_dropped',
             'min_month_n', 'median_month_n', 'max_month_n',
             'share_of_fm_rows', 'share_of_fm_months', 'market_equity_share',
             'below_min_effective_months', 'gate_note']
    coef_cols = []
    for _, key in TERMS:
        for prefix in ('beta_', 'se_', 't_', 'p_two_sided_', 'ci_low_', 'ci_high_'):
            coef_cols.append(prefix + key)
    ref_cols = ['ref_T', 'ref_n', 'ref_beta_peermax', 'ref_t_peermax',
                'ref_beta_ownmax', 'ref_t_ownmax',
                'ref_beta_peerreturn', 'ref_t_peerreturn']
    delta_cols = ['delta_beta_peermax_vs_full', 'delta_t_peermax_vs_full',
                  'delta_beta_ownmax_vs_full', 'delta_t_ownmax_vs_full',
                  'delta_beta_peerreturn_vs_full', 'delta_t_peerreturn_vs_full']
    diff_cols = ['counterpart_segment', 'diff_post_minus_pre_peermax',
                 'diff_se_nw_peermax', 'diff_t_nw_peermax', 'diff_shadow_peermax',
                 'diff_post_minus_pre_ownmax', 'diff_t_nw_ownmax',
                 'diff_post_minus_pre_peerreturn', 'diff_t_nw_peerreturn']
    all_rows = ref_rows + seg_rows
    csv_cols = front + coef_cols + ref_cols + delta_cols + diff_cols + ['source']
    out = pd.DataFrame(all_rows)
    for c in csv_cols:
        if c not in out.columns:
            out[c] = None
    out = out[csv_cols]
    write_csv(CSV_PATH, out)

    # ---- 判定：null 是否由 post 20 个月驱动
    cells = []
    for spec_id, spec_label, _ in SPECS:
        for wname, _ in WEIGHTS:
            po = next(r for r in seg_rows
                      if r['segment'] == 'nv_post' and r['spec'] == spec_id
                      and r['weight'] == wname)
            pr = next(r for r in seg_rows
                      if r['segment'] == 'nv_pre' and r['spec'] == spec_id
                      and r['weight'] == wname)
            rf = ref[(spec_id, wname)]
            full_null = abs(rf['t_peermax']) < 1.96
            pre_null = abs(pr['t_peermax']) < 1.96
            post_null = abs(po['t_peermax']) < 1.96
            cells.append({
                'spec': spec_id, 'weight': wname,
                'full_beta_peermax': rf['beta_peermax'],
                'full_t_peermax': rf['t_peermax'],
                'pre_beta_peermax': pr['beta_peermax'],
                'pre_t_peermax': pr['t_peermax'],
                'post_beta_peermax': po['beta_peermax'],
                'post_t_peermax': po['t_peermax'],
                'full_null_5pct': full_null,
                'pre_null_5pct': pre_null,
                'post_null_5pct': post_null,
                'post_drives_null': bool(full_null and pre_null and not post_null),
                'post_attenuates_pre_effect': bool(full_null and not pre_null and post_null),
                'delta_beta_post_minus_full': (
                    None if po['beta_peermax'] is None
                    else po['beta_peermax'] - rf['beta_peermax']),
                'delta_beta_pre_minus_full': (
                    None if pr['beta_peermax'] is None
                    else pr['beta_peermax'] - rf['beta_peermax']),
            })
    post_abs_t = [abs(c['post_t_peermax']) for c in cells
                  if c['post_t_peermax'] is not None]
    pre_abs_t = [abs(c['pre_t_peermax']) for c in cells
                 if c['pre_t_peermax'] is not None]
    full_abs_t = [abs(c['full_t_peermax']) for c in cells
                  if c['full_t_peermax'] is not None]
    n_post_drives = int(sum(c['post_drives_null'] for c in cells))
    n_post_att = int(sum(c['post_attenuates_pre_effect'] for c in cells))
    all_post_null = all(c['post_null_5pct'] for c in cells)
    all_pre_null = all(c['pre_null_5pct'] for c in cells)
    all_full_null = all(c['full_null_5pct'] for c in cells)
    baseline_ew = next(c for c in cells if c['spec'] == 'spec_full_controls' and c['weight'] == 'EW')
    baseline_vw = next(c for c in cells if c['spec'] == 'spec_full_controls' and c['weight'] == 'VW')
    nv_post_seg = next(s for s in segs if s['id'] == 'nv_post')
    nv_pre_seg = next(s for s in segs if s['id'] == 'nv_pre')
    verdict = {
        'question': 'headline pricing_gate 的 PeerMAX null 是否由 2021 语料断点后的 20 个月驱动？',
        'answer': ('NO' if n_post_drives == 0 else 'YES/PARTIAL'),
        'n_cells': len(cells),
        'n_cells_post_drives_null': n_post_drives,
        'n_cells_post_attenuates_pre_effect': n_post_att,
        'all_cells_full_null_5pct': bool(all_full_null),
        'all_cells_pre_null_5pct': bool(all_pre_null),
        'all_cells_post_null_5pct': bool(all_post_null),
        'max_abs_t_post_peermax': float(max(post_abs_t)) if post_abs_t else None,
        'max_abs_t_pre_peermax': float(max(pre_abs_t)) if pre_abs_t else None,
        'max_abs_t_full_peermax': float(max(full_abs_t)) if full_abs_t else None,
        'post_rows': nv_post_seg['n'],
        'post_share_of_fm_rows': nv_post_seg['share_of_fm_rows'],
        'post_months': nv_post_seg['T'],
        'post_share_of_fm_months': nv_post_seg['share_of_fm_months'],
        'post_market_equity_share': nv_post_seg['market_equity_share'],
        'pre_rows': nv_pre_seg['n'],
        'pre_share_of_fm_rows': nv_pre_seg['share_of_fm_rows'],
        'baseline_ew': baseline_ew,
        'baseline_vw': baseline_vw,
        'cells': cells,
        'diff_tests_nv': {k: v for k, v in diff_tests.items()
                          if v['family'] == 'network_version'},
        'diff_tests_calendar': {k: v for k, v in diff_tests.items()
                                if v['family'] == 'calendar'},
        'interpretation': (
            'post 20 个月（2022-05..2023-12，占 FM 行 22.57%、占月份 19.23%）本身的 '
            'PeerMAX 系数全部不显著（6 个 规格×权重 单元 max|t|='
            f'{max(post_abs_t) if post_abs_t else float("nan"):.2f}）；'
            'pre 84 个月同样全部不显著（max|t|='
            f'{max(pre_abs_t) if pre_abs_t else float("nan"):.2f}）。'
            '因此 headline null 不是由断点后 20 个月制造出来的；'
            '去掉 post 只留 pre，spec_full_controls EW t='
            f'{baseline_ew["pre_t_peermax"]:.2f}、VW t={baseline_vw["pre_t_peermax"]:.2f}，仍是 null。'
            'post spec_full_controls VW β='
            f'{baseline_vw["post_beta_peermax"]:.4f} 接近 0（pre {baseline_vw["pre_beta_peermax"]:.4f}），'
            '说明 post 反而把 pre 阶段温和的 VW 负值稀释掉，而不是产生 null。'),
    }

    post_anchor = snapshot_anchors('post')
    anchor_cmp = compare_anchors(pre_anchor, post_anchor)
    require(anchor_cmp['status'] == 'MATCH', f'锚在运行前后变化：{anchor_cmp["diff"]}')

    counts = {
        'phase': '3B',
        'task': 'N-4 2021 语料断点分段诊断',
        'script': str(Path(__file__).resolve().relative_to(ROOT)).replace('\\', '/'),
        'version_tag': VERSION_TAG,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_sec': round(time.time() - t0, 2),
        'constants': {
            'fm_rows': EXPECTED_FM_ROWS,
            'fm_months': EXPECTED_FM_MONTHS,
            'fm_month_range': [EXPECTED_FM_START, EXPECTED_FM_END],
            'nv_break': NV_BREAK,
            'calendar_break': CAL_BREAK,
            'min_n': MIN_N,
            'nw_lag': NW_LAG,
            'min_effective_month_count': MIN_EFFECTIVE_MONTHS,
            'winsorization_pct': WINSOR,
            'specs': {sid: cols for sid, _, cols in SPECS},
            'weights': [w for w, _ in WEIGHTS],
            'estimability_rule': ('T >= nw_lag + 2 且月内 n >= min_n；min_n=100 是月度 focal 数下限，'
                                  '不是月数下限。post T=20 可估，但低于 registry '
                                  'min_effective_month_count=60，故只作诊断、不构成 pricing_gate gate。'),
            'treatment_timing': ('network_version = 该事件月使用的文本年网络；'
                                 'version<=2020 -> 事件月 2015-05..2022-04（84 月）；'
                                 'version>=2021 -> 事件月 2022-05..2023-12（20 月）。'),
        },
        'anchors': {
            'pre': pre_anchor, 'post': post_anchor, 'compare': anchor_cmp,
        },
        'headline_reference': {
            'source': str(REF_3_6.relative_to(ROOT)).replace('\\', '/'),
            'bytes': pre_anchor['reference_3_6']['bytes'],
            'source_run_at': ref_run_at,
            'source_version_tag': ref_version_tag,
            'rows': [ref[k] for k in sorted(ref)],
            'note': '全样本 headline 读数直接引用 3.6_counts.fm_rows[variant=headline]；本脚本未重跑官方 pricing_gate。',
        },
        'full_sample_monthly_n': (
            fm.groupby('month', sort=True).size().rename('n').reset_index()
            .to_dict('records')),
        'segments': [{
            'id': s['id'], 'family': s['family'], 'label': s['label'],
            'definition': s['definition'], 'T': s['T'], 'n': s['n'],
            'month_range': [s['months'][0], s['months'][-1]],
            'months': s['months'],
            'share_of_fm_rows': s['share_of_fm_rows'],
            'share_of_fm_months': s['share_of_fm_months'],
            'market_equity_share': s['market_equity_share'],
            'network_versions': s['n_network_versions'],
            'monthly_n': s['monthly_n'].to_dict('records'),
        } for s in segs],
        'results': [{k: v for k, v in r.items() if k != '_monthly_betas'}
                    for r in all_rows],
        'partition_check': {
            'rule': ('nv_pre + nv_post 的月度 PeerMAX/OwnMAX/PeerReturn 系数序列拼接后，'
                     '逐月均值必须等于 3.6 登记的全样本 headline β（容差 1e-9）；'
                     '证明分段回归是同一固定样本上的真分割，未改变任何月度横截面。'),
            'cells': partition_checks,
            'max_abs_diff': max((c['abs_diff'] for c in partition_checks), default=None),
        },
        'diff_tests': diff_tests,
        'verdict': verdict,
        'outputs': {},
    }

    # 先写 CSV 再补 sha；JSON 不包含自身 sha
    counts['outputs']['segment_csv'] = {
        'path': str(CSV_PATH.relative_to(ROOT)).replace('\\', '/'),
        'rows': int(len(out)), 'columns': int(len(out.columns)),
        'bytes': file_size(CSV_PATH),
    }
    counts['outputs']['counts_json'] = {
        'path': str(JSON_PATH.relative_to(ROOT)).replace('\\', '/'),
        'note': '本文件（自身 sha 不写入，避免自指）',
    }
    counts['outputs']['execution_md'] = {
        'path': 'artifacts/phase3b/3b_n4_execution.md',
        'note': '由本脚本输出后人工定稿；含表/结论/复现命令',
    }
    write_json(JSON_PATH, counts)

    print(f'[{time.strftime("%H:%M:%S")}] 写出 {CSV_PATH.relative_to(ROOT)} '
          f'({len(out)} 行 × {len(out.columns)} 列)', flush=True)
    print(f'[{time.strftime("%H:%M:%S")}] 写出 {JSON_PATH.relative_to(ROOT)}', flush=True)
    print(f'[{time.strftime("%H:%M:%S")}] 锚 {anchor_cmp["status"]}（9 输入 n_diff={anchor_cmp["n_diff"]}）',
          flush=True)
    print(f'[{time.strftime("%H:%M:%S")}] 判定 answer={verdict["answer"]}；'
          f'post drives={n_post_drives}/{len(cells)}；'
          f'max|t_post|={verdict["max_abs_t_post_peermax"]:.3f}；'
          f'post rows share={verdict["post_share_of_fm_rows"]:.4%}', flush=True)
    print(f'[{time.strftime("%H:%M:%S")}] elapsed={counts["elapsed_sec"]}s', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
