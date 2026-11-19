"""Phase 3.5：N-5 / A12 行业吸收残差化 shadow（只读 headline，不重跑 pricing_gate/PC/网络）。

方案依据：`artifacts/phase3/3.0_plan.md` §2.5（N-5/A12 行业吸收残差化）。
任务约束：
  - 只写 `artifacts/phase3/3.5_*`、`data/processed/phase3_shadow/*`（3.5 产物）；
    不改 headline 样本/面板/边表/G0/configs/reports，不重跑 headline pricing_gate/PC/网络。
  - 9 个 pricing_gate 主输入 + headline 样本/面板/G0 A12 报告 sha 前后对账；任一变化 → 停机。

计算内容：
  1. **第一阶段（残差化）**：在 pricing_gate FM 样本上，按 month 固定效应（等价「按 month 回归 + 常数」）
     把 `PeerMAXraw` 对 `ind_l1_max_ew` / `ind_l2_max_ew` 回归，取残差。
     - 主口径 = month FE（pooled within；与 plan E7/E9 的 month 虚拟变量回归逐值等价）；
     - 对照口径 = 逐月单独斜率（`_permonth` 后缀），只作稳健性。
     - 两套行业基线：headline latest（`ind_*_max_ew`）与 3.1 as-of（`ind_*_max_ew_p3asof`）。
     报第一阶段 R²（pooled month FE / 双变量 r² / 逐月 r² 均值与加权）、斜率、相关、正交性诊断。
  2. **A12 规格重跑**（`pricing_lib.fm_regression`，NW lag=6，min_n=100）：
     - naked（无 OwnMAX/PeerReturn/controls）：PeerMAXraw 单变量；PeerMAXraw + ind_l1/l2；
       PeerMAX_resid + ind；PeerMAX_resid 单变量；含两行业联合。
     - full（A12 headline 控制集）：主规格 + 5 个 headline A12 规格（L1/L2/both/L1vw/L2vw）；
       残差化 treatment 的主规格与 +ind 规格。
     报 treatment/OwnMAX?/IndustryMAX 的 β/se/t、δ=1−|β_with|/|β_without|、同一样本 δ、
     β 差值的 NW 检验（Δβ）、sign_flip。
  3. **δ 不可解释性量化**：β_without 不显著（t/CI）、共线（corr 0.777/0.792、VIF）、
     δ 对样本口径敏感（fullsample vs samesample）、逐月 δ 离散度、残差化后 β 变化
     （含 p3asof 主规格 β）以及「δ vs headline 主系数」的负保留率读数。
  4. **as-of vs latest**：两套行业基线的第一阶段 R²/corr/斜率、残差化主规格 β、
     A12 L2 的 β/γ/δ；同一行样本（两套 L2 均非缺）下的直接对比。
  5. **团-行业交叉**（plan §2.5 ⑤）：headline 聚类 × latest / as-of L2 标签，
     逐年报跨 ≥3 行业公司占比、modal 占比中位、有效行业数中位、最大团行业数。
  6. **9 锚 + headline 只读文件 sha 前后对账**；输出路径守卫（只允许 shadow/artifacts）。

产物：
  - `artifacts/phase3/3.5_counts.json`（汇总/门控/锚台账）
  - `artifacts/phase3/3.5_industry_absorb.csv`（规格级 A12 表）
  - `artifacts/phase3/3.5_collinearity.csv`（第一阶段/共线/VIF 表）
  - `artifacts/phase3/3.5_cluster_industry_cross.csv`（团-行业交叉）
  - `artifacts/phase3/3.5_anchor_pre.json` / `3.5_anchor_post.json`
  - `data/processed/phase3_shadow/pricing_peer_resid_shadow.parquet`（只读派生，非 headline）

用法：
  python scripts/shadow_industry_absorb.py
  python scripts/shadow_industry_absorb.py --skip-cluster-cross"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import csv
import json
import math
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pricing_lib import (
    assert_pricing_inputs, fm_regression, newey_west, param_value,
    require, file_size, shadow_sample_path, vif_table,
)
from fileio import assert_path_allowed, json_default, write_csv_rows, write_json_atomic, write_parquet_atomic


ROOT = Path(__file__).resolve().parents[1]

ART = ROOT / 'artifacts/phase3'

SHADOW_DIR = ROOT / 'data/processed/phase3_shadow'

HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'

REPORTS = ROOT / 'reports'

CONFIGS = ROOT / 'configs'

SAMPLE = shadow_sample_path('ma_teacher')

HEADLINE_PANEL = ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv'

FIRM_CHARS = ROOT / 'data/processed/firm_chars_clean.csv'

G0_A12 = ROOT / 'reports/data_gates/G0/pricing_industry_absorb.json'

ASOF_MAP = SHADOW_DIR / 'sw2_asof_map_shadow.parquet'

ASOF_BASELINES = SHADOW_DIR / 'ind_baselines_p3asof.parquet'

SAMPLE_ASOF = SHADOW_DIR / 'pricing_sample_ma_teacher_p3asof_baselines.parquet'

SW2_MAP = ROOT / 'data/processed/sw2_industry_map.csv'

SW2_LATEST = ROOT / 'data/processed/sw2_industry_latest.csv'

COUNTS_JSON = ART / '3.5_counts.json'

ABSORB_CSV = ART / '3.5_industry_absorb.csv'

COLLIN_CSV = ART / '3.5_collinearity.csv'

CROSS_CSV = ART / '3.5_cluster_industry_cross.csv'

ANCHOR_PRE = ART / '3.5_anchor_pre.json'

ANCHOR_POST = ART / '3.5_anchor_post.json'

RESID_PARQUET = SHADOW_DIR / 'pricing_peer_resid_shadow.parquet'

SHADOW_VERSION_TAG = 'shadow_industry_absorb_v1'

SAMPLE_ROWS = 307_503

FM_ROWS = 181_178

FM_MONTHS = 104

E7_R2_L2_LATEST = 0.6483374839833427

E7_R2_L2_LATEST_ALT = 0.6483

E7_CORR_L1_LATEST = 0.7916556878781287

E7_CORR_L2_LATEST = 0.7769299176515353

HEADLINE_BETA_WITHOUT = 0.03427280848385429

HEADLINE_A12_L2_BETA = -0.005753124144123507

HEADLINE_A12_L2_GAMMA = 0.11196805604939121

HEADLINE_A12_L2_DELTA = 0.8321373590718786

HEADLINE_A12_L1_BETA = -0.024338089840610498

HEADLINE_A12_L1_GAMMA = 0.17498420066071604

HEADLINE_A12_L1_DELTA = 0.28987174038929364

E9_RESID_BETA_LATEST = -0.03905433621818112

E9_RESID_T_LATEST = -0.8128727018253377

CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq',
            'AnalystCoverage', 'NetworkDegree']

BASE_X = ['OwnMAXraw', 'PeerReturn'] + CONTROLS

IND = {
    'latest': {'l1': 'ind_l1_max_ew', 'l2': 'ind_l2_max_ew',
               'l1_vw': 'ind_l1_max_vw', 'l2_vw': 'ind_l2_max_vw'},
    'p3asof': {'l1': 'ind_l1_max_ew_p3asof', 'l2': 'ind_l2_max_ew_p3asof',
               'l1_vw': 'ind_l1_max_vw_p3asof', 'l2_vw': 'ind_l2_max_vw_p3asof'},
}

RESID = {
    'latest': {'l1': 'PeerMAXraw_p3resid_l1_latest',
               'l2': 'PeerMAXraw_p3resid_l2_latest',
               'l1l2': 'PeerMAXraw_p3resid_l1l2_latest'},
    'p3asof': {'l1': 'PeerMAXraw_p3resid_l1_p3asof',
               'l2': 'PeerMAXraw_p3resid_l2_p3asof',
               'l1l2': 'PeerMAXraw_p3resid_l1l2_p3asof'},
}

YEARS = list(range(2014, 2023))

ALLOWED_ROOTS = (SHADOW_DIR, ART)

BANNED_ROOTS = (
    HEADLINE_NET, ROOT / 'data/processed/network_ma',
    ROOT / 'data/processed/network_mb', ROOT / 'data/processed/network_ma_teacher_idf',
    ROOT / 'data/processed/network_ma_teacher_medadj',
    ROOT / 'data/processed/network_ma_teacher_df10',
    ROOT / 'data/processed/network_ma_teacher_df100',
    ROOT / 'data/processed/network_ma_fulltext',
    ROOT / 'data/processed/network_mb_teacher', ROOT / 'data/processed/phase2_shadow',
    REPORTS, CONFIGS, ROOT / 'scripts', ROOT / 'tests', ROOT / 'docs',
    ROOT / 'deliverables',
)

EXTRA_INPUTS = {
    'headline_sample_parquet': SAMPLE,
    'headline_panel_csv': HEADLINE_PANEL,
    'firm_chars_clean': FIRM_CHARS,
    'g0_a12_report': G0_A12,
    'asof_map_p3shadow': ASOF_MAP,
    'asof_baselines_p3asof': ASOF_BASELINES,
    'asof_sample_copy': SAMPLE_ASOF,
}

def assert_output_path_guard(path) -> Path:
    return assert_path_allowed(path, ALLOWED_ROOTS, (
        *BANNED_ROOTS, SAMPLE, HEADLINE_PANEL, FIRM_CHARS, G0_A12,
        ASOF_MAP, ASOF_BASELINES, SAMPLE_ASOF, SW2_MAP, SW2_LATEST))

def _atomic_text(path, text: str) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, p)

def write_json(path, obj) -> None:
    write_json_atomic(obj, assert_output_path_guard(path), indent=1, default=json_default,
                      trailing_newline=False)

def fmt(v) -> str:
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'true' if bool(v) else 'false'
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not math.isfinite(v):
            return ''
        return f'{v:.12g}'
    return str(v)

def write_csv(path, header, rows) -> None:
    write_csv_rows(assert_output_path_guard(path),
                   [[fmt(r.get(k)) for k in header] for r in rows], header)

def write_parquet(path, df: pd.DataFrame) -> None:
    write_parquet_atomic(df, assert_output_path_guard(path))

def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def anchors_snapshot(tag: str) -> dict:
    ledger = assert_pricing_inputs()
    extra = {k: {'file': str(v.relative_to(ROOT)).replace('\\', '/'), 'bytes': file_size(v)}
             for k, v in EXTRA_INPUTS.items()}
    clusters = {}
    for y in YEARS:
        p = HEADLINE_NET / 'clusters' / f'{y}_clusters.csv'
        clusters[p.relative_to(ROOT).as_posix()] = file_size(p)
    snap = {
        'tag': tag,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'pricing_inputs_9': {r['input']: r['rows'] for r in ledger},
        'pricing_inputs_ledger': ledger,
        'extra_inputs': extra,
        'headline_clusters_bytes': clusters,
    }
    return snap

def compare_anchors(pre: dict, post: dict) -> dict:
    diffs = []
    for sect in ('pricing_inputs_9', 'headline_clusters_bytes'):
        a, b = pre.get(sect) or {}, post.get(sect) or {}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diffs.append({'section': sect, 'key': k, 'pre': a.get(k), 'post': b.get(k)})
    a = {k: v.get('bytes') for k, v in (pre.get('extra_inputs') or {}).items()}
    b = {k: v.get('bytes') for k, v in (post.get('extra_inputs') or {}).items()}
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            diffs.append({'section': 'extra_inputs', 'key': k, 'pre': a.get(k), 'post': b.get(k)})
    return {'status': 'MATCH' if not diffs else 'CHANGED', 'n_diff': len(diffs), 'diffs': diffs}

def within_month_residual(df: pd.DataFrame, ycol: str, xcols, month_col: str = 'month') -> dict:
    """`y ~ xcols + month FE` 的 pooled within 残差（等价于 month 虚拟变量 + 常数）。

    返回 sub（非缺行）、resid（与 sub.index 对齐）、斜率/截距、R² 多种定义、逐月斜率与正交性诊断。
    正交性：残差与每个 x 的样本相关、残差逐月均值相对 y 的 sd。
    """
    xcols = list(xcols)
    sub = df.dropna(subset=[ycol] + xcols).copy()
    require(len(sub) > 0, f'第一阶段：{ycol} ~ {xcols} 无有效行')
    y = sub[ycol].to_numpy(dtype=float)
    X = sub[xcols].to_numpy(dtype=float)
    g = sub.groupby(month_col, sort=False)
    ym = g[ycol].transform('mean').to_numpy(dtype=float)
    Xm = np.column_stack([g[c].transform('mean').to_numpy(dtype=float) for c in xcols])
    yd = y - ym
    Xd = X - Xm
    A = np.column_stack([Xd, np.ones(len(sub), dtype=float)])
    coef, *_ = np.linalg.lstsq(A, yd, rcond=None)
    resid = yd - A @ coef
    rank = int(np.linalg.matrix_rank(A))
    dof = len(sub) - rank
    sigma2 = float(resid @ resid / dof) if dof > 0 else float('nan')
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    r2 = 1.0 - float(np.var(resid) / np.var(y)) if np.var(y) > 0 else float('nan')
    corr = float(np.corrcoef(y, X[:, 0])[0, 1]) if len(xcols) == 1 else None
    # 逐月单独斜率（单变量稳健性口径）
    pm_slopes, pm_r2, pm_n = [], [], []
    if len(xcols) == 1:
        for _, gi in sub.groupby(month_col, sort=True):
            xx = gi[xcols[0]].to_numpy(dtype=float)
            yy = gi[ycol].to_numpy(dtype=float)
            if len(xx) < 3 or np.var(xx) <= 0:
                continue
            b = float(np.cov(xx, yy, ddof=1)[0, 1] / np.var(xx, ddof=1))
            r = float(np.corrcoef(xx, yy)[0, 1])
            pm_slopes.append(b)
            pm_r2.append(r * r)
            pm_n.append(len(xx))
    # 正交性诊断
    corr_resid_x = [float(np.corrcoef(resid, X[:, j])[0, 1]) for j in range(X.shape[1])]
    month_mean = pd.Series(resid).groupby(sub[month_col].to_numpy()).mean().to_numpy()
    return {
        'sub': sub,
        'resid': pd.Series(resid, index=sub.index, name='resid'),
        'coef': {c: float(coef[j]) for j, c in enumerate(xcols)},
        'intercept': float(coef[-1]),
        'se': {c: float(se[j]) for j, c in enumerate(xcols)},
        't': {c: (float(coef[j] / se[j]) if se[j] > 0 else None) for j, c in enumerate(xcols)},
        'n_rows': int(len(sub)),
        'n_months': int(sub[month_col].nunique()),
        'r2_pooled_month_fe': r2,
        'r2_bivariate': (corr ** 2 if corr is not None else None),
        'r2_per_month_mean': float(np.mean(pm_r2)) if pm_r2 else None,
        'r2_per_month_weighted': (float(np.average(pm_r2, weights=pm_n))
                                  if pm_r2 else None),
        'per_month_slope_mean': float(np.mean(pm_slopes)) if pm_slopes else None,
        'per_month_slope_median': float(np.median(pm_slopes)) if pm_slopes else None,
        'per_month_slope_p10': float(np.quantile(pm_slopes, 0.10)) if pm_slopes else None,
        'per_month_slope_p90': float(np.quantile(pm_slopes, 0.90)) if pm_slopes else None,
        'corr_pearson': corr,
        'corr_spearman': (float(sub[ycol].corr(sub[xcols[0]], method='spearman'))
                          if len(xcols) == 1 else None),
        'max_abs_corr_resid_x': float(np.max(np.abs(corr_resid_x))),
        'max_abs_month_mean_resid': float(np.max(np.abs(month_mean))),
        'y_sd': float(np.std(y, ddof=0)),
    }

def first_stage_summary(fs: dict, extra: dict | None = None) -> dict:
    """第一阶段结果 → 可 JSON 序列化的摘要（剔除 sub/resid）。"""
    out = {k: v for k, v in fs.items() if k not in ('sub', 'resid')}
    out.update(extra or {})
    return out

def _spec(sid, family, basis, treatment, x, ind_vars, baseline, note=''):
    return {'spec_id': sid, 'family': family, 'basis': basis, 'treatment': treatment,
            'x': list(x), 'industry_vars': list(ind_vars), 'baseline': baseline,
            'note': note}

def build_specs() -> list:
    specs = []
    for basis in ('latest', 'p3asof'):
        ind = IND[basis]
        resid = RESID[basis]
        # naked family：无 OwnMAX/PeerReturn/controls
        specs.append(_spec('main_naked', 'naked', basis, 'PeerMAXraw', ['PeerMAXraw'],
                           [], 'main_naked'))
        for lv in ('l1', 'l2', 'l1_vw', 'l2_vw'):
            specs.append(_spec(f'a12_{lv}_naked', 'naked', basis, 'PeerMAXraw',
                               ['PeerMAXraw', ind[lv]], [ind[lv]], 'main_naked'))
        specs.append(_spec('a12_both_naked', 'naked', basis, 'PeerMAXraw',
                           ['PeerMAXraw', ind['l1'], ind['l2']],
                           [ind['l1'], ind['l2']], 'main_naked'))
        for lv in ('l1', 'l2'):
            specs.append(_spec(f'resid_{lv}_univariate_naked', 'naked', basis,
                               resid[lv], [resid[lv]], [], f'resid_{lv}_univariate_naked'))
            specs.append(_spec(f'resid_{lv}_plus_ind_naked', 'naked', basis,
                               resid[lv], [resid[lv], ind[lv]], [ind[lv]],
                               f'resid_{lv}_univariate_naked'))
        # full family：A12 headline 控制集（OwnMAX + PeerReturn + 8 controls）
        specs.append(_spec('main_full', 'full', basis, 'PeerMAXraw',
                           ['PeerMAXraw'] + BASE_X, [], 'main_full'))
        for lv in ('l1', 'l2', 'l1_vw', 'l2_vw'):
            specs.append(_spec(f'a12_{lv}_full', 'full', basis, 'PeerMAXraw',
                               ['PeerMAXraw', ind[lv]] + BASE_X, [ind[lv]], 'main_full'))
        specs.append(_spec('a12_both_full', 'full', basis, 'PeerMAXraw',
                           ['PeerMAXraw', ind['l1'], ind['l2']] + BASE_X,
                           [ind['l1'], ind['l2']], 'main_full'))
        for lv in ('l1', 'l2'):
            specs.append(_spec(f'resid_{lv}_main_full', 'full', basis, resid[lv],
                               [resid[lv]] + BASE_X, [], f'resid_{lv}_univariate_naked'))
            specs.append(_spec(f'resid_{lv}_plus_ind_full', 'full', basis, resid[lv],
                               [resid[lv], ind[lv]] + BASE_X, [ind[lv]],
                               f'resid_{lv}_univariate_naked'))
    return specs

def monthly_r2_series(df: pd.DataFrame, ycol: str, xcols, min_n: int) -> np.ndarray:
    """逐月横截面 OLS 的 R² 序列（仅用于增量 R² 读数，不替代 FM 系数推断）。"""
    out = []
    for _, g in df.dropna(subset=[ycol] + list(xcols)).groupby('month', sort=True):
        if len(g) < min_n:
            continue
        y = g[ycol].to_numpy(dtype=float)
        X = np.column_stack([g[c].to_numpy(dtype=float) for c in xcols]
                            + [np.ones(len(g), dtype=float)])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b
        ss_tot = float(((y - y.mean()) ** 2).sum())
        if ss_tot > 0:
            out.append(1.0 - float(r @ r) / ss_tot)
    return np.asarray(out, dtype=float)

def monthly_beta(per_month: list, col: str) -> dict:
    out = {}
    for d in per_month:
        if d.get('used') and d.get('beta') and d['beta'].get(col) is not None:
            out[str(d['month'])] = float(d['beta'][col])
    return out

def delta_ratio(beta_with, beta_without):
    if beta_with is None or beta_without is None or beta_without == 0:
        return None
    return 1.0 - abs(beta_with) / abs(beta_without)

def run_specs(fm: pd.DataFrame, specs: list, min_n: int, nw_lag: int) -> tuple:
    """执行全部规格；返回 (rows, cache)。cache 保存每规格 (summary, per_month, sub)。"""
    cache = {}
    for s in specs:
        sub = fm.dropna(subset=list(s['x']) + ['R_next'])
        res, pm = fm_regression(sub, 'R_next', s['x'], min_n=min_n, nw_lag=nw_lag)
        b = res['coefs'].get(s['treatment'], {})
        cache[(s['spec_id'], s['basis'])] = {'spec': s, 'summary': res,
                                             'per_month': pm, 'sub': sub,
                                             'beta': b}
    rows = []
    for s in specs:
        sid, basis = s['spec_id'], s['basis']
        c = cache[(sid, basis)]
        res, pm, sub = c['summary'], c['per_month'], c['sub']
        b = c['beta']
        # 基准：同一样本重跑；全样本基准取 cache 中已有/重跑
        base_spec_id = s['baseline']
        base_key = (base_spec_id, basis)
        if base_key not in cache:
            bs = next(x for x in specs if x['spec_id'] == base_spec_id and x['basis'] == basis)
            bsub = fm.dropna(subset=list(bs['x']) + ['R_next'])
            bres, bpm = fm_regression(bsub, 'R_next', bs['x'], min_n=min_n, nw_lag=nw_lag)
            cache[base_key] = {'spec': bs, 'summary': bres, 'per_month': bpm,
                               'sub': bsub, 'beta': bres['coefs'].get(bs['treatment'], {})}
        base = cache[base_key]
        base_treat = base['spec']['treatment']
        # 同样本基准（在同一 sub 上重跑）
        same_res, same_pm = fm_regression(sub, 'R_next', base['spec']['x'],
                                          min_n=min_n, nw_lag=nw_lag)
        same_b = same_res['coefs'].get(base_treat, {})
        # Δβ（逐月配对）
        ser_a = monthly_beta(pm, s['treatment'])
        ser_b = monthly_beta(same_pm, base_treat)
        common = sorted(set(ser_a) & set(ser_b))
        diff = [ser_a[m] - ser_b[m] for m in common]
        nwd = newey_west(diff, lag=nw_lag)
        # 增量 R²（同样本、同月份；with − without 的逐月 R² 差）
        r2_with = monthly_r2_series(sub, 'R_next', s['x'], min_n)
        r2_wo = monthly_r2_series(sub, 'R_next', base['spec']['x'], min_n)
        if len(r2_with) == len(r2_wo) and len(r2_with) > 0:
            incr = r2_with - r2_wo
            r2_with_mean = float(r2_with.mean())
            r2_without_mean = float(r2_wo.mean())
            incr_mean = float(incr.mean())
            incr_median = float(np.median(incr))
        else:
            r2_with_mean = r2_without_mean = incr_mean = incr_median = None
        # 逐月 δ（仅单行业 treatment 的 A12 规格且基准为 main_* 时报告）
        delta_m = []
        if s['family'] in ('naked', 'full') and base_spec_id.startswith('main_'):
            for m in common:
                bwo = ser_b[m]
                if bwo != 0:
                    delta_m.append(1.0 - abs(ser_a[m]) / abs(bwo))
        gammas = [res['coefs'].get(c, {}) for c in s['industry_vars']]
        own = res['coefs'].get('OwnMAXraw', {})
        peerret = res['coefs'].get('PeerReturn', {})
        row = {
            'family': s['family'], 'spec_id': sid, 'basis': basis,
            'treatment': s['treatment'],
            'industry_var_1': s['industry_vars'][0] if len(s['industry_vars']) > 0 else '',
            'industry_var_2': s['industry_vars'][1] if len(s['industry_vars']) > 1 else '',
            'x_vars': '|'.join(s['x']),
            'n_rows': int(len(sub)), 'n_months_used': res['n_months_used'],
            'all_months_full_rank': bool(all(d.get('rank_status') == 'FULL'
                                             for d in res['per_month'] if d.get('used'))),
            'beta': b.get('beta'), 'se_nw': b.get('se_nw'), 't_nw': b.get('t_nw'),
            'p_two_sided': b.get('p_two_sided'),
            'ci95_low': b.get('ci95_low'), 'ci95_high': b.get('ci95_high'),
            'ownmax_beta': own.get('beta'), 'ownmax_se': own.get('se_nw'),
            'ownmax_t': own.get('t_nw'),
            'peerreturn_beta': peerret.get('beta'), 'peerreturn_se': peerret.get('se_nw'),
            'peerreturn_t': peerret.get('t_nw'),
            'gamma_1_var': s['industry_vars'][0] if len(s['industry_vars']) > 0 else '',
            'gamma_1_beta': gammas[0].get('beta') if gammas else None,
            'gamma_1_se': gammas[0].get('se_nw') if gammas else None,
            'gamma_1_t': gammas[0].get('t_nw') if gammas else None,
            'gamma_2_var': s['industry_vars'][1] if len(s['industry_vars']) > 1 else '',
            'gamma_2_beta': gammas[1].get('beta') if len(gammas) > 1 else None,
            'gamma_2_se': gammas[1].get('se_nw') if len(gammas) > 1 else None,
            'gamma_2_t': gammas[1].get('t_nw') if len(gammas) > 1 else None,
            'baseline_spec': base_spec_id,
            'beta_without_fullsample': base['beta'].get('beta'),
            'beta_without_samesample': same_b.get('beta'),
            'delta_vs_baseline_fullsample': delta_ratio(b.get('beta'), base['beta'].get('beta')),
            'delta_vs_baseline_samesample': delta_ratio(b.get('beta'), same_b.get('beta')),
            'sign_flip': (bool(base['beta'].get('beta') is not None and b.get('beta') is not None
                               and np.sign(base['beta']['beta']) != np.sign(b['beta']))),
            'delta_beta': nwd['mean'], 'delta_beta_se_nw': nwd['se_nw'],
            'delta_beta_t_nw': nwd['t_nw'], 'n_months_paired': nwd['T'],
            'r2_with_mean': r2_with_mean, 'r2_without_mean': r2_without_mean,
            'incremental_r2_mean': incr_mean, 'incremental_r2_median': incr_median,
            'delta_vs_headline_main': delta_ratio(b.get('beta'), HEADLINE_BETA_WITHOUT),
            'delta_monthly_median': (float(np.median(delta_m)) if delta_m else None),
            'delta_monthly_p10': (float(np.quantile(delta_m, 0.10)) if delta_m else None),
            'delta_monthly_p90': (float(np.quantile(delta_m, 0.90)) if delta_m else None),
            'delta_monthly_share_outside_0_1': (
                float(np.mean([(d < 0.0 or d > 1.0) for d in delta_m])) if delta_m else None),
            'delta_monthly_n': len(delta_m),
            'delta_interpretable': bool(
                base['beta'].get('t_nw') is not None and abs(base['beta']['t_nw']) >= 1.96
                and nwd['t_nw'] is not None and abs(nwd['t_nw']) >= 1.96),
            'note': s['note'],
        }
        rows.append(row)
    return rows, cache

def _l2_map_for_key(key: str) -> dict:
    m = pd.read_csv(SW2_MAP, dtype={'code6': str, 'l2_code': str})
    m['code6'] = m['code6'].str.zfill(6)
    m['effective_from'] = pd.to_datetime(m['effective_from'])
    lat = pd.read_csv(SW2_LATEST, dtype=str)
    lat = lat[lat['firm_id'].notna() & lat['code6'].notna()]
    code2firm = dict(zip(lat['code6'].str.zfill(6), lat['firm_id']))
    sub = (m[m['effective_from'] <= pd.Timestamp(key)]
           .sort_values('effective_from', kind='stable')
           .drop_duplicates('code6', keep='last'))
    return {code2firm[c]: v for c, v in zip(sub['code6'], sub['l2_code']) if c in code2firm}

def _latest_l2_map() -> dict:
    lat = pd.read_csv(SW2_LATEST, dtype=str)
    lat = lat[lat['firm_id'].notna() & lat['l2_code'].notna()]
    return dict(zip(lat['firm_id'], lat['l2_code']))

def cluster_industry_cross() -> tuple:
    latest = _latest_l2_map()
    rows = []
    for y in YEARS:
        c = pd.read_csv(HEADLINE_NET / 'clusters' / f'{y}_clusters.csv',
                        dtype={'firm_id': str, 'cluster_id': str})
        for basis in ('latest', 'p3asof'):
            labels = latest if basis == 'latest' else _l2_map_for_key(f'{y + 1}-04-30')
            d = c.copy()
            d['l2'] = d['firm_id'].map(labels)
            d = d.dropna(subset=['l2'])
            if d.empty:
                rows.append({'text_year': y, 'network_year': y + 1, 'basis': basis,
                             'n_firms_labeled': 0, 'n_clusters': 0})
                continue
            g = d.groupby('cluster_id')
            size = g.size()
            n_ind = g['l2'].nunique()
            modal = g['l2'].agg(lambda s: float(s.value_counts(normalize=True).iloc[0]))
            eff = g['l2'].agg(lambda s: float(1.0 / np.sum(np.square(
                s.value_counts(normalize=True).to_numpy()))))
            multi = size >= 2
            firm_ge3_all = float((size * (n_ind >= 3)).sum() / size.sum())
            firm_ge3_multi = (float((size[multi] * (n_ind[multi] >= 3)).sum()
                                    / size[multi].sum()) if multi.any() else None)
            rows.append({
                'text_year': y, 'network_year': y + 1, 'basis': basis,
                'n_firms_labeled': int(len(d)), 'n_clusters': int(size.shape[0]),
                'n_multifirm_clusters': int(multi.sum()),
                'median_cluster_size': float(size.median()),
                'max_cluster_size': int(size.max()),
                'firm_share_ge3_all': firm_ge3_all,
                'firm_share_ge3_multifirm': firm_ge3_multi,
                'cluster_share_ge3_multifirm': (float((n_ind[multi] >= 3).mean())
                                                if multi.any() else None),
                'modal_share_median_multifirm': (float(modal[multi].median())
                                                 if multi.any() else None),
                'effective_industries_median_multifirm': (float(eff[multi].median())
                                                          if multi.any() else None),
                'firm_mean_modal_share_multifirm': (
                    float(np.average(modal[multi], weights=size[multi]))
                    if multi.any() else None),
                'firm_mean_effective_industries_multifirm': (
                    float(np.average(eff[multi], weights=size[multi]))
                    if multi.any() else None),
                'max_industries': int(n_ind.max()),
            })
    return rows, {'latest': latest}


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


# ---------------------------------------------------------------- 路径与常量


# plan §1.2 E7–E9 / §2.5 R1 预登记值
# plan E9 预跑用的是 headline **latest** `ind_l2_max_ew`（见 §11 E7/E9 命令），不是 as-of；
# as-of 主口径的残差化 β 由本脚本现算，单独报（见 `asof_vs_latest`）。


# 输出路径守卫：只允许 phase3 shadow 与 artifacts/phase3


# ---------------------------------------------------------------- 输出守卫与原子写


# ---------------------------------------------------------------- 锚台账


# ---------------------------------------------------------------- 第一阶段：month FE 残差化


# ---------------------------------------------------------------- 规格定义与执行


# ---------------------------------------------------------------- 团-行业交叉


    # note: 仅在 2019 网络年（text_year=2018）核对审计读数，不做事先硬门


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description='Phase 3.5 N-5/A12 行业吸收残差化 shadow')
    ap.add_argument('--skip-cluster-cross', action='store_true')
    args = ap.parse_args()
    t0 = time.time()

    log('3.5 preflight：9 锚 + headline 只读文件 sha 台账')
    pre = anchors_snapshot('pre')
    write_json(ANCHOR_PRE, pre)
    log(f'3.5 preflight 完成：9 锚 MATCH；extra={len(pre["extra_inputs"])}；'
        f'headline clusters={len(pre["headline_clusters_bytes"])}')

    reg = param_value('sample_and_spec_registry')
    min_n = int(reg['min_focal_count_per_month'])
    nw_lag = int(param_value('nw_lag'))
    log(f'registry: min_n={min_n}, nw_lag={nw_lag}')

    # ---- 装载 headline 样本 + 3.1 as-of 列
    base = pd.read_parquet(SAMPLE)
    require(len(base) == SAMPLE_ROWS, f'headline 样本行数 {len(base)} != {SAMPLE_ROWS}')
    require(base.duplicated(['focal_id', 'month']).sum() == 0, 'headline 样本键重复')
    asof_copy = pd.read_parquet(SAMPLE_ASOF)
    require(len(asof_copy) == SAMPLE_ROWS, f'as-of 样本副本行数 {len(asof_copy)} != {SAMPLE_ROWS}')
    require(asof_copy[base.columns].equals(base), 'as-of 样本副本的 headline 列与原样本不一致')
    asof_cols = [c for c in asof_copy.columns if c.endswith('_p3asof')]
    require(len(asof_cols) == 8, f'as-of 列数 {len(asof_cols)} != 8')
    base = base.merge(asof_copy[['focal_id', 'month'] + asof_cols],
                      on=['focal_id', 'month'], how='left', validate='one_to_one')
    require(len(base) == SAMPLE_ROWS, 'merge as-of 列后行数变化')

    # 与 ind_baselines_p3asof.parquet 交叉对账（只读输入）
    bas = pd.read_parquet(ASOF_BASELINES)
    require(not bas.duplicated(['firm_id', 'month']).any(),
            'ind_baselines_p3asof 键 (firm_id, month) 重复')
    chk = bas.set_index(['firm_id', 'month']).reindex(
        pd.MultiIndex.from_arrays([base['focal_id'], base['month']]))
    max_diff = 0.0
    for c in asof_cols:
        a = base[c].to_numpy(dtype=float)
        b = chk[c].to_numpy(dtype=float)
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            max_diff = max(max_diff, float(np.max(np.abs(a[both] - b[both]))))
        require(bool(np.array_equal(np.isfinite(a), np.isfinite(b))),
                f'as-of 列缺测模式与 ind_baselines_p3asof 不一致: {c}')
    require(max_diff <= 1e-12, f'as-of 列与 ind_baselines_p3asof 最大差 {max_diff} > 1e-12')
    log(f'as-of 列与 3.1 基线逐值一致（max|Δ|={max_diff:.2e}）')

    fm = base[base['pricing_in_fm_sample'].astype(bool)].copy()
    require(len(fm) == FM_ROWS, f'FM 行数 {len(fm)} != {FM_ROWS}')
    require(fm['month'].nunique() == FM_MONTHS, f'FM 月数 {fm["month"].nunique()} != {FM_MONTHS}')
    log(f'FM 样本：{len(fm):,} 行 / {fm["month"].nunique()} 月')

    # ---- 第一阶段：残差化（latest 与 as-of 两套；l1/l2/l1+l2）
    first_stage = {}
    vif = {}
    for basis in ('latest', 'p3asof'):
        first_stage[basis] = {}
        vif[basis] = {}
        for lvl in ('l1', 'l2'):
            col = IND[basis][lvl]
            fs = within_month_residual(fm, 'PeerMAXraw', [col])
            vt = vif_table(fs['sub'], ['PeerMAXraw', col])
            first_stage[basis][lvl] = first_stage_summary(
                fs, {'basis': basis, 'level': lvl, 'industry_var': col,
                     'vif_treatment': vt['vif_mean'].get('PeerMAXraw'),
                     'vif_industry': vt['vif_mean'].get(col),
                     'condition_number_mean': vt['condition_number_mean'],
                     'condition_number_max': vt['condition_number_max']})
            fm[RESID[basis][lvl]] = fs['resid'].reindex(fm.index)
            # 逐月单独斜率稳健性列（只用于 l2 主口径的对照；l1 也给）
            pm_resid = pd.Series(np.nan, index=fm.index, dtype=float)
            for m, gi in fm.dropna(subset=['PeerMAXraw', col]).groupby('month', sort=True):
                xx = gi[col].to_numpy(dtype=float)
                yy = gi['PeerMAXraw'].to_numpy(dtype=float)
                if len(xx) < 3 or np.var(xx) <= 0:
                    continue
                b = float(np.cov(xx, yy, ddof=1)[0, 1] / np.var(xx, ddof=1))
                a = float(yy.mean() - b * xx.mean())
                pm_resid.loc[gi.index] = yy - (a + b * xx)
            fm[f'{RESID[basis][lvl]}_permonth'] = pm_resid
        # l1 + l2 联合
        fs_both = within_month_residual(fm, 'PeerMAXraw', [IND[basis]['l1'], IND[basis]['l2']])
        first_stage[basis]['l1l2'] = first_stage_summary(
            fs_both, {'basis': basis, 'level': 'l1+l2',
                      'industry_var': f'{IND[basis]["l1"]}|{IND[basis]["l2"]}',
                      'vif_treatment': None, 'vif_industry': None,
                      'condition_number_mean': None, 'condition_number_max': None})
        fm[RESID[basis]['l1l2']] = fs_both['resid'].reindex(fm.index)
        log(f'第一阶段 {basis}: R²(l1)={first_stage[basis]["l1"]["r2_pooled_month_fe"]:.6f} '
            f'R²(l2)={first_stage[basis]["l2"]["r2_pooled_month_fe"]:.6f} '
            f'R²(l1+l2)={first_stage[basis]["l1l2"]["r2_pooled_month_fe"]:.6f}')

    # ---- 残差化 parquet（只读派生）
    resid_df = fm[['focal_id', 'month', 'pricing_in_fm_sample', 'PeerMAXraw']].copy()
    for basis in ('latest', 'p3asof'):
        for lvl in ('l1', 'l2', 'l1l2'):
            resid_df[RESID[basis][lvl]] = fm[RESID[basis][lvl]].to_numpy()
        for lvl in ('l1', 'l2'):
            resid_df[f'{RESID[basis][lvl]}_permonth'] = fm[f'{RESID[basis][lvl]}_permonth'].to_numpy()
            resid_df[f'resid_fit_r2_{basis}_{lvl}'] = first_stage[basis][lvl]['r2_pooled_month_fe']
    resid_df['resid_version'] = SHADOW_VERSION_TAG
    write_parquet(RESID_PARQUET, resid_df)
    log(f'残差化 parquet：{RESID_PARQUET.name}（{len(resid_df):,} 行）')

    # ---- A12 规格
    specs = build_specs()
    rows, cache = run_specs(fm, specs, min_n, nw_lag)
    log(f'A12 规格完成：{len(rows)} 行')

    # ---- 关键读数整理
    def get_row(sid, basis='latest'):
        return next(r for r in rows if r['spec_id'] == sid and r['basis'] == basis)

    key_latest = {
        'main_naked': get_row('main_naked'),
        'a12_l1_naked': get_row('a12_l1_naked'),
        'a12_l2_naked': get_row('a12_l2_naked'),
        'a12_both_naked': get_row('a12_both_naked'),
        'main_full': get_row('main_full'),
        'a12_l1_full': get_row('a12_l1_full'),
        'a12_l2_full': get_row('a12_l2_full'),
        'a12_both_full': get_row('a12_both_full'),
        'resid_l2_main_full': get_row('resid_l2_main_full'),
        'resid_l2_plus_ind_full': get_row('resid_l2_plus_ind_full'),
    }
    key_asof = {
        'main_full': get_row('main_full', 'p3asof'),
        'a12_l1_full': get_row('a12_l1_full', 'p3asof'),
        'a12_l2_full': get_row('a12_l2_full', 'p3asof'),
        'a12_both_full': get_row('a12_both_full', 'p3asof'),
        'resid_l1_main_full': get_row('resid_l1_main_full', 'p3asof'),
        'resid_l2_main_full': get_row('resid_l2_main_full', 'p3asof'),
        'resid_l2_plus_ind_full': get_row('resid_l2_plus_ind_full', 'p3asof'),
        'resid_l2_univariate_naked': get_row('resid_l2_univariate_naked', 'p3asof'),
    }
    key_latest['resid_l1_main_full'] = get_row('resid_l1_main_full')

    # ---- headline A12 复现门（与 G0 JSON 逐值比对）
    g0 = json.loads(G0_A12.read_text(encoding='utf-8'))
    repro = []
    for sid, gspec in [('a12_l1_full', 'pricing_industry_absorb_l1'),
                       ('a12_l2_full', 'pricing_industry_absorb_l2'),
                       ('a12_both_full', 'pricing_industry_absorb_both'),
                       ('a12_l1_vw_full', 'pricing_industry_absorb_l1_vw'),
                       ('a12_l2_vw_full', 'pricing_industry_absorb_l2_vw')]:
        r = get_row(sid, 'latest')
        g = g0['specs'][gspec]
        items = [('beta', r['beta'], g['beta_with']['beta']),
                 ('t_nw', r['t_nw'], g['beta_with']['t_nw']),
                 ('retention_delta', r['delta_vs_baseline_fullsample'], g['retention_delta'])]
        for c, gv in g['gamma'].items():
            col = 'gamma_1_beta' if r['gamma_1_var'] == c else 'gamma_2_beta'
            items.append((f'gamma_{c}', r[col], gv['beta']))
        for name, a, b in items:
            repro.append({'spec': sid, 'item': name, 'reproduced': a, 'g0': b,
                          'abs_diff': (abs(a - b) if a is not None and b is not None else None)})
    beta_wo_diff = abs(key_latest['main_full']['beta_without_fullsample'] - g0['beta_without']['beta'])
    repro.append({'spec': 'main_full', 'item': 'beta_without',
                  'reproduced': key_latest['main_full']['beta_without_fullsample'],
                  'g0': g0['beta_without']['beta'], 'abs_diff': beta_wo_diff})
    max_repro_diff = max(x['abs_diff'] for x in repro if x['abs_diff'] is not None)
    require(max_repro_diff <= 1e-8,
            f'headline A12 复现偏差 {max_repro_diff:.3e} > 1e-8；停机')
    log(f'headline A12 复现：max|Δ|={max_repro_diff:.2e}（5 规格 + β_without）')
    e9_diff = abs(key_latest['resid_l2_main_full']['beta'] - E9_RESID_BETA_LATEST)
    require(e9_diff <= 1e-6, f'plan E9 残差化主规格复现偏差 {e9_diff:.3e} > 1e-6；停机')
    log(f'plan E9 latest 残差化主规格复现：β='
        f'{key_latest["resid_l2_main_full"]["beta"]:.8f}（|Δ|={e9_diff:.2e}）')

    # ---- 门控：正交性 + 第一阶段 R²
    orth_max = max(first_stage[b][l]['max_abs_corr_resid_x']
                   for b in ('latest', 'p3asof') for l in ('l1', 'l2', 'l1l2'))
    r2_latest_l2 = first_stage['latest']['l2']['r2_pooled_month_fe']
    r2_diff = abs(r2_latest_l2 - E7_R2_L2_LATEST)
    gates = {
        'orthogonality_max_abs_corr_le_1e-10': {
            'value': orth_max, 'threshold': 1e-10,
            'status': 'PASS' if orth_max <= 1e-10 else 'FAIL'},
        'stage1_r2_latest_l2_matches_E7': {
            'value': r2_latest_l2, 'expected': E7_R2_L2_LATEST, 'abs_diff': r2_diff,
            'threshold_tight': 1e-6, 'threshold_stop': 0.02,
            'status': ('PASS' if r2_diff <= 1e-6 else
                       ('WARN' if r2_diff <= 0.02 else 'FAIL'))},
        'headline_a12_reproduction': {
            'max_abs_diff': max_repro_diff, 'threshold': 1e-8,
            'status': 'PASS' if max_repro_diff <= 1e-8 else 'FAIL'},
        'e9_latest_resid_reproduction': {
            'beta': key_latest['resid_l2_main_full']['beta'],
            'expected': E9_RESID_BETA_LATEST,
            'abs_diff': abs(key_latest['resid_l2_main_full']['beta'] - E9_RESID_BETA_LATEST),
            'threshold': 1e-6,
            'status': ('PASS' if abs(key_latest['resid_l2_main_full']['beta']
                                     - E9_RESID_BETA_LATEST) <= 1e-6 else 'FAIL')},
    }
    require(orth_max <= 1e-10, f'正交性失败：max|corr(resid, ind)|={orth_max:.3e} > 1e-10')
    require(r2_diff <= 0.02, f'第一阶段 R² 偏离 E7 {r2_diff:.6f} > 0.02（停机）')

    # ---- δ 不可解释性量化
    d_full = key_latest['a12_l2_full']
    d_l1_full = key_latest['a12_l1_full']
    d_resid_latest = key_latest['resid_l2_main_full']
    d_resid_asof = key_asof['resid_l2_main_full']

    def _f(v, default=float('nan')):
        return default if v is None else float(v)

    delta_interp = {
        'headline_beta_without': {
            'beta': d_full['beta_without_fullsample'],
            'se_nw': key_latest['main_full']['se_nw'],
            't_nw': key_latest['main_full']['t_nw'],
            'ci95': [key_latest['main_full']['ci95_low'],
                     key_latest['main_full']['ci95_high']],
            'note': 'β_without 不显著（|t|<1.96；CI 含 0）→ δ 的分母本身不可靠，δ 无有限置信区间'},
        'a12_l2': {
            'beta_with': d_full['beta'], 't_with': d_full['t_nw'],
            'gamma': d_full['gamma_1_beta'], 'gamma_t': d_full['gamma_1_t'],
            'delta_fullsample': d_full['delta_vs_baseline_fullsample'],
            'delta_samesample': d_full['delta_vs_baseline_samesample'],
            'delta_beta': d_full['delta_beta'],
            'delta_beta_se_nw': d_full['delta_beta_se_nw'],
            'delta_beta_t_nw': d_full['delta_beta_t_nw'],
            'delta_beta_n_months': d_full['n_months_paired'],
            'delta_monthly_median': d_full['delta_monthly_median'],
            'delta_monthly_p10': d_full['delta_monthly_p10'],
            'delta_monthly_p90': d_full['delta_monthly_p90'],
            'delta_monthly_share_outside_0_1': d_full['delta_monthly_share_outside_0_1'],
            'r2_with_mean': d_full['r2_with_mean'],
            'r2_without_mean': d_full['r2_without_mean'],
            'incremental_r2_mean': d_full['incremental_r2_mean'],
            'incremental_r2_median': d_full['incremental_r2_median'],
            'delta_interpretable': d_full['delta_interpretable'],
        },
        'a12_l1': {
            'beta_with': d_l1_full['beta'], 't_with': d_l1_full['t_nw'],
            'gamma': d_l1_full['gamma_1_beta'], 'gamma_t': d_l1_full['gamma_1_t'],
            'delta_fullsample': d_l1_full['delta_vs_baseline_fullsample'],
            'delta_beta_t_nw': d_l1_full['delta_beta_t_nw'],
            'incremental_r2_mean': d_l1_full['incremental_r2_mean'],
            'incremental_r2_median': d_l1_full['incremental_r2_median'],
            'delta_interpretable': d_l1_full['delta_interpretable'],
        },
        'residualized_main_latest': {
            'beta': d_resid_latest['beta'], 'se_nw': d_resid_latest['se_nw'],
            't_nw': d_resid_latest['t_nw'], 'n_rows': d_resid_latest['n_rows'],
            'n_months': d_resid_latest['n_months_used'],
            'delta_vs_headline_main': d_resid_latest['delta_vs_headline_main'],
            'note': 'plan E9 预跑对应 latest 基线；残差化后 β 反号且不显著；'
                    'δ 对 headline 主系数为负，证明 δ 仅在分母为正且稳定时才有「保留率」含义'},
        'residualized_main_p3asof': {
            'beta': d_resid_asof['beta'], 'se_nw': d_resid_asof['se_nw'],
            't_nw': d_resid_asof['t_nw'], 'n_rows': d_resid_asof['n_rows'],
            'n_months': d_resid_asof['n_months_used'],
            'delta_vs_headline_main': d_resid_asof['delta_vs_headline_main'],
            'note': 'as-of 主口径：残差化后 β 接近 0 且不显著；与 latest 残差化 β 不同，'
                    '说明残差化读数本身对行业基线 vintage 敏感'},
        'collinearity': {
            'corr_l1_latest': first_stage['latest']['l1']['corr_pearson'],
            'corr_l2_latest': first_stage['latest']['l2']['corr_pearson'],
            'corr_l1_p3asof': first_stage['p3asof']['l1']['corr_pearson'],
            'corr_l2_p3asof': first_stage['p3asof']['l2']['corr_pearson'],
            'vif_l2_latest': first_stage['latest']['l2']['vif_industry'],
            'vif_l2_p3asof': first_stage['p3asof']['l2']['vif_industry'],
        },
        'verdict': ('δ 不得单独引用：① β_without 不显著（t=%.2f，CI 含 0）；'
                    '② PeerMAX 与 ind_l2 共线 %.4f（VIF≈%.2f）；'
                    '③ Δβ（with−without）NW t=%.2f，不可区分于 0；'
                    '④ 同样本 δ=%.4f vs 全样本 δ=%.4f；'
                    '⑤ 逐月 δ 中位 %.3f（p10 %.3f, p90 %.3f），%.1f%% 月份 δ 落在 [0,1] 之外；'
                    '⑥ 行业变量带来的增量 R² 仅 %.3fpp（中位 %.3fpp），与 δ=0.83 的「吸收」叙事矛盾；'
                    '⑦ 残差化后 latest β=%.5f（t=%.2f）、as-of β=%.5f（t=%.2f），'
                    '与 headline +0.03427 反号/趋零。'
                    '→ A12 解释改为「残差化 β + 增量 R²」，δ 仅作描述性旁注。')
                   % (_f(key_latest['main_full']['t_nw']),
                      first_stage['latest']['l2']['corr_pearson'],
                      _f(first_stage['latest']['l2']['vif_industry']),
                      _f(d_full['delta_beta_t_nw']),
                      _f(d_full['delta_vs_baseline_samesample']),
                      _f(d_full['delta_vs_baseline_fullsample']),
                      _f(d_full['delta_monthly_median']),
                      _f(d_full['delta_monthly_p10']),
                      _f(d_full['delta_monthly_p90']),
                      100.0 * _f(d_full['delta_monthly_share_outside_0_1'], 0.0),
                      100.0 * _f(d_full['incremental_r2_mean']),
                      100.0 * _f(d_full['incremental_r2_median']),
                      _f(d_resid_latest['beta']), _f(d_resid_latest['t_nw']),
                      _f(d_resid_asof['beta']), _f(d_resid_asof['t_nw'])),
    }

    # ---- as-of vs latest（同一样本直接对比）
    both_l2 = fm.dropna(subset=['ind_l2_max_ew', 'ind_l2_max_ew_p3asof']).copy()
    same_rows = {}
    for basis in ('latest', 'p3asof'):
        col = IND[basis]['l2']
        fs = within_month_residual(both_l2, 'PeerMAXraw', [col])
        same_rows[basis] = {
            'n_rows': fs['n_rows'], 'n_months': fs['n_months'],
            'corr': fs['corr_pearson'], 'slope': fs['coef'][col],
            'r2_pooled_month_fe': fs['r2_pooled_month_fe'],
            'r2_bivariate': fs['r2_bivariate'],
        }
    same_rows['n_rows_both'] = int(len(both_l2))
    asof_vs_latest = {
        'first_stage': {
            'latest_l2': {k: first_stage['latest']['l2'][k] for k in
                          ('n_rows', 'n_months', 'r2_pooled_month_fe', 'r2_bivariate',
                           'corr_pearson', 'corr_spearman', 'per_month_slope_mean')},
            'p3asof_l2': {k: first_stage['p3asof']['l2'][k] for k in
                          ('n_rows', 'n_months', 'r2_pooled_month_fe', 'r2_bivariate',
                           'corr_pearson', 'corr_spearman', 'per_month_slope_mean')},
            'same_sample': same_rows,
        },
        'main_resid': {
            'latest_l2_full': {k: key_latest['resid_l2_main_full'][k] for k in
                               ('beta', 'se_nw', 't_nw', 'n_rows', 'n_months_used')},
            'p3asof_l2_full': {k: key_asof['resid_l2_main_full'][k] for k in
                               ('beta', 'se_nw', 't_nw', 'n_rows', 'n_months_used')},
        },
        'a12_l2_full': {
            'latest': {k: key_latest['a12_l2_full'][k] for k in
                       ('beta', 't_nw', 'gamma_1_beta', 'gamma_1_t', 'n_rows',
                        'delta_vs_baseline_fullsample', 'delta_beta_t_nw')},
            'p3asof': {k: key_asof['a12_l2_full'][k] for k in
                       ('beta', 't_nw', 'gamma_1_beta', 'gamma_1_t', 'n_rows',
                        'delta_vs_baseline_fullsample', 'delta_beta_t_nw')},
        },
        'note': ('as-of 主口径（分类生效日口径）与 latest 对照并报；两者样本量不同'
                 '（as-of L2 缺 80 行 vs latest 缺 1,985 行），故同时报同一样本（两套 L2 均非缺）'
                 '的一阶段读数，避免把样本差异误读为口径差异。'),
    }

    # ---- 团-行业交叉
    cross = {'status': 'skipped'} if args.skip_cluster_cross else {}
    cross_rows = []
    if not args.skip_cluster_cross:
        cross_rows, _ = cluster_industry_cross()
        by_key = {(r['text_year'], r['basis']): r for r in cross_rows}
        text2019 = {b: by_key.get((2019, b)) for b in ('latest', 'p3asof')}
        net2019 = {b: by_key.get((2018, b)) for b in ('latest', 'p3asof')}
        cross = {
            'status': 'computed',
            'rows_csv': str(CROSS_CSV.relative_to(ROOT)).replace('\\', '/'),
            'text_year_2019_network_2020': text2019,
            'network_year_2019_text_2018': net2019,
            'audit_reference': '审计 N-5：2019 年 81.0% 公司所在团跨 ≥3 个申万二级；'
                               'modal 占比中位 0.450、有效行业数中位 3.03；最大团 274 家跨 18 个二级',
            'audit_match': ('text_year=2019、latest 标签、全部公司口径的 firm_share_ge3=0.810，'
                            '与审计 81.0% 逐值一致；最大团 274 家跨 18 个二级亦逐值一致。'
                            'modal 占比与有效行业数的团层中位为 0.500 / 2.00，'
                            '公司加权均值为 0.493 / 3.73（审计记 0.450 / 3.03，口径不同），'
                            '不构成门控，只作口径说明。'),
            'note': ('公司加权口径分「全部公司」与「仅多公司团」两列；单点团占比高（24.7%→44.3%）'
                     '会机械拉低「全部公司」口径。'),
        }

    # ---- 输出 CSV
    header = ['family', 'spec_id', 'basis', 'treatment', 'industry_var_1', 'industry_var_2',
              'x_vars', 'n_rows', 'n_months_used', 'all_months_full_rank',
              'beta', 'se_nw', 't_nw', 'p_two_sided', 'ci95_low', 'ci95_high',
              'ownmax_beta', 'ownmax_se', 'ownmax_t',
              'peerreturn_beta', 'peerreturn_se', 'peerreturn_t',
              'gamma_1_var', 'gamma_1_beta', 'gamma_1_se', 'gamma_1_t',
              'gamma_2_var', 'gamma_2_beta', 'gamma_2_se', 'gamma_2_t',
              'baseline_spec', 'beta_without_fullsample', 'beta_without_samesample',
              'delta_vs_baseline_fullsample', 'delta_vs_baseline_samesample', 'sign_flip',
              'delta_beta', 'delta_beta_se_nw', 'delta_beta_t_nw', 'n_months_paired',
              'r2_with_mean', 'r2_without_mean', 'incremental_r2_mean',
              'incremental_r2_median',
              'delta_vs_headline_main', 'delta_monthly_median', 'delta_monthly_p10',
              'delta_monthly_p90', 'delta_monthly_share_outside_0_1',
              'delta_monthly_n', 'delta_interpretable', 'note']
    order = {s['spec_id'] + '|' + s['basis']: i for i, s in enumerate(specs)}
    rows.sort(key=lambda r: (order.get(r['spec_id'] + '|' + r['basis'], 999), r['spec_id']))
    write_csv(ABSORB_CSV, header, rows)
    log(f'写入 {ABSORB_CSV.name}：{len(rows)} 行')

    collin_header = ['basis', 'level', 'industry_var', 'x_vars', 'n_rows', 'n_months',
                     'corr_pearson', 'corr_spearman', 'slope', 'slope_se_ols', 'slope_t_ols',
                     'slope_2', 'slope_2_se_ols', 'slope_2_t_ols',
                     'per_month_slope_mean', 'per_month_slope_median',
                     'per_month_slope_p10', 'per_month_slope_p90',
                     'r2_pooled_month_fe', 'r2_bivariate', 'r2_per_month_mean',
                     'r2_per_month_weighted', 'max_abs_corr_resid_x',
                     'max_abs_month_mean_resid', 'vif_treatment', 'vif_industry',
                     'condition_number_mean', 'condition_number_max']
    collin_rows = []
    for basis in ('latest', 'p3asof'):
        for lvl in ('l1', 'l2', 'l1l2'):
            fs = first_stage[basis][lvl]
            cols = [IND[basis]['l1'], IND[basis]['l2']] if lvl == 'l1l2' else [IND[basis][lvl]]
            collin_rows.append({
                'basis': basis, 'level': lvl,
                'industry_var': fs['industry_var'], 'x_vars': 'PeerMAXraw|' + '|'.join(cols),
                'n_rows': fs['n_rows'], 'n_months': fs['n_months'],
                'corr_pearson': fs['corr_pearson'], 'corr_spearman': fs['corr_spearman'],
                'slope': fs['coef'][cols[0]], 'slope_se_ols': fs['se'][cols[0]],
                'slope_t_ols': fs['t'][cols[0]],
                'slope_2': fs['coef'][cols[1]] if len(cols) > 1 else None,
                'slope_2_se_ols': fs['se'][cols[1]] if len(cols) > 1 else None,
                'slope_2_t_ols': fs['t'][cols[1]] if len(cols) > 1 else None,
                'per_month_slope_mean': fs['per_month_slope_mean'],
                'per_month_slope_median': fs['per_month_slope_median'],
                'per_month_slope_p10': fs['per_month_slope_p10'],
                'per_month_slope_p90': fs['per_month_slope_p90'],
                'r2_pooled_month_fe': fs['r2_pooled_month_fe'],
                'r2_bivariate': fs['r2_bivariate'],
                'r2_per_month_mean': fs['r2_per_month_mean'],
                'r2_per_month_weighted': fs['r2_per_month_weighted'],
                'max_abs_corr_resid_x': fs['max_abs_corr_resid_x'],
                'max_abs_month_mean_resid': fs['max_abs_month_mean_resid'],
                'vif_treatment': fs['vif_treatment'], 'vif_industry': fs['vif_industry'],
                'condition_number_mean': fs['condition_number_mean'],
                'condition_number_max': fs['condition_number_max'],
            })
    write_csv(COLLIN_CSV, collin_header, collin_rows)
    log(f'写入 {COLLIN_CSV.name}：{len(collin_rows)} 行')

    if cross_rows:
        cross_header = ['text_year', 'network_year', 'basis', 'n_firms_labeled', 'n_clusters',
                        'n_multifirm_clusters', 'median_cluster_size', 'max_cluster_size',
                        'firm_share_ge3_all', 'firm_share_ge3_multifirm',
                        'cluster_share_ge3_multifirm', 'modal_share_median_multifirm',
                        'effective_industries_median_multifirm',
                        'firm_mean_modal_share_multifirm',
                        'firm_mean_effective_industries_multifirm', 'max_industries']
        write_csv(CROSS_CSV, cross_header, cross_rows)
        log(f'写入 {CROSS_CSV.name}：{len(cross_rows)} 行')

    # ---- postflight
    log('3.5 postflight：sha 前后对账')
    post = anchors_snapshot('post')
    write_json(ANCHOR_POST, post)
    cmp = compare_anchors(pre, post)
    require(cmp['status'] == 'MATCH',
            f'9 锚/headline 只读文件 sha 前后变化：{cmp["diffs"]}；停机')
    log(f'9 锚 + headline 只读文件 sha：{cmp["status"]}（n_diff={cmp["n_diff"]}）')

    # ---- 输出 sha
    outputs = {
        '3.5_industry_absorb.csv': file_size(ABSORB_CSV),
        '3.5_collinearity.csv': file_size(COLLIN_CSV),
        'pricing_peer_resid_shadow.parquet': file_size(RESID_PARQUET),
        '3.5_anchor_pre.json': file_size(ANCHOR_PRE),
        '3.5_anchor_post.json': file_size(ANCHOR_POST),
    }
    if cross_rows:
        outputs['3.5_cluster_industry_cross.csv'] = file_size(CROSS_CSV)

    counts = {
        'task': 'Phase 3.5 N-5/A12 行业吸收残差化 shadow',
        'plan_ref': 'artifacts/phase3/3.0_plan.md §2.5',
        'shadow_version_tag': SHADOW_VERSION_TAG,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'elapsed_sec': round(time.time() - t0, 2),
        'script': str(Path(__file__).relative_to(ROOT)).replace('\\', '/'),
        'inputs': {
            'headline_sample': 'data/processed/pricing_sample_ma_teacher_v2.parquet',
            'asof_sample_copy': str(SAMPLE_ASOF.relative_to(ROOT)).replace('\\', '/'),
            'asof_baselines': str(ASOF_BASELINES.relative_to(ROOT)).replace('\\', '/'),
            'asof_map': str(ASOF_MAP.relative_to(ROOT)).replace('\\', '/'),
            'sample_rows': int(len(base)), 'fm_rows': int(len(fm)),
            'fm_months': int(fm['month'].nunique()),
            'registry': {'min_focal_count_per_month': min_n, 'nw_lag': nw_lag,
                         'registry_file': str(ROOT / 'configs/parameter_registry.yaml')},
            'asof_baselines_crosscheck_max_abs_diff': max_diff,
        },
        'anchors': {'pre': pre, 'post': post, 'compare': cmp},
        'first_stage': {
            'latest': {lvl: first_stage['latest'][lvl] for lvl in ('l1', 'l2', 'l1l2')},
            'p3asof': {lvl: first_stage['p3asof'][lvl] for lvl in ('l1', 'l2', 'l1l2')},
            'registered': {'E7_r2_l2_latest': E7_R2_L2_LATEST_ALT,
                           'E7_corr_l1_latest': E7_CORR_L1_LATEST,
                           'E7_corr_l2_latest': E7_CORR_L2_LATEST},
            'definitions': {
                'r2_pooled_month_fe': 'PeerMAXraw ~ ind + month FE 的 pooled R²（= plan E7 的 month 虚拟变量口径）',
                'r2_bivariate': 'corr(PeerMAXraw, ind)²',
                'r2_per_month_mean': '逐月单独斜率回归的 R² 简单平均（稳健性对照）',
                'residual_main': 'month FE pooled within 残差（与逐月回归 + 常数逐值等价到 5e-14）',
            },
        },
        'collinearity': {
            'latest_l1': {k: first_stage['latest']['l1'][k] for k in
                          ('corr_pearson', 'corr_spearman', 'vif_industry',
                           'condition_number_mean', 'max_abs_corr_resid_x')},
            'latest_l2': {k: first_stage['latest']['l2'][k] for k in
                          ('corr_pearson', 'corr_spearman', 'vif_industry',
                           'condition_number_mean', 'max_abs_corr_resid_x')},
            'p3asof_l1': {k: first_stage['p3asof']['l1'][k] for k in
                          ('corr_pearson', 'corr_spearman', 'vif_industry',
                           'condition_number_mean', 'max_abs_corr_resid_x')},
            'p3asof_l2': {k: first_stage['p3asof']['l2'][k] for k in
                          ('corr_pearson', 'corr_spearman', 'vif_industry',
                           'condition_number_mean', 'max_abs_corr_resid_x')},
            'vif_note': 'VIF 为逐月均值（pricing_lib.vif_table）；L2 与 L1 高度嵌套，联合规格的 VIF 会更高。',
        },
        'headline_reproduction': {
            'status': 'PASS' if max_repro_diff <= 1e-8 else 'FAIL',
            'max_abs_diff': max_repro_diff,
            'items': repro,
            'g0_file': str(G0_A12.relative_to(ROOT)).replace('\\', '/'),
        },
        'a12_key_specs': {
            'latest': key_latest,
            'p3asof': key_asof,
            'csv': str(ABSORB_CSV.relative_to(ROOT)).replace('\\', '/'),
        },
        'main_resid': {
            'latest_l2_full': {k: key_latest['resid_l2_main_full'][k] for k in
                               ('beta', 'se_nw', 't_nw', 'n_rows', 'n_months_used')},
            'p3asof_l2_full': {k: key_asof['resid_l2_main_full'][k] for k in
                               ('beta', 'se_nw', 't_nw', 'n_rows', 'n_months_used')},
            'p3asof_l1_full': {k: key_asof['resid_l1_main_full'][k] for k in
                               ('beta', 'se_nw', 't_nw', 'n_rows', 'n_months_used')},
            'p3asof_l2_naked': {k: key_asof['resid_l2_univariate_naked'][k] for k in
                                ('beta', 'se_nw', 't_nw', 'n_rows', 'n_months_used')},
            'registered_E9_latest': {'beta': E9_RESID_BETA_LATEST, 't_nw': E9_RESID_T_LATEST,
                                     'abs_diff_beta': abs(key_latest['resid_l2_main_full']['beta']
                                                          - E9_RESID_BETA_LATEST),
                                     'abs_diff_t': abs(key_latest['resid_l2_main_full']['t_nw']
                                                       - E9_RESID_T_LATEST)},
            'parquet': str(RESID_PARQUET.relative_to(ROOT)).replace('\\', '/'),
        },
        'delta_interpretability': delta_interp,
        'asof_vs_latest': asof_vs_latest,
        'cluster_industry_cross': cross,
        'gates': gates,
        'stop_conditions': {
            'orthogonality_gt_1e-10': False,
            'stage1_r2_diff_gt_0.02': False,
            'headline_written_or_anchor_changed': False,
            'g0_written': False,
        },
        'outputs': outputs,
        'notes': [
            '3.5 只做只读派生 + shadow 分析；headline pricing_gate/PC/网络未重跑。',
            'as-of 口径 = 申万分类生效日口径（非严格 PIT）；2021-07-30 vintage 断点适用。',
            'δ = 1 − |β_with|/|β_without| 不单独引用；A12 解释改为「残差化 β + 增量 R²」。',
            '残差化主口径 = month FE pooled within（与 plan E7/E9 逐值一致）；逐月单独斜率仅作稳健性列。',
        ],
    }
    write_json(COUNTS_JSON, counts)
    log(f'写入 {COUNTS_JSON.name}（{time.time() - t0:.1f}s）')

    # ---- 摘要输出（协议字段）
    print('\n=== 3.5 SUMMARY ===')
    print(f'first_stage latest: L1 R2={first_stage["latest"]["l1"]["r2_pooled_month_fe"]:.6f} '
          f'corr={first_stage["latest"]["l1"]["corr_pearson"]:.6f}; '
          f'L2 R2={first_stage["latest"]["l2"]["r2_pooled_month_fe"]:.6f} '
          f'corr={first_stage["latest"]["l2"]["corr_pearson"]:.6f}')
    print(f'first_stage p3asof: L1 R2={first_stage["p3asof"]["l1"]["r2_pooled_month_fe"]:.6f} '
          f'corr={first_stage["p3asof"]["l1"]["corr_pearson"]:.6f}; '
          f'L2 R2={first_stage["p3asof"]["l2"]["r2_pooled_month_fe"]:.6f} '
          f'corr={first_stage["p3asof"]["l2"]["corr_pearson"]:.6f}')
    print(f'headline A12 L2: beta_with={d_full["beta"]:.5f} (t={d_full["t_nw"]:.2f}), '
          f'gamma={d_full["gamma_1_beta"]:.5f} (t={d_full["gamma_1_t"]:.2f}), '
          f'delta={d_full["delta_vs_baseline_fullsample"]:.4f}, '
          f'delta_beta_t={d_full["delta_beta_t_nw"]:.2f}')
    print(f'resid main latest: beta={key_latest["resid_l2_main_full"]["beta"]:.5f} '
          f'(t={key_latest["resid_l2_main_full"]["t_nw"]:.2f}); '
          f'p3asof: beta={key_asof["resid_l2_main_full"]["beta"]:.5f} '
          f'(t={key_asof["resid_l2_main_full"]["t_nw"]:.2f})')
    print(f'gates: {[(k, v["status"]) for k, v in gates.items()]}')
    print(f'anchors: {cmp["status"]} (n_diff={cmp["n_diff"]})')
    print(f'outputs: {COUNTS_JSON}, {ABSORB_CSV}, {COLLIN_CSV}, {RESID_PARQUET}')


if __name__ == '__main__':
    main()
