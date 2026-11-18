"""p3_5_industry_absorb 的辅助函数与常量（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from datetime import datetime, timezone
from lib.io import json_default as _json_default
from pathlib import Path
from s3_m1_lib import (  # noqa: E402
    assert_m1_inputs_sha, fm_regression, newey_west, param_value,
    require, sha256_file, vif_table,
)
import csv
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time

ROOT = Path(__file__).resolve().parents[2]

ART = ROOT / 'artifacts/phase3'

SHADOW_DIR = ROOT / 'data/processed/phase3_shadow'

HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'

REPORTS = ROOT / 'reports'

CONFIGS = ROOT / 'configs'

SAMPLE = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'

HEADLINE_PANEL = ROOT / 'data/processed/monthly_pricing_panel_ma_teacher.csv'

FIRM_CHARS = ROOT / 'data/processed/firm_chars_clean.csv'

G0_A12 = ROOT / 'reports/data_gates/G0/s3_m1_industry_absorb.json'

ASOF_MAP = SHADOW_DIR / 'sw2_asof_map_p3shadow.parquet'

ASOF_BASELINES = SHADOW_DIR / 'ind_baselines_p3asof.parquet'

SAMPLE_ASOF = SHADOW_DIR / 'm1_common_sample_ma_teacher_p3asof_baselines.parquet'

SW2_MAP = ROOT / 'data/processed/sw2_industry_map.csv'

SW2_LATEST = ROOT / 'data/processed/sw2_industry_latest.csv'

COUNTS_JSON = ART / '3.5_counts.json'

ABSORB_CSV = ART / '3.5_industry_absorb.csv'

COLLIN_CSV = ART / '3.5_collinearity.csv'

CROSS_CSV = ART / '3.5_cluster_industry_cross.csv'

ANCHOR_PRE = ART / '3.5_anchor_pre.json'

ANCHOR_POST = ART / '3.5_anchor_post.json'

RESID_PARQUET = SHADOW_DIR / 'm1_peer_resid_p3shadow.parquet'

SHADOW_VERSION_TAG = 'p3shadow_v1_industry_absorb'

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
    p = Path(path).resolve()
    allowed = any(p == Path(a).resolve() or Path(a).resolve() in p.parents
                  for a in ALLOWED_ROOTS)
    banned = [Path(b).resolve() for b in BANNED_ROOTS]
    banned += [Path(SAMPLE).resolve(), Path(HEADLINE_PANEL).resolve(),
               Path(FIRM_CHARS).resolve(), Path(G0_A12).resolve(),
               Path(ASOF_MAP).resolve(), Path(ASOF_BASELINES).resolve(),
               Path(SAMPLE_ASOF).resolve(), Path(SW2_MAP).resolve(),
               Path(SW2_LATEST).resolve()]
    hit = [b for b in banned if p == b or b in p.parents]
    if not allowed or hit:
        print(f'[HARD-GATE] 输出路径命中禁写集合: {p}（allowed={allowed} banned={hit}）',
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    return p

def _atomic_text(path, text: str) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, p)

def write_json(path, obj) -> None:
    _atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default))

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
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(r.get(k)) for k in header])
    os.replace(tmp, p)

def write_parquet(path, df: pd.DataFrame) -> None:
    p = assert_output_path_guard(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp.parquet')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)

def log(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

def anchors_snapshot(tag: str) -> dict:
    ledger = assert_m1_inputs_sha()
    extra = {k: {'file': str(v.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha256_file(v)}
             for k, v in EXTRA_INPUTS.items()}
    clusters = {}
    for y in YEARS:
        p = HEADLINE_NET / 'clusters' / f'{y}_clusters.csv'
        clusters[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    snap = {
        'tag': tag,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'm1_inputs_9': {r['input']: r['sha256'] for r in ledger},
        'm1_inputs_ledger': ledger,
        'extra_inputs': extra,
        'headline_clusters_sha': clusters,
    }
    return snap

def compare_anchors(pre: dict, post: dict) -> dict:
    diffs = []
    for sect in ('m1_inputs_9', 'headline_clusters_sha'):
        a, b = pre.get(sect) or {}, post.get(sect) or {}
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                diffs.append({'section': sect, 'key': k, 'pre': a.get(k), 'post': b.get(k)})
    a = {k: v.get('sha256') for k, v in (pre.get('extra_inputs') or {}).items()}
    b = {k: v.get('sha256') for k, v in (post.get('extra_inputs') or {}).items()}
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

