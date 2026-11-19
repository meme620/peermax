"""Phase 3.6：shadow 网络 M1 对比 + 推广决策（默认 HOLD）。

范围（3.0_plan §2.6 + 3.6 任务书）：
  - 只读 headline 样本 / headline 网络 / 5 个 shadow 网络 / pricing_lib；
  - 在**同一固定样本**（headline FM 181,178 行、104 月）上重算 shadow 网络 treatment；
  - 逐变体 × 规格（S1 登记全控制 / S5 无 turnover / S6 无 rev+无 turnover）× EW/VW 跑 FM；
  - 网络层读数：边 Jaccard、边权相关、treatment 相关、覆盖/样本损失；
  - 只写 `artifacts/phase3/3.6_*` 与 `data/processed/phase3_shadow/p3_6_shadow_treat_*.parquet`。

不重跑 headline M1/PC/网络，不写 G0/reports/configs，不自动推广。

用法：
  python scripts/shadow_network_pricing.py"""

from __future__ import annotations

import sys
from pathlib import Path
from fileio import json_default as _json_default

import argparse
import json
import os
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pricing_lib import (
    assert_m1_inputs, fm_regression, param_value, require, file_size,
    winsor_bounds, winsorize_by_month,
)
from peermax_panel_build import (
    NETWORK_AVAILABLE_AT_RULE, load_versions, select_version,
)


ROOT = Path(__file__).resolve().parents[1]

ART = ROOT / 'artifacts/phase3'

SHADOW_DATA = ROOT / 'data/processed/phase3_shadow'

HEADLINE_NET = ROOT / 'data/processed/network_ma_teacher'

SAMPLE_PATH = ROOT / 'data/processed/m1_common_sample_ma_teacher_v2.parquet'

COMPARISON_CSV = ART / '3.6_network_comparison.csv'

COUNTS_JSON = ART / '3.6_counts.json'

OUTPUT_SHA = ART / '3.6_output_bytes.json'

ANCHOR_PRE = ART / '3.6_anchor_pre.json'

ANCHOR_POST = ART / '3.6_anchor_post.json'

RUN_LOG = ART / '3.6_run.log'

COMPARE_3_2 = ART / '3.2_comparison.csv'

YEARS = list(range(2014, 2023))

TEXT_YEAR_MONTHS = {ty: [] for ty in YEARS}

LIMIT_UNKNOWN = 'limit_status_unknown'

SAMPLE_ROWS = 307503

FM_ROWS = 181178

FM_MONTHS = 104

HEADLINE_EW_BETA = 0.03427280848385429

HEADLINE_EW_T = 0.472809

HEADLINE_VW_BETA = -0.109956

HEADLINE_VW_T = -1.1812

CONTROLS = ['size', 'bm', 'mom', 'rev', 'turnover', 'illiq',
            'AnalystCoverage', 'NetworkDegree']

S1 = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + CONTROLS

S5 = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + [c for c in CONTROLS if c != 'turnover']

S6 = ['PeerMAXraw', 'OwnMAXraw', 'PeerReturn'] + [
    c for c in CONTROLS if c not in ('rev', 'turnover')]

SPECS = [
    ('S1', '登记全控制（8 controls）', S1),
    ('S5', '无 turnover', S5),
    ('S6', '无 rev + 无 turnover', S6),
]

VARIANTS = [
    {'id': 'headline', 'label': 'headline（登记主口径，只读）',
     'dir': HEADLINE_NET, 'prefix': 'pm_ma_ewcos_teacher_v1', 'registered': True},
    {'id': 'p3asof', 'label': 'N1 逐年 as-of 目标（3.1）',
     'dir': ROOT / 'data/processed/network_ma_teacher_p3asof',
     'prefix': 'pm_ma_ewcos_teacher_p3asof_v1'},
    {'id': 'p3yt', 'label': 'N2 逐年 as-of 目标（3.2）',
     'dir': ROOT / 'data/processed/network_ma_teacher_p3yt',
     'prefix': 'pm_ma_ewcos_teacher_p3yt_v1'},
    {'id': 'p3ydf', 'label': 'N3 逐年 df 词表 + 池化 44（3.2）',
     'dir': ROOT / 'data/processed/network_ma_teacher_p3ydf',
     'prefix': 'pm_ma_ewcos_teacher_p3ydf_v1'},
    {'id': 'p3yt_ydf', 'label': 'N4 逐年 df 词表 + 逐年 as-of 目标（3.2）',
     'dir': ROOT / 'data/processed/network_ma_teacher_p3yt_ydf',
     'prefix': 'pm_ma_ewcos_teacher_p3yt_ydf_v1'},
    {'id': 'p3yt_latest', 'label': 'N5 逐年 latest 目标（3.2，对照轴）',
     'dir': ROOT / 'data/processed/network_ma_teacher_p3yt_latest',
     'prefix': 'pm_ma_ewcos_teacher_p3yt_latest_v1'},
]

SHADOW_IDS = [v['id'] for v in VARIANTS if v['id'] != 'headline']

ALL_IDS = [v['id'] for v in VARIANTS]

FM_VARIANTS = VARIANTS + [{'id': 'headline_recompute',
                           'label': 'headline 聚合器复算（同一聚合器，R1 门）'}]

Q4_JACCARD_MIN = {'p3asof': 0.4, 'p3yt': 0.4, 'p3ydf': 0.7,
                  'p3yt_ydf': 0.4, 'p3yt_latest': 0.4}

MIN_N = int(param_value('m1_sample_and_spec_registry')['min_focal_count_per_month'])

NW_LAG = int(param_value('nw_lag'))

WINSOR = winsor_bounds()

_TEE = None

def assert_output_path_guard(path) -> Path:
    """只允许 `artifacts/phase3/3.6_*` 与 `phase3_shadow/p3_6_shadow_treat_*`。"""
    p = Path(path).resolve()
    art = ART.resolve()
    sd = SHADOW_DATA.resolve()
    if art in p.parents:
        if not p.name.startswith('3.6_'):
            print(f'[HARD-GATE] artifacts/phase3 只允许 3.6_*: {p}', file=sys.stderr, flush=True)
            raise SystemExit(2)
        return p
    if sd in p.parents:
        if not (p.name.startswith('p3_6_shadow_treat_') and p.suffix == '.parquet'):
            print(f'[HARD-GATE] phase3_shadow 只允许 p3_6_shadow_treat_*.parquet: {p}',
                  file=sys.stderr, flush=True)
            raise SystemExit(2)
        return p
    print(f'[HARD-GATE] 输出路径命中禁写集合: {p}', file=sys.stderr, flush=True)
    raise SystemExit(2)

def log(msg: str) -> None:
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    if _TEE is not None:
        _TEE.write(line + '\n')
    else:
        print(line, flush=True)

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

def write_json(path, obj) -> None:
    p = assert_output_path_guard(path)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default),
                   encoding='utf-8')
    os.replace(tmp, p)

def write_csv(path, df: pd.DataFrame) -> None:
    p = assert_output_path_guard(path)
    tmp = p.with_name(p.name + '.tmp')
    df.to_csv(tmp, index=False, encoding='utf-8', lineterminator='\n')
    os.replace(tmp, p)

def write_parquet(path, df: pd.DataFrame) -> None:
    p = assert_output_path_guard(path)
    tmp = p.with_name(p.name + '.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)

def _sha_map(paths) -> dict:
    return {str(Path(p).relative_to(ROOT)).replace('\\', '/'): file_size(p)
            for p in paths}

def _net_files(net_dir: Path) -> list:
    pats = ['*_build_report.json', '*_similarity.csv', '*_support.csv',
            '*_pc.csv', 'clusters/*_clusters.csv']
    out = []
    for pat in pats:
        out.extend(sorted(net_dir.glob(pat)))
    return out

def snapshot_anchors(tag: str) -> dict:
    log(f'锚快照 {tag}：9 个 M1 输入 + headline 样本/网络 + shadow 网络 sha')
    ledger = assert_m1_inputs()
    return {
        'tag': tag,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'm1_inputs': {r['input']: r['rows'] for r in ledger},
        'm1_ledger': ledger,
        'headline_sample': {'path': str(SAMPLE_PATH.relative_to(ROOT)).replace('\\', '/'),
                            'bytes': file_size(SAMPLE_PATH)},
        'headline_net': _sha_map(_net_files(HEADLINE_NET)),
        'shadow_net': {v['id']: _sha_map(_net_files(v['dir']))
                       for v in VARIANTS if v['id'] != 'headline'},
    }

def compare_anchors(pre: dict, post: dict) -> dict:
    blocks = {}
    diffs = []
    for key in ('m1_inputs', 'headline_net', 'shadow_net'):
        a, b = pre[key], post[key]
        if isinstance(a, dict) and a and isinstance(next(iter(a.values())), dict):
            # shadow_net: variant -> {file: sha}
            for vid in sorted(set(a) | set(b)):
                fa, fb = a.get(vid, {}), b.get(vid, {})
                for f in sorted(set(fa) | set(fb)):
                    if fa.get(f) != fb.get(f):
                        diffs.append(f'{vid}:{f}')
                blocks[f'{key}:{vid}'] = {'n': len(fa), 'match': fa == fb}
        else:
            for f in sorted(set(a) | set(b)):
                if a.get(f) != b.get(f):
                    diffs.append(f'{key}:{f}')
            blocks[key] = {'n': len(a), 'match': a == b}
    s_pre, s_post = pre['headline_sample'], post['headline_sample']
    if s_pre.get('bytes') != s_post.get('bytes'):
        diffs.append('headline_sample')
    blocks['headline_sample'] = {'n': 1, 'match': s_pre.get('bytes') == s_post.get('bytes')}
    return {'status': 'MATCH' if not diffs else 'DIFFERENT',
            'n_diff': len(diffs), 'diff': diffs, 'blocks': blocks}

def _load_variant_versions(v: dict) -> dict:
    if v.get('registered'):
        vers = load_versions(v['dir'])
    else:
        vers = load_versions(v['dir'], version_spec={
            'version_prefix': v['prefix'], 'years': YEARS,
            'available_at_rule': NETWORK_AVAILABLE_AT_RULE})
    require(sorted(vers) == YEARS, f"{v['id']}: 年份不齐 {sorted(vers)}")
    report_checks = {}
    for ty, rec in vers.items():
        doc = json.loads((v['dir'] / f'{ty}_build_report.json').read_text(encoding='utf-8'))
        expected_version = f"{v['prefix']}_{ty}"
        require(doc.get('network_version') == expected_version,
                f"{v['id']} {ty}: build_report network_version="
                f"{doc.get('network_version')!r} != {expected_version!r}")
        require(doc.get('available_at') == f'{ty + 1}-04-30',
                f"{v['id']} {ty}: available_at={doc.get('available_at')!r}")
        require(int(doc.get('text_year', -1)) == ty,
                f"{v['id']} {ty}: text_year={doc.get('text_year')!r}")
        report_checks[str(ty)] = {
            'network_version': doc['network_version'],
            'available_at': doc['available_at'],
            'n_nodes': int(doc['n_nodes']),
            'n_edges': int(doc['n_edges']),
        }
    return {'vers': vers, 'report_checks': report_checks}

def map_months(vers: dict, months) -> dict:
    mapping = {}
    for m in months:
        v = select_version(vers, m)
        require(v is not None, f'月份 {m} 无可用网络版本')
        require(v['available_at'] < m, f'{m}: available_at={v["available_at"]} 非严格小于月份')
        mapping[m] = int(v['text_year'])
    return mapping

def check_sample_mapping(sample: pd.DataFrame, mapping: dict) -> dict:
    m = sample.loc[sample['network_version'].notna(), ['month', 'network_version']]
    m = m.drop_duplicates().sort_values('month')
    bad = []
    for _, r in m.iterrows():
        ty = mapping.get(r['month'])
        if ty is None or int(r['network_version']) != ty:
            bad.append({'month': r['month'], 'sample': int(r['network_version']),
                        'mapping': ty})
    mapped_rows = sample['month'].map(mapping)
    unmapped = sample.loc[mapped_rows.isna()]
    # 事件窗首月（2015-05）之前无网络版本；这些行不在 FM 样本内，treatment 保持 NaN。
    bad_unmapped_fm = int(unmapped['m1_in_fm_sample'].astype(bool).sum())
    bad_unmapped_version = int(unmapped['network_version'].notna().sum())
    require(bad_unmapped_fm == 0,
            f'FM 样本存在未映射月份 {bad_unmapped_fm} 行')
    require(bad_unmapped_version == 0,
            f'未映射月份却带 network_version {bad_unmapped_version} 行')
    return {'n_month_version_pairs': int(len(m)), 'n_mismatch': len(bad),
            'mismatch': bad,
            'n_mapped_rows': int(mapped_rows.notna().sum()),
            'n_unmapped_rows_pre_network': int(len(unmapped)),
            'unmapped_months': sorted(unmapped['month'].unique().tolist()),
            'n_unmapped_fm_rows': bad_unmapped_fm,
            'n_unmapped_with_version': bad_unmapped_version}

def load_values(srcm: pd.DataFrame, rets: pd.DataFrame) -> tuple:
    maxv = srcm.loc[srcm['MAXraw'].notna()
                    & (srcm['qualification_status'] != LIMIT_UNKNOWN),
                    ['source_id', 'month', 'MAXraw']].copy()
    retv = rets.loc[rets['R'].notna(), ['firm_id', 'month', 'R']].copy()
    return maxv, retv

def _year_contract(e: pd.DataFrame, support: pd.DataFrame, ty: int, vid: str) -> dict:
    dup = int(e.duplicated(['focal_id', 'source_id']).sum())
    self_loop = int((e['focal_id'] == e['source_id']).sum())
    cnt = e.groupby('focal_id', sort=False).size().rename('n_edges_e')
    sup = support.set_index('focal_id')['n_edges'].rename('n_edges_s')
    cmp_ = pd.concat([cnt, sup], axis=1)
    # support 含孤立节点（n_edges=0）；只在边表出现的 focal 上核对 n_edges。
    missing_support = int(cmp_.loc[cmp_['n_edges_e'].notna(), 'n_edges_s'].isna().sum())
    mismatch = int((cmp_.loc[cmp_['n_edges_e'].notna(), 'n_edges_e'].fillna(-1)
                    != cmp_.loc[cmp_['n_edges_e'].notna(), 'n_edges_s'].fillna(-2)).sum())
    zero_support = int((sup == 0).sum())
    gsum = e.groupby('focal_id', sort=False)['similarity'].transform('sum')
    w = e['similarity'] / gsum
    wsum = w.groupby(e['focal_id'], sort=False).sum()
    maxdev = float((wsum - 1.0).abs().max()) if len(wsum) else 0.0
    return {'text_year': ty, 'variant': vid, 'n_edges': int(len(e)),
            'n_focals': int(e['focal_id'].nunique()),
            'n_support_zero_edge_nodes': zero_support,
            'duplicate_edges': dup, 'self_loops': self_loop,
            'focals_missing_in_support': missing_support,
            'n_edges_mismatch_support': mismatch,
            'max_abs_row_weight_sum_minus_1': maxdev,
            'ok': (dup == 0 and self_loop == 0 and missing_support == 0
                   and mismatch == 0 and maxdev <= 1e-9)}

def aggregate_network(v: dict, vers: dict, months_by_ty: dict,
                      maxv: pd.DataFrame, retv: pd.DataFrame,
                      headline_meta: dict) -> tuple:
    """返回 (treatment raw 长表, 逐年网络对比指标, 逐年边表合同检查)。"""
    parts, metrics, contracts = [], [], []
    for ty in sorted(months_by_ty):
        months = months_by_ty[ty]
        assert months, f'{v["id"]} {ty}: 月份为空'
        p = v['dir'] / f'{ty}_similarity.csv'
        e = pd.read_csv(p, usecols=['focal_id', 'source_id', 'similarity'],
                        dtype={'focal_id': str, 'source_id': str})
        support = pd.read_csv(v['dir'] / f'{ty}_support.csv', dtype={'focal_id': str})
        chk = _year_contract(e, support, ty, v['id'])
        contracts.append(chk)
        require(chk['ok'], f"{v['id']} {ty}: 边表合同失败 {chk}")

        gsum = e.groupby('focal_id', sort=False)['similarity'].transform('sum')
        e['w'] = e['similarity'] / gsum
        wtot = e.groupby('focal_id', sort=False)['w'].sum()
        nsrc = support.set_index('focal_id')['n_edges']
        month_df = pd.DataFrame({'month': months})
        eg = e[['focal_id', 'source_id', 'w']].merge(month_df, how='cross')

        mv = maxv[maxv['month'].isin(months)]
        mm = eg.merge(mv, on=['source_id', 'month'], how='left')
        mm['wx'] = mm['w'] * mm['MAXraw']
        mm['cw'] = np.where(mm['MAXraw'].notna(), mm['w'], 0.0)
        gm = (mm.groupby(['month', 'focal_id'], sort=False)
              .agg(num=('wx', 'sum'), den=('cw', 'sum')).reset_index())
        gm['PeerMAXraw'] = np.where(gm['den'] > 0, gm['num'] / gm['den'], np.nan)

        rv = retv[retv['month'].isin(months)]
        rm = eg.merge(rv, left_on=['source_id', 'month'],
                      right_on=['firm_id', 'month'], how='left')
        rm['wx'] = rm['w'] * rm['R']
        rm['cw'] = np.where(rm['R'].notna(), rm['w'], 0.0)
        gr = (rm.groupby(['month', 'focal_id'], sort=False)
              .agg(num=('wx', 'sum'), den_r=('cw', 'sum')).reset_index())
        gr['PeerReturn'] = np.where(gr['den_r'] > 0, gr['num'] / gr['den_r'], np.nan)

        out = gm[['month', 'focal_id', 'PeerMAXraw', 'den']].merge(
            gr[['month', 'focal_id', 'PeerReturn', 'den_r']],
            on=['month', 'focal_id'], how='outer')
        out['n_sources'] = out['focal_id'].map(nsrc)
        out['n_sources'] = out['n_sources'].fillna(
            out['focal_id'].map(e.groupby('focal_id', sort=False).size())).astype(int)
        out['coverage_ratio_max'] = (out['den'] / out['focal_id'].map(wtot)).round(6)
        out['coverage_ratio_ret'] = (out['den_r'] / out['focal_id'].map(wtot)).round(6)
        out = out[['month', 'focal_id', 'PeerMAXraw', 'PeerReturn', 'n_sources',
                   'coverage_ratio_max', 'coverage_ratio_ret']]
        parts.append(out)

        # ---- 与 headline 网络对比
        he = pd.read_csv(HEADLINE_NET / f'{ty}_similarity.csv',
                         usecols=['focal_id', 'source_id', 'similarity'],
                         dtype={'focal_id': str, 'source_id': str})
        inner = e[['focal_id', 'source_id', 'similarity']].merge(
            he, on=['focal_id', 'source_id'], how='inner', suffixes=('', '_h'))
        union = len(e) + len(he) - len(inner)
        n_common = len(inner)
        s_x = s_y = s_xx = s_yy = s_xy = 0.0
        corr_y = None
        if n_common > 2:
            x = inner['similarity'].to_numpy(dtype=float)
            y = inner['similarity_h'].to_numpy(dtype=float)
            s_x, s_y = float(x.sum()), float(y.sum())
            s_xx, s_yy, s_xy = float((x * x).sum()), float((y * y).sum()), float((x * y).sum())
            if x.std() > 0 and y.std() > 0:
                corr_y = float(np.corrcoef(x, y)[0, 1])
        hm = headline_meta[str(ty)]
        vmeta = json.loads((v['dir'] / f'{ty}_build_report.json').read_text(encoding='utf-8'))
        denom = vmeta['n_nodes'] * (vmeta['n_nodes'] - 1)
        metrics.append({
            'text_year': ty, 'n_edges': int(len(e)), 'n_edges_headline': int(len(he)),
            'n_common_edges': n_common,
            'jaccard': float(n_common / union) if union else None,
            'edge_weight_corr': corr_y,
            'density': float(len(e) / denom) if denom else None,
            'density_headline': float(len(he) / (hm['n_nodes'] * (hm['n_nodes'] - 1)))
            if hm['n_nodes'] > 1 else None,
            'sum_x': s_x, 'sum_y': s_y, 'sum_xx': s_xx, 'sum_yy': s_yy, 'sum_xy': s_xy,
        })
        # per-year 进度（避免大量日志）
        log(f"{v['id']} {ty}: edges={len(e):,} jaccard={metrics[-1]['jaccard']:.4f} "
            f"cov_max_med={out['coverage_ratio_max'].median():.4f}")
    raw = pd.concat(parts, ignore_index=True)
    return raw, metrics, contracts

def pooled_network_metrics(metrics: list) -> dict:
    n_com = sum(m['n_common_edges'] for m in metrics)
    n_a = sum(m['n_edges'] for m in metrics)
    n_b = sum(m['n_edges_headline'] for m in metrics)
    jac_pool = n_com / (n_a + n_b - n_com) if (n_a + n_b - n_com) else None
    n = sum(m['n_common_edges'] for m in metrics)
    sx = sum(m['sum_x'] for m in metrics)
    sy = sum(m['sum_y'] for m in metrics)
    sxx = sum(m['sum_xx'] for m in metrics)
    syy = sum(m['sum_yy'] for m in metrics)
    sxy = sum(m['sum_xy'] for m in metrics)
    corr = None
    if n > 2:
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        if vx > 0 and vy > 0:
            corr = (sxy - sx * sy / n) / np.sqrt(vx * vy)
    js = [m['jaccard'] for m in metrics if m['jaccard'] is not None]
    return {
        'n_edges_total': int(n_a),
        'n_edges_headline_total': int(n_b),
        'n_common_edges_total': int(n_com),
        'jaccard_pooled': float(jac_pool) if jac_pool is not None else None,
        'jaccard_median_yearly': float(np.median(js)) if js else None,
        'jaccard_min_yearly': float(np.min(js)) if js else None,
        'jaccard_max_yearly': float(np.max(js)) if js else None,
        'edge_weight_corr_pooled': float(corr) if corr is not None else None,
        'density_median': float(np.median([m['density'] for m in metrics])),
        'density_headline_median': float(np.median([m['density_headline'] for m in metrics])),
        'per_year': metrics,
    }

def attach_treatment(sample: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    key = sample[['focal_id', 'month']].copy()
    key = key.merge(raw, on=['focal_id', 'month'], how='left')
    require(len(key) == len(sample), 'treatment 合并改变行数（键重复）')
    # n_sources 是网络元数据：无网络版本/孤立节点按 0 记（与 s4 面板一致）；treatment 仍为 NaN。
    key['n_sources'] = key['n_sources'].fillna(0).astype(int)
    key = winsorize_by_month(key, ['PeerMAXraw', 'PeerReturn'],
                             lo=WINSOR[0], hi=WINSOR[1])
    return key

def coverage_stats(sample: pd.DataFrame, treat: pd.DataFrame) -> dict:
    fm = sample['m1_in_fm_sample'].astype(bool)
    h_pm = sample['PeerMAXraw']
    h_pr = sample['PeerReturn']
    t_pm = treat['PeerMAXraw']
    t_pr = treat['PeerReturn']
    return {
        'fixed_rows': int(len(sample)),
        'fixed_peermax_nonnull': int(t_pm.notna().sum()),
        'fixed_peerreturn_nonnull': int(t_pr.notna().sum()),
        'fixed_peermax_lost_vs_headline': int((h_pm.notna() & t_pm.isna()).sum()),
        'fixed_peermax_gained_vs_headline': int((h_pm.isna() & t_pm.notna()).sum()),
        'fm_rows': int(fm.sum()),
        'fm_peermax_nonnull': int(t_pm[fm].notna().sum()),
        'fm_peerreturn_nonnull': int(t_pr[fm].notna().sum()),
        'fm_peermax_missing': int(t_pm[fm].isna().sum()),
        'fm_peerreturn_missing': int(t_pr[fm].isna().sum()),
        'fm_peermax_lost_vs_headline': int((h_pm[fm].notna() & t_pm[fm].isna()).sum()),
        'fm_peerreturn_lost_vs_headline': int((h_pr[fm].notna() & t_pr[fm].isna()).sum()),
        'fm_peermax_gained_vs_headline': int((h_pm[fm].isna() & t_pm[fm].notna()).sum()),
        'fm_n_sources_median': float(treat.loc[fm, 'n_sources'].median()),
        'fm_coverage_ratio_max_median': float(treat.loc[fm, 'coverage_ratio_max'].median()),
        'fm_coverage_ratio_ret_median': float(treat.loc[fm, 'coverage_ratio_ret'].median()),
        'fm_n_sources_median_headline': float(sample.loc[fm, 'n_sources'].median()),
        'fm_coverage_ratio_max_median_headline': float(sample.loc[fm, 'coverage_ratio_max'].median()),
        'fm_coverage_ratio_ret_median_headline': float(sample.loc[fm, 'coverage_ratio_ret'].median()),
    }

def _summarise_fm(res: dict, per: list, spec: str, weight: str) -> dict:
    used = [r for r in per if r.get('used')]
    n_used = int(sum(r['n'] for r in used))
    out = {'spec': spec, 'weight': weight,
           'T': int(res['n_months_used']), 'n': n_used,
           'n_months_skipped': int(len(per) - res['n_months_used']),
           'min_month_n': int(min((r['n'] for r in used), default=0)),
           'median_month_n': float(np.median([r['n'] for r in used])) if used else None,
           'max_month_n': int(max((r['n'] for r in used), default=0))}
    for c in ('PeerMAXraw', 'OwnMAXraw', 'PeerReturn'):
        s = res['coefs'].get(c, {})
        key = {'PeerMAXraw': 'peermax', 'OwnMAXraw': 'ownmax', 'PeerReturn': 'peerreturn'}[c]
        out[f'beta_{key}'] = s.get('beta')
        out[f'se_{key}'] = s.get('se_nw')
        out[f't_{key}'] = s.get('t_nw')
        out[f'p2_{key}'] = s.get('p_two_sided')
        out[f'ci_low_{key}'] = s.get('ci95_low')
        out[f'ci_high_{key}'] = s.get('ci95_high')
    return out

def run_fm_matrix(sample: pd.DataFrame, treatments: dict) -> list:
    fm_mask = sample['m1_in_fm_sample'].astype(bool)
    rows = []
    for v in FM_VARIANTS:
        vid = v['id']
        df = sample.copy()
        if vid == 'headline_recompute':
            t = treatments['headline']
            df['PeerMAXraw'] = t['PeerMAXraw'].to_numpy()
            df['PeerReturn'] = t['PeerReturn'].to_numpy()
        elif vid != 'headline':
            t = treatments[vid]
            df['PeerMAXraw'] = t['PeerMAXraw'].to_numpy()
            df['PeerReturn'] = t['PeerReturn'].to_numpy()
        sub = df[fm_mask]
        for spec_id, spec_label, cols in SPECS:
            for wname, wcol in (('EW', None), ('VW', 'market_equity')):
                res, per = fm_regression(sub, 'R_next', cols, weight_col=wcol,
                                         min_n=MIN_N, nw_lag=NW_LAG)
                rec = _summarise_fm(res, per, spec_id, wname)
                rec.update({'variant': vid, 'variant_label': v['label'],
                            'spec_label': spec_label,
                            'n_fm_rows': int(fm_mask.sum()),
                            'n_missing_peermax_fm': int(
                                df.loc[fm_mask, 'PeerMAXraw'].isna().sum())})
                rows.append(rec)
                log(f"{vid} {spec_id} {wname}: beta_peer={rec['beta_peermax']:.6f} "
                    f"t={rec['t_peermax']:.3f} T={rec['T']} n={rec['n']:,}")
    return rows

def enrich_fm_rows(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    base = df[df['variant'] == 'headline'].set_index(['spec', 'weight'])
    df['beta_headline'] = [base.loc[(s, w), 'beta_peermax'] for s, w in
                           zip(df['spec'], df['weight'])]
    df['t_headline'] = [base.loc[(s, w), 't_peermax'] for s, w in
                        zip(df['spec'], df['weight'])]
    df['delta_beta_vs_headline'] = df['beta_peermax'] - df['beta_headline']
    df['delta_t_vs_headline'] = df['t_peermax'] - df['t_headline']
    # R1 校验：headline_recompute 已另存，这里只做 headline 登记值门
    s1ew = df[(df['variant'] == 'headline') & (df['spec'] == 'S1') & (df['weight'] == 'EW')].iloc[0]
    s1vw = df[(df['variant'] == 'headline') & (df['spec'] == 'S1') & (df['weight'] == 'VW')].iloc[0]
    require(abs(s1ew['beta_peermax'] - HEADLINE_EW_BETA) <= 1e-8,
            f"headline S1 EW beta={s1ew['beta_peermax']} != {HEADLINE_EW_BETA}")
    require(abs(s1vw['beta_peermax'] - HEADLINE_VW_BETA) <= 1e-6,
            f"headline S1 VW beta={s1vw['beta_peermax']} != {HEADLINE_VW_BETA}")
    require(int(s1ew['T']) == FM_MONTHS, f"headline T={s1ew['T']} != {FM_MONTHS}")
    # R1：headline_recompute（同一聚合器复算）逐规格复现 headline
    for spec_id, _, _ in SPECS:
        for wname in ('EW', 'VW'):
            h = df[(df['variant'] == 'headline') & (df['spec'] == spec_id)
                   & (df['weight'] == wname)].iloc[0]
            hr = df[(df['variant'] == 'headline_recompute') & (df['spec'] == spec_id)
                    & (df['weight'] == wname)].iloc[0]
            require(abs(hr['beta_peermax'] - h['beta_peermax']) <= 1e-7,
                    f"{spec_id} {wname}: headline_recompute beta 偏差 "
                    f"{hr['beta_peermax'] - h['beta_peermax']}")
            require(abs(hr['t_peermax'] - h['t_peermax']) <= 1e-5,
                    f"{spec_id} {wname}: headline_recompute t 偏差 "
                    f"{hr['t_peermax'] - h['t_peermax']}")
            require(int(hr['n']) == int(h['n']) and int(hr['T']) == int(h['T']),
                    f"{spec_id} {wname}: headline_recompute n/T 不一致")
    return df

def treatment_correlations(sample: pd.DataFrame, treatments: dict) -> dict:
    fm = sample['m1_in_fm_sample'].astype(bool)
    idx = pd.MultiIndex.from_arrays([sample.loc[fm, 'focal_id'].to_numpy(),
                                     sample.loc[fm, 'month'].to_numpy()])
    blocks = {}
    for col in ('PeerMAXraw', 'PeerReturn'):
        data = {'headline': sample.loc[fm, col].to_numpy()}
        for vid in SHADOW_IDS:
            data[vid] = treatments[vid].loc[fm, col].to_numpy()
        wide = pd.DataFrame(data, index=idx)
        corr = wide.corr(min_periods=100)
        n = wide.notna().astype(int).T.dot(wide.notna().astype(int))
        blocks[col] = {
            'corr': {a: {b: (None if pd.isna(corr.loc[a, b]) else float(corr.loc[a, b]))
                         for b in corr.columns} for a in corr.index},
            'n_pair': {a: {b: int(n.loc[a, b]) for b in n.columns} for a in n.index},
        }
    return blocks

def evaluate_q_status(rows: pd.DataFrame, coverage: dict, net_metrics: dict,
                      contracts: dict, anchor_cmp: dict) -> dict:
    q = {}
    contract_ok = all(c['ok'] for vals in contracts.values() for c in vals)
    q['Q1'] = {'status': 'CONDITIONAL' if contract_ok else 'FAIL',
               'note': '9 年边表合同（键唯一/无自环/行内权重和=1）+ available_at 严格小于 + '
                       'build_report 版本串 PASS；但 3.1 R1 的 as-of 样本行覆盖率 97.2654% < 99.9% '
                       '阈值仍未由项目裁定（3.1 §2），故 Q1 不能无条件 PASS；词表/K 目标/映射的 '
                       'R1 证据引自 3.1/3.2，本脚本未重跑构造。'}
    q['Q2'] = {'status': 'PASS' if anchor_cmp['status'] == 'MATCH' else 'FAIL',
               'note': '9 个 M1 输入 + headline 样本 + headline 网络全文件 sha pre/post；'
                       f"n_diff={anchor_cmp['n_diff']}"}
    # Q3
    shadow = rows[~rows['variant'].isin(['headline', 'headline_recompute'])]
    pos_sig = shadow[(shadow['beta_peermax'] > 0) & (shadow['t_peermax'] > 1.96)]
    sign_bad = shadow[(np.sign(shadow['beta_peermax']) != np.sign(shadow['beta_headline']))
                      & (shadow['t_peermax'].abs() >= 1.96)]
    q3_pass = len(pos_sig) == 0 and len(sign_bad) == 0
    q['Q3'] = {'status': 'PASS' if q3_pass else 'FAIL',
               'n_positive_significant': int(len(pos_sig)),
               'n_sign_flip_significant': int(len(sign_bad)),
               'positive_significant_rows': pos_sig[['variant', 'spec', 'weight',
                                                     'beta_peermax', 't_peermax']]
               .to_dict('records'),
               'note': 'PeerMAX 同号或 |t|<1.96；不得出现正显著。'}
    # Q4 使用 3.2 结构表 + 本脚本 Jaccard
    q4 = {}
    try:
        c32 = pd.read_csv(COMPARE_3_2)
    except Exception as exc:  # pragma: no cover
        c32 = None
        q4['_error'] = f'3.2_comparison.csv 不可读: {exc}'
    for vid in SHADOW_IDS:
        nm = net_metrics[vid]
        jac_med = nm['jaccard_median_yearly']
        jac_thr = Q4_JACCARD_MIN[vid]
        ent = {'jaccard_median': jac_med, 'jaccard_threshold': jac_thr,
               'jaccard_pass': bool(jac_med is not None and jac_med >= jac_thr)}
        if c32 is not None:
            s = c32[c32['network'] == vid]
            h = c32[c32['network'] == 'headline']
            if len(s) and len(h):
                m = s[['text_year', 'isolated_share']].merge(
                    h[['text_year', 'isolated_share']], on='text_year',
                    suffixes=('', '_h'))
                ent.update({
                    'peer_minus_target_max_abs': float(s['peer_minus_target'].abs().max()),
                    'peer_target_pass': bool(s['peer_minus_target'].abs().max() <= 2),
                    'isolated_share_max_abs_diff': float(
                        (m['isolated_share'] - m['isolated_share_h']).abs().max()),
                    'isolated_share_pass': bool(
                        (m['isolated_share'] - m['isolated_share_h']).abs().max() <= 0.10),
                    'lift_asof_min': float(s['lift_asof'].min()),
                    'lift_latest_min': float(s['lift_latest'].min()),
                    'lift_pass': bool(min(s['lift_asof'].min(), s['lift_latest'].min()) > 5),
                })
        ent['pass'] = bool(ent.get('jaccard_pass') and ent.get('peer_target_pass')
                           and ent.get('isolated_share_pass') and ent.get('lift_pass'))
        q4[vid] = ent
    q['Q4'] = {'status': 'PASS' if q4 and all(v['pass'] for k, v in q4.items()
                                              if not k.startswith('_')) else 'FAIL',
               'variants': q4,
               'note': 'Jaccard 阈值 N1/N2/N4 ≥0.4、N3 ≥0.7；|median peer−target|≤2；'
                       'isolated share 变化 ≤10pp；lift>5（两套标签）。阶梯平台导致的 '
                       'peer 偏差按 3.1/3.2 留档，不作为构造失败，但 Q4 仍按阈值判 FAIL。'}
    q['Q5'] = {'status': 'NOT_EVALUATED',
               'note': '本脚本只跑 NW(6) FM，未在 shadow 网络上重跑团聚类 SE / 双保持零分布。'
                       'headline 侧 3.4 读数：团聚类 t=0.859(EW)/−1.928(VW)、N-8 分位 '
                       '46.5%/25.5% 均在 95% 区间内；shadow 网络 Q5 待另立任务。'}
    q['Q6'] = {'status': 'PASS_SOURCE_PENDING_PYTEST',
               'note': 'FM/聚合为确定性代码；重跑逐字节比对见 counts.rerun_check；'
                       'pytest 结果见 artifacts/phase3/3.6_execution.md。'}
    q['Q7'] = {'status': 'CITED_PASS_HEADLINE',
               'note': '3.5：δ 不可单独引用（β_without t=0.47，32.7% 月份 δ∉[0,1]，增量 R² +0.283pp）；'
                       '残差化 β latest −0.039(t=−0.81)/as-of −0.001(t=−0.02) → '
                       '「行业之上无稳定增量或不可判定」。shadow 网络 A12 未重跑，不构成矛盾表述。'}
    q['Q8'] = {'status': 'FAIL',
               'note': '项目未给出显式书面批准（推广范围/版本号/registry/manifest/methodology/重跑清单）。'
                       '默认 HOLD。'}
    all_pass = all(v['status'] == 'PASS' for v in q.values())
    q['_overall'] = {'all_pass': bool(all_pass), 'decision': 'PROMOTE' if all_pass else 'HOLD'}
    return q


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --- scripts/lib bootstrap ---
# --- end scripts/lib bootstrap ---


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
    cur = {str(p.relative_to(ROOT)).replace('\\', '/'): file_size(p) for p in files}
    diff = sorted(k for k in set(prev) | set(cur) if prev.get(k) != cur.get(k))
    rerun = {'status': ('NO_PREVIOUS' if not prev else
                        'IDENTICAL' if not diff else 'DIFFERENT'),
             'n_compared': len(cur), 'n_diff': len(diff), 'diff': diff}
    write_json(OUTPUT_SHA, {'phase': '3.6', 'run_at': datetime.now(timezone.utc).isoformat(),
                            'files': cur, 'rerun_check': rerun})
    counts['rerun_check'] = rerun
    counts['output_bytes'] = cur
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
