"""申万二级分类第三源核对（2026-09-15）。

背景
----
项目现行 `data/processed/sw2_industry_latest.csv` 由官方 xls 推导（L2 = 六位码前四位 + "00"），
已用同花顺 F10 抽样核对（`audit_sw2_with_ths.py`，加权一致率 94.26%）。
2026-09-15 获得第三份独立来源 `data/external/sw_industry_all_stkcds_20260915.xlsx`
（5,800 只股票，Stkcd + 申万一级/二级/三级名称，申万 2021 版，31/131/337 类），
**全样本覆盖**（同花顺那次是每 L2 组抽 min(4, size) 只的样本）。

本脚本只做核对与稳定性检验，**不替换、不修改**任何现行分类或网络产物。

四组检查
--------
R1 归档完整性：文件 bytes / 字节数 / 覆盖股票数 / 分类层级数
R2 映射可行性与覆盖：附件二级名称 → l2_code 的可映射比例；两表股票覆盖差集
R3 一致率与归因：L2 名称一致率（全样本 / 在市 / 退市）；不一致按
   naming_variant / counterpart_missing / ours_missing / delisted / classification_diff 归因
R4 结论稳定性（关键）：用第三源重算
   ① τ 校准目标（申万二级池化公司层面中位 peer 数，口径同 network_tau_calibrate.sw2_target）
   ② M-A / M-B 逐年"同申万二级边占比"与行业 lift
   与现行值比较，判断是否需要动 τ 或重建网络。

只读性保证：脚本运行前后对受保护文件做 bytes 比对，任何变动即 FAIL。

用法
----
  python scripts/industry_map_third_source_recheck.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fileio import file_size

ROOT = Path(__file__).resolve().parents[1]
ATT = ROOT / 'data/external/sw_industry_all_stkcds_20260915.xlsx'
LAT = ROOT / 'data/processed/sw2_industry_latest.csv'
NAMES = ROOT / 'data/processed/sw2_l2_names.csv'
MDA = ROOT / 'data/processed/mda_texts.csv'
NETS = {'ma': ROOT / 'data/processed/network_ma', 'mb': ROOT / 'data/processed/network_mb'}
OUT_JSON = ROOT / 'reports/data_gates/G0/sw2_third_source_audit.json'
OUT_CSV = ROOT / 'artifacts/review/sw2_third_source_disagreements.csv'

YEARS = list(range(2017, 2026))
LIFT_YEARS = [2017, 2021, 2025]
CURRENT_TAU_TARGET = 56.0
# 历史值：自建语料（mda_texts）校准的 τ。2026-09-22 语料切换后主口径为老师语料，
# τ = 0.0888212137（见 reports/data_gates/G0/network_tau_calibration_teacher.json）；
# 本脚本原为自建链的第三源稳定性审计，重跑主口径时须显式改用新 τ。
CURRENT_TAU = 0.07093029099632986

# 只读性保护清单：核对脚本绝不允许改动这些
PROTECTED = [LAT, NAMES,
             ROOT / 'data/processed/monthly_pricing_panel_ma.csv',
             ROOT / 'data/processed/monthly_pricing_panel_mb.csv',
             ROOT / 'configs/parameter_registry.yaml']


def sha(p: Path) -> str:
    return file_size(p)


def protected_hashes() -> dict:
    out = {}
    for p in PROTECTED:
        if p.exists():
            out[str(p.relative_to(ROOT)).replace(chr(92), '/')] = sha(p)
    for net, d in NETS.items():
        for f in sorted(d.glob('*_similarity.csv')):
            out[str(f.relative_to(ROOT)).replace(chr(92), '/')] = sha(f)
    return out


def multiset_key(s) -> str:
    s = str(s).replace('Ⅱ', '2').replace('Ⅲ', '3').replace('Ⅰ', '1')
    return ''.join(sorted(s))


def load_sources():
    att = pd.read_excel(ATT, dtype=str)
    att = att.rename(columns={'Stkcd': 'stkcd'})
    att['code6'] = att['stkcd'].str[:6]
    att['board'] = att['stkcd'].str[-2:]

    lat = pd.read_csv(LAT, dtype={'code6': str, 'l1_code': str, 'l2_code': str, 'l3_code': str})
    lat['code6'] = lat['code6'].str.zfill(6)
    nm = pd.read_csv(NAMES, dtype=str)
    lat['our_l2'] = lat['l2_code'].map(dict(zip(nm['l2_code'], nm['l2_name'])))
    lat['our_l1'] = lat['l2_code'].map(dict(zip(nm['l2_code'], nm['l1_name'])))
    return att, lat, nm


def r1_archive(att) -> dict:
    return {
        'file': str(ATT.relative_to(ROOT)).replace(chr(92), '/'),
        'bytes': ATT.stat().st_size,
        'bytes': sha(ATT),
        'n_rows': int(len(att)),
        'n_stocks': int(att['code6'].nunique()),
        'board_split': att['board'].value_counts().to_dict(),
        'n_l1': int(att['l1_name'].nunique(dropna=True)),
        'n_l2': int(att['l2_name'].nunique(dropna=True)),
        'n_l3': int(att['l3_name'].nunique(dropna=True)),
        'rows_without_classification': att.loc[att['l2_name'].isna(), 'stkcd'].tolist(),
    }


def r2_mapping(att, lat) -> dict:
    att_names = set(att['l2_name'].dropna())
    our_names = set(lat['our_l2'].dropna())
    unmappable = sorted(att_names - our_names)
    a, b = set(att['code6']), set(lat['code6'])
    return {
        'att_l2_names': len(att_names),
        'our_l2_names': len(our_names),
        'mappable_names': len(att_names & our_names),
        'unmappable_names': unmappable,
        'our_only_names': sorted(our_names - att_names),
        'n_att_stocks': len(a),
        'n_our_stocks': len(b),
        'both': len(a & b),
        'our_only': len(a - b),
        'att_only': len(b - a),
        'our_only_sample': sorted(a - b)[:15],
        'att_only_sample': sorted(b - a)[:15],
        'match_rate_att': round(len(a & b) / len(a), 4),
    }


def r3_agreement(att, lat) -> tuple[dict, pd.DataFrame]:
    m = lat.merge(att[['code6', 'l1_name', 'l2_name', 'l3_name']], on='code6', how='inner')
    m['same_l2'] = m['our_l2'] == m['l2_name']
    m['same_l1'] = m['our_l1'] == m['l1_name']
    m['is_delisted'] = m['out_date'].notna()

    def reason(r):
        if pd.isna(r['our_l2']):
            return 'ours_missing'
        if pd.isna(r['l2_name']):
            return 'counterpart_missing'
        if r['same_l2']:
            return 'agree'
        if multiset_key(r['our_l2']) == multiset_key(r['l2_name']):
            return 'naming_variant'
        if r['is_delisted']:
            return 'delisted'
        return 'classification_diff'

    m['reason'] = m.apply(reason, axis=1)
    comparable = m[m['reason'].isin(['agree', 'naming_variant', 'delisted', 'classification_diff'])]
    live = comparable[~comparable['is_delisted']]

    stats = {
        'n_comparable': int(len(comparable)),
        'agreement_l2_all': round(float(comparable['same_l2'].mean()), 4),
        'agreement_l1_all': round(float(comparable['same_l1'].mean()), 4),
        'n_live': int(len(live)),
        'agreement_l2_live': round(float(live['same_l2'].mean()), 4),
        'agreement_l1_live': round(float(live['same_l1'].mean()), 4),
        'n_delisted': int(comparable['is_delisted'].sum()),
        'agreement_l2_delisted': round(float(comparable[comparable['is_delisted']]['same_l2'].mean()), 4),
        'reason_counts': comparable['reason'].value_counts().to_dict(),
    }
    dis = comparable[comparable['reason'] != 'agree'][
        ['code6', 'name', 'board', 'is_delisted', 'reason', 'our_l1', 'our_l2', 'l1_name', 'l2_name', 'l3_name']]
    return stats, dis


def _mda_universe():
    d = pd.read_csv(MDA, usecols=['code', 'year', 'status'])
    d = d[d['status'] == 'ok'].copy()
    d['code6'] = d['code'].str[2:].str.zfill(6)
    return d[d['year'].isin(YEARS)]


def tau_target(keycol: str, mapping: pd.DataFrame, label: str, mda) -> dict:
    """口径与 network_tau_calibrate.sw2_target 完全一致。

    未被该源覆盖的 firm-year 归入 sentinel 组（保证池化中位数在两种源上可比），
    并单独报告未覆盖数；不静默丢样本。
    """
    m = mda.merge(mapping[['code6', keycol]], on='code6', how='left')
    n_unmatched = int(m[keycol].isna().sum())
    m[keycol] = m[keycol].fillna('__UNMATCHED__')
    pooled, per_year = [], []
    for y, g in m.groupby('year'):
        sizes = g.groupby(keycol)['code6'].size()
        peers = (g[keycol].map(sizes) - 1).astype(float)
        per_year.append({'text_year': int(y), 'firms': int(len(g)), 'groups': int(sizes.size),
                         'median_peer': float(peers.median())})
        pooled.append(peers.values)
    allp = np.concatenate(pooled)
    return {'label': label,
            'median_peer_pooled_firm_level': float(np.median(allp)),
            'n_firm_years': int(len(allp)),
            'n_firm_years_unmatched': n_unmatched,
            'per_year': per_year}


def industry_lift(net: str, year: int, mapping: dict) -> dict:
    f = NETS[net] / f'{year}_similarity.csv'
    e = pd.read_csv(f, usecols=['focal_id', 'source_id'])
    e['f'] = e['focal_id'].str[2:]
    e['s'] = e['source_id'].str[2:]
    v = e['f'].map(mapping)
    w = e['s'].map(mapping)
    ok = v.notna() & w.notna()
    share = float((v[ok] == w[ok]).mean())
    firms = sorted(set(e['f']) | set(e['s']))
    arr = np.array([mapping.get(x, np.nan) for x in firms], dtype=object)
    arr = arr[pd.notna(arr)]
    iu = np.triu_indices(len(arr), 1)
    base = float((arr[iu[0]] == arr[iu[1]]).mean())
    return {'n_edges': int(len(e)), 'match_rate': round(float(ok.mean()), 4),
            'same_sw2_share': round(share, 4), 'base_rate': round(base, 4),
            'lift': round(share / base, 2) if base else None}


def r4_stability(att, lat) -> dict:
    mda = _mda_universe()
    att_map = att[['code6', 'l2_name']].rename(columns={'l2_name': 'l2_key'}).dropna()
    cur = tau_target('l2_code', lat, 'current_l2_code', mda)
    new = tau_target('l2_key', att_map, 'third_source_l2_name', mda)

    lifts = {}
    cur_map = dict(zip(lat['code6'], lat['l2_code']))
    new_map = dict(zip(att['code6'], att['l2_name']))
    for net in ['ma', 'mb']:
        for y in LIFT_YEARS:
            lifts[f'{net}_{y}'] = {
                'current': industry_lift(net, y, cur_map),
                'third_source': industry_lift(net, y, new_map),
            }
    for k, v in lifts.items():
        c, n = v['current']['lift'], v['third_source']['lift']
        v['lift_delta_pct'] = round((n - c) / c * 100, 2) if c else None

    return {
        'tau_target_current': cur['median_peer_pooled_firm_level'],
        'tau_target_third_source': new['median_peer_pooled_firm_level'],
        'tau_target_unchanged': abs(cur['median_peer_pooled_firm_level']
                                    - new['median_peer_pooled_firm_level']) < 0.5,
        'registered_tau_target': CURRENT_TAU_TARGET,
        'registered_tau': CURRENT_TAU,
        'tau_target_detail': {'current': cur, 'third_source': new},
        'industry_lift': lifts,
        'max_abs_lift_delta_pct': max(abs(v['lift_delta_pct'] or 0) for v in lifts.values()),
    }


def main() -> int:
    before = protected_hashes()
    att, lat, nm = load_sources()

    rep = {
        'run_at': datetime.now(timezone.utc).isoformat(),
        'purpose': '申万二级分类第三源核对与结论稳定性检验（只读，不替换任何现行分类）',
        'source': 'all_stkcds.xlsx（用户 2026-09-15 提供，微信传入；已归档 data/external/）',
        'current_mapping': {
            'file': str(LAT.relative_to(ROOT)).replace(chr(92), '/'),
            'bytes': sha(LAT),
            'derivation': '官方 xls；L2 = 六位码前四位 + "00"',
            'prior_cross_check': '同花顺 F10 抽样（523 只），加权一致率 0.9426',
        },
    }
    rep['R1_archive'] = r1_archive(att)
    rep['R2_mapping'] = r2_mapping(att, lat)
    rep['R3_agreement'], dis = r3_agreement(att, lat)
    rep['R4_stability'] = r4_stability(att, lat)

    after = protected_hashes()
    changed = {k: {'before': before.get(k), 'after': after.get(k)}
               for k in set(before) | set(after) if before.get(k) != after.get(k)}
    rep['R5_readonly'] = {'protected_files': len(before), 'changed': changed,
                          'unchanged': len(changed) == 0}

    checks = {
        'archive_registered': rep['R1_archive']['n_stocks'] > 0 and len(rep['R1_archive']['bytes']) == 64,
        'names_fully_mappable': len(rep['R2_mapping']['unmappable_names']) == 0,
        'coverage_high': rep['R2_mapping']['match_rate_att'] >= 0.99,
        'agreement_live_above_95pct': rep['R3_agreement']['agreement_l2_live'] >= 0.95,
        'tau_target_unchanged': rep['R4_stability']['tau_target_unchanged'],
        'lift_stable_under_5pct': rep['R4_stability']['max_abs_lift_delta_pct'] < 5.0,
        'readonly_verified': rep['R5_readonly']['unchanged'],
    }
    rep['checks'] = checks
    rep['status'] = 'PASS' if all(checks.values()) else 'FAIL'

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    dis.to_csv(OUT_CSV, index=False, encoding='utf-8')

    print(json.dumps(checks, ensure_ascii=False, indent=1))
    print(f"tau target: current={rep['R4_stability']['tau_target_current']} "
          f"third={rep['R4_stability']['tau_target_third_source']}")
    print(f"max |lift delta| = {rep['R4_stability']['max_abs_lift_delta_pct']}%")
    print(f"L2 agreement: all={rep['R3_agreement']['agreement_l2_all']} "
          f"live={rep['R3_agreement']['agreement_l2_live']}")
    print('status =', rep['status'], '->', OUT_JSON.relative_to(ROOT))
    return 0 if rep['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
