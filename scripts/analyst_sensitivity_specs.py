"""analyst-network 敏感性规格（三轮对抗 review 所列敏感性规格落地，2026-08-12 参数冻结后执行）。

变体维度：
  min_common      : {1, 2, 5}  连接阈值（A&H 基线=1；>=2/>=5 为稳健性，无发表先例仅作补充）
  weight          : count_rownorm / ln_rownorm
                      count_rownorm = c / sum c           （A&H 2020 式 1 行归一化，主口径）
                      ln_rownorm   = ln(1+c)/sum ln(1+c)  （Long et al. 2025 先例）
  no_super_analysts : full / no_top1   剔除跨月（全样本 distinct 公司）覆盖数>300 家的超级分析师（v2 实际 6 个，按实际动态剔除）
                      ⚠️ 历史标签 exclude_top1pct 与实际不符（2026-08-14 审查）：真 top-1% 为每月 28-37 人、
                      剔除后边数 −21.4%（degree 152→120），过于激进，不作为报告变体；正式名称 no_super_analysts，
                      输出 key 保持 no_top1 兼容下游引用
  identity        : analyst（主）/ broker_analyst（复合键，daily-panel，IBES 口径近似）

输出:
  data/processed/analyst/sensitivity_specs.json        边层变体跨月汇总 + 抽查月
  data/processed/analyst/sensitivity_broker_key.json   复合键变体月度统计
"""
import json, math
from pathlib import Path
from collections import defaultdict
import pandas as pd

COV = 'data/processed/analyst/firm_analyst_coverage.csv'
CLEAN = 'data/processed/analyst/analyst_step3_clean.csv'  # broker_key 变体用行级输入（v2 起与主口径同 FY 语义）
OUTD = Path('data/processed/analyst')
TOP_EXCLUDE_THRESHOLD = 300
SPOTCHECK_MONTHS = ['2018-06-30', '2021-06-30', '2024-06-30']
MIN_COMMONS = [1, 2, 5]
WEIGHTS = ['count_rownorm', 'ln_rownorm']


def build_pairs(firms_by_analyst):
    """analyst -> set(firms) 投影为 pair 计数（无向、无自环）。
    returns pair_count
    """
    pair_count = defaultdict(int)
    for aid, firms in firms_by_analyst.items():
        fl = sorted(firms)
        for a in range(len(fl)):
            for b in range(a + 1, len(fl)):
                i, j = fl[a], fl[b]
                pair_count[(i, j)] += 1
    return pair_count


def percentile(lst, p):
    if not lst:
        return None
    return lst[int((len(lst) - 1) * p)]


def edge_stats(pair_count, min_common, weight_mode):
    edges = {k: v for k, v in pair_count.items() if v >= min_common}
    if not edges:
        return {'undirected_edges': 0, 'degree_p50': None, 'degree_mean': None,
                'weak_edge_ratio': None, 'top1_weight_p50': None, 'weight_hhi_p50': None}
    denom = defaultdict(float)
    for (i, j), c in edges.items():
        if weight_mode == 'count_rownorm':
            denom[i] += c; denom[j] += c
        else:  # ln_rownorm
            denom[i] += math.log1p(c); denom[j] += math.log1p(c)
    focal_weights = defaultdict(list)
    weak = 0
    for (i, j), c in edges.items():
        if c == 1:
            weak += 1
        if weight_mode == 'count_rownorm':
            wi = c / denom[i] if denom[i] > 0 else 0.0
            wj = c / denom[j] if denom[j] > 0 else 0.0
        else:  # ln_rownorm
            wi = math.log1p(c) / denom[i] if denom[i] > 0 else 0.0
            wj = math.log1p(c) / denom[j] if denom[j] > 0 else 0.0
        focal_weights[i].append(wi)
        focal_weights[j].append(wj)
    deg = defaultdict(int)
    for (i, j) in edges:
        deg[i] += 1; deg[j] += 1
    degs = sorted(deg.values())
    top1 = sorted(max(v) for v in focal_weights.values())
    hhi = sorted(sum(x * x for x in v) for v in focal_weights.values())
    return {
        'undirected_edges': len(edges),
        'degree_p50': percentile(degs, 0.50),
        'degree_mean': round(sum(degs) / len(degs), 2) if degs else None,
        'weak_edge_ratio': round(weak / len(edges), 4) if edges else None,
        'top1_weight_p50': round(percentile(top1, 0.50), 6) if top1 else None,
        'weight_hhi_p50': round(percentile(hhi, 0.50), 6) if hhi else None,
    }


def main():
    cov = pd.read_csv(COV, dtype=str, keep_default_na=False,
                      usecols=['network_cutoff', 'firm_id', 'analyst_id'])
    analyst_firm_count = cov.groupby('analyst_id')['firm_id'].nunique()
    top_analysts = set(analyst_firm_count[analyst_firm_count > TOP_EXCLUDE_THRESHOLD].index)
    print(f'top-analyst exclusion set (> {TOP_EXCLUDE_THRESHOLD} firms): {len(top_analysts)}', flush=True)

    variants = {}
    for mc in MIN_COMMONS:
        for wm in WEIGHTS:
            for ex in ('full', 'no_top1'):
                variants[f'{mc}|{wm}|{ex}'] = []

    spot = {}
    for cutoff in sorted(cov['network_cutoff'].unique()):
        sub = cov[cov['network_cutoff'] == cutoff]
        for exclude in (False, True):
            excluded_sample = sub[~sub['analyst_id'].isin(top_analysts)] if exclude else sub
            fa = defaultdict(set)
            for fid, aid in zip(excluded_sample['firm_id'], excluded_sample['analyst_id']):
                fa[aid].add(fid)
            pair_count = build_pairs(fa)
            for mc in MIN_COMMONS:
                for wm in WEIGHTS:
                    st = edge_stats(pair_count, mc, wm)
                    key = f'{mc}|{wm}|{"full" if not exclude else "no_top1"}'
                    variants[key].append(st)
        if cutoff in SPOTCHECK_MONTHS:
            spot[cutoff] = {}
            for mc in MIN_COMMONS:
                for wm in WEIGHTS:
                    for ex in ('full', 'no_top1'):
                        key = f'{mc}|{wm}|{ex}'
                        spot[cutoff][key] = variants[key][-1]
        if len(variants['1|count_rownorm|full']) % 12 == 0:
            print(f"cutoff {cutoff} done ({len(variants['1|count_rownorm|full'])} months)", flush=True)

    def agg(lst):
        if not lst:
            return None
        def col(k):
            vals = [x[k] for x in lst if x[k] is not None]
            return round(sum(vals) / len(vals), 6) if vals else None
        return {
            'undirected_edges_mean': round(sum(x['undirected_edges'] for x in lst) / len(lst), 1),
            'undirected_edges_min': min(x['undirected_edges'] for x in lst),
            'undirected_edges_max': max(x['undirected_edges'] for x in lst),
            'degree_p50_mean': col('degree_p50'),
            'degree_mean_mean': col('degree_mean'),
            'weak_edge_ratio_mean': col('weak_edge_ratio'),
            'top1_weight_p50_mean': col('top1_weight_p50'),
            'weight_hhi_p50_mean': col('weight_hhi_p50'),
        }

    summary = {k: agg(v) for k, v in variants.items()}
    rep = {
        'note': '变体 key = min_common|weight|exclusion；主口径 = 1|count_rownorm|full。'
                'ln_rownorm 参考 Long et al. 2025。',
        'top_analyst_exclusion_threshold': TOP_EXCLUDE_THRESHOLD,
        'top_analyst_excluded_count': len(top_analysts),
        'summary': summary,
        'spotcheck_months': spot,
    }
    (OUTD / 'sensitivity_specs.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE specs', flush=True)


def broker_key_monthly():
    """daily-panel 复合键变体：identity = (broker_code, analyst_id)，近似 IBES broker 内 estimator 口径。
    仅输出 min_common=1、count_rownorm 的月度统计，与主口径对比边数/degree。
    v2 变更：输入改为行级 clean 数据，并补齐 mapping_status=='ok' 与 A_code 格式校验
    （^A\\d{5}$），FY1/FY2 为行级过滤（先于报告级去重），与主口径 v2 完全同语义。
    """
    rec = pd.read_csv(CLEAN, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'analyst_id', 'broker_code', 'Anndats', 'Fpenddt', 'mapping_status'])
    rec = rec[rec['mapping_status'] == 'ok']
    rec = rec[rec['analyst_id'].str.match(r'^A\d{5}$', na=False)]   # 与主口径 step5 相同的格式校验
    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    rec['Fpenddt'] = pd.to_datetime(rec['Fpenddt'], errors='coerce')
    rec['gap'] = rec['Fpenddt'].dt.year - rec['report_date'].dt.year
    rec = rec[rec['gap'].between(0, 1)]   # FY1/FY2 行级过滤，同主口径 v2
    rec['bkey'] = rec['broker_code'] + '_' + rec['analyst_id']
    out = []
    cutoffs = pd.date_range('2017-12-31', '2025-12-31', freq='ME')
    for cutoff in cutoffs:
        ws = cutoff - pd.DateOffset(months=12)
        sub = rec[(rec['report_date'] > ws) & (rec['report_date'] <= cutoff)]
        fa = defaultdict(set)
        for fid, bk in zip(sub['firm_id'], sub['bkey']):
            fa[bk].add(fid)
        pair_count = build_pairs(fa)
        st = edge_stats(pair_count, 1, 'count_rownorm')
        st['cutoff'] = cutoff.strftime('%Y-%m-%d')
        out.append(st)
    return out


if __name__ == '__main__':
    main()
    bk = broker_key_monthly()
    (OUTD / 'sensitivity_broker_key.json').write_text(
        json.dumps({'identity': 'broker_analyst', 'min_common': 1, 'weight': 'count_rownorm',
                    'note': '复合键 (broker_code, analyst_id)，近似 IBES 口径；与主口径 1|count_rownorm|full 对比',
                    'months': bk}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE broker_key', flush=True)
