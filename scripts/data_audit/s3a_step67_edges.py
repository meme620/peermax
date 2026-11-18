"""S3A Step 6+7：二部图投影 -> 共同分析师边 -> 权重（按月落盘）。

规则（execution_plan S3A Step 5-6）:
  common_analyst_count_{ij,t} = |A_i,t ∩ A_j,t|, i != j
  无向、无自环；边表输出 (i,j) 与 (j,i) 双向
  wA_{ij,t} = n_ij / sum_k n_ik   （focal 行归一化）
  原始强度对称，权重可不对称
  denom=0 -> no-network-support（无共同分析师的公司不在边表，诊断中计数）

输出:
  data/processed/analyst/edges/edges_<network_version>_<cutoff>.csv   每月双向边表（v2 起文件名带版本）
  data/processed/analyst/step67_diagnostics.json    月度诊断累计
"""
import json, csv
from pathlib import Path
from collections import defaultdict
import pandas as pd

COV = 'data/processed/analyst/firm_analyst_coverage.csv'
EDGED = Path('data/processed/analyst/edges')
EDGED.mkdir(parents=True, exist_ok=True)
USE_CASE = 'shared_analyst'
NET_VER = 'coverage_win12_fy12_v3'  # 与 step5 VERSION 同源；写入 network_version 列并进入文件名
SRC_VER = 'forecast_shard_v1_20260811'
# network_support_status 恒为 'support'（2026-08-14 合同说明）：no-network-support 公司
# （denom=0，无共同分析师边）不进边表，其状态由 step8 诊断 no_network_support_monthly_* 报告；
# 该列仅对边表内行有效，不作为状态列使用。
COLS = ['network_use_case', 'network_version', 'focal_id', 'source_id',
        'common_analyst_count', 'link_strength', 'analyst_count_focal',
        'analyst_count_source', 'union_analyst_count', 'jaccard', 'weight',
        'SourceFocalDistinct', 'network_cutoff', 'available_at', 'source_version',
        'coverage_flag', 'missing_reason', 'network_support_status']

def main():
    cov = pd.read_csv(COV, dtype=str,
                      usecols=['network_cutoff', 'firm_id', 'analyst_id'])
    cutoffs = sorted(cov['network_cutoff'].unique())
    print(f"cutoffs: {len(cutoffs)}", flush=True)
    diag = {'months': []}

    for cutoff in cutoffs:
        sub = cov[cov['network_cutoff'] == cutoff]
        # analyst -> set(firms)
        af = defaultdict(set)
        for fid, aid in zip(sub['firm_id'], sub['analyst_id']):
            af[aid].add(fid)
        firm_analyst_count = defaultdict(int)
        for fid, aid in zip(sub['firm_id'], sub['analyst_id']):
            firm_analyst_count[fid] += 1

        # pair 计数
        pair_count = defaultdict(int)
        for aid, firms in af.items():
            fl = sorted(firms)
            for a in range(len(fl)):
                for b in range(a + 1, len(fl)):
                    i, j = fl[a], fl[b]
                    pair_count[(i, j)] += 1

        # 行归一化权重
        denom = defaultdict(int)
        for (i, j), c in pair_count.items():
            denom[i] += c
            denom[j] += c

        focal_weights = defaultdict(list)   # focal -> 归一化权重列表（诊断用）
        path = EDGED / f'edges_{NET_VER}_{cutoff}.csv'
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for (i, j), c in sorted(pair_count.items()):
                ni = firm_analyst_count[i]; nj = firm_analyst_count[j]
                union = ni + nj - c
                jac = c / union if union > 0 else 0.0
                wi = c / denom[i] if denom[i] > 0 else None
                wj = c / denom[j] if denom[j] > 0 else None
                if wi is not None:
                    focal_weights[i].append(wi)
                if wj is not None:
                    focal_weights[j].append(wj)
                for focal, src, wt in ((i, j, wi), (j, i, wj)):
                    w.writerow([USE_CASE, NET_VER, focal, src, c, c, ni if focal == i else nj,
                                nj if focal == i else ni, union, round(jac, 8),
                                round(wt, 8) if wt is not None else '', 1,
                                cutoff, cutoff, SRC_VER, 'covered', '', 'support'])

        n_edges = len(pair_count)
        deg = defaultdict(int)
        for (i, j) in pair_count:
            deg[i] += 1; deg[j] += 1
        degs = sorted(deg.values())
        def pct(p): return degs[int((len(degs) - 1) * p)] if degs else None
        cc = sorted(pair_count.values())
        weak_edge_ratio = round(sum(1 for c in cc if c == 1) / n_edges, 4) if n_edges else None
        top1_ws = sorted([max(v) for v in focal_weights.values()]) if focal_weights else []
        hhi_ws = sorted([sum(x * x for x in v) for v in focal_weights.values()]) if focal_weights else []
        def pct_list(p, lst): return round(lst[int((len(lst) - 1) * p)], 6) if lst else None
        diag['months'].append({
            'cutoff': cutoff,
            'firms_covered': len(firm_analyst_count),
            'firms_with_edges': len(deg),
            'coverage_records': len(sub),   # 覆盖记录数（(firm, analyst) 对数量，诊断 #6）
            'edge_density': round(n_edges / (len(firm_analyst_count) * (len(firm_analyst_count) - 1) / 2), 6)
                           if len(firm_analyst_count) > 1 else None,   # 边数/可能对数（诊断 #6）
            'analysts_active': len(af),
            'undirected_edges': n_edges,
            'degree_mean': round(sum(degs) / len(degs), 2) if degs else None,
            'degree_p50': pct(0.50), 'degree_p90': pct(0.90),
            'common_count_p50': cc[len(cc)//2] if cc else None,
            'common_count_p90': cc[int((len(cc)-1)*0.90)] if cc else None,
            'common_count_max': max(cc) if cc else None,
            'weak_edge_ratio': weak_edge_ratio,
            'top1_weight_p50': pct_list(0.50, top1_ws),
            'weight_hhi_p50': pct_list(0.50, hhi_ws),
        })
        if len(diag['months']) % 12 == 0:
            print(f"{cutoff} done: edges={n_edges} firms={len(firm_analyst_count)}", flush=True)

    (EDGED.parent / 'step67_diagnostics.json').write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE", flush=True)

if __name__ == '__main__':
    main()
