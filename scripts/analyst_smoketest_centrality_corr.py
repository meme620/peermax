"""S3A smoketest：网络中心性共线性检查（2026-08-14，纯 numpy 幂迭代版）。

对抗审查 B1：派生变量 NetworkDegree/AnalystNetworkMass/EffectivePeerCount/Top1Weight
与文献化中心性度量（eigenvector centrality / Perron vector）的相关性。
若高度共线（>0.95），论文报告 degree 即可；若互补，可考虑补充。

只在 3 个抽查月（2018-06/2021-06/2024-06）上计算（幂迭代，无 scipy 依赖）。

用法: python scripts/analyst_smoketest_centrality_corr.py
输出: data/processed/analyst/smoketest_centrality_corr.json
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

EDGES = Path('data/processed/analyst/edges')
DER = 'data/processed/analyst/firm_analyst_derived.csv'
OUT = Path('data/processed/analyst/smoketest_centrality_corr.json')
NET_VER = 'coverage_win12_fy12_v2'
MONTHS = ['2018-06-30', '2021-06-30', '2024-06-30']


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 5:
        return None
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def power_iteration(rows, cols, vals, n, n_iter=300, tol=1e-9):
    """对称邻接矩阵最大特征向量的幂迭代（纯 numpy）。"""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    v = np.ones(n) / np.sqrt(n)
    for _ in range(n_iter):
        out = np.zeros(n)
        np.add.at(out, rows, vals * v[cols])
        np.add.at(out, cols, vals * v[rows])
        nrm = np.linalg.norm(out)
        if nrm == 0:
            break
        out = out / nrm
        if np.abs(out - v).max() < tol:
            v = out
            break
        v = out
    # Perron 向量取正部分（符号由初始化保证为正）
    return v


def main():
    der = pd.read_csv(DER, dtype=str, keep_default_na=False)
    der['month'] = der['month'].str[:7]
    der = der[der['month'].isin([m[:7] for m in MONTHS])]
    for c in ['NetworkDegree', 'AnalystNetworkMass', 'EffectivePeerCount', 'Top1Weight']:
        der[c] = pd.to_numeric(der[c], errors='coerce')

    out = {'method': 'power iteration (pure numpy), Perron vector on common-analyst-count adjacency', 'months': {}}
    for c in MONTHS:
        df = pd.read_csv(EDGES / f'edges_{NET_VER}_{c}.csv', dtype=str,
                         usecols=['focal_id', 'source_id', 'common_analyst_count'])
        df['c'] = pd.to_numeric(df['common_analyst_count'], errors='coerce').fillna(0)
        firms = sorted(set(df['focal_id']) | set(df['source_id']))
        fidx = {f: i for i, f in enumerate(firms)}
        n = len(firms)
        rows = df['focal_id'].map(fidx).values.astype(np.int64)
        cols = df['source_id'].map(fidx).values.astype(np.int64)
        vals = df['c'].values.astype(np.float64)
        evc = power_iteration(rows, cols, vals, n)
        ev_map = {firms[i]: float(evc[i]) for i in range(n)}

        sub = der[der['month'] == c[:7]].copy()
        sub['evc'] = sub['firm_id'].map(ev_map)
        corrs = {}
        for col in ['NetworkDegree', 'AnalystNetworkMass', 'EffectivePeerCount', 'Top1Weight']:
            corrs[f'spearman_{col}_vs_evc'] = round(spearman(sub[col].values, sub['evc'].values), 4)
            corrs[f'spearman_{col}_vs_degree'] = round(
                spearman(sub[col].values, sub['NetworkDegree'].values), 4)
        out['months'][c] = {'firms': n, **corrs}
        print(c, json.dumps(corrs), flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
