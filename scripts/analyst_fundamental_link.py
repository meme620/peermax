"""analyst-network 基本面关联检验（AH 2020 Table 2 式，2026-08-14）。

目的：检验"共同覆盖同行（analyst network peer）的基本面增长是否与 focal 的基本面增长相关"——
     网络捕捉真实经济关联的直接证据（AH Table 2：CF 同行销售增长系数 5.1% 显著）。

数据：
  - eastmoney/finance/*.csv（全市场，TOTALOPERATEREVETZ=营收同比%、PARENTNETPROFITTZ=净利润同比%）
  - data/processed/analyst/edges/edges_<NET_VER>_<cutoff>.csv（共同分析师网络权重；NET_VER 见下方常量，当前 v3 全期）
  - data/processed/firm_chars_clean.csv（Size、BM 控制变量，月度）

时间对齐（无前视）：
  - 财务按 NOTICE_DATE（公告日）<= cutoff 的最新报告期取值（merge_asof backward）
  - 网络用 cutoff 月末权重

回归（两个口径各一次）：
  Growth(i,t) = a + b*PeerGrowth(i,t) + g*Size + d*BM + Year FE
  PeerGrowth(i,t) = sum_j w_ij(t) * Growth(j,t)
  标准误：按 firm 聚类（sandwich + 小样本修正）；Growth/PeerGrowth 两端 winsorize 1%/99%

输出: data/processed/analyst/fundamental_link.json
"""
import json, glob, pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np

from analyst_meta import attach_meta
from fileio import read_pickle

FIN = 'data/raw/eastmoney/finance/*.csv'
EDGES = Path('data/processed/analyst/edges')
CHARS = 'data/processed/firm_chars_clean.csv'
OUT = 'data/processed/analyst/fundamental_link.json'
NET_VER = 'coverage_win12_fy12_v3'  # 2026-08-29 全期扩展
GROWTH_FIELDS = {
    'netprofit': ('PARENTNETPROFITTZ', '净利润同比增速'),
    'revenue': ('TOTALOPERATEREVETZ', '营收同比增速'),
}


def cluster_ols(X, y, cluster):
    """OLS + 按 cluster 聚类标准误（sandwich + G/(G-1)*(N-1)/(N-K) 修正）。"""
    N, K = X.shape
    XpX_inv = np.linalg.inv(X.T @ X)
    beta = XpX_inv @ (X.T @ y)
    e = y - X @ beta
    meat = np.zeros((K, K))
    for g in np.unique(cluster):
        m = cluster == g
        Xg, eg = X[m], e[m]
        meat += (Xg.T @ eg)[:, None] @ (Xg.T @ eg)[None, :]
    G = len(np.unique(cluster))
    V = XpX_inv @ meat @ XpX_inv
    V *= (G / (G - 1)) * ((N - 1) / (N - K))
    return beta, np.sqrt(np.diag(V))


def load_financials():
    """读 eastmoney 财务 -> (firm_id, report_date, notice_date, 增速字段)。缓存优先。"""
    cache = 'data/processed/analyst/financial_panel.csv'
    if Path(cache).exists():
        fin = pd.read_csv(cache, dtype={'firm_id': str})
        fin['report_date'] = pd.to_datetime(fin['report_date'])
        fin['notice_date'] = pd.to_datetime(fin['notice_date'])
        print(f"财务(缓存): {len(fin)} 行, {fin['firm_id'].nunique()} 家", flush=True)
        return fin
    frames = []
    for f in glob.glob(FIN):
        try:
            df = pd.read_csv(f, usecols=['SECURITY_CODE', 'REPORT_DATE', 'NOTICE_DATE',
                                         'TOTALOPERATEREVETZ', 'PARENTNETPROFITTZ'])
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        frames.append(df)
    fin = pd.concat(frames, ignore_index=True)
    fin['firm_id'] = fin['SECURITY_CODE'].astype(int).astype(str).str.zfill(6)
    fin['report_date'] = pd.to_datetime(fin['REPORT_DATE'].str[:10])
    fin['notice_date'] = pd.to_datetime(fin['NOTICE_DATE'].str[:10])
    for col in ['TOTALOPERATEREVETZ', 'PARENTNETPROFITTZ']:
        fin[col] = pd.to_numeric(fin[col], errors='coerce')
    fin = fin[fin['notice_date'].notna()].drop_duplicates(
        ['firm_id', 'report_date'], keep='last')
    print(f"财务: {len(fin)} 行, {fin['firm_id'].nunique()} 家, "
          f"{fin['report_date'].min().date()} ~ {fin['report_date'].max().date()}", flush=True)
    return fin


def winsorize(s, lo=0.01, hi=0.99):
    q = s.quantile([lo, hi])
    return s.clip(q.iloc[0], q.iloc[1])


def main():
    fin = load_financials()
    # 2026-08-29：控制变量改用 Drive 全期市值/BM（原 firm_chars_clean 仅 2018 起，限制 9A 到短样本）
    def _drive_chars(fname, col):
        d = read_pickle(Path('data/raw/drive') / fname).resample('ME').last()
        s = d.stack().reset_index()
        s.columns = ['month', 'firm_id', col]
        s['firm_id'] = s['firm_id'].astype(str).str.split('.').str.get(0)
        s['month'] = pd.to_datetime(s['month']).dt.strftime('%Y-%m')
        return s
    mv_d = _drive_chars('daily_a_share_market_val_3_20260630.pkl', 'mv')
    bm_d = _drive_chars('daily_book_to_market_ratio_ttm_20260630.pkl', 'bm')
    chars = mv_d.merge(bm_d, on=['firm_id', 'month'], how='left')
    chars['size'] = np.log(pd.to_numeric(chars['mv'], errors='coerce'))
    chars['bm'] = pd.to_numeric(chars['bm'], errors='coerce')
    chars = chars.dropna(subset=['size'])

    # 月度 cutoff 网格
    cutoffs = sorted(pd.Timestamp(f.stem.replace(f'edges_{NET_VER}_', ''))
                     for f in EDGES.glob(f'edges_{NET_VER}_*.csv'))
    print(f"cutoffs: {len(cutoffs)} ({cutoffs[0].date()} ~ {cutoffs[-1].date()})", flush=True)

    results = {}
    for gname, (gcol, glabel) in GROWTH_FIELDS.items():
        rows = []
        for cutoff in cutoffs:
            cm = cutoff.strftime('%Y-%m')
            # 1) 每家 focal 的最新可得财务（NOTICE_DATE <= cutoff）
            fsub = fin[fin['notice_date'] <= cutoff].sort_values('notice_date')
            latest = fsub.drop_duplicates('firm_id', keep='last')[['firm_id', gcol]]
            latest = latest.rename(columns={gcol: 'g_focal'})
            # 2) 网络边 + 同行财务 -> 同行加权增长
            edges = pd.read_csv(EDGES / f'edges_{NET_VER}_{cutoff.strftime("%Y-%m-%d")}.csv',
                                dtype=str, usecols=['focal_id', 'source_id', 'weight'])
            edges['w'] = pd.to_numeric(edges['weight'], errors='coerce')
            edges = edges[edges['w'].notna() & (edges['w'] > 0)]
            m = edges.merge(latest, left_on='source_id', right_on='firm_id', how='inner')
            if m.empty:
                continue
            peer = m.groupby('focal_id').apply(
                lambda d: float(np.average(d['g_focal'], weights=d['w'])),
                include_groups=False).reset_index()
            peer.columns = ['firm_id', 'g_peer']
            # 3) 合并 focal 增长 + 控制变量
            d = latest.merge(peer, on='firm_id', how='inner')
            csub = chars[chars['month'] == cm][['firm_id', 'size', 'bm']]
            d = d.merge(csub, on='firm_id', how='inner')
            d['year'] = cutoff.year
            d = d.dropna(subset=['g_focal', 'g_peer', 'size', 'bm'])  # bm 缺失须剔除（Drive 早期大量 NaN 会污染 X）
            rows.append(d)
        panel = pd.concat(rows, ignore_index=True)
        # winsorize + 回归
        panel['g_focal_w'] = winsorize(panel['g_focal'])
        panel['g_peer_w'] = winsorize(panel['g_peer'])
        years = sorted(panel['year'].unique())
        panel['yr'] = panel['year'].map({y: i for i, y in enumerate(years)})
        N = len(panel)
        # 年份 FE：const + 去掉基准年的年份虚拟变量（避免 dummy variable trap，X'X 奇异）
        year_dummies = np.eye(len(years))[panel['yr'].values][:, 1:]
        X = np.column_stack([np.ones(N), panel['g_peer_w'], panel['size'], panel['bm']]
                            + [year_dummies[:, i] for i in range(len(years) - 1)])
        y = panel['g_focal_w'].values
        beta, se = cluster_ols(X, y, panel['firm_id'].values)
        names = ['const', 'PeerGrowth', 'Size', 'BM'] + [f'Year{y}' for y in years[1:]]
        b_peer, se_peer = beta[1], se[1]
        t_peer = b_peer / se_peer
        # R²
        e = y - X @ beta
        r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
        results[gname] = {
            'field': gcol, 'label': glabel,
            'n_obs': int(N), 'n_firms': int(panel['firm_id'].nunique()),
            'n_months': int(panel['year'].nunique()),
            'beta_peer': round(float(b_peer), 4),
            'se_peer_clustered': round(float(se_peer), 4),
            't_peer': round(float(t_peer), 2),
            'r2': round(float(r2), 4),
            'peer_mean_pct': round(float(panel['g_peer_w'].mean()), 3),
            'focal_mean_pct': round(float(panel['g_focal_w'].mean()), 3),
            'note': 'AH Table 2 式：focal 增长 ~ peer 加权增长 + Size + BM + Year FE（firm 聚类 SE）',
        }
        print(json.dumps({k: v for k, v in results[gname].items() if k != 'note'},
                         ensure_ascii=False), flush=True)

    results['benchmark'] = {
        'ah2020_us': 'CF 同行销售增长系数 5.1%（显著），强于行业/地理同行（Table 2）',
        'criterion': 'beta_peer 显著为正（t>2）= 网络捕捉真实经济关联',
    }
    attach_meta(results, __file__, 'coverage_win12_fy12_v3')
    Path(OUT).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE ->', OUT, flush=True)


if __name__ == '__main__':
    main()
