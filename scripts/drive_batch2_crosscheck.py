"""批次2 交叉检验：市值/BM/流动性 vs 自算；suspend/new_stock vs 现有规则；shibor vs rf。"""
import warnings, json
import pandas as pd, numpy as np
from pathlib import Path
from fileio import read_pickle
warnings.filterwarnings('ignore')
d = Path('data/raw/drive'); out = {}

def w(cols): return [c.split('.')[0] for c in cols]

if __name__ == '__main__':
    # ===== A. 市值 vs 自算（market_equity_panel，2018 抽样） =====
    mv = read_pickle(d/'daily_a_share_market_val_3_20260630.pkl')
    mv.columns = w(mv.columns)
    meq = pd.read_csv('data/processed/market_equity_panel.csv', dtype=str, usecols=['firm_id','date','market_equity'])
    meq['market_equity'] = pd.to_numeric(meq['market_equity'], errors='coerce')
    meq18 = meq[meq['date'].str.startswith('2018')].dropna()
    n, corr, diff = 0, 0.0, 0.0
    pairs = []
    for _, r in meq18.sample(min(30000, len(meq18)), random_state=1).iterrows():
        v = mv.at[r['date'], r['firm_id']] if r['date'] in mv.index and r['firm_id'] in mv.columns else np.nan
        if pd.notna(v) and r['market_equity'] > 0:
            n += 1; pairs.append((r['market_equity'], v))
    if pairs:
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        corr = np.corrcoef(a, b)[0, 1]; diff = np.mean(np.abs(a - b) / b)
    out['market_val_vs_calc'] = {'n': n, 'corr': round(float(corr), 5), 'mean_abs_rel_diff': round(float(diff), 5)}
    print(f"A. 市值 vs 自算: n={n} corr={corr:.4f} 均相对差={diff:.4f}", flush=True)

    # ===== B. BM TTM vs 自算（firm_chars_clean bm，2018-01 月） =====
    bm = read_pickle(d/'daily_book_to_market_ratio_ttm_20260630.pkl')
    bm.columns = w(bm.columns)
    fc = pd.read_csv('data/processed/firm_chars_clean.csv', dtype=str, usecols=['firm_id','month','bm'])
    fc['bm'] = pd.to_numeric(fc['bm'], errors='coerce')
    fc18 = fc[fc['month']=='2018-06'].dropna()
    bmp = bm.loc['2018-06-29'] if '2018-06-29' in bm.index else bm.iloc[bm.index.get_indexer([pd.Timestamp('2018-06-29')], method='nearest')[0]]
    bmp.index = w(bmp.index)
    m = fc18.merge(bmp.rename('bm_drive'), left_on='firm_id', right_index=True, how='inner').dropna()
    out['bm_ttm_vs_calc'] = {'n': len(m), 'corr': round(float(m['bm'].corr(m['bm_drive'])), 5),
        'mean_abs_diff': round(float((m['bm']-m['bm_drive']).abs().mean()), 5)}
    print(f"B. BM TTM vs 自算(2018-06): n={len(m)} corr={m['bm'].corr(m['bm_drive']):.4f} 均差={out['bm_ttm_vs_calc']['mean_abs_diff']}", flush=True)

    # ===== C. 换手率 vs 自算（firm_chars turnover） =====
    tr = read_pickle(d/'daily_turnover_rate_20260630.pkl')
    tr.columns = w(tr.columns)
    fc_t = pd.read_csv('data/processed/firm_chars_clean.csv', dtype=str, usecols=['firm_id','month','turnover'])
    fc_t['turnover'] = pd.to_numeric(fc_t['turnover'], errors='coerce')
    fc_t18 = fc_t[fc_t['month']=='2018-06'].dropna()
    trp = tr.loc['2018-06-29'] if '2018-06-29' in tr.index else tr.iloc[tr.index.get_indexer([pd.Timestamp('2018-06-29')], method='nearest')[0]]
    trp.index = w(trp.index)
    mt = fc_t18.merge(trp.rename('turn_drive'), left_on='firm_id', right_index=True, how='inner').dropna()
    out['turnover_vs_calc'] = {'n': len(mt), 'corr': round(float(mt['turnover'].corr(mt['turn_drive'])), 5)}
    print(f"C. 换手率 vs 自算(2018-06): n={len(mt)} corr={mt['turnover'].corr(mt['turn_drive']):.4f}", flush=True)

    # ===== D. suspend_flag vs 停牌推断（daily_panel 缺失） =====
    sf = read_pickle(d/'stock_suspend_flag_20260630.pkl')
    sf.columns = w(sf.columns)
    dp = pd.read_csv('data/processed/daily_panel/2018.csv', dtype=str, usecols=['code','date'])
    dp['firm_id'] = dp['code'].str.replace(r'^(sz|sh|bj)','',regex=True)
    obs = set(zip(dp['firm_id'], dp['date']))
    sus = sf.loc['2018-01-01':'2018-12-31']
    n, agree = 0, 0
    for c in sus.columns[:800]:
        for dt, v in sus[c].items():
            dtstr = dt.strftime('%Y-%m-%d')
            n += 1
            # flag=True(停牌) 应该 ⇔ 面板缺失（该日无记录）
            missing = (c, dtstr) not in obs
            agree += (bool(v) == missing)
    out['suspend_vs_missing'] = {'n': n, 'agreement': round(agree/n, 4)}
    print(f"D. suspend flag vs 面板缺失(2018, 800 股): n={n} 一致率={agree/n:.4f}", flush=True)

    # ===== E. new_stock_flag vs ipo+6月规则 =====
    nf = read_pickle(d/'new_stock_flag_20260630.pkl')
    nf.columns = w(nf.columns)
    ipo_map = dict(zip(pd.read_csv('data/processed/firm_id_map.csv', dtype=str)['code6'],
                       pd.read_csv('data/processed/firm_id_map.csv', dtype=str)['ipo_date']))
    nfp = nf.loc['2020-06-30']
    nfp.index = w(nfp.index)
    ex = nfp[nfp.fillna(False)].index
    exp_rule = [c for c in ex if c in ipo_map and (pd.Timestamp('2020-06-30') - pd.Timestamp(ipo_map[c])).days < 30.4*6]
    out['new_stock_vs_ipo6m'] = {'flag_new': len(ex), 'within_6m': len(exp_rule), 'outside_6m': len(ex)-len(exp_rule)}
    print(f"E. new_stock flag(2020-06-30): flag={len(ex)} 家, 其中上市<6月={len(exp_rule)}, >6月={len(ex)-len(exp_rule)}", flush=True)

    # ===== F. shibor vs 现有 rf（一年期存款阶梯，2006-10 起） =====
    sh = read_pickle(d/'shibor_rates.pkl')
    out['shibor'] = {'shape': list(sh.shape), 'columns': list(sh.columns), 'period': f"{sh.index[0].date()}~{sh.index[-1].date()}"}
    print(f"F. shibor: {sh.shape} cols={list(sh.columns)[:5]}... 期间 {out['shibor']['period']}", flush=True)

    Path('reports/data_gates/G0/drive_batch2_crosscheck.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE -> reports/data_gates/G0/drive_batch2_crosscheck.json")
