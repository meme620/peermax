"""批次1 交叉检验：limit flag vs 自算、ST vs 名称、close_none vs 腾讯 raw。"""
import pickle, warnings, json
import pandas as pd
from pathlib import Path
warnings.filterwarnings('ignore')
d = Path('data/raw/drive')
out = {}

# ============ A. close_none vs 腾讯 raw close（2018 年比对） ============
tc = pd.read_csv('data/processed/daily_panel/2018.csv', dtype=str, keep_default_na=False,
                 usecols=['code','date','close_raw'])
tc['close_raw'] = pd.to_numeric(tc['close_raw'], errors='coerce')
cn = pickle.load(open(d/'daily_close_none_20260630.pkl','rb'))
cn18 = cn.loc['2018-01-01':'2018-12-31']
cn6 = cn18.copy(); cn6.columns = [c.split('.')[0] for c in cn6.columns]
tc18 = tc[tc['close_raw'].notna()].copy()
matched, diff_cnt, diff_gt1pct = 0, 0, 0
rows = []
for _, r in tc18.iterrows():
    c = r['code'].replace('sz','').replace('sh','').replace('bj','')
    if c in cn6.columns:
        v = cn6.at[r['date'], c]
        if pd.notna(v):
            matched += 1
            dd = abs(r['close_raw'] - v)
            if dd > 1e-6: diff_cnt += 1
            if dd / v > 0.01 and v != 0: diff_gt1pct += 1
out['close_none_vs_tencent'] = {'matched': matched, 'exact_diff': diff_cnt,
    'diff_gt_1pct': diff_gt1pct, 'note': '2018 全量逐行比对；值为未复权价格'}
print(f"A. close_none vs 腾讯: 比 {matched} 行, 不等 {diff_cnt}, >1% {diff_gt1pct}", flush=True)

# ============ B. st_flag vs 名称推断 ============
st = pickle.load(open(d/'stock_st_flag_20260630.pkl','rb'))
st_last = st.iloc[-1]  # 最后一日状态
st_last.index = [c.split('.')[0] for c in st_last.index]
fm = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
fm['name_st'] = fm['name'].str.contains('ST|\*', regex=True)
common = fm[fm['code6'].isin(st_last.index)].copy()
common['flag_st'] = common['code6'].map(st_last).fillna(False)
agree = (common['name_st'] == common['flag_st']).mean()
conflict = common[common['name_st'] != common['flag_st']]
out['st_flag_vs_name'] = {'firms': len(common), 'agreement': round(float(agree),4),
    'conflict': len(conflict), 'conflict_examples': conflict.head(8)[['code6','name','name_st','flag_st']].to_dict('records')}
print(f"B. ST flag vs 名称: 一致率 {agree:.2%}, 冲突 {len(conflict)}", flush=True)

# ============ C. limit flag vs 自算（2024-06 抽样） ============
lu = pickle.load(open(d/'daily_limit_up_flag_20260630.pkl','rb'))
ld = pickle.load(open(d/'daily_limit_down_flag_20260630.pkl','rb'))
lu6, ld6 = lu.loc['2024-06-03':'2024-06-28'], ld.loc['2024-06-03':'2024-06-28']
lu6.columns = [c.split('.')[0] for c in lu6.columns]
ld6.columns = [c.split('.')[0] for c in ld6.columns]
# 自算：前收 vs 当日 close_none；阈值按 ST（名称法）5% / 主板10% / 创业20%
close = cn.loc['2024-05-31':'2024-06-28'].copy()
close.columns = [c.split('.')[0] for c in close.columns]
st_last24 = st.loc['2024-06-28'].copy(); st_last24.index = [c.split('.')[0] for c in st_last24.index]
def board_limit(c):
    if c.startswith('3'): return 0.20
    return 0.10
calc_up, calc_dn = {}, {}
for c in close.columns[:1500]:
    s = close[c].dropna()
    if len(s) < 15: continue
    r = s.pct_change()
    is_st = bool(st_last24.get(c, False))
    lim = 0.05 if is_st else board_limit(c)
    idx = r.index[1:]
    calc_up[c] = pd.Series(r > lim - 1e-9, index=idx)
    calc_dn[c] = pd.Series(r < -lim + 1e-9, index=idx)
up_all = []
for c, s in calc_up.items():
    if c in lu6.columns:
        dd = lu6[c].reindex(s.index).fillna(False)
        up_all.append(pd.DataFrame({'calc': s, 'flag': dd}).assign(code=c))
upm = pd.concat(up_all)
agree_up = (upm['calc'] == upm['flag']).mean()
out['limit_up_vs_calc'] = {'rows': len(upm), 'agreement': round(float(agree_up),4),
    'flag_true_calc_false': int(((upm['flag']) & ~upm['calc']).sum()),
    'calc_true_flag_false': int((upm['calc'] & ~upm['flag']).sum())}
print(f"C. limit up flag vs 自算: 一致率 {agree_up:.2%}（{len(upm)} 行, 2024-06 抽样 1500 股）", flush=True)

# ============ D. ind_code 体系判定结论 ============
out['ind_code'] = {'n_firms': int(ind_n if (ind_n:=len(pickle.load(open(d/'ind_code.pkl','rb')))) else 0),
    'classes': 30, 'system': '非申万二级（30 类两位数；40=银行/41=证券/36≈食品饮料/33≈家电/10≈能源/20≈公用）；体系名待确认',
    'usable_for': '行业固定效应、S11 industry alternative link', 'mb_calibration': '仍缺申万二级'}

Path('reports/data_gates/G0/drive_batch1_crosscheck.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print("DONE -> reports/data_gates/G0/drive_batch1_crosscheck.json")
