"""批次 1 交叉检验：
A. close_none vs 腾讯 close_raw（2018/2021/2024 三个代表年全市场）
B. Drive limit flag vs S4 argmax 层 limit_hit_at_argmax（全量）
C. st_flag 抽查（当前名称 ST vs flag；历史案例）
"""
import pandas as pd, numpy as np, json, re
from pathlib import Path

D = Path('data/raw/drive')
OUT = {}

def code6_to_drive(code):  # sz000001 -> 000001.SZ
    m = re.match(r'^(sz|sh|bj)(\d{6})$', code)
    return f"{m.group(2)}.{m.group(1).upper()}" if m else None

def drive6_to_panel(c):  # 000001.SZ -> sz000001
    n, mkt = c.split('.')
    return mkt.lower() + n

# ---------- A. close_none vs 腾讯 ----------
lu = pd.read_pickle(D / 'daily_close_none_20260630.pkl')
lu.columns = [drive6_to_panel(c) if '.' in str(c) else c for c in lu.columns]
res_a = {}
for year in [2018, 2021, 2024]:
    tp = pd.read_csv(f'data/processed/daily_panel/{year}.csv',
                     usecols=['code', 'date', 'close_raw'], dtype={'code': str})
    tp['date'] = pd.to_datetime(tp['date'])
    sub = lu.loc[lu.index.year == year]
    long = sub.stack().rename('close_drive').reset_index()
    long.columns = ['date', 'code', 'close_drive']
    m = tp.merge(long, on=['code', 'date'], how='inner')
    m['diff_pct'] = (m['close_raw'] - m['close_drive']).abs() / m['close_drive']
    res_a[str(year)] = {
        'matched': int(len(m)),
        'exact_match_rate': round(float((m['diff_pct'] < 1e-8).mean()), 6),
        'within_0pct1_rate': round(float((m['diff_pct'] < 0.001).mean()), 6),
        'diff_p99': round(float(m['diff_pct'].quantile(0.99)), 6),
        'n_diff_gt_1pct': int((m['diff_pct'] > 0.01).sum()),
    }
    print(f"A {year}: {res_a[str(year)]}", flush=True)
OUT['A_close_none_vs_tencent'] = res_a

# ---------- B. limit flag vs argmax limit_hit ----------
arg = pd.read_csv('data/processed/source_month_argmax_panel.csv',
                  usecols=['source_id', 'argmax_date', 'limit_hit_at_argmax'], dtype=str)
arg['limit_hit_at_argmax'] = pd.to_numeric(arg['limit_hit_at_argmax'], errors='coerce')
lu_up = pd.read_pickle(D / 'daily_limit_up_flag_20260630.pkl')
lu_dn = pd.read_pickle(D / 'daily_limit_down_flag_20260630.pkl')
for df in (lu_up, lu_dn):
    df.columns = [drive6_to_panel(c) if '.' in str(c) else c for c in df.columns]

def flag_at(code, date_str):
    d = pd.Timestamp(date_str)
    v_up = lu_up.at[d, code] if code in lu_up.columns and d in lu_up.index else np.nan
    v_dn = lu_dn.at[d, code] if code in lu_dn.columns and d in lu_dn.index else np.nan
    return v_up, v_dn

n, hit_up, hit_dn, mismatch = 0, 0, 0, []
for _, r in arg.dropna(subset=['limit_hit_at_argmax']).iterrows():
    v_up, v_dn = flag_at(r['source_id'], r['argmax_date'])
    if pd.isna(v_up) and pd.isna(v_dn):
        continue
    n += 1
    drive_hit = int(bool(v_up) or bool(v_dn))
    if drive_hit == int(r['limit_hit_at_argmax']):
        if drive_hit: hit_up += 1
    else:
        mismatch.append({'source_id': r['source_id'], 'date': r['argmax_date'],
                         'ours': int(r['limit_hit_at_argmax']), 'drive': drive_hit})
OUT['B_limit_flag_vs_argmax'] = {
    'comparable': n, 'agree': n - len(mismatch),
    'agree_rate': round((n - len(mismatch)) / max(n, 1), 6),
    'n_mismatch': len(mismatch),
    'mismatch_sample': mismatch[:10],
}
print(f"B: {OUT['B_limit_flag_vs_argmax']}", flush=True)

# ---------- C. st_flag 抽查 ----------
st = pd.read_pickle(D / 'stock_st_flag_20260630.pkl')
st.columns = [drive6_to_panel(c) if '.' in str(c) else c for c in st.columns]
last = st.iloc[-1]
cur_st_flag = {c for c in st.columns if bool(last[c])}
fm = pd.read_csv('data/processed/firm_id_map.csv', dtype=str)
name_st = set(fm[fm['name'].str.contains('ST', na=False)]['code6']
              .map(lambda c: ('sz' if c.startswith(('0','3')) else 'sh') + c))
OUT['C_st_flag'] = {
    'drive_current_st_count': len(cur_st_flag),
    'name_infer_current_st_count': len(name_st),
    'both': len(cur_st_flag & name_st),
    'flag_only': sorted(cur_st_flag - name_st)[:8],
    'name_only': sorted(name_st - cur_st_flag)[:8],
}
print(f"C: {OUT['C_st_flag']}", flush=True)

Path('reports/data_gates/G0/drive_batch1_crosscheck.json').write_text(
    json.dumps(OUT, ensure_ascii=False, indent=2), encoding='utf-8')
print("DONE -> drive_batch1_crosscheck.json")
