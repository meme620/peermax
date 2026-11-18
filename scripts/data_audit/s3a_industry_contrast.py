"""S3A 9B：行业网络对照（对标 AH 吸收检验）——分析师网络 vs 行业网络的边重叠率 + 行业动量 vs CF 相关性。"""
import pickle, warnings, json
import pandas as pd, numpy as np
from pathlib import Path
import yaml
from s3a_meta import attach_meta

warnings.filterwarnings('ignore')

SPEC = Path('configs/s3a_final_spec_v1.yaml')
_SPEC = yaml.safe_load(SPEC.read_text(encoding='utf-8'))
RET = Path(_SPEC['inputs']['returns'])
EDGE_DIR = Path('data/processed/analyst/edges')

ind = pickle.load(open('data/raw/drive/ind_code.pkl','rb'))
ind.index = [c.split('.')[0] for c in ind.index]
ind = ind.dropna()
print(f"行业分类: {len(ind)} 家, {ind.nunique()} 类", flush=True)

# ===== 1. 边重叠率（抽样 3 个月）=====
rng = np.random.default_rng(1)
out = {}
for cutoff in ['2006-06-30','2013-06-30','2024-06-30']:  # 2026-08-29 全期：早/中/晚抽样
    e = pd.read_csv(f'data/processed/analyst/edges/edges_coverage_win12_fy12_v3_{cutoff}.csv',
                    dtype=str, usecols=['focal_id','source_id'])
    e = e[e['focal_id'] < e['source_id']]  # 无向去重
    e = e[e['focal_id'].isin(ind.index) & e['source_id'].isin(ind.index)]
    same = (ind[e['focal_id']].values == ind[e['source_id']].values)
    # 随机期望：按行业大小算随机同行业概率
    sizes = ind.value_counts(); total = len(ind)
    exp = (sizes * (sizes-1)).sum() / (total * (total-1))
    out[cutoff] = {'edges': len(e), 'same_ind_ratio': round(float(same.mean()), 4),
                   'random_expectation': round(float(exp), 4),
                   'lift': round(float(same.mean()/exp), 2)}
    print(f"{cutoff}: 同行业边占比={same.mean():.4f} vs 随机期望={exp:.4f} (lift={same.mean()/exp:.2f})", flush=True)

# ===== 2) 行业动量 vs CF 相关性（2026-09-01 R7 修订：逐月同期网络，全 241 个月）=====
# 原实现固定用 2024-06 边表回溯 2006/2013 收益（3/4 行前视）且只有 4 个抽查月，
# 支撑不了"全期稳定"措辞。现改为每月用自身 cutoff 边表计算 CF，逐月 rank 相关。
ret = pd.read_csv(RET, dtype=str, usecols=['firm_id','month','R'])
ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)','',regex=True)
ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
ret = ret.dropna()
EDGE_DIR = Path('data/processed/analyst/edges')
files = sorted(EDGE_DIR.glob('edges_coverage_win12_fy12_v3_*.csv'))
corrs, corr_months = {}, []
for k, path in enumerate(files):
    cutoff = path.stem.replace('edges_coverage_win12_fy12_v3_','')
    sig = cutoff[:7]
    r_sig = ret[ret['month']==sig][['firm_id','R']].rename(columns={'R':'rs'})
    if r_sig.empty: continue
    e = pd.read_csv(path, dtype=str, usecols=['focal_id','source_id','weight'])
    e['weight'] = pd.to_numeric(e['weight'], errors='coerce')
    e = e[e['weight'].notna() & (e['weight']>0)]
    m = e.merge(r_sig, left_on='source_id', right_on='firm_id', how='inner')
    num = m.assign(wr=m['weight']*m['rs']).groupby('focal_id')['wr'].sum()
    den = m.groupby('focal_id')['weight'].sum()
    cf = (num/den).rename('cf').reset_index().rename(columns={'focal_id':'firm_id'})
    # 行业动量：同行业等权（含自身，与蒋 IndRet 口径一致即可，目标只是与 CF 对比相关）
    ind_m = r_sig.merge(ind.rename('ind'), left_on='firm_id', right_index=True, how='inner')
    im = ind_m.groupby('ind')['rs'].mean().rename('ind_ret')
    j = cf.merge(ind_m[['firm_id','ind']].merge(im, on='ind'), on='firm_id', how='inner').dropna()
    if len(j) < 50: continue
    c = float(j['cf'].rank().corr(j['ind_ret'].rank()))
    corrs[f'{sig}->{cutoff}'] = c
    corr_months.append(c)
    if (k+1) % 24 == 0:
        print(f"{cutoff}: rank corr={c:.3f} ({k+1}/{len(files)})", flush=True)
arr = np.array(corr_months)
out['cf_vs_industry_momentum'] = {
    'n_months': len(arr), 'months_span': f"{files[0].stem[-10:]}~{files[-1].stem[-10:]}",
    'mean': round(float(arr.mean()),4), 'p10': round(float(np.percentile(arr,10)),4),
    'p90': round(float(np.percentile(arr,90)),4),
    'min': round(float(arr.min()),4), 'max': round(float(arr.max()),4),
    'per_month_first_last': {list(corrs)[0]: list(corrs.values())[0], list(corrs)[-1]: list(corrs.values())[-1]},
}
out['note'] = ('lift>1 = 分析师网络边偏向同行业（早期 lift 低）；CF-行业动量 rank 相关=逐月同期网络全 241 月重算'
               '（2026-09-01 R7 修订，替换旧"2024-06 固定网络×4 抽查月"口径）')
attach_meta(out, __file__)
Path('reports/data_gates/G0/s3a_9b_industry_contrast.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print("DONE -> reports/data_gates/G0/s3a_9b_industry_contrast.json")
