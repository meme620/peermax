"""analyst-network Step 3：ID 标准化 + A_code 拆分 + 落盘 forecast 层（零外部依赖）。

输入: data/raw/analyst/forecast_raw_all.csv（2,922,904 行）
处理:
  1. Scode 清洗：^\\d{6}$ 通过者保留；其余 30 条标记 not_matched
  2. firm 映射：清洗后代码 ∈ firm_chars_clean.firm_id 集合；未映射标记 not_matched
  3. A_code 按 '/' 拆分，A_name 同步按位置拆分；空 A_code 标记 not_matched
  4. 保留 B_code/B_name/预测字段/Anndats_act（forecast 层全保留；
     落盘前统一改小写 broker_code/broker_name/feps/fpe/fnetpro/foperev/anndats_act，
     与下游 step4/5/9/sens 引用一致，契约见 tests/test_analyst_contracts.py）
输出:
  data/processed/analyst/analyst_step3_clean.csv   拆分后 forecast 层
  data/processed/analyst/step3_clean.json             清洗报告
"""
import json, re
from pathlib import Path
import pandas as pd

RAW = 'data/raw/analyst/forecast_raw_all.csv'
OUTD = Path('data/processed/analyst')
OUTD.mkdir(parents=True, exist_ok=True)
CODE_RE = re.compile(r'^\d{6}$')

# 最终输出列序（rename 前的列名；OUT_COLS + OUTPUT_RENAME 构成输出契约，
# tests/test_analyst_contracts.py 校验其与产物表头一致，防源码-产物漂移）
OUT_COLS = ['firm_id', 'Anndats', 'Fpenddt', 'analyst_id', 'analyst_name',
            'B_code', 'B_name', 'Feps', 'Fpe', 'Fnetpro', 'Foperev',
            'Anndats_act', 'mapping_status']
OUTPUT_RENAME = {'B_code': 'broker_code', 'B_name': 'broker_name',
                 'Feps': 'feps', 'Fpe': 'fpe', 'Fnetpro': 'fnetpro',
                 'Foperev': 'foperev', 'Anndats_act': 'anndats_act'}

def main():
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    print(f"input rows: {len(df)}", flush=True)

    # 1. Scode 清洗
    scode_ok = df['Scode'].str.match(r'^\d{6}$')
    # 2. firm 映射
    chars = pd.read_csv('data/processed/firm_chars_clean.csv', usecols=['firm_id'], dtype=str)
    firm_set = set(chars['firm_id'].dropna().unique())
    firm_ok = df['Scode'].isin(firm_set)

    status = df['Scode'].where(scode_ok & firm_ok, other='')
    df['firm_id'] = status
    df['mapping_status'] = 'ok'
    df.loc[~scode_ok, 'mapping_status'] = 'not_matched_bad_scode'
    df.loc[scode_ok & ~firm_ok, 'mapping_status'] = 'not_matched_unlisted'
    n_ok = int((df['mapping_status'] == 'ok').sum())
    print(f"mapped ok: {n_ok}", flush=True)

    # 3. A_code 拆分（含空值处理）
    df['A_code'] = df['A_code'].fillna('')
    df['A_name'] = df['A_name'].fillna('')
    empty_ac = df['A_code'] == ''
    df.loc[empty_ac, 'mapping_status'] = 'not_matched_no_analyst_code'
    print(f"no analyst code: {int(empty_ac.sum())}", flush=True)

    rows = []
    keep_cols = ['firm_id', 'Anndats', 'Fpenddt', 'A_code', 'A_name',
                 'B_code', 'B_name', 'Feps', 'Fpe', 'Fnetpro', 'Foperev',
                 'Anndats_act', 'mapping_status']
    # 向量化拆分（批次三优化：iterrows 292 万行 -> str.split + explode）
    sub = df[keep_cols].copy()
    sub['codes'] = sub['A_code'].str.split('/')
    sub['names'] = sub['A_name'].str.split('/')
    sub['n_codes'] = sub['codes'].str.len()
    sub['n_names'] = sub['names'].str.len()
    mis = sub['n_codes'] != sub['n_names']
    n_mis = int(mis.sum())
    if n_mis:
        # 长度错位行：用第一个名字（或空串）补齐到 codes 数量（等价于原 iterrows 逻辑）。
        # pandas 3.x 下 .loc[mask] 直接赋 ragged list 会触发 numpy 2D 形状检查，
        # 改为整列重建（语义不变）。
        fn = sub.loc[mis, 'names'].str[0].tolist()
        nc = sub.loc[mis, 'n_codes'].tolist()
        pad = [([x] * n) if x else ([''] * n) for x, n in zip(fn, nc)]
        names_all = sub['names'].tolist()
        for pos, idx in enumerate(mis.index[mis]):
            names_all[idx] = pad[pos]
        sub['names'] = names_all
    print(f"code/name misaligned rows: {n_mis}", flush=True)
    out = sub.explode(['codes', 'names'])
    out['analyst_id'] = out['codes'].str.strip()
    out['analyst_name'] = out['names'].str.strip()
    out = out[OUT_COLS]
    out = out.rename(columns=OUTPUT_RENAME)
    print(f"split rows: {len(out)}", flush=True)
    out.to_csv(OUTD / 'analyst_step3_clean.csv', index=False, encoding='utf-8')

    rep = {
        'input_rows': int(len(df)),
        'split_rows': int(len(out)),
        'mapped_ok': int((out['mapping_status'] == 'ok').sum()),
        'status_counts': out['mapping_status'].value_counts().to_dict(),
        'unique_firms_mapped': int(out.loc[out['mapping_status'] == 'ok', 'firm_id'].nunique()),
        'unique_analysts': int(out['analyst_id'].nunique()),
    }
    (OUTD / 'step3_clean.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE", json.dumps(rep, ensure_ascii=False)[:400])

if __name__ == '__main__':
    main()
