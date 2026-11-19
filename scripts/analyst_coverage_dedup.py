"""analyst-network Step 4：报告级去重 -> firm_analyst_coverage_records（零外部依赖）。

规则（protocol §5A 第 5 步）:
  (firm_id, analyst_id, report_date) 至多计一次；
  同日同公司同分析师因不同 Fpenddt 或多预测字段产生多行 -> 不重复计数。
预测字段（feps/fpe/fnetpro/foperev）保留首条，完整预测修订另存 forecast 层。

输出:
  data/processed/analyst/firm_analyst_coverage_records.csv
  data/processed/analyst/step4_dedup.json
"""
import json
from pathlib import Path
import pandas as pd

SRC = 'data/processed/analyst/analyst_step3_clean.csv'
OUTD = Path('data/processed/analyst')
OUTD.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
    df = df[df['mapping_status'] == 'ok'].copy()
    print(f"ok rows: {len(df)}", flush=True)

    n_raw = len(df)
    df['report_key'] = df['firm_id'] + '|' + df['analyst_id'] + '|' + df['Anndats']
    n_dup = int(df['report_key'].duplicated().sum())

    # 保留首条（预测字段取首条；coverage 层只用覆盖事实）
    out = df.drop_duplicates('report_key', keep='first').drop(columns=['report_key'])
    out = out.rename(columns={'Anndats': 'report_date'})
    print(f"after dedup: {len(out)} (dup={n_dup})", flush=True)
    out.to_csv(OUTD / 'firm_analyst_coverage_records.csv', index=False, encoding='utf-8')

    rep = {
        'ok_rows': n_raw,
        'report_level_duplicates': int(n_dup),
        'coverage_records': int(len(out)),
        'unique_firms': int(out['firm_id'].nunique()),
        'unique_analysts': int(out['analyst_id'].nunique()),
        'unique_brokers': int(out['broker_code'].nunique()),
        'year_range': [out['report_date'].min(), out['report_date'].max()],
    }
    (OUTD / 'step4_dedup.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE", json.dumps(rep, ensure_ascii=False)[:400])

if __name__ == '__main__':
    main()
