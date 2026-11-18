"""S3A Step 11：AnalystCoverage 接入 firm_chars_clean.analyst_cov。

口径：
  derived.month（月末 cutoff YYYY-MM-DD）→ chars.month（YYYY-MM）
  t 月末快照覆盖数 → t 月控制变量（available_at <= t，预测 t+1，合规）
  未匹配行 analyst_cov 保持空，missing_reason 追加 analyst_cov_not_covered;
  不填 0（缺失合同）。

注意：build_firm_chars.py 重跑会将 analyst_cov 重置为空，
      重跑后须再执行本脚本（见 build_firm_chars.py 头部注释）。
"""
import shutil
import pandas as pd

DERIVED = 'data/processed/analyst/firm_analyst_derived.csv'
CHARS = 'data/processed/firm_chars_clean.csv'
BACKUP = 'data/processed/firm_chars_clean_backup_pre_analystcov.csv'

def main():
    shutil.copy(CHARS, BACKUP)
    chars = pd.read_csv(CHARS, dtype={'firm_id': str, 'month': str, 'missing_reason': str},
                        keep_default_na=False)
    der = pd.read_csv(DERIVED, dtype={'firm_id': str, 'month': str, 'AnalystCoverage': float},
                      usecols=['firm_id', 'month', 'AnalystCoverage'])
    der['month'] = der['month'].str[:7]  # 2018-01-31 -> 2018-01
    der = der.rename(columns={'AnalystCoverage': 'analyst_cov_new'})
    print(f"chars rows={len(chars)}, derived rows={len(der)}", flush=True)

    merged = chars.merge(der.drop_duplicates(['firm_id', 'month']),
                         on=['firm_id', 'month'], how='left')
    hit = merged['analyst_cov_new'].notna()
    print(f"matched: {int(hit.sum())} / {len(merged)}", flush=True)

    # 填充 analyst_cov（匹配到的填整数值；未匹配保持空）
    merged['analyst_cov'] = merged['analyst_cov_new'].apply(
        lambda v: str(int(v)) if pd.notna(v) else '')
    # 未匹配行追加缺失标记（不覆盖已有标记）
    mask = ~hit
    merged.loc[mask, 'missing_reason'] = (
        merged.loc[mask, 'missing_reason'].fillna('').astype(str)
        .apply(lambda s: s if 'analyst_cov_not_covered' in s else (s + (';' if s and not s.endswith(';') else '') + 'analyst_cov_not_covered;'))
    )
    out = merged[chars.columns]
    out.to_csv(CHARS, index=False, encoding='utf-8')

    # 验证
    chk = pd.read_csv(CHARS, usecols=['analyst_cov'], dtype=str, keep_default_na=False)
    nonempty = (chk['analyst_cov'] != '').sum()
    print(f"DONE analyst_cov non-empty: {nonempty} / {len(chk)}")
    print(f"backup: {BACKUP}")

if __name__ == '__main__':
    main()
