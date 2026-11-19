"""analyst-network Step 2：原始记录清洗审计（零落盘，只读 + 报告）。

检查项（protocol §5A 第 2-5 步对应）:
  1. 精确重复行（13 列全同）计数
  2. Scode 格式（长度/非数字/空值）与 canonical firm_id 映射失败率
  3. A_code：空值率、含 '/' 多分析师率、与 A_name 位置对齐率
  4. 日期列格式合法率（Anndats/Fpenddt/Anndats_act）
  5. (firm_id, analyst_id, report_date) 报告级重复计数
输出: data/raw/analyst/step2_audit.json + 摘要
"""
import json, re
from pathlib import Path
import pandas as pd

RAW = 'data/raw/analyst/forecast_raw_all.csv'
OUT = Path('data/raw/analyst')
OUT.mkdir(parents=True, exist_ok=True)

DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
CODE_RE = re.compile(r'^\d{6}$')

def main():
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    print(f"loaded {len(df)} rows", flush=True)
    rep = {}

    # 1. 精确重复
    dup = df.duplicated().sum()
    rep['exact_duplicate_rows'] = int(dup)
    print(f"exact_duplicates={dup}", flush=True)

    # 2. Scode
    sc = df['Scode']
    rep['scode'] = {
        'empty': int((sc == '').sum()),
        'not_6digit': int((~sc.str.match(r'^\d{6}$')).sum()),
        'unique': int(sc.nunique()),
    }
    print(f"scode unique={sc.nunique()} not6={rep['scode']['not_6digit']} empty={rep['scode']['empty']}", flush=True)

    # Scode -> firm_id 映射（canonical: firm_chars_clean.firm_id 纯六位）
    chars = pd.read_csv('data/processed/firm_chars_clean.csv', usecols=['firm_id'], dtype=str)
    firm_set = set(chars['firm_id'].dropna().unique())
    sc_valid = sc[sc.str.match(r'^\d{6}$')]
    mapped = sc_valid.isin(firm_set)
    rep['firm_mapping'] = {
        'canonical_firm_count': len(firm_set),
        'scode_mapped': int(mapped.sum()),
        'scode_unmapped': int((~mapped).sum()),
        'unmapped_rate': round(float((~mapped).mean()), 6),
    }
    print(f"firm mapping: {rep['firm_mapping']}", flush=True)

    # 3. A_code
    ac = df['A_code']
    an = df['A_name']
    has_slash = ac.str.contains('/')
    rep['a_code'] = {
        'empty': int((ac == '').sum()),
        'with_slash': int(has_slash.sum()),
        'unique': int(ac.nunique()),
    }
    # 对齐率：A_code 与 A_name 的 '/' 分隔段数一致
    def seg_ok(row):
        c, n = row
        if c == '' or n == '':
            return True
        return len(c.split('/')) == len(n.split('/'))
    aligned = df[['A_code', 'A_name']].apply(seg_ok, axis=1)
    rep['a_code']['name_aligned'] = int(aligned.sum())
    rep['a_code']['name_aligned_rate'] = round(float(aligned.mean()), 6)
    print(f"A_code: {rep['a_code']}", flush=True)

    # 4. 日期
    rep['dates'] = {}
    for col in ['Anndats', 'Fpenddt', 'Anndats_act']:
        v = df[col]
        ok = v.str.match(r'^\d{4}-\d{2}-\d{2}$')
        rep['dates'][col] = {'empty': int((v == '').sum()), 'valid_fmt': int(ok.sum()),
                             'invalid': int((~ok).sum())}
        print(f"date {col}: {rep['dates'][col]}", flush=True)

    # 5. 报告级重复 (firm_id, analyst_id, report_date)——用 Scode 清洗后与 A_code 原值
    #    （A_code 含 / 时取首段，多分析师记录拆分会留到 Step 3）
    key = df.assign(k=df['Scode'].str.strip() + '|' + ac.str.split('/').str[0] + '|' + df['Anndats'])
    rep['report_level_dup_rows'] = int(key['k'].duplicated().sum())
    print(f"report-level dup (raw A_code first seg): {rep['report_level_dup_rows']}", flush=True)

    # 6. 冲突审计（2026-08-12 偏差 #2）：同报告键下字段值不一致的键数。
    #    规划要求"非完全重复冲突不能静默删除，保留 conflict audit"。
    #    注意键必须含 Fpenddt：同一报告对多个财年预测（FY1/FY2/FY3）会产生多行且 Fpenddt 不同，
    #    这是正常结构（step4 不重复计数）；真正的冲突是"同报告同预测期下预测值不同"（重复录入）。
    key2 = df.assign(k2=df['Scode'].str.strip() + '|' + ac.str.split('/').str[0]
                     + '|' + df['Anndats'] + '|' + df['Fpenddt'])
    grp2 = key2.groupby('k2')
    conflict_any = (grp2[['Feps', 'Fpe', 'Fnetpro', 'Foperev']].nunique() > 1).any(axis=1)
    rep['conflict_audit'] = {
        'note': '同 (Scode, A_code首段, Anndats, Fpenddt) 键下预测字段值不一致的键数（重复录入冲突，step4 keep-first 丢弃后行）；不含跨 Fpenddt 的正常多预测行',
        'any_conflict_keys': int(conflict_any.sum()),
        'field_conflict_keys': {c: int((grp2[c].nunique() > 1).sum())
                                for c in ['Feps', 'Fpe', 'Fnetpro', 'Foperev']},
    }
    print(f"conflict audit: {rep['conflict_audit']}", flush=True)

    (OUT / 'step2_audit.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE -> data/raw/analyst/step2_audit.json")

if __name__ == '__main__':
    main()
