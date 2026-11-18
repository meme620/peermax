"""S3 收口项：文本层清理审计（错位检测 + 模板文本检测）。

A. 错位检测（跨年一致性法，无需名称表）：
   对每家公司，计算每个 (code,year) 文本与该公司其他年份文本的 max bigram-Jaccard；
   max < 0.05 -> 疑似错位（正常公司相邻年业务文本应保留较高重叠）。
B. 模板文本检测：
   B1 精确重复：全量 text 哈希分组，跨公司同文本 -> 模板组；
   B2 近重复：408 条 sim>=0.99 假边涉及的 105 家公司内两两 Jaccard>0.90 -> 近重复组。

输出:
  reports/data_gates/G0/s3_text_audit.json          汇总
  artifacts/review/s3_misalign_suspect.csv          疑似错位清单
  artifacts/review/s3_template_texts.csv            模板/近重复文本清单
"""
import csv, hashlib, re
from pathlib import Path
from collections import defaultdict
import pandas as pd

BT = 'data/processed/business_texts_v7.csv'
OUTD = Path('reports/data_gates/G0'); OUTD.mkdir(parents=True, exist_ok=True)
ART = Path('artifacts/review'); ART.mkdir(parents=True, exist_ok=True)

def bigrams(t):
    return set(re.findall(r'[\u4e00-\u9fa5A-Za-z0-9]{2}', t or ''))

def jac(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def main():
    bt = pd.read_csv(BT, dtype=str, keep_default_na=False)
    bt['bg'] = bt['text'].apply(bigrams)
    bt['h'] = bt['text'].apply(lambda t: hashlib.md5(t.encode()).hexdigest()[:12])
    print(f"texts: {len(bt)}", flush=True)

    # A. 错位检测
    rows = []
    for code, g in bt.groupby('code'):
        idx = g.index.tolist()
        if len(idx) < 2:
            continue
        for i in idx:
            others_max = max((jac(bt.at[i,'bg'], bt.at[j,'bg']) for j in idx if j != i), default=0.0)
            if others_max < 0.05:
                rows.append({'code': code, 'year': bt.at[i,'year'],
                             'max_jaccard_other_years': round(others_max, 4),
                             'length': bt.at[i,'length'], 'anchor': bt.at[i,'anchor']})
    mis = pd.DataFrame(rows)
    mis.to_csv(ART / 's3_misalign_suspect.csv', index=False, encoding='utf-8')
    print(f"疑似错位: {len(mis)} (涉及公司 {mis['code'].nunique() if len(mis) else 0})", flush=True)

    # B1. 精确重复（跨公司同文本）
    h2codes = defaultdict(set)
    for _, r in bt.iterrows():
        h2codes[r['h']].add(r['code'])
    dup_groups = {h: cs for h, cs in h2codes.items() if len(cs) > 1}
    dup_rows = []
    for h, cs in dup_groups.items():
        sub = bt[bt['h'] == h]
        for _, r in sub.iterrows():
            dup_rows.append({'hash': h, 'code': r['code'], 'year': r['year'],
                             'n_firms_sharing': len(cs), 'length': r['length']})
    dup = pd.DataFrame(dup_rows)
    print(f"精确重复（跨公司同文本）: {len(dup)} 行 / {len(dup_groups)} 组", flush=True)

    # B2. 近重复（假边涉及公司内两两）
    fake = pd.read_csv(ART / 's3_pilot_fake_edges_408.csv', dtype=str)
    comps = set(fake['focal_id']) | set(fake['source_id'])
    cand = set()
    for c in comps:
        cand.update({c, 'sz'+c, 'sh'+c})
    sub = bt[bt['code'].isin(cand)].reset_index(drop=True)
    near = []
    for a in range(len(sub)):
        for b in range(a+1, len(sub)):
            j = jac(sub.at[a,'bg'], sub.at[b,'bg'])
            if j > 0.90 and sub.at[a,'code'] != sub.at[b,'code']:
                near.append({'code_a': sub.at[a,'code'], 'year_a': sub.at[a,'year'],
                             'code_b': sub.at[b,'code'], 'year_b': sub.at[b,'year'],
                             'jaccard': round(j, 4)})
    near_df = pd.DataFrame(near)
    print(f"近重复对（假边公司内，J>0.90）: {len(near_df)}", flush=True)

    dup.to_csv(ART / 's3_template_texts.csv', index=False, encoding='utf-8')
    if len(near_df):
        near_df.to_csv(ART / 's3_near_dup_pairs.csv', index=False, encoding='utf-8')

    summary = {
        'total_texts': int(len(bt)),
        'misalign_suspect': int(len(mis)),
        'misalign_firms': int(mis['code'].nunique()) if len(mis) else 0,
        'exact_dup_cross_firm_rows': int(len(dup)),
        'exact_dup_groups': int(len(dup_groups)),
        'near_dup_pairs_fake_companies': int(len(near_df)),
        'fake_edges_total': int(len(fake)),
        'note': 'A 跨年一致性法（max_jaccard_other_years<0.05）；B1 精确重复跨公司；B2 限假边涉及公司内',
    }
    (OUTD / 's3_text_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE", json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    import json
    main()
