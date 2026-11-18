"""S3A 分析师身份冲突审计（2026-08-12，三轮对抗 review 落地）。

背景：A_code 多姓名的 279 个 ID 中，绝大多数是同一人的姓名拼写变体
（错别字/繁简/拼音-中文/空格/团队署名尾巴），少数是疑似 code 换人/复用。
本脚本输出分类审计表，量化其对共同分析师网络的影响，供身份冲突敏感性使用。

分类规则（对每个多姓名 ID，主名字=记录数最多者，其余为次名字）：
  variant_tail        : 次名字记录数 <= 10           -> 录入/团队署名尾巴，同人，无害保留
  suspected_handover  : 次名字记录数 > 10 且与主名字时间重叠 -> 疑似同一 code 换人，需敏感性
  code_reuse          : 次名字与主名字时间不重叠          -> code 退役后复用，跨期假边，需敏感性

输出:
  data/processed/analyst/identity_audit.json   分类清单 + 影响量级
  data/processed/analyst/identity_conflict_ids.json  疑似换人 + code 复用 ID（供敏感性剔除）
"""
import json
from pathlib import Path
import pandas as pd

REC = 'data/processed/analyst/firm_analyst_coverage_records.csv'
OUTD = Path('data/processed/analyst')
TAIL_THRESHOLD = 10


def main():
    rec = pd.read_csv(REC, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'analyst_id', 'analyst_name', 'report_date'])
    rec = rec[rec['analyst_id'].str.match(r'^A\d{5}$', na=False)]
    rec['y'] = pd.to_datetime(rec['report_date']).dt.year

    out = []
    for aid, sub in rec.groupby('analyst_id'):
        names = sub['analyst_name'].value_counts()
        if len(names) < 2:
            continue
        main_name = names.index[0]
        rows_main = sub[sub['analyst_name'] == main_name]
        y_main = (int(rows_main['y'].min()), int(rows_main['y'].max()))
        detail = []
        conflict = False
        for nm, c in names.items():
            if nm == main_name:
                continue
            s = sub[sub['analyst_name'] == nm]
            y_min, y_max = int(s['y'].min()), int(s['y'].max())
            overlap = not (y_max < y_main[0] or y_min > y_main[1])
            if c <= TAIL_THRESHOLD:
                cls = 'variant_tail'
            elif overlap:
                cls = 'suspected_handover'
                conflict = True
            else:
                cls = 'code_reuse'
                conflict = True
            detail.append({'name': nm, 'records': int(c),
                           'year_range': [y_min, y_max], 'overlap_with_main': overlap, 'class': cls})
        out.append({'analyst_id': aid, 'main_name': main_name,
                    'main_records': int(names.iloc[0]),
                    'main_year_range': list(y_main),
                    'alt_names': detail,
                    'needs_sensitivity': conflict})

    df = pd.DataFrame(out)
    n_tail = sum(1 for r in out if not r['needs_sensitivity'])
    n_conf = sum(1 for r in out if r['needs_sensitivity'])
    n_handover = sum(1 for r in out if any(d['class'] == 'suspected_handover' for d in r['alt_names']))
    n_reuse = sum(1 for r in out if any(d['class'] == 'code_reuse' for d in r['alt_names']))
    print(f"多姓名 ID 总数: {len(df)} | 无害变体尾巴: {n_tail} | 疑似换人: {n_handover} | code 复用: {n_reuse}")

    # 影响量级：疑似换人/复用 ID 的 alt-name 记录数与覆盖公司
    conf_ids = [r['analyst_id'] for r in out if r['needs_sensitivity']]
    sub2 = rec[rec['analyst_id'].isin(conf_ids)]
    alt_rec = 0
    for r in out:
        if r['needs_sensitivity']:
            alt_rec += sum(d['records'] for d in r['alt_names'])
    print(f"疑似换人/复用 ID 数: {len(conf_ids)} | 其 alt-name 记录数合计: {alt_rec} | 覆盖公司数: {sub2['firm_id'].nunique()}")

    rep = {
        'note': 'variant_tail=同人拼写变体（无害保留）；suspected_handover=次名字>10条且时间重叠（疑似code换人）；code_reuse=时间不重叠（code复用）',
        'tail_threshold': TAIL_THRESHOLD,
        'summary': {'multi_name_ids': len(df), 'variant_tail': n_tail,
                    'suspected_handover': n_handover, 'code_reuse': n_reuse,
                    'conflict_ids_total': len(conf_ids),
                    'alt_name_records_total': alt_rec,
                    'conflict_firms_covered': int(sub2['firm_id'].nunique())},
        'records': out,
    }
    (OUTD / 'identity_audit.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTD / 'identity_conflict_ids.json').write_text(json.dumps(
        {'conflict_ids': conf_ids,
         'note': '供身份冲突敏感性：剔除这些 ID 后重建网络，对比边数/degree'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE identity audit')


if __name__ == '__main__':
    main()
