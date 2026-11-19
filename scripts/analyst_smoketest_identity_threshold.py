"""analyst-network smoketest C9：身份冲突阈值敏感性（TAIL_THRESHOLD ∈ {5,10,20}，2026-08-14）。

身份审计把次名字记录数 <= TAIL_THRESHOLD 的归为无害变体尾巴；阈值影响 conflict_ids 集合。
本脚本复现 analyst_identity_audit.py 的分类逻辑，对比三个阈值下的冲突 ID 数量与记录占比
（不重跑网络——inventory 已证明 49 个 ID 影响仅 -0.36%，此处的量级估算足够）。
输出: data/processed/analyst/smoketest_identity_threshold.json
"""
import json
from pathlib import Path
import pandas as pd

REC = 'data/processed/analyst/firm_analyst_coverage_records.csv'
OUT = Path('data/processed/analyst/smoketest_identity_threshold.json')


def classify(rec, tail_thr):
    n_conf = 0
    n_handover = 0
    n_reuse = 0
    alt_records = 0
    for aid, sub in rec.groupby('analyst_id'):
        names = sub['analyst_name'].value_counts()
        if len(names) < 2:
            continue
        main_name = names.index[0]
        rows_main = sub[sub['analyst_name'] == main_name]
        y_main = (int(rows_main['y'].min()), int(rows_main['y'].max()))
        conflict = False
        for nm, c in names.items():
            if nm == main_name:
                continue
            s = sub[sub['analyst_name'] == nm]
            y_min, y_max = int(s['y'].min()), int(s['y'].max())
            overlap = not (y_max < y_main[0] or y_min > y_main[1])
            if c <= tail_thr:
                cls = 'variant_tail'
            elif overlap:
                cls = 'suspected_handover'
                conflict = True
                n_handover += 1
            else:
                cls = 'code_reuse'
                conflict = True
                n_reuse += 1
            if cls != 'variant_tail':
                alt_records += int(c)
        if conflict:
            n_conf += 1
    return {'conflict_ids': n_conf, 'handover_ids': n_handover,
            'reuse_ids': n_reuse, 'alt_name_records': alt_records}


def main():
    rec = pd.read_csv(REC, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'analyst_id', 'analyst_name', 'report_date'])
    rec = rec[rec['analyst_id'].str.match(r'^A\d{5}$', na=False)]
    rec['y'] = pd.to_datetime(rec['report_date']).dt.year
    out = {}
    for thr in (5, 10, 20):
        out[f'tail_thr_{thr}'] = classify(rec, thr)
        print(thr, json.dumps(out[f'tail_thr_{thr}']), flush=True)
    OUT.write_text(json.dumps({
        'note': 'TAIL_THRESHOLD 敏感性（2026-08-14）。主口径阈值=10（49 个冲突 ID）。'
                'conflict_ids 数量变化对网络影响量级参考 inventory：49 个 ID 影响 -0.36%（边）。',
        'multi_name_ids': 311,
        'results': out,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
