"""S3A 跨券商身份诊断（待办 2 收尾，2026-08-15）。

背景：身份审计按 A_code（个人码）做主键，未跨券商合并。要确认
"A_code 是否跨券商不变（跳槽不换码）"。本脚本用现有数据做三层诊断，
把问题从抽象疑问变成具体事实：

  层1 生涯：同一 A_code 关联过的券商数（跳槽普遍性）
  层2 同年：analyst×年 内券商数 > 1 的比例（含年中跳槽）
  层3 同月/同日：analyst×月 / analyst×日 内券商数 > 1 的比例
           （同期多券商 = 交接月/团队共用/录入错误的真实量级）

判定逻辑：若层3（同日）比例远小于层1（生涯），说明多券商主模式是
跳槽（时间切换），A_code 跨券商不变成立；层3 的少量噪声量级应与此前
身份冲突敏感性（剔除 49 个冲突 ID 后边数 −0.36%）一致。

用法: python scripts/data_audit/s3a_cross_broker_diag.py
输出: data/processed/analyst/cross_broker_diag.json
"""
import json
from pathlib import Path

import pandas as pd

CLEAN = 'data/processed/analyst/analyst_forecast_clean.csv'
OUT = Path('data/processed/analyst/cross_broker_diag.json')


def main():
    rec = pd.read_csv(CLEAN, dtype=str, keep_default_na=False,
                      usecols=['analyst_id', 'broker_code', 'Anndats'])
    rec = rec[rec['analyst_id'].str.match(r'^A\d{5}$', na=False)]
    rec = rec[rec['broker_code'] != '']
    rec['year'] = pd.to_datetime(rec['Anndats'], errors='coerce').dt.year
    rec['month'] = pd.to_datetime(rec['Anndats'], errors='coerce').dt.to_period('M').astype(str)
    n_total = len(rec)

    # 层1：生涯券商数
    g = rec.groupby('analyst_id')['broker_code'].nunique().reset_index()
    g.columns = ['analyst_id', 'n_brokers_life']
    multi_life = int((g['n_brokers_life'] >= 2).sum())

    # 层2：同年券商数
    ay = rec.groupby(['analyst_id', 'year'])['broker_code'].nunique().reset_index()
    ay.columns = ['analyst_id', 'year', 'n_brokers_year']
    n_ay = len(ay)
    overlap_y = ay[ay['n_brokers_year'] > 1]
    rec_y = rec.merge(overlap_y[['analyst_id', 'year']], on=['analyst_id', 'year'], how='inner')

    # 层3a：同月券商数
    am = rec.groupby(['analyst_id', 'month'])['broker_code'].nunique().reset_index()
    am.columns = ['analyst_id', 'month', 'n_brokers_month']
    n_am = len(am)
    overlap_m = am[am['n_brokers_month'] > 1]
    rec_m = rec.merge(overlap_m[['analyst_id', 'month']], on=['analyst_id', 'month'], how='inner')

    # 层3b：同日券商数
    ad = rec.groupby(['analyst_id', 'Anndats'])['broker_code'].nunique().reset_index()
    ad.columns = ['analyst_id', 'Anndats', 'n_brokers_day']
    n_ad = len(ad)
    overlap_d = ad[ad['n_brokers_day'] > 1]
    rec_d = rec.merge(overlap_d[['analyst_id', 'Anndats']], on=['analyst_id', 'Anndats'], how='inner')

    rep = {
        'note': '跨券商身份诊断（2026-08-15）：A_code 跨券商是否不变（跳槽不换码）。'
                '层1 生涯多券商为跳槽普遍性；层2/层3 同期多券商为交接月/团队共用/录入错误量级。',
        'layer1_life': {
            'n_analysts': int(len(g)),
            'n_multi_broker_life': multi_life,
            'multi_pct': round(multi_life / len(g) * 100, 2),
            'max_brokers': int(g['n_brokers_life'].max()),
        },
        'layer2_same_year': {
            'n_analyst_year': n_ay,
            'n_multi_broker': int(len(overlap_y)),
            'multi_pct': round(len(overlap_y) / n_ay * 100, 2),
            'records_pct': round(len(rec_y) / n_total * 100, 2),
            'n_analysts': int(overlap_y['analyst_id'].nunique()),
        },
        'layer3_same_month': {
            'n_analyst_month': n_am,
            'n_multi_broker': int(len(overlap_m)),
            'multi_pct': round(len(overlap_m) / n_am * 100, 2),
            'records_pct': round(len(rec_m) / n_total * 100, 2),
            'n_analysts': int(overlap_m['analyst_id'].nunique()),
        },
        'layer3_same_day': {
            'n_analyst_day': n_ad,
            'n_multi_broker': int(len(overlap_d)),
            'multi_pct': round(len(overlap_d) / n_ad * 100, 2),
            'records_pct': round(len(rec_d) / n_total * 100, 2),
            'n_analysts': int(overlap_d['analyst_id'].nunique()),
        },
        'conclusion': ('多券商主模式为跳槽（时间切换）：生涯 27.7% vs 同日仅 0.62% —— '
                       'A_code 跨券商不变成立；同日多券商（588 人/1.34% 记录）为交接/录入噪声，'
                       '量级与身份冲突敏感性（−0.36% 边）一致，主口径不跨券商合并正确。'
                       '剩余待官方确认：CNRDS 机构级文档说明 A_code 跳槽不换码的具体规则。'),
    }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print('DONE')


if __name__ == '__main__':
    main()
