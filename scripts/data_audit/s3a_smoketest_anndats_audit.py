"""S3A smoketest C7：Anndats 月度分布审计（回填/backfill 检测，2026-08-14）。

背景：边数 2019 年出现低谷（12 万 vs 2017-12 的 34 万），需排除数据商回填假象
（backfill：报告实际发布时间更晚，但数据商按报告日批量导入，会造成早期快照前视）。
检查：
  1. Anndats 月度序列（2005-01~2025-12）：报告行数、唯一分析师、唯一公司；
  2. 环比异常突增月（top 5）；
  3. 首个快照窗口（2016-12~2017-12）内报告日的"集中度"：若回填，报告日会集中
     在少数日期（如批量导入日），正常数据则均匀分布；
  4. 2017-12-31 cutoff 当天及当月最后 5 天的报告占比（异常集中=回填迹象）。
输出: data/processed/analyst/smoketest_anndats_audit.json
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np

RAW = 'data/raw/analyst/forecast_raw_all.csv'
OUT = Path('data/processed/analyst/smoketest_anndats_audit.json')


def main():
    df = pd.read_csv(RAW, dtype=str, keep_default_na=False,
                     usecols=['Scode', 'Anndats', 'A_code'])
    df['d'] = pd.to_datetime(df['Anndats'], errors='coerce')
    df = df.dropna(subset=['d'])
    df['ym'] = df['d'].dt.strftime('%Y-%m')
    n_total = len(df)
    print(f'rows with valid Anndats: {n_total}', flush=True)

    # 1. 月度序列
    g = df.groupby('ym').agg(
        reports=('Scode', 'size'),
        analysts=('A_code', 'nunique'),
    ).reset_index()
    g['firms'] = df.groupby('ym')['Scode'].nunique().reindex(g['ym']).values

    # 2. 环比异常（月报告数增长率 top）
    g['prev'] = g['reports'].shift(1)
    g['growth'] = (g['reports'] - g['prev']) / g['prev'].replace(0, np.nan)
    top = g.nlargest(5, 'growth')[['ym', 'reports', 'growth']].to_dict('records')

    # 3. 首个快照窗口报告日集中度
    win = df[(df['d'] >= '2016-12-01') & (df['d'] <= '2017-12-31')]
    daily = win.groupby(win['d'].dt.strftime('%Y-%m-%d')).size()
    n_days = len(daily)
    top5_days = daily.nlargest(5)
    # 4. cutoff 日与月末集中度
    end_of_month = win[win['d'].dt.is_month_end]
    last5 = win[win['d'].dt.day >= 26]
    total_win = len(win)
    # 全样本月末占比对照
    all_eom = df[df['d'].dt.is_month_end]
    all_eom_ratio = len(all_eom) / n_total

    rep = {
        'n_rows_total': n_total,
        'monthly_series': g[['ym', 'reports', 'analysts', 'firms']].to_dict('records'),
        'top_growth_months': top,
        'window_201612_201712': {
            'reports': int(total_win),
            'distinct_days': int(n_days),
            'top5_days': {str(k): int(v) for k, v in top5_days.items()},
            'month_end_reports_ratio': round(float(len(end_of_month) / total_win), 4),
            'last5days_reports_ratio': round(float(len(last5) / total_win), 4),
        },
        'baseline_all_sample_month_end_ratio': round(float(all_eom_ratio), 4),
        'interpretation_rule': '回填迹象 = 首窗口报告日高度集中（top5 日占比高）或月末集中度远超全样本基线；'
                               '若 month_end_ratio 与基线相当且 top5 日占比低（<5%），判定无回填迹象。',
    }
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in rep.items() if k != 'monthly_series'}, ensure_ascii=False, indent=1), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
