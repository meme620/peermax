"""构造月度短期反转因子（REV，AH 2020 五因子第 5 个），2026-08-12。

定义（标准短期反转因子）：每月按"过去 1 个月收益"排序分 10 组，
  REV_t = 输家组（过去收益最低）在 t 月的 EW 收益 - 赢家组（过去收益最高）在 t 月的 EW 收益。
即"买输卖赢"的多空组合月收益，用于 CF MOM alpha 回归控制 1 月反转暴露
（AH 原文：past month's CF return 与股票自身收益高度相关，须控制 reversal）。

输入: data/processed/monthly_returns_clean.csv（firm_id, month, R）
输出: data/processed/rev_factor_monthly.csv（month, REV）
"""
import pandas as pd

RET = 'data/processed/monthly_returns_clean.csv'
OUT = 'data/processed/rev_factor_monthly.csv'


def main():
    ret = pd.read_csv(RET, dtype=str, keep_default_na=False)
    ret['firm_id'] = ret['firm_id'].str.replace(r'^(sz|sh)', '', regex=True)
    ret['R'] = pd.to_numeric(ret['R'], errors='coerce')
    ret = ret[ret['R'].notna()]
    months = sorted(ret['month'].unique())
    rows = []
    for i in range(len(months) - 1):
        sig_m, ret_m = months[i], months[i + 1]     # 用 t-1 收益排序，t 月收益实现因子
        sig = ret[ret['month'] == sig_m][['firm_id', 'R']].rename(columns={'R': 'sig'})
        r = ret[ret['month'] == ret_m][['firm_id', 'R']]
        d = sig.merge(r, on='firm_id', how='inner')
        if len(d) < 30:
            continue
        d['q'] = pd.qcut(d['sig'], 10, labels=False, duplicates='drop')
        loser = d[d['q'] == d['q'].min()]['R'].mean()
        winner = d[d['q'] == d['q'].max()]['R'].mean()
        rows.append({'month': ret_m, 'REV': loser - winner})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding='utf-8')
    print(f"REV 因子: {len(out)} 个月 ({out['month'].min()} ~ {out['month'].max()}) "
          f"均值={out['REV'].mean() * 100:.3f}%/月", flush=True)


if __name__ == '__main__':
    main()
