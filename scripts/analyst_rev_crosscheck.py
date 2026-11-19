"""analyst-network REV 因子交叉验证（P0-3 收尾，2026-08-14）。

背景：项目 FF5 里的短期反转因子 REV 由 build_rev_factor.py 自建
（每月按 t-1 收益分 10 组、买输家卖赢家的 EW 多空，Ken French 标准定义），
无外部因子序列交叉校验（A&H 用 Ken French 官网、Long 用 CSMAR 因子）。
本脚本提供可执行的交叉验证路径：

  1. 读入外部 REV 序列（CSMAR 等公开因子；约定格式：两列 csv，
     month 为 'YYYY-MM'，REV 为小数月收益）；
  2. 与自建 REV 对比：重叠月份数、均值、标准差、Pearson/Spearman 相关；
  3. 用外部 REV 替换自建 REV，重跑 CF MOM（ln/h1）FF5 回归
     （复用 cf_mom_alpha.py 的 Newey-West OLS），与自建口径
     （α=1.5048%/月, t=4.00）对比——替换后 α/t 若基本不变，
     headline 对 REV 构造的依赖即通过外部校验。

用法:
  python scripts/analyst_rev_crosscheck.py                        # 外部数据缺失：输出状态说明
  python scripts/analyst_rev_crosscheck.py data/raw/external/REV_csmar.csv

输出: data/processed/analyst/rev_crosscheck.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from statsutil import nw_ols

SELF_REV = 'data/processed/rev_factor_monthly.csv'
CF = 'data/processed/analyst/cf_mom_replication_ln_h1.json'
FAC = 'data/processed/factor_returns_daily.csv'
OUT = Path('data/processed/analyst/rev_crosscheck.json')


def _monthly_ff4():
    """FF4 日度因子复利转月（与 cf_mom_alpha.py 同逻辑）。"""
    f = pd.read_csv(FAC, dtype=str, keep_default_na=False)
    f['date'] = pd.to_datetime(f['date'])
    f['month'] = f['date'].dt.strftime('%Y-%m')
    for col in ['rf_d', 'MKT_d', 'SMB_d', 'HML_d', 'MOM_d']:
        f[col] = pd.to_numeric(f[col], errors='coerce')
    return f.groupby('month').apply(lambda d: pd.Series({
        'MKT': float((1 + d['MKT_d']).prod() - 1),
        'SMB': float((1 + d['SMB_d']).prod() - 1),
        'HML': float((1 + d['HML_d']).prod() - 1),
        'MOM': float((1 + d['MOM_d']).prod() - 1),
    }), include_groups=False)


def _alpha_with_rev(rev_series):
    """用给定 REV 月序列重跑 CF MOM（ln/h1）FF5/FF4 回归，返回 EW/VW α 摘要。"""
    cf = json.loads(Path(CF).read_text(encoding='utf-8'))
    df = pd.DataFrame(cf['monthly'])
    df = df.merge(_monthly_ff4(), left_on='month', right_index=True, how='inner')
    df = df.merge(rev_series.rename('REV'), on='month', how='left')
    df = df.sort_values('month').reset_index(drop=True)
    out = {}
    for dep in ['ew_ls', 'vw_ls']:
        d5 = df.dropna(subset=[dep, 'MKT', 'SMB', 'HML', 'MOM', 'REV'])
        y5 = d5[dep].values
        X5 = np.column_stack([np.ones(len(y5)), d5['MKT'], d5['SMB'], d5['HML'], d5['MOM'], d5['REV']])
        beta5, se5, t5 = nw_ols(X5, y5, L=1)
        X4 = np.column_stack([np.ones(len(df)), df['MKT'], df['SMB'], df['HML'], df['MOM']])
        y4 = df[dep].values
        beta4, se4, t4 = nw_ols(X4, y4, L=1)
        out[dep] = {
            'ff5_alpha_pct': round(float(beta5[0]) * 100, 4), 'ff5_alpha_t': round(float(t5[0]), 2),
            'ff5_rev_beta_pct': round(float(beta5[5]) * 100, 4), 'ff5_rev_t': round(float(t5[5]), 2),
            'ff4_alpha_pct': round(float(beta4[0]) * 100, 4), 'ff4_alpha_t': round(float(t4[0]), 2),
            'ff5_n_months': int(len(y5)),
        }
    return out


def main():
    self_rev = pd.read_csv(SELF_REV, dtype=str, keep_default_na=False)
    self_rev['REV_self'] = pd.to_numeric(self_rev['REV'], errors='coerce')
    self_rev = self_rev[['month', 'REV_self']]

    external_path = sys.argv[1] if len(sys.argv) > 1 else ''
    if not external_path or not Path(external_path).exists():
        rep = {
            'status': 'external_rev_missing',
            'note': '未提供外部 REV 序列。交叉验证需外部因子（CSMAR 等）的月度 REV 序列，'
                    '约定格式：两列 csv，表头 month,REV；month 为 YYYY-MM（小数月收益，如 0.0123）。'
                    '拿到数据后：python scripts/analyst_rev_crosscheck.py <外部文件路径>',
            'self_rev_stats': {
                'n_months': int(self_rev['REV_self'].notna().sum()),
                'month_range': [self_rev['month'].min(), self_rev['month'].max()],
                'mean_pct': round(float(self_rev['REV_self'].mean()) * 100, 4),
                'std_pct': round(float(self_rev['REV_self'].std()) * 100, 4),
            },
        }
        OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        print('DONE (missing external)')
        return

    ext = pd.read_csv(external_path, dtype=str, keep_default_na=False)
    ext.columns = [c.strip().lower() for c in ext.columns]
    if 'month' not in ext.columns or 'rev' not in ext.columns:
        raise SystemExit(f"外部 REV 文件格式不符：需 month,REV 两列，实际列={list(ext.columns)}")
    ext['REV_ext'] = pd.to_numeric(ext['rev'], errors='coerce')
    ext = ext[['month', 'REV_ext']].dropna(subset=['REV_ext'])

    m = self_rev.merge(ext, on='month', how='inner')
    overlap = len(m)
    pearson = float(m['REV_self'].corr(m['REV_ext'])) if overlap > 2 else None
    spearman = float(m['REV_self'].rank().corr(m['REV_ext'].rank())) if overlap > 2 else None

    alpha_self = _alpha_with_rev(self_rev.set_index('month')['REV_self'])
    alpha_ext = _alpha_with_rev(ext.set_index('month')['REV_ext'])

    rep = {
        'status': 'done',
        'external_source': external_path,
        'overlap': {
            'n_months': overlap,
            'mean_self_pct': round(float(m['REV_self'].mean()) * 100, 4),
            'mean_ext_pct': round(float(m['REV_ext'].mean()) * 100, 4),
            'std_self_pct': round(float(m['REV_self'].std()) * 100, 4),
            'std_ext_pct': round(float(m['REV_ext'].std()) * 100, 4),
            'pearson_corr': pearson,
            'spearman_corr': spearman,
        },
        'alpha_self_rev': alpha_self,
        'alpha_external_rev': alpha_ext,
    }

    if not alpha_ext or overlap < 12:
        conclusion = '外部 REV 与自建口径重叠月过少（或替换失败），需人工评估'
    else:
        dt = abs(alpha_ext['ew_ls']['ff5_alpha_t'] - 4.0)
        da = abs(alpha_ext['ew_ls']['ff5_alpha_pct'] - 1.5048)
        if pearson is not None and pearson >= 0.9 and da < 0.2 and dt < 1.0:
            conclusion = (f'外部 REV 通过交叉验证：Pearson={pearson:.3f}，替换后 EW FF5 α='
                          f'{alpha_ext["ew_ls"]["ff5_alpha_pct"]}%/月（t={alpha_ext["ew_ls"]["ff5_alpha_t"]}），'
                          f'与自建口径（1.5048%/t=4.00）基本一致 → headline 对 REV 构造不敏感')
        elif pearson is not None and pearson >= 0.8 and da < 0.5 and dt < 2.0:
            conclusion = (f'外部 REV 与自建口径有差异但结论稳健（Pearson={pearson:.3f}，'
                          f'α={alpha_ext["ew_ls"]["ff5_alpha_pct"]}%/t={alpha_ext["ew_ls"]["ff5_alpha_t"]}），'
                          f'建议论文双口径报告并说明差异来源')
        else:
            conclusion = (f'外部 REV 与自建口径差异较大（Pearson={pearson}，'
                          f'α={alpha_ext["ew_ls"]["ff5_alpha_pct"]}%/t={alpha_ext["ew_ls"]["ff5_alpha_t"]}），'
                          f'需人工排查 REV 构造口径差异')

    rep['conclusion'] = conclusion
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print('DONE')


if __name__ == '__main__':
    main()
