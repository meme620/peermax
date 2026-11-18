"""申万行业分类映射构建（2026-09-14）。

输入：data/raw/sw2/StockClassifyUse_stock.xls（申万宏源官方 SwClass2021 股票分类使用表）
      字段：股票代码 / 计入日期 / 行业代码 / 更新日期
      层级判据（本脚本验证后固化）：行业代码为 6 位 L3 码，
        L3 = code           （如 480301）
        L2 = code[:4] + '00'（如 480300，银行-股份制银行）
        L1 = code[:3] + '000'（如 480000，银行）

产出：
  data/processed/sw2_industry_map.csv      全量历史（12,909 行）+ l1/l2/l3 派生
  data/processed/sw2_industry_latest.csv   每股最新一条（SW2021 现行口径）+ firm_id
  reports/data_gates/G0/sw2_industry_coverage.json  覆盖率与诊断

时间口径（重要，B1 τ 校准用）：
  - 主口径 = `latest`：取每股"计入日期"最大的一条 → 单一一致分类（SW2021 现行版），
    用于跨 2018-2025 全期稳定的 L2 分组；`is_current_version` 标记该码是否出现在
    2021-07-30 及以后的记录中（该代码集=版本切换后实际分配过的码）。
  - 备选口径 = `asof(t)`：对日期 t 取"计入日期 <= t"的最新一条（保留旧版代码），
    供时变稳健性检验；不改变主口径。

纪律：只做映射与诊断，不删不改原始 xls；派生列显式写入，不做隐式假设。
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLS = ROOT / 'data' / 'raw' / 'sw2' / 'StockClassifyUse_stock.xls'
OUTD = ROOT / 'data' / 'processed'
REPD = ROOT / 'reports' / 'data_gates' / 'G0'
SWITCH = pd.Timestamp('2021-07-30')  # SwClass2021 生效日（版本切换批量计入日）


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(XLS, engine='xlrd', dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    need = ['股票代码', '计入日期', '行业代码', '更新日期']
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"xls 缺少列: {missing}; 实际列={list(df.columns)}")
    df = df[need].copy()
    df['股票代码'] = df['股票代码'].astype(str).str.strip().str.zfill(6)
    df['行业代码'] = df['行业代码'].astype(str).str.strip()
    df['计入日期'] = pd.to_datetime(df['计入日期'], errors='coerce')
    df['更新日期'] = pd.to_datetime(df['更新日期'], errors='coerce')
    df['l3_code'] = df['行业代码']
    df['l2_code'] = df['行业代码'].str[:4] + '00'
    df['l1_code'] = df['行业代码'].str[:3] + '000'
    df = df.rename(columns={'股票代码': 'code6', '计入日期': 'effective_from', '更新日期': 'last_updated'})
    return df[['code6', 'effective_from', 'l1_code', 'l2_code', 'l3_code', 'last_updated']]


def main():
    df = load_raw()
    n_rows = len(df)

    # 诊断
    bad_code = df['l3_code'].str.len().ne(6) | ~df['l3_code'].str.fullmatch(r'\d{6}')
    dup_key = df.duplicated(['code6', 'effective_from'], keep=False)
    switch_codes = set(df.loc[df['effective_from'] >= SWITCH, 'l3_code'].unique())

    df['is_current_version'] = df['l3_code'].isin(switch_codes)
    df = df.sort_values(['code6', 'effective_from'])
    latest = df.groupby('code6', as_index=False).last()

    # 关联 firm_id（sh/sz/bj 前缀）
    idmap = pd.read_csv(OUTD / 'firm_id_map.csv', dtype={'code6': str})
    idmap['code6'] = idmap['code6'].str.zfill(6)
    latest = latest.merge(idmap[['code6', 'firm_id', 'name', 'board', 'out_date']], on='code6', how='left')

    OUTD.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTD / 'sw2_industry_map.csv', index=False, encoding='utf-8')
    latest.to_csv(OUTD / 'sw2_industry_latest.csv', index=False, encoding='utf-8')

    # 覆盖率：对每个语料年，统计有映射的公司数
    cov = {}
    for f, label in [('mda_texts.csv', 'm_b_mda'), ('business_texts.csv', 'm_a_business')]:
        d = pd.read_csv(OUTD / f, usecols=['code', 'year', 'status'])
        d = d[d['status'] == 'ok'].copy()
        d['code6'] = d['code'].str[2:].str.zfill(6)
        merged = d.merge(latest[['code6', 'l2_code', 'is_current_version']], on='code6', how='left')
        g = merged.groupby('year').agg(
            firms=('code', 'size'),
            mapped=('l2_code', lambda s: int(s.notna().sum())),
            current_version=('is_current_version', lambda s: int(s.fillna(False).sum())),
        )
        g['firms'] = g['firms'].astype(int)
        g['unmapped'] = g['firms'] - g['mapped']
        g['legacy_code'] = g['mapped'] - g['current_version']
        cov[label] = g.reset_index().to_dict('records')

    # L2 组规模（主口径）：全市场 + 语料内
    grp_all = latest.groupby('l2_code').agg(n=('code6', 'size'),
                                            exemplars=('name', lambda s: '/'.join(s.dropna().head(4)))).reset_index()
    rep = {
        'source': {
            'file': str(XLS.relative_to(ROOT)).replace('\\', '/'),
            'source_name': '申万宏源证券研究所 SwClass2021 股票分类使用表',
            'rows': int(n_rows),
            'stocks': int(df['code6'].nunique()),
            'code_hierarchy_rule': 'L3=code; L2=code[:4]+"00"; L1=code[:3]+"000"',
        },
        'diagnostics': {
            'bad_l3_code_rows': int(bad_code.sum()),
            'duplicate_code6_effective_from_rows': int(dup_key.sum()),
            'null_effective_from_rows': int(df['effective_from'].isna().sum()),
            'l1_groups': int(df['l1_code'].nunique()),
            'l2_groups': int(df['l2_code'].nunique()),
            'l3_groups': int(df['l3_code'].nunique()),
            'switch_date': str(SWITCH.date()),
            'codes_assigned_on_or_after_switch': len(switch_codes),
            'latest_is_current_version': int(latest['is_current_version'].sum()),
            'latest_is_legacy_code': int((~latest['is_current_version']).sum()),
        },
        'universe_join': {
            'firm_id_map_rows': int(len(idmap)),
            'matched': int(latest['firm_id'].notna().sum()),
            'unmatched_in_map': int(latest['firm_id'].isna().sum()),
            'universe_without_sw_record': int((~idmap['code6'].isin(set(latest['code6']))).sum()),
        },
        'corpus_coverage': cov,
        'l2_group_size_all_market': {
            'groups': int(len(grp_all)),
            'median': float(grp_all['n'].median()),
            'p25': float(grp_all['n'].quantile(0.25)),
            'p75': float(grp_all['n'].quantile(0.75)),
            'min': int(grp_all['n'].min()),
            'max': int(grp_all['n'].max()),
        },
        'top20_l2_by_size': grp_all.sort_values('n', ascending=False).head(20).to_dict('records'),
        'note': '行业名称未随本表提供（官方 xls 只有代码）；名称映射登记为 PENDING（可选来源：Tushare index_member_all 2000 积分 / 同花顺 F10 抽样核对）。L2 组可用成员样本辨识。',
        'run_at': datetime.now(timezone.utc).isoformat(),
    }
    # R3 独立核对（同花顺 F10）：可用时嵌入，使本产物自含三轮验收证据
    ths = REPD / 'sw2_ths_audit.json'
    if ths.exists():
        a = json.loads(ths.read_text(encoding='utf-8'))
        rep['independent_check_ths'] = {
            'source': a['source'], 'sampled': a['sampled'], 'fetched_ok': a['fetched_ok'],
            'l2_groups_covered': a['l2_groups_covered'], 'l2_groups_total': a['l2_groups_total'],
            'weighted_agreement': a['weighted_agreement'],
            'consistency_rate_strict': a['consistency_rate_strict'],
            'disagreement_n': a.get('disagreement_n'),
            'disagreement_legacy_share': a.get('disagreement_legacy_share'),
            'rule_decision': ('记录选取判据经判决性检验：max(计入日期)=规则A 加权一致率 0.9426 > '
                              'max(更新日期)=规则B 0.8757，故采用规则 A（即本表 latest）。'),
        }
    names = OUTD / 'sw2_l2_names.csv'
    if names.exists():
        rep['l2_names_source'] = {
            'file': str(names.relative_to(ROOT)).replace('\\', '/'),
            'note': 'L2 官方名称经同花顺 F10 众数补全（105/133 组完全一致；不一致组见 sw2_ths_audit.json）',
        }
    REPD.mkdir(parents=True, exist_ok=True)
    (REPD / 'sw2_industry_coverage.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({k: rep[k] for k in ['diagnostics', 'universe_join', 'l2_group_size_all_market']},
                     ensure_ascii=False, indent=2))
    print('\ncorpus coverage (M-B mda):')
    print(pd.DataFrame(cov['m_b_mda']).to_string(index=False))
    print(f'\nwrote: {OUTD / "sw2_industry_map.csv"}, {OUTD / "sw2_industry_latest.csv"}, {REPD / "sw2_industry_coverage.json"}')


if __name__ == '__main__':
    main()
