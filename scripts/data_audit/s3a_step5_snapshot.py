"""S3A Step 5：月度覆盖快照 -> firm_analyst_coverage（零外部依赖）。

规则（execution_plan S3A Step 4 / protocol §5A 第 6 步）:
  - 月末 snapshot（自然月末）
  - 窗口：cutoff 前 12 个月（coverage_window=12，2026-08-12 冻结）
  - 只使用 report_date <= cutoff 的记录
  - 首个 cutoff 前不足 12 个月历史 -> coverage_window_incomplete（按数据最早报告日实际计算）

v2 变更（2026-08-12，对抗检查 R2-1 修复，registry fy1_fy2_rule change_log 同日登记）:
  - FY1/FY2 行级过滤移到报告级去重之前：输入由 firm_analyst_coverage_records
    （报告级，v1 链路）改为 analyst_forecast_clean（行级）。凡含至少一条 FY1/FY2
    预测行的报告即计入覆盖，对齐 A&H "at least one FY1 or FY2 earnings forecast
    over the past 12 months" 语义；冻结的 gap 规则本身不变。
  - v1 顺序（先 keep='first' 去重后过滤）误杀 309,341/1,815,139 份报告（17.0%）、
    丢失 31,466/385,085 个公司-分析师 pair（8.2%）；v2 在输出 JSON 的
    report_level_audit 块中固化该对比（误杀数必须为 0）。
  - report_count 语义不变（窗口内报告数），行级输入下用 report_date 的 nunique 实现。
  - coverage_window_incomplete 由硬编码 0 改为按数据最早报告日实际计算。
  - broker_count 不计空串券商码（与 step9 BrokerCoverage 口径一致）。

输出:
  data/processed/analyst/firm_analyst_coverage.csv
  唯一键 (coverage_window_version, network_cutoff, firm_id, analyst_id)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC = 'data/processed/analyst/analyst_forecast_clean.csv'
OUTD = Path('data/processed/analyst')
OUTD.mkdir(parents=True, exist_ok=True)
WINDOW_MONTHS = 12
VERSION = 'coverage_win12_fy12_v3'
SOURCE_VERSION = 'forecast_shard_v1_20260811'


def main():
    rec = pd.read_csv(SRC, dtype=str, keep_default_na=False,
                      usecols=['firm_id', 'Anndats', 'Fpenddt', 'analyst_id',
                               'broker_code', 'mapping_status'])
    rec = rec[rec['mapping_status'] == 'ok']
    # A_code 格式校验（2026-08-12 身份审计发现非法代码 '0'/'S31388'/'A3162' 把不同分析师合并为同一 ID）
    bad_mask = ~rec['analyst_id'].str.match(r'^A\d{5}$', na=False)
    n_bad = int(bad_mask.sum())
    rec = rec[~bad_mask]
    print(f"records after A_code format check: {len(rec)} (dropped {n_bad} bad-code rows)", flush=True)

    rec['report_date'] = pd.to_datetime(rec['Anndats'])
    rec['Fpenddt_dt'] = pd.to_datetime(rec['Fpenddt'], errors='coerce')
    # FY1/FY2 规则（registry fy1_fy2_rule，2026-08-12 冻结）：
    #   gap = Fpenddt.year - report.year；0 -> FY1，1 -> FY2；gap<0 回溯、gap>=2 远期、Fpenddt 缺失剔除
    rec['gap'] = rec['Fpenddt_dt'].dt.year - rec['report_date'].dt.year
    rec['_fy_ok'] = rec['gap'].between(0, 1)
    print(f"records: {len(rec)}", flush=True)

    # ---- 报告级 v1/v2 对比审计（误杀断言：v2 下含 FY1/FY2 行的报告必须全部保留）----
    rkey = rec['firm_id'] + '|' + rec['analyst_id'] + '|' + rec['Anndats']
    rep = rec.groupby(rkey, sort=False)['_fy_ok'].agg(['first', 'any'])
    n_rep = len(rep)
    n_v1 = int(rep['first'].sum())          # v1 规则：报告首行通过才保留
    n_v2 = int(rep['any'].sum())            # v2 规则：报告任一行通过即保留
    pair_key = pd.Index(rep.index).str.split('|').map(lambda p: p[0] + '|' + p[1])
    pairs_v2 = pair_key[rep['any'].values].nunique()
    pairs_v1 = pair_key[rep['first'].values].nunique()
    report_audit = {
        'note': 'v1=报告首行 gap∈[0,1] 才保留（先keep-first去重后过滤）；v2=报告任一行通过即保留（论文 at-least-one 语义）',
        'reports_total': int(n_rep),
        'reports_kept_v1_first_rule': n_v1,
        'reports_kept_v2_any_rule': n_v2,
        'reports_miskilled_under_v1': int(n_v2 - n_v1),
        'pairs_v1_first_rule': int(pairs_v1),
        'pairs_v2_any_rule': int(pairs_v2),
        'pairs_lost_under_v1': int(pairs_v2 - pairs_v1),
    }
    print("report audit:", json.dumps(report_audit, ensure_ascii=False), flush=True)

    # ---- 行级 FY1/FY2 过滤（先于任何形式的报告级去重）----
    fy_tot = len(rec)
    fy_keep = int(rec['_fy_ok'].sum())
    fy_drop_long = int((rec['gap'] > 1).sum())
    fy_drop_past = int((rec['gap'] < 0).sum())
    fy_drop_missing = int(rec['gap'].isna().sum())
    fy = rec[rec['_fy_ok']].copy()
    del rec

    # 月末 cutoff：2005-12-31 起（CNRDS 数据 2005-01 起，首个满足 12 个月窗口的月份；2026-08-29 样本期扩展决策）至 2025-12-31
    cutoffs = pd.date_range('2005-12-31', '2025-12-31', freq='ME')
    print(f"cutoffs: {len(cutoffs)} ({cutoffs[0].date()} ~ {cutoffs[-1].date()})", flush=True)

    # 按 report_date 排序 + searchsorted 取窗口（避免 97 次全表布尔扫描）
    fy = fy.sort_values('report_date', kind='mergesort')
    dates = fy['report_date'].to_numpy(dtype='datetime64[ns]')
    earliest = fy['report_date'].iloc[0]
    incomplete_count = 0

    frames = []
    for cutoff in cutoffs:
        ws = cutoff - pd.DateOffset(months=WINDOW_MONTHS)
        if ws < earliest:
            incomplete_count += 1
        ws64 = np.datetime64(ws)
        c64 = np.datetime64(cutoff)
        lo = dates.searchsorted(ws64, side='right')    # report_date > ws
        hi = dates.searchsorted(c64, side='right')     # report_date <= cutoff
        if hi <= lo:
            continue
        sub = fy.iloc[lo:hi]
        g = sub.groupby(['firm_id', 'analyst_id'], sort=False).agg(
            report_count=('Anndats', 'nunique'),
            first_report_date=('report_date', 'min'),
            last_report_date=('report_date', 'max'),
            broker_count=('broker_code', lambda s: int(s[s != ''].nunique())),
        ).reset_index()
        g['coverage_window_version'] = VERSION
        g['network_cutoff'] = cutoff.strftime('%Y-%m-%d')
        g['coverage_start'] = g['first_report_date'].dt.strftime('%Y-%m-%d')
        g['coverage_end'] = g['last_report_date'].dt.strftime('%Y-%m-%d')
        g['available_at'] = g['last_report_date'].dt.strftime('%Y-%m-%d')
        g['source_version'] = SOURCE_VERSION
        g['coverage_flag'] = 'covered'
        g['missing_reason'] = ''
        frames.append(g[['coverage_window_version', 'network_cutoff', 'firm_id', 'analyst_id',
                         'coverage_start', 'coverage_end', 'report_count',
                         'broker_count', 'first_report_date', 'last_report_date',
                         'available_at', 'source_version', 'coverage_flag', 'missing_reason']])
        if len(frames) % 12 == 0:
            print(f"cutoff {cutoff.date()} done ({len(frames)} months)", flush=True)

    out = pd.concat(frames, ignore_index=True)
    out['first_report_date'] = out['first_report_date'].dt.strftime('%Y-%m-%d')
    out['last_report_date'] = out['last_report_date'].dt.strftime('%Y-%m-%d')
    print(f"coverage rows: {len(out)}", flush=True)
    # 唯一键检查
    dup = out.duplicated(['coverage_window_version', 'network_cutoff', 'firm_id', 'analyst_id']).sum()
    print(f"key duplicates: {dup}", flush=True)
    out.to_csv(OUTD / 'firm_analyst_coverage.csv', index=False, encoding='utf-8')

    monthly = out.groupby('network_cutoff').size()
    rep_out = {
        'rows': int(len(out)),
        'key_duplicates': int(dup),
        'window_months': WINDOW_MONTHS,
        'coverage_window_version': VERSION,
        'cutoffs': int(len(cutoffs)),
        'monthly_covered_pairs_min': int(monthly.min()),
        'monthly_covered_pairs_max': int(monthly.max()),
        'monthly_covered_pairs_mean': int(monthly.mean()),
        'unique_firms_monthly_mean': int(out.groupby('network_cutoff')['firm_id'].nunique().mean()),
        'coverage_window_incomplete_count': int(incomplete_count),
        'fy_filter': {
            'note': 'v2 起为行级统计（全样本，非按窗口实例）；v1 为窗口内报告级统计',
            'rows_total': int(fy_tot),
            'kept_fy1_fy2': fy_keep,
            'dropped_fy3plus': fy_drop_long,
            'dropped_backcast': fy_drop_past,
            'dropped_missing_fpenddt': fy_drop_missing,
        },
        'report_level_audit': report_audit,
    }
    (OUTD / 'step5_snapshot.json').write_text(json.dumps(rep_out, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE", json.dumps(rep_out, ensure_ascii=False)[:600])


if __name__ == '__main__':
    main()
