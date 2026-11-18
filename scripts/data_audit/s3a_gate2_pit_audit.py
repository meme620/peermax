"""S3A Gate 2.5：时点链审计。

逐项检查：
  1. 网络可用性：覆盖表 available_at / coverage_end 不得晚于 network_cutoff；
  2. 窗口上界：窗口内报告的 report_date ≤ cutoff（由 coverage_end 复核）；
  3. 边表时间：available_at == network_cutoff，且 source_version 一致；
  4. 无未来字段：网络快照/边表不得含 Anndats_act 或收益字段；
  5. 边表不变量：无自环、无重复键、focal≠source；
  6. 形成月 q10 与 IPO 过滤：由生产脚本静态检查（代码级），记录证据行；
  7. 权重分母：focal 行归一化（C 线已复核）。

产出：reports/data_gates/G0/s3a_gate2_pit_audit.json（+ Markdown 摘要）
"""
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'configs/s3a_final_spec_v1.yaml'
SPEC_HASH = sha256(SPEC.read_bytes()).hexdigest()
COV = ROOT / 'data/processed/analyst/firm_analyst_coverage.csv'
EDGED = ROOT / 'data/processed/analyst/edges'
VERSION = 'coverage_win12_fy12_v3'
FORBIDDEN_COLS = ['Anndats_act', 'anndats_act', 'r_pred', 'ret', 'R', 'return']


def main():
    spec = yaml.safe_load(SPEC.read_text(encoding='utf-8'))
    cov = pd.read_csv(COV, dtype=str,
                      usecols=['network_cutoff', 'firm_id', 'analyst_id',
                               'available_at', 'coverage_end', 'coverage_start'])
    checks = {}

    # 1-2. 可用性与窗口上界（coverage_end 是窗口内最后一份报告日）
    late = cov[cov['coverage_end'] > cov['network_cutoff']]
    checks['coverage_end_le_cutoff'] = {
        'ok': len(late) == 0, 'violations': int(len(late)),
        'note': '窗口内报告日必须 ≤ cutoff（searchsorted side="right"）'}
    late2 = cov[cov['available_at'] > cov['network_cutoff']]
    checks['available_at_le_cutoff'] = {
        'ok': len(late2) == 0, 'violations': int(len(late2)),
        'note': 'available_at=最后一份报告日，不得晚于 cutoff'}
    # coverage_start 至少早于 cutoff（12 个月窗口语义），允许首月不完整
    bad_start = cov[(cov['coverage_start'] > cov['network_cutoff'])]
    checks['coverage_start_le_cutoff'] = {
        'ok': len(bad_start) == 0, 'violations': int(len(bad_start))}

    # 3-4. 边表列与时间（抽查 3 个月全列 + 全部 241 月表头）
    spot = ['2006-06-30', '2015-06-30', '2024-06-30']
    edge_checks, header_ok, hdr_viol = [], 0, []
    for f in sorted(EDGED.glob(f'edges_{VERSION}_*.csv')):
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if cols == COLS_EXPECTED:
            header_ok += 1
        else:
            hdr_viol.append({'file': f.name,
                             'missing': [c for c in COLS_EXPECTED if c not in cols],
                             'extra': [c for c in cols if c not in COLS_EXPECTED]})
    checks['edge_header_contract'] = {
        'ok': len(hdr_viol) == 0, 'files_checked': header_ok,
        'violations': hdr_viol[:5], 'expected_cols': COLS_EXPECTED}

    for c in spot:
        f = EDGED / f'edges_{VERSION}_{c}.csv'
        e = pd.read_csv(f, dtype={'focal_id': str, 'source_id': str})
        edge_checks.append({
            'cutoff': c,
            'rows': int(len(e)),
            'available_at_eq_cutoff': bool((e['available_at'] == c).all()),
            'network_cutoff_eq': bool((e['network_cutoff'] == c).all()),
            'self_loops': int((e['focal_id'] == e['source_id']).sum()),
            'dup_keys': int(e.duplicated(['focal_id', 'source_id']).sum()),
            'forbidden_cols_present': [x for x in FORBIDDEN_COLS if x in e.columns],
            'source_version_nunique': int(e['source_version'].nunique()),
            'support_status_values': sorted(e['network_support_status'].unique().tolist()),
        })
    checks['edge_spot_time_chain'] = {
        'ok': all(x['available_at_eq_cutoff'] and x['network_cutoff_eq']
                  and x['self_loops'] == 0 and x['dup_keys'] == 0
                  and not x['forbidden_cols_present'] for x in edge_checks),
        'detail': edge_checks}

    # 5. 覆盖表列合同（不得含未来字段）
    cov_cols = pd.read_csv(COV, nrows=0).columns.tolist()
    checks['coverage_header_no_future'] = {
        'ok': not [x for x in FORBIDDEN_COLS if x in cov_cols],
        'cols': cov_cols}

    # 6. 代码级：q10 先算后合并、IPO 过滤
    cf = (ROOT / 'scripts/data_audit/s3a_cf_mom_replication.py').read_text(encoding='utf-8')
    npm = (ROOT / 'scripts/data_audit/s3a_npm_replication.py').read_text(encoding='utf-8')
    checks['code_forming_month_q10'] = {
        'ok': all('quantile(0.10)' in t for t in (cf, npm)),
        'note': '形成月全市场横截面 q10（FROZEN 口径）'}
    checks['code_recent_ipo_filter'] = {
        'ok': all('recent_ipo' in t for t in (cf, npm)),
        'note': '上市不足 6 个月剔除'}
    # 生产 snapshot 是否用 Anndats 而非 Anndats_act
    s5 = (ROOT / 'scripts/data_audit/s3a_step5_snapshot.py').read_text(encoding='utf-8')
    checks['code_uses_anndats_only'] = {
        'ok': ('Anndats' in s5) and ("Anndats_act" not in s5.replace('Anndats_act 未进入', '')),
        'note': 'step5 只读 Anndats；Anndats_act 不出现在取数列'}

    result = {
        'spec_id': spec['spec_id'], 'spec_hash': SPEC_HASH,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv), 'script': Path(__file__).name,
        'network_version': VERSION,
        'checks': checks,
    }
    result['status'] = 'PASS' if all(v['ok'] for v in checks.values()) else 'FAIL'
    out = ROOT / 'reports/data_gates/G0/s3a_gate2_pit_audit.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')

    lines = ['# S3A Gate 2.5 时点链审计\n',
             f'- spec_hash `{SPEC_HASH}`；状态 **{result["status"]}**\n',
             '| 检查项 | 结论 | 证据 |', '|---|---|---|']
    for k, v in checks.items():
        ev = (f"违规 {v.get('violations')}" if 'violations' in v and isinstance(v['violations'], int)
              else ('违规 ' + str(len(v['violations'])) if isinstance(v.get('violations'), list) and v['violations']
                    else '见 JSON'))
        lines.append(f"| {k} | {'通过' if v['ok'] else '不通过'} | {ev} |")
    (ROOT / 'reports/data_gates/G0/s3a_gate2_pit_audit.md').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v['ok'] for k, v in checks.items()}, ensure_ascii=False, indent=1))
    print('status =', result['status'], '->', out.relative_to(ROOT))


COLS_EXPECTED = ['network_use_case', 'network_version', 'focal_id', 'source_id',
                 'common_analyst_count', 'link_strength', 'analyst_count_focal',
                 'analyst_count_source', 'union_analyst_count', 'jaccard', 'weight',
                 'SourceFocalDistinct', 'network_cutoff', 'available_at', 'source_version',
                 'coverage_flag', 'missing_reason', 'network_support_status']

if __name__ == '__main__':
    main()
