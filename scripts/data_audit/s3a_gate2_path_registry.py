"""S3A Gate 2.6：下游路径与依赖登记核对。

核对：
  1. runner 是否包含全部 13 步（含 step11 / 9A–9M 专项是否在 runner 之外）；
  2. 各步实际输出路径与 execution plan / run record 记载是否一致；
  3. 关键产物的月份覆盖（全期 241 月 vs 短样本 96 月）与 network_version；
  4. analyst_edge_members 是否生成（未生成须登记为可选审计缺口）；
  5. build_firm_chars -> step11 -> M1/M2 依赖是否成立。

产出：reports/data_gates/G0/s3a_gate2_path_registry.json
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

RUNNER_STEPS = ['s3a_step1_ingest.py', 's3a_step2_audit.py', 's3a_step3_clean.py',
                's3a_step4_dedup.py', 's3a_step5_snapshot.py', 's3a_step67_edges.py',
                's3a_step8_diag.py', 's3a_step9_derived.py', 's3a_step10_panel_merge.py',
                's3a_sens_specs.py', 's3a_cf_mom_replication.py', 's3a_cf_mom_alpha.py',
                's3a_identity_audit.py']
OUTSIDE_RUNNER = ['s3a_step11_merge_chars.py', 's3a_fundamental_link.py',
                  's3a_9c_fm_regression.py', 's3a_9e_subsample.py', 's3a_9g_cffr.py',
                  's3a_9h_absorb.py', 's3a_9i_additional_tests.py', 's3a_9j_extra.py',
                  's3a_9k_nofy_build.py', 's3a_9k_compare.py', 's3a_9m_cfret2.py']

ARTIFACTS = [
    ('data/processed/analyst/firm_analyst_coverage.csv', 'month'),
    ('data/processed/analyst/firm_analyst_derived.csv', 'month'),
    ('data/processed/firm_chars_clean.csv', 'month'),
    ('data/processed/monthly_pricing_panel_with_analyst.csv', 'month'),
    ('data/processed/monthly_pricing_panel_partial.csv', 'month'),
]


def month_span(path, col):
    try:
        df = pd.read_csv(ROOT / path, usecols=[col], dtype=str)
    except Exception as e:
        return {'error': str(e)[:120]}
    return {'rows': int(len(df)), 'months': int(df[col].nunique()),
            'min': str(df[col].min()), 'max': str(df[col].max())}


def main():
    spec = yaml.safe_load(SPEC.read_text(encoding='utf-8'))
    runner_txt = (ROOT / 'scripts/data_audit/run_s3a_pipeline.py').read_text(encoding='utf-8')
    # STEPS 元组形如 ('1_ingest', 's3a_step1_ingest.py', [])：脚本名是第二个字段
    runner_files = [l.split("'")[3] for l in runner_txt.splitlines()
                    if l.strip().startswith("('") and ".py'" in l]

    artifacts = {}
    for path, col in ARTIFACTS:
        p = ROOT / path
        rec = {'exists': p.exists()}
        if p.exists():
            rec.update(month_span(path, col))
        artifacts[path] = rec

    # 覆盖表用 network_cutoff 而非 month
    cov = pd.read_csv(ROOT / 'data/processed/analyst/firm_analyst_coverage.csv',
                      usecols=['network_cutoff'], dtype=str)
    artifacts['data/processed/analyst/firm_analyst_coverage.csv'] = {
        'exists': True, 'rows': int(len(cov)), 'months': int(cov['network_cutoff'].nunique()),
        'min': cov['network_cutoff'].min(), 'max': cov['network_cutoff'].max()}

    edge_members = list(ROOT.glob('data/**/*edge_member*')) + list(ROOT.glob('artifacts/**/*edge_member*'))
    step11 = (ROOT / 'scripts/data_audit/s3a_step11_merge_chars.py').read_text(encoding='utf-8')

    checks = {
        'runner_has_13_steps': len(runner_files) == 13,
        'runner_covers_base_chain': all(s in runner_files for s in RUNNER_STEPS),
        'step11_outside_runner_documented': 's3a_step11_merge_chars.py' not in runner_files,
        'edge_members_not_generated': len(edge_members) == 0,
        'step11_writes_firm_chars': 'firm_chars_clean.csv' in step11 or 'CHARS' in step11,
        'panel_full_period': artifacts['data/processed/monthly_pricing_panel_with_analyst.csv'].get('months', 0) >= 241,
        'derived_full_period': artifacts['data/processed/analyst/firm_analyst_derived.csv'].get('months', 0) == 241,
        'firm_chars_has_analyst_cov': True,
    }
    # firm_chars analyst_cov 覆盖
    fc = pd.read_csv(ROOT / 'data/processed/firm_chars_clean.csv', usecols=['analyst_cov'])
    checks['firm_chars_has_analyst_cov'] = int(fc['analyst_cov'].notna().sum()) > 0

    gaps = []
    if not checks['panel_full_period']:
        gaps.append({
            'id': 'panel_short_sample',
            'artifact': 'data/processed/monthly_pricing_panel_with_analyst.csv',
            'observed': artifacts['data/processed/monthly_pricing_panel_with_analyst.csv'],
            'impact': '该面板仍为 2018-2025 短样本；S3A 全期（241 月）分析不依赖此面板，'
                      '但 M1/M2 若引用该面板则仅覆盖短样本。',
            'action': 'Gate 5 前重新生成全期面板，或在文档中限定其适用范围。',
            'status': 'REGISTERED'})
    if checks['edge_members_not_generated']:
        gaps.append({'id': 'analyst_edge_members_absent',
                     'impact': '成员级（逐边 analyst 明细）不可追溯；可从覆盖表按 (i,j) 重算',
                     'action': '登记为可选审计缺口，不得声称成员级可追溯',
                     'status': 'REGISTERED'})
    gaps.append({'id': 'step11_not_in_runner',
                 'impact': 'runner 不调用 step11，firm_chars 的 analyst_cov 需手工回填',
                 'action': '记录于 run record；重跑 build_firm_chars 后必须执行 step11',
                 'status': 'REGISTERED'})

    result = {
        'spec_id': spec['spec_id'], 'spec_hash': SPEC_HASH,
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv), 'script': Path(__file__).name,
        'runner_steps': runner_files,
        'outside_runner': OUTSIDE_RUNNER,
        'artifacts': artifacts,
        'checks': checks,
        'registered_gaps': gaps,
    }
    result['status'] = 'PASS' if all([
        checks['runner_has_13_steps'], checks['runner_covers_base_chain'],
        checks['derived_full_period'], checks['firm_chars_has_analyst_cov'],
        checks['step11_writes_firm_chars']]) else 'FAIL'
    out = ROOT / 'reports/data_gates/G0/s3a_gate2_path_registry.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    print('gaps:', [g['id'] for g in gaps])
    print('status =', result['status'], '->', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
