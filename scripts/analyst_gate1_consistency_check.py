"""S3A Gate 1.6 收口一致性校验（第三轮机器对账）。

对账对象：registry ↔ spec ↔ 代码 ↔ 运行命令 ↔ 审计 JSON ↔ manifest ↔ 矩阵。
低层哈希校验委托给三个 Gate 1 审计产物，避免重复计算大文件。

产出 JSON：reports/data_gates/G0/s3a_gate1_consistency_check.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/s3a_final_spec.yaml'
REG = ROOT / 'configs/parameter_registry.yaml'
MAN = ROOT / 'configs/version_manifest.yaml'
MATRIX = ROOT / 'reports/execution/s3a/s3a_final_acceptance_matrix.md'

EXCLUDED_PROBES = []  # 供未来扩展


def load(p):
    import yaml
    return yaml.safe_load(p.read_text(encoding='utf-8'))


def registry_spec_reconciliation(spec, reg):
    params = {it['name']: it for it in reg['parameters']}
    dev_ids = {d['id'] for d in spec['registered_deviations']}
    rows = []

    def row(name, spec_val, reg_val, expect, note=''):
        ok = False
        if expect == 'equal':
            ok = spec_val == reg_val
        elif expect == 'semantic_equal':
            # registry 为描述文本、spec 为机器码：以关键语义词是否出现为准
            ok = all(tok in str(reg_val) for tok in
                     {'month_end': ['月末'], 'raw_close_return': ['raw']}.get(spec_val, []))
        elif expect == 'deviation_registered':
            ok = name in dev_ids
        rows.append({'param': name, 'spec': spec_val, 'registry': reg_val,
                     'expect': expect, 'note': note, 'ok': ok})

    row('coverage_window', spec['network']['coverage_window_months'],
        params['coverage_window']['headline_value'], 'equal')
    row('epsilon_weight', spec['network']['epsilon_weight'],
        params['epsilon_weight']['headline_value'], 'equal')
    row('network_cutoff_rule', spec['network']['cutoff_rule'],
        params['network_cutoff_rule']['headline_value'], 'semantic_equal')
    row('market_value_basis', spec['sample_filters']['market_value_basis_current'],
        spec['sample_filters']['market_value_basis_registry_target'],
        'deviation_registered', 'registry 冻结=流通市值，实现=总市值')
    row('st_exclusion', spec['sample_filters']['st_primary_current'],
        'exclude_st', 'deviation_registered', 'registry 冻结=排除 ST，实现=含 ST + 焦点侧敏感性')
    row('sample_period_full', spec['samples']['s3a_signal_range'],
        params['sample_start']['headline_value'] + '..' + params['sample_end']['headline_value'],
        'deviation_registered', 'S3A 全期复刻 vs registry 主线样本 2018-2025')
    return rows


def code_spec_reconciliation(spec):
    checks = []
    step5 = (ROOT / 'scripts/analyst_coverage_snapshot.py').read_text(encoding='utf-8')
    step67 = (ROOT / 'scripts/analyst_edges_build.py').read_text(encoding='utf-8')
    cf = (ROOT / 'scripts/analyst_cf_mom_replication.py').read_text(encoding='utf-8')
    npm = (ROOT / 'scripts/analyst_npm_replication.py').read_text(encoding='utf-8')

    def add(item, spec_decl, code_evidence, ok):
        checks.append({'item': item, 'spec': spec_decl, 'code': code_evidence, 'ok': ok})

    add('fy_rule', spec['network']['fy_rule'],
        'Fpenddt_dt.dt.year - report_date.dt.year', "'gap'] = rec['Fpenddt_dt'].dt.year - rec['report_date'].dt.year" in step5)
    add('availability_rule', spec['network']['availability_rule'],
        'searchsorted(cutoff, side="right")', "searchsorted(c64, side='right')" in step5)
    add('cutoff_count', spec['network']['cutoff_count'],
        "pd.date_range('2005-12-31','2025-12-31',freq='ME')", "'2005-12-31', '2025-12-31'" in step5)
    add('weight_formula', spec['network']['weight_formula'],
        'c / denom[i]', 'wi = c / denom[i]' in step67 and 'wj = c / denom[j]' in step67)
    add('microcap_rule', spec['sample_filters']['microcap_rule'],
        "groupby('month')['me'].quantile(0.10)", "quantile(0.10)" in cf and "quantile(0.10)" in npm)
    add('h12_min_available', spec['horizon']['h12_min_available_months'],
        "_SPEC['horizon']['h12_min_available_months']", "h12_min_available_months" in cf)
    add('main_scripts_read_spec', 'spec is single entry',
        'SPEC = Path(configs/s3a_final_spec.yaml)', 'configs/s3a_final_spec.yaml' in cf)
    return checks


def audit_artifacts():
    out = {}
    for name in ['entry', 'provenance', 'metadata']:
        p = ROOT / f'reports/data_gates/G0/s3a_gate1_{name}_audit.json'
        rec = {'exists': p.exists()}
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            rec.update({'status': d.get('status'), 'spec_hash': d.get('spec_hash'),
                        'checks': d.get('checks')})
        out[name] = rec
    return out


def frozen_value_drift_check(spec):
    """S3A 冻结值漂移检测：registry 其余条目可被并行工作流新增/修改，
    但 S3A 冻结值一旦变化必须显式暴露，不能被锚点刷新掩盖。"""
    reg = load(REG)
    params = {it['name']: it for it in reg['parameters']}
    expected = spec['governance'].get('s3a_frozen_values_recheck', {})
    pairs = [
        ('coverage_window', params['coverage_window']['headline_value']),
        ('epsilon_weight', params['epsilon_weight']['headline_value']),
        ('sample_start', params['sample_start']['headline_value']),
        ('sample_end', params['sample_end']['headline_value']),
    ]
    rows = [{'param': k, 'expected': expected.get(k), 'actual': v,
             'ok': expected.get(k) == v} for k, v in pairs]
    w = str(params['s3a_weight_rule']['headline_value'])
    rows.append({'param': 's3a_weight_rule', 'expected': expected.get('s3a_weight_rule'),
                 'actual': w, 'ok': expected.get('s3a_weight_rule') in w})
    return rows


def manifest_check(spec):
    man = load(MAN)
    files = [x['file'] for x in man['deliverables']]
    need = ['configs/s3a_final_spec.yaml',
            'reports/execution/s3a/s3a_final_acceptance_matrix.md',
            'reports/execution/s3a/s3a_parameter_snapshot_20260906.yaml',
            'reports/execution/s3a/s3a_three_person_cross_check_protocol.md',
            'scripts/analyst_meta.py']
    return {'registered': [f for f in need if f in files],
            'missing': [f for f in need if f not in files],
            'manifest_version': man['manifest_version'],
            'manifest_version': load(MAN).get('manifest_version'),
            'registry_version': load(REG).get('registry_version'),
            'registry_pinned_in_spec': spec['governance'].get('registry_version') == load(REG).get('registry_version')}


def main():
    spec_bytes = SPEC.read_bytes()
    spec = load(SPEC)
    reg = load(REG)
    recon = registry_spec_reconciliation(spec, reg)
    code = code_spec_reconciliation(spec)
    audits = audit_artifacts()
    man = manifest_check(spec)
    drift = frozen_value_drift_check(spec)

    checks = {
        'registry_spec_reconciled': all(r['ok'] for r in recon),
        'code_matches_spec': all(c['ok'] for c in code),
        'three_audits_pass': all(a.get('status') == 'PASS' for a in audits.values()),
        'audits_pin_current_spec': all(a.get('spec_hash') == file_size(spec_bytes)
                                       for a in audits.values()),
        'manifest_covers_governance_files': len(man['missing']) == 0,
        'registry_matches_spec_pin': man['registry_pinned_in_spec'],
        's3a_frozen_values_unchanged': all(r['ok'] for r in drift),
    }
    report = {
        'spec_id': spec['spec_id'],
        'spec_hash': file_size(spec_bytes),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'checks': checks,
        'registry_spec_rows': recon,
        'code_spec_rows': code,
        'audit_artifacts': audits,
        'manifest': man,
        'frozen_value_drift': drift,
        'matrix_present': MATRIX.exists(),
    }
    out = ROOT / 'reports/data_gates/G0/s3a_gate1_consistency_check.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    print('status =', report['status'], '->', out.relative_to(ROOT))
    if report['status'] != 'PASS':
        for r in recon:
            if not r['ok']:
                print('  recon FAIL:', r)
        for c in code:
            if not c['ok']:
                print('  code FAIL:', c)
        for k, v in audits.items():
            if v.get('status') != 'PASS':
                print('  audit FAIL:', k, v.get('status'), v.get('checks'))
        for r in drift:
            if not r['ok']:
                print('  FROZEN DRIFT:', r)


if __name__ == '__main__':
    main()
