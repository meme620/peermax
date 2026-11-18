"""S3A Gate 1 输出元数据审计（Gate 1.4）。

检查两件事：
1) 脚本侧：CF/NPM 主脚本与 9A–9M 机制脚本是否在写出前注入统一元数据；
2) 产物侧：磁盘上已有 JSON 是否带元数据；未带的登记为「待 Gate 2/3/4 重跑再生」，
   不作为 Gate 1 通过条件（Gate 1 只要求脚本具备能力，并如实登记陈旧产物）。

产出 JSON：reports/data_gates/G0/s3a_gate1_metadata_audit.json
"""
import json
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/s3a_final_spec.yaml'
REQ = ['spec_id', 'spec_hash', 'input_hashes', 'network_version', 'run_at', 'command', 'status']

MAIN_SCRIPTS = ['analyst_cf_mom_replication.py', 'analyst_npm_replication.py']
MECH_SCRIPTS = [
    ('analyst_industry_contrast.py', 'reports/data_gates/G0/s3a_9b_industry_contrast.json'),
    ('analyst_fm_regression.py', 'reports/data_gates/G0/s3a_9c_fm_regression.json'),
    ('analyst_subsample.py', 'reports/data_gates/G0/s3a_9e_subsample.json'),
    ('analyst_cffr.py', 'reports/data_gates/G0/s3a_9g_cffr.json'),
    ('analyst_industry_absorb.py', 'reports/data_gates/G0/s3a_9h_absorb.json'),
    ('analyst_decile_cumret.py', 'reports/data_gates/G0/s3a_9i_bivariate_decile_cumret.json'),
    ('analyst_extra_tests.py', 'reports/data_gates/G0/s3a_9j_extra.json'),
    ('analyst_nofy_compare.py', 'reports/data_gates/G0/s3a_9k_nofy_compare.json'),
    ('analyst_cfret_second_order.py', 'reports/data_gates/G0/analyst_cfret_second_order.json'),
]


def script_side():
    """脚本侧判定：attach_meta 统一机制或内联字面字段（两者产出同一 schema）。"""
    def uses_attach_meta(txt):
        return ('attach_meta(' in txt) and ('from analyst_meta import attach_meta' in txt)

    rows = []
    for name in MAIN_SCRIPTS:
        txt = (ROOT / 'scripts' / name).read_text(encoding='utf-8')
        rows.append({'script': f'scripts/{name}', 'kind': 'main',
                     'emits_metadata': uses_attach_meta(txt) or all(k in txt for k in REQ),
                     'mechanism': 'attach_meta' if uses_attach_meta(txt) else 'inline_literals'})
    for name, _ in MECH_SCRIPTS:
        txt = (ROOT / 'scripts' / name).read_text(encoding='utf-8')
        rows.append({'script': f'scripts/{name}', 'kind': 'mechanism',
                     'emits_metadata': uses_attach_meta(txt) or all(k in txt for k in REQ),
                     'mechanism': 'attach_meta' if uses_attach_meta(txt) else 'inline_literals'})
    return rows


def artifact_side():
    rows = []
    for _, rel in MECH_SCRIPTS:
        p = ROOT / rel
        rec = {'artifact': rel, 'exists': p.exists()}
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding='utf-8'))
                rec['has_metadata'] = all(k in d for k in REQ)
                rec['missing'] = [k for k in REQ if k not in d]
                rec['spec_hash'] = d.get('spec_hash')
            except Exception as e:
                rec.update({'has_metadata': False, 'error': str(e)})
        rows.append(rec)
    # 主复刻产物
    for rel in ['data/processed/analyst/cf_mom_replication.json',
                'data/processed/analyst/npm_replication.json']:
        p = ROOT / rel
        rec = {'artifact': rel, 'exists': p.exists()}
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            rec['has_metadata'] = all(k in d for k in REQ)
            rec['missing'] = [k for k in REQ if k not in d]
            rec['spec_hash'] = d.get('spec_hash')
        rows.append(rec)
    return rows


def main():
    spec_bytes = SPEC.read_bytes()
    import yaml
    spec = yaml.safe_load(spec_bytes.decode('utf-8'))
    scripts = script_side()
    artifacts = artifact_side()
    require_fields = spec.get('provenance', {}).get('output_metadata_required', REQ)
    report = {
        'spec_id': spec['spec_id'],
        'spec_hash': sha256(spec_bytes).hexdigest(),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'status': None,
        'required_fields': require_fields,
        'scripts': scripts,
        'artifacts': artifacts,
        'checks': {
            'all_scripts_emit_metadata': all(r['emits_metadata'] for r in scripts),
        },
        'stale_artifacts': [r['artifact'] for r in artifacts
                            if r.get('exists') and not r.get('has_metadata')],
        'note': ('Gate 1 验收对象是脚本具备元数据能力；磁盘上未带元数据的产物属于旧版，'
                 '须在 Gate 2/3（主复刻）与 Gate 4（9A–9M 重建）重跑再生。'),
    }
    report['status'] = 'PASS' if report['checks']['all_scripts_emit_metadata'] else 'FAIL'
    out = ROOT / 'reports/data_gates/G0/s3a_gate1_metadata_audit.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'checks': report['checks'],
                      'scripts_missing': [r['script'] for r in scripts if not r['emits_metadata']],
                      'artifacts_with_metadata': [r['artifact'] for r in artifacts if r.get('has_metadata')],
                      'stale_artifacts': report['stale_artifacts']},
                     ensure_ascii=False, indent=1))
    print('status =', report['status'], '->', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
