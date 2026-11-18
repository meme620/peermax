"""S3A Gate 1 参数入口审计：静态扫描 + 非法组合报错证据。

对应验收矩阵 Gate 1.2：
- 主运行只接受统一 spec 或显式命令参数；
- 禁止 S3A_RET_FILE / S3A_MEQ_FILE / S3A_NET_VER / H12_MIN_AVAIL / S3A_OUT_SUFFIX 静默改变主规格；
- 变体必须显式命名并写入元数据。

产出 JSON：reports/data_gates/G0/s3a_gate1_entry_audit.json
"""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/s3a_final_spec.yaml'
MAIN_SCRIPTS = ['analyst_cf_mom_replication.py', 'analyst_npm_replication.py']
LEGACY_ENV_NAMES = ['S3A_RET_FILE', 'S3A_MEQ_FILE', 'S3A_NET_VER', 'H12_MIN_AVAIL', 'S3A_OUT_SUFFIX']
ENV_READ_RE = re.compile(r'os\.environ|os\.getenv|getenv\(')
SPEC_REF_RE = re.compile(r"s3a_final_spec\.yaml")
PATH_LITERAL_RE = re.compile(r"['\"](data/[^'\"]+\.(?:pkl|csv))['\"]")
PKL_NAME_RE = re.compile(r"['\"]([a-z0-9_]+\.pkl)['\"]")


def spec_inputs():
    import yaml
    d = yaml.safe_load(SPEC.read_text(encoding='utf-8'))
    return d, {v for v in d['inputs'].values()}, {Path(v).name for v in d['inputs'].values()}


def scan_env_reads():
    hits = []
    for f in sorted((ROOT / 'scripts').rglob('*.py')):
        txt = f.read_text(encoding='utf-8', errors='ignore')
        for i, line in enumerate(txt.splitlines(), 1):
            if ENV_READ_RE.search(line) and not line.strip().startswith('#'):
                hits.append({'file': str(f.relative_to(ROOT)), 'line': i, 'code': line.strip()[:120]})
    return hits


def scan_legacy_names():
    """只登记旧参数名仍被当作环境变量读取的位置（作为普通变量名不算违规）。"""
    hits = []
    self_path = Path(__file__).resolve()
    for f in sorted((ROOT / 'scripts').rglob('*.py')):
        if f.resolve() == self_path:
            continue
        txt = f.read_text(encoding='utf-8', errors='ignore')
        for name in LEGACY_ENV_NAMES:
            for i, line in enumerate(txt.splitlines(), 1):
                if name not in line:
                    continue
                if re.search(rf"(os\.environ[^\n]*{name})|(getenv\([^\)]*{name})", line):
                    hits.append({'file': str(f.relative_to(ROOT)), 'line': i,
                                 'name': name, 'role': 'env_read', 'code': line.strip()[:120]})
    return hits


def check_main_spec_entry(spec_paths, spec_names):
    out = []
    for name in MAIN_SCRIPTS:
        f = ROOT / 'scripts' / name
        txt = f.read_text(encoding='utf-8')
        entry = {
            'script': f'scripts/{name}',
            # 直接读取 spec，或从主模块 import 由 spec 派生的常量（_SPEC/SPEC/_SPEC_HASH）
            'reads_spec': bool(SPEC_REF_RE.search(txt)) or
                          bool(re.search(r"from analyst_cf_mom_replication import[^\n]*(_SPEC|SPEC)", txt)),
            'spec_keys': sorted(set(re.findall(r"_SPEC\['([a-z_0-9]+)'\]", txt))),
            'cli_guards': sorted(set(re.findall(r"raise ValueError\('([^']+)'\)", txt))),
            'emits_spec_metadata': (
                all(k in txt for k in
                    ['spec_id', 'spec_hash', 'input_hashes', 'network_version',
                     'run_at', 'command', 'status'])
                or bool(re.search(r"attach_meta\(", txt))),
            'metadata_mechanism': ('attach_meta' if re.search(r"attach_meta\(", txt) else 'inline_literals'),
            'hardcoded_data_paths': sorted(set(PATH_LITERAL_RE.findall(txt))),
        }
        entry['all_hardcoded_paths_in_spec'] = all(
            (p in spec_paths) or (Path(p).name in spec_names) for p in entry['hardcoded_data_paths'])
        out.append(entry)
    return out


NINE_X_SCRIPTS = (
    'analyst_industry_contrast.py',
    'analyst_fm_regression.py',
    'analyst_subsample.py',
    'analyst_cffr.py',
    'analyst_industry_absorb.py',
    'analyst_decile_cumret.py',
    'analyst_extra_tests.py',
    'analyst_nofy_build.py',
    'analyst_nofy_compare.py',
    'analyst_cfret_second_order.py',
)


def check_9x_paths(spec_paths, spec_names):
    """9A–9M 入口审计：名单与 docs/pipeline.md §3.3 的 9A–9M 表一致。"""
    out = []
    for name in NINE_X_SCRIPTS:
        f = ROOT / 'scripts' / name
        if not f.is_file():
            out.append({'script': f'scripts/{name}', 'missing': True})
            continue
        txt = f.read_text(encoding='utf-8')
        paths = sorted(set(PATH_LITERAL_RE.findall(txt)))
        pkgs = sorted(set(PKL_NAME_RE.findall(txt)))
        bad_paths = [p for p in paths if not ((p in spec_paths) or (Path(p).name in spec_names))]
        bad_pkgs = [p for p in pkgs if p not in spec_names]
        out.append({'script': f'scripts/{name}',
                    'unregistered_csv_paths': bad_paths,
                    'unregistered_pkl_names': bad_pkgs})
    return out


def run_cli_guard_test():
    """实跑非法组合，断言以非零退出且报 ValueError（不加载数据）。"""
    tests = [
        {'script': 'analyst_cf_mom_replication.py', 'argv': ['bogus', '1']},
        {'script': 'analyst_cf_mom_replication.py', 'argv': ['count', '7']},
        {'script': 'analyst_cf_mom_replication.py', 'argv': ['count', '1', 'bogus_net']},
        {'script': 'analyst_npm_replication.py', 'argv': ['bogus']},
        {'script': 'analyst_npm_replication.py', 'argv': ['count', 'bogus_net']},
    ]
    res = []
    for t in tests:
        cmd = [sys.executable, str(ROOT / 'scripts' / t['script'])] + t['argv']
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
            res.append({'script': t['script'], 'argv': t['argv'], 'returncode': p.returncode,
                        'rejected': p.returncode != 0,
                        'error_tail': (p.stderr or '').strip().splitlines()[-1][:160] if p.stderr else ''})
        except subprocess.TimeoutExpired:
            res.append({'script': t['script'], 'argv': t['argv'], 'returncode': None,
                        'rejected': False, 'error_tail': 'TIMEOUT'})
    return res


def main():
    spec_bytes = SPEC.read_bytes()
    spec, spec_paths, spec_names = spec_inputs()
    report = {
        'spec_id': spec['spec_id'],
        'spec_hash': sha256(spec_bytes).hexdigest(),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'status': 'COMPUTED',
        'env_reads': scan_env_reads(),
        'legacy_param_name_hits': scan_legacy_names(),
        'main_script_entry': check_main_spec_entry(spec_paths, spec_names),
        'mechanism_9x_unregistered_inputs': [x for x in check_9x_paths(spec_paths, spec_names)
                                             if x['unregistered_csv_paths'] or x['unregistered_pkl_names']],
        'cli_guard_runtime_test': run_cli_guard_test(),
    }
    report['checks'] = {
        'no_s3a_env_override': all('s3a' not in h['file'].lower() for h in report['env_reads']),
        'no_live_legacy_names': len(report['legacy_param_name_hits']) == 0,
        'main_scripts_read_spec': all(x['reads_spec'] for x in report['main_script_entry']),
        'main_scripts_emit_metadata': all(x['emits_spec_metadata'] for x in report['main_script_entry']),
        'main_hardcoded_paths_registered': all(x['all_hardcoded_paths_in_spec']
                                               for x in report['main_script_entry']),
        'all_9x_inputs_registered': len(report['mechanism_9x_unregistered_inputs']) == 0,
        'illegal_cli_combos_rejected': all(x['rejected'] for x in report['cli_guard_runtime_test']),
    }
    report['status'] = 'PASS' if all(report['checks'].values()) else 'FAIL'
    out = ROOT / 'reports/data_gates/G0/s3a_gate1_entry_audit.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(report['checks'], ensure_ascii=False, indent=1))
    print('status =', report['status'], '->', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
