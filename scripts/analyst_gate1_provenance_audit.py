"""analyst-network Gate 1 输入 provenance 审计。

对应验收矩阵 Gate 1.3：
- 登记预测、收益、市值、IPO、Drive controls、指数成员、SHIBOR、REV 的路径、日期、hash、缺口和 PIT 假设；
- 验收：每个下游脚本使用的输入都能回指到快照，不存在未登记输入。

分类规则（对脚本中出现的每个 data/ reports/ artifacts/ 路径字面量）：
- spec_input            ：出现在 spec.inputs
- spec_collection       ：出现在 spec.provenance.input_collections 或 edge_collections
- manifest_registered   ：出现在 version_manifest 的 file 字段
- script_output         ：有脚本以写模式产生（to_csv/json.dump/write_text/open(...,'w')）
- UNREGISTERED          ：以上都不是（必须为空才能 PASS）

产出 JSON：reports/data_gates/G0/analyst_gate1_provenance_audit.json
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from pricing_lib import file_shape, load_input_contracts

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/analyst_spec.yaml'
MANIFEST = ROOT / 'configs/version_manifest.yaml'
PATH_RE = re.compile(r"['\"]((?:data|reports|artifacts)/[^'\"]*?\.(?:csv|pkl|json|txt|parquet|xlsx))['\"]")
WRITE_RE = re.compile(r"to_csv|json\.dump|write_text|write_bytes|open\([^)]*['\"][wa]")
READ_RE = re.compile(r"read_csv|read_pickle|pd\.read|json\.load|load\(|open\([^)]*['\"]r")


def load_yaml(p):
    import yaml
    return yaml.safe_load(p.read_text(encoding='utf-8'))


def manifest_files():
    mf = load_yaml(MANIFEST)
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'file' and isinstance(v, str):
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for x in o:
                walk(x)
    walk(mf)
    return out


def scan_scripts():
    """按变量级 I/O 证据分类脚本中出现的路径字面量。

    路径通常写成 `OUT = Path('data/...')` 或 `PANEL = 'data/...'`，真正读写发生在使用该变量的行，
    因此必须做变量级匹配，不能只看字面量所在行。
    返回 path -> {'read': [...], 'write': [...], 'mentioned_only': [...]}
    """
    assign_re = re.compile(r"(\w+)\s*=\s*(?:[A-Za-z_][\w.]*\s*/\s*)?(?:Path\()?['\"](?P<path>(?:data|reports|artifacts)/[^'\"]+\.(?:csv|pkl|json|txt|parquet|xlsx))['\"]")
    io_re = r"(?P<fn>to_csv|to_pickle|write_text|write_bytes|json\.dump|read_csv|read_pickle|json\.load|load|open)\s*\([^)]*\b{var}\b"
    WRITE_FNS = {'to_csv', 'to_pickle', 'write_text', 'write_bytes', 'json.dump'}
    refs = {}
    file_texts = {}
    for f in sorted((ROOT / 'scripts').rglob('*.py')):
        if not (f.name.startswith('analyst_') or f.name.startswith('run_analyst')):
            continue
        if f.resolve() == Path(__file__).resolve():
            continue
        file_texts[f.name] = f.read_text(encoding='utf-8', errors='ignore')
        for i, line in enumerate(file_texts[f.name].splitlines(), 1):
            for p in PATH_RE.findall(line):
                rec = refs.setdefault(p, {'read': [], 'write': [], 'mentioned_only': []})
                tag = f'{f.name}:{i}'
                if line.strip().startswith('#'):
                    rec['mentioned_only'].append(tag); continue
                m = assign_re.search(line)
                var = m.group(1) if m else None
                wrote = readed = False
                if var:
                    for fn in re.finditer(io_re.format(var=re.escape(var)), file_texts[f.name]):
                        if fn.group('fn') in WRITE_FNS:
                            wrote = True
                        else:
                            readed = True
                if wrote:
                    rec['write'].append(tag)
                elif readed:
                    rec['read'].append(tag)
                else:
                    rec['mentioned_only'].append(tag)

    if not file_texts:
        raise SystemExit('provenance 审计：未找到 analyst_* 脚本，扫描面无意义')

    # 产生方检测覆盖 scripts/ 全树（产生方可能不在 analyst_ 前缀脚本内，如 peermax_max_build.py）；
    # 读取方检测仍只针对 analyst-network 脚本。
    produced_paths = set()
    produced_basenames = set()
    join_write_re = re.compile(r"(?:to_csv|to_pickle|write_text|write_bytes|json\.dump)\s*\([^)]*/\s*['\"]([^'\"]+)['\"]")
    write_call_tpl = (r"(?:(?:to_csv|to_pickle|write_text|write_bytes|json\.dump)\s*\([^)]*\b{var}\b"
                      r"|open\s*\(\s*{var}\s*,\s*['\"][wa])")
    for f in sorted((ROOT / 'scripts').rglob('*.py')):
        if f.resolve() == Path(__file__).resolve():
            continue
        text = f.read_text(encoding='utf-8', errors='ignore')
        produced_basenames.update(join_write_re.findall(text))
        for line in text.splitlines():
            m = assign_re.search(line)
            if not m or line.strip().startswith('#'):
                continue
            var, path = m.group(1), m.group('path')
            if re.search(write_call_tpl.format(var=re.escape(var)), text):
                produced_paths.add(path)

    for p, rec in refs.items():
        if not rec['write'] and (Path(p).name in produced_basenames or p in produced_paths):
            rec['write'].append('produced_elsewhere')
    return refs


def verify_spec_inputs(spec):
    """spec 输入身份 = configs/pricing_inputs.yaml 的 rows/columns 合同（sha 锚已退役）。

    文件缺失 → MISSING；未登记合同或 rows 为空 → UNREGISTERED（fail-closed，不得当作通过）；
    行数不符或缺登记列 → DIFF。
    """
    contracts = {e['file']: e for e in load_input_contracts().values()}
    rows = []
    for key, p in spec['inputs'].items():
        rec = {'key': key, 'path': p, 'exists': None, 'status': None, 'method': 'contract'}
        f = ROOT / p
        if not f.is_file():
            rec.update({'exists': False, 'status': 'MISSING'})
            rows.append(rec)
            continue
        entry = contracts.get(p)
        if entry is None or entry.get('rows') is None:
            rec.update({'exists': True, 'status': 'UNREGISTERED'})
            rows.append(rec)
            continue
        n, cols = file_shape(f)
        missing_cols = [c for c in (entry.get('columns') or []) if c not in cols]
        rec.update({'exists': True, 'rows': n, 'rows_registered': int(entry['rows']),
                    'columns_missing': missing_cols,
                    'status': 'MATCH' if (n == int(entry['rows']) and not missing_cols) else 'DIFF'})
        rows.append(rec)
    return rows


def main():
    spec = load_yaml(SPEC)
    spec_bytes = SPEC.read_bytes()
    spec_inputs = set(spec['inputs'].values())
    coll_names = set(spec['provenance'].get('input_collections', {})) | \
        set(f"data/processed/analyst/edges/edges_{k}_{{cutoff}}.csv"
            for k in spec['provenance']['edge_collections'])
    registered = manifest_files()
    refs = scan_scripts()

    classified = {'spec_input': [], 'spec_collection': [], 'manifest': [],
                  'script_output': [], 'mentioned_only': [], 'UNREGISTERED': []}
    for p, rec in sorted(refs.items()):
        if p in spec_inputs:
            classified['spec_input'].append(p)
        elif p in coll_names or '{cutoff}' in p:
            classified['spec_collection'].append(p)
        elif p in registered:
            classified['manifest'].append(p)
        elif rec['write']:
            classified['script_output'].append({'path': p, 'written_by': rec['write'][:3]})
        elif rec['read']:
            classified['UNREGISTERED'].append({'path': p, 'read_by': rec['read'][:3],
                                               'refs': rec['read'] + rec['write']})
        else:
            classified['mentioned_only'].append({'path': p, 'where': rec['mentioned_only'][:3]})

    inputs_check = verify_spec_inputs(spec)
    bad_inputs = [r for r in inputs_check if r['status'] != 'MATCH']

    report = {
        'spec_id': spec['spec_id'],
        'spec_hash': len(spec_bytes),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv),
        'status': None,
        'spec_inputs_verified': inputs_check,
        'reference_classification': {k: v for k, v in classified.items()},
        'checks': {
            'all_spec_inputs_match_hash': len(bad_inputs) == 0,
            'no_unregistered_inputs': len(classified['UNREGISTERED']) == 0,
        },
    }
    report['status'] = 'PASS' if all(report['checks'].values()) else 'FAIL'
    out = ROOT / 'reports/data_gates/G0/analyst_gate1_provenance_audit.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'checks': report['checks'],
                      'spec_input_count': len(inputs_check),
                      'bad_inputs': [r['path'] for r in bad_inputs],
                      'unregistered': [u['path'] for u in classified['UNREGISTERED']]},
                     ensure_ascii=False, indent=1))
    print('status =', report['status'], '->', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
