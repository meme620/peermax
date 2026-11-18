"""S3A Gate 5.6 收口三轮校验。

第一轮（数字与路径）：claim matrix 中的关键数字与产物逐项对账；
第二轮（版本与状态）：manifest/registry/spec/矩阵 四方一致，无 active/superseded 语义冲突；
第三轮（claim 与反例）：禁止表述扫描（无证据的"三人复现通过""全部机制完成""无条件超 Long"）。

产出：reports/data_gates/G0/s3a_gate5_closure_check.json
"""
import json
import re
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'configs/s3a_final_spec.yaml'
REG = ROOT / 'configs/parameter_registry.yaml'
MAN = ROOT / 'configs/version_manifest.yaml'
MATRIX = ROOT / 'reports/execution/s3a/s3a_final_acceptance_matrix.md'
CLAIM = ROOT / 'reports/execution/s3a/s3a_claim_evidence_matrix.md'
REPORT = ROOT / 'reports/execution/stage0/s3a_report.md'
PPTX_SCRIPT = ROOT / 'scripts/make_analyst_network_brief_pptx.py'


def load(p):
    return yaml.safe_load(p.read_text(encoding='utf-8'))


def round1_numbers():
    """claim matrix 关键数字 vs 产物逐项对账。"""
    rows = []

    def chk(name, product_path, key_path, expect, tol=0.0):
        d = json.loads((ROOT / product_path).read_text(encoding='utf-8'))
        cur = d
        for k in key_path:
            cur = cur[k]
        ok = abs(float(cur) - float(expect)) <= tol
        rows.append({'item': name, 'expect': expect, 'actual': cur,
                     'product': product_path, 'ok': ok})

    chk('9c_npm_beta', 'reports/data_gates/G0/s3a_9c_fm_regression.json',
        ['coefs', 'npm', 'beta'], 0.12418, 1e-9)
    chk('9c_npm_t_iid', 'reports/data_gates/G0/s3a_9c_fm_regression.json',
        ['coefs', 'npm', 't_iid'], 5.89, 1e-9)
    chk('9c_npm_t_nw1', 'reports/data_gates/G0/s3a_9c_fm_regression.json',
        ['coefs', 'npm', 't_nw1'], 5.94, 1e-9)
    chk('9c_extreme_beta', 'reports/data_gates/G0/s3a_9c_fm_regression_extreme.json',
        ['coefs', 'npm', 'beta'], 0.12767, 1e-9)
    chk('9h_cf_only', 'reports/data_gates/G0/s3a_9h_absorb.json',
        ['col1_cf_only', 'cf', 'beta'], 0.1242, 1e-9)
    chk('9h_ind_only', 'reports/data_gates/G0/s3a_9h_absorb.json',
        ['col2_ind_only', 'ind_ret', 'beta'], 0.0907, 1e-9)
    chk('9h_both_cf', 'reports/data_gates/G0/s3a_9h_absorb.json',
        ['col3_cf_ind', 'cf', 'beta'], 0.1259, 1e-9)
    chk('9h_both_ind', 'reports/data_gates/G0/s3a_9h_absorb.json',
        ['col3_cf_ind', 'ind_ret', 'beta'], 0.008, 1e-9)

    # 报告与 PPTX 脚本不得再出现旧值
    def strip_correction_blocks(text):
        """剔除「> **数字更正…」引用块：其中的旧值属留档，不是失实残留。"""
        out, skip = [], False
        for line in text.splitlines():
            if line.lstrip().startswith('> **数字更正'):
                skip = True
                continue
            if skip and line.lstrip().startswith('>'):
                continue
            skip = False
            out.append(line)
        return chr(10).join(out)

    rep = strip_correction_blocks(REPORT.read_text(encoding='utf-8'))
    sw = PPTX_SCRIPT.read_text(encoding='utf-8')
    stale = {
        'report_has_beta_0_131': 'β=+0.131' in rep or '**+0.131**' in rep,
        'report_claims_t_6_06_as_result': bool(re.search(r'仍显著（t=6\.06', rep)),
        'report_h_0_005': '0.091→0.005' in rep,
        'pptx_script_has_0_131': 'β=0.131' in sw,
        'pptx_script_has_t_6_06': 't=6.06' in sw,
    }
    rows.append({'item': 'report_pptx_stale_values', 'ok': not any(stale.values()),
                 'detail': stale})
    return rows


def round2_versions():
    spec = load(SPEC)
    man = load(MAN)
    reg = load(REG)
    checks = {}
    checks['spec_hash_recorded_in_matrix'] = sha256(SPEC.read_bytes()).hexdigest()[:8] in MATRIX.read_text(encoding='utf-8')
    checks['manifest_anchor_matches'] = \
        spec['governance']['manifest_sha256'] == sha256(MAN.read_bytes()).hexdigest()
    checks['registry_anchor_matches'] = \
        spec['governance']['registry_sha256'] == sha256(REG.read_bytes()).hexdigest()
    checks['claim_matrix_registered'] = any(
        x['file'] == 'reports/execution/s3a/s3a_claim_evidence_matrix.md'
        for x in man['deliverables'])
    checks['claim_matrix_exists'] = CLAIM.exists()

    # active/superseded：同一文件不得同时存在两条 active
    from collections import defaultdict
    by_file = defaultdict(list)
    for x in man['deliverables']:
        by_file[x['file']].append(x.get('status'))
    # 目录型条目（以 / 结尾）可因承载不同网络版本而有多条 active；
    # 文件型条目同一路径不得有两条 active。
    conflict = {f: st for f, st in by_file.items()
                if st.count('active') > 1 and not f.endswith('/')}
    dir_multi = {f: st for f, st in by_file.items()
                 if st.count('active') > 1 and f.endswith('/')}
    checks['no_duplicate_active'] = len(conflict) == 0
    # v2 文档已标 superseded
    v2 = {'reports/execution/stage0/s3a_run_record.md',
          'reports/execution/s3a/s3a_operational_report.md'}
    st = {x['file']: x.get('status') for x in man['deliverables']}
    checks['v2_docs_superseded'] = all(st.get(f) == 'superseded' for f in v2)
    return checks, conflict, dir_multi


def round3_claims():
    """禁止表述扫描。"""
    targets = {
        's3a_report.md': REPORT,
        'claim_matrix': CLAIM,
        'matrix': MATRIX,
    }
    banned = {
        '三人独立复现通过': r'三人(独立)?复现(已)?通过',
        '全部机制已完成': r'全部?机制(均)?已完成',
        '完整复现AH/Long': r'完整复现\s*(AH|Long)',
    }
    out = {}
    for name, p in targets.items():
        t = p.read_text(encoding='utf-8')
        hits = []
        for label, pat in banned.items():
            for m in re.finditer(pat, t):
                seg = t[max(0, m.start() - 60):m.end() + 60].replace('\n', ' ')
                if any(x in seg for x in ('不得', '禁止', 'NOT_RUN', '未做', '未执行')):
                    continue
                hits.append({'rule': label, 'context': seg[:140]})
        out[name] = hits
    return out


def main():
    spec = load(SPEC)
    r1 = round1_numbers()
    r2, conflict, dir_multi = round2_versions()
    r3 = round3_claims()

    checks = {
        'round1_all_numbers_match': all(x['ok'] for x in r1),
        'round2_all_checks': all(r2.values()),
        'round3_no_banned_claims': all(len(v) == 0 for v in r3.values()),
    }
    result = {
        'spec_id': spec['spec_id'], 'spec_hash': sha256(SPEC.read_bytes()).hexdigest(),
        'run_at': datetime.now(timezone.utc).isoformat(),
        'command': ' '.join(sys.argv), 'script': Path(__file__).name,
        'round1_numbers': r1, 'round2_versions': r2, 'round2_conflicts': conflict, 'round2_dir_multi_active': dir_multi,
        'round3_banned_claims': r3, 'checks': checks,
        'status': 'PASS' if all(checks.values()) else 'FAIL',
    }
    out = ROOT / 'reports/data_gates/G0/s3a_gate5_closure_check.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(checks, ensure_ascii=False, indent=1))
    for x in r1:
        if not x['ok']:
            print('  R1 FAIL:', x)
    for k, v in r2.items():
        if not v:
            print('  R2 FAIL:', k)
    for k, v in r3.items():
        if v:
            print('  R3 FAIL:', k, v)
    print('status =', result['status'], '->', out.relative_to(ROOT))


if __name__ == '__main__':
    main()
