"""s3_m1_lib 的 contracts 实现（逐字搬移，逻辑未改）。"""

from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))


from lib.m1.data import sha256_file
from lib.m1.paths import HARD_GATE_KEYS
from lib.m1.paths import M1_INPUTS
from lib.m1.paths import MANIFEST
from lib.m1.paths import REGISTRY
from lib.m1.paths import ROOT
from lib.m1.paths import ROWS_NOT_APPLICABLE
from lib.m1.paths import SPEC
from lib.m1.paths import _M1_RETURN_COLS
from lib.m1.paths import _MONTH_RE
from lib.m1.paths import _SAMPLE_EXCLUSION_VALUE_COL
from lib.m1.paths import _SHA256_HEX
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys

def _registry_fail(msg: str):
    print(f'[REGISTRY] {msg}', file=sys.stderr, flush=True)
    raise SystemExit(2)

def _read_registry_doc(registry_path=None) -> dict:
    """只读 registry YAML；不做任何写回。"""
    import yaml
    p = Path(registry_path) if registry_path is not None else REGISTRY
    if not p.is_file():
        _registry_fail(f'registry 不存在: {p}')
    doc = yaml.safe_load(p.read_text(encoding='utf-8'))
    if not isinstance(doc, dict) or not isinstance(doc.get('parameters'), list):
        _registry_fail(f'registry 结构非法（缺 parameters 列表）: {p}')
    return doc

def _registry_param(name: str, registry_path=None):
    """按 name 取唯一参数的 headline_value；命中数 != 1 / 缺 headline_value → SystemExit(2)。"""
    plist = _read_registry_doc(registry_path)['parameters']
    hits = [p for p in plist if isinstance(p, dict) and p.get('name') == name]
    if len(hits) != 1:
        _registry_fail(f'参数 {name!r} 命中 {len(hits)} 条 != 1；拒绝猜测/回退')
    if 'headline_value' not in hits[0]:
        _registry_fail(f'参数 {name!r} 缺 headline_value')
    return hits[0]['headline_value']

def _registry_month(name: str, registry_path=None) -> str:
    raw = str(_registry_param(name, registry_path)).strip()
    if not _MONTH_RE.fullmatch(raw[:7]) or (len(raw) > 7 and raw[7] not in '-/'):
        _registry_fail(f'{name} 不是 YYYY-MM[-DD] 格式: {raw!r}')
    return raw[:7]

def sample_window_bounds(registry_path=None) -> tuple:
    """registry sample_start/sample_end -> ('YYYY-MM', 'YYYY-MM')；start<=end，否则 SystemExit(2)。"""
    start = _registry_month('sample_start', registry_path)
    end = _registry_month('sample_end', registry_path)
    if start > end:
        _registry_fail(f'sample_start {start} > sample_end {end}')
    return start, end

def load_registry(registry_path=None) -> dict:
    return _read_registry_doc(registry_path)

def params(registry_path=None) -> dict:
    reg = load_registry(registry_path)
    return {p['name']: p for p in reg['parameters']}

def param_value(name: str, registry_path=None):
    """registry headline_value；registry_path 仅测试注入，生产用 configs 真源。"""
    return _registry_param(name, registry_path)

def manifest_input_key(path) -> str:
    """manifest `file` 键归一化：ROOT 下用相对 posix 路径，ROOT 外用绝对 posix 路径。"""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()

def load_manifest_entries(manifest_path=None) -> list:
    """读取 manifest deliverable 条目（只读；不做任何写回）。"""
    import yaml
    p = Path(manifest_path) if manifest_path is not None else MANIFEST
    require(p.is_file(), f'M1 输入锚: manifest 不存在: {p}')
    doc = yaml.safe_load(p.read_text(encoding='utf-8'))
    entries = doc.get('deliverables') if isinstance(doc, dict) else None
    require(isinstance(entries, list) and bool(entries),
            f'M1 输入锚: manifest 无可读 deliverables: {p}')
    return entries

def _manifest_optional_anchor(entries: list, key: str, where: str):
    """manifest 条目可选的 sha256：无条目/无 sha → None；重复/非 active/非 64-hex → 退出。"""
    hits = [e for e in entries
            if str(e.get('file', '')).replace('\\', '/').strip('/') == key]
    if not hits:
        return None
    require(len(hits) == 1,
            f'{where}: manifest 条目数 {len(hits)} != 1（file={key}）；多条目拒绝')
    entry = hits[0]
    status = str(entry.get('status', '')).strip()
    require(status == 'active',
            f'{where}: manifest 条目 status={status!r} != active（file={key}）')
    raw = str(entry.get('sha256') or '').strip().lower()
    if not raw:
        return None
    require(bool(_SHA256_HEX.fullmatch(raw)),
            f'{where}: manifest 条目 sha256 非法（file={key}，当前={raw!r}）')
    return raw

def _manifest_input_anchor(entries: list, key: str, where: str) -> str:
    """取单个 active 条目的 sha256；缺失/重复/非 active/非 64-hex 全部 SystemExit(2)。"""
    hits = [e for e in entries
            if str(e.get('file', '')).replace('\\', '/').strip('/') == key]
    require(len(hits) == 1,
            f'{where}: manifest 条目数 {len(hits)} != 1（file={key}）；缺条目或多条目均拒绝，'
            '不得回退实测值')
    entry = hits[0]
    status = str(entry.get('status', '')).strip()
    require(status == 'active',
            f'{where}: manifest 条目 status={status!r} != active（file={key}）')
    sha = str(entry.get('sha256') or '').strip().lower()
    require(bool(_SHA256_HEX.fullmatch(sha)),
            f'{where}: manifest 条目缺合法 sha256（file={key}，当前={entry.get("sha256")!r}）；'
            '拒绝回退实测值或自动刷新')
    return sha

def manifest_anchor_sha(relpath, entries=None, manifest_path=None):
    """只读 manifest 取可选锚（无条目/无 sha → None）。"""
    entries = load_manifest_entries(manifest_path) if entries is None else entries
    key = str(relpath).replace('\\', '/').strip('/')
    return _manifest_optional_anchor(entries, key, f'manifest_anchor_sha({key})')

def load_spec_input_hashes(spec_path=None) -> dict:
    """读取 spec `provenance.input_hashes`（只读；键归一化为 posix 相对路径）。"""
    import yaml
    p = Path(spec_path) if spec_path is not None else SPEC
    require(p.is_file(), f'spec 输入锚: spec 不存在: {p}')
    doc = yaml.safe_load(p.read_text(encoding='utf-8'))
    prov = doc.get('provenance') if isinstance(doc, dict) else None
    hashes = prov.get('input_hashes') if isinstance(prov, dict) else None
    require(isinstance(hashes, dict) and bool(hashes),
            f'spec 输入锚: 无 provenance.input_hashes: {p}')
    return {str(k).replace('\\', '/').strip('/'): str(v).strip().lower()
            for k, v in hashes.items()}

def anchor_sha(relpath, entries=None, spec_hashes=None, manifest_path=None,
               spec_path=None):
    """manifest / spec 双源锚：返回 (sha256, source)。

    - manifest 有 active + sha，或 spec provenance.input_hashes 有值，至少一个；
    - 两源同时存在且不一致 → SystemExit(2)（不静默择一）；
    - 两源都无 → SystemExit(2)（不回退实测值/不写回）。
    """
    key = str(relpath).replace('\\', '/').strip('/')
    entries = load_manifest_entries(manifest_path) if entries is None else entries
    spec_hashes = load_spec_input_hashes(spec_path) if spec_hashes is None else spec_hashes
    man = _manifest_optional_anchor(entries, key, f'anchor_sha({key})')
    spec = spec_hashes.get(key)
    if spec is not None:
        spec = str(spec).strip().lower()
        require(bool(_SHA256_HEX.fullmatch(spec)),
                f'anchor_sha: spec 锚非法（file={key}，当前={spec!r}）')
    require(man is not None or spec is not None,
            f'anchor_sha: 无锚（file={key}）；manifest/spec 均未登记 sha256，'
            '拒绝回退实测值')
    if man is not None and spec is not None:
        require(man == spec,
                f'anchor_sha: manifest/spec 锚冲突（file={key}）manifest={man} spec={spec}')
    return (man, 'manifest') if man is not None else (spec, 'spec')

def assert_anchored_file_sha(path, manifest_key=None, manifest_path=None,
                             spec_path=None) -> dict:
    """按 manifest/spec 双源锚校验单个文件；失败 SystemExit(2)，信息含期望/实际。"""
    p = Path(path)
    require(p.is_file(), f'输入 sha 锚: 文件不存在: {p}')
    key = str(manifest_key) if manifest_key else manifest_input_key(p)
    expected, source = anchor_sha(key, manifest_path=manifest_path, spec_path=spec_path)
    actual = sha256_file(p)
    require(actual == expected,
            f'输入 sha 不匹配（file={key}，锚来源={source}）期望={expected} 实际={actual}')
    print(f'[INPUT-SHA] {key}: MATCH {actual[:12]}…（{source}）', flush=True)
    return {'file': key, 'path': str(p), 'sha256': actual,
            'expected_sha256': expected, 'actual_sha256': actual,
            'anchor_source': source, 'match': True}

def assert_named_m1_input_sha(name: str, path=None, manifest_path=None) -> dict:
    """校验指定逻辑 M1 输入的实际读取路径（manifest 键仍用登记路径，防止改路径绕过）。"""
    require(name in M1_INPUTS, f'M1 输入锚: 未登记逻辑名 {name!r}')
    p = Path(path) if path is not None else ROOT / M1_INPUTS[name]
    entries = load_manifest_entries(manifest_path)
    expected = _manifest_input_anchor(entries, M1_INPUTS[name], f'M1 输入 {name}')
    require(p.is_file(), f'M1 输入 {name}: 文件不存在: {p}')
    actual = sha256_file(p)
    require(actual == expected,
            f'M1 输入 {name}: sha256 不匹配（file={M1_INPUTS[name]}）'
            f'期望={expected} 实际={actual}')
    print(f'[M1-INPUT-SHA] {name}: MATCH {actual[:12]}… ({p})', flush=True)
    return {'input': name, 'file': M1_INPUTS[name], 'path': str(p),
            'sha256': actual, 'expected_sha256': expected, 'actual_sha256': actual,
            'anchor_sha256': expected, 'anchor_source': 'configs/version_manifest.yaml',
            'match': True}

def assert_m1_inputs_sha(inputs=None, manifest_path=None, spec_path=None, *,
                         _allow_subset: bool = False) -> list:
    """逐个 M1 主输入计算 sha256 并与 manifest/spec 双源锚比对；任一失败 SystemExit(2)。

    每个输入走 `anchor_sha()`：manifest 与 spec 都有锚时必须一致（冲突即退出）；
    只有 manifest 锚按 manifest；只有 spec 锚按 spec；两者都无 → 缺锚退出。

    inputs: 生产调用必须为 None（校验全部 M1_INPUTS）；仅测试可传 {name: path}，
            并显式 `_allow_subset=True`。空 dict 无论是否测试标志都拒绝（不得 0/0 PASS）。
    manifest_path / spec_path: 可选路径注入（测试用）；默认 configs 下真源。
    返回 ledger: [{input, file, path, sha256, expected_sha256, actual_sha256,
                   anchor_sha256, anchor_source, manifest_sha256, spec_sha256, match}, ...]
    不写任何文件；不提供 CLI/环境变量绕过开关。
    """
    entries = load_manifest_entries(manifest_path)
    spec_hashes = load_spec_input_hashes(spec_path)
    if inputs is None:
        inputs = {name: ROOT / rel for name, rel in M1_INPUTS.items()}
    else:
        inputs = dict(inputs)
        require(bool(inputs), 'M1 输入锚: inputs 为空；拒绝 0/0 PASS')
        if not _allow_subset:
            expected_names = set(M1_INPUTS)
            require(set(inputs) == expected_names,
                    f'M1 输入锚: 非测试调用必须校验全部 {len(expected_names)} 个 M1 输入；'
                    f'缺少={sorted(expected_names - set(inputs))} '
                    f'多余={sorted(set(inputs) - expected_names)}')
    ledger = []
    for name, path in inputs.items():
        p = Path(path)
        key = manifest_input_key(p)
        where = f'M1 输入 {name}'
        require(p.is_file(), f'{where}: 文件不存在: {p}（file={key}）')
        expected, anchor_source = anchor_sha(key, entries=entries, spec_hashes=spec_hashes)
        man_sha = _manifest_optional_anchor(entries, key, where)
        spec_sha = spec_hashes.get(key)
        actual = sha256_file(p)
        require(actual == expected,
                f'{where}: sha256 不匹配（file={key}，锚来源={anchor_source}）'
                f'期望={expected} 实际={actual}')
        ledger.append({'input': name, 'file': key, 'path': str(p),
                       'sha256': actual, 'expected_sha256': expected,
                       'actual_sha256': actual, 'anchor_sha256': expected,
                       'anchor_source': anchor_source,
                       'manifest_sha256': man_sha, 'spec_sha256': spec_sha,
                       'match': True})
        print(f'[M1-INPUT-SHA] {name}: MATCH {actual[:12]}… ({key}; {anchor_source})',
              flush=True)
    print(f'[M1-INPUT-SHA] PASS：{len(ledger)}/{len(ledger)} 个输入与 manifest/spec 锚一致',
          flush=True)
    return ledger

def require(cond, msg: str) -> bool:
    """硬门：条件为假时打印 [HARD-GATE] 并以退出码 2 终止（不静默继续）。"""
    if not bool(cond):
        print(f'[HARD-GATE] {msg}', file=sys.stderr, flush=True)
        raise SystemExit(2)
    return True

def _is_zero_int(value) -> bool:
    try:
        return int(value) == 0
    except (TypeError, ValueError):
        return False

def contract_from_frame(df: pd.DataFrame, expected_rows=None, kind: str = 'auto') -> dict:
    """从面板/样本 DataFrame 现算四条硬门（只读，不写文件）。

    kind='panel'：missing_filled_with_zero 按 network_support_status != 'support' 的
                  PeerMAXraw/OwnMAXraw/PeerReturn 零值统计（与 s3_m1_1 A1 面板合同同口径）。
    kind='sample'：按 m1_exclusion_reason -> 对应值列统计（与 s3_m1_1 _integrity_checks 同口径）。
    expected_rows=None：rows_match_registered 记 ROWS_NOT_APPLICABLE（仅自定义面板使用，
    表示“行数门未施加”，不是通过）。
    """
    if kind == 'auto':
        if 'm1_exclusion_reason' in df.columns:
            kind = 'sample'
        elif 'network_support_status' in df.columns:
            kind = 'panel'
        else:
            kind = 'sample'
    c = {'rows': int(len(df)), 'kind': kind}
    if expected_rows is None:
        c['rows_match_registered'] = ROWS_NOT_APPLICABLE
    else:
        c['rows_registered'] = int(expected_rows)
        c['rows_match_registered'] = bool(len(df) == int(expected_rows))
    if {'focal_id', 'month'}.issubset(df.columns):
        c['duplicate_focal_month_keys'] = int(df.duplicated(['focal_id', 'month']).sum())
    else:
        c['duplicate_focal_month_keys'] = None
    if {'available_at', 'month'}.issubset(df.columns):
        av = pd.to_datetime(df['available_at'], errors='coerce')
        mo = pd.to_datetime(df['month'].astype(str) + '-01', errors='coerce')
        c['pit_violations_strict_lt'] = int((av >= mo).sum())
    else:
        c['pit_violations_strict_lt'] = None
    zeros = None
    if kind == 'sample' and 'm1_exclusion_reason' in df.columns:
        exc = df['m1_exclusion_reason'].fillna('')
        zeros = 0
        for name, col in _SAMPLE_EXCLUSION_VALUE_COL.items():
            if col in df.columns:
                zeros += int((df.loc[exc == name, col].fillna(-999) == 0).sum())
    elif kind == 'panel' and 'network_support_status' in df.columns:
        sup = df['network_support_status'] == 'support'
        cols = [c2 for c2 in _M1_RETURN_COLS if c2 in df.columns]
        if cols:
            zeros = int((df.loc[~sup, cols].fillna(-999) == 0).sum().sum())
    c['missing_filled_with_zero'] = None if zeros is None else int(zeros)
    return c

def _contract_entries(source, where: str, expected_rows, kind: str) -> dict:
    """把 report dict / contract dict / 路径 / DataFrame 归一化为 {key: contract}。"""
    if isinstance(source, pd.DataFrame):
        return {'<frame>': contract_from_frame(source, expected_rows=expected_rows, kind=kind)}
    if isinstance(source, (str, Path)):
        p = Path(source)
        require(p.exists(), f'{where}: 契约来源不存在: {p}')
        suffix = p.suffix.lower()
        if suffix == '.json':
            return _contract_entries(json.loads(p.read_text(encoding='utf-8')),
                                     where, expected_rows, kind)
        if suffix == '.parquet':
            return {'<parquet>': contract_from_frame(pd.read_parquet(p),
                                                     expected_rows=expected_rows, kind=kind)}
        if suffix == '.csv':
            return {'<csv>': contract_from_frame(pd.read_csv(p),
                                                 expected_rows=expected_rows, kind=kind)}
        require(False, f'{where}: 不支持的契约来源类型: {p}')
    if isinstance(source, dict):
        if 'contracts' in source:
            contracts = source.get('contracts')
            require(isinstance(contracts, dict) and bool(contracts),
                    f'{where}: 报告中 contracts 为空/缺失，拒绝继续')
            return contracts
        if any(k in source for k in HARD_GATE_KEYS) or 'missing_filled_with_zero_total' in source:
            return {'<contract>': source}
        require(False, f'{where}: dict 既非 build report（无 contracts）也无四条硬门字段')
    require(False, f'{where}: 无法识别的契约来源（{type(source).__name__}）；'
                   '需要 build report dict / contract dict / json|parquet|csv 路径 / DataFrame')
    return {}

def assert_m1_contract(source, expected_rows=None, where: str = 'm1_contract',
                       kind: str = 'auto') -> dict:
    """硬门入口：核验四条输入契约，任一失败即 SystemExit(2)。

    source:
      - s3_m1_sample_report(.json) 的 report dict（使用其 contracts）
      - 单列 contract dict（直接含四条硬门字段）
      - report/parquet/csv 路径
      - DataFrame（此时 expected_rows 给登记行数；None 表示行数门不适用，仅自定义面板）
    返回摘要 dict（便于报告/测试断言）；不写任何文件。
    """
    contracts = _contract_entries(source, where, expected_rows, kind)
    summary = {'where': where, 'n_contracts': len(contracts), 'status': 'PASS', 'contracts': {}}
    for key, c in contracts.items():
        require(isinstance(c, dict), f'{where}/{key}: 契约不是 dict')
        if 'rows_match_registered' in c:
            rows_val = c['rows_match_registered']
        elif isinstance(expected_rows, dict) and key in expected_rows and 'rows' in c:
            rows_val = bool(int(c['rows']) == int(expected_rows[key]))
        elif expected_rows is not None and not isinstance(expected_rows, dict) and 'rows' in c:
            rows_val = bool(int(c['rows']) == int(expected_rows))
        else:
            require(False, f'{where}/{key}: 缺 rows_match_registered，且无法由 rows+期望行数复算')
        rows_enforced = rows_val != ROWS_NOT_APPLICABLE
        if rows_enforced:
            require(isinstance(rows_val, (bool, np.bool_)) and bool(rows_val),
                    f'{where}/{key}: rows_match_registered={rows_val!r}'
                    f'（rows={c.get("rows")}, 登记={c.get("rows_registered")}）')
        pit = c.get('pit_violations_strict_lt', c.get('pit_violations'))
        require(pit is not None, f'{where}/{key}: 缺 pit_violations_strict_lt')
        require(_is_zero_int(pit), f'{where}/{key}: PIT 违规 {pit} != 0')
        dup = c.get('duplicate_focal_month_keys')
        require(dup is not None, f'{where}/{key}: 缺 duplicate_focal_month_keys')
        require(_is_zero_int(dup), f'{where}/{key}: 重复 (focal_id, month) 键 {dup} != 0')
        zero = c.get('missing_filled_with_zero', c.get('missing_filled_with_zero_total'))
        require(zero is not None, f'{where}/{key}: 缺 missing_filled_with_zero')
        require(_is_zero_int(zero), f'{where}/{key}: 缺失被填 0 的行 {zero} != 0')
        summary['contracts'][key] = {
            'rows': c.get('rows'), 'rows_registered': c.get('rows_registered'),
            'rows_match_registered': rows_val, 'rows_gate_enforced': rows_enforced,
            'pit_violations_strict_lt': pit,
            'duplicate_focal_month_keys': dup, 'missing_filled_with_zero': zero}
    print(f'[M1-CONTRACT] {where}: PASS（{len(contracts)} 个契约 × 4 条硬门）', flush=True)
    return summary

