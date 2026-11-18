"""数据不变量门（CLI：python -m peerlab gate）。

每个门守住一条曾由一次性修复脚本手工做过的不变量。判据直接复用生产者的纯规则函数
（board_of、compute_actual_sample_window、refine_row、load_bonus / bonus_factor_by_year），
不在门里复制一份规则，规则改了门跟着改。

三态：pass / fail（判据不满足）/ blocked（要核验的产物或输入不存在，不当通过）。
证据写 reports/data_gates/G0/gate_<id>.json（该目录不入库）。
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = Path('reports') / 'data_gates' / 'G0'
DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
SHARE_STATUSES = {'ok', 'needs_review', 'em_anchor', 'c_anchor', 'no_anchor'}
SHARE_METHODS = {'A', 'C', 'bonus_back', 'no_anchor'}


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _script(name: str):
    """按文件路径载入 scripts/<name>.py，取生产者的规则函数。"""
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / 'scripts' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list:
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _header(path: Path) -> list:
    with open(path, encoding='utf-8', newline='') as f:
        return next(csv.reader(f), [])


def check_edge_build_report(root: Path) -> dict:
    """边表与 build_report 成套：缺报告、缺键、placeholder τ 都算 fail。

    守的是：相对 --out-dir 曾触发 relative_to(ROOT) 的 ValueError，边表写出来了、
    报告整批缺失，只能事后从边表回填元数据。缺口径的报告等于没有可复现的口径。
    """
    sims = sorted((root / 'data' / 'processed').glob('network_*/*_similarity.csv'))
    if not sims:
        return {'problems': [], 'checked': 0,
                'blocked': 'data/processed/network_*/ 下没有 *_similarity.csv'}
    problems = []
    for sim in sims:
        report = sim.with_name(sim.name.replace('_similarity.csv', '_build_report.json'))
        if not report.is_file():
            problems.append(_rel(root, sim) + ': 缺同名 build_report')
            continue
        meta = json.loads(report.read_text(encoding='utf-8'))
        if not isinstance(meta, dict):
            problems.append(_rel(root, report) + ': 顶层不是对象')
            continue
        version = str(meta.get('network_version') or '').strip()
        if not version:
            problems.append(_rel(root, report) + ': network_version 缺失')
        available_at = str(meta.get('available_at') or '')
        if not DATE_RE.fullmatch(available_at):
            problems.append(_rel(root, report) + ': available_at=' + repr(available_at) + ' 不是 YYYY-MM-DD')
        if 'n_edges' not in meta:
            problems.append(_rel(root, report) + ': n_edges 缺失')
        if version.startswith('pm_mb'):
            if meta.get('tau') is None:
                problems.append(_rel(root, report) + ': M-B 报告缺 tau')
            if meta.get('tau_mode') == 'placeholder':
                problems.append(_rel(root, report) + ': tau_mode=placeholder（骨架产物不得当边表交付）')
    return {'problems': problems, 'checked': len(sims), 'blocked': None}


def check_sample_metadata(root: Path) -> dict:
    """样本窗口列现算 + monthly_returns_clean_v2 的 sample_version 由数据算。

    守的是：window_requested / sample_window_actual 曾靠 metadata-only 重写事后补上，
    sample_version 曾写成固定标签 mainline_2018_2025_fullperiod。两处都必须由数据现算：
    标签写死后按版本筛样本会静默选错行。
    """
    processed = root / 'data' / 'processed'
    samples = sorted(processed.glob('m1_common_sample_*.parquet'))
    returns = processed / 'monthly_returns_clean_v2.csv'
    if not samples or not returns.is_file():
        return {'problems': [], 'checked': 0,
                'blocked': '缺 m1_common_sample_*.parquet 或 monthly_returns_clean_v2.csv'}
    import pandas as pd
    import pyarrow.parquet as pq

    rule = _script('pricing_sample_build').compute_actual_sample_window
    need = ['month', 'network_version', 'm1_in_fm_sample', 'window_requested', 'sample_window_actual']
    problems = []
    for path in samples:
        names = pq.ParquetFile(path).schema.names
        missing = [c for c in need if c not in names]
        if missing:
            problems.append(_rel(root, path) + ': 缺列 ' + str(missing))
            continue
        frame = pd.read_parquet(path, columns=need)
        expected = rule(frame)
        actual = sorted(set(frame['sample_window_actual'].dropna().astype(str)))
        if actual != [expected]:
            problems.append(_rel(root, path) + ': sample_window_actual=' + str(actual) +
                            '，按列现算应为 ' + expected)
        requested = sorted(set(frame['window_requested'].dropna().astype(str)))
        if len(requested) != 1:
            problems.append(_rel(root, path) + ': window_requested 不唯一 ' + str(requested))

    columns = _header(returns)
    need_ret = ['month', 'sample_version', 'sample_version_source']
    missing_ret = [c for c in need_ret if c not in columns]
    if missing_ret:
        problems.append(_rel(root, returns) + ': 缺列 ' + str(missing_ret))
    else:
        ret = pd.read_csv(returns, usecols=need_ret, dtype=str)
        expected_version = ret['month'].min() + '..' + ret['month'].max()
        versions = sorted(set(ret['sample_version']))
        if versions != [expected_version]:
            problems.append(_rel(root, returns) + ': sample_version=' + str(versions) +
                            '，按数据应为 ' + expected_version)
        sources = sorted(set(ret['sample_version_source']))
        if sources != ['computed_from_data']:
            problems.append(_rel(root, returns) + ': sample_version_source=' + str(sources) +
                            '（应为 computed_from_data）')
    return {'problems': problems, 'checked': len(samples) + 1, 'blocked': None}


def check_daily_returns_board(root: Path) -> dict:
    """日收益表的 board 列与代码前缀一致，且不多出第二列。

    来源：build_id_map.py 原地给 daily_returns_clean/*.csv 追加 board 列（幂等）。
    缺列或错值会让涨跌停阈值按主板算，静默污染 limit flag。
    """
    returns_dir = root / 'data' / 'processed' / 'daily_returns_clean'
    id_map = root / 'data' / 'processed' / 'firm_id_map.csv'
    if not returns_dir.is_dir() or not id_map.is_file():
        return {'problems': [], 'checked': 0,
                'blocked': '缺 data/processed/daily_returns_clean/ 或 firm_id_map.csv'}
    files = sorted(returns_dir.glob('*.csv'))
    if not files:
        return {'problems': [], 'checked': 0, 'blocked': 'daily_returns_clean/ 下没有 csv'}
    board_of = _script('build_id_map').board_of
    expected = {}
    problems = []
    for row in _rows(id_map):
        code6, board = row.get('code6', ''), row.get('board', '')
        expected[code6] = board
        if board != board_of(code6):
            problems.append('firm_id_map.csv: ' + code6 + ' board=' + repr(board) +
                            ' 与代码前缀规则 ' + repr(board_of(code6)) + ' 不符')
    for path in files:
        code6 = path.stem[:6]
        want = expected.get(code6)
        header = _header(path)
        n_board = header.count('board')
        if want is None:
            if n_board:
                problems.append(_rel(root, path) + ': 未登记代码却有 board 列')
            continue
        if n_board != 1:
            problems.append(_rel(root, path) + ': board 列 ' + str(n_board) + ' 个（应为 1）')
            continue
        index = header.index('board')
        with open(path, encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
            wrong = [r[index] for r in reader if len(r) > index and r[index] != want]
        if wrong:
            problems.append(_rel(root, path) + ': board 值 ' + str(sorted(set(wrong))[:3]) +
                            ' 与代码规则 ' + want + ' 不符')
    return {'problems': problems, 'checked': len(files), 'blocked': None}


def check_mda_source_identity(root: Path) -> dict:
    """MD&A 源身份：无 mismatch、重下队列空、精化名单过自身过滤规则。

    来源：corpus_mda_identity_second_pass.py / corpus_mda_identity_confirm.py 原地改
    artifacts/review/mda_src_identity.csv，corpus_mda_redownload_queue.py 出重下清单，
    name_history_refine.py 收紧公司名词表。unknown 占比只报告，不设自造阈值。
    """
    identity = root / 'artifacts' / 'review' / 'mda_src_identity.csv'
    refined = root / 'data' / 'processed' / 'name_history_refined.csv'
    if not identity.is_file() or not refined.is_file():
        return {'problems': [], 'checked': 0,
                'blocked': '缺 artifacts/review/mda_src_identity.csv 或 name_history_refined.csv'}
    problems = []
    rows = _rows(identity)
    columns = _header(identity)
    if 'identity' not in columns:
        problems.append(_rel(root, identity) + ': 缺 identity 列')
    else:
        mismatch = [r for r in rows if r.get('identity') == 'mismatch']
        if mismatch:
            sample = ', '.join((r.get('code6') or '?') + '/' + (r.get('year') or '?')
                               for r in mismatch[:3])
            problems.append(_rel(root, identity) + ': mismatch ' + str(len(mismatch)) + ' 行（' + sample + '）')
        unknown = sum(1 for r in rows if r.get('identity') == 'unknown')
        if rows and unknown / len(rows) > 0.5:
            problems.append(_rel(root, identity) + ': unknown 占比 ' +
                            format(unknown / len(rows), '.1%') + '（>50%，身份信号失效）')
    queue = root / 'artifacts' / 'review' / 'mda_redownload_pairs.csv'
    if queue.is_file():
        queued = _rows(queue)
        if queued:
            problems.append(_rel(root, queue) + ': 重下队列 ' + str(len(queued)) + ' 条（应为空）')
    refined_rows = _rows(refined)
    refined_columns = _header(refined)
    missing = [c for c in ('action', 'name') if c not in refined_columns]
    if missing:
        problems.append(_rel(root, refined) + ': 缺列 ' + str(missing))
    else:
        refine_row = _script('name_history_refine').refine_row
        dropped = [r for r in refined_rows if refine_row(dict(r)) is None]
        if dropped:
            sample = ', '.join((r.get('action') or '?') + ':' + (r.get('name') or '?')
                               for r in dropped[:3])
            problems.append(_rel(root, refined) + ': ' + str(len(dropped)) +
                            ' 行不过精化规则（' + sample + '）')
    return {'problems': problems, 'checked': len(rows) + len(refined_rows), 'blocked': None}


def check_share_capital_chain(root: Path) -> dict:
    """股本链：连续两年都由送转展开得到时，比值必须等于送转乘数。

    来源：build_share_capital_v2.py 以"锚点 + 双向送转展开"修掉 v1 未展开送转的股本。
    展开链断了，市值/BM 全线错，且不会报错。
    """
    path = root / 'data' / 'processed' / 'share_capital_history_v2.csv'
    bonus_dir = root / 'data' / 'raw' / 'eastmoney' / 'sharebonus'
    if not path.is_file() or not bonus_dir.is_dir():
        return {'problems': [], 'checked': 0,
                'blocked': '缺 share_capital_history_v2.csv 或 data/raw/eastmoney/sharebonus/'}
    module = _script('build_share_capital_v2')
    problems = []
    rows = _rows(path)
    columns = _header(path)
    missing = [c for c in ('code', 'year', 'end_share', 'method', 'status', 'source') if c not in columns]
    if missing:
        return {'problems': [_rel(root, path) + ': 缺列 ' + str(missing)], 'checked': 0, 'blocked': None}
    seen = set()
    by_code = defaultdict(list)
    for row in rows:
        key = (row['code'], row['year'])
        if key in seen:
            problems.append(_rel(root, path) + ': 重复行 ' + row['code'] + '/' + row['year'])
        seen.add(key)
        if row['status'] not in SHARE_STATUSES:
            problems.append(_rel(root, path) + ': ' + row['code'] + '/' + row['year'] +
                            ' status=' + repr(row['status']) + ' 不在登记集合')
        if row['method'] not in SHARE_METHODS:
            problems.append(_rel(root, path) + ': ' + row['code'] + '/' + row['year'] +
                            ' method=' + repr(row['method']) + ' 不在登记集合')
        by_code[row['code']].append(row)
    for code, code_rows in sorted(by_code.items()):
        code_rows.sort(key=lambda r: int(r['year']))
        factor = module.bonus_factor_by_year(module.load_bonus(code, bonus_dir))
        for prev, cur in zip(code_rows, code_rows[1:]):
            if int(cur['year']) != int(prev['year']) + 1:
                continue
            if cur['method'] != 'bonus_back' or prev['method'] != 'bonus_back':
                continue
            if cur['status'] == 'needs_review' or not prev['end_share'] or not cur['end_share']:
                continue
            expected = float(prev['end_share']) * factor.get(int(cur['year']), 1.0)
            actual = float(cur['end_share'])
            if abs(actual - expected) > max(1e-6, 1e-9 * abs(expected)):
                problems.append(code + '/' + cur['year'] + ': end_share=' + format(actual, 'g') +
                                '，按 ' + prev['year'] + ' 展开应为 ' + format(expected, 'g'))
    return {'problems': problems, 'checked': len(rows), 'blocked': None}


def check_firm_chars_analyst_cov(root: Path) -> dict:
    """firm_chars_clean.csv 必须带 analyst_cov 且非空。

    来源：analyst_chars_merge.py 在 runner 之外手工回填这一列（gate2 记为 step11 不在 runner）。
    列丢了，M1 的 AnalystCoverage 控制变量整列 NaN，只在样本报告里表现为 control_missing。
    """
    path = root / 'data' / 'processed' / 'firm_chars_clean.csv'
    if not path.is_file():
        return {'problems': [], 'checked': 0, 'blocked': '缺 data/processed/firm_chars_clean.csv'}
    header = _header(path)
    if 'analyst_cov' not in header:
        return {'problems': [_rel(root, path) + ': 缺 analyst_cov 列'], 'checked': 0, 'blocked': None}
    index = header.index('analyst_cov')
    filled = 0
    total = 0
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            total += 1
            if len(row) > index and row[index]:
                filled += 1
    problems = []
    if filled == 0:
        problems.append(_rel(root, path) + ': analyst_cov 全为空（' + str(total) + ' 行）')
    return {'problems': problems, 'checked': total, 'blocked': None}


@dataclass(frozen=True)
class Gate:
    id: str
    title: str
    check: object


GATES: tuple = (
    Gate('edge-build-report', '边表与 build_report 成套', check_edge_build_report),
    Gate('sample-metadata', '样本窗口列与 sample_version 现算', check_sample_metadata),
    Gate('daily-returns-board', '日收益 board 列与代码前缀一致', check_daily_returns_board),
    Gate('mda-source-identity', 'MD&A 源身份与重下队列干净', check_mda_source_identity),
    Gate('share-capital-chain', '股本送转展开链自洽', check_share_capital_chain),
    Gate('firm-chars-analyst-cov', 'firm_chars 带 analyst_cov', check_firm_chars_analyst_cov),
)


def run_gates(root: Path = ROOT, only=None) -> list:
    root = Path(root)
    results = []
    for gate in GATES:
        if only and gate.id not in only:
            continue
        outcome = gate.check(root)
        problems = outcome['problems']
        blocked = outcome.get('blocked')
        status = 'blocked' if blocked else ('fail' if problems else 'pass')
        evidence = {'gate': gate.id, 'title': gate.title, 'status': status,
                    'checked': outcome['checked'], 'blocked': blocked, 'problems': problems,
                    'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        out = root / EVIDENCE_DIR / ('gate_' + gate.id + '.json')
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        results.append({**evidence, 'evidence': _rel(root, out)})
    return results
