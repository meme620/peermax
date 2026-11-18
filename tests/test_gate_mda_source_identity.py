"""数据门 mda-source-identity：身份无 mismatch、重下队列空、精化名单过规则。"""

from pathlib import Path

from peerlab import gates


def _write(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [','.join(header)] + [','.join(r) for r in rows]
    path.write_text('\n'.join(body) + '\n', encoding='utf-8')


def _identity(root: Path, rows) -> None:
    _write(root / 'artifacts' / 'review' / 'mda_src_identity.csv',
           ['code6', 'year', 'identity'], rows)


def _refined(root: Path, rows) -> None:
    _write(root / 'data' / 'processed' / 'name_history_refined.csv',
           ['action', 'name', 'code', 'year', 'date'], rows)


def test_missing_inputs_blocked(tmp_path: Path) -> None:
    outcome = gates.check_mda_source_identity(tmp_path)
    assert outcome['blocked'] and outcome['problems'] == []


def test_valid_passes(tmp_path: Path) -> None:
    _identity(tmp_path, [['000001', '2020', 'match'], ['000002', '2020', 'unknown']])
    _refined(tmp_path, [['A', '平安银行', '000001', '2020', '2020-03-01']])
    outcome = gates.check_mda_source_identity(tmp_path)
    assert outcome['problems'] == []
    assert outcome['checked'] == 3


def test_mismatch_fails(tmp_path: Path) -> None:
    _identity(tmp_path, [['000001', '2020', 'mismatch']])
    _refined(tmp_path, [['A', '平安银行', '000001', '2020', '2020-03-01']])
    problems = gates.check_mda_source_identity(tmp_path)['problems']
    assert any('mismatch' in p for p in problems)


def test_pending_redownload_fails(tmp_path: Path) -> None:
    _identity(tmp_path, [['000001', '2020', 'match']])
    _refined(tmp_path, [['A', '平安银行', '000001', '2020', '2020-03-01']])
    _write(tmp_path / 'artifacts' / 'review' / 'mda_redownload_pairs.csv',
           ['code', 'year', 'reason'], [['000002', '2019', 'tiny:12B']])
    problems = gates.check_mda_source_identity(tmp_path)['problems']
    assert any('重下队列' in p for p in problems)


def test_refined_name_violating_filter_fails(tmp_path: Path) -> None:
    _identity(tmp_path, [['000001', '2020', 'match']])
    _refined(tmp_path, [['A', '公司名称变更提示', '000001', '2020', '2020-03-01'],
                        ['A', '平安银行', '000001', '2020', '2020-03-01']])
    problems = gates.check_mda_source_identity(tmp_path)['problems']
    assert any('精化规则' in p for p in problems)


def test_unknown_dominance_fails(tmp_path: Path) -> None:
    _identity(tmp_path, [['000001', '2020', 'unknown'], ['000002', '2020', 'unknown'],
                         ['000003', '2020', 'match']])
    _refined(tmp_path, [['A', '平安银行', '000001', '2020', '2020-03-01']])
    problems = gates.check_mda_source_identity(tmp_path)['problems']
    assert any('unknown' in p for p in problems)
