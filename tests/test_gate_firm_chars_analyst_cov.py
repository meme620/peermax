"""数据门 firm-chars-analyst-cov：firm_chars_clean.csv 必须带 analyst_cov。"""

from pathlib import Path

from peerlab import gates


def _write(root: Path, header, rows) -> None:
    path = root / 'data' / 'processed' / 'firm_chars_clean.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [','.join(header)] + [','.join(r) for r in rows]
    path.write_text('\n'.join(body) + '\n', encoding='utf-8')


def test_missing_file_blocked(tmp_path: Path) -> None:
    outcome = gates.check_firm_chars_analyst_cov(tmp_path)
    assert outcome['blocked'] and outcome['problems'] == []


def test_valid_passes(tmp_path: Path) -> None:
    _write(tmp_path, ['firm_id', 'month', 'analyst_cov'], [['000001', '2020-01', '3']])
    outcome = gates.check_firm_chars_analyst_cov(tmp_path)
    assert outcome['problems'] == []
    assert outcome['checked'] == 1


def test_missing_column_fails(tmp_path: Path) -> None:
    _write(tmp_path, ['firm_id', 'month'], [['000001', '2020-01']])
    problems = gates.check_firm_chars_analyst_cov(tmp_path)['problems']
    assert any('缺 analyst_cov' in p for p in problems)


def test_all_empty_fails(tmp_path: Path) -> None:
    _write(tmp_path, ['firm_id', 'month', 'analyst_cov'], [['000001', '2020-01', '']])
    problems = gates.check_firm_chars_analyst_cov(tmp_path)['problems']
    assert any('全为空' in p for p in problems)
