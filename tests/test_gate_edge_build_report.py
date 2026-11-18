"""数据门 edge-build-report：每张边表都要有可复现的 build_report。"""

import json
from pathlib import Path

from peerlab import gates


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def _report(**overrides) -> str:
    meta = {'network_version': 'pm_mb_tfidf_v1_2021', 'available_at': '2022-04-30',
            'n_edges': 12, 'tau': 0.0888, 'tau_mode': 'explicit'}
    meta.update(overrides)
    return json.dumps(meta, ensure_ascii=False)


def test_no_edges_blocked(tmp_path: Path) -> None:
    outcome = gates.check_edge_build_report(tmp_path)
    assert outcome['blocked']
    assert outcome['problems'] == []


def test_valid_passes(tmp_path: Path) -> None:
    net = tmp_path / 'data' / 'processed' / 'network_mb'
    _write(net / '2021_similarity.csv', 'focal_id,source_id\n')
    _write(net / '2021_build_report.json', _report())
    outcome = gates.check_edge_build_report(tmp_path)
    assert outcome['problems'] == []
    assert outcome['checked'] == 1


def test_missing_report_fails(tmp_path: Path) -> None:
    net = tmp_path / 'data' / 'processed' / 'network_mb'
    _write(net / '2021_similarity.csv', 'focal_id,source_id\n')
    outcome = gates.check_edge_build_report(tmp_path)
    assert outcome['problems'] and not outcome['blocked']


def test_placeholder_tau_fails(tmp_path: Path) -> None:
    net = tmp_path / 'data' / 'processed' / 'network_mb'
    _write(net / '2021_similarity.csv', 'focal_id\n')
    _write(net / '2021_build_report.json', _report(tau=None, tau_mode='placeholder'))
    problems = gates.check_edge_build_report(tmp_path)['problems']
    assert any('tau' in p for p in problems)
    assert any('placeholder' in p for p in problems)


def test_missing_keys_fails(tmp_path: Path) -> None:
    net = tmp_path / 'data' / 'processed' / 'network_ma_teacher'
    _write(net / '2020_similarity.csv', 'focal_id\n')
    _write(net / '2020_build_report.json', _report(network_version='', available_at='', n_edges=0))
    problems = gates.check_edge_build_report(tmp_path)['problems']
    assert any('network_version' in p for p in problems)
    assert any('available_at' in p for p in problems)
