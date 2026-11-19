from pathlib import Path

from peerlab import registry, runner


def test_seeds_are_global_and_five():
    assert registry.SEEDS == (0, 1, 2, 3, 4)
    for exp in registry.EXPERIMENTS:
        assert exp.scope in {'prepare', 'per_seed'}


def test_run_id_shape():
    assert runner.make_run_id('fm-main-table', 3, '20260101T000000Z') == 'fm-main-table--s3--20260101T000000Z'
    assert runner.make_run_id('network-business-text', 0).startswith('network-business-text--s0--')


def test_build_command_injects_seed_and_suffix():
    exp = registry.get('fm-main-table')
    cmd = runner.build_command(exp, exp.steps[0], 3)
    assert cmd[-2:] == ['--out-suffix', '_s3']
    exp_seeded = registry.get('shadow-inference')
    cmd_seeded = runner.build_command(exp_seeded, exp_seeded.steps[0], 2)
    assert cmd_seeded[-2:] == ['--seed', '2']


def test_readiness_reports_missing(tmp_path: Path):
    exp = registry.get('fm-main-table')
    check = runner.readiness(exp, tmp_path)
    assert check['ready'] is False
    assert 'configs/parameter_registry.yaml' in check['missing']
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/parameter_registry.yaml').write_text('x: 1', encoding='utf-8')
    # 2026-11-19：目录存在不再算就绪——必须落到具体产物（见 registry.EXPERIMENTS.requires）
    (tmp_path / 'data/processed').mkdir(parents=True)
    check = runner.readiness(exp, tmp_path)
    assert check['ready'] is False
    assert 'data/processed/pricing_sample_ma_teacher_v3.parquet' in check['missing']
    (tmp_path / 'data/processed/pricing_sample_ma_teacher_v3.parquet').write_bytes(b'')
    assert runner.readiness(exp, tmp_path)['ready'] is True


def test_matches_handles_prefix_and_glob():
    patterns = ('data/processed/network_ma_teacher/**', 'reports/data_gates/G0/pricing_*.json')
    assert runner.matches('data/processed/network_ma_teacher/2021_edges.csv', patterns)
    assert runner.matches('reports/data_gates/G0/pricing_table.json', patterns)
    assert not runner.matches('data/processed/other.csv', patterns)


def test_changed_files_detects_new_and_modified():
    before = {'a.csv': (10, 1)}
    after = {'a.csv': (10, 2), 'b.csv': (5, 1)}
    assert runner.changed_files(before, after) == ['a.csv', 'b.csv']


def test_collect_copies_matching_and_skips_large(tmp_path: Path, monkeypatch):
    root = tmp_path
    target = root / 'reports/data_gates/G0/pricing_table.json'
    target.parent.mkdir(parents=True)
    target.write_text('{}', encoding='utf-8')
    big = root / 'reports/data_gates/G0/pricing_big.json'
    big.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(runner, 'MAX_COPY_BYTES', 1)
    run_dir = root / 'result/runs/r1'
    run_dir.mkdir(parents=True)
    collected = runner.collect(run_dir, ['reports/data_gates/G0/pricing_table.json',
                                        'reports/data_gates/G0/pricing_big.json'],
                               ('reports/data_gates/G0/pricing_*.json',), root)
    flags = {item['path']: item['copied'] for item in collected}
    assert flags['reports/data_gates/G0/pricing_table.json'] is False
    assert flags['reports/data_gates/G0/pricing_big.json'] is False
    monkeypatch.setattr(runner, 'MAX_COPY_BYTES', 1024 * 1024)
    collected = runner.collect(run_dir, ['reports/data_gates/G0/pricing_table.json'],
                               ('reports/data_gates/G0/pricing_*.json',), root)
    assert collected[0]['copied'] is True
    assert (run_dir / 'outputs/reports/data_gates/G0/pricing_table.json').is_file()
