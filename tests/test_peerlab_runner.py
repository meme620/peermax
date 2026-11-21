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


def test_smoke_args_replace_per_step_not_append():
    """-smoke 必须按 step 整块替换 args；实验级追加会把 --years 塞进每个 step（口径冲突）。"""
    exp = registry.get('network-mda-text')
    for step in exp.steps:
        assert step.args_smoke is not None, f'{step.script} 缺 args_smoke'
        smoke = runner.build_command(exp, step, 0, smoke=True)
        # 追加语义的典型症状：同一 step 里 --years 出现两次，argparse 取末值而脚本按区间解析
        assert smoke.count('--years') <= 1, f'{step.script} 出现重复 --years: {smoke}'
    full = runner.build_command(exp, exp.steps[0], 0)
    assert full[full.index('--years') + 1] == '2014-2022'
    smoke = runner.build_command(exp, exp.steps[0], 0, smoke=True)
    assert smoke[smoke.index('--years') + 1] == '2021-2021'
    assert '--corpus' in smoke and smoke[smoke.index('--corpus') + 1] == 'teacher'


def test_sfx_placeholder_expands_per_seed():
    """requires / outputs / args 里的 {sfx} 必须按 seed 展开。

    gate-report 的 --input-suffix 与 requires 若一边带后缀一边不带，就绪判定与
    实际读取会对不上（status 报 ready 而 run 读不到文件）。
    """
    exp = registry.get('gate-report')
    e3 = exp.expanded(3)
    assert [p for p in e3.requires if 'pricing_table' in p] == [
        'reports/data_gates/G0/pricing_table_s3.json']
    assert e3.steps[0].args == ('--input-suffix', '_s3')
    cmd = runner.build_command(exp, exp.steps[0], 3)
    assert '--input-suffix' in cmd and cmd[cmd.index('--input-suffix') + 1] == '_s3'


def test_seed_placeholder_in_step_args():
    """随 seed 变化的实验必须把 seed 传进脚本，否则多 seed 跑出同一份产物。"""
    exp = registry.get('robustness-checks')
    e2 = exp.expanded(2)
    assert e2.steps[0].args == ('--seed', '2')
    assert e2.outputs[0].endswith('pricing_robustness_placebo_s2.json')
    cmd = runner.build_command(exp, exp.steps[0], 2)
    assert cmd[cmd.index('--seed') + 1] == '2'
    assert cmd[cmd.index('--out-suffix') + 1] == '_s2'


def test_seed_dependent_outputs_do_not_collide():
    """两个 seed 的产物路径集合必须不同，否则后跑的覆盖先跑的。"""
    for exp_id in ('robustness-checks', 'density-null', 'shadow-inference', 'fm-main-table'):
        exp = registry.get(exp_id)
        a, b = exp.expanded(0).outputs, exp.expanded(1).outputs
        assert set(a) != set(b), f'{exp_id} 的产物不随 seed 变化'


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
    # 文件名随 headline 样本版本变，从 registry 指针推导，避免每次升版都要改测试
    label = registry.headline_sample_label()
    sample = f'data/processed/pricing_sample_ma_teacher_{label}.parquet'
    assert sample in check['missing']
    (tmp_path / sample).write_bytes(b'')
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
