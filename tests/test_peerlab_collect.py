import json
from pathlib import Path

from peerlab import collect, registry


def test_dotted_get_nested_and_list():
    doc = {'main': {'ma_teacher': {'coefs': {'PeerMAXraw': {'beta': 0.13}}}}, 'xs': [1, 2, 3]}
    assert collect.dotted_get(doc, 'main.ma_teacher.coefs.PeerMAXraw.beta') == 0.13
    assert collect.dotted_get(doc, 'xs.1') == 2
    assert collect.dotted_get(doc, 'missing.key') is collect.MISSING


def _fake_run(result_dir: Path, exp_id: str, seed: int, value: float) -> None:
    run_id = exp_id + '--s' + str(seed) + '--20260101T000000Z'
    run_dir = result_dir / 'runs' / run_id
    out = run_dir / 'outputs/reports/data_gates/G0'
    out.mkdir(parents=True)
    doc = {'main': {'ma_teacher': {'coefs': {'PeerMAXraw': {'beta': value, 't': 6.0}}}}}
    (out / ('s3_m1_pricing_table_s' + str(seed) + '.json')).write_text(json.dumps(doc), encoding='utf-8')
    (run_dir / 'run.json').write_text('{}', encoding='utf-8')
    index = result_dir / 'index.json'
    doc_index = json.loads(index.read_text()) if index.is_file() else {'runs': []}
    doc_index['runs'].append({'run_id': run_id, 'exp': exp_id, 'seed': seed,
                              'status': 'ok', 'smoke': False})
    index.write_text(json.dumps(doc_index), encoding='utf-8')


def test_extract_metrics_uses_suffix_glob(tmp_path: Path):
    exp = registry.get('fm-main-table')
    _fake_run(tmp_path, 'fm-main-table', 2, 0.131)
    values = collect.extract_metrics(tmp_path / 'runs/fm-main-table--s2--20260101T000000Z', exp, 2)
    assert values['beta_peer'] == 0.131
    assert values['t_peer'] == 6.0


def test_build_tables_marks_identical_across_seeds(tmp_path: Path):
    exp = registry.get('fm-main-table')
    for seed in registry.SEEDS:
        _fake_run(tmp_path, 'fm-main-table', seed, 0.131)
    written = collect.build_tables(tmp_path, (exp,))
    table = (tmp_path / 'tables/fm-main-table.csv').read_text(encoding='utf-8').strip().splitlines()
    assert table[0].startswith('metric,s0,s1,s2,s3,s4,mean,sd,identical')
    beta_row = [line for line in table if line.startswith('beta_peer')][0]
    assert beta_row.endswith('True')
    assert any(path.name == 'metrics.csv' for path in written)


def test_summary_without_runs(tmp_path: Path):
    (tmp_path / 'index.json').write_text('{"runs": []}', encoding='utf-8')
    path = collect.write_summary(tmp_path, registry.EXPERIMENTS)
    text = path.read_text(encoding='utf-8')
    assert '当前没有有效运行' in text
    assert '## fm-main-table' in text
