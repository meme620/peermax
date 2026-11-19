from pathlib import Path

from peerlab import lint, registry, spec

ROOT = Path(__file__).resolve().parents[1]


def test_spec_resolves_known_refs() -> None:
    assert spec._resolve('script:pricing_gate_report', ROOT) is None
    assert spec._resolve('gate:daily-returns-board', ROOT) is None
    assert spec._resolve('lint:check_paths', ROOT) is None
    assert spec._resolve('test:tests/test_manifest_input_guard.py', ROOT) is None


def test_spec_flags_unknown_refs() -> None:
    assert spec._resolve('gate:no-such-gate', ROOT)
    assert spec._resolve('script:no_such_script', ROOT)
    assert spec._resolve('weird:thing', ROOT)


def test_repo_spec_is_clean() -> None:
    assert spec.validate(ROOT) == []


def test_every_experiment_has_notes() -> None:
    for exp in registry.EXPERIMENTS:
        note = registry.notes_for(exp.id)
        assert note['purpose'] and note['notes'], exp.id


def test_stage_code_in_id_rejected() -> None:
    class Fake:
        id = 'm1-main'
    problems = lint.check_ids([Fake()])
    assert any('阶段代号' in p for p in problems)


def test_stage_coded_path_rejected() -> None:
    problems = lint.check_path_names(['scripts/s3_legacy_module.py'])
    assert any('阶段代号' in p for p in problems)


def test_number_prefix_dir_rejected() -> None:
    problems = lint.check_path_names(['01_proposal/x.md'])
    assert any('数字前缀' in p for p in problems)


def test_deep_and_single_file_dirs_rejected() -> None:
    problems = lint.check_path_names(['a/b/c/x.py', 'solo/only.md'])
    assert any('层级' in p for p in problems)
    assert any('只装一个文件' in p for p in problems)

def test_gitkeep_placeholder_ignores_depth() -> None:
    problems = lint.check_path_names(['data/raw/drive/.gitkeep', 'data/processed/analyst/edges/.gitkeep'])
    assert problems == []


def test_data_dir_with_single_file_allowed() -> None:
    problems = lint.check_path_names(['data/external/reference.xlsx'])
    assert not any('只装一个文件' in p for p in problems)

def test_config_yaml_stage_code_allowed() -> None:
    problems = lint.check_path_names(['configs/m1_contracts.yaml'])
    assert not any('阶段代号' in p for p in problems)


def test_dangling_script_literal_reported(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'real.py').write_text('', encoding='utf-8')
    (tmp_path / 'spec').mkdir()
    (tmp_path / 'spec' / 'note.yaml').write_text('python scripts/gone.py', encoding='utf-8')
    problems = lint.check_script_literals(tmp_path, paths=['spec/note.yaml'])
    assert any('gone' in p for p in problems)


def test_existing_script_literal_ok(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'real.py').write_text('', encoding='utf-8')
    (tmp_path / 'spec').mkdir()
    (tmp_path / 'spec' / 'note.yaml').write_text('python scripts/real.py', encoding='utf-8')
    assert lint.check_script_literals(tmp_path, paths=['spec/note.yaml']) == []


def test_text_io_without_encoding_reported(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'bad.py').write_text(
        "open('x')\nPath('y').read_text()\n", encoding='utf-8')
    problems = lint.check_text_io_encoding(tmp_path)
    assert len(problems) == 2


def test_hygiene_flags_blanket_except(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'bad.py').write_text(
        "try:\n    pass\nexcept Exception:\n    pass\n", encoding='utf-8')
    problems = lint.check_hygiene(tmp_path)
    assert any('pass/continue' in p for p in problems)


def test_hygiene_flags_path_insert_but_exempts_bootstrap(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'bad.py').write_text(
        "import sys\nsys.path.insert(0, 'x')\n", encoding='utf-8')
    (tmp_path / 'peerlab').mkdir()
    (tmp_path / 'peerlab' / 'bootstrap.py').write_text(
        "import sys\nsys.path.insert(0, 'x')\n", encoding='utf-8')
    problems = lint.check_hygiene(tmp_path)
    assert any('scripts/bad.py' in p for p in problems)
    assert not any('bootstrap' in p for p in problems)


def test_binary_open_needs_no_encoding(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'ok.py').write_text("open('x', 'rb')\n", encoding='utf-8')
    assert lint.check_text_io_encoding(tmp_path) == []


def test_script_literal_in_tests_ignored(tmp_path: Path) -> None:
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'test_x.py').write_text("(tmp / 'scripts/writer.py')", encoding='utf-8')
    assert lint.check_script_literals(tmp_path, paths=['tests/test_x.py']) == []
