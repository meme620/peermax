from pathlib import Path

from peerlab import lint, registry

SECTIONS = lint.SECTIONS


def write_doc(path: Path, ids, sections) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = []
    for exp_id in ids:
        body.append('## ' + exp_id)
        body.append('')
        for section in sections:
            body.append('### ' + section)
            body.append('')
            body.append('内容')
            body.append('')
    path.write_text('\n'.join(body), encoding='utf-8')


def test_experiments_docs_complete(tmp_path: Path) -> None:
    ids = [exp.id for exp in registry.EXPERIMENTS]
    write_doc(tmp_path / 'docs' / 'experiments.md', ids, SECTIONS)
    assert lint.check_experiments(tmp_path, registry.EXPERIMENTS) == []


def test_missing_section_reported(tmp_path: Path) -> None:
    ids = [exp.id for exp in registry.EXPERIMENTS]
    write_doc(tmp_path / 'docs' / 'experiments.md', ids, SECTIONS[1:])
    problems = lint.check_experiments(tmp_path, registry.EXPERIMENTS)
    assert any('缺小节' in p for p in problems)


def test_unregistered_block_reported(tmp_path: Path) -> None:
    ids = [exp.id for exp in registry.EXPERIMENTS] + ['bogus-experiment']
    write_doc(tmp_path / 'docs' / 'experiments.md', ids, SECTIONS)
    problems = lint.check_experiments(tmp_path, registry.EXPERIMENTS)
    assert any('不在注册表里' in p for p in problems)


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

def test_data_dir_with_single_file_allowed() -> None:
    problems = lint.check_path_names(['data/external/reference.xlsx'])
    assert not any('只装一个文件' in p for p in problems)

def test_config_yaml_stage_code_allowed() -> None:
    problems = lint.check_path_names(['configs/m1_contracts.yaml'])
    assert not any('阶段代号' in p for p in problems)


def test_dangling_script_literal_reported(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'real.py').write_text('', encoding='utf-8')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'note.md').write_text('python scripts/gone.py', encoding='utf-8')
    problems = lint.check_script_literals(tmp_path, paths=['docs/note.md'])
    assert any('gone' in p for p in problems)


def test_existing_script_literal_ok(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'real.py').write_text('', encoding='utf-8')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs' / 'note.md').write_text('python scripts/real.py', encoding='utf-8')
    assert lint.check_script_literals(tmp_path, paths=['docs/note.md']) == []


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
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'test_x.py').write_text("(tmp / 'scripts/writer.py')", encoding='utf-8')
    assert lint.check_script_literals(tmp_path, paths=['tests/test_x.py']) == []


def _inventory(root: Path, body: str) -> None:
    (root / 'docs').mkdir(parents=True, exist_ok=True)
    header = 'name,former_name,path,role,imported_by,test_ref\n'
    (root / 'docs' / 'script_inventory.csv').write_text(header + body, encoding='utf-8')


def test_inventory_roundtrip_ok(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'alpha.py').write_text('', encoding='utf-8')
    _inventory(tmp_path, 'alpha,,scripts/alpha.py,stage,0,\n')
    assert lint.check_inventory(tmp_path) == []


def test_inventory_missing_row_reported(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'alpha.py').write_text('', encoding='utf-8')
    (tmp_path / 'scripts' / 'beta.py').write_text('', encoding='utf-8')
    _inventory(tmp_path, 'alpha,,scripts/alpha.py,stage,0,\n')
    problems = lint.check_inventory(tmp_path)
    assert any('beta' in p for p in problems)


def test_inventory_bad_role_and_test_ref(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'alpha.py').write_text('', encoding='utf-8')
    _inventory(tmp_path, 'alpha,,scripts/alpha.py,misc,0,test_nope.py\n')
    problems = lint.check_inventory(tmp_path)
    assert any('role' in p for p in problems)
    assert any('test_ref' in p for p in problems)


def test_oneoff_import_reported(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'alpha.py').write_text('import beta\n', encoding='utf-8')
    (tmp_path / 'scripts' / 'beta.py').write_text('', encoding='utf-8')
    _inventory(tmp_path, 'alpha,,scripts/alpha.py,stage,0,\nbeta,,scripts/beta.py,oneoff,0,\n')
    problems = lint.check_oneoff_imports(tmp_path)
    assert any('oneoff' in p for p in problems)
