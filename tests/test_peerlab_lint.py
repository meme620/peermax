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


def test_registry_refs_cover_every_layer() -> None:
    design = {
        'sample': {'registry_keys': {'window': ['sample_start', 'sample_end']}},
        'pricing_gates': [{'id': 'P1', 'registry_key': None},
                     {'id': 'P2', 'registry_key': 'alpha_pricing'}],
    }
    assert sorted(spec._registry_refs(design)) == [
        ('design.pricing_gates[P2].registry_key', 'alpha_pricing'),
        ('design.sample.registry_keys.window', 'sample_end'),
        ('design.sample.registry_keys.window', 'sample_start'),
    ]


def test_unknown_registry_key_flagged() -> None:
    design = {'iqr': {'registry_keys': {'value': 'no_such_param'}}}
    assert spec.check_registry_refs(design, {'iqr_pool_value'}) == [
        'design.iqr.registry_keys.value: registry 无参数 no_such_param']


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

def test_config_yaml_stage_code_rejected() -> None:
    problems = lint.check_path_names(['configs/s3a_final_spec.yaml'])
    assert any('阶段代号' in p for p in problems)


def test_clean_config_yaml_passes() -> None:
    problems = lint.check_path_names(['configs/pricing_inputs.yaml', 'configs/analyst_spec.yaml'])
    assert not any('阶段代号' in p or 'final' in p for p in problems)


def test_artifact_name_with_stage_code_rejected(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'a.py').write_text(
        "OUT = 'reports/G0/s3_m1_pricing_table.json'\n", encoding='utf-8')
    problems = lint.check_artifact_names(tmp_path, paths=['scripts/a.py'])
    assert any('阶段代号' in p for p in problems)


def test_artifact_name_with_final_rejected(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'a.py').write_text(
        "SPEC = ROOT / 'configs/analyst_final_spec.yaml'\n", encoding='utf-8')
    problems = lint.check_artifact_names(tmp_path, paths=['scripts/a.py'])
    assert any('final' in p for p in problems)


def test_clean_artifact_names_pass(tmp_path: Path) -> None:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts' / 'a.py').write_text(
        "OUT = 'reports/G0/pricing_table.json'\n"
        "X = final.stat\n"
        "Y = s3_status.items\n", encoding='utf-8')
    assert lint.check_artifact_names(tmp_path, paths=['scripts/a.py']) == []


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

# ---------------------------------------------------------------- undefined-name 规则
def _mk_script(root: Path, name: str, body: str) -> Path:
    d = root / 'scripts'
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(body, encoding='utf-8')
    return p


def test_undefined_name_call_flagged(tmp_path: Path) -> None:
    """批量改名事故的形状：调用 file_size() 但本模块未定义/未导入。"""
    _mk_script(tmp_path, 'a.py', 'def f():\n    return file_size()\n')
    problems = lint.check_undefined_names(tmp_path)
    assert any('file_size' in p for p in problems), problems


def test_defined_and_imported_names_pass(tmp_path: Path) -> None:
    _mk_script(tmp_path, 'b.py',
               'from os.path import join\n'
               'def helper():\n    return join("a", "b")\n'
               'def g():\n    return helper()\n')
    assert lint.check_undefined_names(tmp_path) == []


def test_builtins_and_args_not_flagged(tmp_path: Path) -> None:
    _mk_script(tmp_path, 'c.py',
               'def f(x, *args, **kwargs):\n'
               '    return len([x]) + int(1) + max(0, 1)\n')
    assert lint.check_undefined_names(tmp_path) == []


def test_class_and_local_names_not_flagged(tmp_path: Path) -> None:
    _mk_script(tmp_path, 'd.py',
               'class K:\n'
               '    def m(self):\n'
               '        return K()\n'
               'def f():\n'
               '    for i in range(2):\n'
               '        pass\n'
               '    with open("x") as fh:\n'
               '        pass\n'
               '    try:\n'
               '        pass\n'
               '    except ValueError as exc:\n'
               '        return K()\n')
    assert lint.check_undefined_names(tmp_path) == []


def test_attribute_call_not_confused_with_bare_name(tmp_path: Path) -> None:
    _mk_script(tmp_path, 'e.py', 'import os\ndef f():\n    return os.path.join("a")\n')
    assert lint.check_undefined_names(tmp_path) == []
