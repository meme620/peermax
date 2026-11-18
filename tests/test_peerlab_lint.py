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
