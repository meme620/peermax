from peerlab import imports


def test_imports_clean_removes_files_written_during_import(tmp_path):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'reports/data_gates/G0').mkdir(parents=True)
    (tmp_path / 'scripts/writer.py').write_text(
        'from pathlib import Path' + chr(10) +
        "Path('reports/data_gates/G0/probe.json').write_text('{}')" + chr(10),
        encoding='utf-8',
    )
    code = imports.run(tmp_path, jobs=1, clean=True)
    assert code == 0
    assert not (tmp_path / 'reports/data_gates/G0/probe.json').exists()


def test_classify_config_when_registry_missing():
    stderr = '[REGISTRY] registry 不存在: configs/parameter_registry.yaml'
    assert imports.classify(2, stderr) == 'config'


def test_classify_dependency_and_missing():
    assert imports.classify(1, "ModuleNotFoundError: No module named 'yaml'") == 'dep'
    assert imports.classify(1, 'FileNotFoundError: data/x.csv') == 'missing'
    assert imports.classify(1, 'OSError: Cannot save file into a non-existent directory') == 'missing'
    assert imports.classify(1, 'RuntimeError: 未知') == 'other'
    assert imports.classify(0, '') == 'ok'
