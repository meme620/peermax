from peerlab import imports


def test_classify_config_when_registry_missing():
    stderr = '[REGISTRY] registry 不存在: configs/parameter_registry.yaml'
    assert imports.classify(2, stderr) == 'config'


def test_classify_dependency_and_missing():
    assert imports.classify(1, "ModuleNotFoundError: No module named 'yaml'") == 'dep'
    assert imports.classify(1, 'FileNotFoundError: data/x.csv') == 'missing'
    assert imports.classify(1, "OSError: Cannot save file into a non-existent directory") == 'missing'
    assert imports.classify(1, 'RuntimeError: 未知') == 'other'
    assert imports.classify(0, '') == 'ok'
