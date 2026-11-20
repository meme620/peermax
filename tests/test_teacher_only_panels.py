"""C7：样本只许老师语料两列（ma_teacher / mb_teacher）。

执行面两条：
  1. 三个面板键映射（PANELS / PANEL_PATHS_V4 / NET_DIRS）的键集合恰为两列；
  2. 计算链的四个文件里不得出现自建语料产物路径（monthly_pricing_panel_ma.csv 等）。
自建链的产物不在本仓计算链内；键位一回来，check_panel_contracts 会遍历到不存在的
文件抛 FileNotFoundError，把 estimation-sample 卡在就绪判定。
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peerlab import lint  # noqa: E402

EXPECTED = {"ma_teacher", "mb_teacher"}
CHAIN_FILES = ["scripts/pricing_lib.py", "scripts/pricing_sample_build.py",
               "configs/pricing_inputs.yaml", "configs/version_manifest.yaml"]


def _literal_dict_keys(path: Path, name: str):
    """用 AST 取字面量 dict 的键集合（不 import，避免导入期 registry 校验）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            if isinstance(node.value, ast.Dict):
                return {k.value for k in node.value.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return None


def test_panel_key_sets_are_teacher_only():
    path = ROOT / "scripts/pricing_lib.py"
    for name in ("PANELS", "PANEL_PATHS_V4", "NET_DIRS"):
        keys = _literal_dict_keys(path, name)
        assert keys is not None, f"pricing_lib.py 找不到 {name} 的字面量定义"
        assert keys == EXPECTED, f"{name} 键集合 {sorted(keys)} != {sorted(EXPECTED)}"


def test_chain_files_have_no_self_corpus_paths():
    for rel in CHAIN_FILES:
        p = ROOT / rel
        assert p.is_file(), f"{rel}: 文件缺失"
        text = p.read_text(encoding="utf-8", errors="ignore")
        for token in lint.SELF_CORPUS_PATHS:
            assert token not in text, f"{rel}: 出现自建语料产物路径 {token!r}（C7）"


def test_lint_rule_passes_on_current_tree():
    assert lint.check_teacher_only_panels(ROOT) == []


def test_lint_rule_flags_injected_key(tmp_path):
    """反例：把自建语料键放回去，规则必须报出来。"""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    src = (ROOT / "scripts/pricing_lib.py").read_text(encoding="utf-8")
    bad = src.replace("NET_DIRS = {\n    'ma_teacher'",
                      "NET_DIRS = {\n    'ma': ROOT / 'data/processed/network_ma',\n    'ma_teacher'")
    assert bad != src, "注入失败：pricing_lib.py 的 NET_DIRS 形态变了"
    (tmp_path / "scripts/pricing_lib.py").write_text(bad, encoding="utf-8")
    for rel in ("scripts/pricing_sample_build.py", "configs/pricing_inputs.yaml",
                "configs/version_manifest.yaml"):
        (tmp_path / rel).write_text((ROOT / rel).read_text(encoding="utf-8"), encoding="utf-8")
    problems = lint.check_teacher_only_panels(tmp_path)
    assert any("NET_DIRS" in p or "network_ma" in p for p in problems), problems
