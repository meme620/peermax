"""数据门按路径载入生产者脚本：脚本目录必须在搜索路径上。

gates._script 用 importlib 在进程内载入 scripts/<name>.py 取规则函数；脚本顶层 import
同级模块（pricing_sample_build → pricing_lib）。缺这一条时门从 blocked 变成
ModuleNotFoundError 崩，而它只在产物齐备、门真的走到规则函数时才暴露。
"""
import subprocess
import sys
from pathlib import Path

from peerlab import gates

ROOT = Path(__file__).resolve().parents[1]


def test_loader_puts_scripts_on_path_and_imports_sibling_module():
    module = gates._script('pricing_sample_build')
    assert callable(module.compute_actual_sample_window)
    assert str(ROOT / 'scripts') in sys.path


def test_gate_cli_reaches_rule_function_instead_of_crashing():
    """子进程跑真实 CLI：崩在 import 时退出码非 0 且 stderr 有 ModuleNotFoundError。"""
    proc = subprocess.run([sys.executable, '-m', 'peerlab', 'gate', '--only', 'sample-metadata'],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert 'ModuleNotFoundError' not in proc.stderr, proc.stderr[-2000:]
    assert proc.returncode in (0, 1, 2), (proc.returncode, proc.stdout[-2000:])
