"""逐个导入 scripts/ 下每个 .py，把"缺依赖"与"预期缺配置"分开。

用法：python tools/smoke_import.py [--jobs N]

判读：
  ok      导入成功
  config  因 configs/ 缺失被 fail-closed 拦下（SystemExit 2 / [REGISTRY]）
  missing 因数据或报告目录缺失抛 FileNotFoundError（预期）
  dep     ModuleNotFoundError —— 依赖没装，真失败
  other   其它异常 —— 真失败
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CHILD = '''
import importlib.util, sys, pathlib
path = sys.argv[1]
# 真实执行时 sys.path[0] 就是脚本所在目录，同级 import 依赖它
sys.path.insert(0, str(pathlib.Path(path).resolve().parent))
spec = importlib.util.spec_from_file_location("_smoke_target", path)
if spec is None or spec.loader is None:
    raise SystemExit("spec 无法加载")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
'''


def classify(returncode: int, stderr: str) -> str:
    if returncode == 0:
        return 'ok'
    if '[REGISTRY]' in stderr or (returncode == 2 and 'SystemExit' in stderr):
        return 'config'
    if 'ModuleNotFoundError' in stderr or 'ImportError' in stderr:
        return 'dep'
    if 'FileNotFoundError' in stderr or 'non-existent directory' in stderr:
        return 'missing'
    return 'other'


def run_one(path: pathlib.Path, timeout: int) -> tuple[str, str, str]:
    try:
        proc = subprocess.run(
            [sys.executable, '-c', CHILD, str(path)],
            cwd=ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return str(path.relative_to(ROOT)), 'other', f'超时 {timeout}s'
    kind = classify(proc.returncode, proc.stderr)
    detail = ''
    if kind in {'dep', 'other'}:
        tail = proc.stderr.strip().splitlines()
        detail = tail[-1] if tail else f'rc={proc.returncode}'
    return str(path.relative_to(ROOT)), kind, detail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='逐脚本导入冒烟测试')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args(argv)

    targets = sorted(p for p in (ROOT / 'scripts').rglob('*.py') if '__pycache__' not in p.parts)
    results: list[tuple[str, str, str]] = []
    with futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for item in pool.map(lambda p: run_one(p, args.timeout), targets):
            results.append(item)

    buckets: dict[str, list[tuple[str, str, str]]] = {}
    for rel, kind, detail in results:
        buckets.setdefault(kind, []).append((rel, kind, detail))

    print(f'扫描 {len(targets)} 个脚本')
    for kind in ('ok', 'config', 'missing', 'dep', 'other'):
        print(f'  {kind:<8} {len(buckets.get(kind, []))}')
    for kind in ('dep', 'other'):
        for rel, _, detail in buckets.get(kind, []):
            print(f'  [{kind}] {rel}: {detail}')

    failures = len(buckets.get('dep', [])) + len(buckets.get('other', []))
    print('\n结论:', '通过' if failures == 0 else f'{failures} 个真失败')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
