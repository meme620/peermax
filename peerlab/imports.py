"""逐脚本导入冒烟（CLI：python -m peerlab smoke --imports）。"""

from __future__ import annotations

import concurrent.futures as futures
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CHILD = '''
import importlib.util, sys, pathlib
path = sys.argv[1]
sys.path.insert(0, str(pathlib.Path(path).resolve().parent))
spec = importlib.util.spec_from_file_location('_smoke_target', path)
if spec is None or spec.loader is None:
    raise SystemExit('spec 无法加载')
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


def run_one(path: pathlib.Path, root: pathlib.Path, timeout: int = 180):
    try:
        proc = subprocess.run(
            [sys.executable, '-c', CHILD, str(path)],
            cwd=root, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return str(path.relative_to(root)), 'other', f'超时 {timeout}s'
    kind = classify(proc.returncode, proc.stderr)
    detail = ''
    if kind in {'dep', 'other'}:
        tail = proc.stderr.strip().splitlines()
        detail = tail[-1] if tail else f'rc={proc.returncode}'
    return str(path.relative_to(root)), kind, detail


def targets(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in (root / 'scripts').rglob('*.py') if '__pycache__' not in p.parts)


def run(root: pathlib.Path = ROOT, jobs: int = 8, timeout: int = 180) -> int:
    files = targets(root)
    results = []
    with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for item in pool.map(lambda p: run_one(p, root, timeout), files):
            results.append(item)
    buckets = {}
    for rel, kind, detail in results:
        buckets.setdefault(kind, []).append((rel, kind, detail))
    print(f'扫描 {len(files)} 个脚本')
    for kind in ('ok', 'config', 'missing', 'dep', 'other'):
        print(f'  {kind:<8} {len(buckets.get(kind, []))}')
    for kind in ('dep', 'other'):
        for rel, _, detail in buckets.get(kind, []):
            print(f'  [{kind}] {rel}: {detail}')
    failures = len(buckets.get('dep', [])) + len(buckets.get('other', []))
    print(chr(10) + '结论:', '通过' if failures == 0 else f'{failures} 个真失败')
    return 0 if failures == 0 else 1
