"""运行单个实验：就绪判定、命令组装、子进程、日志、产物收集与索引。"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import registry

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / 'result'
WATCH_DIRS = ('reports/data_gates/G0', 'data/processed', 'artifacts')
MAX_COPY_BYTES = 20 * 1024 * 1024


def now_stamp(now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    return moment.strftime('%Y%m%dT%H%M%SZ')


def make_run_id(exp_id: str, seed: int, stamp: str | None = None) -> str:
    return f'{exp_id}--s{seed}--{stamp or now_stamp()}'


def git_commit(root: Path = ROOT) -> str:
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or 'unknown'
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def suffix_for(exp: registry.Experiment, seed: int) -> str:
    if exp.suffix_arg and exp.scope == 'per_seed':
        return f'_s{seed}'
    return ''


def build_command(exp: registry.Experiment, step: registry.Step, seed: int,
                  smoke: bool = False) -> list[str]:
    args = step.args_smoke if (smoke and step.args_smoke is not None) else step.args
    sfx = suffix_for(exp, seed)
    args = [a.replace('{sfx}', sfx).replace('{seed}', str(seed)) for a in args]
    cmd = [sys.executable, '-m', 'peerlab.bootstrap', step.script, *args]
    if exp.accepts_seed:
        cmd += ['--seed', str(seed)]
    if exp.suffix_arg and exp.scope == 'per_seed':
        cmd += [exp.suffix_arg, suffix_for(exp, seed)]
    return cmd


def env_for(seed: int) -> dict[str, str]:
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = str(seed)
    env['PEERLAB_SEED'] = str(seed)
    return env


def readiness(exp: registry.Experiment, root: Path = ROOT, seed: int = 0) -> dict:
    missing = exp.missing_inputs(root, seed)
    return {'ready': not missing, 'missing': missing}


def snapshot(root: Path, dirs: tuple[str, ...] = WATCH_DIRS) -> dict[str, tuple[int, int]]:
    state: dict[str, tuple[int, int]] = {}
    for rel_dir in dirs:
        base = root / rel_dir
        if not base.is_dir():
            continue
        for path in base.rglob('*'):
            if path.is_file():
                try:
                    stat = path.stat()
                except OSError:
                    continue
                state[path.relative_to(root).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return state


def changed_files(before: dict[str, tuple[int, int]],
                  after: dict[str, tuple[int, int]]) -> list[str]:
    return sorted(
        rel for rel, sig in after.items()
        if before.get(rel) != sig
    )


def matches(rel: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern.endswith('/**'):
            prefix = pattern[:-3].rstrip('/')
            if rel == prefix or rel.startswith(prefix + '/'):
                return True
        elif fnmatch.fnmatchcase(rel, pattern):
            return True
    return False


def collect(run_dir: Path, changed: list[str], patterns: tuple[str, ...],
            root: Path = ROOT) -> list[dict]:
    collected: list[dict] = []
    dest_root = run_dir / 'outputs'
    for rel in changed:
        if not matches(rel, patterns):
            continue
        src = root / rel
        size = src.stat().st_size
        if size > MAX_COPY_BYTES:
            collected.append({'path': rel, 'bytes': size, 'copied': False,
                              'reason': 'over-20MB'})
            continue
        dest = dest_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        collected.append({'path': rel, 'bytes': size, 'copied': True})
    return collected


def _write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def append_index(result_dir: Path, record: dict) -> None:
    index_path = result_dir / 'index.json'
    doc = {'runs': []}
    if index_path.is_file():
        doc = json.loads(index_path.read_text(encoding='utf-8'))
        doc.setdefault('runs', [])
    doc['runs'].append(record)
    _write_json(index_path, doc)


def run_experiment(exp: registry.Experiment, seed: int, *, dry_run: bool = False,
                   smoke: bool = False, root: Path = ROOT,
                   result_dir: Path | None = None) -> dict:
    root = Path(root)
    result_dir = Path(result_dir) if result_dir else root / 'result'
    check = readiness(exp, root, seed)
    commands = [build_command(exp, step, seed, smoke=smoke) for step in exp.steps]
    if not check['ready']:
        return {'status': 'blocked', 'exp': exp.id, 'seed': seed,
                'missing': check['missing'], 'commands': commands}
    if dry_run:
        return {'status': 'dry-run', 'exp': exp.id, 'seed': seed,
                'missing': [], 'commands': commands}

    run_id = make_run_id(exp.id, seed)
    run_dir = result_dir / ('smoke' if smoke else 'runs') / run_id
    (run_dir / 'logs').mkdir(parents=True, exist_ok=True)

    before = snapshot(root)
    started = datetime.now(timezone.utc)
    steps_log: list[dict] = []
    status = 'ok'
    for idx, command in enumerate(commands, start=1):
        t0 = time.time()
        proc = subprocess.run(command, cwd=root, env=env_for(seed),
                              capture_output=True, text=True)
        elapsed = round(time.time() - t0, 2)
        (run_dir / 'logs' / f'{idx:02d}.stdout.log').write_text(proc.stdout, encoding='utf-8')
        (run_dir / 'logs' / f'{idx:02d}.stderr.log').write_text(proc.stderr, encoding='utf-8')
        steps_log.append({'command': command, 'returncode': proc.returncode,
                          'seconds': elapsed, 'log': f'logs/{idx:02d}.stdout.log'})
        if proc.returncode != 0:
            status = 'failed'
            break
    after = snapshot(root)
    collected = collect(run_dir, changed_files(before, after), exp.expanded(seed).outputs, root)
    finished = datetime.now(timezone.utc)

    record = {
        'run_id': run_id,
        'exp': exp.id,
        'title': exp.title,
        'seed': seed,
        'status': status,
        'smoke': smoke,
        'started_at': started.isoformat(timespec='seconds'),
        'finished_at': finished.isoformat(timespec='seconds'),
        'seconds': round((finished - started).total_seconds(), 2),
        'git_commit': git_commit(root),
        'steps': steps_log,
        'outputs': collected,
    }
    _write_json(run_dir / 'run.json', record)
    if not smoke:
        append_index(result_dir, {k: record[k] for k in
                                  ('run_id', 'exp', 'seed', 'status', 'smoke',
                                   'started_at', 'seconds', 'git_commit')})
    record['run_dir'] = str(run_dir)
    return record
