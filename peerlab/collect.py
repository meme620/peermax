"""指标抽取与结果聚合：runs -> metrics.csv -> tables/ -> summary.md。"""

from __future__ import annotations

import csv
import fnmatch
import json
import statistics
from pathlib import Path

from . import registry, runner

MISSING = object()


def dotted_get(doc, path: str):
    cur = doc
    for part in path.split('.'):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return MISSING
    return cur


def _find_output(run_dir: Path, pattern: str, exp: registry.Experiment, seed: int):
    sfx = runner.suffix_for(exp, seed)
    target = pattern.replace('{sfx}', sfx)
    base = run_dir / 'outputs'
    if not base.is_dir():
        return None
    for path in sorted(base.rglob('*')):
        if path.is_file() and fnmatch.fnmatchcase(path.relative_to(base).as_posix(), target):
            return path
    return None


def extract_metrics(run_dir: Path, exp: registry.Experiment, seed: int) -> dict:
    values = {}
    for spec in exp.metrics:
        path = _find_output(run_dir, spec.file_glob, exp, seed)
        if path is None:
            values[spec.name] = None
            continue
        try:
            doc = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            values[spec.name] = None
            continue
        raw = dotted_get(doc, spec.path)
        values[spec.name] = None if raw is MISSING or raw is None else raw
    return values


def load_index(result_dir: Path) -> list:
    index_path = result_dir / 'index.json'
    if not index_path.is_file():
        return []
    return json.loads(index_path.read_text(encoding='utf-8')).get('runs', [])


def summarise(values: list) -> dict:
    present = [v for v in values if v is not None]
    mean = statistics.fmean(present) if present else None
    sd = statistics.stdev(present) if len(present) > 1 else (0.0 if present else None)
    identical = bool(present) and len(present) == len(values) and len(set(present)) == 1
    return {'mean': mean, 'sd': sd, 'identical': identical}


def build_tables(result_dir: Path, experiments, seeds=registry.SEEDS) -> list:
    runs = [r for r in load_index(result_dir) if r.get('status') == 'ok' and not r.get('smoke')]
    written = []
    table_dir = result_dir / 'tables'
    table_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    for exp in experiments:
        if not exp.metrics:
            continue
        per_run = [r for r in runs if r['exp'] == exp.id]
        if not per_run:
            continue
        by_seed = {}
        for record in per_run:
            run_dir = result_dir / 'runs' / record['run_id']
            if run_dir.is_dir():
                by_seed[record['seed']] = extract_metrics(run_dir, exp, record['seed'])
        if not by_seed:
            continue

        lines = ['metric,' + ','.join('s' + str(s) for s in seeds) + ',mean,sd,identical']
        for spec in exp.metrics:
            vals = [by_seed.get(s, {}).get(spec.name) for s in seeds]
            stats = summarise(vals)
            for seed, value in zip(seeds, vals):
                metric_rows.append({'run_id': exp.id + '--s' + str(seed), 'exp': exp.id,
                                    'seed': seed, 'metric': spec.name, 'value': value})
            cells = ','.join('' if v is None else repr(v) for v in vals)
            mean = '' if stats['mean'] is None else repr(stats['mean'])
            sd = '' if stats['sd'] is None else repr(stats['sd'])
            lines.append(','.join([spec.name, cells, mean, sd, str(stats['identical'])]))
        path = table_dir / (exp.id + '.csv')
        path.write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
        written.append(path)

    if metric_rows:
        metrics_path = result_dir / 'metrics.csv'
        with metrics_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['run_id', 'exp', 'seed', 'metric', 'value'])
            writer.writeheader()
            writer.writerows(metric_rows)
        written.append(metrics_path)
    return written


def schema_probe(result_dir: Path, experiments) -> 'Path | None':
    runs = [r for r in load_index(result_dir) if r.get('status') == 'ok' and not r.get('smoke')]
    seen = {}
    for record in runs:
        outputs = result_dir / 'runs' / record['run_id'] / 'outputs'
        if not outputs.is_dir():
            continue
        for path in sorted(outputs.rglob('*.json')):
            rel = path.relative_to(outputs).as_posix()
            try:
                doc = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(doc, dict):
                seen.setdefault(rel, sorted(doc.keys()))
    if not seen:
        return None
    lines = ['# 产物 JSON 顶层键（首轮运行实测）', '']
    for rel, keys in sorted(seen.items()):
        lines.append('- ' + rel + '：' + ', '.join(keys))
    path = result_dir / 'tables' / '_schema_probe.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(chr(10).join(lines) + chr(10), encoding='utf-8')
    return path


def write_summary(result_dir: Path, experiments, seeds=registry.SEEDS) -> Path:
    runs = load_index(result_dir)
    ok_runs = [r for r in runs if r.get('status') == 'ok' and not r.get('smoke')]
    lines = ['# 结果摘要', '', '种子集合：' + str(tuple(seeds)) + '；成功运行 ' + str(len(ok_runs)) + ' 次。', '']
    if not ok_runs:
        lines.append('当前没有有效运行：实验在就绪检查阶段被拦下（缺 configs/ 或上游产物）。')
        lines.append('用 python -m peerlab status 查看每个实验缺什么。')
        lines.append('')
    for exp in experiments:
        exp_runs = [r for r in ok_runs if r['exp'] == exp.id]
        lines.append('## ' + exp.id + ' — ' + exp.title)
        lines.append('')
        if not exp_runs:
            lines.append('无成功运行。')
            lines.append('')
            continue
        seeds_seen = sorted({r['seed'] for r in exp_runs})
        lines.append('- 种子：' + str(seeds_seen))
        for spec in exp.metrics:
            vals = []
            for record in sorted(exp_runs, key=lambda r: r['seed']):
                run_dir = result_dir / 'runs' / record['run_id']
                value = extract_metrics(run_dir, exp, record['seed']).get(spec.name)
                if value is not None:
                    vals.append(value)
            stats = summarise(vals) if vals else {'mean': None, 'sd': None, 'identical': False}
            if stats['mean'] is None:
                lines.append('- ' + spec.name + '：未取到值（产物缺失或 spec 未校准）')
            elif stats['identical']:
                lines.append('- ' + spec.name + '：' + format(stats['mean'], '.6g') + '（' + str(len(vals)) + ' 个种子完全一致）')
            else:
                lines.append('- ' + spec.name + '：' + format(stats['mean'], '.6g') + ' ± ' + format(stats['sd'], '.3g'))
        lines.append('')
    path = result_dir / 'summary.md'
    path.write_text(chr(10).join(lines), encoding='utf-8')
    return path


def aggregate(result_dir: Path, experiments=registry.EXPERIMENTS,
              seeds=registry.SEEDS) -> list:
    written = build_tables(result_dir, experiments, seeds)
    probe = schema_probe(result_dir, experiments)
    if probe:
        written.append(probe)
    written.append(write_summary(result_dir, experiments, seeds))
    return written
