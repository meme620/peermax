"""命名与实验文档规范检查（CLI：python -m peerlab lint）。"""

from __future__ import annotations

import pathlib
import re
import subprocess

from . import registry

ROOT = pathlib.Path(__file__).resolve().parents[1]

SECTIONS = ('目的', '参数', 'Baseline', '实现细节', '输入与前置', '产物与验收', '耗时与资源', '已知坑')
STAGE_CODE = re.compile(r'(?:^|-)(?:s[1-4]|s25|s3a|m1|p[23]b?)(?:-|$)')
SNAKE = re.compile(r'^(?:__[a-z]+__|[a-z][a-z0-9_]*)$')
NUMBER_PREFIX = re.compile(r'^\d+_')


def check_experiments(root: pathlib.Path, experiments) -> list:
    problems = []
    base = root / 'experiments'
    if not base.is_dir():
        return ['缺少 experiments/ 目录']
    wanted = {exp.id for exp in experiments}
    for exp in experiments:
        readme = base / exp.id / 'README.md'
        if not readme.is_file():
            problems.append(f'缺文档 experiments/{exp.id}/README.md')
            continue
        text = readme.read_text(encoding='utf-8')
        missing = [s for s in SECTIONS if f'## {s}' not in text]
        if missing:
            problems.append(f'experiments/{exp.id}/README.md 缺小节：' + '、'.join(missing))
    for path in sorted(p for p in base.iterdir() if p.is_dir()):
        if path.name not in wanted:
            problems.append(f'experiments/{path.name}/ 不在注册表里')
    return problems


def check_ids(experiments) -> list:
    problems = []
    for exp in experiments:
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', exp.id):
            problems.append(f'实验 id {exp.id!r} 不是 kebab-case 全小写')
        if STAGE_CODE.search(exp.id):
            problems.append(f'实验 id {exp.id!r} 含阶段代号（s1/s2/s3/s4/m1/p2/p3）')
    return problems


def check_new_files(root: pathlib.Path) -> list:
    problems = []
    out = subprocess.run(['git', 'ls-files', '*.py'], cwd=root,
                         capture_output=True, text=True)
    for rel in out.stdout.splitlines():
        if rel.startswith(('scripts/', 'tests/')):
            continue
        stem = pathlib.Path(rel).stem
        if not SNAKE.match(stem):
            problems.append(f'{rel}: 文件名不是 snake_case')
        elif re.search(r'(?<!_v)\d$', stem):
            problems.append(f'{rel}: 文件名以裸数字结尾（版本请用 _v2）')
    return problems


def check_dirs(root: pathlib.Path) -> list:
    problems = []
    candidates = [p for p in root.iterdir() if p.is_dir()]
    experiments = root / 'experiments'
    if experiments.is_dir():
        candidates += [p for p in experiments.iterdir() if p.is_dir()]
    for path in sorted(candidates):
        if NUMBER_PREFIX.match(path.name):
            problems.append(f'{path.relative_to(root)}: 目录带数字前缀')
    return problems


def run(root: pathlib.Path = ROOT, experiments=None) -> tuple:
    experiments = experiments or registry.EXPERIMENTS
    problems = (check_experiments(root, experiments)
                + check_ids(experiments)
                + check_new_files(root)
                + check_dirs(root))
    return (0 if not problems else 1), problems
