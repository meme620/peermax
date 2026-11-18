"""命名与结构规范检查（CLI：python -m peerlab lint）。"""

from __future__ import annotations

import collections
import pathlib
import re
import subprocess

from . import registry

ROOT = pathlib.Path(__file__).resolve().parents[1]

SECTIONS = ('目的', '参数', 'Baseline', '实现细节', '输入与前置', '产物与验收', '耗时与资源', '已知坑')
STAGE_CODE = re.compile(r'(?<![a-z0-9])(?:s[1-4]|s25|s3a|s3b|m[1-3]|p[1-4]b?)(?![a-z0-9])', re.I)
NUMBER_PREFIX = re.compile(r'^\d+_')
SNAKE = re.compile(r'^[a-z][a-z0-9_]*$')
MAX_DEPTH = 2
ROOT_FILES = {'Makefile', 'README.md', 'requirements.txt', 'pytest.ini'}


def check_experiments(root: pathlib.Path, experiments) -> list:
    """docs/experiments.md 是唯一实验文档：每个注册 id 一个 ## 块，块内八节齐全。"""
    doc = root / 'docs' / 'experiments.md'
    if not doc.is_file():
        return ['缺少 docs/experiments.md']
    text = doc.read_text(encoding='utf-8')
    blocks = {}
    for chunk in re.split(r'(?m)^## ', text)[1:]:
        title = chunk.splitlines()[0].strip()
        blocks[title] = chunk
    problems = []
    for exp in experiments:
        block = blocks.get(exp.id)
        if block is None:
            problems.append(f'docs/experiments.md 缺 {exp.id} 一节')
            continue
        missing = [s for s in SECTIONS if f'### {s}' not in block]
        if missing:
            problems.append(f'docs/experiments.md 的 {exp.id} 缺小节：' + '、'.join(missing))
    wanted = {exp.id for exp in experiments}
    for title in blocks:
        if title not in wanted:
            problems.append(f'docs/experiments.md 的 {title!r} 不在注册表里')
    return problems


def check_ids(experiments) -> list:
    problems = []
    for exp in experiments:
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', exp.id):
            problems.append(f'实验 id {exp.id!r} 不是 kebab-case 全小写')
        if STAGE_CODE.search(exp.id):
            problems.append(f'实验 id {exp.id!r} 含阶段代号（s1/s2/s3/s4/m1/p2/p3）')
    return problems


def check_path_names(paths) -> list:
    """paths 为相对仓库根的路径串；检查阶段代号、大小写、数字前缀、层级与单文件目录。"""
    problems = []
    for rel in paths:
        pure = pathlib.PurePosixPath(rel)
        for part in pure.parts[:-1]:
            if STAGE_CODE.search(part):
                problems.append(f'{rel}: 目录名含阶段代号 {part!r}')
            if NUMBER_PREFIX.match(part):
                problems.append(f'{rel}: 目录名带数字前缀 {part!r}')
        if STAGE_CODE.search(pure.stem):
            problems.append(f'{rel}: 文件名含阶段代号')
        if NUMBER_PREFIX.match(pure.stem):
            problems.append(f'{rel}: 文件名带数字前缀')
        if pure.suffix in {'.py', '.sh', '.md'} and pure.name not in ROOT_FILES:
            if pure.stem.startswith('__') and pure.stem.endswith('__'):
                pass
            elif not SNAKE.match(pure.stem):
                problems.append(f'{rel}: 文件名不是 snake_case')
            elif re.search(r'(?<!_v)\d$', pure.stem) and not re.search(r'_batch\d+$', pure.stem):
                problems.append(f'{rel}: 文件名以裸数字结尾（版本请用 _v2）')
        if len(pure.parts) - 1 > MAX_DEPTH:
            problems.append(f'{rel}: 目录层级超过 {MAX_DEPTH} 层')
    counts = collections.Counter(str(pathlib.PurePosixPath(p).parent) for p in paths)
    for directory, count in sorted(counts.items()):
        if directory == '.' or count != 1:
            continue
        if directory == 'data' or directory.startswith('data/'):
            continue  # 数据目录按载荷存放，允许只放一个参照文件
        problems.append(f'{directory}/: 目录只装一个文件')
    return problems


def check_paths(root: pathlib.Path) -> list:
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, text=True)
    paths = [line for line in out.stdout.split('\0') if line]
    return check_path_names(paths)


def check_local_imports(root: pathlib.Path) -> list:
    import ast

    local = {p.stem for p in (root / 'scripts').glob('*.py')}
    problems = []
    for base in ('scripts', 'peerlab', 'tests'):
        for path in sorted((root / base).rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for name in names:
                    first = name.split('.')[0]
                    if STAGE_CODE.search(first):
                        problems.append(f'{rel}: 残留阶段代号模块 {first}')
                    elif first in local and not (root / 'scripts' / (first + '.py')).is_file():
                        problems.append(f'{rel}: 本地模块 {first} 找不到对应文件')
    return problems


def check_script_sources(root: pathlib.Path) -> list:
    """scripts/ 已拍平到一层：再出现两层相对回退或 parents[2] 就是把仓库根算错。"""
    problems = []
    for path in sorted((root / 'scripts').glob('*.py')):
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding='utf-8')
        for pattern, why in (('"..", ".."', '两层相对回退'), ("'..', '..'", '两层相对回退'),
                             ('parents[2]', 'parents[2] 定位')):
            if pattern in text:
                problems.append(f'{rel}: 含 {pattern}（{why}；scripts/ 已平铺，父目录只有一层）')
    return problems


def run(root: pathlib.Path = ROOT, experiments=None) -> tuple:
    experiments = experiments or registry.EXPERIMENTS
    problems = (check_experiments(root, experiments)
                + check_ids(experiments)
                + check_paths(root)
                + check_local_imports(root)
                + check_script_sources(root))
    return (0 if not problems else 1), problems
