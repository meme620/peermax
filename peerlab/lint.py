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
        config_exception = pure.parts[0] == 'configs' and pure.suffix in {'.yaml', '.yml'}
        if STAGE_CODE.search(pure.stem) and not config_exception:
            problems.append(f'{rel}: 文件名含阶段代号')
        if NUMBER_PREFIX.match(pure.stem):
            problems.append(f'{rel}: 文件名带数字前缀')
        if pure.suffix in {'.py', '.sh', '.md'} and pure.name not in ROOT_FILES:
            if pure.stem.startswith('__') and pure.stem.endswith('__'):
                pass
            elif not SNAKE.match(pure.stem):
                problems.append(f'{rel}: 文件名不是 snake_case')
            elif re.search(r'(?<!_v)\d$', pure.stem) and not re.search(r'_batch\d+$', pure.stem):
                problems.append(f'{rel}: 文件名以裸数字结尾')
        if re.search(r'_v\d+$', pure.stem):
            problems.append(f'{rel}: 文件名带版本号后缀（口径版本只进产物名与 network_version）')
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


SCRIPT_LITERAL = re.compile(r'scripts/([a-z0-9_]+)\.py')
ALLOWED_ROLES = {'library', 'stage', 'analysis', 'support', 'oneoff', 'deliverable', 'external'}
TEXT_SUFFIXES = {'.py', '.md', '.csv', '.yaml', '.yml', '.sh'}


def check_script_literals(root: pathlib.Path, paths=None) -> list:
    """代码、文档、清单里出现的 scripts/<name>.py 必须真的存在。

    删除或改名脚本后，注册表、文档、gate2 清单里的字面量会静默指向空气；
    这一条把断链变成 lint 失败。tests/ 跳过：测试会往 tmp_path 里合成同形 fixture。
    """
    if paths is None:
        out = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, text=True)
        paths = [line for line in out.stdout.split('\0') if line]
    existing = {p.stem for p in (root / 'scripts').glob('*.py')}
    problems = []
    seen = set()
    for rel in paths:
        if rel.startswith('tests/'):
            continue
        if pathlib.PurePosixPath(rel).suffix not in TEXT_SUFFIXES and rel != 'Makefile':
            continue
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for match in SCRIPT_LITERAL.finditer(text):
            name = match.group(1)
            if name not in existing and (rel, name) not in seen:
                seen.add((rel, name))
                problems.append(f'{rel}: 引用不存在的 scripts/{name}.py')
    return problems


def check_inventory(root: pathlib.Path) -> list:
    """docs/script_inventory.csv 与 scripts/ 双向一致；role 白名单；test_ref 必须存在。"""
    import csv

    path = root / 'docs' / 'script_inventory.csv'
    if not path.is_file():
        return ['缺少 docs/script_inventory.csv']
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    scripts = {p.stem for p in (root / 'scripts').glob('*.py')}
    problems = []
    seen = set()
    for row in rows:
        name = (row.get('name') or '').strip()
        rel = (row.get('path') or '').strip()
        role = (row.get('role') or '').strip()
        if name in seen:
            problems.append(f'script_inventory.csv: {name} 重复登记')
        seen.add(name)
        if name not in scripts:
            problems.append(f'script_inventory.csv: {name} 没有对应脚本')
        elif rel != f'scripts/{name}.py':
            problems.append(f'script_inventory.csv: {name} 的 path 应为 scripts/{name}.py（现 {rel!r}）')
        if role not in ALLOWED_ROLES:
            problems.append(f'script_inventory.csv: {name} 的 role {role!r} 不在白名单')
        for test in (row.get('test_ref') or '').split(';'):
            test = test.strip()
            if test and not (root / 'tests' / test).is_file():
                problems.append(f'script_inventory.csv: {name} 的 test_ref {test} 不存在')
    for name in sorted(scripts - seen):
        problems.append(f'script_inventory.csv: 缺 {name} 的一行（新脚本要登记）')
    return problems


def check_oneoff_imports(root: pathlib.Path) -> list:
    """非 oneoff 脚本不得 import oneoff 脚本：证据脚本不能变成主链依赖。"""
    import ast
    import csv

    path = root / 'docs' / 'script_inventory.csv'
    if not path.is_file():
        return []
    with open(path, encoding='utf-8-sig', newline='') as f:
        roles = {row['name']: row['role'] for row in csv.DictReader(f)}
    problems = []
    for script in sorted((root / 'scripts').glob('*.py')):
        if roles.get(script.stem) == 'oneoff':
            continue
        tree = ast.parse(script.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split('.')[0]]
            for name in names:
                if roles.get(name) == 'oneoff':
                    problems.append(f'scripts/{script.stem}.py: import 了 oneoff 脚本 {name}')
    return problems


def check_text_io_encoding(root: pathlib.Path) -> list:
    """文本模式 open() 与 Path.read_text/write_text 必须显式 encoding。

    不写 encoding 就吃系统 locale：macOS 是 UTF-8、Windows 是 cp936，同一脚本跨平台
    读同一文件会解出不同字节。二进制模式（mode 含 b）不受此约束。
    """
    import ast

    problems = []
    for base in ('scripts', 'peerlab', 'tests'):
        for path in sorted((root / base).glob('*.py')):
            rel = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name) and func.id == 'open':
                    mode = None
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                        mode = node.args[1].value
                    for kw in node.keywords:
                        if kw.arg == 'mode' and isinstance(kw.value, ast.Constant):
                            mode = kw.value.value
                    if mode and 'b' in str(mode):
                        continue
                    if not any(kw.arg == 'encoding' for kw in node.keywords):
                        problems.append(f'{rel}:{node.lineno}: open() 未指定 encoding')
                elif isinstance(func, ast.Attribute) and func.attr in {'read_text', 'write_text'}:
                    if not any(kw.arg == 'encoding' for kw in node.keywords):
                        problems.append(f'{rel}:{node.lineno}: {func.attr}() 未指定 encoding')
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
                + check_script_sources(root)
                + check_text_io_encoding(root)
                + check_script_literals(root)
                + check_inventory(root)
                + check_oneoff_imports(root))
    return (0 if not problems else 1), problems
