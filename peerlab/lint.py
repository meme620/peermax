"""命名与结构规范检查（CLI：python -m peerlab lint）。"""

from __future__ import annotations

import collections
import pathlib
import re
import subprocess

from . import registry

ROOT = pathlib.Path(__file__).resolve().parents[1]

STAGE_CODE = re.compile(r'(?<![a-z0-9])(?:s[1-4]|s25|s3a|s3b|m[1-3]|p[1-4]b?)(?![a-z0-9])', re.I)
NUMBER_PREFIX = re.compile(r'^\d+_')
SNAKE = re.compile(r'^[a-z][a-z0-9_]*$')
MAX_DEPTH = 2
ROOT_FILES = {'Makefile', 'README.md', 'AGENTS.md', 'requirements.txt', 'pytest.ini'}


def check_spec(root: pathlib.Path) -> list:
    """spec/*.yaml 与 registry 的 purpose/notes 必须可解析：引用脚本、gate、lint 检查、测试都要存在。"""
    from . import spec
    try:
        return spec.validate(root)
    except FileNotFoundError as exc:
        return [str(exc)]


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
        if pure.name == '.gitkeep':
            continue  # 占位文件不承载内容，不参与命名与层级约束
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
TEXT_SUFFIXES = {'.py', '.md', '.csv', '.yaml', '.yml', '.sh'}


def check_hygiene(root: pathlib.Path) -> list:
    """禁止回流的三种坏味道：脚本里 sys.path.insert、裸 except 吞异常、except Exception: pass。"""
    import ast

    problems = []
    # runpy 不会把脚本目录放进 sys.path[0]，bootstrap 必须自己插；其余脚本禁插。
    PATH_INSERT_EXEMPT = {'peerlab/bootstrap.py'}
    for base in ('scripts', 'peerlab'):
        for path in sorted((root / base).glob('*.py')):
            rel = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if (rel not in PATH_INSERT_EXEMPT and isinstance(node, ast.Call)
                        and ast.unparse(node.func) in {'sys.path.insert', '_sys.path.insert'}):
                    problems.append(f'{rel}:{node.lineno}: sys.path.insert（脚本目录已在 sys.path[0]）')
                if isinstance(node, ast.ExceptHandler):
                    if node.type is None:
                        problems.append(f'{rel}:{node.lineno}: 裸 except')
                    elif isinstance(node.type, ast.Name) and node.type.id == 'Exception':
                        if len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue)):
                            problems.append(f'{rel}:{node.lineno}: except Exception 后直接 pass/continue（收窄异常类型）')
    return problems


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
    problems = (check_spec(root)
                + check_ids(experiments)
                + check_paths(root)
                + check_local_imports(root)
                + check_script_sources(root)
                + check_text_io_encoding(root)
                + check_hygiene(root)
                + check_script_literals(root))
    return (0 if not problems else 1), problems
