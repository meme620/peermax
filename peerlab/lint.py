"""命名与结构规范检查（CLI：python -m peerlab lint）。"""

from __future__ import annotations

import ast
import builtins
import collections
import pathlib
import re
import subprocess
import sys

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
    # 按路径载入脚本的宿主必须自己把 scripts/ 插进搜索路径：bootstrap 走 runpy、
    # gates 走 importlib（进程内取生产者规则函数）。其余脚本禁插。
    PATH_INSERT_EXEMPT = {'peerlab/bootstrap.py', 'peerlab/gates.py'}
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


ARTIFACT_EXT = r'(?:json|md|csv|parquet|pptx|xlsx|yaml|yml|txt|log|pkl)'
ARTIFACT_TOKEN = re.compile(r'[A-Za-z0-9_{}\-]*[A-Za-z0-9_{}]\.' + ARTIFACT_EXT + r'\b')
FINAL_TOKEN = re.compile(r'(?i)(?:^|[^a-z0-9])final(?:[^a-z0-9]|$)')


def check_artifact_names(root: pathlib.Path, paths=None) -> list:
    """产物文件名字面量不得含阶段代号或 final。

    磁盘上最终叫什么由脚本里写的字符串决定，只查路径本身管不住产物名。
    tests/ 跳过：测试会往 tmp_path 合成同形 fixture。
    """
    if paths is None:
        out = subprocess.run(['git', 'ls-files', '-z'], cwd=root, capture_output=True, text=True)
        paths = [line for line in out.stdout.split('\0') if line]
    problems = []
    for rel in paths:
        if rel.startswith(('tests/', 'ref/', 'data/')):
            continue
        if pathlib.PurePosixPath(rel).suffix not in TEXT_SUFFIXES and rel != 'Makefile':
            continue
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for match in ARTIFACT_TOKEN.finditer(text):
            token = match.group(0)
            stem = token.rsplit('.', 1)[0]
            if stem.isupper() or len(stem) < 3:
                continue
            if STAGE_CODE.search(stem) or FINAL_TOKEN.search(stem):
                problems.append(f'{rel}: 产物名含阶段代号或 final {token!r}')
    return problems


SELF_CORPUS_PANEL_KEYS = ('ma', 'mb')

SELF_CORPUS_PATHS = ('monthly_pricing_panel_ma.csv', 'monthly_pricing_panel_mb.csv',
                     'monthly_pricing_panel_ma_v4.csv', 'monthly_pricing_panel_mb_v4.csv',
                     'network_ma/', 'network_mb/')


def check_teacher_only_panels(root: pathlib.Path) -> list:
    """样本只许老师语料两列（spec/design.yaml C7）：自建 CNINFO 语料键与路径不得回流。

    自建链的语料产物与网络目录都不在本仓计算链内；键位一回来，check_panel_contracts
    就会遍历到不存在的文件抛 FileNotFoundError，把 estimation-sample 卡在就绪判定。
    """
    problems = []
    targets = [root / 'scripts/pricing_lib.py', root / 'scripts/pricing_sample_build.py',
               root / 'configs/pricing_inputs.yaml', root / 'configs/version_manifest.yaml']
    for path in targets:
        if not path.is_file():
            problems.append(f'{path.relative_to(root).as_posix()}: 文件缺失')
            continue
        rel = path.relative_to(root).as_posix()
        text = path.read_text(encoding='utf-8', errors='ignore')
        for token in SELF_CORPUS_PATHS:
            if token in text:
                problems.append(f'{rel}: 出现自建语料产物路径 {token!r}（C7 只许老师语料两列）')
    return problems + _check_panel_key_sets(root)


def _check_panel_key_sets(root: pathlib.Path) -> list:
    """用 AST 读三个面板键映射的键集合，核对恰好是老师语料两列。

    不 import pricing_lib：导入即校验整个 registry 与输入合同，lint 会因数据缺失而假红。
    """
    expected = {'ma_teacher', 'mb_teacher'}
    problems = []
    path = root / 'scripts/pricing_lib.py'
    if not path.is_file():
        return ['scripts/pricing_lib.py: 文件缺失']
    tree = ast.parse(path.read_text(encoding='utf-8'))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ('PANELS', 'PANEL_PATHS_V4', 'NET_DIRS'):
                keys = set()
                if isinstance(node.value, ast.Dict):
                    for k in node.value.keys:
                        if isinstance(k, ast.Constant) and isinstance(k.value, str):
                            keys.add(k.value)
                found[target.id] = keys
    for name in ('PANELS', 'PANEL_PATHS_V4', 'NET_DIRS'):
        keys = found.get(name)
        if keys is None:
            problems.append(f'scripts/pricing_lib.py: 找不到 {name} 的字面量定义')
        elif keys != expected:
            problems.append(f'scripts/pricing_lib.py: {name} 键集合 {sorted(keys)} != '
                            f'老师语料两列 {sorted(expected)}（C7）')
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


_UNDEF_SKIP = {
    'self', 'cls', '__file__', '__name__', 'True', 'False', 'None',
}


def check_undefined_names(root: pathlib.Path) -> list:
    """scripts/ 内被调用但未在本模块定义/导入的名字：批量改名事故的机械兜底。

    只看 Call 节点里的裸 Name（不含属性调用与内建）；模块级 def/import/赋值与
    函数内外层参数都算已定义。误报代价低（能改到），漏报代价高（2026-11-19 的
    sizes→file_size 事故 15 处全属此类，compileall 与既有 lint 都放行）。
    """
    problems = []
    builtin_names = set(dir(builtins))
    for path in sorted((root / 'scripts').glob('*.py')):
        rel = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'))
        except SyntaxError:
            continue
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    defined.add((a.asname or a.name).split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    defined.add(a.asname or a.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                defined.update(node.names)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Name):
                continue
            name = func.id
            if name in defined or name in builtin_names or name in _UNDEF_SKIP:
                continue
            problems.append(f'{rel}:{node.lineno}: 调用未定义的名字 {name}()')
    return problems


def run(root: pathlib.Path = ROOT, experiments=None) -> tuple:
    experiments = experiments or registry.EXPERIMENTS
    problems = (check_spec(root)
                + check_ids(experiments)
                + check_paths(root)
                + check_local_imports(root)
                + check_script_sources(root)
                + check_artifact_names(root)
                + check_text_io_encoding(root)
                + check_hygiene(root)
                + check_undefined_names(root)
                + check_script_literals(root)
                + check_teacher_only_panels(root))
    return (0 if not problems else 1), problems
