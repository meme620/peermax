"""环境 / 配置 / 数据三级自检（CLI：python -m peerlab doctor）。"""

from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CORE = ['pandas', 'numpy', 'scipy', 'sklearn', 'yaml', 'pyarrow', 'pytest']
CONDITIONAL = ['jieba', 'ahocorasick', 'pymupdf', 'pdfminer', 'fontTools', 'pptx']
FETCH = ['baostock']
ALL_PACKAGES = CORE + CONDITIONAL + FETCH

DIST_NAMES = {
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'ahocorasick': 'pyahocorasick',
    'pdfminer': 'pdfminer.six',
    'pptx': 'python-pptx',
    'fontTools': 'fonttools',
}

CONFIGS = [
    'parameter_registry.yaml',
    's3a_final_spec.yaml',
    'version_manifest.yaml',
    'm1_contracts.yaml',
    'claim_registry.yaml',
    'scope_card.yaml',
]

RAW_DIRS = [
    'analyst', 'baostock', 'cfnd', 'cninfo', 'cninfo_prospectus', 'drive',
    'eastmoney', 'institutional', 'lexicon', 'sina', 'sw2', 'teacher',
    'tencent', 'wsvi',
]

EXTERNAL_FILE = 'data/external/sw_industry_all_stkcds_20260915.xlsx'
PROCESSED_DIR = 'data/processed'
GATES_DIR = 'reports/data_gates/G0'


def check_python(root: pathlib.Path = ROOT):
    marker = root / '.python-version'
    want = marker.read_text(encoding='utf-8').strip() if marker.is_file() else ''
    have = '.'.join(str(x) for x in sys.version_info[:2])
    if not want:
        return False, [f'Python {have}: 缺少 .python-version']
    ok = want == have
    return ok, [f'Python {have}：与 .python-version（{want}）' + ('一致' if ok else '不一致')]


def check_imports(packages=None):
    packages = packages or ALL_PACKAGES
    lines = []
    failed = []
    for name in packages:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed.append(f'{name}: {type(exc).__name__}: {exc}')
            lines.append(f'  [缺失] {name}')
            continue
        try:
            version = importlib.metadata.version(DIST_NAMES.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            version = '(无元数据)'
        lines.append(f'  [ok]   {name} {version}')
    return not failed, lines + failed


def check_configs(root: pathlib.Path = ROOT):
    import yaml

    lines = []
    missing = []
    for name in CONFIGS:
        path = root / 'configs' / name
        if not path.is_file():
            missing.append(name)
            lines.append(f'  [缺失] configs/{name}')
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding='utf-8'))
        except yaml.YAMLError as exc:
            missing.append(name)
            lines.append(f'  [解析失败] configs/{name}: {exc}')
            continue
        if not isinstance(doc, dict):
            missing.append(name)
            lines.append(f'  [结构异常] configs/{name}: 顶层不是映射')
            continue
        lines.append(f'  [ok]   configs/{name}')
    return not missing, lines


def payload_count(base: pathlib.Path) -> int:
    """只数真实载荷；.gitkeep 一类占位文件不算数据就绪。"""
    if not base.is_dir():
        return 0
    return sum(1 for p in base.rglob('*') if p.is_file() and not p.name.startswith('.'))


def check_data(root: pathlib.Path = ROOT):
    lines = []
    ready = 0
    for name in RAW_DIRS:
        path = root / 'data' / 'raw' / name
        count = payload_count(path)
        if count:
            ready += 1
            lines.append(f'  [就绪] data/raw/{name}: {count} 个文件')
        else:
            lines.append(f'  [空]   data/raw/{name}')
    external = root / EXTERNAL_FILE
    lines.append(f"  [{'就绪' if external.is_file() else '缺失'}] {EXTERNAL_FILE}")
    for label, rel in (('产物', PROCESSED_DIR), ('产物', GATES_DIR)):
        base = root / rel
        count = payload_count(base)
        lines.append(f'  [{label}] {rel}: {count} 个文件')
    lines.append(f'  小结: raw 目录 {ready}/{len(RAW_DIRS)} 就绪')
    return True, lines


def run(levels, root: pathlib.Path = ROOT):
    blocks = []
    ok = True
    if 'env' in levels:
        py_ok, py_lines = check_python(root)
        im_ok, im_lines = check_imports()
        ok = ok and py_ok and im_ok
        blocks.append('== 解释器 ==' + chr(10) + chr(10).join(py_lines))
        blocks.append('== 依赖 ==' + chr(10) + chr(10).join(im_lines))
    if 'configs' in levels:
        c_ok, c_lines = check_configs(root)
        ok = ok and c_ok
        blocks.append('== 配置 ==' + chr(10) + chr(10).join(c_lines))
    if 'data' in levels:
        _, d_lines = check_data(root)
        blocks.append('== 数据 ==' + chr(10) + chr(10).join(d_lines))
    return ok, (chr(10) + chr(10)).join(blocks)
