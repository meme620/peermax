"""环境 / 配置 / 数据三级自检。

用法：
    python tools/preflight.py --level env|configs|data|all

退出码：env 或 configs 不通过 -> 1；data 级只报告不判失败（数据完备性取决于仓库外的决定）。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

CORE = ['pandas', 'numpy', 'scipy', 'sklearn', 'yaml', 'pyarrow', 'pytest']
CONDITIONAL = ['jieba', 'ahocorasick', 'pymupdf', 'pdfminer', 'fontTools', 'pptx']
FETCH = ['baostock']
ALL_PACKAGES = CORE + CONDITIONAL + FETCH

# import 名与发行包名不同的几项，取版本时要用发行包名
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
    's3a_final_spec_v1.yaml',
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


def check_python(root: pathlib.Path = ROOT) -> tuple[bool, list[str]]:
    marker = root / '.python-version'
    want = marker.read_text(encoding='utf-8').strip() if marker.is_file() else ''
    have = '.'.join(str(x) for x in sys.version_info[:2])
    if not want:
        return False, [f'Python {have}: 缺少 .python-version']
    ok = want == have
    verdict = '一致' if ok else '不一致'
    return ok, [f'Python {have}：与 .python-version（{want}）{verdict}']


def check_imports(packages: list[str] = ALL_PACKAGES) -> tuple[bool, list[str]]:
    lines: list[str] = []
    failed: list[str] = []
    for name in packages:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - 自检要把任何失败都报出来
            failed.append(f'{name}: {type(exc).__name__}: {exc}')
            lines.append(f'  [缺失] {name}')
            continue
        try:
            version = importlib.metadata.version(DIST_NAMES.get(name, name))
        except importlib.metadata.PackageNotFoundError:
            version = '(无元数据)'
        lines.append(f'  [ok]   {name} {version}')
    return not failed, lines + failed


def check_configs(root: pathlib.Path = ROOT) -> tuple[bool, list[str]]:
    import yaml

    lines: list[str] = []
    missing: list[str] = []
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


def check_data(root: pathlib.Path = ROOT) -> tuple[bool, list[str]]:
    lines: list[str] = []
    ready = 0
    for name in RAW_DIRS:
        path = root / 'data' / 'raw' / name
        count = sum(1 for p in path.rglob('*') if p.is_file()) if path.is_dir() else 0
        if count:
            ready += 1
            lines.append(f'  [就绪] data/raw/{name}: {count} 个文件')
        else:
            lines.append(f'  [空]   data/raw/{name}')
    external = root / EXTERNAL_FILE
    lines.append(f"  [{'就绪' if external.is_file() else '缺失'}] {EXTERNAL_FILE}")
    processed = root / PROCESSED_DIR
    n_processed = sum(1 for p in processed.rglob('*') if p.is_file()) if processed.is_dir() else 0
    lines.append(f'  [产物] {PROCESSED_DIR}: {n_processed} 个文件')
    gates = root / GATES_DIR
    n_gates = sum(1 for p in gates.rglob('*') if p.is_file()) if gates.is_dir() else 0
    lines.append(f'  [产物] {GATES_DIR}: {n_gates} 个文件')
    lines.append(f'  小结: raw 目录 {ready}/{len(RAW_DIRS)} 就绪')
    return True, lines


def run(levels: list[str], root: pathlib.Path = ROOT) -> tuple[bool, str]:
    blocks: list[str] = []
    ok = True
    if 'env' in levels:
        py_ok, py_lines = check_python(root)
        im_ok, im_lines = check_imports()
        ok = ok and py_ok and im_ok
        blocks.append('== 解释器 ==\n' + '\n'.join(py_lines))
        blocks.append('== 依赖 ==\n' + '\n'.join(im_lines))
    if 'configs' in levels:
        c_ok, c_lines = check_configs(root)
        ok = ok and c_ok
        blocks.append('== 配置 ==\n' + '\n'.join(c_lines))
    if 'data' in levels:
        _, d_lines = check_data(root)
        blocks.append('== 数据 ==\n' + '\n'.join(d_lines))
    return ok, '\n\n'.join(blocks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='PeerMAX 环境/配置/数据自检')
    parser.add_argument('--level', default='all', choices=['env', 'configs', 'data', 'all'])
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args(argv)
    levels = ['env', 'configs', 'data'] if args.level == 'all' else [args.level]
    ok, text = run(levels, pathlib.Path(args.root))
    print(text)
    print('\n结论:', '通过' if ok else '未通过')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
