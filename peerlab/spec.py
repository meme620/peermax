"""spec/ 下的 YAML 是文档的可执行替身：每条引用都要能解析到实体。

校验对象：
- spec/design.yaml：变量 implemented_by 的脚本、任何层级的 registry_key / registry_keys
  参数名、数据门与约束的 enforced_by 引用、红线引用；
- spec/stages.yaml：阶段脚本存在；
- spec/inputs.yaml：消费者脚本存在，required 的路径必须在盘上。
"""

from __future__ import annotations

import pathlib

import yaml

from . import data_contracts, gates, lint, registry

SPEC_FILES = ('design.yaml', 'stages.yaml', 'inputs.yaml')


def _load(root: pathlib.Path, name: str):
    path = pathlib.Path(root) / 'spec' / name
    if not path.is_file():
        raise FileNotFoundError(f'缺 spec/{name}')
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def _registry_keys(root: pathlib.Path) -> set:
    path = pathlib.Path(root) / 'configs' / 'parameter_registry.yaml'
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    return {item['name'] for item in data.get('parameters', [])}


def _registry_names(value, where):
    """registry_key(s) 的值可以是标量、列表或「标签 → 键名」映射。"""
    if isinstance(value, dict):
        for label, item in value.items():
            yield from _registry_names(item, f'{where}.{label}')
    elif isinstance(value, list):
        for item in value:
            yield from _registry_names(item, where)
    elif value:
        yield where, value


def _registry_refs(node, where='design'):
    """遍历 design，产出 (位置, 参数名)：字段名 registry_key / registry_keys 即引用。"""
    if isinstance(node, dict):
        label = node.get('id') or node.get('name')
        here = f'{where}[{label}]' if label else where
        for key, value in node.items():
            if key in ('registry_key', 'registry_keys'):
                yield from _registry_names(value, f'{here}.{key}')
            elif isinstance(value, (dict, list)):
                yield from _registry_refs(value, f'{here}.{key}')
    elif isinstance(node, list):
        for item in node:
            yield from _registry_refs(item, where)


def check_registry_refs(design, keys) -> list:
    return [f'{where}: registry 无参数 {key}'
            for where, key in _registry_refs(design) if key not in keys]


def _resolve(token: str, root: pathlib.Path) -> str | None:
    """返回问题描述；None 表示解析成功。"""
    kind, _, name = token.partition(':')
    base = pathlib.Path(root)
    if kind == 'script':
        return None if (base / 'scripts' / (name + '.py')).is_file() else f'缺脚本 scripts/{name}.py'
    if kind == 'gate':
        ids = {gate.id for gate in gates.GATES}
        return None if name in ids else f'缺数据门 {name}'
    if kind == 'lint':
        return None if hasattr(lint, name) else f'缺 lint 检查 {name}'
    if kind == 'contract':
        return None if name in data_contracts.CHECKS else f'缺数据契约 {name}'
    if kind == 'test':
        return None if (base / name).is_file() else f'缺测试 {name}'
    return f'未知引用类型 {token!r}'


def _check_refs(entries, field, root, label, problems):
    for entry in entries or []:
        for token in entry.get(field) or []:
            issue = _resolve(token, root)
            if issue:
                problems.append(f'{label} {entry.get("id") or entry.get("name")}: {issue}')


def _validate_contracts(root, problems) -> None:
    path = pathlib.Path(root) / 'spec' / 'data_contracts.yaml'
    if not path.is_file():
        problems.append('缺 spec/data_contracts.yaml')
        return
    doc = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    ids = [c.get('id') for c in doc.get('contracts', []) if isinstance(c, dict)]
    if len(ids) != len(set(ids)):
        problems.append('spec/data_contracts.yaml: contract id 重复')
    missing = sorted(set(data_contracts.CHECKS) - set(ids))
    extra = sorted(set(ids) - set(data_contracts.CHECKS))
    if missing:
        problems.append('spec/data_contracts.yaml: 缺 contract ' + ', '.join(missing))
    if extra:
        problems.append('spec/data_contracts.yaml: 未知 contract ' + ', '.join(extra))


def validate(root) -> list:
    root = pathlib.Path(root)
    problems = []
    design = _load(root, 'design.yaml')
    stages = _load(root, 'stages.yaml')
    inputs = _load(root, 'inputs.yaml')
    keys = _registry_keys(root)

    for var in design.get('variables', []):
        for script in var.get('implemented_by', []):
            if not (root / 'scripts' / (script + '.py')).is_file():
                problems.append(f'design.variables {var["name"]}: 缺脚本 scripts/{script}.py')
    problems.extend(check_registry_refs(design, keys))
    for gate in design.get('pricing_gates', []):
        _check_refs([gate], 'enforced_by', root, 'design.pricing_gates', problems)
    _check_refs(design.get('data_gates'), 'enforced_by', root, 'design.data_gates', problems)
    _check_refs(design.get('data_contracts'), 'enforced_by', root, 'design.data_contracts', problems)
    _check_refs([design.get('sample_filters') or {}], 'enforced_by', root, 'design.sample_filters', problems)
    _check_refs([design.get('max_definition') or {}], 'enforced_by', root, 'design.max_definition', problems)
    _check_refs(design.get('constraints'), 'enforced_by', root, 'design.constraints', problems)
    _check_refs(design.get('red_lines'), 'enforced_by', root, 'design.red_lines', problems)

    for stage in stages.get('stages', []):
        for script in stage.get('scripts', []):
            if not (root / 'scripts' / (script + '.py')).is_file():
                problems.append(f'stages {stage["id"]}: 缺脚本 scripts/{script}.py')

    for item in inputs.get('inputs', []):
        consumer = item.get('consumer')
        if consumer and not (root / 'scripts' / (consumer + '.py')).is_file():
            problems.append(f'inputs {item["path"]}: 缺消费者 scripts/{consumer}.py')
        if item.get('required', True) and not (root / item['path']).exists():
            problems.append(f'inputs {item["path"]}: 标注 required 但不在盘上')

    _validate_contracts(root, problems)

    for exp in registry.EXPERIMENTS:
        note = registry.EXPERIMENT_NOTES.get(exp.id)
        if not note or not note.get('purpose') or not note.get('notes'):
            problems.append(f'registry {exp.id}: 缺 purpose 或 notes（实验 spec 不完整）')
    return problems


def summary(root) -> dict:
    root = pathlib.Path(root)
    design = _load(root, 'design.yaml')
    stages = _load(root, 'stages.yaml')
    inputs = _load(root, 'inputs.yaml')
    return {
        'variables': len(design.get('variables', [])),
        'pricing_gates': len(design.get('pricing_gates', [])),
        'data_gates': len(design.get('data_gates', [])),
        'constraints': len(design.get('constraints', [])),
        'red_lines': len(design.get('red_lines', [])),
        'stages': len(stages.get('stages', [])),
        'inputs': len(inputs.get('inputs', [])),
        'experiments': len(registry.EXPERIMENTS),
        'enforced_refs': sum(len(entry.get('enforced_by') or [])
                             for group in ('pricing_gates', 'data_gates', 'constraints', 'red_lines')
                             for entry in design.get(group, [])),
    }
