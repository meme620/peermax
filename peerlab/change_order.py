"""change order 登记校验（configs/README.md 的流程 1 + 2 的执行面）。

configs/README.md 要求「改登记值必须走 change order」：在 parameter_registry.yaml 的
change_log 追加一条，并升 registry_version。此前这两步只靠人工遵守，没有代码检查，
于是登记表可以在不留下 change order 的情况下被改动。本模块把该流程变成可执行的不变量。

五条不变量（全部 fail-closed，逐条给可定位的行/字段）：
  C1 字段完整：每条 change_log 必须有 id/date/scope/rationale/params/version_from/version_to；
  C2 版本成链：把各条的 (version_from, version_to) 串成链后必须连续、无环，且终点
     等于 registry_version；version_to 大于 version_from；
  C3 变更覆盖：某条登记的 params 必须逐项存在于 parameters，且该条不能是「空改」
      （params 为空且没有 renames）；
  C4 id 唯一且成格式：CO-YYYY-MM-DD-slug；
  C5 日期不倒序：change_log 按 date 倒序排列（最新在上），与现有条目一致。

不做的事：不校验「参数值是否真的被改过」——那需要历史版本比对，仓库里没有旧值快照；
不校验 approval 的自由文本内容。
"""

from __future__ import annotations

import datetime as _dt
import pathlib
import re

REGISTRY_CONFIG = 'configs/parameter_registry.yaml'
_ID_RE = re.compile(r'^CO-(\d{4})-(\d{2})-(\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)$')
_REQUIRED = ('id', 'date', 'scope', 'rationale', 'params', 'version_from', 'version_to')


def _parse_version(text: str) -> tuple[int, ...]:
    """'1.7.0-peermax-baseline' -> (1, 7, 0)。非 '<digits>.<digits>.<digits>' 前缀即抛错。"""
    m = re.match(r'^(\d+)\.(\d+)\.(\d+)', str(text))
    if not m:
        raise ValueError(f'版本串不可解析: {text!r}（须以 <major>.<minor>.<patch> 开头）')
    return tuple(int(g) for g in m.groups())


def check(root: pathlib.Path | None = None) -> list[str]:
    """返回问题列表（空 = 通过）。"""
    import yaml

    base = pathlib.Path(root) if root is not None else pathlib.Path(__file__).resolve().parents[1]
    path = base / REGISTRY_CONFIG
    if not path.is_file():
        return [f'{REGISTRY_CONFIG}: 文件缺失，change order 无从校验']
    doc = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    problems: list[str] = []

    log = doc.get('change_log')
    if not isinstance(log, list) or not log:
        return [f'{REGISTRY_CONFIG}: 缺 change_log（改登记值必须留 change order）']

    # params 条目允许两种写法：顶层参数名（'iqr_pool_value'）或
    # 顶层参数名.子键（'sample_filter_thresholds.limit_hit_source'）。
    # 后者是必要的：登记表里多个参数是带子键的映射，change order 只改其中一个子键，
    # 光写父名无法定位改了什么（CO-2026-11-19-parameters 就登记了 limit_hit_source）。
    param_names: set[str] = set()
    for p in doc.get('parameters') or []:
        if not isinstance(p, dict) or not p.get('name'):
            continue
        name = p['name']
        param_names.add(name)
        hv = p.get('headline_value')
        if isinstance(hv, dict):
            param_names.update(f'{name}.{k}' for k in hv)
    registry_version = doc.get('registry_version')

    # C1 字段完整（顺带 C4 格式、C5 日期）
    seen_ids: set[str] = set()
    parsed: list[tuple[str, _dt.date, tuple[int, ...], tuple[int, ...]]] = []
    for idx, entry in enumerate(log):
        where = f'change_log[{idx}]'
        if not isinstance(entry, dict):
            problems.append(f'{where}: 不是映射')
            continue
        # params 允许为空列表（交给 C3「空改」判定），其余字段不得为空
        missing = [k for k in _REQUIRED
                   if (k == 'params' and entry.get(k) is None)
                   or (k != 'params' and entry.get(k) in (None, '', []))]
        if missing:
            problems.append(f'{where}: 缺字段 {missing}')
            continue

        cid = str(entry['id'])
        if not _ID_RE.match(cid):
            problems.append(f'{where}.id={cid!r}: 须为 CO-YYYY-MM-DD-slug')
        if cid in seen_ids:
            problems.append(f'{where}.id={cid!r}: 重复')
        seen_ids.add(cid)

        try:
            day = _dt.date.fromisoformat(str(entry['date']))
        except ValueError:
            problems.append(f'{where}.date={entry["date"]!r}: 不是 YYYY-MM-DD')
            continue

        params = entry.get('params')
        if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
            problems.append(f'{where}.params: 须为字符串列表')
            continue
        unknown = [p for p in params if p not in param_names]
        if unknown:
            problems.append(f'{where}.params: 登记了不存在的参数 {unknown}')
        if not params and not entry.get('renames'):
            problems.append(f'{where}: 空改（params 为空且无 renames）')

        try:
            v_from = _parse_version(entry['version_from'])
            v_to = _parse_version(entry['version_to'])
        except ValueError as exc:
            problems.append(f'{where}: {exc}')
            continue
        if v_to <= v_from:
            problems.append(f'{where}: version_to {entry["version_to"]!r} 不大于 '
                            f'version_from {entry["version_from"]!r}')

        if 'sample_version_from' in entry or 'sample_version_to' in entry:
            if entry.get('sample_version_from') == entry.get('sample_version_to'):
                problems.append(f'{where}: sample_version_from == sample_version_to（无变更）')

        parsed.append((cid, day, v_from, v_to))

    # C5 日期倒序（最新在上）
    for i in range(1, len(log)):
        a, b = log[i - 1], log[i]
        if isinstance(a, dict) and isinstance(b, dict):
            if str(a.get('date') or '') < str(b.get('date') or ''):
                problems.append(f'change_log[{i}]: 日期 {b.get("date")!r} 晚于上一条 '
                                f'{a.get("date")!r}（须最新在上）')

    # C2 版本成链
    if parsed:
        try:
            head = max(parsed, key=lambda x: x[3])
            if registry_version is not None:
                if head[3] != _parse_version(registry_version):
                    problems.append(
                        f'registry_version={registry_version!r} 与最新 change order '
                        f'{head[0]!r} 的 version_to 不一致')
            # 每条 version_from 必须能在另一条 version_to 里找到（除最旧一条）
            tos = {x[3] for x in parsed}
            froms = {x[2] for x in parsed}
            orphans = sorted(froms - tos)
            if len(orphans) > 1:
                problems.append(
                    'change_log 版本链断裂：'
                    + ', '.join('.'.join(map(str, o)) for o in orphans)
                    + ' 均无前一条 change order 的 version_to 指向')
        except ValueError as exc:
            problems.append(f'版本链校验失败: {exc}')

    return problems
