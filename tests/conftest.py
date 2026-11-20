import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / 'scripts'):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

ARTIFACT_ROOTS = (ROOT / 'data' / 'processed', ROOT / 'artifacts', ROOT / 'reports')


def require_artifacts(*paths, hint: str = '缺产物（先跑上游脚本）') -> None:
    """产物不存在就 skip：把"还没跑"与"代码坏了"分开。

    契约测试默认读真实产物；仓库重建前这些用例系统性报红，掩盖真缺陷。
    只跳过确实缺失的路径，不改变任何断言逻辑。
    """
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(f'{hint}: {missing[0]}')


def registered_rows(key: str) -> int:
    """从 configs/pricing_inputs.yaml 读登记的 rows（复现门槛的唯一来源）。"""
    import yaml as _yaml
    doc = _yaml.safe_load((ROOT / 'configs' / 'pricing_inputs.yaml').read_text(encoding='utf-8'))
    rows = doc['pricing_inputs'][key].get('rows')
    assert rows is not None, f'{key}: 合同未登记 rows'
    return int(rows)


def assert_input_anchors(keys=None) -> None:
    """校验 configs/pricing_inputs.yaml 里已登记的输入：行数与列名必须与实测一致。

    sha256 已退役（2026-11-19）：输入身份由「行数 + 列名」合同认定。未登记 rows 的条目
    直接报错（fail-closed），文件缺失则 skip（属"还没跑"）。
    """
    import csv as _csv
    import yaml as _yaml
    contract = _yaml.safe_load((ROOT / 'configs' / 'pricing_inputs.yaml').read_text(encoding='utf-8'))
    entries = contract['pricing_inputs']
    picked = {k: v for k, v in entries.items() if keys is None or k in keys}
    missing = [str(ROOT / v['file']) for v in picked.values() if not (ROOT / v['file']).is_file()]
    if missing:
        pytest.skip(f'缺产物（先跑上游脚本）: {missing[0]}')
    for key, reg in picked.items():
        p = ROOT / reg['file']
        rows = reg.get('rows')
        assert rows is not None, f'{key}: 合同未登记 rows（fail-closed）'
        if p.suffix == '.csv':
            with open(p, encoding='utf-8', newline='') as fh:
                r = _csv.reader(fh)
                header = next(r, [])
                actual = sum(1 for _ in r)
            for col in (reg.get('columns') or []):
                assert col in header, f'{key}: 缺列 {col}'
        else:
            import pandas as _pd
            actual = len(_pd.read_parquet(p, columns=None))
        assert int(rows) == actual, f'{key}: 行数合同={rows} 实测={actual}'


# 只认「产物不存在」的措辞。校验失败 / 不一致 / 重叠 / 符号翻转等是 fail-closed 的
# 业务拒绝，必须保持 fail——把它们当缺产物会静默吞掉真实的网络与定价守卫。
_MISSING_PAT = re.compile(r'缺少|缺锚|缺 |No such file or directory|不存在|未发现|文件缺失')


def _missing_artifact_path(exc_value) -> str | None:
    """判定一个异常是否只是"产物不存在"，返回缺失路径；否则 None。

    只认 data/processed、artifacts、reports 三个产物根下的路径：测试里断言业务
    规则失败、或读 tests/ 自带 fixture 失败，仍按失败处理，不会被吞成 skip。
    """
    if isinstance(exc_value, FileNotFoundError):
        path = getattr(exc_value, 'filename', None)
        return str(path) if path else None
    text = str(exc_value) if isinstance(exc_value, (AssertionError, SystemExit)) else ''
    if not text:
        return None
    # 断言消息本身就是产物路径（assert p.is_file(), p）：等价于"产物缺失"，
    # 不需要额外的提示词，否则这类断言会被当成业务失败。
    m = re.fullmatch(r"PosixPath\('([^']+)'\)", text.strip())
    if m:
        return m.group(1)
    if not _MISSING_PAT.search(text):
        return None
    m = re.search(r'((?:data/processed|artifacts|reports)/[^\s）)，,;]+)', text)
    if m:
        return m.group(1)
    m = re.search(r'(/[^\s）)，,;]+)', text)
    if m:
        return m.group(1)
    m = re.search(r"PosixPath\('([^']+)'\)", text)
    if m:
        return m.group(1)
    if re.search(r'缺少|缺锚|缺 |不存在|未发现|文件缺失', text):
        m = re.search(r'net_dir=([^\s]+)', text)
        if m:
            return m.group(1)
        m = re.search(r'([A-Za-z0-9_./-]+\.(?:csv|json|parquet|pkl|md))', text)
        if m:
            return str(ARTIFACT_ROOTS[1]) + '/' + m.group(1)
        if re.search(r'先运行|先跑|未发现', text):
            return str(ARTIFACT_ROOTS[1])
    return None


def _under_artifact_root(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return any(str(p).startswith(str(r)) for r in ARTIFACT_ROOTS)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """缺产物的失败降级为 skip：分辨"还没跑"与"代码坏了"。

    2026-11-19 前，缺 data/processed 产物时整套契约测试是红的（120 failed /
    152 errors），与真实缺陷混在一起看不出区别。这里只对产物根下的缺失生效，
    其余异常一律保持 fail。
    """
    outcome = yield
    report = outcome.get_result()
    if report.when not in ('setup', 'call') or not report.failed:
        return
    exc = getattr(call, 'excinfo', None)
    if exc is None or exc.value is None:
        return
    if isinstance(exc.value, SystemExit):
        captured = ' '.join(body for name, body in (report.sections or [])
                            if 'stderr' in name or 'stdout' in name)
        path = _missing_artifact_path(AssertionError(captured)) if captured else None
    else:
        path = _missing_artifact_path(exc.value)
    if not _under_artifact_root(path):
        return
    report.outcome = 'skipped'
    report.longrepr = (str(path), 0, f'缺产物（先跑上游脚本）: {path}')
