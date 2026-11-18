"""月度字符串运算。函数体逐字搬自 5 处 `_next_month`。"""

from __future__ import annotations


def next_month(ym: str) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    m += 1
    if m == 13:
        y, m = y + 1, 1
    return f'{y:04d}-{m:02d}'
