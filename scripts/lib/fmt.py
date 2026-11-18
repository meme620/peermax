"""表格数值格式化。函数体逐字搬自 p2_2/p2_3。"""

from __future__ import annotations

import math


def fmt_float(x, nd: int = 8) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    if s in ("", "-0"):
        return "0"
    return s


def fmt_int(x) -> str:
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return str(int(round(v)))
