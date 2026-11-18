"""子进程引导：播种后以 __main__ 方式执行快照脚本。

用法：python -m peerlab.bootstrap <script> [args...]
环境变量：PEERLAB_SEED（种子），PYTHONHASHSEED（由 runner 设置）。

脚本用 Path(__file__).resolve().parents[1] 定位仓库根，runpy 保留 __file__，
因此层级不变；同级 import 依赖脚本目录在 sys.path[0]，这里补上。
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys


def seed_everything(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit('用法: python -m peerlab.bootstrap <script> [args...]')
    script = pathlib.Path(argv[0]).resolve()
    if not script.is_file():
        raise SystemExit(f'脚本不存在: {script}')
    seed_everything(int(os.environ.get('PEERLAB_SEED', '0')))
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *argv[1:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main(sys.argv[1:])
