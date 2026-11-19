"""JSON / CSV 原子写与哈希。函数体逐字搬自既有实现，未改逻辑。"""

from __future__ import annotations

import sys
import csv
import datetime
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_json_atomic(obj: dict, path: Path, *, indent: int = 2, default=None,
                      trailing_newline: bool = True) -> None:
    """JSON 原子写。indent / trailing_newline 保留既有两种产物的字节形态，勿随手统一。"""
    text = json.dumps(obj, ensure_ascii=False, indent=indent, default=default)
    if trailing_newline:
        text += "\n"
    atomic_write_text(Path(path), text)


def write_parquet_atomic(df, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def file_size(path):
    return Path(path).stat().st_size


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    return str(o)


def assert_path_allowed(path, allowed_roots=(), banned_roots=()) -> Path:
    """写出路径必须在 allowed_roots 内（给定时），且不命中 banned_roots（含其后代）。

    违反 → stderr 打印命中集合并 SystemExit(2)。返回 resolve 后的路径。
    各 shadow 脚本只保留自己的 allow/banned 常量，谓词只此一份。
    """
    p = Path(path).resolve()
    allowed = True
    if allowed_roots:
        allowed = any(p == Path(a).resolve() or Path(a).resolve() in p.parents
                      for a in allowed_roots)
    banned = [Path(b).resolve() for b in banned_roots]
    hit = [b for b in banned if p == b or b in p.parents]
    if not allowed or hit:
        print(f'[HARD-GATE] 输出路径命中禁写集合: {p}（allowed={allowed} banned={hit}）',
              file=sys.stderr, flush=True)
        raise SystemExit(2)
    return p


def read_csv_dicts(path, encoding: str = 'utf-8') -> list:
    with open(path, encoding=encoding, newline='') as f:
        return list(csv.DictReader(f))


def read_csv_rows(path, encoding: str = 'utf-8') -> list:
    with open(path, encoding=encoding, newline='') as f:
        return list(csv.reader(f))


def write_csv_rows(path, rows, header) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, p)
