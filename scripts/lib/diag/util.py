"""s3_pc_sign_diag_h1h8 的 util 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.diag.consts import DIAG
from lib.diag.consts import PROGRESS
from lib.diag.consts import ROOT
import json
import math
import numpy as np

def psay(step: str, msg: str) -> None:
    line = f'[D-{step}] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')

def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)

def pct(x):
    return None if x is None or not np.isfinite(x) else round(float(x) * 100.0, 4)

def dump(tag: str, obj) -> None:
    p = DIAG / f'{tag}.json'
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    psay(tag, f'产出 {p.relative_to(ROOT)}')

def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def power_mde(beta_pct, t_nw, effect_ref_pct: float = 0.152) -> dict | None:
    """REV2-F1(f)：由 (β, t) 反推 se → 95% CI、80% 功效 MDE、原文量级效应下的功效。"""
    if beta_pct is None or t_nw in (None, 0) or not np.isfinite(t_nw):
        return None
    se = abs(float(beta_pct) / float(t_nw))
    z = effect_ref_pct / se
    return {
        'beta_pct': beta_pct, 't_nw': t_nw, 'se_pct': r4(se),
        'ci95_pct': [r4(beta_pct - 1.96 * se), r4(beta_pct + 1.96 * se)],
        'mde80_pct': r4(2.8016 * se),
        'power_at_effect_ref': r4(_phi(z - 1.96) + _phi(-z - 1.96)),
        'effect_ref_pct': effect_ref_pct,
        'note': 'MDE(80%) = 2.8016 × se（双侧 α=0.05）；功效 = P(拒绝 β=0 | β=effect_ref) 正态近似',
    }

def fmt_fm(blk: dict, keys=('PC',)) -> dict:
    out = {'n_months_used': blk['n_months_used'], 'coefs_pct': {}}
    for c in keys:
        s = blk['coefs'][c]
        out['coefs_pct'][c] = {'beta': pct(s['beta']), 't_nw': r4(s['t_nw']), 't_iid': r4(s['t_iid'])}
    return out

