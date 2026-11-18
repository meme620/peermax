"""s3_pc_sign_diag_h1h8 的 data 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.diag.consts import PANEL
from lib.diag.consts import PC_DIR
from lib.diag.consts import PC_YEARS
from lib.diag.consts import REV_PKL
from lib.diag.consts import SW2_MAP
from pathlib import Path
import pandas as pd

def load_pc(pc_dir: Path = PC_DIR, years=PC_YEARS) -> pd.DataFrame:
    frames = []
    for y in years:
        f = pc_dir / f'{y}_pc.csv'
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={'focal_id': str})
        frames.append(d)
    pc = pd.concat(frames, ignore_index=True)
    for c in ('hhi', 'pc', 'hhi_excl_self', 'revenue_coverage'):
        pc[c] = pd.to_numeric(pc[c], errors='coerce')
    return pc

def load_panel() -> pd.DataFrame:
    cols = ['focal_id', 'month', 'network_version', 'R_next', 'next_R', 'R_next_postadj',
            'market_equity', 'size', 'bm', 'mom', 'rev', 'illiq', 'turnover',
            'm1_in_fm_sample', 'm1_exclusion_reason', 'me_below_q10', 'me_q10_market',
            'AnalystCoverage', 'ind_l2_ret_vw', 'ind_l1_ret_vw', 'st_flag_focal', 'recent_ipo']
    d = pd.read_parquet(PANEL, columns=cols)
    d['text_year'] = pd.to_numeric(d['network_version'], errors='coerce')
    return d

def merge_pc(panel: pd.DataFrame, pc: pd.DataFrame) -> pd.DataFrame:
    m = panel.merge(pc[['focal_id', 'text_year', 'pc', 'hhi', 'hhi_excl_self',
                        'n_competitors', 'revenue_coverage', 'revenue_year']],
                    on=['focal_id', 'text_year'], how='left')
    m['PC'] = pd.to_numeric(m['pc'], errors='coerce')
    return m

def load_rev(year: int) -> dict:
    d = pd.read_pickle(REV_PKL)
    d.index = [ts.strftime('%Y') if hasattr(ts, 'strftime') else str(ts)[:4] for ts in d.index]
    y = str(year)
    if y not in d.index:
        return {}
    row = d.loc[y]
    return {str(c).split('.')[0].zfill(6): float(v) for c, v in row.items() if v == v and v > 0}

def sw2_asof(month_series: pd.Series, firm_series: pd.Series) -> pd.Series:
    mp = pd.read_csv(SW2_MAP, dtype={'code6': str})
    mp = mp[mp['is_current_version'].astype(bool)]
    mp['firm_id'] = mp['firm_id'].astype(str)
    lut = mp.drop_duplicates('firm_id').set_index('firm_id')['l2_code'].astype(str)
    return firm_series.map(lut)

