"""TNIC-3 structure benchmark summary.

Reads the HP TNIC-3 archive (tab-delimited year/gvkey1/gvkey2/score, symmetric with
self-records) and produces per-year structure statistics used only as an external
benchmark for the PeerMAX MA/MB product-market networks.

Per HP readme Technical Notes 1-3: rows are symmetric (each pair twice), self-records
are included and must be dropped before counting peers, and `score` is the excess of
the pairwise similarity over the inclusion threshold (not a raw cosine).
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / 'artifacts/external_benchmarks/tnic/tnic3_data.zip'
OUT_CSV = ROOT / 'artifacts/external_benchmarks/tnic/tnic_structure_summary.csv'
OUT_JSON = ROOT / 'artifacts/external_benchmarks/tnic/tnic_structure_summary.json'
CHUNK = 5_000_000


def main() -> int:
    if not ZIP.exists():
        print(f'BLOCKED: missing {ZIP}', file=sys.stderr)
        return 2

    peer_counts: dict[int, pd.Series] = {}
    scores = []
    n_rows_total = 0

    zf = zipfile.ZipFile(ZIP)
    with zf.open('tnic3_data.txt') as raw:
        reader = pd.read_csv(io.TextIOWrapper(raw, encoding='utf-8'), sep='\t',
                             usecols=['year', 'gvkey1', 'gvkey2', 'score'],
                             chunksize=CHUNK)
        for chunk in reader:
            n_rows_total += len(chunk)
            self_rows = chunk['gvkey1'] == chunk['gvkey2']
            scores.append(float(self_rows.sum()))  # self-record count = n firms
            n_self = int(self_rows.sum())
            pc = chunk.loc[~self_rows]
            grp = pc.groupby(['year', 'gvkey1']).size()
            for (yr, gv), n in grp.items():
                prev = peer_counts.get(yr)
                add = pd.Series({gv: n}, dtype='int64')
                peer_counts[yr] = add if prev is None else prev.add(add, fill_value=0)

    rows = []
    for yr in sorted(peer_counts):
        s = peer_counts[yr]
        rows.append({
            'year': yr,
            'n_firms': int(len(s)),
            'peer_mean': float(s.mean()),
            'peer_median': float(s.median()),
            'peer_p10': float(s.quantile(0.10)),
            'peer_p90': float(s.quantile(0.90)),
            'peer_min': int(s.min()),
            'peer_max': int(s.max()),
        })
    df = pd.DataFrame(rows).sort_values('year')
    df['density'] = df['peer_mean'] / (df['n_firms'] - 1)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    payload = {
        'source_zip': str(ZIP.relative_to(ROOT)).replace(chr(92), '/'),
        'source_zip_bytes': ZIP.stat().st_size,
        'n_rows_total_including_self': int(n_rows_total),
        'n_years': int(len(df)),
        'year_min': int(df['year'].min()),
        'year_max': int(df['year'].max()),
        'note': ('score = excess similarity over inclusion threshold (not raw cosine); '
                 'rows symmetric with self-records; TNIC calendar year = Compustat datadate year, '
                 'readme states data is not lagged'),
        'latest_year_stats': df.iloc[-1].to_dict(),
        'year_2016_stats': df[df['year'] == 2016].iloc[0].to_dict() if (df['year'] == 2016).any() else None,
    }
    import json
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding='utf-8')
    print(df.tail(12).to_string(index=False))
    print(f'\nrows(incl self)={n_rows_total:,}  -> {OUT_CSV.relative_to(ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
