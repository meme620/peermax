"""批次 1 schema 验收：索引/代码格式/日期覆盖/缺失率/与现有数据对齐。"""
import pandas as pd, json
from pathlib import Path

D = Path('data/raw/drive')
report = {}

def audit(name, df):
    if isinstance(df, pd.Series):
        r = {'type': 'Series', 'rows': len(df), 'name': str(df.name), 'dtype': str(df.dtype)}
        s = pd.Series(df.index.astype(str))
        r['index_sample'] = [str(x) for x in df.index[:3]]
        r['index_unique'] = int(df.index.nunique())
        r['index_len_mode'] = int(s.str.len().mode().iloc[0]) if len(s) else 0
        r['value_sample'] = [str(x) for x in df[:3]]
        r['value_unique'] = int(df.nunique())
        return r
    r = {'rows': len(df)}
    r['columns'] = [str(c) for c in df.columns[:8]]
    r['index_name'] = df.index.name or '(unnamed)'
    idx = df.index
    if isinstance(df, pd.Series):
        r = {'type': 'Series', 'rows': len(df), 'name': str(df.name), 'dtype': str(df.dtype)}
        s = pd.Series(df.index.astype(str))
        r['index_sample'] = [str(x) for x in df.index[:3]]
        r['index_unique'] = int(df.index.nunique())
        r['index_len_mode'] = int(s.str.len().mode().iloc[0]) if len(s) else 0
        try:
            v = pd.to_numeric(df.astype(str), errors='coerce')
            r['value_sample'] = [str(x) for x in df[:3]]
        except (TypeError, ValueError, AttributeError):
            pass
        return r
    if idx.nlevels == 1:
        s = pd.Series(idx.astype(str))
        r['index_sample'] = [str(x) for x in idx[:3]]
        r['index_unique'] = int(idx.nunique())
        r['index_len_mode'] = int(s.str.len().mode().iloc[0]) if len(s) else 0
    else:
        r['index_sample'] = [str(tuple(x)) for x in idx[:3]]
        r['index_levels'] = [str(df.index.get_level_values(i).dtype) for i in range(idx.nlevels)]
    # 日期范围（列或索引中的日期）
    for c in list(df.columns[:4]) + [None]:
        vals = idx if c is None else df[c]
        try:
            dates = pd.to_datetime(pd.Series(vals).astype(str), errors='coerce').dropna()
            valid = dates.between('2000-01-01', '2026-12-31')
            if valid.sum() > len(dates) * 0.5 and len(dates):
                r['date_range'] = [str(dates[valid].min().date()), str(dates[valid].max().date())]
                break
        except (TypeError, ValueError, AttributeError):
            continue
    # 缺失率（数值列）
    num = df.select_dtypes('number') if hasattr(df, 'select_dtypes') else df.iloc[:0]
    if len(num.columns):
        r['null_rate_max'] = round(float(num.isna().mean().max()), 4)
    return r

if __name__ == '__main__':
    for f in sorted(D.glob('*.pkl')):
        name = f.name
        try:
            df = pd.read_pickle(f)
            if isinstance(df, dict):
                report[name] = {'type': 'dict', 'keys': list(df.keys())[:5], 'n_keys': len(df)}
                continue
            report[name] = audit(name, df)
            print(f"--- {name}: {report[name]}", flush=True)
        except Exception as e:
            report[name] = {'error': str(e)[:200]}
            print(f"FAIL {name}: {e}", flush=True)

    Path('reports/data_gates/G0/drive_batch1_schema.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print("DONE -> reports/data_gates/G0/drive_batch1_schema.json")
