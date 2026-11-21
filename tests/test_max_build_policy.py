import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import peermax_max_build as m  # noqa: E402


def _panel(root: Path) -> Path:
    panel = root / 'daily_panel'
    panel.mkdir()
    # headline 收益基期是 r_adj；夹具里与 rraw 同值，只验涨跌停/复牌日剔除逻辑
    rows = ['code,date,close_raw,close_qfq,volume,amount,rraw,ca_flag,r_adj']
    for d, v in (('2020-01-02', 0.0), ('2020-01-03', 0.10),
                 ('2020-01-06', 0.03), ('2020-01-07', -0.02)):
        rows.append(f'sz000001,{d},10,10,1000,1000,{v},0,{v}')
    rows.append('sz000002,2020-01-02,10,10,1000,1000,0.0,0,0.0')
    rows.append('sz000002,2020-01-06,10,10,1000,1000,-0.01,0,-0.01')
    (panel / '2020.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return panel


def _flags(root: Path) -> Path:
    idx = pd.to_datetime(['2020-01-02', '2020-01-03', '2020-01-06', '2020-01-07'])
    cols = ['000001.SZ', '000002.SZ']
    pd.DataFrame(False, index=idx, columns=cols).to_pickle(root / 'st.pkl')
    up = pd.DataFrame(False, index=idx, columns=cols)
    up.loc[pd.Timestamp('2020-01-03'), '000001.SZ'] = True
    up.to_pickle(root / 'up.pkl')
    up.to_pickle(root / 'daily_limit_down_flag_20260630.pkl')
    return root / 'up.pkl'


def test_run_excludes_limit_and_resume_days(tmp_path: Path, monkeypatch) -> None:
    panel = _panel(tmp_path)
    flag = _flags(tmp_path)
    monkeypatch.setattr(m, 'LIMIT_HIT_FLAG', flag)
    m.run(panel_dir=panel, out=tmp_path / 'out.csv', out_json=tmp_path / 'rep.json',
          st_flag=tmp_path / 'st.pkl')
    df = pd.read_csv(tmp_path / 'out.csv', dtype=str, keep_default_na=False)
    by = {r['source_id']: r for r in df.to_dict('records')}
    assert by['sz000001']['MAXraw'] == '0.03'
    assert by['sz000001']['MAXraw_incl_all'] == '0.1'
    assert by['sz000001']['n_excluded_limit'] == '1'
    assert by['sz000002']['MAXraw'] == '0.0'
    assert by['sz000002']['n_excluded_resume'] == '1'


def test_max_uses_r_adj_not_rraw(tmp_path: Path, monkeypatch) -> None:
    """判别力测试：r_adj 与 rraw 取不同值时，MAXraw 必须来自 r_adj。

    本文件其余夹具令两者同值，实现回退到 rraw 时仍会绿；这一条专门堵住该缺口。
    """
    panel = tmp_path / 'daily_panel'
    panel.mkdir()
    rows = ['code,date,close_raw,close_qfq,volume,amount,rraw,ca_flag,r_adj']
    # 除权日形态：不复权 rraw 出现伪极端值，后复权 r_adj 正常
    rows.append('sz000001,2020-01-02,10,10,1000,1000,0.0,0,0.0')
    rows.append('sz000001,2020-01-03,5,10,1000,1000,-0.50,1,0.0')
    rows.append('sz000001,2020-01-06,5,11,1000,1000,0.0,0,0.10')
    (panel / '2020.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')

    flag = _flags(tmp_path)
    monkeypatch.setattr(m, 'LIMIT_HIT_FLAG', flag)
    out = tmp_path / 'out.csv'
    m.run(panel_dir=panel, out=out, out_json=tmp_path / 'rep.json',
          st_flag=tmp_path / 'st.pkl')
    df = pd.read_csv(out, dtype=str, keep_default_na=False)
    row = next(r for r in df.to_dict('records') if r['source_id'].endswith('000001'))
    # r_adj 月内最大值 0.10；若实现回退到 rraw 会得到 0.0
    assert abs(float(row['MAXraw']) - 0.10) < 1e-9, f"MAXraw 未取自 r_adj: {row['MAXraw']}"
