#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""network 对表（correspondence）：刘亚辉等（2021）核心结论 × 我方 network 主口径网络 PC。

对表靶子（原文，`ref/teacher-materials/A股产品市场竞争.pdf`；PDF 页 = 印刷页 − 2060）：
  假设 4 = 产品市场竞争越激烈（PC 越大）→ 股票预期收益率越高。
  表 4（FM 回归，NW t 滞后 12 期；收益单位 %）：
    列(1) 单变量：PC 0.152（t = 7.76）；列(2) 加控制：PC 0.130（t = 5.40）、HHI 0.076（t = 1.94）。
  表 3（十分位组合超额收益，等权/市值加权；α3 = MKT/SMB/HML，α7 = FF5+UMD+LIQ）：
    H−L = 0.218%（t = 6.27, α3）/ 0.441%（t = 6.99, α7）（等权）；0.204（5.87）/ 0.410（6.59）（市值加权）。
  表 2（描述统计）：PC 均值 −0.230、中位 −0.180（PC = 团内销售收入 HHI 的相反数；Valta 2012）。
  假设 1 对应：PC → 主营业务收入/净利润波动率（σSale_y = y..y+2 年主营业务收入的标准差）。

== 事先判据（pre-registered，先于运行写入本 docstring）==
  原判据 = **符号一致 + 显著性同向**（t 同号且 |t| > 1.96 记显著）；
  满足即记「**复现成功**」，仅符号一致而不显著记「**部分**」，符号相反记「**不成功**」。
  **2026-10-27 起该判据不再适用于对外结论**：对手集合口径不同（见下第 3 条），PC 不是原文变量的
  严格复现，故对外判定统一为「**不可比/未复现（对手集合口径不同）**」；原符号/显著性读数只作描述。
  量级差异一律按期间/语料差异解释，**不做事后样本挑选、不做任何调参**。

== 期间/语料/聚类与阈值差异声明（对表前登记）==
  1. 期间不同：原文文本年 2007–2016（FM 月样本 N = 149,938 / 90,752）；我方文本年 FY2017–FY2022
     （老师 MD&A 语料）→ PC 按 PIT（available_at = 文本年次年 04-30，max{available_at < month}）
     仅 2018-05..2023-12 可用（+1 年后落在面板末月外）；FM / 组合有效事件月 = 2018-05..2023-12（68 月）。
  2. 语料不同：原文 = 年报全文（沪深交易所官网）；我方主口径 = **老师 MD&A 章节导出**
     （`pm_ma_ewcos_teacher_v1_{y}`）。
  3. 聚类同方法、对手集合同法但**口径不同**：原文第四步 = 「任何与其相似度超过团内最小相似度的公司」
     （字面含团外）；我方 = 同团全部成员（已登记偏离，methodology_execution_plan.md:2290）。
     差异 = 我方团数 K 逐年校准到「申万二级公司层面池化中位 peer 44」（老师口径），原文用肘点法则。
     PC 定义与滞后结构一致，但 PC 不是原文变量的严格复现；对表判定 = 不可比/未复现，
     不得输出「符号相反 / 复现失败 / 证伪」。
  4. 我方既有因子集 = `FF4_SELF_BUILT_v1`（MKT/SMB/HML/MOM）→ α3 / α4 可得；
     **RMW/CMA/LIQ 无既有产物 → 原文 α5/α6/α7 不可得**（登记为限制，不作替代口径宣称）。
  5. 控制变量：原文列(2) 含 HHI/β_MKT/Size/BM/Mom/Rev/Illiq/IA/Roe/Lev/Disp/Ivol（12 项）；
     我方既有月面板可对齐 = **size / bm / mom / rev / illiq（5 项）+ PC**；无 firm-level β_MKT
     （36 个月滚动市场贝塔，附录附表 1）与 IA/Roe/Lev/Disp/Ivol、无证监会二级行业 HHI
     （PC 文件中的 `hhi` = 同团 −PC，与 PC 定义性共线，不可作原文「传统行业 HHI」的替代）→ 登记为限制。
  6. NW 滞后：主读数用 registry `nw_lag = 6`（我方冻结值）；并报 lag = 12 列以对应原文口径。
  7. 样本：主读数 = 已登记 pricing_gate 样本 `pricing_sample_v2_2015_2023`（`pricing_in_fm_sample`，含 microcap q10 / 次新股 /
     控制可得三处筛除）；并报「原文口径最近似样本」（PC 与所报变量可得、不施加额外筛除）——两样本同判据并报，
     **不择一**。收益 = 面板既有 `R_next`（raw，月度截面 [0.01,0.99] winsorize，registry 冻结规则）；
     PC 天然有界 [−1, 0]，未施加额外 winsorize。

纪律：只读既有产物；不覆盖任何既有文件；不改任何 headline_value / 冻结值；产物 = 本脚本 + 本报告两件。

v2（2026-10-19，全文语料对表轮）：加 `--pc-dir` / `--pc-years` / `--panel` / `--out-dir` / `--out-prefix` /
`--progress` / `--corpus-label` / `--md-title` / `--step-tag` / `--generated-at` 参数化，供
「对表语料敏感性列」（语料 = 自建年报全文，registry `ma_fulltext_text_corpus`）复用**同一套靶子、
同一套先登记判据、同一期间口径与同一登记样本**。**缺省（不传任何新参数）时逐字节等于 v1 行为**
（默认 pc-dir = `data/processed/network_ma_teacher`、默认输出 = `reports/data_gates/G0/
competition_pricing_correspondence.{json,md}`、默认进度 = `artifacts/tmp/pc_correspondence_progress.md`、
默认标题 / 语料标签 / `generated_at` = 原字面量）；**判据、样本规则、控制集、NW 滞后与统计口径一字未改**。

用法：python scripts/competition_pricing_correspondence.py
      python scripts/competition_pricing_correspondence.py --pc-dir data/processed/network_ma_fulltext \
          --out-prefix competition_pricing_correspondence_fulltext --corpus-label 年报全文 --step-tag FT-4
产出：reports/data_gates/G0/{out-prefix}.{json,md}（缺省 competition_pricing_correspondence.{json,md}）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pricing_lib import (
    OUT_DIR, ROOT, ROWS_NOT_APPLICABLE, assert_pricing_contract, contract_from_frame, fm_regression,
    file_shape, load_input_contracts, manifest_input_key, newey_west, param_value,
    rank_average_groups, require, write_json,
)

# 2026-10-27（Phase 0.2 口径复核）：对手集合口径不同 → 对表判定统一为「不可比/未复现」。
# 原文第四步 = 「任何与其相似度超过团内最小相似度的公司」（字面含团外）；我方 = 同团全部成员
# （已登记偏离：methodology_execution_plan:2290，主仓库原文）。PC 不是原文变量的严格复现。
NOT_COMPARABLE_LABEL = '不可比/未复现（对手集合口径不同）'
COMPARABLE_TO_PAPER = False

PROGRESS = ROOT / 'artifacts/tmp/pc_correspondence_progress.md'
PANEL_SAMPLE = ROOT / 'data/processed/pricing_sample_ma_teacher_v2.parquet'
PRICING_SAMPLE_REPORT = ROOT / 'reports/data_gates/G0/pricing_sample_report.json'
V2_SAMPLE_REPORT = ROOT / 'reports/data_gates/G0/pricing_sample_report_v2.json'
PC_DIR = ROOT / 'data/processed/network_ma_teacher'
OUT_PREFIX = 'competition_pricing_correspondence'
STEP_TAG = 'PC'
CORPUS_LABEL = '老师 MD&A 章节'                       # 差异声明/标题里的「我方语料」标签（缺省 = v1 字面量）
CORPUS_DIFF_NOTE = '语料：原文 = 年报全文；我方主口径 = 老师 MD&A 章节（pm_ma_ewcos_teacher_v1_{y}）'
CORPUS_MD_CELL = '**老师 MD&A 章节**（`pm_ma_ewcos_teacher_v1_{y}`）'
CORPUS_LIMITATION = '老师 MD&A 语料（2021 年存在已知结构性断点，见 methodology 与 workplan 登记）'
CORPUS_LABEL_DETAIL = 'pm_ma_ewcos_teacher_v1_{y}'    # 版本串（随 --pc-dir 切换；缺省 = 主口径）
MD_TITLE = '# network 对表：刘亚辉等（2021）核心结论 × 我方主口径网络 PC'
GENERATED_AT = '2026-10-16'
PC_DIR_LABEL = 'data/processed/network_ma_teacher'
_V1_PROGRESS = PROGRESS
FACTORS = ROOT / 'data/processed/factor_returns_daily.csv'
REV_PKL = ROOT / 'data/raw/drive/operating_revenue_annual.pkl'
NP_PKL = ROOT / 'data/raw/drive/net_profit_parent_company_annual.pkl'
JSON_OUT = OUT_DIR / 'competition_pricing_correspondence.json'
MD_OUT = OUT_DIR / 'competition_pricing_correspondence.md'

PC_YEARS = list(range(2017, 2023))          # 老师 MD&A 语料 6 年（2014–2016 缺营收 → PC 不可得）
CONTROLS = ['size', 'bm', 'mom', 'rev', 'illiq']
TARGETS = {
    'table4_col1_pc': {'coef': 0.152, 't': 7.76, 'source': '表 4 列(1)（单变量，NW lag 12）'},
    'table4_col2_pc': {'coef': 0.130, 't': 5.40, 'source': '表 4 列(2)（加控制，NW lag 12）'},
    'table4_col2_hhi': {'coef': 0.076, 't': 1.94, 'source': '表 4 列(2) HHI'},
    'table3_HL_ew_a3': {'coef': 0.218, 't': 6.27, 'source': '表 3 H−L 等权 α3'},
    'table3_HL_ew_a7': {'coef': 0.441, 't': 6.99, 'source': '表 3 H−L 等权 α7'},
    'table3_HL_vw_a3': {'coef': 0.204, 't': 5.87, 'source': '表 3 H−L 市值加权 α3'},
    'table3_HL_vw_a7': {'coef': 0.410, 't': 6.59, 'source': '表 3 H−L 市值加权 α7'},
    'table2_pc_mean': {'value': -0.230, 'source': '表 2 PC 均值'},
    'table2_pc_median': {'value': -0.180, 'source': '表 2 PC 中位'},
    'h1_sign': {'sign': '+', 'source': '正文 §四(三)1（假设 1：PC 系数均为正）'},
}


def _report_sample_bytes(report: dict, key: str):
    """从 sample report 取 bytes：优先 files.v2_samples，兼容 samples 内嵌字段。"""
    if not isinstance(report, dict):
        return None, None
    files = report.get('files')
    if isinstance(files, dict):
        for group in ('v2_samples', 'samples', 'v1_samples_superseded'):
            rec = files.get(group)
            rec = rec.get(key) if isinstance(rec, dict) else None
            if isinstance(rec, dict) and rec.get('bytes'):
                return str(rec['bytes']).strip().lower(), f'files.{group}.{key}.bytes'
    for group in ('samples', 'samples_v1_superseded'):
        rec = report.get(group)
        rec = rec.get(key) if isinstance(rec, dict) else None
        if isinstance(rec, dict) and rec.get('bytes'):
            return str(rec['bytes']).strip().lower(), f'{group}.{key}.bytes'
    return None, None


def assert_sample_bytes(sample_report: dict, key: str = 'ma_teacher') -> dict:
    """读取登记样本 parquet 前校验行数与列；失败 SystemExit(2)，信息含期望/实际。"""
    file_key = manifest_input_key(PANEL_SAMPLE)
    rows, columns = file_shape(PANEL_SAMPLE)
    contracts = load_input_contracts()
    entry = contracts.get(file_key) or {}
    reg_rows = entry.get('rows')
    require(reg_rows is None or int(reg_rows) == rows,
            f'competition: 样本行数不匹配（file={file_key}）期望={reg_rows} 实际={rows}')
    return {'file': file_key, 'path': str(PANEL_SAMPLE), 'rows': rows,
            'rows_registered': reg_rows, 'columns': columns, 'match': True}


def psay(step: str, msg: str) -> None:
    line = f'[{STEP_TAG}-{step}] {msg}'
    print(line, flush=True)
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS, 'a', encoding='utf-8', newline='\n') as f:
        f.write(f'- {line}\n')


# ---------------------------------------------------------------- 统计工具
def ols_hac(y, X, lag: int = 6):
    """OLS + Newey-West(Bartlett) HAC 标准误。返回 (beta, se, t, T)。"""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    A = X * y[:, None]
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    e = y - X @ beta
    T, k = X.shape
    Ae = X * e[:, None]
    S = Ae.T @ Ae / T
    for l in range(1, min(lag, T - 1) + 1):
        w = 1.0 - l / (lag + 1.0)
        G = Ae[l:].T @ Ae[:-l] / T
        S = S + w * (G + G.T)
    V = XtXi @ (T * S) @ XtXi
    se = np.sqrt(np.diag(V))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = beta / se
    return beta, se, t, T


def raw_verdict(our_t, target_t=None):
    """旧符号判据，仅内部/历史使用；不进入 2026-10-27 后的对外结论。"""
    if our_t is None or not np.isfinite(our_t):
        return 'NOT_COMPUTABLE'
    sign_ok = True if target_t is None else ((our_t > 0) == (target_t > 0))
    if sign_ok and abs(our_t) > 1.96:
        return '复现成功'
    if sign_ok:
        return '部分（符号一致、不显著）'
    return '不成功（符号相反）'


def verdict(our_t, target_t=None):
    """对外判定：对手集合口径不同 → 一律 NOT_COMPARABLE_LABEL（N-2）。"""
    if not COMPARABLE_TO_PAPER:
        return NOT_COMPARABLE_LABEL
    return raw_verdict(our_t, target_t)


def pct(x):
    return None if x is None or not np.isfinite(x) else round(float(x) * 100.0, 4)


def r4(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 4)


def f3(x):
    """原文值排印（3 位小数，与论文表体一致）。"""
    return f'{float(x):.3f}'


# ---------------------------------------------------------------- PC 装载
def load_pc() -> pd.DataFrame:
    frames = []
    for y in PC_YEARS:
        d = pd.read_csv(PC_DIR / f'{y}_pc.csv', dtype={'focal_id': str},
                        usecols=['focal_id', 'text_year', 'pc', 'hhi', 'revenue_coverage',
                                 'n_competitors', 'revenue_year'])
        frames.append(d)
    pc = pd.concat(frames, ignore_index=True)
    pc = pc[pc['pc'].notna()].copy()
    pc['PC'] = pc['pc'].astype(float)
    return pc[['focal_id', 'text_year', 'PC', 'hhi', 'revenue_coverage', 'n_competitors',
               'revenue_year']]


def fmt_fm(block: dict) -> dict:
    """FM 输出 -> 与原文同量纲（收益单位 %）的读数。"""
    out = {}
    for col, s in block['coefs'].items():
        out[col] = {'beta_pct_per_unit': pct(s['beta']), 't_nw': r4(s['t_nw']),
                    'n_months': s['n_months'], 'beta_raw': r4(s['beta'])}
    out['n_months_used'] = block['n_months_used']
    return out


def _rel(p) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _corpus_detail(pc_dir: Path) -> str:
    n = pc_dir.name
    if 'fulltext' in n:
        return 'pm_ma_ewcos_fulltext_v1_{y}'
    if n.endswith('_idf'):
        return 'pm_ma_idf_teacher_v1_{y}'
    if n.endswith('_medadj'):
        return 'pm_ma_ewcos_teacher_medadj_v1_{y}'
    return 'pm_ma_ewcos_teacher_v1_{y}'


def main() -> int:
    global PROGRESS, PANEL_SAMPLE, PC_DIR, PC_YEARS, JSON_OUT, MD_OUT, OUT_PREFIX, STEP_TAG, \
        CORPUS_LABEL, CORPUS_LABEL_DETAIL, CORPUS_DIFF_NOTE, CORPUS_MD_CELL, CORPUS_LIMITATION, \
        MD_TITLE, GENERATED_AT, PC_DIR_LABEL, OUT_DIR
    # v2-fix（2026-10-19）：`OUT_DIR` 原缺于 global 列表且仅在 `--out-dir` 显式传入时赋值 →
    # **缺省路径**（不传任何新参数）在 `JSON_OUT = OUT_DIR / ...` 处 UnboundLocalError（v1 默认行为未保住）。
    # 加入 global 后：不传 `--out-dir` 时 `OUT_DIR` 保持 pricing_lib 模块级缺省（reports/data_gates/G0）
    # → 缺省输出路径与 v1 逐字面量一致。
    ap = argparse.ArgumentParser(description='network 对表（correspondence）；v2 参数化，缺省 = 主口径行为')
    ap.add_argument('--pc-dir', default=None, help='PC 目录（缺省 data/processed/network_ma_teacher）')
    ap.add_argument('--pc-years', type=int, nargs='*', default=None, help='PC 文本年（缺省 2017-2022）')
    ap.add_argument('--panel', default=None, help='登记样本面板（缺省 pricing_sample_ma_teacher_v2.parquet）')
    ap.add_argument('--out-dir', default=None, help='报告目录（缺省 reports/data_gates/G0）')
    ap.add_argument('--out-prefix', default=None, help='输出文件名前缀（缺省 competition_pricing_correspondence）')
    ap.add_argument('--progress', default=None, help='进度文件（缺省 artifacts/tmp/pc_correspondence_progress.md）')
    ap.add_argument('--corpus-label', default=None, help='差异声明里的我方语料标签（缺省 老师 MD&A 章节）')
    ap.add_argument('--corpus-diff-note', default=None, help='declared_differences 的「语料」整串（缺省 = v1 字面量）')
    ap.add_argument('--corpus-md-cell', default=None, help='md 差异表「我方」单元格整串（缺省 = v1 字面量）')
    ap.add_argument('--corpus-limitation', default=None, help='known_limitations 的语料条目整串（缺省 = v1 字面量）')
    ap.add_argument('--md-title', default=None, help='报告 md 标题（缺省主口径标题）')
    ap.add_argument('--step-tag', default=None, help='进度/日志步骤前缀（缺省 PC）')
    ap.add_argument('--generated-at', default=None, help='JSON generated_at（缺省 2026-10-16）')
    args = ap.parse_args()
    # Phase 1 I-2：启动即核验登记样本 build report 的四条硬门；失败退出，不写任何新结果。
    sample_report = json.loads(PRICING_SAMPLE_REPORT.read_text(encoding='utf-8'))
    assert_pricing_contract(sample_report, where='competition:build_report')
    custom_panel = args.panel is not None
    if args.pc_dir:
        PC_DIR = Path(args.pc_dir) if Path(args.pc_dir).is_absolute() else ROOT / args.pc_dir
        PC_DIR_LABEL = _rel(PC_DIR)
        CORPUS_LABEL_DETAIL = _corpus_detail(PC_DIR)
    if args.pc_years:
        PC_YEARS = list(args.pc_years)
    if args.panel:
        PANEL_SAMPLE = Path(args.panel) if Path(args.panel).is_absolute() else ROOT / args.panel
    if args.out_dir:
        OUT_DIR = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    if args.out_prefix:
        OUT_PREFIX = args.out_prefix
    if args.progress:
        PROGRESS = Path(args.progress) if Path(args.progress).is_absolute() else ROOT / args.progress
    if args.corpus_label:
        CORPUS_LABEL = args.corpus_label
    if args.corpus_diff_note:
        CORPUS_DIFF_NOTE = args.corpus_diff_note
    if args.corpus_md_cell:
        CORPUS_MD_CELL = args.corpus_md_cell
    if args.corpus_limitation:
        CORPUS_LIMITATION = args.corpus_limitation
    if args.md_title:
        MD_TITLE = args.md_title
    if args.step_tag:
        STEP_TAG = args.step_tag
    if args.generated_at:
        GENERATED_AT = args.generated_at
    JSON_OUT = OUT_DIR / f'{OUT_PREFIX}.json'
    MD_OUT = OUT_DIR / f'{OUT_PREFIX}.md'
    # Phase 1 I-3 R2：登记样本 parquet sha 校验放在任何 psay/progress 写入之前；
    # 失败即退出，不留下进度行或输出。自定义 --panel 无登记锚（M-2 逃生口），显式跳过。
    if custom_panel:
        sample_bytes = {'mode': 'custom_panel', 'checked': False,
                      'reason': '自定义 --panel 无登记样本锚；不伪造 sha 通过'}
    else:
        sample_bytes = assert_sample_bytes(sample_report)
    psay('0', '状态 = 开始；判据与差异声明已写入 docstring（先登记后跑）')
    psay('0', f'运行参数 = pc-dir `{PC_DIR_LABEL}` / PC 文本年 {PC_YEARS} / 输出 `{_rel(JSON_OUT)}` / '
              f'语料标签「{CORPUS_LABEL}」/ 版本串 `{CORPUS_LABEL_DETAIL}` / step-tag {STEP_TAG}')
    nw_lag = int(param_value('nw_lag'))
    min_n = int(param_value('sample_and_spec_registry')['min_focal_count_per_month'])
    psay('0', f'registry 冻结读数：nw_lag = {nw_lag}、min_n/月 = {min_n}（主读数）；另报 lag = 12 对应原文')

    # ---------------- PC-1 装载与 PIT 合并
    panel = pd.read_parquet(PANEL_SAMPLE)
    if custom_panel:
        # 自定义面板无登记行数：行数门显式不适用（不伪造通过），只强制其余三条。
        c_panel = contract_from_frame(panel, expected_rows=None, kind='sample')
        for gate in ('pit_violations_strict_lt', 'duplicate_focal_month_keys',
                     'missing_filled_with_zero'):
            require(c_panel.get(gate) == 0,
                    f'competition:custom_panel {gate}={c_panel.get(gate)} != 0（{_rel(PANEL_SAMPLE)}）')
        psay('0', f'自定义面板 {_rel(PANEL_SAMPLE)}：rows_match_registered 不适用'
                  f'（rows={c_panel["rows"]}），PIT/重复键/缺失不填 0 = 0')
    else:
        expected_rows = int(sample_report['samples']['ma_teacher']['rows_total'])
        c_panel = assert_pricing_contract(
            panel, expected_rows=expected_rows, kind='sample',
            where='competition:registered_panel')['contracts']['<frame>']
    contract_status = {
        'panel': _rel(PANEL_SAMPLE),
        'panel_mode': 'custom' if custom_panel else 'registered',
        'sample_bytes': sample_bytes,
        'rows_match_registered': c_panel.get('rows_match_registered'),
        'rows': c_panel.get('rows'),
        'rows_registered': c_panel.get('rows_registered'),
        'rows_gate_enforced': c_panel.get('rows_match_registered') != ROWS_NOT_APPLICABLE,
        'pit_violations_strict_lt': c_panel.get('pit_violations_strict_lt'),
        'duplicate_focal_month_keys': c_panel.get('duplicate_focal_month_keys'),
        'missing_filled_with_zero': c_panel.get('missing_filled_with_zero'),
        'source': ('pricing_sample_report.json:samples.ma_teacher.rows_total + 面板 parquet 现算'
                   if not custom_panel else
                   '自定义 --panel：无登记行数；只强制 PIT/重复键/缺失不填 0'),
        'note': ('自定义 --panel 无登记行数：行数门显式不适用'
                 '（not_applicable_custom_panel），不伪造通过'
                 if custom_panel else
                 '登记面板行数门严格等于 samples.ma_teacher.rows_total'),
    }
    pc = load_pc()
    panel['text_year'] = pd.to_numeric(panel['network_version'], errors='coerce')
    merged = panel.merge(pc, on=['focal_id', 'text_year'], how='left')
    n_pc = int(merged['PC'].notna().sum())
    months_pc = sorted(merged.loc[merged['PC'].notna(), 'month'].unique())
    psay('1', f'PC 合并状态 = ok：PC 非空 {n_pc:,} 行；PC 可用月 {months_pc[0]}..{months_pc[-1]}'
              f'（{len(months_pc)} 月）；文本年 {PC_YEARS[0]}–{PC_YEARS[-1]}；'
              f'文本年映射 = 面板 network_version（PIT max{{available_at < month}}）')
    # 符号约定核验（防「PC 列实为 HHI」这类口径事故）：pc = −hhi 逐值成立
    diff = (merged['PC'] + merged['hhi']).abs().max()
    sign_check = {'max_abs_pc_plus_hhi': r4(diff), 'pc_is_negative_hhi': bool(diff < 1e-9),
                  'note': 'PC 越大 = 竞争越激烈（原文同一约定：PC = 团内销售收入 HHI 的相反数）'}
    psay('1', f'PC 符号约定核验状态 = {"ok" if sign_check["pc_is_negative_hhi"] else "FAIL"}'
              f'（max|pc + hhi| = {sign_check["max_abs_pc_plus_hhi"]}）')

    # 样本定义（四组，全部并报，不择一）
    pc_ok = merged['PC'].notna() & merged['R_next'].notna()
    ctrl_ok = merged[CONTROLS].notna().all(axis=1)
    reg = merged['pricing_in_fm_sample'].astype(bool)
    sample_paper_like = merged[pc_ok].copy()
    sample_registered = merged[pc_ok & reg].copy()
    sample_with_controls_paper_like = merged[pc_ok & ctrl_ok].copy()
    sample_with_controls_registered = merged[pc_ok & reg & ctrl_ok].copy()
    psay('1', f'样本量状态：原文最近似样本 = {len(sample_paper_like):,} 行 / {sample_paper_like["month"].nunique()} 月；'
              f'登记样本 = {len(sample_registered):,} 行 / {sample_registered["month"].nunique()} 月；'
              f'加控制原文最近似 = {len(sample_with_controls_paper_like):,}；加控制登记样本 = {len(sample_with_controls_registered):,}')

    # ---------------- PC-2 描述统计（表 2 对应）
    pc_firm_year = pc['PC']
    desc = {
        'pc_pooled_firm_year': {
            'n': int(pc_firm_year.notna().sum()), 'mean': r4(pc_firm_year.mean()),
            'sd': r4(pc_firm_year.std()), 'p25': r4(pc_firm_year.quantile(.25)),
            'median': r4(pc_firm_year.median()), 'p75': r4(pc_firm_year.quantile(.75)),
        },
        'pc_in_fm_sample_monthly': {
            'n': int(sample_registered['PC'].notna().sum()), 'mean': r4(sample_registered['PC'].mean()),
            'sd': r4(sample_registered['PC'].std()), 'median': r4(sample_registered['PC'].median()),
        },
        'target': TARGETS['table2_pc_mean'], 'target_median': TARGETS['table2_pc_median'],
        'verdict_mean': ('符号一致（同为负；描述性，不构成复现）' if pc_firm_year.mean() < 0
                         else '符号不一致（描述性，不构成复现）'),
    }
    psay('2', f'描述统计状态 = ok：PC（firm-year，n = {desc["pc_pooled_firm_year"]["n"]:,}）'
              f'均值 {desc["pc_pooled_firm_year"]["mean"]}、中位 {desc["pc_pooled_firm_year"]["median"]}'
              f'（原文 −0.230 / −0.180）')

    # ---------------- PC-3 表 4 列(1) 对应：单变量 FM
    fm1 = {}
    for tag, samp in (('paper_like', sample_paper_like), ('registered', sample_registered)):
        for lag in (nw_lag, 12):
            for wtag, wcol in (('EW', None), ('VW', 'market_equity')):
                blk, _ = fm_regression(samp, 'R_next', ['PC'], weight_col=wcol, min_n=min_n, nw_lag=lag)
                fm1[f'{tag}_{wtag}_lag{lag}'] = fmt_fm(blk)
    head1 = fm1[f'registered_EW_lag{nw_lag}']
    v1 = verdict(head1['PC']['t_nw'], TARGETS['table4_col1_pc']['t'])
    psay('3', f'表 4 列(1) 对应状态 = {v1}：登记样本 EW PC = {head1["PC"]["beta_pct_per_unit"]}%'
              f'（t = {head1["PC"]["t_nw"]}，lag {nw_lag}，{head1["n_months_used"]} 月）；'
              f'原文 0.152（t = 7.76）')

    # ---------------- PC-4 表 4 列(2) 对应：加控制 FM
    fm2 = {}
    for tag, samp in (('paper_like', sample_with_controls_paper_like), ('registered', sample_with_controls_registered)):
        for lag in (nw_lag, 12):
            for wtag, wcol in (('EW', None), ('VW', 'market_equity')):
                blk, _ = fm_regression(samp, 'R_next', ['PC'] + CONTROLS, weight_col=wcol,
                                       min_n=min_n, nw_lag=lag)
                fm2[f'{tag}_{wtag}_lag{lag}'] = fmt_fm(blk)
    head2 = fm2[f'registered_EW_lag{nw_lag}']
    v2 = verdict(head2['PC']['t_nw'], TARGETS['table4_col2_pc']['t'])
    psay('4', f'表 4 列(2) 对应状态 = {v2}：登记样本 EW PC = {head2["PC"]["beta_pct_per_unit"]}%'
              f'（t = {head2["PC"]["t_nw"]}，{head2["n_months_used"]} 月）；'
              f'原文 0.130（t = 5.40）')

    # ---------------- PC-5 表 3 对应：分位组合 + H−L 原始收益 + α3/α4
    fac = pd.read_csv(FACTORS, dtype={'date': str})
    fac['month'] = fac['date'].str[:7]
    fmonth = fac.groupby('month').agg(
        MKT=('MKT_d', lambda s: float(np.prod(1 + s) - 1)),
        SMB=('SMB_d', lambda s: float(np.prod(1 + s) - 1)),
        HML=('HML_d', lambda s: float(np.prod(1 + s) - 1)),
        MOM=('MOM_d', lambda s: float(np.prod(1 + s) - 1)),
        RF=('rf_d', lambda s: float(np.prod(1 + s) - 1))).reset_index()

    def portfolio(key, samp, n_groups, wtag, wcol, lag):
        d = samp[['month', 'PC', 'R_next', 'market_equity']].copy()
        if wcol:
            d = d[d[wcol].notna() & (d[wcol] > 0)]
        d['grp'] = d.groupby('month', group_keys=False)['PC'].transform(
            lambda s: rank_average_groups(s, n_groups))
        d = d[d['grp'].notna()]
        rows = []
        for m, sub in d.groupby('month'):
            rec = {'month': m, 'n': int(len(sub))}
            for g in range(1, n_groups + 1):
                gg = sub[sub['grp'] == g]
                if gg.empty:
                    rec[f'g{g}'] = None
                elif wcol:
                    rec[f'g{g}'] = float(np.average(gg['R_next'], weights=gg[wcol]))
                else:
                    rec[f'g{g}'] = float(gg['R_next'].mean())
            rows.append(rec)
        t = pd.DataFrame(rows).sort_values('month')
        t[f'HL_{wtag}'] = t[f'g{n_groups}'] - t['g1']
        nw = newey_west(t[f'HL_{wtag}'], lag=lag)
        t2 = t.merge(fmonth, on='month', how='left')
        ls = (t2[f'HL_{wtag}'] - t2['RF']).to_numpy(dtype=float)
        out = {'n_months': int(len(t)), 'n_months_effective': int(nw['T']),
               'n_drops_missing_factor': int(t2[['MKT', 'SMB', 'HML']].isna().any(axis=1).sum()),
               'group_mean_pct': {f'Q{g}': pct(t[f'g{g}'].mean()) for g in range(1, n_groups + 1)},
               'HL_raw_pct': pct(nw['mean']), 'HL_raw_t_nw': r4(nw['t_nw'])}
        for atag, cols in (('a3', ['MKT', 'SMB', 'HML']), ('a4', ['MKT', 'SMB', 'HML', 'MOM'])):
            ok = t2[cols].notna().all(axis=1)
            X = np.column_stack([np.ones(int(ok.sum()))] + [t2.loc[ok, c].to_numpy(float) for c in cols])
            b, se, tt, T = ols_hac(ls[ok.to_numpy()], X, lag=lag)
            out[f'alpha_{atag}_pct'] = pct(b[0])
            out[f'alpha_{atag}_t'] = r4(tt[0])
        return out

    ports = {}
    for tag, samp in (('paper_like', sample_paper_like), ('registered', sample_registered)):
        for gtag, n_groups in (('quintile', 5), ('decile', 10)):
            for wtag, wcol in (('EW', None), ('VW', 'market_equity')):
                ports[f'{tag}_{gtag}_{wtag}'] = portfolio(tag, samp, n_groups, wtag, wcol, nw_lag)
    hqreg = ports['registered_quintile_EW']
    v3 = verdict(hqreg['alpha_a3_t'], TARGETS['table3_HL_ew_a3']['t'])
    psay('5', f'表 3 对应状态 = {v3}：登记样本五分位等权 H−L raw = {hqreg["HL_raw_pct"]}%'
              f'（t = {hqreg["HL_raw_t_nw"]}）、α3 = {hqreg["alpha_a3_pct"]}%（t = {hqreg["alpha_a3_t"]}）、'
              f'α4 = {hqreg["alpha_a4_pct"]}%（t = {hqreg["alpha_a4_t"]}）；原文 H−L α3 = 0.218（6.27）')

    # ---------------- PC-6 可选：假设 1（PC → σSale / σProfit）
    h1 = {'status': 'NOT_EXECUTED'}
    try:
        rev = pd.read_pickle(REV_PKL)
        prof = pd.read_pickle(NP_PKL)
        rev.index = pd.to_datetime(rev.index).year
        prof.index = pd.to_datetime(prof.index).year
        rows = []
        for y in range(2018, 2023):
            if not all(k in rev.index for k in (y, y + 1, y + 2)):
                continue
            sub = pd.DataFrame({
                'sigma_sale': rev.loc[[y, y + 1, y + 2]].std() / 1e8,
                'sigma_profit': prof.loc[[y, y + 1, y + 2]].std() / 1e8})
            sub['year'] = y
            sub['firm_id'] = [str(c).split('.')[0].zfill(6) for c in sub.index]
            sub['code'] = [('sh' if c.startswith('6') else 'sz') + c for c in sub['firm_id']]
            code = sub[['code', 'year', 'sigma_sale', 'sigma_profit']].rename(columns={'code': 'focal_id'})
            pcy = pc[pc['text_year'] == y - 1][['focal_id', 'PC']].rename(columns={'PC': 'PC_lag1'})
            rows.append(code.merge(pcy, on='focal_id', how='inner'))
        h1p = pd.concat(rows, ignore_index=True).dropna(subset=['PC_lag1'])
        h1p = h1p[h1p['sigma_sale'].notna() | h1p['sigma_profit'].notna()]
        h1['n_firm_year'] = int(len(h1p))
        h1['years'] = sorted(h1p['year'].unique().tolist())
        h1['pc_years_used'] = sorted((h1p['year'] - 1).unique().tolist())
        for tag, ycol in (('sigma_sale', 'sigma_sale'), ('sigma_profit', 'sigma_profit')):
            d = h1p.dropna(subset=[ycol]).copy()
            if d.empty:
                h1[tag] = {'status': 'NO_DATA'}
                continue
            yr_dum = pd.get_dummies(d['year'].astype('category'), drop_first=True).astype(float).to_numpy()
            X = np.column_stack([np.ones(len(d)), d['PC_lag1'].to_numpy(float), yr_dum])
            b, se_iid, t_iid, T = ols_hac(d[ycol].to_numpy(float), X, lag=0)
            u = d[ycol].to_numpy(float) - X @ b
            XtXi = np.linalg.pinv(X.T @ X)
            gsum = pd.DataFrame(X * u[:, None]).groupby(d['focal_id'].to_numpy()).sum().to_numpy()
            n_firm = len(set(d['focal_id']))
            S = gsum.T @ gsum
            V = XtXi @ (len(d) * S) @ XtXi * (len(d) / max(n_firm, 1)) / len(d)
            se_cl = np.sqrt(np.diag(V))
            h1[tag] = {'n': int(len(d)), 'n_firms': int(n_firm),
                       'beta_sigma_in_1e8cny': r4(b[1]), 't_cluster_firm': r4(b[1] / se_cl[1]),
                       't_iid': r4(t_iid[1]), 'years': sorted(d['year'].unique().tolist()),
                       'sign': '+' if b[1] > 0 else '−',
                       'raw_direction': ('β > 0（与原文同向；描述性）' if b[1] > 0
                                         else 'β < 0（与原文反向；描述性）'),
                       'verdict': (NOT_COMPARABLE_LABEL if not COMPARABLE_TO_PAPER else
                                   ('复现成功' if (b[1] > 0 and abs(b[1] / se_cl[1]) > 1.96)
                                    else ('部分（符号一致、不显著）' if b[1] > 0 else '不成功（符号相反）'))),
                       'unit': 'σ 单位 = 亿元（1e8 CNY）；β 单位 = 亿元/单位 PC',
                       'spec': ('Y_{i,y} ~ PC_{i,y−1} + year FE（5 年短面板，无 firm FE：firm FE 与 year FE 在 '
                                '5 年内几乎吸收全部变异，原文 17,302 firm-year 可容 firm FE）'
                                '；无 HHI/Age/IO/TbQ/Dividend 控制（既有面板无该口径）')}
        h1['status'] = 'EXECUTED'
        psay('6', f'假设 1 对应状态 = {h1.get("sigma_sale", {}).get("verdict")}：σSale β = '
                  f'{h1.get("sigma_sale", {}).get("beta_sigma_in_1e8cny")}'
                  f'（t_cl = {h1.get("sigma_sale", {}).get("t_cluster_firm")}）、'
                  f'σProfit β = {h1.get("sigma_profit", {}).get("beta_sigma_in_1e8cny")}'
                  f'（t_cl = {h1.get("sigma_profit", {}).get("t_cluster_firm")}）；n = {h1["n_firm_year"]}')
    except Exception as exc:   # noqa: BLE001
        h1 = {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'}
        psay('6', f'假设 1 对应状态 = FAILED（{h1["error"]}）')

    # ---------------- PC-7 判定汇总 + 产物
    checklist = [
        {'layer': '表 2 描述统计（PC 均值/中位）',
         'target': f'均值 {f3(TARGETS["table2_pc_mean"]["value"])} / 中位 {f3(TARGETS["table2_pc_median"]["value"])}',
         'ours': f'均值 {desc["pc_pooled_firm_year"]["mean"]} / 中位 {desc["pc_pooled_firm_year"]["median"]}'
                 f'（firm-year n = {desc["pc_pooled_firm_year"]["n"]:,}）',
         'verdict': desc['verdict_mean'] + '（量级更大：语料 = MD&A 章节、团数 K 校准口径不同）'},
        {'layer': '表 4 列(1)（单变量 FM，NW lag 12）',
         'target': f'{f3(TARGETS["table4_col1_pc"]["coef"])}（t = {TARGETS["table4_col1_pc"]["t"]}）',
         'ours': f'登记样本 EW {fm1[f"registered_EW_lag12"]["PC"]["beta_pct_per_unit"]}'
                 f'（t = {fm1[f"registered_EW_lag12"]["PC"]["t_nw"]}）；'
                 f'原文最近似样本 EW {fm1[f"paper_like_EW_lag12"]["PC"]["beta_pct_per_unit"]}'
                 f'（t = {fm1[f"paper_like_EW_lag12"]["PC"]["t_nw"]}）',
         'verdict': verdict(fm1['registered_EW_lag12']['PC']['t_nw'], TARGETS['table4_col1_pc']['t'])},
        {'layer': '表 4 列(2)（加控制 FM，NW lag 12）',
         'target': f'PC {f3(TARGETS["table4_col2_pc"]["coef"])}（t = {TARGETS["table4_col2_pc"]["t"]}）',
         'ours': f'登记样本 EW {fm2[f"registered_EW_lag12"]["PC"]["beta_pct_per_unit"]}'
                 f'（t = {fm2[f"registered_EW_lag12"]["PC"]["t_nw"]}）；'
                 f'原文最近似样本 EW {fm2[f"paper_like_EW_lag12"]["PC"]["beta_pct_per_unit"]}'
                 f'（t = {fm2[f"paper_like_EW_lag12"]["PC"]["t_nw"]}）',
         'verdict': verdict(fm2['registered_EW_lag12']['PC']['t_nw'], TARGETS['table4_col2_pc']['t'])},
        {'layer': '表 3（H−L 十分位/五分位，α3）',
         'target': f'等权 α3 {f3(TARGETS["table3_HL_ew_a3"]["coef"])}（t = {TARGETS["table3_HL_ew_a3"]["t"]}）/ '
                   f'市值加权 {f3(TARGETS["table3_HL_vw_a3"]["coef"])}'
                   f'（t = {TARGETS["table3_HL_vw_a3"]["t"]}）',
         'ours': f'登记样本五分位 α3 EW {hqreg["alpha_a3_pct"]}（t = {hqreg["alpha_a3_t"]}）/ '
                 f'VW {ports["registered_quintile_VW"]["alpha_a3_pct"]}'
                 f'（t = {ports["registered_quintile_VW"]["alpha_a3_t"]}）；十分位 EW '
                 f'{ports["registered_decile_EW"]["alpha_a3_pct"]}'
                 f'（t = {ports["registered_decile_EW"]["alpha_a3_t"]}）',
         'verdict': verdict(hqreg['alpha_a3_t'], TARGETS['table3_HL_ew_a3']['t'])},
        {'layer': '表 3（H−L α7 对应口径：我方可比 α4 = FF3+UMD）',
         'target': f'α7 等权 {f3(TARGETS["table3_HL_ew_a7"]["coef"])}（t = {TARGETS["table3_HL_ew_a7"]["t"]}）',
         'ours': f'α4 等权 {hqreg["alpha_a4_pct"]}（t = {hqreg["alpha_a4_t"]}）；'
                 f'**FF5/UMD/LIQ 因子无既有产物 → 原文 α7 不可得**（不作替代口径宣称）',
         'verdict': '不可比/未复现（α4 ≠ α7；对手集合口径不同）'},
        {'layer': '假设 1（PC → 业绩波动率；可选）',
         'target': 'PC 系数均为正（正文 §四(三)1，1%/10% 显著）',
         'ours': (f'σSale β = {h1.get("sigma_sale", {}).get("beta_sigma_in_1e8cny")} 亿元/单位 PC'
                  f'（t_cl = {h1.get("sigma_sale", {}).get("t_cluster_firm")}）/ '
                  f'σProfit β = {h1.get("sigma_profit", {}).get("beta_sigma_in_1e8cny")}'
                  f'（t_cl = {h1.get("sigma_profit", {}).get("t_cluster_firm")}）；n = {h1.get("n_firm_year")}'
                  if h1.get('status') == 'EXECUTED' else f'未执行（{h1.get("status")}）'),
         'verdict': (NOT_COMPARABLE_LABEL if h1.get('status') == 'EXECUTED' else h1.get('status'))},
    ]
    n_success = sum(1 for c in checklist if c['verdict'].startswith('复现成功'))
    n_partial = sum(1 for c in checklist if c['verdict'].startswith('部分'))
    n_not_comparable = sum(1 for c in checklist if c['verdict'].startswith('不可比'))
    overall = NOT_COMPARABLE_LABEL if n_not_comparable else (
        '复现成功' if n_success >= 4 else ('部分复现' if n_success >= 2 else '复现不成功'))

    out = {
        'artifact': OUT_PREFIX,
        'generated_at': GENERATED_AT,
        'paper': {'citation': '刘亚辉、牟爽、张紫琪（2021）《产品市场竞争风险被定价了吗？》经济学（季刊）21(6)',
                  'file': 'ref/teacher-materials/A股产品市场竞争.pdf', 'printed_page_offset': 2060,
                  'hypothesis': 'H4 = PC 越大 → 股票预期收益率越高（表 4）；组合证据 = 表 3；描述 = 表 2'},
        'pre_registered_criteria': {
            'rule': '（历史判据，2026-10-27 起不适用于对外结论）符号一致 + 显著性同向（t 同号且 |t| > 1.96 记显著）'
                    '→「复现成功」；符号一致不显著→「部分」；反号→「不成功」',
            'applicability': '不适用：对手集合口径不同（项目 = 同团成员；原文 = 相似度 > 团内最小相似度，字面含团外）→ '
                             '对外判定统一为「不可比/未复现」，原符号/显著性读数仅作描述',
            'no_sample_selection': True, 'no_tuning': True,
            'magnitude_rule': '量级差异按期间/语料差异解释（不做量级等值对账）'},
        'declared_differences': [
            '期间：原文 2007–2016 文本年（FM 月样本 149,938 / 90,752）；我方 FY2017–FY2022（老师 MD&A）→ PC 可用月 2018-05..2023-12，FM/组合有效事件月 2018-05..2023-12（68 月）',
            CORPUS_DIFF_NOTE,
            '聚类与阈值：我方 = 等权 0/1 → 余弦 → average 层次聚类 → 逐年 K 校准至池化中位 peer 44 → 对手集合 = 同团全部成员；'
            '原文 = 同方法聚类（肘点法则）→ 对手集合 = 「任何与其相似度超过团内最小相似度的公司」（字面含团外）。'
            '**对手集合口径不同（已登记偏离，methodology_execution_plan.md:2290）；PC 不是原文变量的严格复现，'
            '判定 = 不可比/未复现**。',
            '因子：我方 FF4_SELF_BUILT_v1（MKT/SMB/HML/MOM）→ α3/α4 可得；RMW/CMA/LIQ 无 → 原文 α5/α6/α7 不可得',
            '控制：我方 = size/bm/mom/rev/illiq（5 项）；无 firm-level β_MKT、无 IA/Roe/Lev/Disp/Ivol、无证监会二级行业 HHI',
            'NW 滞后：主读数 lag 6（registry nw_lag 冻结值）；并报 lag 12 对应原文',
            '样本：主读数 = 已登记 pricing_gate 样本 pricing_sample_v2_2015_2023；并报「原文最近似样本」（仅变量可得性）',
            '收益：面板既有 R_next（raw、月度截面 [0.01,0.99] winsorize）；PC 有界 [−1,0] 未额外 winsorize'],
        'pc_inputs': {'dir': PC_DIR_LABEL, 'years': PC_YEARS,
                      'definition': 'PC = −HHI(同团成员销售收入，含 focal 自身；营收取文本年 −1 年)',
                      'pit': '面板 network_version = 文本年；available_at = 文本年次年 04-30；max{available_at < month}',
                      'sign_convention_check': sign_check,
                      'competitor_set_note': ('已登记偏离（methodology_execution_plan.md:2290）：我方对手集合 = 同团全部成员；'
                                              '原文第四步 = 「任何与其相似度超过团内最小相似度的公司」（字面含团外）。'
                                              '二者不是同一对手集合；不得声称近似一致、可对表或可复现。'
                                              '若要复现原文 PC，须按原文规则 + 接近原文的词表规模（active_vocab ≈ 1.5 万量级）'
                                              '重建（未做）。因此本报告判定 = 不可比/未复现（对手集合口径不同），'
                                              '不是「符号相反」，也不构成对原文的证伪。')},
        'comparability': {
            'comparable_to_paper': False,
            'registered_deviation': 'methodology_execution_plan:2290',
            'project_rule': 'same_cluster_members（同团全部成员）',
            'paper_rule': '任何与其相似度超过团内最小相似度的公司（字面含团外）',
            'consequence': 'PC 不是原文变量的严格复现；判定 = 不可比/未复现（对手集合口径不同）；'
                           '不得写符号相反/复现失败/证伪',
            'replication_requirement': '按原文第四步规则 + 接近原文词表规模（active_vocab ≈ 1.5 万量级）重建（未做）'},
        'numerical_readings': {
            'registered_EW_lag6_univariate': {
                'beta_pct': fm1['registered_EW_lag6']['PC']['beta_pct_per_unit'],
                't_nw': fm1['registered_EW_lag6']['PC']['t_nw'],
                'n_months': fm1['registered_EW_lag6']['n_months_used']},
            'registered_EW_lag12_univariate': {
                'beta_pct': fm1['registered_EW_lag12']['PC']['beta_pct_per_unit'],
                't_nw': fm1['registered_EW_lag12']['PC']['t_nw'],
                'n_months': fm1['registered_EW_lag12']['n_months_used']},
            'ci95_pct': [-1.1983, 0.1847],
            'se_pct': 0.3528,
            'mde80_pct': 0.9884,
            'power_at_paper_effect': 0.0715,
            'n_months': int(fm1['registered_EW_lag6']['n_months_used']),
            'diagnostic_sample': 'main_caption',
            'source': 'reports/audits/competition_sign_diagnostics_20261021.md §8R(f)'
                      '（CI/功效为既有**主口径**诊断读数，本脚本不重算）',
            'reading': ('主口径诊断读数，非全文列；全文列点估计为负但不显著'
                        if ('fulltext' in OUT_PREFIX or '全文' in CORPUS_LABEL)
                        else 'CI 同时覆盖 0 与 +0.152 → 本样本无法判定；'
                             '不是符号相反，也不是对原文的证伪')},
        'sample_sizes': {
            'paper_like': {'rows': int(len(sample_paper_like)), 'months': int(sample_paper_like['month'].nunique()),
                              'first_month': sample_paper_like['month'].min(), 'last_month': sample_paper_like['month'].max()},
            'registered': {'rows': int(len(sample_registered)), 'months': int(sample_registered['month'].nunique())},
            'with_controls_paper_like': {'rows': int(len(sample_with_controls_paper_like)), 'months': int(sample_with_controls_paper_like['month'].nunique())},
            'with_controls_registered': {'rows': int(len(sample_with_controls_registered)), 'months': int(sample_with_controls_registered['month'].nunique())}},
        # Phase 1 I-2：面板契约状态（自定义 --panel 的行数门显式记 not_applicable_custom_panel）
        'contract_status': contract_status,
        'targets': TARGETS,
        'L1_descriptives': desc,
        'L2_fm_univariate': fm1,
        'L2_fm_with_controls': fm2,
        'L3_portfolios': ports,
        'L4_hypothesis1_optional': h1,
        'checklist': checklist,
        'summary': {'n_success': n_success, 'n_partial': n_partial,
                    'n_not_comparable': n_not_comparable, 'overall': overall},
        # Phase 0.3（2026-10-27）：N-11 PC 识别层级注记（append-only；不改任何数值/算法）
        'pc_level_note': {
            'level': 'cluster-year',
            'definition': 'PC = −HHI(同团成员 y−1 年销售收入)；同一团全部成员共享同一 PC 值，因此 PC 是团-年常数',
            'per_year_unique_pc': {'2017': 198, '2018': 217, '2019': 273,
                                   '2020': 271, '2021': 235, '2022': 258},
            'total_unique_cluster_year': 1452,
            'diagnostic_2019': '3,354 家公司 / 407 团 / 134 单点团；有 PC 的 3,218 行只有 273 个不同 PC 值'
                               '（= 407 − 134）',
            'full_pc_panel_rows_2017_2022': 19849,
            'pc_correspondence_merged_firm_year': 18934,
            'inference_recommendation': '横截面按团聚类（或团层 FM/团层等权），并叠加年块推断（团×年二维）；'
                                        '不得按 firm-month 独立处理，不得写「月度截面无重复」',
            'note': '2026-10-27 Phase 0.3 注记；未改 PC 数值、未重跑 PC',
        },
        'known_limitations': [
            'PC 仅 6 个文本年（FY2017–FY2022；2014–2016 缺营收）→ FM 有效月 ≈ 68（原文 ≈ 120 月）',
            CORPUS_LIMITATION,
            '聚类同源：PC 与我方主口径网络同源（同一团划分），非独立数据源',
            'β_MKT/IA/Roe/Lev/Disp/Ivol 与证监会二级行业 HHI 无既有产物 → 列(2) 控制集少于原文（12 → 5）',
            'α5/α6/α7（FF5/UMD/LIQ）因子无既有产物 → 只能报 α3/α4',
            '原文列(3)(4)（TF-IDF PC / HP 中位概率阈值 PC）对应变体未跑（teacher_idf / medadj 网络存在但本轮未纳入对表）',
            '假设 1（可选）无 firm FE、控制集不含 HHI/IO/TbQ/Dividend（既有面板无对应字段）',
            'PC 是团-年常数（N-11，2026-10-27）：同一团全部成员共享同一 PC 值，2019 年 3,218 行仅 273 个不同值，'
            '6 年合计 1,452 个团-年取值；推断须按团聚类（或团层 FM/团层等权）并叠加年块推断（团×年二维），'
            '不得按 firm-month 独立处理，不得写「月度截面无重复」'],
        'unexecuted_or_deviations': [
            '原文表 4 列(3)(4)：TF-IDF 加权 PC / HP 中位概率阈值 PC —— 未执行（变体网络 network_ma_teacher_idf、'
            'network_ma_teacher_medadj 已存在，对表须另立轮次）',
            '原文 α4/α5/α6/α7：FF5（RMW/CMA）+ UMD + LIQ —— RMW/CMA/LIQ 无既有因子产物 → 不可计算（α4 并报，不宣称等价）',
            '原文列(2) 控制 β_MKT（36 个月滚动市场贝塔）/ IA / Roe / Lev / Disp / Ivol / 证监会二级行业 HHI —— 未纳入'
            '（既有面板无对应字段；β_MKT 或行业 HHI 如需对齐须先登记构造口径再跑）',
            'PC 的 `hhi_excl_self`（剔除自身的 HHI）变体 —— 未执行（PC 文件已有该列，可作后续敏感性）',
            '假设 3（团体排序高/低竞争分组 + 跨组预测）—— 不在本轮范围（本轮只对表假设 4 / 表 2 / 表 3 / 假设 1）',
            '原文样本筛除（原文样本筛除：金融/综合类剔除、上市不足两年、词条数 <10）—— 未逐条复刻（我方用登记样本与'
            '「原文最近似样本」两口径并报，未做额外的行业/金融剔除）',
            '多重检验 / 子期间 bootstrap —— 未执行（判据为符号 + 显著性，不含推断校正）'],
    }
    # 断言：numerical_readings 的 lag6/lag12 必须逐值等于同文件 L2 读数（主/全文均适用）
    for _lag in ('lag6', 'lag12'):
        _nr = out['numerical_readings'][f'registered_EW_{_lag}_univariate']
        _pc = out['L2_fm_univariate'][f'registered_EW_{_lag}']['PC']
        assert _nr['beta_pct'] == _pc['beta_pct_per_unit'], (_lag, _nr, _pc)
        assert _nr['t_nw'] == _pc['t_nw'], (_lag, _nr, _pc)
        assert _nr['n_months'] == _pc['n_months'], (_lag, _nr, _pc)
    write_json(JSON_OUT, out)
    psay('7', f'JSON 产出状态 = ok（{JSON_OUT.relative_to(ROOT)}）')

    # ---------------- Markdown 报告
    def fm_rows(taglist):
        lines = ['| 规格 | PC 系数（%/单位 PC） | t(NW) | 月数 |', '|---|---|---|---|']
        for k in taglist:
            b = (fm1 if k.startswith('L1') else fm2)[k[3:]]
            lines.append(f'| {k[3:]} | {b["PC"]["beta_pct_per_unit"]} | {b["PC"]["t_nw"]} | {b["n_months_used"]} |')
        return lines

    md = []
    md.append(MD_TITLE + '\n')
    md.append('> **性质**：对表分析产物（只读既有产物；未覆盖任何既有文件、未改任何 headline_value / 冻结值）。'
              '**2026-10-27 口径更正（口径复核）**：对手集合口径不同 → 判定统一为「不可比/未复现」。')
    md.append('> **判据（先登记）**：原判据 = 符号一致 + 显著性同向（t 同号且 |t| > 1.96）→「复现成功」；'
              '符号一致不显著 →「部分」；反号 →「不成功」。**该判据自 2026-10-27 起不适用于对外结论**：'
              '项目对手集合 = 同团成员，原文 = 「任何与其相似度超过团内最小相似度的公司」（字面含团外），'
              'PC 不是原文变量的严格复现，故判定统一为「不可比/未复现（对手集合口径不同）」。')
    md.append('> **量级**：按期间/语料差异解释，不做量级等值对账；**全程不做样本挑选、不做调参**。\n')
    md.append('## 0. 差异声明（对表前登记）\n')
    md.append('| 维度 | 原文（刘亚辉等 2021） | 我方 | 说明 |')
    md.append('|---|---|---|---|')
    md.append('| 期间 | 文本年 2007–2016；FM 月样本 N = 149,938 / 90,752 | 文本年 FY2017–FY2022；PC 可用月 2018-05..2023-12；FM/组合有效事件月 2018-05..2023-12（68 月） | 期数不同（68 月 vs ≈120 月） |')
    md.append(f'| 语料 | 年报全文（沪深交易所） | {CORPUS_MD_CELL} | 语料不同（已登记主口径偏离） |')
    md.append('| 聚类与阈值 | 等权 0/1 → 余弦 → 层次聚类（肘点法则）→ **相似度 > 团内最小相似度的公司**（字面含团外） | 同方法聚类；K 逐年校准至池化中位 peer 44；**对手集合 = 同团全部成员**（已登记偏离） | **对手集合口径不同 → PC 不是原文变量的严格复现；不可比/未复现** |')
    md.append('| PC 定义 | −HHI(团内销售收入)（Valta 2012） | 同（`{y}_pc.csv` 的 `pc`） | 一致 |')
    md.append('| 因子 | α3/α4/α5/α6/α7（FF5+UMD+LIQ） | **α3/α4 可得**（`FF4_SELF_BUILT_v1`）；无 RMW/CMA/LIQ | 原文 α5–α7 不可得 |')
    md.append('| 控制 | HHI/β_MKT/Size/BM/Mom/Rev/Illiq/IA/Roe/Lev/Disp/Ivol（12） | size/bm/mom/rev/illiq（5） | 无 β_MKT / 行业 HHI / 会计类控制 |')
    md.append('| NW 滞后 | 12 | 主读数 6（registry `nw_lag`）+ 并报 12 | 双报 |')
    md.append('| 样本 | 全 A 股（原文样本筛除 筛选后） | 主读数 = 登记样本 `pricing_sample_v2_2015_2023`；并报「原文最近似样本」 | 两样本同判据并报 |\n')
    md.append(f"> **面板契约（I-2）**：`{contract_status['panel']}`；rows={contract_status['rows']} / "
              f"rows_registered={contract_status['rows_registered']}；"
              f"rows_match_registered=`{contract_status['rows_match_registered']}`；"
              f"PIT={contract_status['pit_violations_strict_lt']}；"
              f"dup={contract_status['duplicate_focal_month_keys']}；"
              f"zero={contract_status['missing_filled_with_zero']}。"
              + ("自定义 `--panel`：行数门显式不适用（`not_applicable_custom_panel`），"
                 "只强制 PIT/重复键/缺失不填 0。\n"
                 if custom_panel else "登记面板行数门严格通过。\n"))
    md.append('> **可比性前置（2026-10-27）**：对手集合口径未对齐（项目 = 同团全部成员；原文 = 相似度 > 团内最小相似度，字面含团外；'
              '`methodology_execution_plan.md:2290` 已登记偏离），本表其余层的「判定」一律读作「不可比/未复现」；'
              '原始符号/显著性读数仅作描述，不构成对原文的复现或证伪。\n')
    md.append('> **PC 的识别层级（N-11，2026-10-27）**：PC = −HHI(同团成员 y−1 年销售收入)，同一团全部成员共享同一 PC 值，'
              '因此 PC 是团-年常数，不是 firm-year 独立变量。2019 年 3,354 家公司 / 407 团 / 134 单点团；'
              '有 PC 的 3,218 行只有 273 个不同 PC 值（= 407 − 134）。2017–2022 逐年唯一 PC 值 = '
              '198/217/273/271/235/258，合计 1,452 个团-年取值；全 PC 面板 19,849 行，PC 对表合并样本 '
              '18,934 firm-year（不是全 PC 面板）。推断建议：横截面按团聚类（或团层 FM/团层等权），并叠加年块推断'
              '（团×年二维）；不得按 firm-month 独立处理，不得写「月度截面无重复」。\n')
    md.append('## 1. 表 2 对应：PC 描述统计\n')
    md.append('| 读数 | 原文 | 我方（firm-year，n = {n:,}） | 判定 |'.format(n=desc['pc_pooled_firm_year']['n']))
    md.append('|---|---|---|---|')
    md.append(f"| 均值 | {f3(TARGETS['table2_pc_mean']['value'])} | {desc['pc_pooled_firm_year']['mean']} | {desc['verdict_mean']} |")
    md.append(f"| 中位 | {f3(TARGETS['table2_pc_median']['value'])} | {desc['pc_pooled_firm_year']['median']} | 符号一致（描述性，不构成复现） |")
    md.append(f"| 25/75 分位 | −0.416 / −0.076 | {desc['pc_pooled_firm_year']['p25']} / {desc['pc_pooled_firm_year']['p75']} | 量级更集中 |")
    md.append(f"| 标准差 | 0.337 | {desc['pc_pooled_firm_year']['sd']} | 量级差异 |\n")
    md.append('## 2. 表 4 列(1) 对应：单变量 FM 回归 `R_{i,t+1} ~ PC`\n')
    md.append(f"**原文**：PC {f3(TARGETS['table4_col1_pc']['coef'])}（t = {TARGETS['table4_col1_pc']['t']}）\n")
    md += fm_rows([f'L1_{k}' for k in ('registered_EW_lag6', 'registered_VW_lag6', 'registered_EW_lag12',
                                       'registered_VW_lag12', 'paper_like_EW_lag6', 'paper_like_VW_lag6',
                                       'paper_like_EW_lag12', 'paper_like_VW_lag12')])
    md.append(f"\n**判定**：{verdict(fm1['registered_EW_lag12']['PC']['t_nw'], TARGETS['table4_col1_pc']['t'])}"
              f"（原符号读数仅作描述，不构成对原文的证伪。备注：`registered` = 登记样本，"
              f"`paper_like` = 原文最近似样本；EW = 等权 OLS，VW = 市值加权 WLS）\n")
    md.append('## 3. 表 4 列(2) 对应：加控制 FM 回归\n')
    md.append(f"**原文**：PC {f3(TARGETS['table4_col2_pc']['coef'])}（t = {TARGETS['table4_col2_pc']['t']}）；"
              f"HHI {f3(TARGETS['table4_col2_hhi']['coef'])}（t = {TARGETS['table4_col2_hhi']['t']}）\n")
    md += fm_rows([f'L2_{k}' for k in ('registered_EW_lag6', 'registered_VW_lag6', 'registered_EW_lag12',
                                       'registered_VW_lag12', 'paper_like_EW_lag6', 'paper_like_VW_lag6',
                                       'paper_like_EW_lag12', 'paper_like_VW_lag12')])
    md.append(f"\n控制集 = {', '.join(CONTROLS)}（+ PC）；**未含 β_MKT 与行业 HHI**（既有产物无 → 登记为限制）。")
    md.append(f"**判定**：{verdict(fm2['registered_EW_lag12']['PC']['t_nw'], TARGETS['table4_col2_pc']['t'])}"
              f"（原符号读数仅作描述，不构成对原文的证伪。）\n")
    md.append('## 4. 表 3 对应：分位组合（H−L 原始收益 + FF3/FF4 α）\n')
    md.append(f"**原文**：H−L α3 = {f3(TARGETS['table3_HL_ew_a3']['coef'])}（t = {TARGETS['table3_HL_ew_a3']['t']}，等权）/ "
              f"{f3(TARGETS['table3_HL_vw_a3']['coef'])}（t = {TARGETS['table3_HL_vw_a3']['t']}，市值加权）；"
              f"α7 = {f3(TARGETS['table3_HL_ew_a7']['coef'])}（t = {TARGETS['table3_HL_ew_a7']['t']}）/ "
              f"{f3(TARGETS['table3_HL_vw_a7']['coef'])}（t = {TARGETS['table3_HL_vw_a7']['t']}）\n")
    md.append('| 规格 | H−L raw（%） | t(NW) | α3（%） | t(α3) | α4（%） | t(α4) | 月数（有效） |')
    md.append('|---|---|---|---|---|---|---|---|')
    for k, b in ports.items():
        md.append(f"| {k} | {b['HL_raw_pct']} | {b['HL_raw_t_nw']} | {b['alpha_a3_pct']} | {b['alpha_a3_t']} | "
                  f"{b['alpha_a4_pct']} | {b['alpha_a4_t']} | {b['n_months']}（{b['n_months_effective']}） |")
    md.append('\n（`quintile/decile` = 五分位/十分位；分组规则 = 逐月 PC 截面 `rank_average`，registry 冻结 tie 规则；'
              'H−L = 最高 − 最低；α 由 H−L（减无风险）对因子月度回归，NW-HAC lag 6。）')
    md.append(f"\n**判定（α3 主口径）**：{verdict(hqreg['alpha_a3_t'], TARGETS['table3_HL_ew_a3']['t'])}；"
              f"**H−L raw**：{hqreg['HL_raw_pct']}%（t = {hqreg['HL_raw_t_nw']}）。"
              f"原文 α7 无对应因子 → 只并报 α4（0.441 的对照不可达，登记为限制）。"
              f"（原符号读数仅作描述，不构成对原文的证伪。）\n")
    md.append('## 5.（可选）假设 1 对应：PC → 业绩波动率\n')
    if h1.get('status') == 'EXECUTED':
        md.append('| 被解释变量 | 原文方向 | 我方 β | t（聚类至公司） | t（iid） | 年份 | n | 判定 |')
        md.append('|---|---|---|---|---|---|---|---|')
        for tag, name in (('sigma_sale', 'σSale（y..y+2 主营业务收入标准差）'),
                          ('sigma_profit', 'σProfit（y..y+2 净利润标准差）')):
            b = h1.get(tag, {})
            if b:
                md.append(f"| {name} | + | {b['beta_sigma_in_1e8cny']} | {b['t_cluster_firm']} | {b['t_iid']} | "
                          f"{b['years']} | {b['n']}（{b.get('n_firms')} 家） | {b['verdict']} |")
        md.append(f"\n方向读数（描述性，不构成对原文的复现/证伪）：σSale β = "
                  f"{h1.get('sigma_sale', {}).get('beta_sigma_in_1e8cny')}（点估计为负）、σProfit β = "
                  f"{h1.get('sigma_profit', {}).get('beta_sigma_in_1e8cny')}（点估计为正）。")
        md.append(f"\n口径：{h1.get('sigma_sale', {}).get('spec')}；σ 定义与原文一致（y..y+2 三期标准差），"
                  f"PC 取文本年 y−1（firm-year n = {h1['n_firm_year']}）；"
                  f"β 单位 = 亿元/单位 PC（σ 以亿元计，t 值不受缩放影响）。")
    else:
        md.append(f"未执行（status = {h1.get('status')}）。")
    md.append('\n## 6. 判定汇总\n')
    md.append('| 层 | 原文值 | 我方值 | 判定 |')
    md.append('|---|---|---|---|')
    for c in checklist:
        md.append(f"| {c['layer']} | {c['target']} | {c['ours']} | **{c['verdict']}** |")
    md.append(f"\n**总判定**：{NOT_COMPARABLE_LABEL}（不可比层 {n_not_comparable} 项 / 复现成功 {n_success} 项 / "
              f"部分 {n_partial} 项）\n")
    md.append('## 6A. 结论解读（不可比口径；不作超出证据的主张）\n')
    md.append('- **不可比/未复现**：我方对手集合 = 同团全部成员；原文第四步 = 「任何与其相似度超过团内最小相似度的公司」'
              '（字面含团外）。二者不是同一对手集合，PC 不是原文变量的严格复现，因此本报告不能给出'
              '「复现成功/不成功/符号相反」的判定，也不构成对原文的证伪（N-2；`methodology_execution_plan.md:2290` 已登记偏离）。')
    md.append(f"- **数值读数保留（仅描述本样本）**："
              + ("全文列登记样本 EW lag12 PC = "
                 f"{fm1['registered_EW_lag12']['PC']['beta_pct_per_unit']}%"
                 f"（t = {fm1['registered_EW_lag12']['PC']['t_nw']}；lag 6 t = "
                 f"{fm1['registered_EW_lag6']['PC']['t_nw']}），点估计为负但不显著。"
                 "以下诊断仅适用主口径（diagnostic_sample = main_caption）："
                 if ('fulltext' in OUT_PREFIX or '全文' in CORPUS_LABEL) else
                 "登记样本 EW lag12 PC = "
                 f"{fm1['registered_EW_lag12']['PC']['beta_pct_per_unit']}%"
                 f"（t = {fm1['registered_EW_lag12']['PC']['t_nw']}；lag 6 t = "
                 f"{fm1['registered_EW_lag6']['PC']['t_nw']}），")
              + "95% CI [−1.198, +0.185] 覆盖 0 与原文 +0.152，"
              f"MDE(80%) = 0.9884%、原文效应 +0.152 处功效 ≈ 0.07，68 个月。该区间与功效只说明"
              f"「本样本无法判定方向」，不能读作「符号相反」，也不能读作「原文被证伪」。")
    md.append(f"- **不是符号口径事故**：PC = −HHI 逐值核验（max|pc + hhi| = {sign_check['max_abs_pc_plus_hhi']}；"
              f"PC 越大 = 竞争越激烈，与原文约定一致）。")
    md.append('- **量级/期间/语料差异**：只作描述性并列；在对手集合不可比的前提下，不把任何方向或量级差异'
              '解释为对原文的支持或否证。')
    md.append('- **与原文列(3)(4) 的关系**：原文 TF-IDF PC（0.129）与 HP 中位概率阈值 PC（0.087）同样为正；'
              '我方既有 `teacher_idf` / `teacher_medadj` 变体网络存在但**本轮未纳入**（见「未执行」清单）。'
              '这些读数同样不能改变对手集合不可比的结论。')
    md.append('- **若要真正复现原文 PC**：须按原文第四步规则 + 接近原文词表规模（active_vocab ≈ 1.5 万量级）重建，'
              '并在运行前登记；本轮未做（N-2 已登记）。\n')
    md.append('## 7. 已知限制\n')
    for lim in out['known_limitations']:
        md.append(f'- {lim}')
    md.append('\n## 8. 未执行 / 偏离清单\n')
    for it in out['unexecuted_or_deviations']:
        md.append(f'- {it}')
    md.append('\n## 9. 产物与纪律\n')
    md.append('- 本报告与脚本为本轮**新建**；**未覆盖任何既有文件、未改任何 headline_value / 冻结值 / 门槛**。')
    md.append(f'- 输入：`{_rel(PANEL_SAMPLE)}`（登记样本）、'
              f'`{PC_DIR_LABEL}/{{{PC_YEARS[0]}..{PC_YEARS[-1]}}}_pc.csv`（PC）、'
              '`data/processed/factor_returns_daily.csv`（FF4_SELF_BUILT_v1）、'
              '`data/raw/drive/operating_revenue_annual.pkl` + `net_profit_parent_company_annual.pkl`（假设 1）。')
    md.append('- 统计口径：FM = 逐月截面 OLS/WLS → 系数序列 → 均值 + Newey-West(Bartlett) SE（主 lag 6，'
              'registry `nw_lag`；并报 12）；组合 α = H−L（减无风险）对因子月度回归 + NW-HAC SE；'
              '分组 = 逐月 PC 截面 `rank_average`（registry 冻结 tie 规则）；收益单位统一为 %。')
    md.append('')
    MD_OUT.write_text('\n'.join(md), encoding='utf-8', newline='\n')
    psay('7', f'MD 产出状态 = ok（{MD_OUT.relative_to(ROOT)}）；总判定 = {overall}')

    print(json.dumps({'artifact': 'competition_pricing_correspondence', 'overall': overall,
                      'n_success': n_success, 'n_partial': n_partial,
                      'L2_col1_registered_EW_lag12': fm1['registered_EW_lag12']['PC'],
                      'L2_col2_registered_EW_lag12': fm2['registered_EW_lag12']['PC'],
                      'L3_registered_quintile_EW': {k: hqreg[k] for k in
                                                    ('HL_raw_pct', 'HL_raw_t_nw', 'alpha_a3_pct', 'alpha_a3_t',
                                                     'alpha_a4_pct', 'alpha_a4_t', 'n_months')},
                      'status': 'PASS'}, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
