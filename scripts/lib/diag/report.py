"""s3_pc_sign_diag_h1h8 的 report 模块（逐字搬移，逻辑未改）。"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data_audit'))
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'data'))

from lib.diag.consts import DIAG
from lib.diag.consts import OUT_JSON
from lib.diag.consts import OUT_MD
from lib.diag.consts import ROOT
from lib.diag.util import psay
import json

def assemble():
    parts = {}
    for tag in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'):
        p = DIAG / f'{tag}.json'
        parts[tag] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'status': 'MISSING'}
    S = {
        'artifact': 's3_pc_sign_diagnostics_20261021',
        'generated_at': '2026-10-21',
        'nature': '项目自建 PC 系数为负的只读诊断（复用既有产物 + 新建诊断脚本/产物；未改任何既有文件/冻结值/主口径）。'
                  '**2026-10-27 口径更正**：仅描述项目自建 PC（对手集合 = 同团全部成员），不是对原文 PC 的复现或证伪。',
        'comparability': {
            'comparable_to_paper': False,
            'registered_deviation': 'docs/methodology/methodology_execution_plan.md:2290',
            'project_rule': 'same_cluster_members（同团全部成员）',
            'paper_rule': '任何与其相似度超过团内最小相似度的公司（字面含团外）',
            'consequence': 'PC 不是原文变量的严格复现；判定 = 不可比/未复现（对手集合口径不同）；'
                           '不得写符号相反/复现失败/证伪',
            'replication_requirement': '按原文第四步规则 + 接近原文词表规模（active_vocab ≈ 1.5 万量级）重建（未做）'},
        'pre_registered_criteria': {
            'rule': '成立 = 主读数层（登记样本 EW lag6 单变量）PC 系数 ≥ 0 且 4 子层 ≥3 层同向转正；'
                    '部分 = 向 0 移动 ≥50% 仍为负；排除 = 移动 <25% 或同号；待定 = 数据不可得/证据冲突',
            'h1_special': '注入检验：合成「PC 与下月收益正相关」面板 → 期望显著为正；为负则管线方向 bug 确诊',
            'h8_special': 'PC 对团规模回归残差的 FM 系数 ≥0 → 机械/结构解释成立',
            'no_sample_selection_no_tuning': True,
        },
        'inputs': {
            'panel': 'data/processed/m1_common_sample_ma_teacher_v2.parquet',
            'pc_dir': 'data/processed/network_ma_teacher/{2017..2022}_pc.csv',
            'k_tier_cache': 'artifacts/tmp/pc_rc/h1_net_t{5,10,20,30,44,60,80,120}/（既有只读复用）',
            'returns': 'data/processed/monthly_returns_clean_v2.csv（持有期口径）',
            'factors': 'data/processed/factor_returns_daily.csv（FF4 月聚合）',
            'sw2': 'data/processed/sw2_industry_latest.csv',
        },
        'baseline_readings': {
            'paper_targets': {'table4_col1': '+0.152 (t=7.76)', 'table4_col2': '+0.130 (t=5.40)',
                              'table2_pc_mean_median': '-0.230 / -0.180', 'table3_HL_a3_ew': '+0.218 (t=6.27)'},
            'ours_MDA_teacher': {'registered_EW_lag6_univariate': '-0.5068 (t=-1.4365)',
                                 'registered_EW_lag6_with_controls': '-0.4116 (t=-1.1902)',
                                 'paper_like_EW_lag12_univariate': '-0.4559 (t=-1.9998)',
                                 'quintile_EW_alpha_a3': '-0.3482 (t=-1.4744)'},
            'ours_fulltext_corpus': {'registered_EW_lag6_univariate': '-0.3926 (t=-1.1227)',
                                     'paper_like_EW_lag12_univariate': '-0.4194 (t=-2.0059)',
                                     'quintile_EW_alpha_a3': '-0.4974 (t=-2.0349)',
                                     'src': 'reports/data_gates/G0/s3_pc_pricing_correspondence_fulltext.json'},
            'note': '换语料不改变项目 PC 的点估计符号（历史读数，描述性）；本报告数值为 MD&A 主口径（除注明外）。'
                    'PC 与原文对手集合不可比，判定统一为「不可比/未复现」。',
        },
        'parts': parts,
    }
    v = {}
    h1 = parts.get('H1', {})
    inj = h1.get('injection_test', {})
    pos = inj.get('syn_rank_pos', {})
    h1_ok = (pos.get('beta_pct') is not None and pos.get('beta_pct') > 0 and pos.get('HL_decile_EW_pct', -1) > 0)
    ga = h1.get('group_audit_decile', {})
    v['H1'] = {
        'verdict': '排除' if (h1_ok and ga.get('H_group_is_high_pc') and ga.get('H_group_is_low_hhi')) else '成立（方向 bug）',
        'reason': (f"注入检验：合成正相关 PC 的 β = {pos.get('beta_pct')}%（t = {pos.get('t_nw')}）、"
                   f"H−L(十分位 EW) = {pos.get('HL_decile_EW_pct')}%（t = {pos.get('HL_decile_EW_t')}）；"
                   f"镜像负相关 β = {inj.get('syn_rank_neg', {}).get('beta_pct')}%；随机 β = {inj.get('syn_random', {}).get('beta_pct')}%；"
                   f"D10−D1 PC 均值差 = {ga.get('D10_minus_D1_pc')}、HHI 差 = {ga.get('D10_minus_D1_hhi')}"
                   f"（H 组 PC {'更大' if ga.get('H_group_is_high_pc') else '更小'}）"),
    }
    h2 = parts.get('H2', {})
    ex = h2.get('excl_self_and_coverage_variants', {})
    r_ex = ex.get('registered', {}).get('univariate', {}).get('beta_pct')
    wpa = h2.get('write_precision_audit', {})
    v['H2'] = {
        'verdict': '排除' if (r_ex is not None and r_ex < 0) else '部分/成立',
        'reason': (f"剔除自身变体 β = {r_ex}%（t = {ex.get('registered',{}).get('univariate',{}).get('t_nw')}）；"
                   f"覆盖率=1 子样本 β = {ex.get('coverage_eq_1',{}).get('registered',{}).get('beta_pct')}%；"
                   f"手核团逐位对拍全精度 max|Δhhi| ≤ 5e-9（写盘 round(x,8) 精度；最大团实测 "
                   f"{wpa.get('csv_hhi_max_abs_diff_raw') if wpa.get('csv_hhi_max_abs_diff_raw') is not None else '—'}"
                   f"，原报「= 0」为 r4 显示假象，REV2-F4 修正）；"
                   f"营收字段降级为「盘上 pkl = 营业收入（854.30 亿），原文附录表未核」"),
    }
    h3 = parts.get('H3', {})
    tg = h3.get('tier_grid', {})
    k_vals = [x['k_2019'] for x in tg.values() if x.get('k_2019')]
    t_abs = [abs(x['fm_reg_registered_lag12']['t_nw']) for x in tg.values()
             if x['fm_reg_registered_lag12'].get('t_nw') is not None]
    v['H3'] = {'verdict': '排除' if h3.get('all_negative') else '部分/成立',
               'reason': f"{len(tg)} 档 K（中位 peer 5→300，K(2019) {min(k_vals)}–{max(k_vals)}，含请求的粗 K≈100–200 区）"
                         f"主读数层**无正号** = {h3.get('all_negative')}（β 区间 "
                         f"{min([x['fm_reg_registered_lag12']['beta_pct'] for x in tg.values()], default=None)}"
                         f"..{max([x['fm_reg_registered_lag12']['beta_pct'] for x in tg.values()], default=None)}；"
                         f"多档 |t| ≤ 0.55、全部不显著，最大 |t| = {max(t_abs) if t_abs else None:.4f}）；"
                         f"粗档 tier=150/200 的加控制与 α3 子层转正但不显著（|t| ≤ 0.95）；"
                         f"tier=10 与原文**中心四分位吻合**（p25 −0.3962 vs −0.416、中位 −0.1799 vs −0.180、"
                         f"p75 −0.0733 vs −0.076）但 **sd 0.2204 vs 原文 0.337**（尾部不一致，REV2-F4 限定表述）"
                         f"仍为 −0.3348%"}
    h4 = parts.get('H4', {})
    va = h4.get('variants', {})
    pos4 = {k: x.get('fm_univariate', {}).get('beta_pct') for k, x in va.items()
            if isinstance(x, dict) and x.get('fm_univariate', {}).get('beta_pct') is not None
            and x['fm_univariate']['beta_pct'] > 0}
    v['H4'] = {'verdict': '排除' if not pos4 else '部分/成立',
               'reason': f"7 个样本变体（含无分析师门、含微盘、全 PC 宇宙）系数全部为负；正号变体 = {pos4 or '无'}"}
    h5 = parts.get('H5', {})
    py = h5.get('positive_years_registered', [])
    psig5 = h5.get('positive_and_significant_years', [])
    h5_year_txt = {y: x.get('beta_pct') for y, x in h5.get('by_return_year', {}).get('registered', {}).items()}
    h5_paper_txt = {y: x.get('beta_pct') for y, x in h5.get('by_return_year', {}).get('paper_like', {}).items()}
    h5_reg19 = h5.get('by_return_year', {}).get('registered', {}).get('2019', {})
    h5_pap19 = h5.get('by_return_year', {}).get('paper_like', {}).get('2019', {})
    v['H5'] = {'verdict': '待定（无正面证据；2007–2016 窗口不可测）',
               'reason': f"登记样本逐年 β = {h5_year_txt}（原文最近似样本 = {h5_paper_txt}）；"
                         f"正号年 = {py or '无'}，但**bootstrap CI95 含 0 的正号年 = "
                         f"{psig5 or '无'}**（判据 REV2-F2 = 月块 bootstrap CI，非 NW t）；"
                         f"2019：登记 {h5_reg19.get('beta_pct')}%/CI95 {h5_reg19.get('boot_ci95_pct')}、"
                         f"原文最近似样本 {h5_pap19.get('beta_pct')}%"
                         f"（t_nw = {h5_pap19.get('t_nw')} 但 t_iid = {h5_pap19.get('t_iid')}、"
                         f"CI95 {h5_pap19.get('boot_ci95_pct')} 含 0、LOO β 区间 {h5_pap19.get('loo_beta_range_pct')}）"
                         f"→ 不得写作「显著为正」的正面证据；逐月负号占比 = "
                         f"{h5.get('monthly_beta_sign_distribution', {}).get('share_negative')}"}
    h6 = parts.get('H6', {})
    cco = h6.get('control_coefs_vs_paper', {}).get('registered', {}).get('coefs', {})
    cs6 = h6.get('cluster_size_control', {})
    psa = h6.get('paper_sign_agreement', {})
    _ncp = parts.get('H8', {}).get('revision_tests', {}).get('control_function_form', {}).get('variants', {})
    v['H6'] = {'verdict': '部分（归因收窄：团规模控制仅 log 形式转正且不显著；1/N 仍负；控制项 3/5 与原文反号）',
               'reason': f"加控制后我方 size {cco.get('size')}、bm {cco.get('bm')}、rev {cco.get('rev')}"
                         f"（原文 +0.339 / −0.358 / +1.835）→ 符号相反仅 "
                         f"{psa.get('n_opposite_sign')}/{psa.get('n_controls')} 项（{psa.get('opposite_vars')}），"
                         f"mom/illiq 同号；"
                         f"行业 FE 后 β = {h6.get('industry_fe', {}).get('univariate', {}).get('beta_pct')}%"
                         f"（t = {h6.get('industry_fe', {}).get('univariate', {}).get('t_nw')}）；"
                         f"团规模控制：log 形式 β_PC = {cs6.get('PC_plus_log_peers', {}).get('PC_beta_pct')}%"
                         f"（t = {cs6.get('PC_plus_log_peers', {}).get('PC_t')}，不显著）、"
                         f"**1/N 形式 β_PC = {_ncp.get('PC+invN', {}).get('PC', {}).get('beta_pct')}%"
                         f"（t = {_ncp.get('PC+invN', {}).get('PC', {}).get('t_nw')}，团成员数口径，H8 §8R）仍为负**"
                         f"、1/n_competitors 口径 {cs6.get('PC_plus_inv_peers', {}).get('PC_beta_pct')}%"
                         f"（t = {cs6.get('PC_plus_inv_peers', {}).get('PC_t')}）同样为负 → 归因仅限 log 形式"} 
    h7 = parts.get('H7', {})
    yv = h7.get('y_variants', {})
    h7_hor_txt = {k: x.get('beta_pct_per_unit_period') for k, x in h7.get('holding_horizons', {}).items()
                  if k.startswith('h_')}
    h7_hor_monthly = {k: x.get('beta_pct_per_unit_per_month_avg') for k, x in h7.get('holding_horizons', {}).items()
                      if k.startswith('h_')}
    v['H7'] = {'verdict': '排除' if not h7.get('positive_specs') else '待定（部分口径转正）',
               'reason': f"收益口径/持有期变体正号项 = {h7.get('positive_specs') or '无（全部为负）'}；"
                         f"持有期 β（%/单位/期）= {h7_hor_txt}；**折算月均 = {h7_hor_monthly}** → "
                         f"无反转、月均幅度基本不变（t+2..6 ≈ −0.66%/月、t+7..12 ≈ −0.79%/月 vs t+1 −0.57%/月）；"
                         f"各窗口为**重叠窗**且 NW(6) 未做重叠校正 → t 不作推断依据（REV2-F5）"}
    h8 = parts.get('H8', {})
    rt = h8.get('residual_test', {})
    resid_beta = rt.get('fm_PC_resid', {}).get('beta_pct')
    rv8 = h8.get('revision_tests', {})
    idn = rv8.get('identity_decomposition', {})
    ncp = rv8.get('control_function_form', {}).get('variants', {})
    wnb = rv8.get('within_N_bins', {}).get('bins', {})
    drr = rv8.get('distribution_reweighting', {})
    pmp = rv8.get('cluster_label_permutation', {})
    pmd = rv8.get('power_mde', {}).get('main_reading_registered_univariate', {})
    v['H8'] = {'verdict': '待定 / 未被识别（原「根因 #1 = PC 是团结构代理、残差化后负号消失」结论**撤回**）',
               'reason': f"撤回理由 = 对处理变量 PC 的定义性结构分量（团规模/自身份额）回归取残差属过度控制/bad control；"
                         f"且函数形式不稳健（log1p(N) → {cs6.get('PC_plus_log_peers', {}).get('PC_beta_pct')}% 转正，"
                         f"1/N → {ncp.get('PC+invN', {}).get('PC', {}).get('beta_pct')}%、"
                         f"PC+1/N+own_share → {ncp.get('PC+invN+own_share', {}).get('PC', {}).get('beta_pct')}% 仍为负）；"
                         f"恒等式两分量各自为负（FM(−own²) = {idn.get('t_own', {}).get('beta_pct')}%/"
                         f"t = {idn.get('t_own', {}).get('t_nw')}、FM(−他人集中度) = {idn.get('t_peer', {}).get('beta_pct')}%/"
                         f"t = {idn.get('t_peer', {}).get('t_nw')}）；within-N 分箱 β = "
                         f"{ {k: v2.get('beta_pct') for k, v2 in wnb.items()} }；"
                         f"分布重加权：WLS β = {drr.get('registered|w_dist', {}).get('beta_pct')}%"
                         f"（t = {drr.get('registered|w_dist', {}).get('t_nw')}）、分位匹配 β = "
                         f"{drr.get('registered|quantile_match', {}).get('beta_pct')}%"
                         f"（t = {drr.get('registered|quantile_match', {}).get('t_nw')}）；"
                         f"团标签置换 1000 次：单侧 p = {pmp.get('p_one_sided_le_observed')}；"
                         f"功效/MDE：se = {pmd.get('se_pct')}%、MDE(80%) = {pmd.get('mde80_pct')}%、"
                         f"原文效应处功效 = {pmd.get('power_at_effect_ref')}"}
    S['verdicts'] = v
    # ---- REV2 修订元信息（依独立复核 F1–F6）
    S['revision'] = {
        'revision_id': 'REV2-20261021',
        'date': '2026-10-21',
        'trigger': '独立复核（artifacts/tmp/review_diag/{rv_lib.py,A_injection.py,B_h2.py,C_h3h4.py,D_h5.py,'
                   'E_h6.py,F_h8.py,F2_h8_identity.py} 及同名 JSON）判定原报告三处方法论/表述问题（F1–F6）',
        'changes': [
            'F1｜H8 撤回「PC ≈ 团结构代理、残差化后负号消失」（过度控制/bad control；函数形式不稳健；恒等式两分量各自为负）'
            '→ 改判待定/未被识别；补做 6 项替代检验（恒等式分解 / 1/N 控制 / within-N 分箱 / 分布匹配重加权 / 1000 次团标签置换 / 功效-MDE），'
            '全部由本脚本独立重算（不引用复核脚本产物数值）',
            'F2｜H5 降级为「待定（无正面证据；2007–2016 窗口不可测）」；年度/子期间并报 iid t + 月块 bootstrap CI95 + LOO；判据改以 bootstrap CI 为准',
            'F3｜H6 归因收窄：「团规模控制使 PC 转正」仅限 log 形式且不显著；补 1/N 线性控制；控制项符号 3/5 与原文相反（size/bm/rev），mom/illiq 同号',
            'F4｜精度/强度表述：H2 max|Δhhi| ≤5e-9（round(x,8) 写盘精度）、营收字段降级；H3「11/11 为负」→「无正号（多档 |t| ≤ 0.55）」、'
            '分布「高度吻合」→「中心四分位吻合（sd 0.2204 vs 0.337）」',
            'F5｜H7 持有期 β 折算月均 + 注明重叠窗/NW(6) 未做重叠校正',
            'F6｜治理登记（manifest / registry / spec 三角刷新 + 四审计重跑 + 三哈希闭环）',
        ],
        'principle': '只改本诊断轮自己的产物；未改主口径 / 冻结值 / headline_value；新增检验判据（docstring）先登记后跑。',
    }
    # ---- 根因排序 / 已排除 / 未排除 / 下一步
    h8c = h8.get('pc_structure_corr', {})
    h5_yearly = {y: x.get('beta_pct') for y, x in h5.get('by_return_year', {}).get('registered', {}).items()}
    h5_sub = '；'.join(f"{k.split('|')[-1]} = {x.get('beta_pct')}%（t = {x.get('t_nw')}；"
                       f"bootstrap CI95 {x.get('boot_ci95_pct')}）"
                       for k, x in h5.get('subperiods', {}).items() if k.startswith('registered'))
    h7_hor = {k: x.get('beta_pct_per_unit_per_month_avg') for k, x in h7.get('holding_horizons', {}).items()
              if k.startswith('h_')}
    S['summary_ranking'] = [
        {'candidate': '**（撤回）PC 的横截面信息 = 团结构代理（自身收入份额 + 团规模）**（H8 结构层）',
         'support': '**撤回**（原「强（唯一使负号消失的口径）」不成立）',
         'evidence': f"过度控制/bad control（对处理变量的定义性分量取残差）；函数形式不稳健（log1p(N) → "
                     f"{cs6.get('PC_plus_log_peers', {}).get('PC_beta_pct')}% 对照 1/N → "
                     f"{ncp.get('PC+invN', {}).get('PC', {}).get('beta_pct')}% 仍为负）；恒等式两分量各自为负"
                     f"（FM(−own²) = {idn.get('t_own', {}).get('beta_pct')}%、FM(−他人集中度) = "
                     f"{idn.get('t_peer', {}).get('beta_pct')}%）→ 改判「待定/未被识别」"},
        {'candidate': '**期间 / 市场状态（2018–2023 vs 原文 2007–2016）**（H5）',
         'support': '待定（无正面证据；2007–2016 窗口不可测）',
         'evidence': f"登记样本逐年 β = {h5_yearly}；原文最近似样本逐年 β = {h5_paper_txt}"
                     f"（2019 = +0.3391%/t_nw = 2.92 但 **t_iid = 1.06、月块 bootstrap CI95 [−0.2645, +0.9235] 含 0、"
                     f"LOO β 区间 [0.1990, 0.4915]、去 2019-09 后 t = 1.47** → 非显著为正，不得作为正面证据；"
                     f"2023 = −1.3857%/t = −4.7958 但 bootstrap CI95 [−2.6438, −0.2771]）；子期间 {h5_sub}；"
                     f"逐月负号占比 {h5.get('monthly_beta_sign_distribution', {}).get('share_negative')}；"
                     f"控制变量符号相反仅 3/5（size/bm/rev）→ 该窗口横截面定价结构不同；"
                     f"持有期折算月均不改善（{h7_hor}）→ 排除「信号延迟」型期间解释"},
        {'candidate': '残余口径（原文 12 控制集 / S1–S7 样本筛选 / 原文肘点团结构不可得 / 分布匹配重加权）',
         'support': '弱–中（未穷尽；分布口径已用重加权检验，未使主读数转正）',
         'evidence': f"列(1) 单变量已为负 → 控制集翻转概率低；K 网格 11 档主读数层无正号；"
                     f"within-N 分箱内无一致正号（{ {k: v2.get('beta_pct') for k, v2 in wnb.items()} }）；"
                     f"分布匹配重加权（WLS β = {drr.get('registered|w_dist', {}).get('beta_pct')}%、"
                     f"分位匹配 β = {drr.get('registered|quantile_match', {}).get('beta_pct')}%）亦未转正且不显著；"
                     f"原文 S1–S7 与 βMKT/IA/Roe/Lev/Disp/Ivol 未逐条复刻"},
    ]
    S['retracted_conclusions'] = [
        '**撤回**：H8「PC 的横截面信息 = 团结构代理……残差化后负号消失」→ 复核三条反驳（原文收录）：'
        + ' ／ '.join(rv8.get('withdrawal', {}).get('review_rebuttals_verbatim', [])),
        '**收回**：H5 根因排序 #2 中「2019 显著为正」的表述（NW t = 2.92 由月系数强自相关放大 2.63×：'
        't_iid = 1.06；月块 bootstrap CI95 含 0；去 2019-09 后 t = 1.47）→ 改为「无正面证据」',
        '**修正**：H3「11/11 为负」→「无正号（多档 |t| ≤ 0.55，均不显著）」；'
        'H2「max|Δhhi| = 0」→「≤ 5e-9（round(x,8) 写盘精度）」；'
        'H2 营收字段「与原文一致」→「盘上 pkl = 营业收入，原文附录表未核」；'
        'H3 分布「高度吻合」→「中心四分位吻合、sd 0.2204 vs 原文 0.337」',
    ]
    S['revised_root_cause'] = {
        'excluded_strong': [
            f"H1 方向/排序 bug — 注入检验 + PC×(−1) 反证：合成正相关 PC β = "
            f"{inj.get('syn_rank_pos', {}).get('beta_pct')}%（t = {inj.get('syn_rank_pos', {}).get('t_nw')}）、"
            f"镜像 β = {inj.get('syn_rank_neg', {}).get('beta_pct')}%、随机 β = {inj.get('syn_random', {}).get('beta_pct')}%；"
            f"PC×(−1) 反证 β = {inj.get('adv_neg_PC', {}).get('beta_pct')}%"
            f"（t = {inj.get('adv_neg_PC', {}).get('t_nw')}，逐位反号；对照复核实测 +0.5068/+1.4365）；"
            f"D10−D1 PC {ga.get('D10_minus_D1_pc')} / HHI {ga.get('D10_minus_D1_hhi')}",
            'H2 PC 定义细节 — 手核团全精度 ≤5e-9（274 成员团）；剔除自身 −0.3516（t = −1.56）；'
            '营收覆盖=1 子样本 −0.5838；hhi_excl_self 列语义已登记',
            'H3 K 口径 — 11 档（K(2019) 116–1042）主读数层无正号、全部不显著（多档 |t| ≤ 0.55，最大 1.80）；'
            'tier=10 中心四分位对齐原文后仍 −0.3348%',
            'H4 样本口径 — 7 变体全为负（全 PC 宇宙 n = 645,782 → −0.3796；无分析师门 −0.189；纳微盘 −0.783）',
            'H7 收益口径与横截面 — 14 个收益规格全负（EW/VW、raw/α、winsor 变体、行业/市场调整、后复权）；'
            '规模三分位全负（大盘最负 −0.839）；持有期折算月均无反转',
            '（跨轮一致）T1–T6 批与 rootcause H1–H10 已排除：尾部极值、小市值风格、NW 实现、语料、词库粒度、'
            '行业马甲、后复权、PIT 对齐',
        ],
        'pending': [
            '期间/制度（H5）：无正面证据；原文 2007–2016 窗口不可测（语料仅文本年 2017+）',
            '残余口径：原文 12 控制集（βMKT/IA/Roe/Lev/Disp/Ivol + 证监会二级行业 HHI）、S1–S7 样本筛选、'
            '原文肘点团结构不可得',
            'H8 结构层（原「团结构代理」）：改正为**待定/未被识别**（过度控制方法论不成立；'
            f"分布匹配重加权（重要性 WLS β = {drr.get('registered|w_dist', {}).get('beta_pct')}%/t = "
            f"{drr.get('registered|w_dist', {}).get('t_nw')}；分位匹配 β = "
            f"{drr.get('registered|quantile_match', {}).get('beta_pct')}%/t = "
            f"{drr.get('registered|quantile_match', {}).get('t_nw')}）与 within-N 分箱均未提供支持",
            '推断层强度：团标签置换 1000 次零分布下观测 β 单侧 p = '
            f"{pmp.get('p_one_sided_le_observed')}（恰在 5% 临界、分辨率 0.1%）、双侧 p = {pmp.get('p_two_sided')}"
            f" → 按双侧口径「反号」不显著区别于零分布（单侧口径处于临界）、置换 MDE(80%) ≈ "
            f"{pmp.get('mde80_from_perm_sd_pct')}%；"
            f"功效/MDE：se = {pmd.get('se_pct')}%、CI95 = {pmd.get('ci95_pct')}%、MDE(80%) = {pmd.get('mde80_pct')}%、"
            f"原文效应 +0.152 处功效 = {pmd.get('power_at_effect_ref')} → **本设计对原文量级功效不足**",
        ],
        'retracted': [
            'H8「PC ≈ 团结构代理、残差化后负号消失」（过度控制/bad control；函数形式不稳健；恒等式两分量各自为负）',
            'H5 排序 #2 的「2019 显著为正」表述（NW t 被自相关放大；bootstrap CI 含 0）',
        ],
        'final_conclusion': (
            f"两套语料、四类样本、11 档 K、14 种收益口径下**均未检出中文论文的正号**；"
            f"主读数不显著（β = {pmd.get('beta_pct')}%/单位、t = {pmd.get('t_nw')}；"
            f"MDE(80%) ≈ {pmd.get('mde80_pct')}%/单位、原文效应 +0.152 处功效 ≈ "
            f"{pmd.get('power_at_effect_ref')}；CI95 {pmd.get('ci95_pct')} 同时覆盖 0 与 +0.152 → 不能读作「已证伪原文效应」；"
            f"团标签置换 1000 次单侧 p = {pmp.get('p_one_sided_le_observed')}、双侧 p = {pmp.get('p_two_sided')}），"
            f"故应表述为「**未检出正号**」而非「稳健反号」；H8「团结构代理」根因撤回、期间层无正面证据 → 未识别根因。"
            f"但 PC 与原文对手集合不可比（项目 = 同团成员；原文 = 相似度 > 团内最小相似度，字面含团外），"
            f"故该结论只描述项目自建 PC，不构成对原文的复现或证伪。"),
    }
    S['excluded'] = [
        f"H1 排序/高低方向 bug（注入检验：合成正相关 PC β = {inj.get('syn_rank_pos', {}).get('beta_pct')}%"
        f"（t = {inj.get('syn_rank_pos', {}).get('t_nw')}）、镜像 β = {inj.get('syn_rank_neg', {}).get('beta_pct')}%、"
        f"随机 β = {inj.get('syn_random', {}).get('beta_pct')}%；PC×(−1) 反证 = +0.5068（t = +1.4365）；"
        f"D10−D1 PC {ga.get('D10_minus_D1_pc')} / HHI {ga.get('D10_minus_D1_hhi')}）",
        'H2 PC 定义/算式细节（手核团全精度 max|Δhhi| ≤ 5e-9 = 写盘 round(x,8) 精度（含 274 成员团）；'
        '剔除自身 −0.3516（t=−1.56）；营收覆盖=1 子样本 −0.5838；营收字段 = 盘上 pkl 营业收入 854.30 亿、'
        '**原文附录表未核**（REV2-F4 降级））',
        'H3 K 校准口径（11 档 K：中位 peer 5→300、K(2019) 116–1042，含请求的粗 K≈100–200 区；'
        '**主读数层无正号（多档 |t| ≤ 0.55，全部不显著，最大 |t| = 1.80）**；'
        'tier=10 与原文**中心四分位吻合**（p25 −0.3962 vs −0.416、中位 −0.1799 vs −0.180、p75 −0.0733 vs −0.076）'
        '但 sd 0.2204 vs 0.337（尾部不一致）仍为 −0.3348%）',
        'H4 样本口径（7 变体全为负：全 PC 宇宙 n=645,782 β=−0.3796；无分析师门 −0.189；纳微盘 −0.783）',
        'H7 收益口径/横截面结构/持有期（14 个收益规格全负；规模三分位全负且大盘最负 −0.839；'
        '持有期 β 折算月均无反转、幅度基本不变（≈−0.57 / −0.66 / −0.79 %/月）；α3 仍为负）',
        '（跨轮一致）T1–T6 批与 rootcause H1–H10 已排除：尾部极值、小市值风格、NW 实现、语料、词库粒度、'
        '行业马甲、后复权、PIT 对齐',
    ]
    S['open_items'] = [
        '原文 2007–2016 窗口**不可测**：我方 MD&A 与年报全文语料均仅覆盖文本年 2017+（营收面板 FY2016+）'
        '→ 无法在同窗口对照；期间的判别只能靠**向前延展**（FY2013–2015 营收重建 → 文本年 2014–2022）。',
        '原文肘点法的确切团结构（K 与团形）无法从表 2 唯一反推 → K 网格覆盖了"分布对齐"维度，'
        '但无法证明"原文团结构下符号会变"。',
        '原文 12 项控制集（βMKT 36 月滚动 / IA / Roe / Lev / Disp / Ivol / 证监会二级行业 HHI）与 '
        'S1–S7 样本筛选（金融/综合类剔除、上市不足两年、词条数 <10 等）未逐条复刻。',
        '**H8 结构层 = 待定/未被识别（原「团结构代理」结论已撤回）**：残差化路线属过度控制/bad control，'
        '不得作为根因；round-2 替代检验（恒等式分解 / 1/N 控制 / within-N 分箱 / 分布匹配重加权 / 1000 次团标签置换）'
        '均未提供正号或机制证据。',
        '**统计功效不足**：主读数 se ≈ '
        f"{pmd.get('se_pct')}%、MDE(80%) ≈ {pmd.get('mde80_pct')}%/单位，原文效应 +0.152 处功效 ≈ "
        f"{pmd.get('power_at_effect_ref')} → 「未检出正号」不能读作「已证伪原文效应」；"
        '要检验原文量级需扩展窗口/提高截面 N（或与原文同窗口）。',
    ]
    S['next_steps'] = [
        '**决定性实验（期间层）**：MRQ→FY2013–2015 营收重建（前置门：FY2016+ 与年报相对误差 ≤1%）→ '
        'PC 文本年扩至 2014–2022（事件月 ≈104）→ 检验 2015–2017 是否出现正号（以月块 bootstrap CI 为判据）；'
        '若仍无正号，则期间层也被压缩为"窗口内不可定案"。',
        '**规格补全（残余口径）**：构造 βMKT(36 月滚动)/IA/Roe/Lev/Disp/Ivol + 证监会二级行业 HHI，'
        '复刻表 4 列(2) 的 12 控制集；同步按 S1–S7 逐条复刻样本（预计不翻转，列(1) 已为负）。',
        '**功效补强（推断层）**：本轮 1000 次团标签置换 + MDE 已完成（单侧 p = '
        f"{pmp.get('p_one_sided_le_observed')}、MDE(80%) ≈ {pmd.get('mde80_pct')}%）→ 下一步是压 MDE："
        '延长文本年（同期间层实验）、提高月内有效截面、或改用重叠窗正确校正（HAC 以事件月聚合）。',
        '**结构层路线更正**：撤回「对 log 团员数/自身收入份额正交化」的最小修复建议（bad control）；'
        '若仍需机制解释，须用**团层随机化/工具**或明确外生结构变量，不得对 PC 的定义性分量做控制。',
        '**表述层兜底**：论文按「项目 PC 与原文不可比（对手集合口径不同）+ 同窗口点估计为负 + 功效不足」并列陈述；'
        '不构成对原文的复现或证伪；写"未检出正号（不显著）"，不得写"复现成功"、不得写"稳健反号"、不得写"已找到唯一根因"。',
    ]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding='utf-8')
    psay('asm', f'JSON 报告状态 = ok（{OUT_JSON.relative_to(ROOT)}）')
    make_md(S)
    return S

def make_md(S: dict) -> None:
    p = S['parts']

    def row(x, y):
        return f'| {x} | {y} |'

    md = []
    md.append('# S3「项目自建 PC → 预期收益」系数为负 · 深度诊断 H1–H8（2026-10-21；2026-10-27 口径更正）\n')
    md.append('> **2026-10-27 口径更正（Phase 0.2 / V-P1-2 + N-2）**：本报告的「负号/反号」分析仅针对**项目自建 PC**'
              '（对手集合 = 同团全部成员）的描述性排查，**不是对刘亚辉等（2021）PC 变量的复现或证伪**。'
              '原文第四步的对手集合 = 「任何与其相似度超过团内最小相似度的公司」（字面含团外），项目实现 = 同团成员'
              '（`methodology_execution_plan.md:2290` 已登记偏离）；二者不是同一对手集合。因此：① 本报告 H1–H8 的裁决与数值全部保留，'
              '但解释范围限于项目 PC；② 不得把本报告读作「原文复现失败 / 符号相反 / 原文结论被证伪」；'
              '③ 如需对原文 PC 做复现，须按原文第四步规则 + 接近原文词表规模重建（active_vocab ≈ 1.5 万量级；未做）。'
              '主读数数值不变：β = −0.5068%（t = −1.4365）、95% CI [−1.198, +0.185]（覆盖原文 +0.152）、'
              '原文效应处功效 ≈ 0.07、68 个月。\n')
    md.append('> **性质**：只读诊断（复用既有产物 + 新建诊断脚本 `scripts/data_audit/s3_pc_sign_diag_h1h8.py`）；'
              '未覆盖任何既有产物、未改 headline_value / 冻结值 / 主口径网络。')
    md.append('> **修订（REV2，2026-10-21）**：按独立复核意见 F1–F6 修正本报告：H8「团结构代理」结论**撤回**、'
              'H5 降级为**待定（无正面证据）**、H6 归因收窄（仅 log 形式）、H2/H3/H7 精度与单位表述修正 + '
              '补做复核指定替代检验（恒等式分解 / 1/N 控制 / within-N 分箱 / 分布匹配重加权 / 1000 次团标签置换 / 功效-MDE）。')
    md.append('> **判据（先登记，REV2 追加）**：成立 = 主读数层 PC 系数 ≥ 0 且 4 子层 ≥3 层同向转正；'
              '部分 = 向 0 移动 ≥50% 仍为负；排除 = 移动 <25% 或同号；待定 = 数据不可得/证据冲突。')
    md.append('> **背景靶子（不可直接对表）**：`docs/literature/A股产品市场竞争.pdf` 表 4 列(1) +0.152（t=7.76）/ 列(2) +0.130（t=5.40）'
              '；现行读数 = `reports/data_gates/G0/s3_pc_pricing_correspondence.{json,md}`（判定 = 不可比/未复现）。\n')
    md.append('## 0. 裁决总表\n')
    md.append('| # | 假设 | 判定 | 关键数字 |')
    md.append('|---|---|---|---|')
    for tag in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8'):
        vv = S['verdicts'][tag]
        md.append(f"| {tag} | {p[tag].get('hypothesis', '')} | **{vv['verdict']}** | {vv['reason']} |")
    md.append('\n## 1. H1 方向/排序 bug（最高优先）\n')
    h1 = p['H1']
    md.append(f"- 静态读码：`pc = −hhi` 逐值成立（max|pc+hhi| = {h1['code_audit']['max_abs_pc_plus_hhi']}）；"
              f"分组 = 逐月 `rank_average`（升序 → g10 = PC 最大）；H−L = g10 − g1；FM 系数 = dR/dPC。")
    ga = h1['group_audit_decile']
    md.append('- 分组核验（登记样本十分位，pooled）：\n')
    md.append('| 组 | n | mean PC | median PC | mean HHI | 中位团员数 | 组均收益 % |')
    md.append('|---|---|---|---|---|---|---|')
    for i in range(1, 11):
        g = ga['group_stat'][f'D{i}']
        md.append(f"| D{i} | {g['n']:,} | {g['pc_mean']} | {g['pc_median']} | {g['hhi_mean']} | "
                  f"{g['peer_median']:.0f} | {g['r_mean_pct']} |")
    md.append(f"\n- D10−D1：PC 均值差 {ga['D10_minus_D1_pc']}、HHI 均值差 {ga['D10_minus_D1_hhi']}；"
              f"**H 组（D10）PC {'更大' if ga['H_group_is_high_pc'] else '更小'}、HHI {'更小' if ga['H_group_is_low_hhi'] else '更大'}**。")
    inj = h1['injection_test']
    md.append('- 注入检验（合成列，真实数据零改动）：\n')
    md.append('| 合成变量 | FM β（%/单位） | t(NW) | 十分位 H−L EW %（t） | 十分位 H−L VW %（t） |')
    md.append('|---|---|---|---|---|')
    for k, lab in (('syn_rank_pos', 'PC_syn = 月内 R_next 秩 − 0.5（与下月收益正相关）'),
                   ('syn_rank_neg', 'PC_syn = −(月内 R_next 秩 − 0.5)（镜像）'),
                   ('syn_random', '月内均匀随机'), ('real_PC', '真实 PC（对照）'),
                   ('adv_neg_PC', 'PC×(−1) 反证（REV2，应逐位反号）')):
        x = inj[k]
        md.append(f"| {lab} | {x['beta_pct']} | {x['t_nw']} | "
                  f"{x.get('HL_decile_EW_pct', '—')}（{x.get('HL_decile_EW_t', '—')}） | "
                  f"{x.get('HL_decile_VW_pct', '—')}（{x.get('HL_decile_VW_t', '—')}） |")
    md.append(f"\n- 手算 OLS 对拍（逐月 Σ(x−x̄)(y−ȳ)/Σ(x−x̄)² vs 库函数逐月系数）：")
    for mo, hh in h1['hand_ols_check'].items():
        md.append(f"  - {mo}：n = {hh['n']}、手算 β = {hh['beta_pct']}%（库 = {hh['lib_beta_raw']*100:.4f}%）、"
                  f"逐值一致 = {hh['match']}、同月 corr(PC, R_next) = {hh['corr_PC_R_next']}")
    md.append('\n## 2. H2 PC 定义/算式细节\n')
    h2 = p['H2']
    md.append(f"- 原文定义：{h2['paper_definition']['quote']}")
    md.append('\n- 手核团（clusters + 营收 pkl 现算，与 `{y}_pc.csv` 逐位对拍，**全精度**）：\n')
    md.append('| 文本年 | 团 id | 成员数 | 营收覆盖 | 现算 HHI（含自身） | max|Δhhi| vs csv（全精度） | 列 = −HHI_excl（剔除自身的 PC） |')
    md.append('|---|---|---|---|---|---|---|')
    for a in h2['manual_cluster_audit']:
        raw = a.get('csv_hhi_max_abs_diff_raw')
        md.append(f"| {a['text_year']} | {a['cluster_id']} | {a['n_members']} | {a['n_rev_covered']} | "
                  f"{a['hhi_recomputed_incl_self']} | {raw:.3e} | "
                  f"{a.get('csv_col_equals_minus_HHI_excl')} |")
    wpa = h2.get('write_precision_audit', {})
    if wpa:
        md.append(f"- 写盘精度核验（REV2-F4）：最大团 id = {wpa.get('largest_cluster_2019_id')}"
                  f"（{wpa.get('n_members')} 成员）全精度 max|Δhhi| = {wpa.get('csv_hhi_max_abs_diff_raw'):.3e}"
                  f"、max|Δpc| = {wpa.get('csv_pc_max_abs_diff_raw'):.3e}；"
                  f"**理论界 ≤5e-9（pc.csv 写盘 = round(x,8)）** → 原报告「max|Δhhi| = 0」为 r4 显示精度假象，已改记 ≤5e-9")
    ex = h2['excl_self_and_coverage_variants']
    md.append('\n- 变体读数：\n')
    md.append('| 变体 | 样本 | n 行 | 单变量 β %（t） | 加控制 β %（t） | 五分位 α3 %（t） |')
    md.append('|---|---|---|---|---|---|')
    for tag, s in ex.items():
        if tag == 'coverage_eq_1':
            continue
        md.append(f"| 剔除自身（PC_x） | {tag} | {s['n_rows']:,} | {s['univariate']['beta_pct']}"
                  f"（{s['univariate']['t_nw']}） | {s['with_controls']['beta_pct']}（{s['with_controls']['t_nw']}） | "
                  f"{s['quintile_EW_alpha_a3_pct']}（{s['quintile_EW_alpha_a3_t']}） |")
    for tag, s in ex['coverage_eq_1'].items():
        md.append(f"| 营收覆盖 = 1 | {tag} | {s['n_rows']:,} | {s['beta_pct']}（{s['t_nw']}） | — | — |")
    dv = h2['distribution_vs_paper']
    md.append(f"\n- 分布对照：我方 firm-(month) n = {dv['ours_firm_year']['n']:,}、均值 {dv['ours_firm_year']['mean']} / "
              f"中位 {dv['ours_firm_year']['median']} / p25 {dv['ours_firm_year']['p25']} / p75 {dv['ours_firm_year']['p75']} / "
              f"sd {dv['ours_firm_year']['sd']}（原文表 2 n = {dv['paper_table2']['n']:,}、−0.230 / −0.180 / −0.416 / −0.076 / 0.337）；"
              f"营收覆盖率中位 {dv['coverage']['median']}、覆盖 < 1 占比 {dv['coverage']['share_lt1']}")
    md.append(f"- 营收字段：{h2['revenue_field']['interpretation']}；"
              f"`hhi_excl_self` 列语义 = {h2['revenue_field']['excl_self_column_semantics']}")
    md.append('\n## 3. H3 K 校准口径\n')
    h3 = p['H3']
    md.append('| tier（中位 peer 目标） | K(2019) | PC 均值 | PC 中位 | FM 登记 lag12 %（t） | FM 加控制 %（t） | 五分位 EW α3 %（t） |')
    md.append('|---|---|---|---|---|---|---|')
    for t, x in h3['tier_grid'].items():
        md.append(f"| {t} | {x['k_2019']} | {x['pc_mean']} | {x['pc_median']} | "
                  f"{x['fm_reg_registered_lag12']['beta_pct']}（{x['fm_reg_registered_lag12']['t_nw']}） | "
                  f"{x['fm_reg_ctrl_lag12']['beta_pct']}（{x['fm_reg_ctrl_lag12']['t_nw']}） | "
                  f"{x['quintile_EW_alpha_a3_pct']}（{x['quintile_EW_alpha_a3_t']}） |")
    t_abs = [abs(x['fm_reg_registered_lag12']['t_nw']) for x in h3['tier_grid'].values()
             if x['fm_reg_registered_lag12'].get('t_nw') is not None]
    tr10 = h3['tier_grid'].get('10', {})
    md.append(f"\n- {len(h3['tier_grid'])} 档主读数层**无正号 = {h3['all_negative']}**（多档 |t| ≤ 0.55、"
              f"全部不显著，最大 |t| = {max(t_abs):.4f}；细档 PC 由既有 K 重建缓存只读提供、粗档 K≈100–200 由本轮新建 tag "
              f"`dcoarse` 重建；FM 由本脚本逐档重跑）。")
    md.append(f"- tier=10 与原文**中心四分位吻合**：p25 {tr10.get('pc_p25')} vs −0.416、中位 {tr10.get('pc_median')} vs −0.180、"
              f"p75 {tr10.get('pc_p75')} vs −0.076（firm-year n = {tr10.get('n_firm_year')} vs 15,482）；"
              f"但 **sd 0.2204 vs 原文 0.337**（尾部不一致）→ 表述限定为「中心四分位吻合」，不得写「分布高度吻合」（REV2-F4）；"
              f"该档 FM 仍为 −0.3348%（t = −1.04）")
    if h3.get('coarse_K_note'):
        md.append(f"- {h3['coarse_K_note']}")
    md.append('\n## 4. H4 样本口径\n')
    md.append('| 变体 | n 行 | 月数 | 均值市值(亿) | 均值分析师覆盖 | β %（t） | 加控制 β % |')
    md.append('|---|---|---|---|---|---|---|')
    for k, x in p['H4']['variants'].items():
        if 'fm_univariate' not in x:
            md.append(f"| {k} | {x.get('n_rows', '—')} | — | — | — | {x.get('error', x.get('note', '—'))} | — |")
            continue
        md.append(f"| {k} | {x['n_rows']:,} | {x.get('n_months', '—')} | {x.get('mean_market_equity_1e8', '—')} | "
                  f"{x.get('analyst_cov_mean', '—')} | {x['fm_univariate']['beta_pct']}（{x['fm_univariate']['t_nw']}） | "
                  f"{x.get('fm_with_controls', {}).get('beta_pct', '—')}"
                  f"（{x.get('fm_with_controls', {}).get('t_nw', '—')}） |")
    md.append('\n## 5. H5 期间（REV2-F2：判据改以月块 bootstrap CI 为准）\n')
    h5 = p['H5']
    md.append('| 回归年 | 登记样本 β %（t_nw / t_iid） | bootstrap CI95 | LOO β 区间 | 原文最近似样本 β %（t_nw / t_iid） |')
    md.append('|---|---|---|---|---|')
    for y in sorted(h5['by_return_year']['registered']):
        a = h5['by_return_year']['registered'][y]
        b = h5['by_return_year']['paper_like'].get(y, {})
        md.append(f"| {y} | {a.get('beta_pct', '—')}（{a.get('t_nw', '—')} / {a.get('t_iid', '—')}） | "
                  f"{a.get('boot_ci95_pct', '—')} | {a.get('loo_beta_range_pct', '—')} | "
                  f"{b.get('beta_pct', '—')}（{b.get('t_nw', '—')} / {b.get('t_iid', '—')}） |")
    md.append('\n- 子期间：' + '；'.join(f"{k} = {v['beta_pct']}%（t_nw = {v['t_nw']}、t_iid = {v.get('t_iid')}；"
                                        f"bootstrap CI95 {v.get('boot_ci95_pct')}）" for k, v in h5['subperiods'].items()))
    md.append(f"- 逐月系数：负号月占比 {h5['monthly_beta_sign_distribution']['share_negative']}"
              f"、均值 {h5['monthly_beta_sign_distribution']['mean_beta_pct']}%"
              f"、区间 [{h5['monthly_beta_sign_distribution']['min_pct']}, {h5['monthly_beta_sign_distribution']['max_pct']}]%")
    pc5 = h5.get('period_criterion', {})
    md.append(f"- 判据（REV2-F2）：{pc5.get('rule_rev2', '')}")
    md.append(f"- **结论改判：待定（无正面证据；2007–2016 窗口不可测）**；正号且 bootstrap CI 不含 0 的年 = "
              f"{h5.get('positive_and_significant_years') or '无'}（原报告把「2019 显著为正」当作积极证据的表述已收回）")
    md.append('\n## 6. H6 控制集与规格\n')
    h6 = p['H6']
    md.append('| 变量 | 原文表 4 列(2) β %（t） | 我方 β %（t） |')
    md.append('|---|---|---|')
    for c, (pb, pt) in [('size', (0.339, 23.42)), ('bm', (-0.358, -8.79)), ('mom', (1.791, 15.28)),
                        ('rev', (1.835, 8.27)), ('illiq', (0.028, 4.68))]:
        o = h6['control_coefs_vs_paper']['registered']['coefs'].get(c, {})
        md.append(f"| {c} | {pb}（{pt}） | {o.get('beta_pct')}（{o.get('t_nw')}） |")
    md.append(f"- 行业 FE（月×申万二级 去均值，组内 ≥5 家，n = {h6['industry_fe']['n_rows']:,}）："
              f"单变量 β = {h6['industry_fe']['univariate']['beta_pct']}%（t = {h6['industry_fe']['univariate']['t_nw']}）、"
              f"加 5 控制 β = {h6['industry_fe']['with_controls_5']['beta_pct']}%"
              f"（t = {h6['industry_fe']['with_controls_5']['t_nw']}）")
    cs = h6['cluster_size_control']
    psa = h6.get('paper_sign_agreement', {})
    _ncp6 = p.get('H8', {}).get('revision_tests', {}).get('control_function_form', {}).get('variants', {})
    md.append(f"- 团规模控制（REV2-F3 收窄）：log(1+团员数) 单独 β = {cs['log_peers_only']['beta_pct']}%（t = {cs['log_peers_only']['t_nw']}）；"
              f"PC + log 团员 → β_PC = {cs['PC_plus_log_peers']['PC_beta_pct']}%"
              f"（t = {cs['PC_plus_log_peers']['PC_t']}）；再加 5 控制 → β_PC = "
              f"{cs['PC_plus_log_peers_plus_controls']['PC_beta_pct']}%（t = {cs['PC_plus_log_peers_plus_controls']['PC_t']}）；"
              f"**换 1/N 线性形式（1/n_competitors 口径，H6）→ PC + 1/N β_PC = "
              f"{cs.get('PC_plus_inv_peers', {}).get('PC_beta_pct')}%/t = "
              f"{cs.get('PC_plus_inv_peers', {}).get('PC_t')}；PC + 1/N + 5 控制 = "
              f"{cs.get('PC_plus_inv_peers_plus_controls', {}).get('PC_beta_pct')}%/t = "
              f"{cs.get('PC_plus_inv_peers_plus_controls', {}).get('PC_t')}；"
              f"H8 §8R 团成员数口径 1/N = {_ncp6.get('PC+invN', {}).get('PC', {}).get('beta_pct')}%/t = "
              f"{_ncp6.get('PC+invN', {}).get('PC', {}).get('t_nw')}、+ own_share = "
              f"{_ncp6.get('PC+invN+own_share', {}).get('PC', {}).get('beta_pct')}%/t = "
              f"{_ncp6.get('PC+invN+own_share', {}).get('PC', {}).get('t_nw')} → 均仍为负**；"
              f"两种形式均不显著 → 「团规模控制使 PC 转正」**仅限 log 形式**，不构成结构层证据")
    md.append(f"- 控制项符号与原文一致计数（REV2-F3）：同号 {psa.get('n_same_sign')}/{psa.get('n_controls')}、"
              f"相反 {psa.get('n_opposite_sign')}/{psa.get('n_controls')}（{psa.get('opposite_vars')}）；"
              f"mom/illiq 同号 → 表述须限定为「符号相反仅 3/5 控制项」")
    md.append('\n## 7. H7 收益口径/横截面结构\n')
    h7 = p['H7']
    md.append('| 被解释变量 | EW β %（t） | VW β %（t） |')
    md.append('|---|---|---|')
    for k in sorted(set(x.split('|')[0] for x in h7['y_variants'])):
        a = h7['y_variants'].get(f'{k}|EW', {})
        b = h7['y_variants'].get(f'{k}|VW', {})
        md.append(f"| {k} | {a.get('beta_pct', '—')}（{a.get('t_nw', '—')}） | {b.get('beta_pct', '—')}（{b.get('t_nw', '—')}） |")
    md.append(f"\n- 规模三分位内：{ {k: v['beta_pct'] for k, v in h7['size_tercile'].items()} }")
    md.append('- 持有期（收益 = monthly_returns_clean_v2 复利；REV2-F5 附月均折算）：\n')
    md.append('| 窗口 | 窗长(月) | n 行 | β %/单位 PC/期 | β 月均折算 %/月 | t(NW) |')
    md.append('|---|---|---|---|---|---|')
    for k, v in h7['holding_horizons'].items():
        if k == 'note':
            continue
        md.append(f"| {k} | {v.get('window_months', '—')} | {v['n_rows']:,} | {v['beta_pct_per_unit_period']} | "
                  f"{v.get('beta_pct_per_unit_per_month_avg', '—')} | {v['t_nw']} |")
    if h7.get('holding_horizons_rev2'):
        md.append(f"- {h7['holding_horizons_rev2']['reading']}")
    md.append('\n- 组合层：' + '；'.join(f"{k} α3 = {v.get('alpha_a3_pct')}%（t = {v.get('alpha_a3_t')}）"
                                        for k, v in h7['portfolios_alpha'].items()))
    md.append('\n## 8. H8 机械/结构\n')
    h8 = p['H8']
    sc = h8['pc_structure_corr']
    md.append(f"- corr(PC, log 团员数) = {sc['corr_PC_log_cluster_size']}、corr(PC, 自身收入份额) = {sc['corr_PC_own_revenue_share']}、"
              f"corr(PC, 团内收入 CV) = {sc['corr_PC_cluster_rev_cv']}、corr(PC, −行业 HHI) = {h8['industry_hhi']['corr_PC_indHHI_neg']}"
              f"（n = {sc['n_rows']:,}）")
    md.append('- PC 按团规模五分位：\n')
    md.append('| 组 | n | mean PC | mean HHI | 自身收入份额均值 | 中位团员数 |')
    md.append('|---|---|---|---|---|---|')
    for k, v in sc['pc_group_by_size'].items():
        md.append(f"| {k} | {v['n']:,} | {v['pc_mean']} | {v['hhi_mean']} | {v['own_share_mean']} | {v['peer_median']:.0f} |")
    rt = h8['residual_test']
    md.append(f"\n- 残差检验：{rt['spec']}；载荷 = {rt['loadings']}、R² = {rt['r2']}")
    for k, v in rt.items():
        if k.startswith('fm_'):
            md.append(f"  - FM({k[3:]}) = {v.get('beta_pct')}%（t = {v.get('t_nw')}）"
                      if 'beta_pct' in v else f"  - FM({k[3:]}) = {v}")
    md.append(f"- 持续性：池化 AR(1) corr = {h8['persistence']['pooled_AR1_corr']}、逐年 = "
              f"{[x['corr_PC'] for x in h8['persistence']['by_year_pairs']]}、"
              f"均值 |ΔPC| = {h8['persistence']['mean_pc_change_abs']}")

    # ---- REV2（F1）：H8 撤回与替代检验
    rv8 = h8.get('revision_tests', {})
    if rv8:
        wd = rv8.get('withdrawal', {})
        md.append('\n### 8R. H8 修订（REV2-F1）：撤回与替代检验（独立重算）\n')
        md.append(f"- **撤回**：{wd.get('retracted_conclusion', '')}")
        md.append('- 复核三条反驳（原文收录；每条后附本轮独立重算对照）：\n')
        for i, r in enumerate(wd.get('review_rebuttals_verbatim', []), 1):
            md.append(f"  {i}. {r}")
        idn = rv8.get('identity_decomposition', {})
        md.append(f"\n- (a) 恒等式分解（n = {idn.get('n_rows'):,}，恒等式残差 max = {idn.get('max_abs_resid_identity'):.2e}，"
                  f"两分量 corr = {idn.get('corr_two_components')}）：\n")
        md.append('| 分量 | mean | sd | FM β %/单位 | t(NW) |')
        md.append('|---|---|---|---|---|')
        for tag, lab in (('t_own', '−own_share²'), ('t_peer', '−同团他人集中度')):
            z = idn.get(tag, {})
            d0 = idn.get('descriptives', {}).get(tag, {})
            md.append(f"| {lab} | {d0.get('mean')} | {d0.get('sd')} | {z.get('beta_pct')} | {z.get('t_nw')} |")
        md.append(f"- 联合 FM：{idn.get('fm_joint')}（两分量各自为负、均不显著 → 不存在「某结构分量吸收负号」的证据）")
        cf = rv8.get('control_function_form', {}).get('variants', {})
        md.append(f"\n- (b) 机械控制函数形式：{cf.get('PC+invN', {}).get('PC', {})}（PC+1/N）、"
                  f"{cf.get('PC+invN+own_share', {}).get('PC', {})}（PC+1/N+own_share）、"
                  f"{cf.get('PC+own_share', {}).get('PC', {})}（PC+own_share）→ **1/N 形式仍为负**"
                  f"（对照 log 形式 +0.3221）")
        wnb = rv8.get('within_N_bins', {}).get('bins', {})
        md.append('\n- (c) within-N 分箱（同团规模内部梯度；min_n 放宽到 20，已标注）：\n')
        md.append('| 分箱 | n | 样本占比 | 中位 N | PC sd | FM β %/单位 | t(NW) |')
        md.append('|---|---|---|---|---|---|---|')
        for k, v2 in wnb.items():
            md.append(f"| {k} | {v2.get('n'):,} | {v2.get('sample_share')} | {v2.get('median_N')} | "
                      f"{v2.get('pc_sd')} | {v2.get('beta_pct', '—（无变异/样本不足）')} | {v2.get('t_nw', '—')} |")
        drr = rv8.get('distribution_reweighting', {})
        md.append('\n- (d) 分布匹配重加权（原文表 2 PC 分位密度；REV2 新增检验）：\n')
        md.append(f"  - 观测四分位份额 = {drr.get('observed_firm_year_bins', {}).get('obs_shares')}"
                  f"（目标 0.25×4；firm-year n = {drr.get('observed_firm_year_bins', {}).get('n_firm_year'):,}）、"
                  f"重要性权重 = {drr.get('observed_firm_year_bins', {}).get('bin_weights')}")
        md.append('| 实现 | 登记样本 β %/单位 | t(NW) | 原文最近似样本 β %/单位 | t(NW) |')
        md.append('|---|---|---|---|---|')
        for tag, lab in (('w_dist', '重要性重加权 WLS（四分位权重）'), ('w_dist_cap10', '同上 + 权重上限 10'),
                         ('quantile_match', '分位匹配（秩 → 原文分段线性分位函数）')):
            r_ = drr.get(f'registered|{tag}', {})
            pl = drr.get(f'paper_like|{tag}', {})
            md.append(f"| {lab} | {r_.get('beta_pct')} | {r_.get('t_nw')} | {pl.get('beta_pct')} | {pl.get('t_nw')} |")
        md.append(f"  - 重加权后加 5 控制：登记 β_PC = {drr.get('registered|w_dist+controls', {}).get('PC_beta_pct')}%"
                  f"（t = {drr.get('registered|w_dist+controls', {}).get('PC_t')}）；"
                  f"分位匹配五分位 EW：{drr.get('registered|quantile_match|quintile_EW')}")
        md.append(f"  - 变换后分布：{drr.get('transformed_distribution', {}).get('PC_qm')}"
                  f"（firm-year 均值/中位/p25/p75/sd）；权重摘要 {drr.get('transformed_distribution', {}).get('weight_summary')}")
        pmp = rv8.get('cluster_label_permutation', {})
        md.append(f"\n- (e) 团标签置换 **{pmp.get('n_iter')} 次**（逐年重贴「团 → HHI」对应）："
                  f"零分布 mean = {pmp.get('perm_beta_mean_pct')}%、sd = {pmp.get('perm_beta_sd_pct')}%、"
                  f"CI95 = {pmp.get('perm_ci95_pct')}%（min {pmp.get('perm_min_pct')} / max {pmp.get('perm_max_pct')}）；"
                  f"观测 β = {pmp.get('observed_beta_pct')}% → **单侧 p = {pmp.get('p_one_sided_le_observed')}**、"
                  f"双侧 p = {pmp.get('p_two_sided')}；置换 MDE(80%) ≈ {pmp.get('mde80_from_perm_sd_pct')}%")
        pmd = rv8.get('power_mde', {}).get('main_reading_registered_univariate', {})
        md.append(f"\n- (f) 功效 / MDE（主读数）：se = {pmd.get('se_pct')}%、CI95 = {pmd.get('ci95_pct')}%、"
                  f"MDE(80% 功效) = {pmd.get('mde80_pct')}%/单位、原文效应 +{pmd.get('effect_ref_pct')} 处功效 = "
                  f"**{pmd.get('power_at_effect_ref')}** → 在 t = {pmd.get('t_nw')}、四层子读数全不显著的前提下，"
                  f"只能写「**未检出正号（不显著）**」，不得写「稳健反号」")
        md.append(f"- 8R 判据：{rv8.get('status', '')}")

    md.append('\n## 9. 根因判定（修正版，REV2）\n')
    rc = S.get('revised_root_cause', {})
    if rc.get('final_conclusion'):
        md.append(f"**总结论（一句话）**：{rc['final_conclusion']}\n")
    if rc.get('retracted'):
        md.append('**撤回**：\n')
        for it in rc['retracted']:
            md.append(f'- {it}')
    if rc.get('excluded_strong'):
        md.append('\n**已排除（强证据）**：\n')
        for it in rc['excluded_strong']:
            md.append(f'- {it}')
    if rc.get('pending'):
        md.append('\n**待定**：\n')
        for it in rc['pending']:
            md.append(f'- {it}')
    md.append('\n（候选排序与支持度见下表；#1 已撤回，#2/#3 保持待定）\n')
    if 'summary_ranking' in S:
        md.append('| 排序 | 候选 | 支持度 | 依据 |')
        md.append('|---|---|---|---|')
        for i, it in enumerate(S['summary_ranking'], 1):
            md.append(f"| {i} | {it['candidate']} | {it['support']} | {it['evidence']} |")
    if 'retracted_conclusions' in S:
        md.append('\n**撤回/收回/修正明细**：\n')
        for it in S['retracted_conclusions']:
            md.append(f'- {it}')
    if 'open_items' in S:
        md.append('\n**仍未排除 / 不可测**：\n')
        for it in S['open_items']:
            md.append(f'- {it}')
    if 'next_steps' in S:
        md.append('\n**下一步建议**：\n')
        for it in S['next_steps']:
            md.append(f'- {it}')

    md.append('\n## 10. 修订记录（REV2，复核意见 F1–F6 响应）\n')
    rv = S.get('revision', {})
    md.append(f"- 修订轮：{rv.get('revision_id')}（{rv.get('date')}）；触发：{rv.get('trigger')}")
    for it in rv.get('changes', []):
        md.append(f'- {it}')
    md.append('\n**修法 → 修正后数值/原文摘录（重点）**：\n')
    # 自包含取值（不依赖上文块内变量，防块间耦合）
    _rt10 = p.get('H8', {}).get('revision_tests', {})
    _drr = _rt10.get('distribution_reweighting', {})
    _pmp = _rt10.get('cluster_label_permutation', {})
    _wnb = _rt10.get('within_N_bins', {}).get('bins', {})
    _ncp = _rt10.get('control_function_form', {}).get('variants', {})
    _cs6 = p.get('H6', {}).get('cluster_size_control', {})
    _psa = p.get('H6', {}).get('paper_sign_agreement', {})
    _y19 = p.get('H5', {}).get('by_return_year', {}).get('paper_like', {}).get('2019', {})
    _wpa = p.get('H2', {}).get('write_precision_audit', {})
    md.append(f'- F1 分布匹配重加权（新检验）：重要性 WLS（原文四分位份额 → 权重 '
              f'{_drr.get("observed_firm_year_bins", {}).get("bin_weights")}）β = '
              f'{_drr.get("registered|w_dist", {}).get("beta_pct")}%（t = {_drr.get("registered|w_dist", {}).get("t_nw")}）、'
              f'分位匹配（秩 → 原文分段线性分位函数）β = {_drr.get("registered|quantile_match", {}).get("beta_pct")}%'
              f'（t = {_drr.get("registered|quantile_match", {}).get("t_nw")}）；原文最近似样本 = '
              f'{_drr.get("paper_like|w_dist", {}).get("beta_pct")}%（WLS）/'
              f'{_drr.get("paper_like|quantile_match", {}).get("beta_pct")}%（分位匹配）→ **两者均未转正**（分布口径不解释负号）')
    md.append(f'- F1 团标签置换 **{_pmp.get("n_iter")} 次**：mean = {_pmp.get("perm_beta_mean_pct")}%、'
              f'sd = {_pmp.get("perm_beta_sd_pct")}、CI95 = {_pmp.get("perm_ci95_pct")}%、'
              f'**单侧 p = {_pmp.get("p_one_sided_le_observed")}**（恰在 5% 临界）、双侧 p = {_pmp.get("p_two_sided")}、'
              f'置换 MDE(80%) ≈ {_pmp.get("mde80_from_perm_sd_pct")}%'
              f'（复核实测 200 次对照：mean +0.049 / sd 0.296 / CI95 [−0.498, +0.585] / P(≤−0.5068) = 2.5%）')
    md.append('- F1 within-N 表（§8R(c) 汇总）：' +
              '；'.join(f"{k} β={v2.get('beta_pct')}（t={v2.get('t_nw')}；share={v2.get('sample_share')}）"
                       for k, v2 in _wnb.items()))
    md.append(f'- F2 H5 统计量并报（数值见 §5）：2019 原文最近似样本 t_nw = {_y19.get("t_nw")}、'
              f't_iid = {_y19.get("t_iid")}、bootstrap CI95 {_y19.get("boot_ci95_pct")} 含 0、'
              f'LOO β 区间 {_y19.get("loo_beta_range_pct")}；正号且 CI 不含 0 的年 = 无 → 判据改 bootstrap CI 后'
              f'**无正面证据**；根因排序 #2 的「2019 显著为正」已收回')
    md.append(f'- F3 H6 归因收窄：log 形式 β_PC = {_cs6.get("PC_plus_log_peers", {}).get("PC_beta_pct")}%'
              f'（t = {_cs6.get("PC_plus_log_peers", {}).get("PC_t")}，不显著）；1/N（团成员数）'
              f'{_ncp.get("PC+invN", {}).get("PC", {}).get("beta_pct")}%、+own_share '
              f'{_ncp.get("PC+invN+own_share", {}).get("PC", {}).get("beta_pct")}% 仍为负；'
              f'控制项符号相反仅 {_psa.get("n_opposite_sign")}/{_psa.get("n_controls")}（{_psa.get("opposite_vars")}）')
    md.append('- F4 精度/强度表述：H2「max|Δhhi| = 0」→「≤ 5e-9（round(x,8) 写盘精度；最大团实测 '
              f'{_wpa.get("csv_hhi_max_abs_diff_raw", float("nan")):.3e}）」；H3「11/11 为负」→「无正号'
              '（多档 |t| ≤ 0.55，均不显著，最大 |t| = 1.80）」；H3 分布「高度吻合」→「中心四分位吻合（p25/中位/p75），'
              'sd 0.2204 vs 原文 0.337」；H2 营收字段 →「盘上 pkl = 营业收入（854.30 亿），原文附录表未核」')
    md.append('- F5 H7 单位与重叠窗：持有期 β 折算月均 = ' +
              str({k: v.get('beta_pct_per_unit_per_month_avg')
                   for k, v in p.get('H7', {}).get('holding_horizons', {}).items() if k.startswith('h_')}) +
              ' → 无反转、月均幅度基本不变；t+2..6 / t+7..12 等为重叠窗且 NW(6) 未做重叠校正 → t 不作推断依据')
    md.append('- F6 治理：本轮产物登记进 version_manifest（诊断脚本 + 8 分片 + 报告两件 + dcoarse 证据 = 12 条）；'
              'registry/spec 三角刷新；四审计顺序重跑；五 YAML + 三哈希闭环 + manifest 全量 sha 扫描（见 §11）')
    if 'excluded' in S:
        md.append('\n**已排除（原表）**：' + '；'.join(S['excluded']))
    md.append('\n## 11. 产物与纪律\n')
    md.append('- 新建脚本：`scripts/data_audit/s3_pc_sign_diag_h1h8.py`；分片 JSON：`artifacts/tmp/pc_diag/H1..H8.json`；'
              '进度：`artifacts/tmp/pc_sign_diag_progress.md`；本报告两件。')
    md.append('- 复用（只读）：`artifacts/tmp/pc_rc/h1_net_t*/`（K 档 PC 缓存）、'
              '`data/processed/network_ma_teacher/{2017..2022}_pc.csv` + `clusters/`、'
              '`data/processed/m1_common_sample_ma_teacher_v2.parquet`、`data/processed/monthly_returns_clean_v2.csv`、'
              '`data/processed/factor_returns_daily.csv`、`data/raw/drive/operating_revenue_annual.pkl`、'
              '`data/processed/sw2_industry_latest.csv`。')
    md.append('- 本轮新建（粗 K 档，未覆盖既有）：`artifacts/tmp/pc_rc/dcoarse_net_t{150,200,300}/`（仓库重建器新建 tag）、'
              '`artifacts/tmp/pc_rc/rc_rebuild_dcoarse.json`、`artifacts/tmp/pc_diag/{H1..H8}.json` + '
              '`artifacts/tmp/pc_diag/{pc_report_dcoarse_t*,dcoarse_*,run_*,final_h3}.log`。')
    md.append('- REV2 复核证据（只读引用，不在本轮重算其数值）：`artifacts/tmp/review_diag/{rv_lib.py,A_injection.py,'
              'B_h2.py,C_h3h4.py,D_h5.py,E_h6.py,F_h8.py,F2_h8_identity.py}` 及同名 JSON。')
    md.append('- 治理登记（REV2-F6，与本报告同轮）：本轮产物（诊断脚本 + 8 分片 + 报告两件 + dcoarse 证据）登记进 '
              '`configs/version_manifest.yaml`；registry 追加 change_log + spec 双锚点刷新（三角刷新）；'
              'Gate 1 四审计按 entry → provenance → metadata → consistency 顺序重跑；五 YAML safe_load + 重复键；'
              '三哈希闭环 + manifest 全量 sha 扫描 —— 终态证据 = `artifacts/tmp/pc_diag/diag_governance.json` '
              '与 `artifacts/tmp/pc_diag/diag_closeout_check.json`。')
    md.append('- 未覆盖任何既有产物、未改任何 headline_value / 冻结值 / 主口径网络；注入检验仅在内存构造合成列；'
              'REV2 的分布重加权/置换/功效检验全部只读（内存计算），不改任何盘上数据。')
    md.append('')
    OUT_MD.write_text('\n'.join(md), encoding='utf-8', newline='\n')
    print(f'  -> {OUT_MD}')

