"""实验注册表与全局种子。

约定：
- 种子只有一处定义（SEEDS），实验不得自带种子集合。
- 命令一律以仓库根为 cwd，脚本路径相对仓库根。
- 产物文件名沿用 snapshot 脚本既定命名，这里只负责声明与归拢。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

_LABELS = {'pricing_sample_v1_2015_2023': 'v1', 'pricing_sample_v2_2015_2023': 'v2',
           'pricing_sample_v3_20261115': 'v3', 'pricing_sample_v4_20261116': 'v4'}


def headline_sample_label(root=None) -> str:
    """headline 样本版本标签：读 registry 指针，不在实验表里写死 'v3'。

    缺 sample_and_spec_registry.sample_version 或版本不在已登记集合时抛错
    （fail-closed，不静默回退）。
    """
    import yaml
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    doc = yaml.safe_load((base / REGISTRY_CONFIG).read_text(encoding='utf-8'))
    entry = next(p for p in doc['parameters'] if p.get('name') == 'sample_and_spec_registry')
    return _LABELS[entry['headline_value']['sample_version']]


def headline_sample_path(root=None) -> str:
    """headline 样本 parquet 的相对路径（与 pricing_lib.sample_path 同口径）。"""
    return f'data/processed/pricing_sample_ma_teacher_{headline_sample_label(root)}.parquet'

CONFIG_FILES: tuple[str, ...] = (
    'configs/parameter_registry.yaml',
    'configs/analyst_spec.yaml',
    'configs/version_manifest.yaml',
    'configs/pricing_inputs.yaml',
    'configs/claim_registry.yaml',
    'configs/scope_card.yaml',
)

REGISTRY_CONFIG = 'configs/parameter_registry.yaml'


@dataclass(frozen=True)
class Step:
    """一次子进程调用。"""

    script: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricSpec:
    """从产物 JSON 中取一个标量。file_glob 支持 {seed} 占位。"""

    name: str
    file_glob: str
    path: str


@dataclass(frozen=True)
class Experiment:
    id: str
    title: str
    steps: tuple[Step, ...]
    requires: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    metrics: tuple[MetricSpec, ...] = ()
    smoke_args: tuple[str, ...] = ()
    accepts_seed: bool = False
    suffix_arg: str | None = None
    scope: str = 'per_seed'  # prepare | per_seed
    runtime_hint: str = ''

    def missing_inputs(self, root) -> list[str]:
        from pathlib import Path

        base = Path(root)
        return [p for p in self.requires if not (base / p).exists()]


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        id='estimation-sample',
        title='pricing_gate 样本构建（run-all 只跑一次）',
        steps=(Step('scripts/pricing_sample_build.py'),),
        requires=(REGISTRY_CONFIG, 'data/processed/monthly_pricing_panel_ma_teacher.csv'),
        outputs=(
            'reports/data_gates/G0/pricing_sample_report*.json',
            'data/processed/pricing_sample_*.parquet',
        ),
        scope='prepare',
        runtime_hint='分钟级到小时级',
    ),
    Experiment(
        id='network-business-text',
        title='M-A 产品市场竞争网络构边（老师 MD&A 语料）',
        steps=(Step('scripts/network_business_edges_build.py', ('--corpus', 'teacher')),),
        requires=('configs/version_manifest.yaml', 'data/raw/teacher/mda_data_2023.csv',
                  'data/processed/product_lexicon_v2_teacher.csv'),
        outputs=('data/processed/network_ma_teacher/**',),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='network-mda-text',
        title='M-B 文本竞争网络：全期矩阵 → τ 校准 → 构边',
        steps=(
            Step('scripts/network_mda_full_matrix.py', ('--years', '2017-2025')),
            Step('scripts/network_tau_calibrate.py'),
            Step('scripts/network_mda_edges_build.py', ('--year', '2021')),
        ),
        requires=(REGISTRY_CONFIG, 'data/raw/teacher/mda_data_2023.csv',
                  'data/processed/teacher_mda_panel.csv'),
        outputs=(
            'data/processed/network_mb_teacher/**',
            'reports/data_gates/G0/network_tau_calibration*.json',
        ),
        smoke_args=('--years', '2021'),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='peermax-panel',
        title='panel：MAX/PeerMAX 面板（主口径 ma_teacher）',
        steps=(
            Step('scripts/peermax_max_build.py'),
            Step(
                'scripts/peermax_panel_build.py',
                (
                    '--use-case', 'ma',
                    '--net-dir', 'data/processed/network_ma_teacher',
                    '--corpus-id', 'ma_teacher',
                ),
            ),
        ),
        requires=(REGISTRY_CONFIG, 'data/processed/network_ma_teacher', 'data/processed/daily_panel'),
        outputs=(
            'data/processed/monthly_pricing_panel_*_v4.csv',
            'reports/data_gates/G0/peermax_panel_build_*_v4_report.json',
        ),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='fm-main-table',
        title='pricing_gate 主表：Fama-MacBeth + horse race',
        steps=(Step('scripts/pricing_fm_main.py',
                    ('--sample-version', headline_sample_label())),),
        requires=(REGISTRY_CONFIG, headline_sample_path()),
        outputs=(
            'reports/data_gates/G0/pricing_table*.json',
            'reports/data_gates/G0/pricing_table*.md',
            'reports/data_gates/G0/pricing_horse_race*.json',
        ),
        metrics=(
            MetricSpec('beta_peer', 'reports/data_gates/G0/pricing_table{sfx}.json',
                       'main.ma_teacher.coefs.PeerMAXraw.beta'),
            MetricSpec('t_peer', 'reports/data_gates/G0/pricing_table{sfx}.json',
                       'main.ma_teacher.coefs.PeerMAXraw.t'),
        ),
        suffix_arg='--out-suffix',
        runtime_hint='分钟级',
    ),
    Experiment(
        id='portfolio-sorts',
        title='pricing_gate 组合表：五分位/十分位 spread',
        steps=(Step('scripts/pricing_portfolio_sorts.py',
                    ('--sample-version', headline_sample_label())),),
        requires=(REGISTRY_CONFIG, headline_sample_path()),
        outputs=(
            'reports/data_gates/G0/pricing_portfolio_deciles*.json',
            'reports/data_gates/G0/pricing_diagnostics*.json',
        ),
        suffix_arg='--out-suffix',
        runtime_hint='分钟级',
    ),
    Experiment(
        id='robustness-checks',
        title='pricing_gate 稳健性：placebo 与长度/行业控制',
        steps=(
            Step('scripts/pricing_robustness_placebo.py'),
            Step('scripts/pricing_length_industry_control.py'),
        ),
        requires=(REGISTRY_CONFIG, headline_sample_path()),
        outputs=(
            'reports/data_gates/G0/pricing_robustness_placebo.json',
            'reports/data_gates/G0/pricing_length_control.json',
            'reports/data_gates/G0/pricing_industry_absorb.json',
        ),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='gate-report',
        title='pricing_gate 数据门报告',
        steps=(Step('scripts/pricing_gate_report.py'),),
        requires=(REGISTRY_CONFIG,
                  'reports/data_gates/G0/pricing_table.json',
                  'reports/data_gates/G0/pricing_portfolio_deciles.json'),
        outputs=(
            'reports/data_gates/G0/pricing_gate_report*.json',
            'reports/data_gates/G0/pricing_gate_report*.md',
        ),
        suffix_arg='--out-suffix',
        runtime_hint='秒级到分钟级',
    ),
    Experiment(
        id='shadow-inference',
        title='Phase3 shadow 推断层（bootstrap，随 seed 变化）',
        steps=(Step('scripts/shadow_inference.py'),),
        requires=(REGISTRY_CONFIG, 'artifacts/phase3'),
        outputs=('artifacts/phase3/3.4_*.csv',),
        accepts_seed=True,
        runtime_hint='分钟级',
    ),
    Experiment(
        id='density-null',
        title='网络密度零模型（随机重连，随 seed 变化）',
        steps=(Step('scripts/network_density_null.py'),),
        requires=(REGISTRY_CONFIG, 'data/processed/network_ma_teacher'),
        outputs=('reports/data_gates/G0/network_density_null.json',),
        accepts_seed=True,
        runtime_hint='分钟级到小时级（R=1000 重连）',
    ),
)

BY_ID: dict[str, Experiment] = {exp.id: exp for exp in EXPERIMENTS}


def get(exp_id: str) -> Experiment:
    try:
        return BY_ID[exp_id]
    except KeyError:
        raise KeyError(f'未知实验 {exp_id!r}；可选：{", ".join(BY_ID)}') from None

# 每个实验的 purpose 与 notes：实验 spec 从 md 迁到代码，peerlab experiments 直接打印，lint 校验非空。
EXPERIMENT_NOTES: dict[str, dict] = {
    'estimation-sample': {
        'purpose': '构建 pricing_gate/spec_no_turnover 估计样本（四列口径，v3 为主）',
        'notes': (
            '参数：--v2-report-only、--force/--force-reason、--rebuild-amount-cache',
            '口径：完整月 n_valid_days>=15、microcap 形成月 q10、次新股 <6 月、IQR 池化登记值',
            '坑：缺 registry 导入即 SystemExit(2)；覆盖既有样本必须给 --force-reason',
        ),
    },
    'network-business-text': {
        'purpose': 'M-A 产品市场网络构边（--corpus teacher 为主口径）',
        'notes': (
            '上游：text_business_extract -> network_business_cluster_grid -> cluster_freeze',
            '产物：data/processed/network_ma_teacher/，不覆盖 network_ma/',
            '坑：本实验 steps 只做构边，聚类冻结要先跑；缺聚类目录会 blocked',
        ),
    },
    'network-mda-text': {
        'purpose': 'M-B 老师 MD&A 语料的 TF-IDF 文本网络',
        'notes': (
            '链路：network_mda_full_matrix -> network_tau_calibrate -> network_mda_edges_build',
            '产物：data/processed/network_mb_teacher/；network_mb_stub/ 只是占位产物',
            '坑：不给 --tau 走占位模式，产物不入分析',
        ),
    },
    'peermax-panel': {
        'purpose': 'panel MAX/PeerMAX 定价面板（主口径 ma_teacher）',
        'notes': (
            '上游：两条网络 + 日/月收益 + 股本；缺 network_ma_teacher 即 blocked',
            '主口径 ma_teacher，mb_teacher 为方法稳健列',
            '验收：peermax_panel_acceptance 的 C1 三轮全过',
        ),
    },
    'fm-main-table': {
        'purpose': 'pricing_gate Fama-MacBeth 主表与 distinctness horse race',
        'notes': (
            '参数：--sample-version 缺省 v3、--out-suffix _s{seed}',
            'registry：nw_lag、min_focal_count_per_month、min_effective_month_count、iqr_pool_value',
            '验收：n_months_used 与样本报告一致',
        ),
    },
    'portfolio-sorts': {
        'purpose': '五分位/十分位组合表与 spread',
        'notes': (
            '参数：--sample-version 缺省 v2（与 fm-main-table 的 v3 不同，属既有口径）',
            'registry：portfolio_breakpoints、K_Spread_Pricing',
            '验收：spread 与连续变量系数同号',
        ),
    },
    'robustness-checks': {
        'purpose': 'placebo、篇幅控制、行业吸收',
        'notes': (
            '参数：--sample-version 缺省 v2',
            '产物：G0/pricing_robustness_placebo.json、pricing_length_control.json、pricing_industry_absorb.json',
            '验收：placebo 不显著；行业吸收后符号方向与主表一致',
        ),
    },
    'gate-report': {
        'purpose': '五项判定 汇总成 pricing_gate_status 与停机动作',
        'notes': (
            '参数：--out-suffix 与 --input-suffix 成对，读同种子上游产物',
            '上游缺失记 NOT_RUN，不中断整份报告',
            '验收：无 fail 项，NOT_RUN 逐条有原因',
        ),
    },
    'shadow-inference': {
        'purpose': 'Phase3 shadow 推断层（bootstrap，随种子变化）',
        'notes': (
            '参数：--seed（缺省 20261026；本仓按全局 5 seed 跑）',
            '产物落 artifacts/phase3/，不进 reports/data_gates/G0',
            '验收：四个 CSV 齐全，分层读数与主表方向一致',
        ),
    },
    'density-null': {
        'purpose': '网络密度零模型（同 N 同 E 随机重连）',
        'notes': (
            '参数：--reps 缺省读 registry rewiring_reps=1000，冒烟用 --reps 10',
            '真随种子变化；产物 G0/network_density_null.json',
            '验收：观测密度与零分布同时落盘，R 与 seed 记在产物里',
        ),
    },
}


def notes_for(exp_id: str) -> dict:
    return EXPERIMENT_NOTES.get(exp_id, {'purpose': '', 'notes': ()})
