"""实验注册表与全局种子。

约定：
- 种子只有一处定义（SEEDS），实验不得自带种子集合。
- 命令一律以仓库根为 cwd，脚本路径相对仓库根。
- 产物文件名沿用 snapshot 脚本既定命名，这里只负责声明与归拢。
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

CONFIG_FILES: tuple[str, ...] = (
    'configs/parameter_registry.yaml',
    'configs/s3a_final_spec_v1.yaml',
    'configs/version_manifest.yaml',
    'configs/m1_contracts.yaml',
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
        title='M1 样本构建（run-all 只跑一次）',
        steps=(Step('scripts/data_audit/s3_m1_1_build_sample.py'),),
        requires=(REGISTRY_CONFIG, 'data/processed'),
        outputs=(
            'reports/data_gates/G0/s3_m1_sample_report*.json',
            'data/processed/m1_common_sample_*.parquet',
        ),
        scope='prepare',
        runtime_hint='分钟级到小时级',
    ),
    Experiment(
        id='network-business-text',
        title='M-A 产品市场竞争网络构边（老师 MD&A 语料）',
        steps=(Step('scripts/data_audit/s3_ma_build_edges.py', ('--corpus', 'teacher')),),
        requires=('configs/version_manifest.yaml', 'data/processed'),
        outputs=('data/processed/network_ma_teacher/**',),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='network-mda-text',
        title='M-B 文本竞争网络：全期矩阵 → τ 校准 → 构边',
        steps=(
            Step('scripts/data_audit/s3_mb_full_matrix.py', ('--years', '2017-2025')),
            Step('scripts/data_audit/s3_b1_calibrate_tau.py'),
            Step('scripts/data_audit/s3_mb_build_edges.py', ('--year', '2021')),
        ),
        requires=(REGISTRY_CONFIG, 'data/raw/teacher/mda_data_2023.csv'),
        outputs=(
            'data/processed/network_mb_teacher/**',
            'reports/data_gates/G0/s3_b1_tau_calibration*.json',
        ),
        smoke_args=('--years', '2021'),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='peermax-panel',
        title='S4：MAX/PeerMAX 面板（主口径 ma_teacher）',
        steps=(
            Step('scripts/data_audit/s4_step1_max.py'),
            Step(
                'scripts/data_audit/s4_step2_peermax_formal.py',
                (
                    '--use-case', 'ma',
                    '--net-dir', 'data/processed/network_ma_teacher',
                    '--corpus-id', 'ma_teacher',
                ),
            ),
        ),
        requires=(REGISTRY_CONFIG, 'data/processed/network_ma_teacher'),
        outputs=(
            'data/processed/monthly_pricing_panel_*_v4.csv',
            'reports/data_gates/G0/s4_step2_*_v4_report.json',
        ),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='fm-main-table',
        title='M1 主表：Fama-MacBeth + horse race',
        steps=(Step('scripts/data_audit/s3_m1_2_fm_main.py', ('--sample-version', 'v3')),),
        requires=(REGISTRY_CONFIG, 'data/processed'),
        outputs=(
            'reports/data_gates/G0/s3_m1_pricing_table*.json',
            'reports/data_gates/G0/s3_m1_pricing_table*.md',
            'reports/data_gates/G0/s3_m1_horse_race*.json',
        ),
        metrics=(
            MetricSpec('beta_peer', 'reports/data_gates/G0/s3_m1_pricing_table{sfx}.json',
                       'main.ma_teacher.coefs.PeerMAXraw.beta'),
            MetricSpec('t_peer', 'reports/data_gates/G0/s3_m1_pricing_table{sfx}.json',
                       'main.ma_teacher.coefs.PeerMAXraw.t'),
        ),
        suffix_arg='--out-suffix',
        runtime_hint='分钟级',
    ),
    Experiment(
        id='portfolio-sorts',
        title='M1 组合表：五分位/十分位 spread',
        steps=(Step('scripts/data_audit/s3_m1_3_portfolio_shape.py', ('--sample-version', 'v2')),),
        requires=(REGISTRY_CONFIG, 'data/processed'),
        outputs=(
            'reports/data_gates/G0/s3_m1_portfolio_deciles*.json',
            'reports/data_gates/G0/s3_m1_diagnostics*.json',
        ),
        suffix_arg='--out-suffix',
        runtime_hint='分钟级',
    ),
    Experiment(
        id='robustness-checks',
        title='M1 稳健性：placebo 与长度/行业控制',
        steps=(
            Step('scripts/data_audit/s3_m1_4_robustness_placebo.py'),
            Step('scripts/data_audit/s3_m1_5_length_industry.py'),
        ),
        requires=(REGISTRY_CONFIG, 'data/processed'),
        outputs=(
            'reports/data_gates/G0/s3_m1_robustness_placebo.json',
            'reports/data_gates/G0/s3_m1_length_control.json',
            'reports/data_gates/G0/s3_m1_industry_absorb.json',
        ),
        runtime_hint='分钟级',
    ),
    Experiment(
        id='gate-report',
        title='M1 数据门报告',
        steps=(Step('scripts/data_audit/s3_m1_6_gate_report.py'),),
        requires=(REGISTRY_CONFIG, 'data/processed'),
        outputs=(
            'reports/data_gates/G0/s3_m1_gate_report*.json',
            'reports/data_gates/G0/s3_m1_gate_report*.md',
        ),
        suffix_arg='--out-suffix',
        runtime_hint='秒级到分钟级',
    ),
    Experiment(
        id='shadow-inference',
        title='Phase3 shadow 推断层（bootstrap，随 seed 变化）',
        steps=(Step('scripts/data_audit/p3_4_inference_shadow.py'),),
        requires=(REGISTRY_CONFIG, 'artifacts/phase3'),
        outputs=('artifacts/phase3/3.4_*.csv',),
        accepts_seed=True,
        runtime_hint='分钟级',
    ),
    Experiment(
        id='density-null',
        title='网络密度零模型（随机重连，随 seed 变化）',
        steps=(Step('scripts/data_audit/s3_network_density_null.py'),),
        requires=(REGISTRY_CONFIG, 'data/processed'),
        outputs=('reports/data_gates/G0/s3_network_density_null.json',),
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
