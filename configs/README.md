# configs/

六个冻结配置。参数来源分三类：契约测试断言、源码 docstring、设计口径取值，每条在文件里带 `note`；数值只登记在本目录，`spec/` 用 `registry_key` / `registry_keys` 引用参数名。

| 文件 | 作用 | 谁读 |
|---|---|---|
| `parameter_registry.yaml` | 参数登记表：样本窗口、winsor、NW 滞后与敏感性网格、pricing 门禁阈值、headline/shadow 样本指针、主控制集、MAX 日样本口径、预注册推断参数 | `pricing_lib` 导入即校验；`pricing_*`、`shadow_*`、`peermax_panel_build`、`analyst_gate1_consistency_check` 等 |
| `version_manifest.yaml` | 版本清单：网络版本白名单（10 个目录） | `pricing_lib.load_manifest_entries`、`peermax_panel_build.load_versions` |
| `analyst_spec.yaml` | 分析师网络规格：输入路径、网络版本与规则、样本偏差登记、governance 版本串 | `analyst_*` 全套 |
| `pricing_inputs.yaml` | 定价层九个主输入的合同：路径 / `rows` / `columns` | `pricing_lib.check_input_contract`、`shadow_monthly_status` 等 Phase2 脚本 |
| `claim_registry.yaml` | claim 与七类连接登记（pricing_headline_claim、pricing_portfolio_claim） | `check_link_audit.py`、`tests/test_link_audit_semantics.py` |
| `scope_card.yaml` | 最小口径卡（treatment/outcome/窗口/上限） | `check_link_audit.py` |

## 改口径的流程

改登记值、headline 规则或样本定义：

1. 在 `parameter_registry.yaml` 的 `change_log` 追加一条（id / date / scope / rationale / params / version 前后）；
2. 升 `registry_version`，同步 `analyst_spec.yaml` 的 `governance.registry_version`（`pricing_lib` 导入即核对，不一致退出 2）；
3. 跑 `python -m peerlab spec`、`make lint` 与相称的 pytest。

派生参数不得手填：`K_Consist_Pricing` 必须等于 `K_Econ_Pricing / iqr_pool_value`，`pricing_lib` 导入即断言。样本指针只有两个：`sample_and_spec_registry.sample_version`（headline）与 `shadow_legacy_sample_version`（shadow 全套，重跑后删除）。

## 输入合同的登记步骤

`data/processed/` 产出一份主输入后：

1. 数出行数，写进 `pricing_inputs.yaml` 对应条目的 `rows`；
2. 需要锁列名的，把列名列表写进 `columns`；
3. 跑 `python -m peerlab smoke --imports` 与 `python -m pytest tests -q`。

`rows` 为 null 时 `check_input_contract` 直接 fail-closed（不会退化成"文件存在"）；`columns` 为 null 表示只锁行数。
