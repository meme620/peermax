# 运行环境

## 解释器

| 项 | 值 |
|---|---|
| 版本 | **Python 3.13**（`.python-version`） |
| 实际验证 | 3.13.12（uv 下载的解释器） |
| 原始环境 | 主仓库为系统 Python 3.13（`docs/pipeline.md` §6） |

选用 3.13 是为了与主仓库一致。本机另有 3.12 与 3.14，仅作后备：若某个二进制轮子在 3.13 缺失，退 3.12 并在本节记下偏差。当前 14 个直接依赖在 3.13 全部有可用轮子，未退版。

## 依赖

`requirements.txt` 单文件，直接依赖与传递依赖都写死版本；解析与验证日期 2026-11-18。

| 包 | 版本 | 用途 | 真实 import 的文件数 |
|---|---|---|---|
| pandas | 2.3.3 | 全链数据操作 | 146 |
| numpy | 2.5.5 | 数值运算 | 110 |
| scipy | 1.18.1 | 统计分布与检验 | 10 |
| scikit-learn | 1.9.1 | TF-IDF 向量化（M-B 侧） | 14 |
| PyYAML | 6.0.3 | registry / spec / manifest | 40 |
| pyarrow | 21.0.0 | parquet 样本读写 | 4 |
| pytest | 8.4.2 | 契约测试 | 29 |
| jieba | 0.42.1 | M-B 分词 | 6 |
| pyahocorasick | 2.3.1 | M-A 词库多模式匹配 | 2 |
| pymupdf | 1.28.2 | 年报/招股书 PDF 文本层 | 4 |
| pdfminer-six | 20260107 | 招股书锚 CID→GID 通道 | 1 |
| fonttools | 4.68.0 | 同上，字形映射 | 1 |
| python-pptx | 1.0.2 | 只给 `make_analyst_network_brief_pptx.py` | 1 |
| baostock | 0.9.4 | 只给 `fetch_baostock_*` | 2 |

**pandas 锁 2.x**：全树没有 `applymap`/`swapaxes`/`.ix`/`DataFrame.append` 这类已删除 API（逐项检索为 0），但 `groupby(...).apply(...)` 有 25 处。pandas 3.0 把 `include_groups` 的默认值改为 False，分组列不再出现在传入的帧里，这 25 处可能静默改变数字。研究数字出自 pandas 2.x，因此锁 2.x。

## 安装

```bash
make env        # uv venv --python 3.13 .venv + uv pip install -r requirements.txt
```

`uv` 创建的 venv 不带 `pip`，装包与查看都用 uv：`uv pip list --python .venv/bin/python`、`uv pip freeze --python .venv/bin/python`。需要网络（PyPI）。

## 检查与当前基线

| 命令 | 做什么 | 2026-11-18 结果 |
|---|---|---|
| `make preflight` | 解释器 / 14 个依赖 / 6 个 configs / 数据目录，三块一次打印 | env 通过；configs 6 个缺失；raw 目录 5/14 就绪 |
| `make imports` | 逐个导入 `scripts/` 下 203 个脚本 | 192 ok、0 config、11 missing、**0 dep、0 other** |
| `make smoke` | peerlab L0：10 个实验逐个判脚本与输入是否就绪 | 10 个全部 blocked（缺 configs），退出码 3 |
| `make status` | 逐实验列出缺什么 | 0/10 就绪 |
| `make lint` | 命名与实验文档规范检查 | 通过 |
| `make selftest` | `pytest tests/test_peerlab_*.py tests/test_script_utils.py` | 34 passed |
| `make test` | 全量 `pytest`（`tests/`） | 配置已就位，收集不再中断；当前 309 passed / 123 failed / 149 errors，全部因 `data/processed/` 与 `reports/` 为空 |

`make imports`（即 `python -m peerlab smoke --imports --clean`）会执行每个脚本的模块级代码，部分脚本会在 `reports/data_gates/G0/`、`data/processed/` 下写出文件（如 `drive_batch1_schema.json`）。命令跑完会列出这些文件并删除；不加 `--clean` 时只列出，便于先看脚本写了什么。

`make imports` 的判读：`config` 是被 fail-closed 的 registry 拦下、`missing` 是缺数据/报告目录，两者属预期；`dep`（`ModuleNotFoundError`）与 `other` 才算环境问题。

种子协议、实验注册表与结果聚合规则见 `docs/result_pipeline.md`：全局 `SEEDS = (0,1,2,3,4)`，每次运行注入 `PYTHONHASHSEED` 与 random/numpy 种子，脚本自带 `--seed` 的直传。

`make preflight` 的退出码：env 或 configs 不通过返回 1；data 级只报告不判失败，因为数据完备性取决于仓库外的决定。

## 等级与现状

| 级 | 要求 | 现状 |
|---|---|---|
| T1 环境 | venv 可重建、依赖齐、脚本全部能导入 | 达成 |
| T2 数据 | 教师数据在代码读取的路径 | 达成（42 个文件，5 个 raw 目录 + external） |
| T3 配置 | `configs/` 六个 YAML 到位 | 达成（2026-11-18 新基线重建，来源与未锚项见 `configs/README.md`） |
| T4 运行 | 跑通管线并产出 `data/processed/` | **未达成**：依赖 T3 |

## 平台边界

- `scripts/network_prospectus_anchor.py` 的 OCR 回退经 `wsl.exe -u 27697 bash` 调用 `ocr_tesseract_chunk.sh`，WSL 用户 ID 写死在代码里（`:718`）。macOS 上这条通道不可用，只影响招股书锚审计；该脚本的 pdfminer + CID→GID 通道不受影响。
- 抓数脚本用标准库 `urllib`，没有 `requests` 依赖。
- 全部脚本要求 **cwd = 仓库根**：`scripts/<文件>.py` 与 `tests/*.py` 都用 `parents[1]` 定位根。
