# TE-RAG

历史 Query–SQL 驱动的表/字段 Schema Grounding。当前代码直接在本目录迭代。

统一实验入口是 `run_experiment.py`，电网和 BIRD 共用根目录的 `terag/`、`retrievers/`、`datasets/`、`eval/`。数据集专用评分与运行接线在 `experiments/enterprise/`、`experiments/bird/`。当前参数在 `configs/enterprise.json`、`configs/bird.json`。

本地论文：`TE-RAG_DASFAA/writing_round5_20261123/main.tex` / `main.pdf`，含附录版本同目录。`main` 分支只分发当前核心源码、参数、测试和使用说明；论文工作稿、私有实验输入、逐题结果、归档和本机环境不随代码提交。

代码仓库不包含企业数据、训练折接受 B/C、索引、BIRD 派生输入或核对预测。新 clone 可以安装依赖并运行组件回归；精确复现论文数值需额外恢复授权输入，见 [PRIVATE_INPUTS.md](PRIVATE_INPUTS.md)。

## 新环境安装

使用 Python 3.11，在项目根目录执行：

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_experiment.py enterprise --list
```

恢复授权输入后才使用 `--check` 和正式运行命令。新环境使用正常 Python 启动，不加 `-S`；下方 `-S` 示例针对原本机保留的 `.runtime/` 数值环境。

## 运行

本机使用 IDE 配置的 Python 3.11.5。`.runtime/site-packages/` 保留已经验证的数值依赖，`-S` 避开 SDK 全局环境的 NumPy ABI 冲突。不需要旧实验目录。

```sh
PY=/Users/apple/miniconda3/envs/low-code-gpt/bin/python

# 只读检查输入身份和当前模块导入
"$PY" -S run_experiment.py enterprise --check
"$PY" -S run_experiment.py bird --check

# 查看各轨道的方法；enterprise 的 90 份接受参数直接保存在 configs 中
"$PY" -S run_experiment.py enterprise --list
"$PY" -S run_experiment.py bird --track independent --list

# 电网：原两协议五折，固定消融；独立优化消融改为 --track independent
PYTHONHASHSEED=0 "$PY" -S run_experiment.py enterprise --track fixed --output results/enterprise_fixed

# BIRD：原两协议五折，当前各协议选中的 Full
PYTHONHASHSEED=0 "$PY" -S run_experiment.py bird --output results/bird_selected

# BIRD 完整固定/独立优化消融：使用 --track fixed / --track independent
# 原 core 配置与其消融：使用 --track core；参数默认值不冒充优化版 Full

# 小规模运行并核对保存预测；--limit 的输出明确标为 smoke
PYTHONHASHSEED=0 "$PY" -S run_experiment.py bird --protocol question --fold 0 --db california_schools --limit 3 --verify --output results/bird_smoke

# 六个常用离线基线；两个数据集均支持相同的 --methods 名称
PYTHONHASHSEED=0 "$PY" -S run_experiment.py bird --protocol question --fold 0 --db california_schools --methods BM25 MetadataMatched ExampleRRF TFIDF Hybrid CooccurRRF --output results/bird_baselines

# 组件、物理字段匹配、门控执行和输出边界回归
PYTHONHASHSEED=0 PYTHONPATH="$PWD:$PWD/.runtime/site-packages" "$PY" -S -m unittest discover -s tests -v
```

输出目录必须是 `results/` 下尚不存在的子目录。输出包含压缩逐题预测和 `summary.json`，不会覆盖旧结果。`--fold`、`--protocol`、`--methods` 可以缩小执行范围；`--limit` 仅供检查运行。

Dense 需要本机已有 MiniLM 缓存及 torch/transformers。三个 API 基线名称为 `DenseDeepSeek`、`KaSLA-Adapted`、`LinkAlign-SingleDB-Adapted`，只在 question 协议显式请求，并增加 `--allow-api`。它们需要额外模型/API 依赖及运行时凭据；整理工作未调用 API。BIRD API 仍使用原预算 broker，默认上限 CNY 20，配置在 `experiments/bird/api/config.py`；模型/API 可用性不属于本次验证范围。

## 保留内容

| 路径 | 用途 |
|---|---|
| `run_experiment.py` | 当前实验唯一命令行入口 |
| `terag/`、`retrievers/`、`datasets/`、`eval/` | 共用核心和论文基线 |
| `experiments/` | 当前电网评分、BIRD 适配/门控、基线接线 |
| `configs/` | 当前实际接受参数，可直接编辑迭代 |
| `source_dataset/` | 原始企业数据和完整 Gold，原路径保留 |
| `data/` | 当前协议、BIRD Schema/训练行/Gold、必要 B/C 与索引、压缩核对预测 |
| `.runtime/` | 本机已经验证的数值依赖 |
| `results/` | 今后新实验输出 |
| `tests/` | 当前组件与入口回归 |
| `TE-RAG_DASFAA/writing_round5_20261123/` | 最新自包含论文源文件及 PDF |
| `_archive/20261124_cleanup/` | 移出的旧代码、旧实验、报告和论文版本 |

`data/` 是运行输入和核对资料，不能当作旧实验缓存删除。电网 C 仍消费接受的训练折资产，保留原来源和预算校验；不会因为目录整理重新生成或重新选参。源码与参数不绑定旧 release 哈希，未来直接修改当前文件即可。输入身份由 `data/INPUT_MANIFEST.json` 管理；有意改变数据时需同步更新该清单并记录实验变化。

`--verify` 对比表顺序和字段顺序：电网覆盖当前 Full/消融；BIRD 另覆盖保存的基线输出。它不表示重新核验全部历史统计。电网基线单独运行，不使用 `--verify`。

归档目录不参与当前实验导入或数据读取，可以整体移到项目外保存。归档内历史入口及其绝对路径不再维护。移动到同一硬盘的归档目录不会释放整个项目的磁盘空间。
