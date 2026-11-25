# 授权复现输入

`main` 只包含当前算法源码、实际接受参数、组件测试和使用说明，不包含私有数据、逐题预测或本机依赖环境。单独 clone 不是企业数值实验的完整复现包。

授权使用者需在项目根目录恢复以下层级，当前代码无需历史 `research_runs/` 或 `_archive/`：

| 路径 | 内容 |
|---|---|
| `source_dataset/processed_qa_data.csv` | 企业 Query–SQL 数据 |
| `source_dataset/processed_field_schema.csv` | 企业字段 Schema |
| `source_dataset/processed_table_schema.csv` | 企业表 Schema |
| `source_dataset/derived/full_gold_v2.jsonl` | 已核验完整 SQL Gold |
| `data/enterprise/protocol.json` | 原两协议五折与基础参数 |
| `data/enterprise/metadata_index.pkl`、`metadata_index_identity.json` | 当前接受的元数据索引及身份 |
| `data/enterprise/accepted/<setting>_fold<fold>/` | 每折 `B.json`、`B_texts.json`、`C.json` |
| `data/bird/` | 派生的 `q5_manifest.json`、`h5_manifest.json`、`gold.json`、`schema/`、`train/` |
| `data/INPUT_MANIFEST.json` | 授权输入文件的 SHA256 清单 |
| `data/reference/enterprise.jsonl.gz`、`bird.jsonl.gz` | 仅 `--verify` 排序核对需要的保存预测 |

不能用缺失的历史文件、空数据或其他划分近似替代上述输入；已有 B/C 资产只能作为保存证据消费，不能称为本轮重新生成。修改其生成代码、Schema 或数据时，应重建对应资产并记录新实验身份。

源码与选定参数不锁定历史 Git HEAD。今后的修改直接在当前代码上进行。

本机 `.runtime/` 不是可移植的 Python 环境，不随 Git 分发。在其他机器使用 Python 3.11 安装 `requirements.txt`。组件回归不需要私有输入；Dense/API 方法还需要各自额外依赖、模型缓存和运行时凭据。此仓库不承诺新机器无需准备数据即可重算论文数字。

`config.yaml` 原地保留于所有者机器并被 Git 忽略。API 凭据也可以通过运行时环境变量提供；不要将实际值写入配置示例、日志或提交。默认实验不会调用 API，付费基线须显式选择并设置 `--allow-api`。
