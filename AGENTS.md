# TE-RAG：当前工作入口（2026-11-24 整理后）

先读 [PROJECT_GUIDE.md](PROJECT_GUIDE.md)。该文件是当前论文、代码、实验与复现路径的总览；按任务再读其中指向的源码或报告。

- 当前论文是 `TE-RAG_DASFAA/writing_round5_20261123/main.tex` / `main.pdf`，含附录版同目录。旧 merged/main/original_acm 与其他写作轮次已归档，不作为当前入口。
- 核心定位：历史 Query–SQL 驱动的领域知识获取、约束扩展与检索利用。研究任务是表/字段 Schema Grounding；SQL 生成和执行不是当前论文实验。
- 当前实验入口是 `run_experiment.py`；接受参数在 `configs/enterprise.json`、`configs/bird.json`。共享源码为 `terag/`、`retrievers/`、`datasets/`、`eval/`；专用评分和接线为 `experiments/enterprise/`、`experiments/bird/`。BIRD 必须经 `unified_entry.run_one` 执行门控后评价。
- 历史版本、旧结果、中间报告和旧 release 统一在 `_archive/20261124_cleanup/`。正常开发与运行不依赖归档，不恢复旧版本开发。`data/`、`source_dataset/`、`.runtime/` 是当前运行依赖，不能当历史缓存清理。
- 后续直接修改当前源码和配置，不校验旧封存 HEAD/代码 SHA。改变 B/C 生成、Schema 或输入时需同步重建当前接受资产、更新输入清单并记录新实验身份，不能将缓存旧结果冒充新计算。
- 新基线必须准确命名为 `LinkAlign-SingleDB-Adapted`、`KaSLA-Adapted`；`DenseDeepSeek` 是独立的自定义基线，不能冒充 LinkAlign。
- 先检查 `git status`，保留用户工作树改动。整理前的原文件和改动按原字节保存在归档对应路径。`config.yaml` 原地保留：允许按已授权任务运行时读取，禁止打印、复制进报告/日志或提交其凭据。不要自动暂存、commit、push 或重写历史。
- 正式指标以当前核验 CSV 和来源预测为准。88 是运行记录数，76 是有效评价分母，65 是预算可行补充；FGCS 已有大小写后处理勘误，不能直接抄旧均值。
- 不把完整方法 +9.2pp 归因于单个注释模块，不声称迁移改善未见表，不把适配版结果称为原模型完整复现。
- `SESSION_HANDOFF.md` 按时间追加，整理前的完整历史交接已保存在归档。旧 README 示例和历史待办不构成当前执行指令；不要读到旧“下一步”就重复开发、调用 API 或重跑全矩阵。
- 根据当前用户任务继续工作；已有授权仍有效。涉及论文、算法或统计的实质变更时，同步维护总览对应部分及来源记录，避免再产生多份互相冲突的“最新状态”。
