# TE-RAG 当前工作入口

**当前分支（2026-11-25 用户更正后）：仅使用 `main`，跟踪 `origin/main`。核心源码已同步为 `509e29a`；本地和远端 `master` 均已删除。下方首次推送 master 的记录仅为历史。**

2026-11-24：按用户要求集中归档历史版本，保留当前版本供后续直接迭代。

先读 [README.md](README.md) 的运行方法。当前唯一实验 CLI 为 [run_experiment.py](run_experiment.py)，不再通过历史 release 或 `research_runs` 回放来启动实验。

当前论文是 [第五轮主文](TE-RAG_DASFAA/writing_round5_20261123/main.tex) / [PDF](TE-RAG_DASFAA/writing_round5_20261123/main.pdf)，含附录版本也在同目录。旧 merged/main/original_acm 与第一至四轮全部进入 `_archive/20261124_cleanup/original_tree/TE-RAG_DASFAA/`，正文和数值未改。

## 当前方法与代码

- 研究任务是表/字段 Schema Grounding；SQL 用于历史监督和评价，不生成或执行 SQL。
- 电网当前完整实现是 `experiments/enterprise/model.py::Engine`；共用构建与索引操作在 `engine_base.py`，训练折数据/接受 C 在 `runtime.py`。当前接受参数为 `configs/enterprise.json`，含两协议五折、九变体的 90 份选择。固定消融和独立优化消融分别运行，不从外层评价重新选参。
- BIRD 当前实现是 `experiments/bird/engine/`，必须通过 `unified_entry.run_one` 执行 hg 和 field_rerank，再从最终输出计算指标。`configs/bird.json` 保留当前协议专用 Full、固定消融和独立优化消融的原参数。
- 共用的 `terag/`、`retrievers/`、`datasets/`、`eval/` 来自最终接受电网实现；已核对修复后的 BIRD 共享源码相同。两套实验现在直接导入这一份源码。
- `terag/terag_retriever_v2.py` 提供底层状态构建；其默认参数不是当前优化 Full。今后改评分逻辑需同时检查 dataset 专用 Engine，而不能只改默认权重。
- 基线必须准确命名 `LinkAlign-SingleDB-Adapted`、`KaSLA-Adapted`；`DenseDeepSeek` 为自定义基线。基线接线在 `experiments/baselines.py`，BIRD 专用实现保留在 `experiments/bird/baselines/` 与 `api/`。

## 数据、来源与结论边界

- `source_dataset/` 保持原路径和字节；当前 Gold/划分不改。电网 88 条运行记录、76 有效主分母、65 预算可行补充；FGCS 按规范化物理字段匹配。
- 电网必要接受 B/C、元数据索引和训练协议集中在 `data/enterprise/`。BIRD 五折清单、训练行、11 库 Schema 和 Gold 在 `data/bird/`。均不依赖归档目录。
- `data/reference/` 保存压缩的现版本逐题输出供排序一致性核对。完整原始分数、统计及每阶段报告在归档，论文已有表格保留在最新写作目录。
- 完整方法的整体收益不能归因于单个模块；不声称迁移改善未见表，不将适配版称为原模型完整复现。BIRD 已有 BIndex 符号、H 门控逐题效应及 Q 门控召回取舍勘误，论文第五轮沿用的是补正值。
- 当前电网调优属于探索性嵌套验证，不因整理目录变为独立最终泛化测试。未来变更需记录实际源码/参数、输入身份和新输出位置，不能冒充已冻结旧结果。

## 本次整理与后续约定

归档位置为 `_archive/20261124_cleanup/`。`MOVES.json` 记录原路径和新路径；原 PROJECT_GUIDE、SESSION_HANDOFF、README 与历史报告都保留在 `original_tree/`。`.runtime/site-packages/` 从旧运行目录独立移出，供当前配置 SDK 使用。

当前入口不再校验旧 Git HEAD、旧代码封存 SHA 或旧工作树状态；当前参数/源码可直接迭代。`--check` 核对当前输入及依赖。旧归档脚本中的待办和授权不是新任务，不执行它们。

先检查 `git status`，保留用户修改。不自动 commit、push 或重跑完整矩阵。根 `config.yaml` 原地保留；允许授权运行时读取，禁止打印、复制进报告或提交凭据。

此次验证：43 项回归；483 条本地抽样运行中，465 条与保存预测核对零差异，其余 18 条为电网六个离线基线的运行检查。覆盖电网固定/独立两轨、BIRD 当前 Q/独立 Q/固定 H、core 消融和常用离线基线。归档目录临时不可用时，两数据集输入检查及 BIRD 推理仍通过。范围和输出见归档内 `VALIDATION.json`；这些不是新正式实验结果，未调用 API、SQL 执行或新参数搜索。

## 2026-11-25：核心源码提交与推送

用户明确授权提交并推送核心代码到 master。远端原来只有 main；现已创建独立源码历史的 master，提交 `9a6234f97d0ea6f71711c4108d1827a385103c8c`，并推送到 `origin/master`。98 个文件约 1.1 MB，包含核心源码、当前参数、测试、README 和 PRIVATE_INPUTS；不包含论文工作稿、归档、私有数据/逐题输出、本机环境或凭据文件。

原本地主线 main 的 49 个未推送历史提交仍保留，未上传；远端 main 和 GitHub 默认分支不改。本地当前分支已切换至 master 并跟踪 origin/master，所有本地论文、数据、凭据和归档保留原文件。以后正常在当前 master 源码上迭代，勿推送旧 main 历史。

43 项组件回归在不含私有输入的隔离源码目录中通过，暂存内容与当前文件逐字节一致；Python/JSON 语法和凭据检查通过。新 clone 的安装与授权输入边界见 README 和 PRIVATE_INPUTS.md；不称为无数据准备即可复现论文数值。当前本地 PROJECT_GUIDE、SESSION_HANDOFF、AGENTS/CLAUDE 仍留在工作目录，不在本次源码提交内。

## 2026-11-25：按用户更正统一至 main

用户明确要求只保留 main，将 master 与本地当前核心内容同步至 main，再删除 master。以远端原 main 和 master 为父提交创建合并提交 `509e29a346e55c5887455c2cd1d68faa2aa4e818`，保留远端已有历史；当前源码/参数/测试与原 master 一致，仅三个文档中的分支名改为 main。正常快进推送并核验成功后，删除了远端及本地 master。

本地 main 现跟踪 origin/main，旧 main 的未推送历史继续由既有冻结标签 `terag-final-ablation-v1` 保留，没有上传。论文、凭据、私有数据、环境和归档未改动、未上传。后续仅在 main 迭代。此次仅分支整理和文档名称修正，未改算法，也未重跑实验或调用 API。

## 2026-11-25：项目说明纳入版本管理

此前将 AGENTS.md、CLAUDE.md、PROJECT_GUIDE.md、SESSION_HANDOFF.md 一并忽略，导致这些说明只存在于本地，未随源码同步。现取消四个文件的忽略规则，并纳入 main。此前“说明未提交”的条目保留为当时记录，不代表当前状态。

本地与远端一致指 Git 已跟踪的源码和说明一致，不表示完整本地目录全部上传。论文链接指向本地工作稿；论文、凭据、私有数据、接受资产、逐题输出、环境和归档仍不进入 Git。
