# 当前工作交接

此前完整的时间顺序记录已原样移动到 `_archive/20261124_cleanup/original_tree/SESSION_HANDOFF.md`。

## 2026-11-24：当前实验代码整理

用户要求只保留能运行当前版本的代码，将中间报告、旧版本和历史实验统一移到一个目录，未来直接在当前版本迭代。

当前入口改为 `run_experiment.py`；共享源码集中在根 `terag/`、`retrievers/`、`datasets/`、`eval/`，数据集专用实现为 `experiments/enterprise/`、`experiments/bird/`。当前接受参数为 `configs/enterprise.json`、`configs/bird.json`。必要数据、固定划分、接受 C 和压缩参考预测集中到 `data/`；原 `source_dataset/`、凭据 `config.yaml` 原地保留。工作目录不再依赖历史 release、research_runs 或 archived 源码。

旧源代码、旧结果、失败实验、报告、release、审计记录和旧论文版本全部移动到 `_archive/20261124_cleanup/original_tree/`；旧数值依赖移至 `.runtime/site-packages/`。归档路径映射见该目录 MOVES.json。原用户未提交文件按原字节保存在对应归档位置；最新第五轮论文目录保持原路径及字节。

保留原协议、参数与预算；没有新选参、付费 API、SQL 执行或论文数值修改。43 项当前回归通过，483 条本地抽样运行中 465 条与保存预测核对零差异，其余 18 条为电网离线基线运行检查。覆盖电网固定/独立两轨、BIRD core/选中 Full/固定 H 与独立 Q 消融及六个离线基线。归档临时不可用时，两数据集输入检查及 BIRD 推理仍通过。没有完整重跑五折矩阵、重新计算显著性或验证 API 方法在线可用性。

源码和配置可直接迭代，不再校验历史封存的 HEAD/工作树/代码 SHA。若修改 B/C 生成、Schema 或输入数据，需同步重建当前接受资产、更新输入清单并标明新实验身份。不要将缓存的旧证据冒充新生成结果。

未执行 Git 暂存、提交、推送或历史重写。下一步按用户新的研究任务继续，入口和命令以根 README 与 PROJECT_GUIDE 为准。

## 2026-11-25：按明确授权推送核心代码到 master

用户要求检查核心代码和文件并提交、推送 master。远端仅有 main，本地 main 尚有 49 个未推送且曾跟踪凭据配置的历史提交。以当前核心文件建立独立源码历史的 master，避免上传旧历史；未改写或推送原 main。

提交为 `9a6234f97d0ea6f71711c4108d1827a385103c8c`，98 文件约 1.1 MB，推送到 `git@github.com:geenquan/TE-RAG.git` 的 master。源码、接受参数、测试、使用说明及 PRIVATE_INPUTS 保留；论文、私有数据、接受资产、逐题输出、环境和归档不上传。根 config 原地保留，检查过程只在内存比对凭据值，未打印或复制凭据。原 HEAD 和暂存工作树先保留，导入源码提交后仅切换 HEAD/索引至 master，工作文件未被覆盖。本地现在跟踪 origin/master，GitHub 默认分支仍为 main。

43 项测试在独立源码目录通过，未带入企业输入；待提交文件与当前源码一致，凭据/私有路径排除、Python/JSON 语法检查通过。README 增补新环境安装和授权输入边界，检查器为缺失授权输入给出明确说明，现版本测试兼容新 clone 的空 results 目录。没有新正式矩阵、API 或 SQL 执行。

## 2026-11-25：用户更正，仅保留 main

用户要求将 master 和本地当前内容同步至 main，并在同步成功后删除 master。已将远端原 main 与源码 master 合并为 `509e29a346e55c5887455c2cd1d68faa2aa4e818`，当前源码/参数/测试保持原 master 字节，仅 README、PRIVATE_INPUTS、gitignore 注释改用 main。正常快进推送至 origin/main，核验后删除了远端和本地 master。

当前唯一工作分支为 main，跟踪 origin/main。原本地 main 的 49 个未推送历史提交保留于已有冻结标签，未上传；不强制覆盖远端历史。所有本地论文、私有数据、凭据、环境、归档仍原地保留，不随核心源码上传。此次无算法变更、无新实验/API/SQL。首次 master 推送条目保留为历史，不再作为后续分支使用说明。

## 2026-11-25：修正项目说明被忽略的问题

用户询问线上线下是否一致，以及 AGENTS.md 显示黄色、没有提交的原因。核对时两端 main 均为 509e29a；四份项目说明被 .gitignore 排除，Git 状态为 ignored/untracked（!!），因此不在普通 status 和提交列表中。此前的同步仅覆盖已跟踪代码，并非整个本地目录。

按已有同步本地项目内容的授权，取消 AGENTS.md、CLAUDE.md、PROJECT_GUIDE.md、SESSION_HANDOFF.md 四项忽略并纳入 main。凭据、私有数据、论文、环境和归档继续忽略；不修改算法或实验结果，本次仅 Git 规则和项目说明变更。
