# TE-RAG 实验框架

**表格增强检索增强生成（Table-Enhanced Retrieval-Augmented Generation）**

用于自然语言查询转换为SQL的表格检索和字段选择任务。

---

## 一键运行指南

### 第一步：安装依赖

```bash
# 进入项目目录
cd /Users/apple/Documents/浙大工作/论文/分层查询数据表/code

# 安装依赖包
pip install pandas numpy networkx jieba matplotlib psutil
```

### 第二步：运行实验

```bash
# 运行所有实验（对比实验 + 冷启动实验 + 消融实验 + 可视化）
python experiment/run_experiments.py
```

### 第三步：查看结果

```bash
# 查看实验结果目录
ls results/

# 查看对比实验结果
cat results/comparison_results.csv

# 查看消融实验结果
cat results/ablation_results.csv

# 查看冷启动实验结果
cat results/cold_start_results.csv

# 查看总结报告
cat results/experiment_summary.md
```

---

## 命令详解

### 命令1：运行全部实验

```bash
python experiment/run_experiments.py
```

**执行内容：**
1. **对比实验** - 比较 BM25、Vector、LLM、TE-RAG 四种方法的性能
2. **冷启动实验** - 测试在新表上的检索效果
3. **消融实验** - 分析TE-RAG各组件的贡献
4. **可视化** - 生成图表和报告

**输出文件：**
| 文件名 | 内容 |
|-------|------|
| `comparison_results.csv` | 对比实验结果 |
| `cold_start_results.csv` | 冷启动实验结果 |
| `ablation_results.csv` | 消融实验结果 |
| `comparison_accuracy.png` | 准确性对比图 |
| `cold_start_comparison.png` | 冷启动对比图 |
| `ablation_results.png` | 消融实验图 |
| `experiment_summary.md` | 总结报告 |
| `comparison_table.tex` | LaTeX表格 |

### 命令2：只运行对比实验

```bash
python experiment/comparison.py
```

### 命令3：只运行消融实验

```bash
python experiment/ablation.py
```

### 命令4：只生成可视化图表

```bash
python experiment/visualize.py
```

### 命令5：预处理数据（可选）

如果需要重新处理原始数据：

```bash
cd source_dataset
python process_dataset_new.py
```

**输入：** `dataset.py` 中的 DDL 和 QA 数据
**输出：**
- `processed_field_schema.csv` - 字段结构
- `processed_table_schema.csv` - 表结构
- `processed_qa_data.csv` - QA数据

---

## 项目结构

```
code/
├── experiment/                    # 实验模块
│   ├── run_experiments.py        # ⭐ 主入口：一键运行所有实验
│   ├── comparison.py             # 对比实验
│   ├── ablation.py               # 消融实验
│   └── visualize.py              # 可视化工具
│
├── retrievers/                    # 检索器模块
│   ├── __init__.py               # 模块入口
│   ├── base_retriever.py         # 基类
│   ├── bm25_retriever.py         # BM25检索器
│   ├── vector_retriever.py       # 向量检索器
│   ├── llm_retriever.py          # LLM检索器
│   ├── terag_retriever.py        # TE-RAG检索器（核心）
│   └── retriever_factory.py      # 工厂类
│
├── source_dataset/               # 数据集
│   ├── processed_field_schema.csv   # 字段结构
│   ├── processed_table_schema.csv   # 表结构
│   ├── processed_qa_data.csv        # QA数据
│   └── process_dataset_new.py       # 数据预处理脚本
│
├── results/                      # 实验结果（自动生成）
│   ├── *.csv                     # 结果数据
│   ├── *.png                     # 图表
│   └── *.md                      # 报告
│
├── base/                         # 旧版代码（兼容保留）
├── client/                       # LLM客户端
├── utils/                        # 工具函数
├── main.py                       # 旧入口
└── requirements.txt              # 依赖列表
```

---

## 实验说明

### 1. 对比实验

比较四种检索方法的性能：

| 方法 | 说明 |
|-----|------|
| **BM25** | 基于关键词的传统检索 |
| **Vector** | 基于TF-IDF向量的检索 |
| **LLM** | 基于大语言模型的检索 |
| **TE-RAG** | 本方法：表格增强RAG |

**评估指标：**
- Table Accuracy：表格选择准确率
- SQL Accuracy：SQL生成准确率（表+字段都正确）
- Query Time：查询时间
- Memory：内存占用

### 2. 消融实验

分析TE-RAG各组件的贡献：

| 配置 | 说明 |
|-----|------|
| Full TE-RAG | 完整方法 |
| w/o Graph Weight | 去掉二分图加权 |
| w/o Template Mining | 去掉模板挖掘 |
| w/o Pattern Generalization | 去掉模式泛化 |
| w/o Business Rules | 去掉业务规则 |
| w/o Enhanced Index | 去掉增强索引 |

### 3. 冷启动实验

测试系统对新表的适应能力：
- 训练时排除某些表
- 测试时只使用这些新表

---

## 数据格式

### QA数据 (processed_qa_data.csv)

| 列名 | 说明 | 示例 |
|-----|------|------|
| question | 自然语言问题 | "2025年2月杭州公司的售电量是多少？" |
| table | 目标表名 | "db.table_name" |
| field | 目标字段 | "field1\|field2" |
| sql | 参考SQL | "SELECT field1 FROM table..." |

### 表结构 (processed_table_schema.csv)

| 列名 | 说明 |
|-----|------|
| table | 表名 |
| db | 数据库名 |
| table_desc | 表描述 |
| field | 所有字段（\|分隔） |
| schema | DDL语句 |

### 字段结构 (processed_field_schema.csv)

| 列名 | 说明 |
|-----|------|
| table | 所属表 |
| db | 数据库 |
| field_name | 字段名 |
| field_name_desc | 字段描述 |
| field_type | 字段类型 |

---

## 常见问题

### Q1: 如何修改训练/测试比例？

编辑 `experiment/run_experiments.py`，修改：
```python
n_train = int(len(indices) * 0.8)  # 改为80%训练，20%测试
```

### Q2: 如何只测试特定方法？

编辑 `experiment/run_experiments.py`，修改：
```python
methods = ['BM25', 'TE-RAG']  # 只测试这两个
```

### Q3: 如何添加新的检索方法？

1. 在 `retrievers/` 创建新文件 `my_retriever.py`
2. 继承 `BaseRetriever` 类
3. 实现 `fit()` 和 `_retrieve()` 方法
4. 在 `retrievers/__init__.py` 中导入

详见 `retrievers/README.md`

### Q4: 运行报错怎么办？

1. 检查依赖是否安装完整
2. 检查数据文件是否存在
3. 检查Python版本（建议3.8+）

```bash
# 重新安装依赖
pip install -r requirements.txt
```

---

## 快速测试代码

```python
# 测试单个检索器
from retrievers import RetrieverFactory
import pandas as pd

# 数据路径
field_csv = 'source_dataset/processed_field_schema.csv'
table_csv = 'source_dataset/processed_table_schema.csv'
qa_csv = 'source_dataset/processed_qa_data.csv'

# 创建TE-RAG检索器
retriever = RetrieverFactory.create('TE-RAG', field_csv, table_csv)

# 读取数据并训练
qa_df = pd.read_csv(qa_csv)
train_data = qa_df.iloc[:100]  # 使用前100条训练
retriever.fit(train_data)

# 测试检索
query = "2025年2月杭州公司的售电量是多少？"
results = retriever.retrieve(query, k=5)

# 打印结果
for r in results:
    print(f"表: {r.table}, 得分: {r.table_score:.3f}")
    for col, score in r.columns[:3]:
        print(f"  - {col}: {score:.3f}")

# 评估
test_data = qa_df.iloc[100:150]  # 使用50条测试
metrics = retriever.evaluate(test_data, k=5)
print(f"Table Accuracy: {metrics.table_accuracy:.1%}")
print(f"SQL Accuracy: {metrics.sql_accuracy:.1%}")
```

---

## 依赖版本

```
pandas>=1.0.0
numpy>=1.18.0
networkx>=2.5
jieba>=0.42
matplotlib>=3.0.0
psutil>=5.0.0
```

---

## 联系方式

如有问题，请提交Issue或联系项目维护者。

---

## 许可证

MIT License
