"""
码值映射器

从历史 SQL 中挖掘 phrase → code 的映射，用于扩展查询

使用方式:
    mapper = CodeMapper(config)
    phrase_mapping = mapper.mine_from_sql(qa_data)
    expanded_query = mapper.expand_query(query)
"""

import os
import re
import json
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from terag.config import TERAGConfig


@dataclass
class PhraseMapping:
    """短语映射"""
    phrase: str
    field: str
    code_value: str
    source_sql: str
    frequency: int = 1


class CodeMapper:
    """
    知值映射器

    从历史 SQL 的 WHERE 条件挖掘 (field=value) 映射，    用于扩展查询（命中 phrase 就追加 code token）
    """

    def __init__(self, config: TERAGConfig):
        """
        初始化码值映射器

        Args:
            config: TE-RAG 配置
        """
        self.config = config
        self.phrase_mappings: Dict[str, PhraseMapping] = {}
        self.field_df = pd.read_csv(config.data.field_csv)

    def mine_from_sql(self, qa_data: pd.DataFrame) -> Dict[str, List[PhraseMapping]]:
        """
        从 SQL 挖掘码值映射

        Args:
            qa_data: QA 数据，包含 question, sql 列

        Returns:
            {field: [PhraseMapping, ...]}
        """
        mappings = defaultdict(list)

        for _, row in qa_data.iterrows():
            sql = row.get('sql', '')
            if pd.isna(sql) or not sql:
                continue

            # 解析 WHERE 条件
            conditions = self._extract_where_conditions(str(sql))

            for condition in conditions:
                # 提取 field=value 对
                field_value_pairs = self._extract_field_value_pairs(condition)

                for field, value in field_value_pairs:
                    # 创建映射
                    mapping = PhraseMapping(
                        phrase=self._infer_phrase(field, value, row['question']),
                        field=field,
                        code_value=value,
                        source_sql=str(sql),
                        frequency=1
                    )
                    mappings[field].append(mapping)

        # 合并相同短语和值的映射，累加频率
        merged = defaultdict(list)
        for field, mapping_list in mappings.items():
            seen = {}
            for m in mapping_list:
                key = (m.phrase, m.code_value)
                if key in seen:
                    seen[key].frequency += 1
                else:
                    seen[key] = m
            merged[field] = list(seen.values())

        # 存储到 self.phrase_mappings
        self.phrase_mappings = dict(merged)

        return dict(merged)

    def _extract_where_conditions(self, sql: str) -> List[str]:
        """提取 WHERE 条件"""
        conditions = []

        # 简单提取 WHERE 子句
        where_match = re.search(
            r'\bWHERE\s+(.*?)(?:\bGROUP BY\b|\bORDER BY\b|\bHAVING\b|\bLIMIT\b|$)',
            sql,
            re.IGNORECASE | re.DOTALL
        )

        if where_match:
            where_clause = where_match.group(1)
            # 分割 AND/OR 条件
            cond_parts = re.split(r'\bAND\b|\bOR\b', where_clause, flags=re.IGNORECASE)
            conditions.extend([c.strip() for c in cond_parts if c.strip()])

        return conditions

    def _extract_field_value_pairs(self, condition: str) -> List[Tuple[str, str]]:
        """提取条件中的字段=值对"""
        pairs = []

        # 移除括号
        condition = condition.strip('()')

        # 匹配 field = 'value' 或 field != 'value'
        # 支持多种引号
        patterns = [
            r"(\w+)\s*=\s*'([^']*)'",   # field = 'value'
            r"(\w+)\s*=\s*\"([^\"]*)\"",  # field = "value"
            r"(\w+)\s*=\s*(\d+)",       # field = number
            r"(\w+)\s*LIKE\s+'%'",      # field LIKE '%'
        ]

        for pattern in patterns:
            match = re.search(pattern, condition, re.IGNORECASE)
            if match:
                field = match.group(1)
                value = match.group(2)
                pairs.append((field, value))

        return pairs

    def _infer_phrase(self, field: str, value: str, question: str) -> str:
        """推断短语（简化版）"""
        # 简单处理：移除引号，转小写
        value = value.strip("'\"").lower()
        field = field.lower()

        # 尝试从问题中提取短语
        # 如果值包含中文，直接使用值作为短语
        if re.search(r'[\u4e00-\u9fa5]+', value):
            return value
        else:
            # 否则使用字段名作为短语
            return field

        return f"{field}_{value}"

    def build_phrase_mapping(self, qa_data: pd.DataFrame) -> Dict[str, List[PhraseMapping]]:
        """
        构建短语映射词典

        Args:
            qa_data: QA 数据

        Returns:
            {field: [PhraseMapping, ...]}
        """
        all_mappings = defaultdict(list)

        # 按字段分组
        for field, mappings in self.mine_from_sql(qa_data).items():
            all_mappings[field].extend(mappings)

        # 按频率排序
        for field in all_mappings:
            all_mappings[field].sort(key=lambda m: -m.frequency, reverse=True)

        return dict(all_mappings)

    def expand_query(self, query: str) -> str:
        """
        扩展查询（追加 code token)

        Args:
            query: 原始查询

        Returns:
            扩展后的查询
        """
        expanded_tokens = []

        # 检查每个字段映射
        for field, mappings in self.phrase_mappings.items():
            for mapping in mappings:
                # 如果查询中包含短语，追加 code token
                if mapping.phrase in query:
                    code_token = f"CODE:{mapping.field}='{mapping.code_value}'"
                    expanded_tokens.append(code_token)

        if expanded_tokens:
            return f"{query} {' '.join(expanded_tokens)}"
        else:
            return query

    def save(self, output_path: str):
        """
        保存映射到文件

        Args:
            output_path: 输出路径
        """
        data = {}
        for field, mappings in self.phrase_mappings.items():
            data[field] = [
                {
                    'phrase': m.phrase,
                    'field': m.field,
                    'code_value': m.code_value,
                    'source_sql': m.source_sql,
                    'frequency': m.frequency,
                }
                for m in mappings
            ]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"码值映射已保存到: {output_path}")

    def load(self, input_path: str):
        """
        从文件加载映射

        Args:
            input_path: 输入路径

        Returns:
            加载的映射数量
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        count = 0
        for field, mappings in data.items():
            for m in mappings:
                mapping = PhraseMapping(
                    phrase=m['phrase'],
                    field=m['field'],
                    code_value=m['code_value'],
                    source_sql=m['source_sql'],
                    frequency=m['frequency']
                )
                self.phrase_mappings[field].append(mapping)
                count += 1

        print(f"从 {input_path} 加载了 {count} 个码值映射")
        return count


class QueryProcessor:
    """
    查询处理器

    整合码值映射、用于扩展查询
    """

    def __init__(self, config: TERAGConfig, code_mapper: CodeMapper = None):
        self.config = config
        self.code_mapper = code_mapper or CodeMapper(config)
        self._token_cache = {}

    def process(self, query: str) -> str:
        """
        处理查询

        Args:
            query: 原始查询

        Returns:
            处理后的查询（可能包含 code token)
        """
        if query in self._token_cache:
            return self._token_cache[query]

        # 1. 扩展查询（追加 code token)
        expanded_query = self.code_mapper.expand_query(query)

        # 2. 分词
        import jieba
        tokens = list(jieba.cut(expanded_query))

        self._token_cache[query] = tokens

        return tokens

    def train_from_qa_data(self, qa_path: str, output_path: str = None):
        """
        从 QA 数据训练码值映射

        Args:
            qa_path: QA 数据路径
            output_path: 映射输出路径
        """
        # 加载 QA 数据
        qa_data = pd.read_csv(qa_path)

        # 挖掘映射
        mappings = self.code_mapper.mine_from_sql(qa_data)

        # 保存
        self.code_mapper.save(output_path)

        return len(mappings)


def main():
    """演示码值映射"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.config import TERAGConfig

    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("码值映射演示")
    print("=" * 60)

    # 讣练
    mapper = CodeMapper(config)
    train_path = config.data.qa_csv

    mappings_path = config.get_artifact_path('code_mappings.json')

    if not os.path.exists(mappings_path):
        print(f"\n从 {train_path} 训练码值映射...")
        n_mappings = mapper.mine_from_sql(pd.read_csv(train_path))
        mapper.save(mappings_path)
        print(f"提取了 {sum(len(v) for v in mappings.values())} 个字段的映射")
    else:
        print(f"\n加载已有映射: {mappings_path}")
        mapper.load(mappings_path)

    # 测试扩展
    print("\n测试查询扩展:")
    test_queries = [
        "查询杭州供电公司的售电量",
        "统计浙江省所有用户的电费总额",
        "分析供电所A的回收率排名",
    ]

    processor = QueryProcessor(config, mapper)

    for query in test_queries:
        tokens = processor.process(query)
        print(f"\n查询: {query}")
        print(f"Tokens: {tokens}")


if __name__ == "__main__":
    main()
