"""
模式挖掘器

挖掘查询模式（实体-属性、时间、聚合）并支持模式泛化
"""

import os
import json
import re
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from terag.config import TERAGConfig
from terag.tokenization import tokenize_text

PATTERN_IMPLEMENTATION = 'deterministic_patterns_v1'


@dataclass
class Pattern:
    """查询模式"""
    pattern_id: str
    pattern_type: str  # entity_attr, time, aggregation, etc.
    pattern_text: str
    count: int = 1
    source_element: str = ""
    transferred: bool = False

    def to_dict(self) -> dict:
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'pattern_text': self.pattern_text,
            'count': self.count,
            'source_element': self.source_element,
            'transferred': self.transferred,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Pattern':
        return cls(
            pattern_id=data['pattern_id'],
            pattern_type=data['pattern_type'],
            pattern_text=data['pattern_text'],
            count=data['count'],
            source_element=data['source_element'],
            transferred=data['transferred'],
        )


class PatternMiner:
    """
    模式挖掘器

    功能：
    1. 实体-属性模式挖掘
    2. 时间模式识别
    3. 聚合模式识别
    4. 模式泛化（基于相似度迁移）

    使用方式:
        miner = PatternMiner(config)
        patterns = miner.mine(element_to_queries)
        miner.save(patterns, "artifacts/patterns.jsonl")
    """

    def __init__(self, config: TERAGConfig):
        """
        初始化模式挖掘器

        Args:
            config: TE-RAG 配置
        """
        self.config = config

        # 挖掘模式: keywords（中文业务词表）/ generic（英文，从历史 Query-SQL 自动学习）
        self.mode = getattr(config.pattern_mining, 'mode', 'keywords')
        self.language = getattr(config.tokenizer, 'language', 'chinese')

        # 关键词配置（全部来自 config，不在代码中硬编码）
        self.entity_keywords = config.pattern_mining.entity_keywords
        self.attribute_keywords = config.pattern_mining.attribute_keywords
        self.similarity_threshold = config.pattern_mining.similarity_threshold
        self.min_pattern_count = config.pattern_mining.min_pattern_count

        # 聚合关键词 / 时间模式（按模式选择中英文版本）
        if self.mode == 'generic':
            self.aggregation_keywords = dict(config.pattern_mining.english_aggregation_keywords)
            self.time_patterns = [tuple(p) for p in config.pattern_mining.english_time_patterns
                                  if isinstance(p, (list, tuple)) and len(p) == 2]
        else:
            self.aggregation_keywords = dict(config.pattern_mining.aggregation_keywords)
            self.time_patterns = [tuple(p) for p in config.pattern_mining.time_patterns
                                  if isinstance(p, (list, tuple)) and len(p) == 2]

        # generic 模式：延迟加载 schema 词表（用于 query↔schema 签名 token 提取）
        self._schema_vocab: Optional[Set[str]] = None

    def mine(self, element_to_queries: Dict[str, Set[str]]) -> Dict[str, List[Pattern]]:
        """
        挖掘模式

        Args:
            element_to_queries: 元素到查询的映射 {element_node: set of queries}

        Returns:
            {element_node: [Pattern, ...]}
        """
        pattern_library = defaultdict(list)

        # 第一阶段：直接挖掘
        for element_node, queries in sorted(element_to_queries.items()):
            patterns = self._mine_patterns_for_element(sorted(queries))
            for pattern_key, pattern_data in sorted(patterns.items()):
                if pattern_data['count'] >= self.min_pattern_count:
                    pattern = Pattern(
                        pattern_id=f"{element_node}_{len(pattern_library[element_node])}",
                        pattern_type=pattern_data['type'],
                        pattern_text=pattern_key,
                        count=pattern_data['count'],
                        source_element=element_node,
                        transferred=False
                    )
                    pattern_library[element_node].append(pattern)

        # 第二阶段：模式泛化（如果启用）
        if self.config.ablation.use_pattern_generalization:
            self._generalize_patterns(pattern_library)

        return dict(pattern_library)

    def _mine_patterns_for_element(self, queries: List[str]) -> Dict[str, dict]:
        """
        为单个元素挖掘模式

        Returns:
            {pattern_text: {'type': str, 'count': int}}
        """
        patterns = defaultdict(lambda: {'type': '', 'count': 0})

        for query in queries:
            # 1. 实体-属性模式
            entity_attr_pattern = self._extract_entity_attr_pattern(query)
            if entity_attr_pattern:
                patterns[entity_attr_pattern]['type'] = 'entity_attr'
                patterns[entity_attr_pattern]['count'] += 1

            # 2. 时间模式
            time_pattern = self._extract_time_pattern(query)
            if time_pattern:
                patterns[time_pattern]['type'] = 'time'
                patterns[time_pattern]['count'] += 1

            # 3. 聚合模式
            agg_pattern = self._extract_aggregation_pattern(query)
            if agg_pattern:
                patterns[agg_pattern]['type'] = 'aggregation'
                patterns[agg_pattern]['count'] += 1

        return dict(patterns)

    def _load_schema_vocab(self) -> Set[str]:
        """
        加载 schema 词表（generic 模式）

        从 table/field 名称与描述分词构建词表，用于识别历史查询中的 schema 相关
        签名 token —— 使 TE-RAG 的 annotation learning 仍然来自 historical Query-SQL
        与 schema 的关联，而不依赖固定业务词表。
        """
        if self._schema_vocab is not None:
            return self._schema_vocab

        vocab: Set[str] = set()
        try:
            field_csv = self.config.data.field_csv
            table_csv = self.config.data.table_csv
            if field_csv and os.path.exists(field_csv):
                field_df = pd.read_csv(field_csv)
                for _, row in field_df.iterrows():
                    vocab.update(t for t in tokenize_text(str(row['field_name']), self.language) if len(t) > 1)
                    vocab.update(t for t in tokenize_text(str(row.get('field_name_desc', '')), self.language) if len(t) > 1)
            if table_csv and os.path.exists(table_csv):
                table_df = pd.read_csv(table_csv)
                for _, row in table_df.iterrows():
                    vocab.update(t for t in tokenize_text(str(row['table']), self.language) if len(t) > 1)
                    vocab.update(t for t in tokenize_text(str(row.get('table_desc', '')), self.language) if len(t) > 1)
        except Exception as e:
            print(f"  Warning: 加载 schema 词表失败: {e}")

        self._schema_vocab = vocab
        return vocab

    def _extract_entity_attr_pattern(self, query: str) -> Optional[str]:
        """提取实体-属性模式"""
        if self.mode == 'generic':
            return self._extract_signature_pattern(query)

        # keywords 模式（中文业务词表，来自配置）
        words = list(tokenize_text(query, self.language))

        entities = [w for w in words if any(k in w for k in self.entity_keywords)]
        attributes = [w for w in words if any(k in w for k in self.attribute_keywords)]

        if entities or attributes:
            parts = []
            if entities:
                parts.append(f"ENTITY:{'|'.join(sorted(set(entities)))}")
            if attributes:
                parts.append(f"ATTR:{'|'.join(sorted(set(attributes)))}")
            return ' '.join(parts)

        return None

    def _extract_signature_pattern(self, query: str) -> Optional[str]:
        """
        提取 schema 签名模式（generic 模式）

        历史查询中与 schema 词表相交的 content token 构成签名：
        SIG:tok1|tok2|...
        """
        vocab = self._load_schema_vocab()
        if not vocab:
            return None

        tokens = [t for t in tokenize_text(query, self.language) if len(t) > 1]
        signature = sorted({t for t in tokens if t in vocab})

        if not signature:
            return None
        return f"SIG:{'|'.join(signature[:8])}"

    def _extract_time_pattern(self, query: str) -> Optional[str]:
        """提取时间模式"""
        text = query.lower() if self.mode == 'generic' else query
        for pattern, pattern_type in self.time_patterns:
            if re.search(pattern, text):
                return f"TIME:{pattern_type}"
        return None

    def _extract_aggregation_pattern(self, query: str) -> Optional[str]:
        """提取聚合模式"""
        text = query.lower() if self.mode == 'generic' else query
        for keyword, agg_type in self.aggregation_keywords.items():
            if keyword in text:
                return f"AGG:{agg_type}"
        return None

    def _generalize_patterns(self, pattern_library: Dict[str, List[Pattern]]):
        """
        模式泛化

        对于模式较少的元素，从相似元素迁移模式
        """
        element_list = sorted(pattern_library.keys())

        for i, e1 in enumerate(element_list):
            # 如果元素的模式数量较少
            if len(pattern_library[e1]) < 3:
                for j, e2 in enumerate(element_list):
                    if i != j and len(pattern_library[e2]) >= 3:
                        # 计算模式相似度
                        similarity = self._compute_pattern_similarity(
                            pattern_library[e1],
                            pattern_library[e2]
                        )

                        if similarity > self.similarity_threshold:
                            # 迁移模式
                            for pattern in pattern_library[e2][:2]:
                                transferred = Pattern(
                                    pattern_id=f"{e1}_{len(pattern_library[e1])}_transferred",
                                    pattern_type=pattern.pattern_type,
                                    pattern_text=pattern.pattern_text,
                                    count=int(pattern.count * similarity),
                                    source_element=e2,
                                    transferred=True
                                )
                                pattern_library[e1].append(transferred)

    def _compute_pattern_similarity(self, patterns1: List[Pattern], patterns2: List[Pattern]) -> float:
        """计算两个元素的模式相似度"""
        set1 = set(p.pattern_text for p in patterns1)
        set2 = set(p.pattern_text for p in patterns2)

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def save(self, pattern_library: Dict[str, List[Pattern]], output_path: str):
        """保存模式库"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for element_node, patterns in pattern_library.items():
                record = {
                    'element': element_node,
                    'patterns': [p.to_dict() for p in patterns]
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"模式库已保存到: {output_path}")

    def load(self, input_path: str) -> Dict[str, List[Pattern]]:
        """加载模式库"""
        pattern_library = {}

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                element = record['element']
                patterns = [Pattern.from_dict(p) for p in record['patterns']]
                pattern_library[element] = patterns

        return pattern_library

    def get_pattern_stats(self, pattern_library: Dict[str, List[Pattern]]) -> Dict:
        """获取模式统计"""
        stats = {
            'total_elements': len(pattern_library),
            'total_patterns': sum(len(p) for p in pattern_library.values()),
            'transferred_patterns': sum(
                sum(1 for p in patterns if p.transferred)
                for patterns in pattern_library.values()
            ),
            'pattern_types': defaultdict(int),
        }

        for patterns in pattern_library.values():
            for pattern in patterns:
                stats['pattern_types'][pattern.pattern_type] += 1

        stats['pattern_types'] = dict(stats['pattern_types'])

        return stats


def main():
    """演示模式挖掘"""
    import sys
    from pathlib import Path
    import networkx as nx

    # 添加项目根目录
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.config import TERAGConfig
    from terag.graph_builder import BipartiteGraphBuilder

    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    # 加载训练数据
    train_data = []
    with open(config.get_split_path('train'), 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line))
    train_df = pd.DataFrame(train_data)

    print(f"加载训练数据: {len(train_df)} 条")

    # 构建图
    builder = BipartiteGraphBuilder(config)
    graph = builder.build(train_df)

    # 获取元素到查询的映射
    element_to_queries = builder.get_element_to_queries(graph, train_df)
    print(f"元素数量: {len(element_to_queries)}")

    # 挖掘模式
    miner = PatternMiner(config)
    patterns = miner.mine(element_to_queries)

    # 打印统计
    stats = miner.get_pattern_stats(patterns)
    print(f"\n模式统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 保存
    if config.output.save_patterns:
        miner.save(patterns, config.get_artifact_path('patterns.jsonl'))


if __name__ == "__main__":
    main()
