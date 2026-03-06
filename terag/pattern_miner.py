"""
模式挖掘器

挖掘查询模式（实体-属性、时间、聚合）并支持模式泛化
"""

import os
import json
import re
import jieba
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field

from terag.config import TERAGConfig


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

        # 关键词配置
        self.entity_keywords = config.pattern_mining.entity_keywords
        self.attribute_keywords = config.pattern_mining.attribute_keywords
        self.similarity_threshold = config.pattern_mining.similarity_threshold
        self.min_pattern_count = config.pattern_mining.min_pattern_count

        # 聚合关键词
        self.aggregation_keywords = {
            '多少': 'VALUE',
            '哪些': 'LIST',
            '总计': 'SUM',
            '合计': 'SUM',
            '平均': 'AVG',
            '最大': 'MAX',
            '最小': 'MIN',
            '数量': 'COUNT',
            '排名': 'RANK',
        }

        # 时间模式
        self.time_patterns = [
            (r'\d{4}年\d{1,2}月', 'YEAR_MONTH'),
            (r'\d{4}年', 'YEAR'),
            (r'\d{1,2}月', 'MONTH'),
            (r'今年', 'THIS_YEAR'),
            (r'去年', 'LAST_YEAR'),
            (r'上月', 'LAST_MONTH'),
            (r'本季度', 'THIS_QUARTER'),
        ]

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
        for element_node, queries in element_to_queries.items():
            patterns = self._mine_patterns_for_element(list(queries))
            for pattern_key, pattern_data in patterns.items():
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

    def _extract_entity_attr_pattern(self, query: str) -> Optional[str]:
        """提取实体-属性模式"""
        words = list(jieba.cut(query))

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

    def _extract_time_pattern(self, query: str) -> Optional[str]:
        """提取时间模式"""
        for pattern, pattern_type in self.time_patterns:
            if re.search(pattern, query):
                return f"TIME:{pattern_type}"
        return None

    def _extract_aggregation_pattern(self, query: str) -> Optional[str]:
        """提取聚合模式"""
        for keyword, agg_type in self.aggregation_keywords.items():
            if keyword in query:
                return f"AGG:{agg_type}"
        return None

    def _generalize_patterns(self, pattern_library: Dict[str, List[Pattern]]):
        """
        模式泛化

        对于模式较少的元素，从相似元素迁移模式
        """
        element_list = list(pattern_library.keys())

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
