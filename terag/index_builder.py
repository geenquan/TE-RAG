"""
索引构建器

构建 BM25 倒排索引，支持多权重增强
"""

import os
import json
import pickle
import math
import jieba
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from terag.config import TERAGConfig


class IndexBuilder:
    """
    索引构建器

    功能：
    1. 构建 BM25 倒排索引
    2. 多权重增强（表名=2.0, 描述=1.5, 字段名=1.0）
    3. 支持加载模式库增强索引

    使用方式:
        builder = IndexBuilder(config)
        index = builder.build()
        builder.save(index, "artifacts/bm25_index/")
    """

    def __init__(self, config: TERAGConfig):
        """
        初始化索引构建器

        Args:
            config: TE-RAG 配置
        """
        self.config = config

        # BM25 参数
        self.k1 = config.index.bm25.k1
        self.b = config.index.bm25.b

        # 字段权重
        self.field_weights = config.index.field_weights

        # 加载表和字段数据
        self.field_df = pd.read_csv(config.data.field_csv)
        self.table_df = pd.read_csv(config.data.table_csv)

        # 模式库（可选）
        self.pattern_library = None

    def load_patterns(self, pattern_path: str):
        """加载模式库"""
        from terag.pattern_miner import PatternMiner

        miner = PatternMiner(self.config)
        self.pattern_library = miner.load(pattern_path)
        print(f"加载模式库: {len(self.pattern_library)} 个元素")

    def build(self) -> Dict:
        """
        构建索引

        Returns:
            索引字典，包含：
            - inverted_index: 倒排索引 {term: [(doc_id, tf, doc_length), ...]}
            - document_norms: 文档长度 {doc_id: length}
            - documents: 文档内容 {doc_id: document_text}
            - avgdl: 平均文档长度
            - N: 文档总数
        """
        inverted_index = defaultdict(list)
        document_norms = {}
        documents = {}

        # 是否使用增强索引
        use_enhanced = self.config.ablation.use_enhanced_index

        for _, row in self.table_df.iterrows():
            table_name = row['table']
            table_node = f"T:{table_name}"
            table_desc = row.get('table_desc', '')

            # 获取该表的列
            columns = self.field_df[self.field_df['table'] == table_name]

            # 构建文档
            document_parts = []

            if use_enhanced:
                # 增强索引：多权重
                document_parts.append((table_name, self.field_weights.table_name))

                if pd.notna(table_desc):
                    document_parts.append((table_desc, self.field_weights.table_desc))

                for _, col_row in columns.iterrows():
                    document_parts.append((col_row['field_name'], self.field_weights.field_name))

                    field_desc = col_row.get('field_name_desc', '')
                    if pd.notna(field_desc):
                        document_parts.append((field_desc, self.field_weights.field_desc))

                # 添加模式
                if self.pattern_library and table_node in self.pattern_library:
                    for pattern in self.pattern_library[table_node]:
                        document_parts.append((pattern.pattern_text, self.field_weights.pattern))
            else:
                # 基础索引：等权重
                document_parts.append((table_name, 1.0))

                if pd.notna(table_desc):
                    document_parts.append((table_desc, 1.0))

                for _, col_row in columns.iterrows():
                    document_parts.append((col_row['field_name'], 1.0))

            # 索引文档
            self._index_document(
                table_node, document_parts,
                inverted_index, document_norms, documents
            )

        # 计算统计信息
        N = len(document_norms)
        avgdl = np.mean(list(document_norms.values())) if document_norms else 1.0

        return {
            'inverted_index': dict(inverted_index),
            'document_norms': document_norms,
            'documents': documents,
            'avgdl': avgdl,
            'N': N,
            'k1': self.k1,
            'b': self.b,
        }

    def _index_document(
        self,
        doc_id: str,
        document_parts: List[Tuple[str, float]],
        inverted_index: Dict,
        document_norms: Dict,
        documents: Dict
    ):
        """
        索引单个文档

        Args:
            doc_id: 文档 ID
            document_parts: [(text, weight), ...]
            inverted_index: 倒排索引
            document_norms: 文档长度
            documents: 文档内容
        """
        term_frequencies = defaultdict(float)
        document_text_parts = []

        for text, weight in document_parts:
            words = list(jieba.cut(str(text)))
            document_text_parts.append(str(text))

            for word in words:
                if len(word) > 1:
                    term_frequencies[word] += weight

        # 计算文档长度
        doc_length = math.sqrt(sum(tf ** 2 for tf in term_frequencies.values()))
        document_norms[doc_id] = doc_length if doc_length > 0 else 1.0

        # 保存文档内容
        documents[doc_id] = ' '.join(document_text_parts)

        # 添加到倒排索引
        for term, tf in term_frequencies.items():
            inverted_index[term].append((doc_id, tf, doc_length))

    def save(self, index: Dict, output_dir: str):
        """保存索引"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存倒排索引
        with open(os.path.join(output_dir, 'inverted_index.pkl'), 'wb') as f:
            pickle.dump(index['inverted_index'], f)

        # 保存文档长度
        with open(os.path.join(output_dir, 'document_norms.pkl'), 'wb') as f:
            pickle.dump(index['document_norms'], f)

        # 保存文档内容
        with open(os.path.join(output_dir, 'documents.jsonl'), 'w', encoding='utf-8') as f:
            for doc_id, doc_text in index['documents'].items():
                f.write(json.dumps({'doc_id': doc_id, 'text': doc_text}, ensure_ascii=False) + '\n')

        # 保存元数据
        metadata = {
            'avgdl': index['avgdl'],
            'N': index['N'],
            'k1': index['k1'],
            'b': index['b'],
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"索引已保存到: {output_dir}")
        print(f"  文档数: {index['N']}")
        print(f"  平均文档长度: {index['avgdl']:.2f}")

    def load(self, input_dir: str) -> Dict:
        """加载索引"""
        with open(os.path.join(input_dir, 'inverted_index.pkl'), 'rb') as f:
            inverted_index = pickle.load(f)

        with open(os.path.join(input_dir, 'document_norms.pkl'), 'rb') as f:
            document_norms = pickle.load(f)

        documents = {}
        with open(os.path.join(input_dir, 'documents.jsonl'), 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                documents[record['doc_id']] = record['text']

        with open(os.path.join(input_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return {
            'inverted_index': inverted_index,
            'document_norms': document_norms,
            'documents': documents,
            'avgdl': metadata['avgdl'],
            'N': metadata['N'],
            'k1': metadata['k1'],
            'b': metadata['b'],
        }


class BM25Retriever:
    """
    BM25 检索器

    使用预构建的索引进行检索
    """

    def __init__(self, index: Dict):
        """
        初始化检索器

        Args:
            index: 索引字典
        """
        self.inverted_index = index['inverted_index']
        self.document_norms = index['document_norms']
        self.avgdl = index['avgdl']
        self.N = index['N']
        self.k1 = index['k1']
        self.b = index['b']

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        检索

        Args:
            query: 查询文本
            k: 返回的 top-k 结果

        Returns:
            [(doc_id, score), ...]
        """
        query_terms = [t for t in jieba.cut(query) if len(t) > 1]
        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            postings = self.inverted_index[term]
            df = len(postings)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf, doc_length in postings:
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                score = idf * numerator / denominator
                scores[doc_id] += score

        # 排序并返回 top-k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


def main():
    """演示索引构建"""
    import sys
    from pathlib import Path

    # 添加项目根目录
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from terag.config import TERAGConfig

    # 加载配置
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = TERAGConfig.from_yaml(str(config_path))

    print("=" * 60)
    print("索引构建")
    print("=" * 60)

    # 构建索引
    builder = IndexBuilder(config)

    # 尝试加载模式库
    pattern_path = config.get_artifact_path('patterns.jsonl')
    if os.path.exists(pattern_path):
        builder.load_patterns(pattern_path)

    index = builder.build()

    # 保存
    if config.output.save_index:
        builder.save(index, config.get_artifact_path('bm25_index'))

    # 测试检索
    print("\n测试检索:")
    retriever = BM25Retriever(index)

    test_queries = [
        "查询公司的售电量",
        "统计用户的电费金额",
        "分析供电所的回收率",
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        results = retriever.search(query, k=3)
        for doc_id, score in results:
            print(f"  {doc_id}: {score:.4f}")


if __name__ == "__main__":
    main()
