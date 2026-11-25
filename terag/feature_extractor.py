"""
特征提取器

提取表级和字段级特征
"""

import pandas as pd
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

from terag.config import TERAGConfig, FeatureConfig
from terag.index_builder import BM25Retriever
from terag.pattern_miner import Pattern, PatternMiner
from terag.code_mapper import CodeMapper
from terag.tokenization import tokenize_text
from datasets.gt_utils import parse_ground_truth_tables, parse_ground_truth_fields


class FeatureExtractor:
    """
    特征提取器

    提取两类特征：
    1. 表级特征：bm25_score, graph_score, pattern_score
    2. 字段级特征：direct_match, graph_propagation, role_prior, train_recommend

    使用方式:
        extractor = FeatureExtractor(config, graph, index, patterns, train_data)
        table_features = extractor.extract_table_features(query)
        field_features = extractor.extract_field_features(query, table_name)
    """

    def __init__(
        self,
        config: TERAGConfig,
        graph: nx.Graph,
        index: Dict,
        patterns: Dict[str, List[Pattern]],
        train_data: pd.DataFrame,
        field_df: pd.DataFrame,
        table_df: pd.DataFrame
    ):
        """
        初始化特征提取器

        Args:
            config: TE-RAG 配置
            graph: 二分图
            index: BM25 索引
            patterns: 模式库
            train_data: 训练数据
            field_df: 字段表
            table_df: 数据表
        """
        self.config = config
        self.graph = graph
        self.patterns = patterns
        self.train_data = train_data
        self.field_df = field_df
        self.table_df = table_df

        # BM25 检索器
        self.bm25_retriever = BM25Retriever(index)

        # 分词语言（PowerGrid=chinese / BIRD=english）
        self.language = getattr(config.tokenizer, 'language', 'chinese')

        # 获取特征配置参数（支持消融）
        self._load_feature_params()

        # 初始化 CodeMapper 用于查询扩展
        self.code_mapper = None
        self._init_code_mapper()

        # 同义词词典（从配置加载）
        self.synonyms_dict = getattr(config.feature, 'synonyms_dict', {})

        # 预计算训练数据的分词结果与 gold 标注（多表兼容）
        self._train_tokens_cache = {}
        self._train_gold_cache: Dict[int, Tuple[List[str], List[Tuple[str, str]]]] = {}
        if train_data is not None and not train_data.empty:
            for idx in range(len(train_data)):
                row = train_data.iloc[idx]
                self._train_tokens_cache[idx] = set(tokenize_text(str(row['question']), self.language))
                self._train_gold_cache[idx] = (
                    parse_ground_truth_tables(row),
                    parse_ground_truth_fields(row),
                )

        # 查询分词缓存
        self._query_tokens_cache = {}

    def _tokenize(self, text: str) -> List[str]:
        """按当前数据集语言分词"""
        return tokenize_text(text, self.language)

    def _init_code_mapper(self):
        """初始化 CodeMapper 用于查询扩展

        P1-A 泄漏修复：不再加载 artifacts/code_mappings.json（历史划分产物），
        改为仅从当前 fit(train_data) 传入的训练数据挖掘码值映射。
        """
        try:
            self.code_mapper = CodeMapper(self.config)
            if self.train_data is not None and not self.train_data.empty \
                    and 'sql' in self.train_data.columns:
                mined = self.code_mapper.mine_from_sql(self.train_data)
                n_mappings = sum(len(v) for v in mined.values()) if mined else 0
                print(f"CodeMapper: 从当前训练集挖掘码值映射（{n_mappings} 条）")
        except Exception as e:
            print(f"Warning: CodeMapper initialization failed: {e}")
            self.code_mapper = None

    def _load_feature_params(self):
        """从配置加载特征计算参数"""
        feat = self.config.feature

        # 表级特征参数
        self.graph_propagation_weight = getattr(feat, 'graph_propagation_weight', 3.0)
        self.template_match_weight = getattr(feat, 'template_match_weight', 0.6)

        # 字段级特征参数
        self.direct_match_multiplier = getattr(feat, 'direct_match_multiplier', 1.5)
        self.field_name_match_bonus = getattr(feat, 'field_name_match_bonus', 0.4)
        self.field_graph_score_cap = getattr(feat, 'field_graph_score_cap', 5.0)
        self.train_recommend_multiplier = getattr(feat, 'train_recommend_multiplier', 1.5)
        self.train_recommend_cap = getattr(feat, 'train_recommend_cap', 3.0)

    def _get_query_tokens(self, query: str) -> Set[str]:
        """获取查询的分词结果（带缓存）

        包含 Phase 1 的 expand 步骤：
        1. 分词（中文 jieba / 英文正则，按数据集配置）
        2. CodeMapper 扩展（码值映射）
        3. 同义词扩展
        """
        if query not in self._query_tokens_cache:
            # 1. 基础分词
            tokens = set(self._tokenize(query))

            # 2. CodeMapper 扩展（码值映射）
            if self.code_mapper is not None:
                try:
                    expanded_query = self.code_mapper.expand_query(query)
                    # 扩展后的查询重新分词
                    if expanded_query != query:
                        expanded_tokens = set(self._tokenize(expanded_query))
                        tokens = tokens | expanded_tokens
                except Exception:
                    pass

            # 3. 同义词扩展
            if self.synonyms_dict:
                expanded = set()
                for token in tokens:
                    if token in self.synonyms_dict:
                        expanded.update(self.synonyms_dict[token])
                tokens = tokens | expanded

            self._query_tokens_cache[query] = tokens
        return self._query_tokens_cache[query]

    def extract_table_features(self, query: str) -> Dict[str, Dict[str, float]]:
        """
        提取表级特征

        Args:
            query: 查询文本

        Returns:
            {table_node: {'bm25_score': float, 'graph_score': float, 'pattern_score': float}}
        """
        features = {}
        query_tokens = self._get_query_tokens(query)

        # 1. BM25 检索
        bm25_results = self.bm25_retriever.search(query, k=len(self.table_df))
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}

        # 2. 图传播得分
        graph_scores = self._compute_graph_scores(query, query_tokens)

        # 3. 模式匹配得分
        pattern_scores = self._compute_pattern_scores(query, query_tokens)

        # 合并特征（排序保证跨进程确定性：set 迭代顺序受哈希随机化影响）
        all_tables = sorted(set(bm25_scores.keys()) | set(graph_scores.keys()) | set(pattern_scores.keys()))

        for table_node in all_tables:
            features[table_node] = {
                'bm25_score': bm25_scores.get(table_node, 0.0),
                'graph_score': graph_scores.get(table_node, 0.0),
                'pattern_score': pattern_scores.get(table_node, 0.0),
            }

        return features

    def _compute_graph_scores(self, query: str, query_tokens: Set[str]) -> Dict[str, float]:
        """计算图传播得分

        消融开关: use_graph_weight=False 时返回空字典
        """
        # 消融：如果关闭图权重，直接返回空
        if not self.config.ablation.use_graph_weight:
            return {}

        if self.train_data is None or self.train_data.empty:
            return {}

        scores = defaultdict(float)
        threshold = self.config.feature.graph_propagation_threshold

        for idx in range(len(self.train_data)):
            hist_tokens = self._train_tokens_cache.get(idx, set())
            if not hist_tokens:
                continue

            # 计算 Jaccard 相似度
            intersection = len(query_tokens & hist_tokens)
            if intersection == 0:
                continue

            union = len(query_tokens | hist_tokens)
            sim = intersection / union if union > 0 else 0

            if sim > threshold:
                # 该历史查询对应的 gold 表（多表查询为每张表传播得分）
                gold_tables, _ = self._train_gold_cache.get(idx, ([], []))
                for table_id in gold_tables:
                    table_node = f"T:{table_id}"
                    # 使用配置参数
                    scores[table_node] += sim * self.graph_propagation_weight

        return dict(scores)

    def _compute_pattern_scores(self, query: str, query_tokens: Set[str]) -> Dict[str, float]:
        """计算模式匹配得分

        消融开关: use_template_mining=False 时返回空字典
        """
        if not self.config.ablation.use_template_mining:
            return {}

        scores = {}

        for table_node, pattern_list in self.patterns.items():
            if not pattern_list:
                continue

            max_score = 0.0
            for pattern in pattern_list:
                pattern_tokens = set(self._tokenize(pattern.pattern_text))
                if pattern_tokens:
                    overlap = len(query_tokens & pattern_tokens) / len(pattern_tokens)
                    max_score = max(max_score, overlap)

            if max_score > 0:
                # 使用配置参数
                scores[table_node] = max_score * self.template_match_weight

        return scores

    def extract_field_features(
        self,
        query: str,
        table_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        提取字段级特征

        Args:
            query: 查询文本
            table_name: 表名

        Returns:
            {field_name: {'direct_match': float, 'graph_propagation': float,
                          'role_prior': float, 'train_recommend': float}}
        """
        features = {}
        query_tokens = self._get_query_tokens(query)
        query_terms = [t for t in self._tokenize(query) if len(t) > 1]
        query_term_set = set(query_terms)

        # 获取该表的字段
        columns = self.field_df[self.field_df['table'] == table_name]

        for _, col_row in columns.iterrows():
            field_name = col_row['field_name']
            field_desc = str(col_row.get('field_name_desc', ''))
            col_node = f"C:{table_name}.{str(field_name).lower()}"

            # 1. 直接匹配得分
            s_direct = self._compute_direct_match_score(
                field_name, field_desc, query_terms, query_term_set
            )

            # 2. 图传播得分
            s_graph = self._compute_field_graph_score(col_node, query, query_tokens)

            # 3. 模式匹配得分（role_prior）
            s_pattern = self._compute_field_pattern_score(col_node, query_tokens)

            # 4. 训练数据字段推荐
            s_train = self._compute_train_field_score(table_name, field_name, query_tokens)

            # 5. B/C 注解加成（P2-D3 接口）：查询词命中该列的 confirmed
            #    术语注解（B 别名/共现 + C 迁移），w_ann=0.5 与 D1 对照一致；
            #    annotation_terms 由 TERAGRetrieverV2.fit 注入，缺省空=关闭
            s_ann = 0.0
            ann_terms = getattr(self, 'annotation_terms', None) or {}
            ann_key = f"{table_name.lower()}.{field_name.lower()}"
            for term in ann_terms.get(ann_key, []):
                if term in query_term_set:
                    s_ann += 0.5

            features[field_name] = {
                'direct_match': s_direct,
                'graph_propagation': s_graph,
                'role_prior': s_pattern,
                'train_recommend': s_train,
                'annotation_hit': s_ann,
            }

        return features

    def _compute_direct_match_score(
        self,
        field_name: str,
        field_desc: str,
        query_terms: List[str],
        query_term_set: Set[str]
    ) -> float:
        """计算直接匹配得分"""
        field_text = f"{field_name} {field_desc}"
        field_terms = set(t for t in self._tokenize(field_text) if len(t) > 1)

        overlap = len(query_term_set & field_terms)
        # 使用配置参数
        s_direct = overlap / max(len(query_terms), 1) * self.direct_match_multiplier

        # 字段名直接包含查询词
        for term in query_terms:
            if term in field_name or term in field_desc:
                # 使用配置参数
                s_direct += self.field_name_match_bonus

        return s_direct

    def _compute_field_graph_score(
        self,
        col_node: str,
        query: str,
        query_tokens: Set[str]
    ) -> float:
        """计算字段图传播得分

        消融开关: use_graph_weight=False 时直接返回 0.0
        """
        # 消融：如果关闭图权重，直接返回 0
        if not self.config.ablation.use_graph_weight:
            return 0.0

        if self.train_data is None or self.train_data.empty:
            return 0.0

        if not self.graph.has_node(col_node):
            return 0.0

        score = 0.0
        threshold = self.config.feature.graph_propagation_threshold

        for idx in range(len(self.train_data)):
            hist_tokens = self._train_tokens_cache.get(idx, set())
            if not hist_tokens:
                continue

            intersection = len(query_tokens & hist_tokens)
            if intersection == 0:
                continue

            union = len(query_tokens | hist_tokens)
            sim = intersection / union if union > 0 else 0

            if sim > threshold:
                query_node = f"Q_{idx}"
                if self.graph.has_edge(query_node, col_node):
                    weight = self.graph[query_node][col_node].get('weight', 1.0)
                    score += sim * weight

        # 使用配置参数
        return min(score, self.field_graph_score_cap)

    def _compute_field_pattern_score(self, col_node: str, query_tokens: Set[str]) -> float:
        """计算字段模式匹配得分

        消融开关: use_template_mining=False 时返回 0.0
        """
        if not self.config.ablation.use_template_mining:
            return 0.0

        if col_node not in self.patterns:
            return 0.0

        max_sim = 0.0
        for pattern in self.patterns[col_node]:
            pattern_tokens = set(self._tokenize(pattern.pattern_text))
            if pattern_tokens:
                sim = len(query_tokens & pattern_tokens) / len(pattern_tokens)
                max_sim = max(max_sim, sim)

        return max_sim

    def _compute_train_field_score(
        self,
        table_name: str,
        field_name: str,
        query_tokens: Set[str]
    ) -> float:
        """计算训练数据字段推荐得分"""
        if self.train_data is None or self.train_data.empty:
            return 0.0

        score = 0.0
        threshold = self.config.feature.train_similarity_threshold

        for idx in range(len(self.train_data)):
            gold_tables, gold_fields = self._train_gold_cache.get(idx, ([], []))

            # 检查是否是同一个表
            if table_name not in gold_tables:
                continue

            # 计算查询相似度
            hist_tokens = self._train_tokens_cache.get(idx, set())
            if not hist_tokens:
                continue

            union = len(query_tokens | hist_tokens)
            if union == 0:
                continue

            sim = len(query_tokens & hist_tokens) / union

            if sim > threshold:
                # 检查该字段是否在训练数据的 gold 字段列表中
                if any(str(t).lower() == str(table_name).lower() and str(f).lower() == str(field_name).lower() for t, f in gold_fields):
                    # 使用配置参数
                    score += sim * self.train_recommend_multiplier

        # 使用配置参数
        return min(score, self.train_recommend_cap)
