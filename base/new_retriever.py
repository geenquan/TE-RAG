import pandas as pd
import networkx as nx
import re
import jieba
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import math


class TE_RAG:
    """
    TE-RAG: 表格增强检索增强生成框架

    主要功能：
    1. 构建查询-数据库元素的加权二分图
    2. 基于图的模式匹配与注解生成
    3. 倒排索引构建与增强检索
    """

    def __init__(self, field_csv: str, table_csv: str, qa_csv: str):
        # 读取CSV文件
        self.field_df = pd.read_csv(field_csv)
        self.table_df = pd.read_csv(table_csv)
        self.qa_df = pd.read_csv(qa_csv)

        # 初始化二分图
        self.G = nx.Graph()

        # 角色权重（可根据验证集优化）
        self.role_weights = {
            'SELECT': 1.0,      # 选择列
            'WHERE': 1.5,       # 过滤条件（更重要）
            'JOIN': 1.2,        # 连接条件
            'GROUP_BY': 1.1,    # 分组
            'ORDER_BY': 0.8,    # 排序
            'HAVING': 1.3,      # 分组过滤
            'FROM': 1.0,        # 表引用
        }

        # 存储元素到查询的映射
        self.element_to_queries: Dict[str, Set[str]] = defaultdict(set)
        # 存储查询到元素的映射
        self.query_to_elements: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # 模板库
        self.template_library: Dict[str, List[Dict]] = defaultdict(list)

        # 注解库
        self.annotations: Dict[str, Dict] = {}

        # 倒排索引
        self.inverted_index: Dict[str, List[Tuple]] = defaultdict(list)
        self.document_norms: Dict[str, float] = {}

    def parse_sql_elements(self, sql: str) -> Dict[str, List[str]]:
        """
        从SQL中解析出引用的表和列，并识别它们的角色

        返回: {
            'tables': [(table_name, role), ...],
            'columns': [(column_name, role, table_name), ...]
        }
        """
        result = {
            'tables': [],
            'columns': []
        }

        if pd.isna(sql) or not isinstance(sql, str):
            return result

        sql_upper = sql.upper()

        # 解析FROM子句中的表
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_\.]*)'
        from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
        for table in from_matches:
            # 简化表名（去掉schema前缀）
            table_simple = table.split('.')[-1] if '.' in table else table
            result['tables'].append((table, 'FROM', table_simple))

        # 解析JOIN子句中的表
        join_pattern = r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_\.]*)'
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        for table in join_matches:
            table_simple = table.split('.')[-1] if '.' in table else table
            result['tables'].append((table, 'JOIN', table_simple))

        # 解析SELECT子句中的列
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # 提取列名（处理别名）
            columns = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:,|\s+AS|\s+FROM|\s*$)', select_clause, re.IGNORECASE)
            for col in columns:
                if col.upper() not in ['SELECT', 'DISTINCT', 'AS', 'AND', 'OR', 'WHERE']:
                    result['columns'].append((col, 'SELECT', ''))

        # 解析WHERE子句中的列
        where_pattern = r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|HAVING|LIMIT|;|$)'
        where_match = re.search(where_pattern, sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # 提取列名（处理 a.column 格式）
            columns = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)\s*(?:=|>|<|>=|<=|<>|!=|LIKE|IN|IS)', where_clause, re.IGNORECASE)
            for col in columns:
                col_name = col.split('.')[-1] if '.' in col else col
                result['columns'].append((col_name, 'WHERE', ''))

        # 解析GROUP BY子句
        group_pattern = r'GROUP BY\s+(.*?)(?:HAVING|ORDER BY|LIMIT|;|$)'
        group_match = re.search(group_pattern, sql, re.IGNORECASE | re.DOTALL)
        if group_match:
            group_clause = group_match.group(1)
            columns = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', group_clause)
            for col in columns:
                if col.upper() not in ['BY', 'AND', 'OR']:
                    result['columns'].append((col, 'GROUP_BY', ''))

        # 解析ORDER BY子句
        order_pattern = r'ORDER BY\s+(.*?)(?:LIMIT|;|$)'
        order_match = re.search(order_pattern, sql, re.IGNORECASE | re.DOTALL)
        if order_match:
            order_clause = order_match.group(1)
            columns = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', order_clause)
            for col in columns:
                if col.upper() not in ['BY', 'ASC', 'DESC', 'AND', 'OR']:
                    result['columns'].append((col, 'ORDER_BY', ''))

        return result

    def build_bipartite_graph(self):
        """
        构建加权二分图

        G = (Q ∪ E, E, W)
        - Q: 历史查询节点（左分区）
        - E = T ∪ C: 数据库元素节点（右分区），包括表和列
        - 边权重基于元素在SQL中的角色
        """
        print("开始构建二分图...")

        # 1. 添加查询节点（左分区）
        query_nodes = set()
        for idx, row in self.qa_df.iterrows():
            question = row['question']
            if pd.notna(question):
                query_nodes.add(question)
                self.G.add_node(f"Q_{idx}", question=question, bipartite=0)

        # 2. 添加表节点和列节点（右分区）
        table_nodes = set()
        column_nodes = set()

        # 从field_df中提取列节点
        for _, row in self.field_df.iterrows():
            table_name = row['table']
            field_name = row['field_name']
            field_desc = row.get('field_name_desc', '')

            # 添加列节点
            col_node = f"C:{table_name}.{field_name}"
            column_nodes.add(col_node)
            self.G.add_node(col_node,
                          node_type='column',
                          table=table_name,
                          field=field_name,
                          description=field_desc,
                          bipartite=1)

        # 从table_df中提取表节点
        for _, row in self.table_df.iterrows():
            table_name = row['table']
            table_desc = row.get('table_desc', '')

            # 添加表节点
            table_node = f"T:{table_name}"
            table_nodes.add(table_node)
            self.G.add_node(table_node,
                          node_type='table',
                          table=table_name,
                          description=table_desc,
                          bipartite=1)

        # 3. 构建边（基于SQL解析）
        for idx, row in self.qa_df.iterrows():
            question = row['question']
            sql = row.get('sql', '')
            table_from_csv = row.get('table', '')
            fields_from_csv = row.get('field', '')

            query_node = f"Q_{idx}"

            # 从CSV字段中获取元素
            if pd.notna(fields_from_csv) and isinstance(fields_from_csv, str):
                field_list = fields_from_csv.split('|')
                for field in field_list:
                    field = field.strip()
                    if field:
                        # 确定表名
                        table_name = table_from_csv.split('.')[-1] if pd.notna(table_from_csv) else ''

                        # 添加列边
                        col_node = f"C:{table_name}.{field}"
                        if col_node in column_nodes:
                            # 从SQL解析角色确定权重
                            role = self._determine_field_role(field, sql)
                            weight = self.role_weights.get(role, 1.0)

                            if self.G.has_edge(query_node, col_node):
                                # 如果边已存在，累加权重
                                self.G[query_node][col_node]['weight'] += weight
                                self.G[query_node][col_node]['roles'].append(role)
                            else:
                                self.G.add_edge(query_node, col_node, weight=weight, roles=[role])

                            # 更新映射
                            self.element_to_queries[col_node].add(question)
                            self.query_to_elements[question]['columns'].append(col_node)

            # 添加表边
            if pd.notna(table_from_csv):
                table_simple = table_from_csv.split('.')[-1]
                table_node = f"T:{table_simple}"
                if table_node in table_nodes:
                    weight = self.role_weights.get('FROM', 1.0)
                    self.G.add_edge(query_node, table_node, weight=weight, roles=['FROM'])
                    self.element_to_queries[table_node].add(question)
                    self.query_to_elements[question]['tables'].append(table_node)

            # 从SQL中解析额外元素
            sql_elements = self.parse_sql_elements(sql)

            for table_full, role, table_simple in sql_elements['tables']:
                table_node = f"T:{table_simple}"
                if table_node in table_nodes:
                    weight = self.role_weights.get(role, 1.0)
                    if not self.G.has_edge(query_node, table_node):
                        self.G.add_edge(query_node, table_node, weight=weight, roles=[role])

            for col_name, role, table_name in sql_elements['columns']:
                # 尝试匹配列节点
                for col_node in column_nodes:
                    if col_node.endswith(f".{col_name}"):
                        weight = self.role_weights.get(role, 1.0)
                        if self.G.has_edge(query_node, col_node):
                            self.G[query_node][col_node]['weight'] += weight * 0.5  # 额外解析的权重降低
                        else:
                            self.G.add_edge(query_node, col_node, weight=weight * 0.5, roles=[role])

        print(f"二分图构建完成：")
        print(f"  - 查询节点数: {len(query_nodes)}")
        print(f"  - 表节点数: {len(table_nodes)}")
        print(f"  - 列节点数: {len(column_nodes)}")
        print(f"  - 总边数: {len(self.G.edges)}")

    def _determine_field_role(self, field: str, sql: str) -> str:
        """
        根据SQL确定字段的角色
        """
        if pd.isna(sql) or not isinstance(sql, str):
            return 'SELECT'

        sql_upper = sql.upper()
        field_upper = field.upper()

        # 检查WHERE子句
        if 'WHERE' in sql_upper:
            where_start = sql_upper.find('WHERE')
            where_clause = sql_upper[where_start:]
            if field_upper in where_clause.split('GROUP BY')[0].split('ORDER BY')[0]:
                return 'WHERE'

        # 检查GROUP BY
        if 'GROUP BY' in sql_upper and field_upper in sql_upper:
            group_start = sql_upper.find('GROUP BY')
            if field_upper in sql_upper[group_start:group_start+100]:
                return 'GROUP_BY'

        return 'SELECT'

    def extract_knowledge_patterns(self):
        """
        基于图的模式匹配与注解生成

        包括：
        1. 术语-元素模板挖掘
        2. 模式泛化与迁移
        3. 业务规则模式识别
        4. 注解组装
        """
        print("\n开始提取知识模式...")

        # 1. 术语-元素模板挖掘
        self._mine_templates()

        # 2. 模式泛化与迁移
        self._generalize_patterns()

        # 3. 业务规则模式识别
        self._recognize_business_rules()

        # 4. 组装注解
        self._assemble_annotations()

        print(f"知识模式提取完成：")
        print(f"  - 模板数量: {sum(len(v) for v in self.template_library.values())}")
        print(f"  - 注解元素数量: {len(self.annotations)}")

    def _mine_templates(self):
        """
        术语-元素模板挖掘

        对每个数据库元素，分析引用它的查询，识别自然语言模板
        """
        print("  挖掘术语-元素模板...")

        # 对每个元素，收集引用它的查询
        for element_node in self.element_to_queries:
            queries = list(self.element_to_queries[element_node])

            if len(queries) < 1:
                continue

            # 提取查询中的共同模式
            patterns = self._extract_query_patterns(queries)

            for pattern, count in patterns.items():
                if count >= 1:  # 至少出现1次
                    template = {
                        'pattern': pattern,
                        'count': count,
                        'queries': queries[:5],  # 保存示例查询
                        'type': 'direct'
                    }
                    self.template_library[element_node].append(template)

    def _extract_query_patterns(self, queries: List[str]) -> Dict[str, int]:
        """
        从查询列表中提取共同模式
        """
        patterns = defaultdict(int)

        for query in queries:
            # 分词
            words = list(jieba.cut(query))

            # 提取关键词模式
            # 1. 实体-属性模式
            entity_pattern = self._extract_entity_pattern(words)
            if entity_pattern:
                patterns[entity_pattern] += 1

            # 2. 时间模式
            time_pattern = self._extract_time_pattern(query)
            if time_pattern:
                patterns[f"TIME:{time_pattern}"] += 1

            # 3. 聚合模式
            agg_pattern = self._extract_aggregation_pattern(query)
            if agg_pattern:
                patterns[f"AGG:{agg_pattern}"] += 1

            # 4. 查询意图模式
            intent_pattern = self._extract_intent_pattern(query)
            if intent_pattern:
                patterns[f"INTENT:{intent_pattern}"] += 1

        return patterns

    def _extract_entity_pattern(self, words: List[str]) -> Optional[str]:
        """提取实体-属性模式"""
        # 识别实体词和属性词
        entity_keywords = ['公司', '供电所', '单位', '用户', '客户']
        attribute_keywords = ['售电量', '电费', '欠费', '回收率', '户数']

        entities = []
        attributes = []

        for word in words:
            for ek in entity_keywords:
                if ek in word:
                    entities.append(word)
                    break
            for ak in attribute_keywords:
                if ak in word:
                    attributes.append(word)
                    break

        if entities or attributes:
            return f"ENTITY:{'|'.join(entities)} ATTR:{'|'.join(attributes)}"
        return None

    def _extract_time_pattern(self, query: str) -> Optional[str]:
        """提取时间模式"""
        # 年月模式
        year_month = re.search(r'(\d{4})年(\d{1,2})月', query)
        if year_month:
            return f"年月查询"

        # 年份模式
        year = re.search(r'(\d{4})年', query)
        if year:
            return f"年份查询"

        return None

    def _extract_aggregation_pattern(self, query: str) -> Optional[str]:
        """提取聚合模式"""
        agg_keywords = {
            '多少': 'VALUE',
            '总计': 'SUM',
            '平均': 'AVG',
            '最大': 'MAX',
            '最小': 'MIN',
            '数量': 'COUNT',
            '哪些': 'LIST',
            '排行': 'RANK'
        }

        for keyword, agg_type in agg_keywords.items():
            if keyword in query:
                return agg_type

        return None

    def _extract_intent_pattern(self, query: str) -> Optional[str]:
        """提取查询意图"""
        intent_keywords = {
            '是多少': 'QUERY_VALUE',
            '有哪些': 'QUERY_LIST',
            '排名': 'QUERY_RANK',
            '趋势': 'QUERY_TREND',
            '对比': 'QUERY_COMPARE'
        }

        for keyword, intent in intent_keywords.items():
            if keyword in query:
                return intent

        return None

    def _generalize_patterns(self):
        """
        模式泛化与迁移

        对于查询覆盖较少的元素，从语义相似的元素中泛化模式
        """
        print("  执行模式泛化与迁移...")

        # 计算元素相似度
        element_similarity = self._compute_element_similarity()

        # 对每个元素进行模式迁移
        for element_node in self.template_library:
            current_templates = self.template_library[element_node]

            # 如果模板数量较少，尝试从相似元素迁移
            if len(current_templates) < 3:
                similar_elements = element_similarity.get(element_node, [])

                for similar_element, similarity in similar_elements[:3]:  # 取前3个最相似的
                    if similar_element in self.template_library:
                        for template in self.template_library[similar_element]:
                            # 迁移模板并调整
                            adapted_template = {
                                'pattern': template['pattern'],
                                'count': template['count'] * similarity,  # 降低权重
                                'source': similar_element,
                                'type': 'transferred',
                                'similarity': similarity
                            }
                            self.template_library[element_node].append(adapted_template)

    def _compute_element_similarity(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        计算元素间的相似度

        sim(ei, ej) = template_overlap(ei, ej) + schema_similarity(ei, ej)
        """
        similarity = defaultdict(list)

        elements = list(self.template_library.keys())

        for i, e1 in enumerate(elements):
            for j, e2 in enumerate(elements):
                if i >= j:
                    continue

                # 模板重叠度
                templates1 = set(t['pattern'] for t in self.template_library[e1])
                templates2 = set(t['pattern'] for t in self.template_library[e2])
                template_overlap = len(templates1 & templates2) / max(len(templates1 | templates2), 1)

                # 架构相似度
                schema_sim = self._compute_schema_similarity(e1, e2)

                total_sim = template_overlap * 0.6 + schema_sim * 0.4

                if total_sim > 0.1:  # 只保留相似度大于0.1的
                    similarity[e1].append((e2, total_sim))
                    similarity[e2].append((e1, total_sim))

        # 排序
        for e in similarity:
            similarity[e].sort(key=lambda x: x[1], reverse=True)

        return similarity

    def _compute_schema_similarity(self, e1: str, e2: str) -> float:
        """
        计算两个元素的架构相似度
        """
        # 如果都是列节点
        if e1.startswith('C:') and e2.startswith('C:'):
            # 检查是否在同一表中
            table1 = e1.split('.')[0].replace('C:', '')
            table2 = e2.split('.')[0].replace('C:', '')

            if table1 == table2:
                return 0.5  # 同表列更相似

            # 检查数据类型是否相同
            type1 = self._get_field_type(e1)
            type2 = self._get_field_type(e2)

            if type1 and type2 and type1 == type2:
                return 0.3

        return 0.1

    def _get_field_type(self, col_node: str) -> Optional[str]:
        """获取字段的数据类型"""
        parts = col_node.replace('C:', '').split('.')
        if len(parts) >= 2:
            table = parts[0]
            field = parts[1]

            matches = self.field_df[
                (self.field_df['table'] == table) &
                (self.field_df['field_name'] == field)
            ]

            if not matches.empty:
                return matches.iloc[0].get('field_type', None)

        return None

    def _recognize_business_rules(self):
        """
        业务规则模式识别

        识别跨查询的重复约束模式
        """
        print("  识别业务规则模式...")

        # 分析WHERE子句中的条件模式
        for idx, row in self.qa_df.iterrows():
            question = row['question']
            sql = row.get('sql', '')

            if pd.isna(sql):
                continue

            # 提取条件模式
            conditions = self._extract_conditions(sql)

            # 将条件模式与元素关联
            elements = self.query_to_elements.get(question, {})
            for col_node in elements.get('columns', []):
                if col_node not in self.annotations:
                    self.annotations[col_node] = {
                        'reference_templates': [],
                        'constraint_templates': [],
                        'value_interpretations': {},
                        'common_usage': []
                    }

                for cond_type, cond_value in conditions:
                    self.annotations[col_node]['constraint_templates'].append({
                        'type': cond_type,
                        'value': cond_value,
                        'query': question
                    })

    def _extract_conditions(self, sql: str) -> List[Tuple[str, str]]:
        """
        从SQL中提取条件模式
        """
        conditions = []

        if pd.isna(sql) or not isinstance(sql, str):
            return conditions

        # 阈值模式
        threshold_patterns = [
            (r'>\s*(\d+)', 'GREATER_THAN'),
            (r'<\s*(\d+)', 'LESS_THAN'),
            (r'>=\s*(\d+)', 'GREATER_EQUAL'),
            (r'<=\s*(\d+)', 'LESS_EQUAL'),
        ]

        for pattern, cond_type in threshold_patterns:
            matches = re.findall(pattern, sql)
            for match in matches:
                conditions.append((cond_type, match))

        # 成员模式
        in_pattern = r'IN\s*\(([^)]+)\)'
        in_matches = re.findall(in_pattern, sql, re.IGNORECASE)
        for match in in_matches:
            conditions.append(('IN_SET', match))

        # 等值模式
        eq_pattern = r"=\s*'([^']+)'"
        eq_matches = re.findall(eq_pattern, sql)
        for match in eq_matches:
            conditions.append(('EQUALS_STRING', match))

        eq_num_pattern = r"=\s*(\d+)"
        eq_num_matches = re.findall(eq_num_pattern, sql)
        for match in eq_num_matches:
            conditions.append(('EQUALS_NUMBER', match))

        return conditions

    def _assemble_annotations(self):
        """
        注解组装

        为每个表和列组装综合性注解
        """
        print("  组装注解...")

        # 为表生成注解
        for _, row in self.table_df.iterrows():
            table_name = row['table']
            table_node = f"T:{table_name}"

            if table_node not in self.annotations:
                self.annotations[table_node] = {}

            self.annotations[table_node].update({
                'description': row.get('table_desc', ''),
                'query_templates': [],
                'join_templates': [],
                'domain_context': []
            })

            # 从模板库获取相关模板
            if table_node in self.template_library:
                self.annotations[table_node]['query_templates'] = [
                    t['pattern'] for t in self.template_library[table_node]
                ]

        # 为列生成注解
        for _, row in self.field_df.iterrows():
            table_name = row['table']
            field_name = row['field_name']
            col_node = f"C:{table_name}.{field_name}"

            if col_node not in self.annotations:
                self.annotations[col_node] = {}

            self.annotations[col_node].update({
                'description': row.get('field_name_desc', ''),
                'field_type': row.get('field_type', ''),
                'reference_templates': [],
                'constraint_templates': [],
                'value_interpretations': {},
                'common_usage': []
            })

            # 从模板库获取相关模板
            if col_node in self.template_library:
                self.annotations[col_node]['reference_templates'] = [
                    t['pattern'] for t in self.template_library[col_node][:10]
                ]

    def build_inverted_index(self):
        """
        构建增强的数据库元素倒排索引
        """
        print("\n开始构建倒排索引...")

        # 为每个表创建复合文档
        for table_node in [n for n in self.G.nodes if n.startswith('T:')]:
            table_name = table_node.replace('T:', '')

            # 获取表信息
            table_info = self.table_df[self.table_df['table'] == table_name]
            table_desc = table_info.iloc[0]['table_desc'] if not table_info.empty else ''

            # 获取该表的所有列
            columns = self.field_df[self.field_df['table'] == table_name]

            # 构建复合文档
            document_parts = []

            # 表名（加权）
            document_parts.append((table_name, 2.0))
            # 表描述（加权）
            if pd.notna(table_desc):
                document_parts.append((table_desc, 1.5))
            # 列名和描述
            for _, col_row in columns.iterrows():
                document_parts.append((col_row['field_name'], 1.0))
                if pd.notna(col_row.get('field_name_desc', '')):
                    document_parts.append((col_row['field_name_desc'], 0.8))

            # 添加注解中的模板
            if table_node in self.annotations:
                for template in self.annotations[table_node].get('query_templates', []):
                    document_parts.append((template, 0.5))

            # 分词并构建索引
            self._index_document(table_node, document_parts)

        print(f"倒排索引构建完成，词汇表大小: {len(self.inverted_index)}")

    def _index_document(self, doc_id: str, document_parts: List[Tuple[str, float]]):
        """
        为文档构建倒排索引
        """
        term_frequencies = defaultdict(float)

        for text, weight in document_parts:
            # 分词
            words = list(jieba.cut(str(text)))
            for word in words:
                if len(word) > 1:  # 忽略单字
                    term_frequencies[word] += weight

        # 计算文档长度归一化因子
        doc_length = math.sqrt(sum(tf ** 2 for tf in term_frequencies.values()))
        self.document_norms[doc_id] = doc_length if doc_length > 0 else 1

        # 添加到倒排索引
        for term, tf in term_frequencies.items():
            self.inverted_index[term].append((doc_id, tf, doc_length))

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        两阶段检索：表格检索 + 列排名

        Args:
            query: 用户查询
            k: 返回的top-k结果

        Returns:
            [(table_node, score, info), ...]
        """
        print(f"\n处理查询: {query}")

        # 阶段1：查询处理
        query_terms = list(jieba.cut(query))
        query_terms = [t for t in query_terms if len(t) > 1]

        # 模板匹配
        matched_templates = self._match_templates(query)

        # 阶段2：表格检索（BM25）
        table_scores = self._bm25_retrieval(query_terms)

        # 加入模板匹配贡献
        for table_node in table_scores:
            template_score = 0
            for template_info in matched_templates:
                if template_info['element'] == table_node:
                    template_score += template_info['confidence']
            table_scores[table_node] += template_score * 0.5

        # 筛选top-K表格
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # 阶段3：列排名
        results = []
        for table_node, table_score in top_tables:
            table_name = table_node.replace('T:', '')

            # 获取该表的列
            columns = self.field_df[self.field_df['table'] == table_name]

            column_scores = {}
            for _, col_row in columns.iterrows():
                col_node = f"C:{table_name}.{col_row['field_name']}"

                # 计算三个得分
                s_dir = self._direct_term_match(col_node, query_terms)
                s_graph = self._graph_propagation_score(col_node, query)
                s_pat = self._pattern_similarity_score(col_node, query)

                # 加权组合
                total_score = 0.4 * s_dir + 0.3 * s_graph + 0.3 * s_pat
                column_scores[col_node] = total_score

            # 获取top列
            top_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            results.append({
                'table': table_node,
                'table_score': table_score,
                'columns': top_columns,
                'annotation': self.annotations.get(table_node, {})
            })

        return results

    def _match_templates(self, query: str) -> List[Dict]:
        """
        将查询与模板库匹配
        """
        matches = []

        for element_node, templates in self.template_library.items():
            for template in templates:
                pattern = template['pattern']

                # 计算匹配度
                confidence = self._compute_template_similarity(query, pattern)

                if confidence > 0.3:  # 阈值
                    matches.append({
                        'element': element_node,
                        'pattern': pattern,
                        'confidence': confidence
                    })

        return sorted(matches, key=lambda x: x['confidence'], reverse=True)[:10]

    def _compute_template_similarity(self, query: str, pattern: str) -> float:
        """
        计算查询与模板的相似度
        """
        query_words = set(jieba.cut(query))
        pattern_words = set(jieba.cut(pattern))

        if not pattern_words:
            return 0

        overlap = len(query_words & pattern_words)
        return overlap / len(pattern_words)

    def _bm25_retrieval(self, query_terms: List[str], k1: float = 1.5, b: float = 0.75) -> Dict[str, float]:
        """
        BM25检索算法
        """
        scores = defaultdict(float)

        # 计算平均文档长度
        avgdl = sum(self.document_norms.values()) / len(self.document_norms) if self.document_norms else 1

        # 文档总数
        N = len(self.document_norms)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            postings = self.inverted_index[term]
            df = len(postings)  # 文档频率
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf, doc_length in postings:
                # BM25公式
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_length / avgdl)
                scores[doc_id] += idf * numerator / denominator

        return scores

    def _direct_term_match(self, col_node: str, query_terms: List[str]) -> float:
        """
        直接术语匹配得分
        """
        score = 0

        if col_node in self.annotations:
            annotation = self.annotations[col_node]

            # 检查描述
            desc = annotation.get('description', '')
            if pd.notna(desc) and isinstance(desc, str):
                desc_words = set(jieba.cut(desc))
                score += len(desc_words & set(query_terms)) * 0.5

            # 检查模板
            for template in annotation.get('reference_templates', []):
                if pd.notna(template) and isinstance(template, str):
                    template_words = set(jieba.cut(template))
                    score += len(template_words & set(query_terms)) * 0.3

        return score

    def _graph_propagation_score(self, col_node: str, query: str) -> float:
        """
        通过历史查询的图传播得分
        """
        score = 0

        # 找到与当前查询相似的历史查询
        for q_idx in range(len(self.qa_df)):
            hist_question = self.qa_df.iloc[q_idx]['question']

            # 计算查询相似度
            query_sim = self._compute_query_similarity(query, hist_question)

            if query_sim > 0.3:
                # 获取该历史查询关联的元素
                query_node = f"Q_{q_idx}"

                if self.G.has_node(query_node) and self.G.has_node(col_node):
                    if self.G.has_edge(query_node, col_node):
                        edge_weight = self.G[query_node][col_node].get('weight', 1.0)
                        score += query_sim * edge_weight

        return min(score, 5.0)  # 限制最大值

    def _compute_query_similarity(self, q1: str, q2: str) -> float:
        """
        计算两个查询的相似度
        """
        words1 = set(jieba.cut(q1))
        words2 = set(jieba.cut(q2))

        if not words1 or not words2:
            return 0

        return len(words1 & words2) / len(words1 | words2)

    def _pattern_similarity_score(self, col_node: str, query: str) -> float:
        """
        与模板的模式相似度得分
        """
        max_sim = 0

        if col_node in self.template_library:
            for template in self.template_library[col_node]:
                sim = self._compute_template_similarity(query, template['pattern'])
                max_sim = max(max_sim, sim)

        return max_sim

    def process(self):
        """
        执行整个TE-RAG流程
        """
        # 1. 构建二分图
        self.build_bipartite_graph()

        # 2. 提取知识模式
        self.extract_knowledge_patterns()

        # 3. 构建倒排索引
        self.build_inverted_index()

        # 4. 测试检索
        test_queries = [

            "2025年2月钱江供电所的售电量是多少？",
            "2025年2月杭州公司的电费回收率是多少？",
            "2025年2月浙江公司欠费用户有哪些？"
        ]

        for query in test_queries:
            results = self.retrieve(query, k=3)
            print(f"\n查询: {query}")
            for i, result in enumerate(results):
                print(f"  Top-{i+1} 表: {result['table']} (得分: {result['table_score']:.3f})")
                for col, col_score in result['columns'][:3]:
                    print(f"    - {col}: {col_score:.3f}")

    def export_annotations(self, output_path: str = 'annotations_output.csv'):
        """
        导出注解到CSV文件
        """
        rows = []
        for element, annotation in self.annotations.items():
            rows.append({
                'element': element,
                'description': annotation.get('description', ''),
                'templates': '|'.join(annotation.get('reference_templates', [])[:5]),
                'constraints': str(annotation.get('constraint_templates', []))[:200]
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"\n注解已导出到: {output_path}")


def main():
    # 输入CSV文件路径
    field_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_field_schema.csv'
    table_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_table_schema.csv'
    qa_csv = '/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/source_dataset/processed_qa_data.csv'

    # 创建TE-RAG实例并执行处理
    te_rag = TE_RAG(field_csv, table_csv, qa_csv)
    te_rag.process()

    # 导出注解
    te_rag.export_annotations('/Users/apple/Documents/浙大工作/论文/分层查询数据表/code/base/annotations_output.csv')


if __name__ == "__main__":
    main()
