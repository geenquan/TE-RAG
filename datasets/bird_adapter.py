"""
BIRD 数据集适配器

将官方 BIRD 数据（dev.json / dev_tables.json / dev_databases）转换为实验框架的
标准化格式（table_df / field_df / qa_df），与 PowerGrid 实验接口保持一致。

核心约定：
- 表唯一 ID: "{db_id}.{table_name}"（如 california_schools.frpm），db 列保存 db_id
- field_name 必须使用 column_names_original（Gold SQL 使用真实物理字段名）
- Gold table / column 从 Gold SQL 自动解析（sqlglot, sqlite 方言 + 别名/子查询处理），
  并对照 dev_tables.json 校验，不匹配的记录到解析错误清单，绝不 silent accept
- evidence 是否拼接进查询文本由配置 use_evidence 控制
"""

import json
import os
import zipfile
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from datasets.base_adapter import BaseDatasetAdapter
from terag.config import BirdDatasetConfig
from terag.sql_role_parser import SQLRoleParser
from terag.tokenization import tokenize_text


class BirdDatasetAdapter(BaseDatasetAdapter):
    """BIRD 数据集适配器"""

    def __init__(self, bird_config: BirdDatasetConfig, seed: int = 42,
                 source_split: Optional[str] = None):
        """
        初始化适配器

        Args:
            bird_config: BIRD 数据集配置
            seed: 随机种子（数据划分可复现）
            source_split: 使用的官方 split（默认取配置中的 source_split）
        """
        self.config = bird_config
        self.seed = seed
        self.source_split = source_split or bird_config.source_split
        self.split_config = bird_config.get_source_split_config(self.source_split)

        # 原始数据
        self.queries: List[Dict] = []
        self.db_schemas: Dict[str, Dict] = {}          # db_id -> dev_tables.json 单库条目
        self.db_descriptions: Dict[str, Dict] = {}     # db_id -> {table: {col: desc dict}}

        # 解析器（SQLite 方言）
        self.parser = SQLRoleParser(dialect=bird_config.sql_dialect)

        # 解析错误清单
        self.parse_errors: List[Dict] = []

        # 简单表名（小写） -> canonical 表 ID 映射，按 db 分组
        self._table_id_maps: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------
    # 数据加载
    # ------------------------------------------------------------------

    def ensure_databases_extracted(self):
        """检查 databases 目录，若只有 zip 则自动解压"""
        databases_dir = self.split_config.databases_dir
        if not databases_dir or os.path.isdir(databases_dir):
            return

        zip_path = databases_dir.rstrip('/') + '.zip'
        if os.path.exists(zip_path):
            extract_to = os.path.dirname(databases_dir)
            print(f"未找到 {databases_dir}，自动解压 {zip_path} ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_to)
            print(f"解压完成: {databases_dir}")
        else:
            raise FileNotFoundError(
                f"BIRD databases 目录不存在且未找到 zip: {databases_dir}")

    def load_queries(self) -> List[Dict]:
        """加载 dev.json 查询数据"""
        path = self.split_config.json
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"BIRD 查询文件不存在: {path}（请检查 data.bird.{self.source_split}.json 配置）")

        with open(path, 'r', encoding='utf-8') as f:
            self.queries = json.load(f)
        print(f"加载 BIRD 查询: {len(self.queries)} 条（split={self.source_split}）")
        return self.queries

    def load_schema(self) -> Dict:
        """加载 dev_tables.json schema 数据"""
        path = self.split_config.tables_json
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"BIRD schema 文件不存在: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            tables = json.load(f)

        self.db_schemas = {t['db_id']: t for t in tables}
        print(f"加载 BIRD schema: {len(self.db_schemas)} 个数据库")

        # 构建简单表名 -> canonical ID 映射
        for db_id, schema in self.db_schemas.items():
            self._table_id_maps[db_id] = {
                t.lower(): self.canonical_table_id(db_id, t)
                for t in schema['table_names_original']
            }

        return self.db_schemas

    def load_database_descriptions(self) -> Dict:
        """
        加载各库 database_description/<table>.csv 字段描述

        Returns:
            {db_id: {table_name_original: {original_column_name: {desc 字段}}}}
        """
        self.db_descriptions = {}
        databases_dir = self.split_config.databases_dir

        for db_id, schema in self.db_schemas.items():
            desc_dir = os.path.join(databases_dir, db_id, 'database_description')
            db_desc = {}

            if os.path.isdir(desc_dir):
                for table_name in schema['table_names_original']:
                    csv_path = os.path.join(desc_dir, f"{table_name}.csv")
                    if not os.path.exists(csv_path):
                        continue
                    try:
                        # BIRD 描述文件带 BOM（utf-8-sig），少数文件为 cp1252/latin-1 编码
                        try:
                            desc_df = pd.read_csv(csv_path, encoding='utf-8-sig')
                        except UnicodeDecodeError:
                            desc_df = pd.read_csv(csv_path, encoding='cp1252')
                        table_desc = {}
                        for _, row in desc_df.iterrows():
                            col = row.get('original_column_name')
                            if pd.isna(col) or not str(col).strip():
                                continue
                            table_desc[str(col).strip()] = {
                                'column_name': '' if pd.isna(row.get('column_name')) else str(row.get('column_name')),
                                'column_description': '' if pd.isna(row.get('column_description')) else str(row.get('column_description')).strip(),
                                'data_format': '' if pd.isna(row.get('data_format')) else str(row.get('data_format')).strip(),
                                'value_description': '' if pd.isna(row.get('value_description')) else str(row.get('value_description')).strip(),
                            }
                        db_desc[table_name] = table_desc
                    except Exception as e:
                        print(f"  Warning: 读取描述文件失败 {csv_path}: {e}")

            self.db_descriptions[db_id] = db_desc

        n_desc = sum(len(t) for t in self.db_descriptions.values())
        print(f"加载数据库字段描述: {n_desc} 张表"
              f"（use_database_description={self.config.use_database_description}）")
        return self.db_descriptions

    # ------------------------------------------------------------------
    # Schema 转换
    # ------------------------------------------------------------------

    def canonical_table_id(self, db_id: str, table_name: str) -> str:
        """生成唯一表 ID: prefix_table_with_db 时为 {db_id}.{table}，否则为 {table}"""
        if self.config.schema.prefix_table_with_db:
            return f"{db_id}.{table_name}"
        return table_name

    def _schema_columns_by_db(self, db_id: str) -> Dict[str, Set[str]]:
        """获取某库 {简单表名: {物理列名}}（供 SQL 解析消歧用）"""
        schema = self.db_schemas[db_id]
        columns = defaultdict(set)
        for table_idx, col_name in schema['column_names_original']:
            if table_idx < 0:
                continue  # 跳过全局 '*'
            table_name = schema['table_names_original'][table_idx]
            columns[table_name].add(col_name)
        return dict(columns)

    def _build_field_desc(self, db_id: str, table_name: str, col_original: str,
                          col_nl_name: str) -> str:
        """
        组装字段描述

        优先级: database_description.column_description > BIRD column_names 自然语言名
        可选拼接 value_description / data_format（由配置控制）
        """
        desc_entry = (self.db_descriptions.get(db_id, {})
                      .get(table_name, {})
                      .get(col_original, {}))

        parts = []
        if self.config.use_database_description:
            col_desc = desc_entry.get('column_description', '')
            if col_desc:
                parts.append(col_desc)
            data_format = desc_entry.get('data_format', '')
            if data_format:
                parts.append(f"[format: {data_format}]")
        if not parts and col_nl_name and col_nl_name != col_original:
            # 无描述文件时回退到 BIRD 自然语言列名
            parts.append(col_nl_name)
        if self.config.use_value_description:
            value_desc = desc_entry.get('value_description', '')
            if value_desc:
                parts.append(f"Value: {value_desc}")
        return ' '.join(parts)

    def build_table_dataframe(self) -> pd.DataFrame:
        """构建标准化表 schema DataFrame"""
        rows = []

        for db_id, schema in sorted(self.db_schemas.items()):
            table_names = schema['table_names_original']
            table_names_nl = schema['table_names']
            columns_by_table = self._schema_columns_by_db(db_id)

            for idx, table_name in enumerate(table_names):
                table_id = self.canonical_table_id(db_id, table_name)
                cols = sorted(columns_by_table.get(table_name, []))

                # 表描述：优先 BIRD 自然语言表名
                table_desc = table_names_nl[idx] if idx < len(table_names_nl) else table_name
                if not table_desc:
                    table_desc = table_name

                # 合成 DDL（与 PowerGrid schema 列格式保持一致）
                col_lines = []
                for col in cols:
                    col_lines.append(f"  {col}")
                ddl = (f"CREATE TABLE {table_id} (\n" + ",\n".join(col_lines) + "\n);")

                rows.append({
                    'table': table_id,
                    'db': db_id,
                    'table_desc': table_desc,
                    'field': '|'.join(cols),
                    'schema': ddl,
                })

        df = pd.DataFrame(rows)
        print(f"构建 BIRD 表 schema: {len(df)} 张表")
        return df

    def build_field_dataframe(self) -> pd.DataFrame:
        """构建标准化字段 schema DataFrame"""
        rows = []
        schema_cfg = self.config.schema

        for db_id, schema in sorted(self.db_schemas.items()):
            table_names = schema['table_names_original']
            column_names_original = schema['column_names_original']
            column_names = schema['column_names']
            column_types = schema['column_types']
            primary_keys = schema.get('primary_keys', [])
            foreign_keys = schema.get('foreign_keys', [])

            # 主键列索引集合
            pk_cols = set()
            for pk in primary_keys:
                if isinstance(pk, list):
                    pk_cols.update(pk)
                else:
                    pk_cols.add(pk)

            # 外键: 列索引 -> 参考列索引
            fk_map = {}
            for fk_col, fk_ref in foreign_keys:
                fk_map[fk_col] = fk_ref

            for col_idx, (table_idx, col_original) in enumerate(column_names_original):
                if table_idx < 0:
                    continue
                table_name = table_names[table_idx]
                table_id = self.canonical_table_id(db_id, table_name)
                col_nl = column_names[col_idx][1] if col_idx < len(column_names) else col_original
                col_type = column_types[col_idx] if col_idx < len(column_types) else 'text'

                field_desc = self._build_field_desc(db_id, table_name, col_original, col_nl)

                # PK/FK 非破坏性 metadata
                is_pk = col_idx in pk_cols
                fk_target = ''
                if col_idx in fk_map:
                    ref_idx = fk_map[col_idx]
                    if 0 <= ref_idx < len(column_names_original):
                        ref_table_idx, ref_col = column_names_original[ref_idx]
                        if 0 <= ref_table_idx < len(table_names):
                            fk_target = f"{table_names[ref_table_idx]}.{ref_col}"

                if schema_cfg.include_pk_fk:
                    if is_pk:
                        field_desc = (field_desc + " [primary key]").strip()
                    if fk_target:
                        field_desc = (field_desc + f" [foreign key -> {fk_target}]").strip()

                rows.append({
                    'table': table_id,
                    'db': db_id,
                    'field_name': col_original,
                    'field_name_desc': field_desc,
                    'field_type': col_type,
                    'is_primary_key': is_pk,
                    'foreign_key_target': fk_target,
                })

        df = pd.DataFrame(rows)
        print(f"构建 BIRD 字段 schema: {len(df)} 个字段")
        return df

    # ------------------------------------------------------------------
    # Gold SQL 解析
    # ------------------------------------------------------------------

    def parse_gold_sql(self, item: Dict) -> Dict:
        """
        解析单条 Gold SQL，提取 gold tables / gold columns

        Returns:
            {
                'parse_ok': bool,
                'gold_tables': [canonical_table_id, ...],
                'gold_fields': [[canonical_table_id, column_name], ...],
                'gold_joins': [[table, table], ...],      # 简单表名
                'gold_predicates': ["table.column", ...], # 简单表名.列名
                'errors': [错误信息], 'warnings': [告警]
            }
        """
        db_id = item['db_id']
        sql = item.get('SQL', '')
        result = {
            'parse_ok': False,
            'gold_tables': [],
            'gold_fields': [],
            'gold_joins': [],
            'gold_predicates': [],
            'errors': [],
            'warnings': [],
        }

        id_map = self._table_id_maps.get(db_id, {})
        schema_columns = self._schema_columns_by_db(db_id)
        # schema 列名大小写规范化映射: (table_lower, col_lower) -> schema 原始列名
        schema_col_lookup = {}
        for t, cols in schema_columns.items():
            for c in cols:
                schema_col_lookup[(t.lower(), str(c).lower())] = c

        parsed = self.parser.parse(
            sql, dialect=self.config.sql_dialect, schema_columns=schema_columns)

        if not parsed.success:
            result['errors'].append(f"SQL parse failed ({parsed.parse_method}): {parsed.error}")
            return result

        # gold 表: 简单表名（小写） -> canonical ID，并校验是否存在于 schema
        unknown_tables = []
        for simple_table in parsed.get_gold_tables():
            if simple_table in id_map:
                canonical = id_map[simple_table]
                if canonical not in result['gold_tables']:
                    result['gold_tables'].append(canonical)
            else:
                unknown_tables.append(simple_table)
        if unknown_tables:
            result['errors'].append(
                f"parsed tables not in schema: {unknown_tables}")

        # gold 列: (简单表名, SQL 中的列名) -> (canonical 表 ID, schema 原始列名)
        for table_simple, col in sorted(parsed.get_all_fields()):
            canonical = id_map.get(table_simple)
            if canonical is None:
                continue
            schema_col = schema_col_lookup.get((table_simple, col.lower()), col)
            pair = [canonical, schema_col]
            if pair not in result['gold_fields']:
                result['gold_fields'].append(pair)

        result['gold_joins'] = [list(p) for p in parsed.join_pairs]
        result['gold_predicates'] = [f"{t}.{c}" for t, c in parsed.predicates]
        result['warnings'] = list(parsed.warnings)

        result['parse_ok'] = bool(result['gold_tables']) and not unknown_tables
        if not result['gold_tables']:
            result['errors'].append("no gold tables extracted")
        return result

    # ------------------------------------------------------------------
    # QA 数据
    # ------------------------------------------------------------------

    def build_query_text(self, item: Dict) -> str:
        """
        构建检索查询文本

        use_evidence=True 时拼接 evidence（实验输出必须记录该开关）
        """
        question = str(item.get('question', '')).strip()
        evidence = str(item.get('evidence', '') or '').strip()
        if self.config.use_evidence and evidence:
            return f"{question} {evidence}"
        return question

    def build_qa_dataframe(self) -> pd.DataFrame:
        """构建标准化 QA DataFrame（含结构化 gold 标注）"""
        if not self.db_schemas:
            self.load_schema()
        if not self.queries:
            self.load_queries()
        if self.config.use_database_description and not self.db_descriptions:
            self.load_database_descriptions()

        rows = []
        n_parse_ok = 0

        for item in self.queries:
            gold = self.parse_gold_sql(item)
            if gold['parse_ok']:
                n_parse_ok += 1

            # 解析错误 / 告警统一记录（不 silent accept）
            if gold['errors'] or gold['warnings']:
                self.parse_errors.append({
                    'question_id': item.get('question_id'),
                    'db_id': item.get('db_id'),
                    'sql': item.get('SQL', ''),
                    'errors': gold['errors'],
                    'warnings': gold['warnings'],
                    'parsed_tables': gold['gold_tables'],
                    'parsed_columns': [f"{t}.{c}" for t, c in gold['gold_fields']],
                })

            # 简单表名展示形式（legacy 兼容字段）
            simple_tables = [t.split('.')[-1] for t in gold['gold_tables']]
            field_display = '|'.join(
                f"{t.split('.')[-1]}.{c}" for t, c in gold['gold_fields'])

            rows.append({
                'question': self.build_query_text(item),
                'table': '|'.join(gold['gold_tables']),
                'field': field_display,
                'sql': item.get('SQL', ''),
                'db_id': item.get('db_id', ''),
                'evidence': str(item.get('evidence', '') or ''),
                'difficulty': item.get('difficulty', ''),
                'question_id': item.get('question_id'),
                'question_orig': str(item.get('question', '')),
                'gold_tables': gold['gold_tables'],
                'gold_fields': gold['gold_fields'],
                'gold_joins': gold['gold_joins'],
                'gold_predicates': gold['gold_predicates'],
                'parse_ok': gold['parse_ok'],
            })

        qa_df = pd.DataFrame(rows)
        total = len(qa_df)
        print(f"构建 BIRD QA 数据: {total} 条, SQL 解析成功 {n_parse_ok} 条"
              f"（成功率 {n_parse_ok / total * 100:.1f}%）")
        n_multi = sum(1 for r in rows if len(r['gold_tables']) > 1)
        print(f"  多表查询: {n_multi} 条, 单表查询: {total - n_multi} 条")
        return qa_df

    # ------------------------------------------------------------------
    # Within-database 数据划分
    # ------------------------------------------------------------------

    def split_within_database(self, qa_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        同库内划分 train/dev/test（历史查询与测试问题共享同一数据库 schema）

        对查询数不足 min_queries_per_db 的库：整库跳过并打印日志（不 silent drop）

        Returns:
            (train_df, dev_df, test_df, split_info)
        """
        split_cfg = self.config.split
        assert abs(split_cfg.train_ratio + split_cfg.dev_ratio + split_cfg.test_ratio - 1.0) < 1e-6, \
            "划分比例之和必须为 1"

        rng = np.random.RandomState(self.seed)
        train_idx, dev_idx, test_idx = [], [], []
        skipped_dbs = {}

        for db_id in sorted(qa_df['db_id'].unique()):
            db_rows = qa_df[qa_df['db_id'] == db_id]
            n = len(db_rows)

            if n < split_cfg.min_queries_per_db:
                skipped_dbs[db_id] = n
                print(f"  Warning: 数据库 {db_id} 仅 {n} 条查询 "
                      f"< min_queries_per_db={split_cfg.min_queries_per_db}，整库跳过")
                continue

            indices = db_rows.index.to_numpy()
            permuted = indices[rng.permutation(n)]

            n_train = int(n * split_cfg.train_ratio)
            n_dev = int(n * split_cfg.dev_ratio)

            train_idx.extend(permuted[:n_train])
            dev_idx.extend(permuted[n_train:n_train + n_dev])
            test_idx.extend(permuted[n_train + n_dev:])

        train_df = qa_df.loc[sorted(train_idx)].reset_index(drop=True)
        dev_df = qa_df.loc[sorted(dev_idx)].reset_index(drop=True)
        test_df = qa_df.loc[sorted(test_idx)].reset_index(drop=True)

        print(f"\nBIRD within-database 划分 (seed={self.seed}, strategy={split_cfg.strategy}):")
        print(f"  训练集: {len(train_df)} 条 ({len(train_df) / len(qa_df) * 100:.1f}%)")
        print(f"  验证集: {len(dev_df)} 条 ({len(dev_df) / len(qa_df) * 100:.1f}%)")
        print(f"  测试集: {len(test_df)} 条 ({len(test_df) / len(qa_df) * 100:.1f}%)")

        split_info = {
            'strategy': split_cfg.strategy,
            'seed': self.seed,
            'train_ratio': split_cfg.train_ratio,
            'dev_ratio': split_cfg.dev_ratio,
            'test_ratio': split_cfg.test_ratio,
            'min_queries_per_db': split_cfg.min_queries_per_db,
            'train_size': len(train_df),
            'dev_size': len(dev_df),
            'test_size': len(test_df),
            'train_dbs': sorted(train_df['db_id'].unique().tolist()),
            'dev_dbs': sorted(dev_df['db_id'].unique().tolist()),
            'test_dbs': sorted(test_df['db_id'].unique().tolist()),
            'skipped_dbs': skipped_dbs,
        }
        return train_df, dev_df, test_df, split_info

    # ------------------------------------------------------------------
    # 完整准备流程
    # ------------------------------------------------------------------

    def prepare(self) -> Dict[str, pd.DataFrame]:
        """
        执行完整数据准备流程

        Returns:
            {'table_df', 'field_df', 'qa_df', 'train_df', 'dev_df', 'test_df'}
        """
        print("=" * 60)
        print(f"BIRD 数据准备 (source_split={self.source_split}, seed={self.seed})")
        print("=" * 60)

        self.ensure_databases_extracted()
        self.load_schema()
        self.load_queries()
        self.load_database_descriptions()

        table_df = self.build_table_dataframe()
        field_df = self.build_field_dataframe()
        qa_df = self.build_qa_dataframe()

        train_df, dev_df, test_df, split_info = self.split_within_database(qa_df)

        return {
            'table_df': table_df,
            'field_df': field_df,
            'qa_df': qa_df,
            'train_df': train_df,
            'dev_df': dev_df,
            'test_df': test_df,
            'split_info': split_info,
        }
