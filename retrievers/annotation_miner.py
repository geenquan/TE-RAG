"""
直接领域注解挖掘（P2-D1，原论文 §3.3 注解生成的"直接证据"子集）

从**当前训练折**的 NLQ–SQL 提取可追溯注解（B 类：历史日志直接证据；
区别于 A 类原始 schema 元数据与 C 类迁移注解——后者属阶段 2）。

注解类型与规则（全部显式、可解释，无外部模型）：
- TERM（业务短语→表列）：
  R1 select_alias——训练 SQL 的 SELECT 输出别名中的术语 ↔ 该输出列
  （别名即训练证据；术语须不在该列名/描述与表名/描述内——元数据已覆盖者
  不重复记，仅留 lexicon_overlap 标记）。状态 confirmed。
  R2 co-occurrence——问题中不在 schema 词表的术语 t 与被引用列 e 共现：
  support(t,e)≥2 且集中度 support(t,e)/Σ_e' support(t,e') ≥ 0.8 → confirmed；
  多元素在 20% 内并列 → ambiguous（记录、不注入）；support=1 → rejected_single。
  不做"整句所有词 ↔ 所有字段"绑定。
- TEMPLATE（表级查询模板）：问题做槽位归一化（日期→#DATE、数字→#NUM、
  ENTITY 类谓词值→#ORG），按表聚合，支持度排序取前 K=5。
- PREDICATE（谓词注解）：训练 WHERE 的 (列, 操作符, 值类型[, 值])；值分类
  DATE/ENTITY/CODE/THRESHOLD/UNKNOWN（规则见 _classify_value）。无 NL 证据
  的代码值只记值共现，不编造业务含义。
- ROLE（列角色先验）：来自 AST 解析的列角色集合（复用 sql_role_parser）。

每条注解：type/target/content/source_qids/rule/support/confidence/status。
索引通道：table_annotation_texts()（confirmed TERM + TEMPLATE，权重固定
field_weights.pattern）；列排序通道：column_bonus()（w_ann 固定 0.5）。
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd

from terag.sql_role_parser import SQLRoleParser, SQLGLOT_AVAILABLE

try:
    import sqlglot
    from sqlglot import exp
except ImportError:  # pragma: no cover
    sqlglot = None

W_ANN = 0.5            # 列排序注解命中加成（固定，运行前声明）
TEMPLATE_TOPK = 5      # 每表注入模板数（固定）
TERM_MIN_SUPPORT = 2   # R2 最小支持
TERM_CONCENTRATION = 0.8
AMBIGUITY_WINDOW = 0.2

# 术语提取停用词（纯功能词/动词；组织与数字由槽位规则处理）
STOP_TERMS = {'的', '是', '有哪些', '多少', '哪些', '什么', '请问', '查询',
              '统计', '显示', '列出', '分析', '怎么', '如何'}

_DATE_RE = re.compile(r'\d{4}年\d{1,2}月|\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?|\d{6}|\d{8}')
_NUM_RE = re.compile(r'\d+')
_CODE_RE = re.compile(r'^[A-Za-z0-9]{1,4}$')
_ORG_COL_RE = re.compile(r'org|name|dept|unit|station|company', re.I)


def _norm_key(table: str, col: str) -> str:
    return f"{table.lower()}.{col.lower()}"


class DirectAnnotationMiner:
    """从当前训练折提取直接注解（见模块 docstring 的规则）"""

    def __init__(self, field_df: pd.DataFrame, table_df: pd.DataFrame,
                 dialect: str = 'mysql'):
        self.field_df = field_df
        self.table_df = table_df
        self.dialect = dialect
        self.schema_columns = {t: set(g['field_name'].astype(str))
                               for t, g in field_df.groupby('table')}
        self.annotations: List[Dict] = []
        self._term_by_key: Dict[str, List[Tuple[str, float]]] = {}
        self._templates_by_table: Dict[str, List[Tuple[str, int]]] = {}
        self._entity_values: List[str] = []

    # ---------------------------------------------------------------- schema 词表

    def _local_lexicon(self, table: str, col: Optional[str] = None) -> set:
        from terag.tokenization import tokenize_text
        words = set()
        trow = self.table_df[self.table_df['table'] == table]
        for _, r in trow.iterrows():
            words |= {w for w in tokenize_text(str(r.get('table_desc', '') or ''), 'chinese')
                      if len(w) > 1}
        cols = self.field_df[self.field_df['table'] == table]
        if col is not None:
            cols = cols[cols['field_name'] == col]
        for _, r in cols.iterrows():
            words.add(str(r['field_name']).lower())
            words |= {w for w in tokenize_text(
                str(r.get('field_name_desc', '') or ''), 'chinese') if len(w) > 1}
        return words

    def _global_lexicon(self) -> set:
        from terag.tokenization import tokenize_text
        words = set()
        for _, r in self.field_df.iterrows():
            words.add(str(r['field_name']).lower())
            words |= {w for w in tokenize_text(
                str(r.get('field_name_desc', '') or ''), 'chinese') if len(w) > 1}
        for _, r in self.table_df.iterrows():
            words |= {w for w in tokenize_text(
                str(r.get('table_desc', '') or ''), 'chinese') if len(w) > 1}
        return words

    # ---------------------------------------------------------------- 挖掘主入口

    def _append(self, ann: Dict) -> None:
        """装配注解并分配稳定 annotation_id（供迁移溯源）。"""
        ann['annotation_id'] = f"a{len(self.annotations):04d}"
        self.annotations.append(ann)

    def mine(self, train_df: pd.DataFrame) -> 'DirectAnnotationMiner':
        from terag.tokenization import tokenize_text
        assert SQLGLOT_AVAILABLE and sqlglot is not None, '注解挖掘需要 sqlglot（AST）'
        parser = SQLRoleParser(dialect=self.dialect)
        glob_lex = self._global_lexicon()

        alias_pairs = defaultdict(lambda: defaultdict(set))  # (t,c) -> term -> {真实出现该别名的 qid}
        alias_meta = defaultdict(lambda: defaultdict(set))   # (t,c) -> term -> {完整别名文本}
        cooc = defaultdict(lambda: defaultdict(set))  # term -> element_key -> {qid}
        templates = defaultdict(lambda: defaultdict(set))  # table -> norm_q -> {qid}
        tmpl_bindings = defaultdict(lambda: defaultdict(list))  # table -> norm_q -> [(element, role)]
        pred_counts = defaultdict(set)        # (t,c,op,vclass[,v]) -> {qid}
        role_sets = defaultdict(set)          # element -> roles
        role_qids = defaultdict(set)
        entity_vals = defaultdict(set)        # value -> {qid}

        for _, row in train_df.iterrows():
            qid = str(row.get('qid') or row.name)
            sql = str(row.get('sql', ''))
            question = str(row.get('question', ''))
            pr = parser.parse(sql, schema_columns=self.schema_columns)
            if not pr.ast_parsed:
                continue
            tables = pr.get_gold_tables()

            # --- R1 别名对齐（局部作用域解析输出列）---
            for t, c in pr.get_all_fields():
                for r_ in pr.field_roles[t][c].roles:
                    role_sets[(t, c)].add(r_)
                role_qids[(t, c)].add(qid)
            for term, target, alias_text in self._alias_terms(sql):
                alias_pairs[target][term].add(qid)
                alias_meta[target][term].add(alias_text)

            # --- R2 共现术语 ---
            q_tokens = [w for w in tokenize_text(question, 'chinese') if len(w) > 1]
            unknown_terms = [w for w in q_tokens
                             if w not in glob_lex and w not in STOP_TERMS
                             and not _NUM_RE.search(w)]
            ref_cols = [k for k in pr.get_all_fields()]
            for term in set(unknown_terms):
                for key in ref_cols:
                    cooc[term][_norm_key(*key)].add(qid)

            # --- 谓词 ---
            for (t, c), op, val, vclass in self._predicates(sql):
                pred_counts[(t, c, op, vclass, val)].add(qid)
                if vclass == 'ENTITY':
                    entity_vals[val].add(qid)

            # --- 模板（含最小绑定结构：该题引用的 (元素, 角色) 列表）---
            norm_q = self._normalize_template(question)
            for t in tables:
                templates[t][norm_q].add(qid)
                binds = sorted({( _norm_key(tt, cc), rr)
                                for (tt, cc) in pr.get_all_fields()
                                for rr in pr.field_roles[tt][cc].roles})
                tmpl_bindings[t][norm_q].append({'qid': qid, 'bindings': binds})

        self._entity_values = sorted(entity_vals.keys())
        # 二次遍历模板（ENTITY 值归一化需先收集实体值）
        templates2 = defaultdict(lambda: defaultdict(set))
        parser2 = SQLRoleParser(dialect=self.dialect)
        for _, row in train_df.iterrows():
            qid = str(row.get('qid') or row.name)
            pr2 = parser2.parse(str(row.get('sql', '')),
                                schema_columns=self.schema_columns)
            if not pr2.ast_parsed:
                continue
            for t in pr2.get_gold_tables():
                templates2[t][self._normalize_template(str(row.get('question', '')))].add(qid)

        # --- 装配注解 ---
        self.annotations = []
        self._annotation_ids = {}
        for (t, c), per_term in alias_pairs.items():
            for term, qids in per_term.items():
                local = self._local_lexicon(t, c)
                lex_overlap = term in local
                # P2-D2 修正：来源/支持度 = 真实出现该别名的 qid 集合
                # （role_qids 只作 ROLE 注解来源）；confidence 为启发式
                # 支持度分（=别名出现题数，非校准概率）
                self._append({
                    'type': 'TERM', 'rule': 'select_alias',
                    'target': _norm_key(t, c), 'content': term,
                    'alias_texts': sorted(alias_meta[(t, c)][term]),
                    'source_qids': sorted(qids), 'support': len(qids),
                    'confidence': f'heuristic:support={len(qids)}',
                    'status': 'confirmed',
                    'lexicon_overlap': lex_overlap})
        for term, per_elem in cooc.items():
            counts = {k: len(v) for k, v in per_elem.items()}
            total = sum(counts.values())
            best = max(counts.values())
            top = [k for k, n in counts.items() if n >= best * (1 - AMBIGUITY_WINDOW)]
            if best < TERM_MIN_SUPPORT:
                status = 'rejected_single'
            elif best / total >= TERM_CONCENTRATION and len(top) == 1:
                status = 'confirmed'
            else:
                status = 'ambiguous'
            qids = sorted(set().union(*per_elem.values()))
            self._append({
                'type': 'TERM', 'rule': 'co_occurrence',
                'target': top[0] if len(top) == 1 and status == 'confirmed' else None,
                'candidates': sorted(counts.items(), key=lambda x: -x[1])[:5],
                'content': term, 'source_qids': qids, 'support': total,
                'confidence': f'heuristic:concentration={round(best / total, 3)}',
                'status': status})
        for (t, c, op, vclass, val), qids in pred_counts.items():
            self._append({
                'type': 'PREDICATE', 'rule': 'where_ast',
                'target': _norm_key(t, c), 'content': {'operator': op,
                                                       'value_class': vclass,
                                                       'value': val},
                'source_qids': sorted(qids), 'support': len(qids),
                'confidence': 'heuristic:ast_exact', 'status': 'confirmed'})
        for (t, c), roles in role_sets.items():
            self._append({
                'type': 'ROLE', 'rule': 'ast_roles',
                'target': _norm_key(t, c), 'content': sorted(roles),
                'source_qids': sorted(role_qids[(t, c)]),
                'support': len(role_qids[(t, c)]),
                'confidence': 'heuristic:ast_exact', 'status': 'confirmed'})
        for t, per_tmpl in templates2.items():
            ranked = sorted(per_tmpl.items(), key=lambda x: (-len(x[1]), x[0]))
            for text, qids in ranked[:TEMPLATE_TOPK]:
                binds = []
                for rec in tmpl_bindings[t].get(text, []):
                    binds.append(rec['bindings'])
                self._append({
                    'type': 'TEMPLATE', 'rule': 'slot_normalization',
                    'target': t, 'content': text,
                    'bindings': binds,
                    'source_qids': sorted(qids), 'support': len(qids),
                    'confidence': f'heuristic:support={len(qids)}',
                    'status': 'confirmed'})

        # 注入视图
        self._term_by_key = defaultdict(list)
        for a in self.annotations:
            if a['type'] == 'TERM' and a['status'] == 'confirmed' and a.get('target'):
                self._term_by_key[a['target']].append((a['content'], a['confidence']))
        self._templates_by_table = {
            t: [(text, len(qids)) for text, qids in
                sorted(per_tmpl.items(), key=lambda x: (-len(x[1]), x[0]))[:TEMPLATE_TOPK]]
            for t, per_tmpl in templates2.items()}
        return self

    # ---------------------------------------------------------------- SQL 局部解析

    def _parse_sql(self, sql):
        for d in (self.dialect, 'hive', 'mysql'):
            try:
                return sqlglot.parse_one(sql, dialect=d), d
            except Exception:  # noqa: BLE001
                continue
        return None, None

    def _scope_sources(self, select):
        """一个 SELECT 的 FROM/JOIN 物理表与别名（简化局部解析）"""
        tables = {}
        frm = select.args.get('from_') or select.args.get('from')
        sources = []
        if frm is not None:
            sources.append(frm.this)
        sources += [j.this for j in (select.args.get('joins') or [])]
        for src in sources:
            if isinstance(src, exp.Table) and src.name:
                name = src.name
                alias = (src.alias or name).lower()
                tables[alias] = name
                tables[name.lower()] = name
        return tables

    def _resolve(self, col, tables, n_physical):
        if col.table:
            return tables.get(col.table.lower())
        if n_physical == 1:
            return next(iter({t for t in tables.values()}))
        return None

    def _alias_terms(self, sql):
        """R1：SELECT 输出别名中的新术语 ↔ 输出列（限可唯一归属的列）。

        返回 [(term, (table, column), alias_text), ...]——alias_text 保留完整
        别名供来源审计（P2-D2：support 必须来自真实出现该别名的题）。
        """
        from terag.tokenization import tokenize_text
        out = []
        tree, _ = self._parse_sql(sql)
        if tree is None:
            return out
        for select in tree.find_all(exp.Select):
            tables = self._scope_sources(select)
            n_phys = len({t for t in tables.values()})
            for e in select.expressions:
                alias = e.alias
                if not alias:
                    continue
                cols = [c for c in e.find_all(exp.Column)
                        if not isinstance(c.this, exp.Star)]
                if len(cols) != 1:
                    continue
                t = self._resolve(cols[0], tables, n_phys)
                if t is None:
                    continue
                local = self._local_lexicon(t, cols[0].name)
                for w in tokenize_text(alias, 'chinese'):
                    if len(w) > 1 and w not in STOP_TERMS \
                            and not _NUM_RE.search(w) and w not in local:
                        out.append((w, (t, cols[0].name), alias))
        return out

    def _predicates(self, sql):
        """WHERE 谓词：(列, 操作符, 值, 值类型)。仅取可唯一归属列的比较条件。"""
        out = []
        tree, _ = self._parse_sql(sql)
        if tree is None:
            return out
        op_map = {exp.EQ: '=', exp.GT: '>', exp.LT: '<', exp.GTE: '>=',
                  exp.LTE: '<=', exp.Like: 'LIKE', exp.In: 'IN',
                  exp.Between: 'BETWEEN'}
        for select in tree.find_all(exp.Select):
            w = select.args.get('where')
            if w is None:
                continue
            tables = self._scope_sources(select)
            n_phys = len({t for t in tables.values()})
            for node in w.find_all(exp.Binary):
                op = op_map.get(type(node))
                if op is None:
                    continue
                cols = [c for c in node.find_all(exp.Column)]
                lits = [l for l in node.find_all(exp.Literal)]
                if len(cols) != 1 or not lits:
                    continue
                t = self._resolve(cols[0], tables, n_phys)
                if t is None:
                    continue
                val = lits[0].this if hasattr(lits[0], 'this') else str(lits[0])
                out.append(((t, cols[0].name), op, str(val),
                            self._classify_value(str(val), cols[0].name, op)))
            for node in w.find_all(exp.Is):
                cols = [c for c in node.find_all(exp.Column)]
                if len(cols) == 1:
                    t = self._resolve(cols[0], tables, n_phys)
                    if t is not None:
                        out.append(((t, cols[0].name), 'IS', 'NULL', 'NULL'))
        return out

    def _classify_value(self, val: str, col: str, op: str) -> str:
        """值分类（声明规则）：DATE/ENTITY/CODE/THRESHOLD/UNKNOWN/NULL"""
        if _DATE_RE.fullmatch(val):
            return 'DATE'
        if re.search(r'[一-鿿]', val) and len(val) >= 3 \
                and _ORG_COL_RE.search(col):
            return 'ENTITY'
        if _CODE_RE.match(val) and op == '=' and re.search(
                r'cls|type|stat|flag|code|level|kind', col, re.I):
            return 'CODE'
        if op in ('>', '<', '>=', '<=') and _NUM_RE.match(val):
            return 'THRESHOLD'
        return 'UNKNOWN'

    def _normalize_template(self, question: str) -> str:
        """槽位归一化：日期→#DATE；ENTITY 值→#ORG；剩余数字→#NUM"""
        q = question
        for v in self._entity_values:
            if v in q:
                q = q.replace(v, '#ORG')
        q = _DATE_RE.sub('#DATE', q)
        q = _NUM_RE.sub('#NUM', q)
        return re.sub(r'\s+', '', q)

    # ---------------------------------------------------------------- 检索通道

    def table_annotation_texts(self, pattern_weight: float) -> Dict[str, List[Tuple[str, float]]]:
        """表级索引注入内容：confirmed TERM（列的）+ 模板文本。"""
        out = defaultdict(list)
        for key, terms in self._term_by_key.items():
            t = key.split('.', 1)[0]
            for term, _conf in terms:
                out[t].append((term, pattern_weight))
        for t, tmpls in self._templates_by_table.items():
            for text, _sup in tmpls:
                out[t].append((text, pattern_weight))
        return dict(out)

    def column_bonus(self, query_tokens, table: str, column: str) -> float:
        """列排序注解命中加成：查询词命中该列的 confirmed TERM。"""
        key = _norm_key(table, column)
        bonus = 0.0
        for term, _conf in self._term_by_key.get(key, []):
            if term in query_tokens:
                bonus += W_ANN
        return bonus

    def summary(self) -> Dict:
        by = defaultdict(lambda: defaultdict(int))
        for a in self.annotations:
            by[a['type']][a['status']] += 1
        return {t: dict(v) for t, v in by.items()}

    def source_qids(self) -> set:
        return {q for a in self.annotations for q in a['source_qids']}
