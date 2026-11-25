"""
SQL 角色解析器

使用 sqlglot 进行 AST 解析，准确识别每个字段在 SQL 中的角色。

支持：
- 多方言（默认 mysql；ODPS/Hive 风格注释与函数自动回退 hive 方言；BIRD 可配 sqlite）
- 多表 SQL（JOIN / 子查询 / 派生表 / CTE），gold tables 取所有物理表
- 作用域别名解析（T1/T2、派生表别名、CTE 名；子查询作用域独立，不全局混淆）
- 无限定列解析（可见表唯一化 / schema 唯一匹配；多表歧义显式记录，不静默取首个）
- 星号三态：SELECT * 与 t.* 按 schema 展开；聚合参数 * 不展开只记录
- 解析完整性独立标记：ast_parsed ≠ 字段全部可解析；unresolved / ambiguous /
  star_expansions 逐项落盘，不返回"成功且零字段"掩盖失败
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

try:
    import sqlglot
    from sqlglot import exp
    SQLGLOT_AVAILABLE = True
except ImportError:
    SQLGLOT_AVAILABLE = False
    print("Warning: sqlglot not available, falling back to regex-based parsing")


class _Unresolved(Exception):
    """列无法解析为物理表列"""


class _Ambiguous(Exception):
    """列在多张可见表中同名（无法唯一归属）"""

    def __init__(self, column: str, candidates):
        super().__init__(column)
        self.column = column
        self.candidates = list(candidates)


@dataclass
class FieldRoleInfo:
    """字段角色信息"""
    table: str
    field: str
    roles: Set[str] = field(default_factory=set)  # SELECT, WHERE, JOIN, GROUP_BY, ORDER_BY, HAVING
    role_weights: Dict[str, float] = field(default_factory=dict)

    def add_role(self, role: str, weight: float = 1.0):
        """添加角色"""
        self.roles.add(role)
        self.role_weights[role] = max(self.role_weights.get(role, 0), weight)

    def get_max_weight(self, role_weights: Dict[str, float]) -> float:
        """获取最大权重"""
        if not self.roles:
            return 1.0
        return max(role_weights.get(r, 1.0) for r in self.roles)


class SQLRoleParser:
    """
    SQL 角色解析器

    使用 sqlglot 进行 AST 解析，按 SELECT 作用域解析每个字段的物理归属与角色。

    支持的角色：
    - SELECT / WHERE / JOIN / GROUP_BY / ORDER_BY / HAVING / FROM

    使用方式:
        parser = SQLRoleParser(dialect='sqlite')
        result = parser.parse(sql, schema_columns={'frpm': {'County Name', ...}})
        print(result.tables)        # 所有物理表
        print(result.field_roles)   # {table_name: {field_name: FieldRoleInfo}}
        print(result.unresolved)    # 未解析项（限定义务：gold 构建必须检查）
        print(result.ambiguous)     # 多表歧义列
    """

    def __init__(self, default_role_weights: Dict[str, float] = None,
                 dialect: str = 'mysql', mode: str = 'ast'):
        """
        初始化解析器

        Args:
            default_role_weights: 默认角色权重
            dialect: SQL 方言（PowerGrid 默认 mysql；ODPS 注释/函数自动回退 hive）
            mode: 'ast'（默认，经 P2-A 验证的 AST 解析）或 'regex'（历史正则
                  回退，仅用于诊断对照；不构成完整解析）
        """
        self.default_role_weights = default_role_weights or {
            'SELECT': 1.2,
            'WHERE': 2.0,
            'JOIN': 1.8,
            'GROUP_BY': 1.5,
            'ORDER_BY': 1.3,
            'HAVING': 1.4,
            'FROM': 1.0,
        }
        self.dialect = dialect
        if mode not in ('ast', 'regex'):
            raise ValueError(f"未知解析模式: {mode}")
        self.mode = mode
        self._scopes_cache: Dict[int, 'SQLRoleParser._Scope'] = {}

    def parse(self, sql: str, expected_table: str = None,
              dialect: str = None,
              schema_columns: Optional[Dict[str, Set[str]]] = None) -> 'ParseResult':
        """
        解析 SQL 语句

        Args:
            sql: SQL 语句
            expected_table: 期望的表名（保留参数；作用域解析不再依赖它猜测归属）
            dialect: SQL 方言（覆盖默认值）
            schema_columns: 物理 schema {table: {column, ...}}，
                            用于无限定列消歧、派生表/CTE 输出匹配与星号展开。
                            完整 gold 构建必须提供；缺省时多表无限定列记为
                            unresolved 而非猜测。

        Returns:
            ParseResult（ast_parsed 表示 AST 解析成功；字段级完整性看
            unresolved / ambiguous / star_expansions）
        """
        if self.mode == 'regex' or not SQLGLOT_AVAILABLE:
            result = self._parse_with_regex(sql, expected_table)
            result.ast_parsed = False
            reason = ('显式 regex 模式（诊断对照）' if self.mode == 'regex'
                      else 'sqlglot-not-installed')
            result.unresolved.append({'item': reason,
                                      'kind': 'parser_missing',
                                      'reason': 'regex 回退，不构成完整 AST 解析'})
            return result

        dialect = dialect or self.dialect

        # 方言链：配置方言 → hive（ODPS 风格注释/函数）→ mysql（BIRD 反引号）
        tried = []
        for d in [dialect] + [x for x in ('hive', 'mysql') if x != dialect]:
            tried.append(d)
            result = self._parse_with_sqlglot(sql, expected_table, d, schema_columns)
            if result.success:
                return result
        result = self._parse_with_regex(sql, expected_table)
        result.ast_parsed = False
        result.unresolved.append(
            {'item': f'sqlglot-failed:{",".join(tried)}', 'kind': 'parse_error',
             'reason': (result.error or '')[:300]})
        return result

    # ------------------------------------------------------------------ 作用域

    class _Scope:
        """一个 SELECT 的作用域：可见的物理表别名、派生表别名与 CTE"""

        def __init__(self, node, parent):
            self.node = node      # exp.Select
            self.parent = parent  # 外层 _Scope（相关子查询可见外层表）
            self.table_aliases: Dict[str, str] = {}       # 别名/表名(小写) -> 物理表名
            self.derived_aliases: Dict[str, object] = {}  # 派生表/CTE 别名 -> 内层查询

    def _build_scopes(self, parsed) -> Dict[int, '_Scope']:
        """为每个 exp.Select 建立作用域（外层关系 + FROM/JOIN 源 + 可见 CTE）"""
        selects = list(parsed.find_all(exp.Select))
        parent_select = {}
        for s in selects:
            p = s.parent
            while p is not None and not isinstance(p, exp.Select):
                p = p.parent
            parent_select[id(s)] = p

        # CTE 定义（WITH 子句）：名称 -> 内层查询
        cte_defs: Dict[str, object] = {}
        for cte in parsed.find_all(exp.CTE):
            cte_defs[cte.alias_or_name.lower()] = cte.this

        scopes: Dict[int, 'SQLRoleParser._Scope'] = {}
        for s in selects:
            parent_scope = scopes.get(id(parent_select[id(s)])) \
                if parent_select.get(id(s)) is not None else None
            sc = self._Scope(s, parent_scope)

            sources = []
            # sqlglot >=30 使用 'from_'（避开 Python 关键字），旧版为 'from'
            frm = s.args.get('from_') or s.args.get('from')
            if frm is not None:
                # From 的全部直接子源（正常只有 this；括号化 JOIN 会拆成多个）
                for v in frm.args.values():
                    vs = v if isinstance(v, list) else [v]
                    for x in vs:
                        if isinstance(x, exp.Expression):
                            sources.append(x)
            for join in (s.args.get('joins') or []):
                sources.append(join.this)
            for src in sources:
                self._register_source(src, sc, cte_defs)
                # FROM 树预序走查（Select 边界截止）：括号化 JOIN / 异常包装下
                # 的别名化 Subquery/Table 也在本层可见
                self._collect_nested_sources(src, sc, cte_defs, depth=0)

            scopes[id(s)] = sc
        return scopes

    def _collect_nested_sources(self, node, sc: '_Scope',
                                cte_defs: Dict[str, object], depth: int):
        """走查 FROM 源树，登记别名化 Subquery/Table；不进入内层 Select。"""
        if node is None or depth > 12:
            return
        if isinstance(node, exp.Select):
            return  # Select 边界：内层作用域由其自身 scope 处理
        if isinstance(node, exp.Table):
            self._register_source(node, sc, cte_defs)
            return
        if isinstance(node, exp.Subquery):
            if node.alias:
                self._register_source(node, sc, cte_defs)
            for child in node.args.values():
                vs = child if isinstance(child, list) else [child]
                for c in vs:
                    if isinstance(c, exp.Expression):
                        self._collect_nested_sources(c, sc, cte_defs, depth + 1)
            return
        # Join / 其他容器节点：继续下探
        for child in node.args.values():
            vs = child if isinstance(child, list) else [child]
            for c in vs:
                if isinstance(c, exp.Expression):
                    self._collect_nested_sources(c, sc, cte_defs, depth + 1)

    def _register_source(self, src, sc: '_Scope', cte_defs: Dict[str, object]):
        """登记一个 FROM/JOIN 源（物理表 / 派生表 / CTE 引用 / 括号化 JOIN）"""
        if src is None:
            return
        if isinstance(src, exp.Table):
            name = src.name
            if not name:
                return
            if name.lower() in cte_defs:
                sc.derived_aliases[name.lower()] = cte_defs[name.lower()]
                return
            alias = (src.alias or name).lower()
            sc.table_aliases[alias] = name
            sc.table_aliases[name.lower()] = name
            return

        if isinstance(src, exp.Subquery):
            alias = (src.alias or '')
            # 递归剥掉匿名括号包装；真别名可能在里层 Subquery 上
            core, depth = src.this, 0
            while isinstance(core, exp.Subquery) and depth < 8:
                if not alias and core.alias:
                    alias = core.alias
                if not core.alias:
                    core = core.this
                    depth += 1
                else:
                    break

            if not alias and isinstance(core, exp.Join):
                # 括号化 JOIN 包装：FROM ( (…) a JOIN (…) b ) —— 透明展开
                self._register_source(core.this, sc, cte_defs)
                for j in (core.args.get('joins') or []):
                    self._register_source(j.this, sc, cte_defs)
                for deep in list(core.find_all(exp.Subquery)):
                    self._register_source(deep, sc, cte_defs)
                for deep in list(core.find_all(exp.Table)):
                    self._register_source(deep, sc, cte_defs)
                return

            inner = core.this if isinstance(core, exp.Subquery) else core
            key = alias.lower() if alias else f'__anon_derived_{len(sc.derived_aliases)}'
            sc.derived_aliases[key] = inner
            return

        if hasattr(src, 'alias'):
            # Union 等集合操作直接作派生源
            alias = (getattr(src, 'alias', '') or '')
            key = alias.lower() if alias else f'__anon_derived_{len(sc.derived_aliases)}'
            sc.derived_aliases[key] = src

    def _visible_physical_tables(self, sc: '_Scope') -> List[str]:
        """作用域链上可见的全部物理表（去重，保持出现序）"""
        out, seen = [], set()
        s = sc
        while s is not None:
            for name in s.table_aliases.values():
                low = name.lower()
                if low not in seen:
                    seen.add(low)
                    out.append(name)
            s = s.parent
        return out

    def _schema_cols(self, table: str, schema_columns) -> Optional[Set[str]]:
        if not schema_columns:
            return None
        cols = schema_columns.get(table) or schema_columns.get(table.lower())
        if cols is None:
            return None
        return {str(c).lower() for c in cols}

    def _scope_for_node(self, node):
        """node 所属最近 SELECT 的作用域"""
        p = node.parent
        while p is not None:
            if isinstance(p, exp.Select):
                sc = self._scopes_cache.get(id(p))
                if sc is not None:
                    return sc
            p = p.parent
        return None

    def _resolve_in_derived(self, inner, column_name: str, schema_columns,
                            depth: int = 0) -> Optional[Set[Tuple[str, str]]]:
        """在派生表/CTE 输出中定位 column_name 对应的物理 (table, column) 集合"""
        if depth > 8:
            return None
        if not isinstance(inner, exp.Select):
            # UNION 等集合操作：在两侧分别找
            found = set()
            for side in (getattr(inner, 'this', None),
                         getattr(inner, 'expression', None)):
                if side is not None:
                    sub = self._resolve_in_derived(side, column_name,
                                                   schema_columns, depth + 1)
                    if sub:
                        found |= sub
            return found or None

        has_star = False
        for e in inner.expressions:
            if list(e.find_all(exp.Star)):
                has_star = True
                continue
            output_name = (e.alias_or_name or '').lower()
            if output_name and output_name == column_name.lower():
                phys = set()
                for col in e.find_all(exp.Column):
                    col_scope = self._scope_for_node(col)
                    if col_scope is None:
                        continue
                    try:
                        phys |= self._resolve_column(col, col_scope, schema_columns)
                    except (_Unresolved, _Ambiguous):
                        continue
                return phys or None
        if has_star:
            # 显式输出列未命中但内层含星号：在星号展开的物理列中找
            phys = set()
            sc = self._scopes_cache.get(id(inner))
            if sc is None:
                return None
            for t in self._visible_physical_tables(sc):
                cols = self._schema_cols(t, schema_columns)
                if cols and column_name.lower() in cols:
                    phys.add((t, column_name))
            return phys or None
        return None

    def _resolve_column(self, col: 'exp.Column', sc: '_Scope',
                        schema_columns) -> Set[Tuple[str, str]]:
        """解析一个列节点 → 物理 (table, column) 集合

        限定列：沿作用域链找别名/派生表/CTE（内层自然遮蔽外层）。
        无限定列（local-first，符合 SQL 名字解析）：
        本层物理表唯一 → 归属；本层 schema 唯一匹配 → 归属；本层歧义 →
        记歧义；本层无匹配 → 尝试本层派生表输出；再向外层作用域重复。

        Raises:
            _Unresolved / _Ambiguous
        """
        qualifier = (col.table or '').lower()
        name = col.name

        if qualifier:
            s = sc
            while s is not None:
                if qualifier in s.table_aliases:
                    return {(s.table_aliases[qualifier], name)}
                if qualifier in s.derived_aliases:
                    phys = self._resolve_in_derived(
                        s.derived_aliases[qualifier], name, schema_columns)
                    if phys:
                        return phys
                    raise _Unresolved(
                        f"限定符 '{qualifier}' 的输出中未找到列 '{name}'")
                s = s.parent
            raise _Unresolved(f"限定符 '{qualifier}' 不是可见表/派生表/CTE")

        s = sc
        while s is not None:
            local = []
            for t in s.table_aliases.values():
                if t not in local:
                    local.append(t)
            if len(local) == 1:
                return {(local[0], name)}
            if schema_columns and len(local) > 1:
                matches = [t for t in local
                           if name.lower() in (self._schema_cols(t, schema_columns) or set())]
                if len(matches) == 1:
                    return {(matches[0], name)}
                if len(matches) > 1:
                    raise _Ambiguous(name, matches)
                # 本层无匹配 → 尝试本层派生表输出，再向外层
            if s.derived_aliases:
                # 逐派生源解析：单源命中（含 union 多侧多归属）→ 返回；
                # 多源命中 → 歧义
                per_source = []
                for inner in s.derived_aliases.values():
                    r = self._resolve_in_derived(inner, name, schema_columns)
                    if r:
                        per_source.append(r)
                if len(per_source) == 1:
                    return per_source[0]
                if len(per_source) > 1:
                    tables_of = {t for r in per_source for t, _ in r}
                    raise _Ambiguous(name, tables_of)
            s = s.parent
        raise _Unresolved(f"无限定列 '{name}' 在作用域链上无法唯一归属")

    # ------------------------------------------------------------------ 主解析

    def _parse_with_sqlglot(self, sql: str, expected_table: str, dialect: str,
                            schema_columns: Optional[Dict[str, Set[str]]]) -> 'ParseResult':
        """使用 sqlglot 解析（作用域树实现）"""
        result = ParseResult(sql=sql, parse_method=f'sqlglot({dialect})')

        try:
            parsed = sqlglot.parse_one(sql, dialect=dialect)
        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error = f"{type(e).__name__}: {str(e)[:300]}"
            return result

        try:
            self._scopes_cache = self._build_scopes(parsed)
            scopes = self._scopes_cache

            # 1. 物理表集合（CTE 名不算物理表；同名表按简名合并）
            physical = set()
            for sc in scopes.values():
                physical.update(sc.table_aliases.values())
            for name in sorted(physical):
                result.add_table(name)

            # 2. 按作用域逐子句提取字段角色
            for sc in scopes.values():
                select = sc.node
                # SELECT 列表（星号三态 + 普通列）
                for e in select.expressions:
                    stars = list(e.find_all(exp.Star))
                    if stars:
                        self._handle_star(stars, 'SELECT', sc, schema_columns, result)
                    for col in e.find_all(exp.Column):
                        if isinstance(col.this, exp.Star):
                            continue
                        self._add_column(col, 'SELECT', sc, schema_columns, result)

                for role, node in (('WHERE', select.args.get('where')),
                                   ('HAVING', select.args.get('having'))):
                    if node is None:
                        continue
                    for col in node.find_all(exp.Column):
                        self._add_column(col, role, sc, schema_columns, result)

                for join in (select.args.get('joins') or []):
                    on = join.args.get('on')
                    if on is not None:
                        for col in on.find_all(exp.Column):
                            self._add_column(col, 'JOIN', sc, schema_columns, result)

                group = select.args.get('group')
                if group is not None:
                    for e in group.expressions:
                        for col in e.find_all(exp.Column):
                            self._add_column(col, 'GROUP_BY', sc, schema_columns, result)

                order = select.args.get('order')
                if order is not None:
                    for ordered in order.expressions:
                        for col in ordered.find_all(exp.Column):
                            self._add_column(col, 'ORDER_BY', sc, schema_columns, result)

            # 语句级 ORDER BY（UNION 外层等无 SELECT 祖先的情形）
            for order in parsed.find_all(exp.Order):
                if order.find_ancestor(exp.Select) is not None:
                    continue
                root_scopes = list(scopes.values())
                if not root_scopes:
                    continue
                for ordered in order.expressions:
                    for col in ordered.find_all(exp.Column):
                        self._add_column(col, 'ORDER_BY', root_scopes[0],
                                         schema_columns, result)

            # 顶层投影集合（clean_projection 口径）：无 SELECT 祖先的最外层
            # SELECT 的输出表达式解析到的物理表列（含星号展开目标；UNION 两侧
            # 均计）。区别于 field_roles 中混合子查询 SELECT 的全部角色列。
            projection = set()
            for sel in list(parsed.find_all(exp.Select)):
                if sel.find_ancestor(exp.Select) is not None:
                    continue
                sc = scopes.get(id(sel))
                if sc is None:
                    continue
                for e in sel.expressions:
                    if list(e.find_all(exp.Star)):
                        agg = e.find_ancestor(exp.AggFunc)
                        if agg is not None:
                            continue  # 聚合星号不是投影列
                        node = e.find(exp.Star).parent
                        qualifier = (node.table or '').lower() \
                            if isinstance(node, exp.Column) else ''
                        if qualifier:
                            s2 = sc
                            while s2 is not None:
                                if qualifier in s2.table_aliases:
                                    projection |= {
                                        (s2.table_aliases[qualifier].lower(), c)
                                        for c in (self._schema_cols(
                                            s2.table_aliases[qualifier],
                                            schema_columns) or set())}
                                    break
                                if qualifier in s2.derived_aliases:
                                    for t in self._inner_physical_tables(
                                            s2.derived_aliases[qualifier]):
                                        projection |= {
                                            (t.lower(), c) for c in
                                            (self._schema_cols(t, schema_columns) or set())}
                                    break
                                s2 = s2.parent
                        else:
                            for t in self._visible_physical_tables(sc):
                                projection |= {
                                    (t.lower(), c) for c in
                                    (self._schema_cols(t, schema_columns) or set())}
                        continue
                    for col in e.find_all(exp.Column):
                        if isinstance(col.this, exp.Star):
                            continue
                        try:
                            for t, f in self._resolve_column(col, sc, schema_columns):
                                projection.add((t.lower(), f.lower()))
                        except (_Unresolved, _Ambiguous):
                            continue
            result.projection_fields = sorted(projection)

            # 3. JOIN 连接表对（物理表名，逗号连接不含）
            result.join_pairs = self._extract_join_pairs(parsed)

            # 4. WHERE 谓词字段
            for sc in scopes.values():
                w = sc.node.args.get('where')
                if w is None:
                    continue
                for col in w.this.find_all(exp.Column):
                    try:
                        for t, f in self._resolve_column(col, sc, schema_columns):
                            result.predicates.append((t, f))
                    except (_Unresolved, _Ambiguous):
                        continue

            # 兼容标志
            result.has_where = any(sc.node.args.get('where') for sc in scopes.values())
            result.has_join = bool(list(parsed.find_all(exp.Join)))
            result.has_group_by = any(sc.node.args.get('group') for sc in scopes.values())
            result.has_order_by = bool(list(parsed.find_all(exp.Order)))
            result.has_having = any(sc.node.args.get('having') for sc in scopes.values())

            result.success = True
            result.ast_parsed = True

        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error = f"{type(e).__name__}: {str(e)[:300]}"

        return result

    # ------------------------------------------------------------------ 星号与列登记

    def _handle_star(self, stars, role: str, sc: '_Scope',
                     schema_columns, result: 'ParseResult'):
        """星号三态：SELECT * / t.* 按 schema 展开；聚合参数 * 只记录不展开"""
        for star in stars:
            agg = star.find_ancestor(exp.AggFunc)
            if agg is not None:
                result.star_expansions.append(
                    {'form': 'aggregate_star', 'qualifier': '', 'tables': [],
                     'expanded_columns': 0,
                     'basis': f'聚合 {agg.sql_name()}(*) 参数星号，不展开为物理列'})
                continue

            qualifier = ''
            node = star.parent
            if isinstance(node, exp.Column):
                qualifier = (node.table or '').lower()

            if qualifier:
                target: Optional[List[str]] = None
                s = sc
                while s is not None and target is None:
                    if qualifier in s.table_aliases:
                        target = [s.table_aliases[qualifier]]
                    elif qualifier in s.derived_aliases:
                        target = self._inner_physical_tables(
                            s.derived_aliases[qualifier])
                    s = s.parent
                if not target:
                    result.unresolved.append(
                        {'item': f'{qualifier}.*', 'kind': 'star_qualifier',
                         'reason': f'限定符 {qualifier!r} 无法解析'})
                    continue
                self._expand_star_tables(target, role, schema_columns, result,
                                         form='table_star', qualifier=qualifier)
            else:
                tables = self._visible_physical_tables(sc)
                self._expand_star_tables(tables, role, schema_columns, result,
                                         form='select_star', qualifier='')

    def _inner_physical_tables(self, inner) -> List[str]:
        """派生表/CTE 内层查询引用的全部物理表（简名，去重）"""
        out, seen = [], set()
        if inner is None:
            return out
        nodes = [inner] if isinstance(inner, exp.Select) else list(inner.find_all(exp.Select))
        for node in nodes:
            sc = self._scopes_cache.get(id(node))
            if sc is None:
                continue
            for name in self._visible_physical_tables(sc):
                if name.lower() not in seen:
                    seen.add(name.lower())
                    out.append(name)
        return out

    def _expand_star_tables(self, tables: List[str], role: str, schema_columns,
                            result: 'ParseResult', form: str, qualifier: str):
        if not schema_columns:
            result.unresolved.append(
                {'item': f'{qualifier or "SELECT"}.*', 'kind': 'star_no_schema',
                 'reason': '星号展开需要 schema_columns，未提供'})
            return
        n = 0
        for t in tables:
            cols = self._schema_cols(t, schema_columns)
            if cols is None:
                result.unresolved.append(
                    {'item': f'{t}.*', 'kind': 'star_table_not_in_schema',
                     'reason': f'表 {t} 不在 schema 中，星号无法展开'})
                continue
            for c in sorted(cols):
                result.add_field_role(t, c, role)
                n += 1
        result.star_expansions.append(
            {'form': form, 'qualifier': qualifier,
             'tables': sorted({t.lower() for t in tables}),
             'expanded_columns': n, 'basis': 'schema_columns 全列展开'})

    def _add_column(self, col: 'exp.Column', role: str, sc: '_Scope',
                    schema_columns, result: 'ParseResult'):
        """解析并登记一个列节点（unresolved / ambiguous 显式记录）。

        列一律按其**最近 SELECT 作用域**解析（外层子句 find_all 会下钻进
        子查询，须防外层作用域重复/错误归属）；找不到时回退子句作用域。
        """
        name = col.name
        if not name:
            return
        scope = self._scope_for_node(col) or sc
        try:
            resolved = self._resolve_column(col, scope, schema_columns)
        except _Ambiguous as amb:
            result.ambiguous.append(
                {'column': name, 'candidates': sorted({c.lower() for c in amb.candidates}),
                 'role': role})
            result.warnings.append(
                f"ambiguous column '{name}' across {sorted({c.lower() for c in amb.candidates})}")
            return
        except _Unresolved as un:
            result.unresolved.append(
                {'item': f"{(col.table + '.') if col.table else ''}{name}",
                 'kind': 'unresolved_column', 'role': role, 'reason': str(un)})
            result.warnings.append(
                f"cannot resolve column '{name}' (qualifier='{col.table}'): {un}")
            return
        for table, fld in resolved:
            if schema_columns is not None:
                cols = self._schema_cols(table, schema_columns)
                if cols is not None and fld.lower() not in cols:
                    result.unresolved.append(
                        {'item': f'{table}.{fld}', 'kind': 'not_in_schema',
                         'role': role,
                         'reason': f'列不在表 {table} 的 schema 中（可能为输出别名/派生列）'})
                    result.warnings.append(f"column '{table}.{fld}' not in schema")
                    continue
            result.add_field_role(table, fld, role)

    def _extract_join_pairs(self, parsed) -> List[Tuple[str, str]]:
        """提取 JOIN 连接的表对（物理表名，逗号连接不含）"""
        pairs = []
        try:
            for join in parsed.find_all(exp.Join):
                names: List[str] = []
                for t in join.find_all(exp.Table):
                    if t.name and t.name not in names:
                        names.append(t.name)
                if len(names) >= 2:
                    for other in names[1:]:
                        pair = (names[0].lower(), other.lower())
                        if pair not in pairs:
                            pairs.append(pair)
        except Exception:  # noqa: BLE001
            pass
        return pairs

    # ------------------------------------------------------------------ 正则回退

    def _parse_with_regex(self, sql: str, expected_table: str = None) -> 'ParseResult':
        """使用正则表达式解析（回退方案；不构成完整 AST 解析，ast_parsed=False）"""
        result = ParseResult(sql=sql, parse_method='regex')

        try:
            sql_upper = sql.upper()

            from_match = re.search(r'FROM\s+([`\w]+)', sql_upper)
            if from_match:
                result.add_table(from_match.group(1).strip('`').lower())

            join_match = re.search(r'JOIN\s+([`\w]+)', sql_upper)
            if join_match:
                result.add_table(join_match.group(1).strip('`').lower())

            if 'WHERE' in sql_upper:
                result.has_where = True
            if 'JOIN' in sql_upper:
                result.has_join = True
            if 'GROUP BY' in sql_upper:
                result.has_group_by = True
            if 'ORDER BY' in sql_upper:
                result.has_order_by = True
            if 'HAVING' in sql_upper:
                result.has_having = True

            result.success = True

        except Exception as e:  # noqa: BLE001
            result.success = False
            result.error = str(e)

        return result

    def get_role_statistics(self, sql_list: List[str]) -> Dict[str, int]:
        """
        统计 SQL 中各角色的频率

        Args:
            sql_list: SQL 语句列表

        Returns:
            角色频率字典
        """
        stats = defaultdict(int)

        for sql in sql_list:
            result = self.parse(sql)
            if result.success:
                for table, fields in result.field_roles.items():
                    for field, info in fields.items():
                        for role in info.roles:
                            stats[role] += 1

        return dict(stats)


@dataclass
class ParseResult:
    """解析结果"""
    sql: str
    success: bool = False
    error: str = ""
    parse_method: str = "unknown"

    # AST 解析成功标志（独立于 success：regex 回退 success=True 但 ast_parsed=False）
    ast_parsed: bool = False

    # 表名集合（小写）
    tables: Set[str] = field(default_factory=set)

    # 字段角色信息 {table: {field: FieldRoleInfo}}（字段保留原始大小写）
    field_roles: Dict[str, Dict[str, FieldRoleInfo]] = field(default_factory=lambda: defaultdict(dict))

    # JOIN 连接表对（真实表名，小写）
    join_pairs: List[Tuple[str, str]] = field(default_factory=list)

    # WHERE 谓词字段 [(table, column), ...]
    predicates: List[Tuple[str, str]] = field(default_factory=list)

    # 解析告警（人类可读摘要；结构化信息见 unresolved/ambiguous/star_expansions）
    warnings: List[str] = field(default_factory=list)

    # 未解析项（gold 构建必须逐项处理；不得视为已覆盖）
    unresolved: List[Dict] = field(default_factory=list)

    # 多表歧义列（无法唯一归属；不得静默填入 gold）
    ambiguous: List[Dict] = field(default_factory=list)

    # 星号处理记录（form: select_star/table_star/aggregate_star）
    star_expansions: List[Dict] = field(default_factory=list)

    # 顶层投影物理列 [(table, column), ...]（clean_projection 口径；
    # 仅最外层 SELECT 输出，不含子查询内部 SELECT 列）
    projection_fields: List[Tuple[str, str]] = field(default_factory=list)

    # 简化的标志（用于正则解析回退）
    has_where: bool = False
    has_join: bool = False
    has_group_by: bool = False
    has_order_by: bool = False
    has_having: bool = False

    def add_table(self, table: str):
        """添加表"""
        if table:
            self.tables.add(table.lower())

    def add_field_role(self, table: str, field: str, role: str):
        """添加字段角色"""
        table = table.lower()
        if not field:
            return

        if field not in self.field_roles[table]:
            self.field_roles[table][field] = FieldRoleInfo(table=table, field=field)

        self.field_roles[table][field].add_role(role)

    def find_field_info(self, table: str, field: str) -> Optional[FieldRoleInfo]:
        """大小写不敏感地查找字段角色信息"""
        table = table.lower() if table else ''
        field_lower = field.lower() if field else ''
        table_dict = self.field_roles.get(table, {})
        if field_lower in table_dict:
            return table_dict[field_lower]
        for f, info in table_dict.items():
            if f.lower() == field_lower:
                return info
        return None

    def get_field_weight(self, table: str, field: str,
                         role_weights: Dict[str, float] = None) -> float:
        """
        获取字段权重

        Args:
            table: 表名
            field: 字段名
            role_weights: 角色权重字典

        Returns:
            字段权重
        """
        role_weights = role_weights or {
            'SELECT': 1.2,
            'WHERE': 2.0,
            'JOIN': 1.8,
            'GROUP_BY': 1.5,
            'ORDER_BY': 1.3,
            'HAVING': 1.4,
        }

        info = self.find_field_info(table, field)
        if info is not None:
            return info.get_max_weight(role_weights)

        # 正则解析的回退逻辑
        weight = 1.0
        if self.has_where:
            weight = max(weight, 2.0)
        if self.has_join:
            weight = max(weight, 1.8)
        if self.has_group_by:
            weight = max(weight, 1.5)
        if self.has_order_by:
            weight = max(weight, 1.3)

        return weight

    def get_all_fields(self) -> Set[Tuple[str, str]]:
        """获取所有字段 (table, field)，table 为小写、field 保留原始大小写"""
        fields = set()
        for table, field_dict in self.field_roles.items():
            for field in field_dict:
                fields.add((table, field))
        return fields

    def get_gold_columns(self) -> List[str]:
        """获取 gold 列（table.column，用于多表评估）"""
        return sorted(f"{t}.{f}" for t, f in self.get_all_fields())

    def get_gold_tables(self) -> List[str]:
        """获取 gold 表（小写）"""
        return sorted(self.tables)


def demo():
    """演示 SQL 角色解析"""
    parser = SQLRoleParser(dialect='sqlite')

    schema = {
        'schools': {'CDSCode', 'Street', 'City'},
        'satscores': {'cds', 'AvgScrRead'},
        'frpm': {'Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name'},
    }

    # 测试 SQL（覆盖典型场景）
    test_sqls = [
        # 单表 + 反引号列名
        "SELECT `Free Meal Count (K-12)` FROM frpm WHERE `County Name` = 'Alameda'",
        # JOIN + 别名
        "SELECT T2.Street, T2.City FROM schools AS T2 JOIN satscores AS T1 ON T2.CDSCode = T1.cds",
        # 聚合 + 排序 + LIMIT
        ("SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm "
         "WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / "
         "`Enrollment (K-12)`) DESC LIMIT 1"),
        # 子查询
        "SELECT name FROM schools WHERE CDSCode IN (SELECT cds FROM satscores WHERE AvgScrRead > 500)",
    ]

    for sql in test_sqls:
        print(f"\nSQL: {sql[:80]}...")
        result = parser.parse(sql, dialect='sqlite', schema_columns=schema)

        if result.success:
            print(f"  解析方法: {result.parse_method}")
            print(f"  gold tables: {result.get_gold_tables()}")
            print(f"  gold columns: {result.get_gold_columns()}")
            if result.warnings:
                print(f"  warnings: {result.warnings}")
        else:
            print(f"  解析失败: {result.error}")


if __name__ == "__main__":
    demo()
