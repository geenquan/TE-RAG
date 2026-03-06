"""
SQL 角色解析器

使用 sqlglot 进行 AST 解析，准确识别每个字段在 SQL 中的角色
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

    使用 sqlglot 进行 AST 解析，准确识别每个字段在 SQL 中的角色。

    支持的角色：
    - SELECT: 选择列表中的字段
    - WHERE: WHERE 子句中的字段（过滤条件）
    - JOIN: JOIN 条件中的字段
    - GROUP_BY: GROUP BY 子句中的字段
    - ORDER_BY: ORDER BY 子句中的字段
    - HAVING: HAVING 子句中的字段
    - FROM: FROM 子句中的表

    使用方式:
        parser = SQLRoleParser()
        result = parser.parse(sql, expected_table="users")
        print(result.field_roles)  # {field_name: FieldRoleInfo}
    """

    def __init__(self, default_role_weights: Dict[str, float] = None):
        """
        初始化解析器

        Args:
            default_role_weights: 默认角色权重
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

    def parse(self, sql: str, expected_table: str = None) -> 'ParseResult':
        """
        解析 SQL 语句

        Args:
            sql: SQL 语句
            expected_table: 期望的表名（用于字段匹配）

        Returns:
            ParseResult 对象
        """
        if SQLGLOT_AVAILABLE:
            return self._parse_with_sqlglot(sql, expected_table)
        else:
            return self._parse_with_regex(sql, expected_table)

    def _parse_with_sqlglot(self, sql: str, expected_table: str = None) -> 'ParseResult':
        """使用 sqlglot 解析"""
        result = ParseResult(sql=sql, parse_method='sqlglot')

        try:
            # 解析 SQL
            parsed = sqlglot.parse_one(sql, dialect='mysql')

            # 提取表名
            tables = set()
            for table in parsed.find_all(exp.Table):
                table_name = table.name
                tables.add(table_name)
                result.add_table(table_name)

            # 解析 SELECT 列
            for select in parsed.find_all(exp.Select):
                for expr in select.expressions:
                    self._extract_fields_from_expr(expr, 'SELECT', result, expected_table)

            # 解析 WHERE 子句
            where = parsed.find(exp.Where)
            if where:
                self._extract_fields_from_expr(where.this, 'WHERE', result, expected_table)

            # 解析 JOIN 条件
            for join in parsed.find_all(exp.Join):
                if join.args.get('on'):
                    self._extract_fields_from_expr(join.args['on'], 'JOIN', result, expected_table)

            # 解析 GROUP BY
            group = parsed.find(exp.Group)
            if group:
                for expr in group.expressions:
                    self._extract_fields_from_expr(expr, 'GROUP_BY', result, expected_table)

            # 解析 ORDER BY
            order = parsed.find(exp.Order)
            if order:
                for expr in order.expressions:
                    self._extract_fields_from_expr(expr, 'ORDER_BY', result, expected_table)

            # 解析 HAVING
            having = parsed.find(exp.Having)
            if having:
                self._extract_fields_from_expr(having.this, 'HAVING', result, expected_table)

            result.success = True

        except Exception as e:
            result.success = False
            result.error = str(e)
            # 回退到正则解析
            return self._parse_with_regex(sql, expected_table)

        return result

    def _extract_fields_from_expr(self, expr, role: str, result: 'ParseResult',
                                   expected_table: str = None):
        """从表达式中提取字段"""
        # 处理 Column 对象
        for col in expr.find_all(exp.Column):
            field_name = col.name
            table_name = col.table if col.table else expected_table

            if table_name:
                result.add_field_role(table_name, field_name, role)

        # 处理 Identifier（可能是字段名）
        for ident in expr.find_all(exp.Identifier):
            # 只处理不是表名的标识符
            if not isinstance(ident.parent, exp.Table):
                field_name = ident.name
                if expected_table:
                    result.add_field_role(expected_table, field_name, role)

    def _parse_with_regex(self, sql: str, expected_table: str = None) -> 'ParseResult':
        """使用正则表达式解析（回退方案）"""
        result = ParseResult(sql=sql, parse_method='regex')

        try:
            sql_upper = sql.upper()

            # 提取表名
            from_match = re.search(r'FROM\s+(\w+)', sql_upper)
            if from_match:
                result.add_table(from_match.group(1).lower())

            join_match = re.search(r'JOIN\s+(\w+)', sql_upper)
            if join_match:
                result.add_table(join_match.group(1).lower())

            # 简单的角色识别
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

            # 从字段列表推断角色
            # 这是一个简化版本，实际上无法准确识别每个字段的角色
            result.success = True

        except Exception as e:
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

    # 表名集合
    tables: Set[str] = field(default_factory=set)

    # 字段角色信息 {table: {field: FieldRoleInfo}}
    field_roles: Dict[str, Dict[str, FieldRoleInfo]] = field(default_factory=lambda: defaultdict(dict))

    # 简化的标志（用于正则解析）
    has_where: bool = False
    has_join: bool = False
    has_group_by: bool = False
    has_order_by: bool = False
    has_having: bool = False

    def add_table(self, table: str):
        """添加表"""
        self.tables.add(table.lower())

    def add_field_role(self, table: str, field: str, role: str):
        """添加字段角色"""
        table = table.lower()
        field = field.lower()

        if field not in self.field_roles[table]:
            self.field_roles[table][field] = FieldRoleInfo(table=table, field=field)

        self.field_roles[table][field].add_role(role)

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
        table = table.lower()
        field = field.lower()

        role_weights = role_weights or {
            'SELECT': 1.2,
            'WHERE': 2.0,
            'JOIN': 1.8,
            'GROUP_BY': 1.5,
            'ORDER_BY': 1.3,
            'HAVING': 1.4,
        }

        if table in self.field_roles and field in self.field_roles[table]:
            return self.field_roles[table][field].get_max_weight(role_weights)

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
        """获取所有字段 (table, field)"""
        fields = set()
        for table, field_dict in self.field_roles.items():
            for field in field_dict:
                fields.add((table, field))
        return fields


def demo():
    """演示 SQL 角色解析"""
    parser = SQLRoleParser()

    # 测试 SQL
    test_sqls = [
        "SELECT name, age FROM users WHERE id = 1",
        "SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id WHERE o.amount > 100",
        "SELECT department, COUNT(*) FROM employees GROUP BY department ORDER BY COUNT(*) DESC",
        "SELECT category, SUM(price) FROM products WHERE price > 10 GROUP BY category HAVING SUM(price) > 1000",
    ]

    for sql in test_sqls:
        print(f"\nSQL: {sql}")
        result = parser.parse(sql)

        if result.success:
            print(f"  解析方法: {result.parse_method}")
            print(f"  表: {result.tables}")

            for table, fields in result.field_roles.items():
                for field, info in fields.items():
                    print(f"    {table}.{field}: {info.roles}")
                    print(f"      权重: {info.get_max_weight(parser.default_role_weights):.1f}")
        else:
            print(f"  解析失败: {result.error}")


if __name__ == "__main__":
    demo()
