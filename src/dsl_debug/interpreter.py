"""DSL interpreter for a DAG-based dataflow language.

The language operates on tabular data (list of dicts) with pipe-based operations.
Supports: load, filter, select, group_by+aggregate, join, compute, sort_by, take, emit.

This interpreter has zero external dependencies — it uses only Python stdlib.
"""

from __future__ import annotations

import re
import operator
import copy
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

@dataclass
class LoadNode:
    dataset_name: str

@dataclass
class PipeOp:
    """A single pipe operation like filter(...), select(...), etc."""
    op: str
    args: dict[str, Any]

@dataclass
class Assignment:
    target: str
    source: Any  # LoadNode, JoinNode, or str (variable reference)
    pipes: list[PipeOp] = field(default_factory=list)
    line_num: int = 0

@dataclass
class JoinNode:
    left: str
    right: str
    left_key: str
    right_key: str

@dataclass
class EmitNode:
    variable: str
    line_num: int = 0

@dataclass
class Result:
    success: bool
    data: list[dict] | None = None
    error: str | None = None
    line_num: int | None = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Comparison operators for filter/join conditions
CMP_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}

# Regex patterns for parsing DSL syntax
RE_COMMENT = re.compile(r"^\s*--.*$")
RE_EMIT = re.compile(r"^\s*emit\s+(\w+)\s*$")
RE_ASSIGN = re.compile(r"^\s*(\w+)\s*=\s*(.+)$")
RE_LOAD = re.compile(r'^load\(\s*"([^"]+)"\s*\)$')
RE_JOIN = re.compile(
    r"^join\(\s*(\w+)\s*,\s*(\w+)\s*,\s*on:\s*(\w+)\s*(==|!=|>|<|>=|<=)\s*(\w+)\s*\)$"
)
RE_PIPE = re.compile(r"\|>")
RE_FILTER = re.compile(
    r'^filter\(\s*(\w+)\s*(==|!=|>|<|>=|<=)\s*(?:"([^"]*)"|(\d+(?:\.\d+)?)|(\w+))\s*\)$'
)
RE_SELECT = re.compile(r"^select\((.+)\)$")
RE_GROUP_BY = re.compile(r"^group_by\((.+)\)$")
RE_AGGREGATE = re.compile(r"^aggregate\((.+)\)$")
RE_COMPUTE = re.compile(r"^compute\((.+)\)$")
RE_SORT_BY = re.compile(r"^sort_by\(\s*(\w+)\s*(?:,\s*(asc|desc))?\s*\)$")
RE_TAKE = re.compile(r"^take\(\s*(\d+)\s*\)$")

# Aggregate function pattern: name: func(col) or name: count()
RE_AGG_FUNC = re.compile(r"(\w+)\s*:\s*(sum|count|avg|min|max)\(\s*(\w*)\s*\)")

# Compute expression pattern: name: expr
RE_COMPUTE_EXPR = re.compile(r"(\w+)\s*:\s*(.+)")


def _parse_value(s: str) -> Any:
    """Try to parse a string as int, float, or leave as string."""
    try:
        return int(s)
    except (ValueError, TypeError):
        pass
    try:
        return float(s)
    except (ValueError, TypeError):
        pass
    return s


class ParseError(Exception):
    def __init__(self, message: str, line_num: int | None = None):
        self.line_num = line_num
        super().__init__(f"Line {line_num}: {message}" if line_num else message)


class DSLRuntimeError(Exception):
    def __init__(self, message: str, node_name: str | None = None, line_num: int | None = None):
        self.node_name = node_name
        self.line_num = line_num
        prefix = f"Node '{node_name}'" if node_name else f"Line {line_num}"
        super().__init__(f"{prefix}: {message}")


def parse(code: str) -> tuple[list[Assignment], list[EmitNode]]:
    """Parse DSL code into a list of assignments and emit statements."""
    if not isinstance(code, str):
        if isinstance(code, list):
            code = "\n".join(str(line) for line in code)
        else:
            code = str(code) if code else ""

    assignments: list[Assignment] = []
    emits: list[EmitNode] = []

    lines = code.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        line_num = i + 1

        # Skip empty lines and comments
        if not line or RE_COMMENT.match(line):
            i += 1
            continue

        # Check for emit
        m = RE_EMIT.match(line)
        if m:
            emits.append(EmitNode(variable=m.group(1), line_num=line_num))
            i += 1
            continue

        # Must be an assignment — collect continuation lines (lines starting with |>)
        m = RE_ASSIGN.match(line)
        if not m:
            raise ParseError(f"Unexpected syntax: {line}", line_num)

        target = m.group(1)
        rhs = m.group(2).strip()

        # Collect continuation lines
        while i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith("|>"):
                rhs += " " + next_line
                i += 1
            else:
                break

        # Parse the RHS
        assignment = _parse_rhs(target, rhs, line_num)
        assignments.append(assignment)
        i += 1

    if not emits:
        raise ParseError("No 'emit' statement found")

    return assignments, emits


def _parse_rhs(target: str, rhs: str, line_num: int) -> Assignment:
    """Parse the right-hand side of an assignment."""
    # Split on pipe operators
    parts = RE_PIPE.split(rhs)
    parts = [p.strip() for p in parts]

    source_str = parts[0]
    pipe_strs = parts[1:]

    # Parse source
    m = RE_LOAD.match(source_str)
    if m:
        source = LoadNode(dataset_name=m.group(1))
    else:
        m = RE_JOIN.match(source_str)
        if m:
            source = JoinNode(
                left=m.group(1), right=m.group(2),
                left_key=m.group(3), right_key=m.group(5),
            )
        elif re.match(r"^\w+$", source_str):
            source = source_str  # variable reference
        else:
            raise ParseError(f"Cannot parse source: {source_str}", line_num)

    pipes = [_parse_pipe_op(ps, line_num) for ps in pipe_strs]
    return Assignment(target=target, source=source, pipes=pipes, line_num=line_num)


def _parse_pipe_op(s: str, line_num: int) -> PipeOp:
    """Parse a single pipe operation."""
    # filter
    m = RE_FILTER.match(s)
    if m:
        col = m.group(1)
        op = m.group(2)
        # Value: string literal, number, or column reference
        if m.group(3) is not None:
            val = m.group(3)  # string literal
        elif m.group(4) is not None:
            val = _parse_value(m.group(4))  # number
        else:
            val = {"col_ref": m.group(5)}  # column reference
        return PipeOp(op="filter", args={"col": col, "cmp": op, "val": val})

    # select
    m = RE_SELECT.match(s)
    if m:
        cols = [c.strip() for c in m.group(1).split(",")]
        return PipeOp(op="select", args={"cols": cols})

    # group_by
    m = RE_GROUP_BY.match(s)
    if m:
        cols = [c.strip() for c in m.group(1).split(",")]
        return PipeOp(op="group_by", args={"cols": cols})

    # aggregate
    m = RE_AGGREGATE.match(s)
    if m:
        agg_str = m.group(1)
        aggs = []
        for am in RE_AGG_FUNC.finditer(agg_str):
            aggs.append({
                "name": am.group(1),
                "func": am.group(2),
                "col": am.group(3) if am.group(3) else None,
            })
        if not aggs:
            raise ParseError(f"Cannot parse aggregate functions: {agg_str}", line_num)
        return PipeOp(op="aggregate", args={"aggs": aggs})

    # compute
    m = RE_COMPUTE.match(s)
    if m:
        expr_str = m.group(1)
        cm = RE_COMPUTE_EXPR.match(expr_str)
        if not cm:
            raise ParseError(f"Cannot parse compute expression: {expr_str}", line_num)
        return PipeOp(op="compute", args={"name": cm.group(1), "expr": cm.group(2).strip()})

    # sort_by
    m = RE_SORT_BY.match(s)
    if m:
        return PipeOp(op="sort_by", args={"col": m.group(1), "order": m.group(2) or "asc"})

    # take
    m = RE_TAKE.match(s)
    if m:
        return PipeOp(op="take", args={"n": int(m.group(1))})

    raise ParseError(f"Unknown pipe operation: {s}", line_num)


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

def _topo_sort(assignments: list[Assignment]) -> list[Assignment]:
    """Topological sort of assignments based on variable dependencies."""
    name_to_node = {a.target: a for a in assignments}
    visited: set[str] = set()
    result: list[Assignment] = []
    visiting: set[str] = set()

    def visit(name: str):
        if name in visited:
            return
        if name in visiting:
            raise DSLRuntimeError(f"Circular dependency involving '{name}'")
        visiting.add(name)
        node = name_to_node.get(name)
        if node is None:
            return  # external (dataset) or unknown — will error at runtime
        # Find dependencies
        deps = _get_deps(node)
        for d in deps:
            visit(d)
        visiting.remove(name)
        visited.add(name)
        result.append(node)

    for a in assignments:
        visit(a.target)

    return result


def _get_deps(a: Assignment) -> list[str]:
    """Get variable names that this assignment depends on."""
    if isinstance(a.source, str):
        return [a.source]
    if isinstance(a.source, JoinNode):
        return [a.source.left, a.source.right]
    return []


def _exec_filter(data: list[dict], args: dict) -> list[dict]:
    col = args["col"]
    cmp_op = CMP_OPS[args["cmp"]]
    val = args["val"]

    result = []
    for row in data:
        if col not in row:
            raise DSLRuntimeError(f"Column '{col}' not found. Available: {list(row.keys())}")
        row_val = row[col]
        if isinstance(val, dict) and "col_ref" in val:
            cmp_val = row.get(val["col_ref"])
        else:
            cmp_val = val
        # Type coercion for comparison
        try:
            if cmp_op(row_val, cmp_val):
                result.append(row)
        except TypeError:
            # Try string comparison
            if cmp_op(str(row_val), str(cmp_val)):
                result.append(row)
    return result


def _exec_select(data: list[dict], args: dict) -> list[dict]:
    cols = args["cols"]
    result = []
    for row in data:
        new_row = {}
        for c in cols:
            if c not in row:
                raise DSLRuntimeError(f"Column '{c}' not found. Available: {list(row.keys())}")
            new_row[c] = row[c]
        result.append(new_row)
    return result


def _exec_group_by_aggregate(data: list[dict], group_cols: list[str], aggs: list[dict]) -> list[dict]:
    # Group rows
    groups: dict[tuple, list[dict]] = {}
    for row in data:
        key = tuple(row.get(c) for c in group_cols)
        groups.setdefault(key, []).append(row)

    result = []
    for key, rows in groups.items():
        new_row = dict(zip(group_cols, key))
        for agg in aggs:
            func = agg["func"]
            col = agg["col"]
            name = agg["name"]
            if func == "count":
                new_row[name] = len(rows)
            elif func == "sum":
                new_row[name] = sum(r.get(col, 0) for r in rows)
            elif func == "avg":
                vals = [r.get(col, 0) for r in rows]
                new_row[name] = sum(vals) / len(vals) if vals else 0
            elif func == "min":
                new_row[name] = min(r.get(col, 0) for r in rows)
            elif func == "max":
                new_row[name] = max(r.get(col, 0) for r in rows)
        result.append(new_row)
    return result


def _exec_join(left_data: list[dict], right_data: list[dict], left_key: str, right_key: str) -> list[dict]:
    result = []
    for lr in left_data:
        lv = lr.get(left_key)
        for rr in right_data:
            rv = rr.get(right_key)
            if lv == rv:
                merged = {**lr, **rr}
                result.append(merged)
    return result


def _eval_compute_expr(expr: str, row: dict) -> Any:
    """Evaluate a simple arithmetic expression with column references."""
    # Tokenize: split on operators while keeping them
    tokens = re.split(r"(\s*[+\-*/]\s*)", expr)
    values = []
    ops = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in "+-*/":
            ops.append(token)
        else:
            # Try as number first
            try:
                values.append(float(token) if "." in token else int(token))
            except ValueError:
                # Column reference
                if token not in row:
                    raise DSLRuntimeError(f"Column '{token}' not found in compute. Available: {list(row.keys())}")
                val = row[token]
                if not isinstance(val, (int, float)):
                    raise DSLRuntimeError(f"Column '{token}' is not numeric (got {type(val).__name__}: {val})")
                values.append(val)

    if not values:
        raise DSLRuntimeError(f"Empty compute expression: {expr}")

    # Two-pass evaluation: multiplication/division first, then addition/subtraction
    for priority_ops in [{"*", "/"}, {"+", "-"}]:
        new_values = [values[0]]
        new_ops = []
        for j, op in enumerate(ops):
            if op in priority_ops:
                left = new_values.pop()
                right = values[j + 1]
                if op == "+":
                    new_values.append(left + right)
                elif op == "-":
                    new_values.append(left - right)
                elif op == "*":
                    new_values.append(left * right)
                elif op == "/":
                    if right == 0:
                        raise DSLRuntimeError("Division by zero")
                    new_values.append(left / right)
            else:
                new_values.append(values[j + 1])
                new_ops.append(op)
        values = new_values
        ops = new_ops

    return values[0]


def _exec_compute(data: list[dict], args: dict) -> list[dict]:
    name = args["name"]
    expr = args["expr"]
    result = []
    for row in data:
        new_row = dict(row)
        new_row[name] = _eval_compute_expr(expr, row)
        result.append(new_row)
    return result


def _exec_sort_by(data: list[dict], args: dict) -> list[dict]:
    col = args["col"]
    reverse = args["order"] == "desc"
    try:
        # (is_none, value) — nulls sort last
        return sorted(data, key=lambda r: (r.get(col) is None, r.get(col, 0)), reverse=reverse)
    except TypeError:
        return sorted(data, key=lambda r: str(r.get(col, "")), reverse=reverse)


def _exec_take(data: list[dict], args: dict) -> list[dict]:
    return data[:args["n"]]


PIPE_EXECUTORS = {
    "filter": _exec_filter,
    "select": _exec_select,
    "compute": _exec_compute,
    "sort_by": _exec_sort_by,
    "take": _exec_take,
}


class Interpreter:
    """Execute DSL code against provided datasets."""

    def run(self, code: str, datasets: dict[str, list[dict]]) -> Result:
        """Execute DSL code and return the emitted result."""
        try:
            assignments, emits = parse(code)
        except ParseError as e:
            return Result(success=False, error=str(e), line_num=e.line_num)

        env, err = self._execute_assignments(assignments, datasets)
        if err is not None:
            return err

        # Get emitted variable
        emit = emits[0]
        if emit.variable not in env:
            return Result(
                success=False,
                error=f"Variable '{emit.variable}' not defined",
                line_num=emit.line_num,
            )

        return Result(success=True, data=env[emit.variable])

    def inspect(self, code: str, datasets: dict[str, list[dict]], node_name: str) -> Result:
        """Execute up to and including the named node, return its output."""
        try:
            assignments, emits = parse(code)
        except ParseError as e:
            return Result(success=False, error=str(e), line_num=e.line_num)

        env, err = self._execute_assignments(assignments, datasets, stop_at=node_name)
        if err is not None:
            return err

        if node_name not in env:
            available = [a.target for a in assignments]
            return Result(
                success=False,
                error=f"Node '{node_name}' not found. Available nodes: {available}",
            )

        return Result(success=True, data=env[node_name])

    def _execute_assignments(
        self,
        assignments: list[Assignment],
        datasets: dict[str, list[dict]],
        stop_at: str | None = None,
    ) -> tuple[dict[str, list[dict]], Result | None]:
        """Parse, topo-sort, and execute assignments. Returns (env, error_or_None).

        If stop_at is given, execution halts after the named node is computed.
        """
        try:
            sorted_assignments = _topo_sort(assignments)
        except DSLRuntimeError as e:
            return {}, Result(success=False, error=str(e))

        env: dict[str, list[dict]] = {}

        try:
            for a in sorted_assignments:
                data = self._resolve_source(a, env, datasets)
                data = self._apply_pipes(a, data)
                env[a.target] = data
                if stop_at is not None and a.target == stop_at:
                    break
        except DSLRuntimeError as e:
            return env, Result(success=False, error=str(e), line_num=e.line_num)
        except Exception as e:
            return env, Result(success=False, error=f"Execution error: {e}")

        return env, None

    def get_node_names(self, code: str) -> list[str]:
        """Return all variable names defined in the code."""
        try:
            assignments, _ = parse(code)
            return [a.target for a in assignments]
        except ParseError:
            return []

    def _resolve_source(self, a: Assignment, env: dict, datasets: dict) -> list[dict]:
        """Resolve the source of an assignment to actual data."""
        if isinstance(a.source, LoadNode):
            name = a.source.dataset_name
            if name not in datasets:
                raise DSLRuntimeError(
                    f"Dataset '{name}' not found. Available: {list(datasets.keys())}",
                    node_name=a.target, line_num=a.line_num,
                )
            return copy.deepcopy(datasets[name])

        if isinstance(a.source, JoinNode):
            j = a.source
            for var in (j.left, j.right):
                if var not in env:
                    raise DSLRuntimeError(
                        f"Variable '{var}' not defined",
                        node_name=a.target, line_num=a.line_num,
                    )
            return _exec_join(env[j.left], env[j.right], j.left_key, j.right_key)

        if isinstance(a.source, str):
            if a.source not in env:
                raise DSLRuntimeError(
                    f"Variable '{a.source}' not defined",
                    node_name=a.target, line_num=a.line_num,
                )
            return copy.deepcopy(env[a.source])

        raise DSLRuntimeError(f"Unknown source type: {type(a.source)}", node_name=a.target)

    def _apply_pipes(self, a: Assignment, data: list[dict]) -> list[dict]:
        """Apply pipe operations sequentially."""
        pending_group_by = None

        for pipe in a.pipes:
            if pipe.op == "group_by":
                pending_group_by = pipe.args["cols"]
                continue

            if pipe.op == "aggregate":
                if pending_group_by is None:
                    raise DSLRuntimeError(
                        "aggregate() requires a preceding group_by()",
                        node_name=a.target, line_num=a.line_num,
                    )
                data = _exec_group_by_aggregate(data, pending_group_by, pipe.args["aggs"])
                pending_group_by = None
                continue

            if pending_group_by is not None:
                raise DSLRuntimeError(
                    "group_by() must be followed by aggregate()",
                    node_name=a.target, line_num=a.line_num,
                )

            executor = PIPE_EXECUTORS.get(pipe.op)
            if executor is None:
                raise DSLRuntimeError(
                    f"Unknown operation: {pipe.op}",
                    node_name=a.target, line_num=a.line_num,
                )

            try:
                data = executor(data, pipe.args)
            except DSLRuntimeError:
                raise
            except Exception as e:
                raise DSLRuntimeError(
                    f"Error in {pipe.op}(): {e}",
                    node_name=a.target, line_num=a.line_num,
                )

        if pending_group_by is not None:
            raise DSLRuntimeError(
                "group_by() must be followed by aggregate()",
                node_name=a.target, line_num=a.line_num,
            )

        return data


def format_table(data: list[dict], max_rows: int = 20) -> str:
    """Format data as a readable text table."""
    if not data:
        return "(empty table)"

    cols = list(data[0].keys())
    # Calculate column widths
    widths = {c: len(str(c)) for c in cols}
    display_data = data[:max_rows]
    for row in display_data:
        for c in cols:
            widths[c] = max(widths[c], len(str(row.get(c, ""))))

    # Header
    header = " | ".join(str(c).ljust(widths[c]) for c in cols)
    separator = "-+-".join("-" * widths[c] for c in cols)
    lines = [header, separator]

    # Rows
    for row in display_data:
        line = " | ".join(str(row.get(c, "")).ljust(widths[c]) for c in cols)
        lines.append(line)

    if len(data) > max_rows:
        lines.append(f"... ({len(data) - max_rows} more rows)")

    lines.append(f"({len(data)} rows total)")
    return "\n".join(lines)
