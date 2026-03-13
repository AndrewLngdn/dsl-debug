"""Tests for the DSL interpreter."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from dsl_debug.interpreter import Interpreter, parse, ParseError, format_table


@pytest.fixture
def interp():
    return Interpreter()


@pytest.fixture
def sales_data():
    return {
        "sales": [
            {"id": 1, "region": "west", "amount": 100, "status": "completed"},
            {"id": 2, "region": "east", "amount": 200, "status": "completed"},
            {"id": 3, "region": "west", "amount": 150, "status": "pending"},
            {"id": 4, "region": "east", "amount": 300, "status": "completed"},
            {"id": 5, "region": "north", "amount": 250, "status": "completed"},
            {"id": 6, "region": "west", "amount": 200, "status": "completed"},
        ]
    }


@pytest.fixture
def join_data():
    return {
        "orders": [
            {"order_id": 1, "customer_id": 10, "total": 100},
            {"order_id": 2, "customer_id": 20, "total": 200},
            {"order_id": 3, "customer_id": 10, "total": 150},
        ],
        "customers": [
            {"id": 10, "name": "Alice", "tier": "gold"},
            {"id": 20, "name": "Bob", "tier": "silver"},
            {"id": 30, "name": "Carol", "tier": "bronze"},
        ],
    }


class TestParser:
    def test_parse_simple(self):
        code = '''
x = load("data")
  |> filter(status == "active")
emit x
'''
        assignments, emits = parse(code)
        assert len(assignments) == 1
        assert len(emits) == 1
        assert emits[0].variable == "x"

    def test_parse_no_emit(self):
        code = 'x = load("data")'
        with pytest.raises(ParseError, match="No 'emit' statement"):
            parse(code)

    def test_parse_comment_and_blank_lines(self):
        code = '''
-- This is a comment
x = load("data")

-- Another comment

emit x
'''
        assignments, emits = parse(code)
        assert len(assignments) == 1

    def test_parse_multiple_pipes(self):
        code = '''
result = load("data")
  |> filter(status == "active")
  |> select(name, amount)
  |> sort_by(amount, desc)
  |> take(5)
emit result
'''
        assignments, emits = parse(code)
        assert len(assignments) == 1
        assert len(assignments[0].pipes) == 4

    def test_parse_join(self):
        code = '''
a = load("orders")
b = load("customers")
result = join(a, b, on: customer_id == id)
emit result
'''
        assignments, emits = parse(code)
        assert len(assignments) == 3

    def test_parse_error_bad_syntax(self):
        code = '''
this is not valid syntax
emit x
'''
        with pytest.raises(ParseError, match="Unexpected syntax"):
            parse(code)


class TestExecution:
    def test_load_and_filter(self, interp, sales_data):
        code = '''
x = load("sales")
  |> filter(status == "completed")
emit x
'''
        result = interp.run(code, sales_data)
        assert result.success
        assert len(result.data) == 5
        assert all(r["status"] == "completed" for r in result.data)

    def test_select(self, interp, sales_data):
        code = '''
x = load("sales")
  |> select(region, amount)
emit x
'''
        result = interp.run(code, sales_data)
        assert result.success
        assert set(result.data[0].keys()) == {"region", "amount"}

    def test_group_by_aggregate(self, interp, sales_data):
        code = '''
x = load("sales")
  |> filter(status == "completed")
  |> group_by(region)
  |> aggregate(total: sum(amount), count: count())
emit x
'''
        result = interp.run(code, sales_data)
        assert result.success
        assert len(result.data) == 3
        for row in result.data:
            assert "total" in row
            assert "count" in row

    def test_sort_by_desc(self, interp, sales_data):
        code = '''
x = load("sales")
  |> sort_by(amount, desc)
emit x
'''
        result = interp.run(code, sales_data)
        assert result.success
        amounts = [r["amount"] for r in result.data]
        assert amounts == sorted(amounts, reverse=True)

    def test_take(self, interp, sales_data):
        code = '''
x = load("sales")
  |> take(3)
emit x
'''
        result = interp.run(code, sales_data)
        assert result.success
        assert len(result.data) == 3

    def test_compute(self, interp, sales_data):
        code = '''
x = load("sales")
  |> compute(doubled: amount * 2)
emit x
'''
        result = interp.run(code, sales_data)
        assert result.success
        for row in result.data:
            assert row["doubled"] == row["amount"] * 2

    def test_join(self, interp, join_data):
        code = '''
o = load("orders")
c = load("customers")
result = join(o, c, on: customer_id == id)
emit result
'''
        result = interp.run(code, join_data)
        assert result.success
        assert len(result.data) == 3
        for row in result.data:
            assert "name" in row
            assert "total" in row

    def test_full_pipeline(self, interp, sales_data):
        code = '''
sales = load("sales")
  |> filter(status == "completed")
  |> select(region, amount)

totals = sales
  |> group_by(region)
  |> aggregate(revenue: sum(amount), count: count())
  |> sort_by(revenue, desc)
  |> take(3)

emit totals
'''
        result = interp.run(code, sales_data)
        assert result.success
        assert len(result.data) <= 3
        revenues = [r["revenue"] for r in result.data]
        assert revenues == sorted(revenues, reverse=True)


class TestAggregateCorrectness:
    """Aggregate functions must produce deterministic, correct values."""

    def test_avg(self, interp, sales_data):
        code = '''
x = load("sales")
  |> filter(status == "completed")
  |> group_by(region)
  |> aggregate(avg_amount: avg(amount))
emit x
'''
        result = interp.run(code, sales_data)
        assert result.success
        by_region = {r["region"]: r["avg_amount"] for r in result.data}
        assert by_region["west"] == 150.0
        assert by_region["east"] == 250.0
        assert by_region["north"] == 250.0

    def test_avg_determinism(self, interp):
        """avg must produce identical output across runs."""
        datasets = {"data": [{"x": 1}, {"x": 2}, {"x": 3}]}
        code = '''
x = load("data")
  |> group_by(x)
  |> aggregate(val: avg(x))
emit x
'''
        r1 = interp.run(code, datasets)
        r2 = interp.run(code, datasets)
        assert r1.data == r2.data


class TestErrorHandling:
    def test_missing_dataset(self, interp):
        code = '''
x = load("nonexistent")
emit x
'''
        result = interp.run(code, {})
        assert not result.success
        assert "not found" in result.error

    def test_missing_column(self, interp, sales_data):
        code = '''
x = load("sales")
  |> filter(nonexistent == "foo")
emit x
'''
        result = interp.run(code, sales_data)
        assert not result.success
        assert "not found" in result.error

    def test_aggregate_without_group_by(self, interp, sales_data):
        code = '''
x = load("sales")
  |> aggregate(total: sum(amount))
emit x
'''
        result = interp.run(code, sales_data)
        assert not result.success
        assert "group_by" in result.error


class TestInspect:
    def test_inspect_node(self, interp, sales_data):
        code = '''
raw = load("sales")
filtered = raw
  |> filter(status == "completed")
result = filtered
  |> take(2)
emit result
'''
        result = interp.inspect(code, sales_data, "filtered")
        assert result.success
        assert all(r["status"] == "completed" for r in result.data)
        assert len(result.data) > 2  # Not yet truncated by take

    def test_inspect_nonexistent_node(self, interp, sales_data):
        code = '''
x = load("sales")
emit x
'''
        result = interp.inspect(code, sales_data, "nonexistent")
        assert not result.success
        assert "not found" in result.error


class TestFormatTable:
    def test_format_basic(self):
        data = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        output = format_table(data)
        assert "a" in output
        assert "b" in output
        assert "hello" in output
        assert "2 rows" in output

    def test_format_empty(self):
        assert "empty" in format_table([])

    def test_format_truncation(self):
        data = [{"x": i} for i in range(50)]
        output = format_table(data, max_rows=10)
        assert "40 more rows" in output
