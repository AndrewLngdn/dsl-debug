"""Random dataset and program generator for the DSL debugger.

Generates realistic tabular datasets across multiple domains, creates correct
DAG programs that process them, then uses BugInjector to create buggy variants.

This module uses only Python stdlib — no external dependencies required.
"""

from __future__ import annotations

import random
import copy
from dataclasses import dataclass
from typing import Any

from .interpreter import Interpreter, Result
from .bug_injector import BugInjector, BugInfo, verify_bug
from .environment import Example


# ---------------------------------------------------------------------------
# Domain definitions
# ---------------------------------------------------------------------------

@dataclass
class ColumnDef:
    name: str
    dtype: str  # "str", "int", "float"
    values: list[Any] | tuple[int, int] | tuple[float, float]  # enum values or (min, max) range

@dataclass
class DomainDef:
    name: str
    dataset_name: str
    columns: list[ColumnDef]
    # Optional second dataset for joins
    secondary_name: str | None = None
    secondary_columns: list[ColumnDef] | None = None
    join_left_key: str | None = None
    join_right_key: str | None = None


# Five built-in domains covering common business data patterns
DOMAINS = [
    DomainDef(
        name="sales",
        dataset_name="sales",
        columns=[
            ColumnDef("id", "int", (1, 1000)),
            ColumnDef("region", "str", ["west", "east", "north", "south", "central"]),
            ColumnDef("amount", "int", (50, 500)),
            ColumnDef("status", "str", ["completed", "pending", "cancelled", "refunded"]),
            ColumnDef("rep_id", "int", (1, 20)),
            ColumnDef("quarter", "str", ["Q1", "Q2", "Q3", "Q4"]),
        ],
        secondary_name="reps",
        secondary_columns=[
            ColumnDef("rep_id", "int", (1, 20)),
            ColumnDef("name", "str", ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]),
            ColumnDef("tier", "str", ["junior", "senior", "lead"]),
        ],
        join_left_key="rep_id",
        join_right_key="rep_id",
    ),
    DomainDef(
        name="employees",
        dataset_name="employees",
        columns=[
            ColumnDef("emp_id", "int", (100, 999)),
            ColumnDef("dept", "str", ["engineering", "sales", "marketing", "ops", "hr"]),
            ColumnDef("salary", "int", (40000, 150000)),
            ColumnDef("level", "str", ["junior", "mid", "senior", "staff", "principal"]),
            ColumnDef("years", "int", (0, 20)),
            ColumnDef("rating", "int", (1, 5)),
        ],
        secondary_name="departments",
        secondary_columns=[
            ColumnDef("dept_id", "str", ["engineering", "sales", "marketing", "ops", "hr"]),
            ColumnDef("budget", "int", (100000, 5000000)),
            ColumnDef("head_count_limit", "int", (10, 100)),
        ],
        join_left_key="dept",
        join_right_key="dept_id",
    ),
    DomainDef(
        name="products",
        dataset_name="products",
        columns=[
            ColumnDef("product_id", "int", (1, 500)),
            ColumnDef("category", "str", ["electronics", "clothing", "food", "books", "toys"]),
            ColumnDef("price", "float", (5.0, 200.0)),
            ColumnDef("stock", "int", (0, 1000)),
            ColumnDef("rating", "float", (1.0, 5.0)),
            ColumnDef("supplier_id", "int", (1, 10)),
        ],
        secondary_name="suppliers",
        secondary_columns=[
            ColumnDef("supplier_id", "int", (1, 10)),
            ColumnDef("supplier_name", "str", ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne"]),
            ColumnDef("country", "str", ["US", "CN", "DE", "JP", "KR"]),
        ],
        join_left_key="supplier_id",
        join_right_key="supplier_id",
    ),
    DomainDef(
        name="orders",
        dataset_name="orders",
        columns=[
            ColumnDef("order_id", "int", (1000, 9999)),
            ColumnDef("customer_id", "int", (1, 50)),
            ColumnDef("total", "int", (10, 1000)),
            ColumnDef("status", "str", ["shipped", "delivered", "returned", "processing"]),
            ColumnDef("channel", "str", ["web", "mobile", "store", "phone"]),
            ColumnDef("items", "int", (1, 20)),
        ],
        secondary_name="customers",
        secondary_columns=[
            ColumnDef("customer_id", "int", (1, 50)),
            ColumnDef("segment", "str", ["premium", "standard", "basic"]),
            ColumnDef("city", "str", ["NYC", "LA", "Chicago", "Houston", "Phoenix"]),
        ],
        join_left_key="customer_id",
        join_right_key="customer_id",
    ),
    DomainDef(
        name="students",
        dataset_name="students",
        columns=[
            ColumnDef("student_id", "int", (1, 200)),
            ColumnDef("major", "str", ["cs", "math", "physics", "biology", "english"]),
            ColumnDef("gpa", "float", (1.0, 4.0)),
            ColumnDef("year", "int", (1, 4)),
            ColumnDef("credits", "int", (0, 160)),
            ColumnDef("status", "str", ["active", "graduated", "on_leave", "suspended"]),
        ],
        secondary_name="courses",
        secondary_columns=[
            ColumnDef("course_dept", "str", ["cs", "math", "physics", "biology", "english"]),
            ColumnDef("course_name", "str", ["Intro", "Advanced", "Seminar", "Lab", "Workshop"]),
            ColumnDef("difficulty", "int", (1, 5)),
        ],
        join_left_key="major",
        join_right_key="course_dept",
    ),
]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _generate_column_value(col: ColumnDef, rng: random.Random) -> Any:
    if col.dtype == "str":
        return rng.choice(col.values)
    elif col.dtype == "int":
        return rng.randint(col.values[0], col.values[1])
    elif col.dtype == "float":
        return round(rng.uniform(col.values[0], col.values[1]), 2)


def _generate_dataset(columns: list[ColumnDef], n_rows: int, rng: random.Random) -> list[dict]:
    """Generate a random dataset with given columns."""
    return [
        {col.name: _generate_column_value(col, rng) for col in columns}
        for _ in range(n_rows)
    ]


def _generate_secondary_dataset(columns: list[ColumnDef], rng: random.Random) -> list[dict]:
    """Generate a secondary (lookup) dataset — one row per unique key value."""
    key_col = columns[0]
    if key_col.dtype == "str":
        keys = list(key_col.values)
    else:
        # Cap at 15 unique keys so lookup tables stay small (realistic dimension table size)
        keys = list(range(key_col.values[0], min(key_col.values[0] + 15, key_col.values[1] + 1)))

    data = []
    for key_val in keys:
        row = {key_col.name: key_val}
        for col in columns[1:]:
            row[col.name] = _generate_column_value(col, rng)
        data.append(row)
    return data


# ---------------------------------------------------------------------------
# Program generation
# ---------------------------------------------------------------------------

class ProgramBuilder:
    """Build a random valid DSL program for a given domain."""

    def __init__(self, domain: DomainDef, rng: random.Random):
        self.domain = domain
        self.rng = rng
        self.lines: list[str] = []
        self.available_columns: dict[str, list[str]] = {}

    def build(self, difficulty: str = "medium") -> str:
        """Generate a random valid DSL program."""
        self.lines = []

        if difficulty == "easy":
            n_pipes = self.rng.randint(2, 3)
            use_join = False
        elif difficulty == "medium":
            n_pipes = self.rng.randint(3, 5)
            use_join = self.domain.secondary_name is not None and self.rng.random() < 0.5
        else:  # hard
            n_pipes = self.rng.randint(4, 7)
            use_join = self.domain.secondary_name is not None and self.rng.random() < 0.8

        # Load primary dataset
        main_var = self.domain.dataset_name
        self.lines.append(f'-- Load {self.domain.name} data')
        self.lines.append(f'{main_var} = load("{self.domain.dataset_name}")')
        all_cols = [c.name for c in self.domain.columns]
        self.available_columns[main_var] = all_cols

        # Determine string and numeric columns for filtering
        str_cols = [c for c in self.domain.columns if c.dtype == "str"]
        num_cols = [c for c in self.domain.columns if c.dtype in ("int", "float")]

        # Build a pipeline of operations
        current_var = main_var
        pipes_added = 0

        # Possibly add a filter
        if pipes_added < n_pipes and str_cols and self.rng.random() < 0.7:
            col = self.rng.choice(str_cols)
            val = self.rng.choice(col.values)
            self.lines.append(f'')
            new_var = f"filtered_{self.domain.dataset_name}"
            self.lines.append(f'{new_var} = {current_var}')
            self.lines.append(f'  |> filter({col.name} == "{val}")')
            self.available_columns[new_var] = list(self.available_columns[current_var])
            current_var = new_var
            pipes_added += 1

        # Possibly add numeric filter
        if pipes_added < n_pipes and num_cols and self.rng.random() < 0.4:
            col = self.rng.choice(num_cols)
            op = self.rng.choice([">", "<", ">=", "<="])
            if col.dtype == "int":
                mid = (col.values[0] + col.values[1]) // 2
                val = self.rng.randint(col.values[0], mid)
            else:
                mid = (col.values[0] + col.values[1]) / 2
                val = round(self.rng.uniform(col.values[0], mid), 0)
                val = int(val)
            if current_var.startswith("filtered_"):
                self.lines.append(f'  |> filter({col.name} {op} {val})')
            else:
                new_var = f"filtered_{self.domain.dataset_name}"
                self.lines.append(f'')
                self.lines.append(f'{new_var} = {current_var}')
                self.lines.append(f'  |> filter({col.name} {op} {val})')
                self.available_columns[new_var] = list(self.available_columns[current_var])
                current_var = new_var
            pipes_added += 1

        # Possibly add select (trim columns)
        if pipes_added < n_pipes and self.rng.random() < 0.3:
            avail = self.available_columns[current_var]
            n_select = min(self.rng.randint(2, 4), len(avail))
            selected = self.rng.sample(avail, n_select)
            # Ensure we keep columns needed for later operations
            if str_cols:
                group_col = str_cols[0].name
                if group_col not in selected and group_col in avail:
                    selected[0] = group_col
            if num_cols:
                agg_col = num_cols[0].name
                if agg_col not in selected and agg_col in avail:
                    if len(selected) < len(avail):
                        selected.append(agg_col)
            if self.domain.join_left_key and self.domain.join_left_key in avail:
                if self.domain.join_left_key not in selected:
                    selected.append(self.domain.join_left_key)
            self.lines.append(f'  |> select({", ".join(selected)})')
            self.available_columns[current_var] = selected
            pipes_added += 1

        # Group by + aggregate
        if pipes_added < n_pipes and str_cols and num_cols:
            group_col_def = self.rng.choice(str_cols)
            group_col = group_col_def.name
            if group_col in self.available_columns[current_var]:
                agg_col_def = self.rng.choice(num_cols)
                agg_col = agg_col_def.name
                if agg_col in self.available_columns[current_var]:
                    agg_func = self.rng.choice(["sum", "avg", "min", "max"])
                    agg_name = f"total_{agg_col}" if agg_func == "sum" else f"{agg_func}_{agg_col}"

                    self.lines.append(f'')
                    new_var = "summary"
                    self.lines.append(f'{new_var} = {current_var}')
                    self.lines.append(f'  |> group_by({group_col})')
                    self.lines.append(f'  |> aggregate({agg_name}: {agg_func}({agg_col}), count: count())')
                    self.available_columns[new_var] = [group_col, agg_name, "count"]
                    current_var = new_var
                    pipes_added += 2

        # Join with secondary dataset
        if use_join and self.domain.secondary_name:
            self.lines.append(f'')
            self.lines.append(f'-- Load lookup data')
            sec_var = self.domain.secondary_name
            self.lines.append(f'{sec_var} = load("{self.domain.secondary_name}")')
            sec_cols = [c.name for c in self.domain.secondary_columns]
            if len(sec_cols) > 2 and self.rng.random() < 0.5:
                keep = self.rng.sample(sec_cols, min(3, len(sec_cols)))
                if self.domain.join_right_key not in keep:
                    keep[0] = self.domain.join_right_key
                self.lines.append(f'  |> select({", ".join(keep)})')
                sec_cols = keep
            self.available_columns[sec_var] = sec_cols

            self.lines.append(f'')
            join_var = "enriched"
            left_key = self.domain.join_left_key
            right_key = self.domain.join_right_key
            self.lines.append(f'{join_var} = join({current_var}, {sec_var}, on: {left_key} == {right_key})')
            merged_cols = list(set(self.available_columns[current_var] + sec_cols))
            self.available_columns[join_var] = merged_cols
            current_var = join_var
            pipes_added += 1

        # Compute (add derived column)
        if pipes_added < n_pipes and self.rng.random() < 0.4:
            avail = self.available_columns[current_var]
            num_avail = [c for c in avail if any(
                nc.name == c and nc.dtype in ("int", "float") for nc in self.domain.columns
            )]
            for c in avail:
                if c.startswith(("total_", "sum_", "avg_", "min_", "max_")) or c == "count":
                    num_avail.append(c)

            if len(num_avail) >= 2:
                c1, c2 = self.rng.sample(num_avail, 2)
                op = self.rng.choice(["+", "-", "*", "/"])
                compute_name = "computed_val"
                new_var = current_var if current_var != self.domain.dataset_name else "processed"
                if current_var == new_var:
                    self.lines.append(f'  |> compute({compute_name}: {c1} {op} {c2})')
                else:
                    self.lines.append(f'')
                    self.lines.append(f'{new_var} = {current_var}')
                    self.lines.append(f'  |> compute({compute_name}: {c1} {op} {c2})')
                    self.available_columns[new_var] = list(avail) + [compute_name]
                    current_var = new_var
                pipes_added += 1

        # Sort + take (common ending pattern)
        avail = self.available_columns[current_var]
        sort_candidates = [c for c in avail if any(
            nc.name == c and nc.dtype in ("int", "float") for nc in self.domain.columns
        )]
        for c in avail:
            if c.startswith(("total_", "sum_", "avg_", "min_", "max_", "count", "computed")):
                sort_candidates.append(c)

        if sort_candidates:
            sort_col = self.rng.choice(sort_candidates)
            order = self.rng.choice(["asc", "desc"])
            take_n = self.rng.choice([3, 5, 10])

            result_var = "result"
            self.lines.append(f'')
            self.lines.append(f'{result_var} = {current_var}')
            self.lines.append(f'  |> sort_by({sort_col}, {order})')
            self.lines.append(f'  |> take({take_n})')
            self.available_columns[result_var] = list(avail)
            current_var = result_var
        else:
            result_var = current_var

        self.lines.append(f'')
        self.lines.append(f'emit {current_var}')

        return "\n".join(self.lines)


class DataGenerator:
    """Generate complete examples: datasets + correct code + buggy code.

    Can use either built-in domains (default) or custom domain definitions.
    """

    def __init__(
        self,
        seed: int | None = None,
        domains: list[DomainDef] | None = None,
        preloaded_datasets: dict[str, dict[str, list[dict]]] | None = None,
    ):
        """
        Args:
            seed: Random seed for reproducibility.
            domains: Custom domain definitions. Falls back to built-in DOMAINS if not provided.
            preloaded_datasets: Pre-loaded data keyed by domain.name.
                     If provided, this data is used instead of generating random data.
        """
        self.rng = random.Random(seed)
        self.interpreter = Interpreter()
        self.bug_injector = BugInjector(rng=self.rng)
        self.domains = domains or DOMAINS
        self.preloaded_datasets = preloaded_datasets

    def generate_example(
        self,
        difficulty: str = "medium",
        domain_name: str | None = None,
        bug_category: str | None = None,
        num_bugs: int = 1,
        max_retries: int = 10,
    ) -> Example | None:
        """Generate a random example with datasets, correct code, and buggy code.

        Returns None if generation fails after max_retries attempts.
        """
        for _ in range(max_retries):
            example = self._try_generate(difficulty, domain_name, bug_category, num_bugs)
            if example is not None:
                return example
        return None

    def _try_generate(
        self,
        difficulty: str,
        domain_name: str | None,
        bug_category: str | None,
        num_bugs: int,
    ) -> Example | None:
        # Pick domain
        if domain_name:
            domain = next((d for d in self.domains if d.name == domain_name), None)
            if domain is None:
                raise ValueError(f"Unknown domain: {domain_name}")
        else:
            domain = self.rng.choice(self.domains)

        # Get datasets: use preloaded data if available, otherwise generate
        if self.preloaded_datasets and domain.name in self.preloaded_datasets:
            datasets = copy.deepcopy(self.preloaded_datasets[domain.name])
        else:
            n_rows = self.rng.randint(20, 80)
            datasets = {
                domain.dataset_name: _generate_dataset(domain.columns, n_rows, self.rng)
            }
            if domain.secondary_name:
                datasets[domain.secondary_name] = _generate_secondary_dataset(
                    domain.secondary_columns, self.rng
                )

        # Generate correct program
        builder = ProgramBuilder(domain, self.rng)
        correct_code = builder.build(difficulty)

        # Verify correct code runs
        result = self.interpreter.run(correct_code, datasets)
        if not result.success:
            return None  # Generated invalid code, retry

        if not result.data:
            return None  # Empty result, retry

        expected_output = result.data

        # Inject bugs
        injector = BugInjector(rng=random.Random(self.rng.randint(0, 2**31)))
        buggy_code, bug_infos = injector.inject(
            correct_code,
            num_bugs=num_bugs,
            difficulty=difficulty,
            category=bug_category,
        )

        if not bug_infos:
            return None  # Failed to inject bugs, retry

        # Verify bug is effective
        if not verify_bug(correct_code, buggy_code, datasets):
            return None  # Bug didn't change output, retry

        # Count nodes
        node_names = self.interpreter.get_node_names(correct_code)
        has_join = "join(" in correct_code

        return Example(
            datasets=datasets,
            correct_code=correct_code,
            buggy_code=buggy_code,
            expected_output=expected_output,
            bug_info=[{
                "category": b.category,
                "description": b.description,
                "line_num": b.line_num,
                "original": b.original,
                "mutated": b.mutated,
            } for b in bug_infos],
            difficulty=difficulty,
            domain=domain.name,
            num_nodes=len(node_names),
            has_join=has_join,
        )

    def generate_batch(
        self,
        count: int,
        difficulty: str = "medium",
        num_bugs: int = 1,
    ) -> list[Example]:
        """Generate a batch of examples."""
        examples = []
        for _ in range(count):
            ex = self.generate_example(difficulty=difficulty, num_bugs=num_bugs)
            if ex:
                examples.append(ex)
        return examples
