"""Programmatic bug injection for DSL programs.

Takes a correct program and injects bugs of various categories:
- Syntax: missing pipe, wrong keyword
- Wrong argument: sum vs avg, asc vs desc
- Wrong column: filter on wrong column
- Join bug: wrong join key
- Logic bug: wrong filter value, off-by-one in take
- Multi-bug: two bugs in different locations
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .interpreter import Interpreter


@dataclass
class BugInfo:
    category: str       # syntax, wrong_arg, wrong_col, join_bug, logic_bug
    description: str    # human-readable description of the bug
    line_num: int       # line number where bug was injected
    original: str       # original line text
    mutated: str        # mutated line text


# ---------------------------------------------------------------------------
# Bug injection strategies
# ---------------------------------------------------------------------------

def _find_pipe_lines(lines: list[str]) -> list[tuple[int, str]]:
    """Find lines that contain pipe operations (|> ...)."""
    return [(i, line.strip()) for i, line in enumerate(lines) if line.strip().startswith("|>")]


def _replace_line(lines: list[str], idx: int, new_content: str) -> None:
    """Replace line at idx with new_content, preserving original indentation."""
    indent = len(lines[idx]) - len(lines[idx].lstrip())
    lines[idx] = " " * indent + new_content


# Keyword → common typos, used by _inject_syntax for typo-style bugs
_SYNTAX_KEYWORDS = {
    "filter": ["fliter", "filtre", "filer"],
    "select": ["selct", "slect", "selecct"],
    "group_by": ["groupby", "group_bi", "grp_by"],
    "aggregate": ["aggreate", "agregate", "aggregte"],
    "compute": ["compue", "comput", "compote"],
    "sort_by": ["sortby", "sort_bi", "srt_by"],
    "take": ["tke", "taek"],
}


class BugInjector:
    """Inject bugs into correct DSL programs."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def inject(
        self,
        code: str,
        num_bugs: int = 1,
        difficulty: str = "medium",
        category: str | None = None,
    ) -> tuple[str, list[BugInfo]]:
        """Inject bugs into correct code.

        Args:
            code: Correct DSL code
            num_bugs: Number of bugs to inject (1 for single, 2 for multi-bug)
            difficulty: easy, medium, or hard
            category: Specific bug category, or None for random

        Returns:
            (buggy_code, list of BugInfo descriptions)
        """
        lines = code.split("\n")
        bugs_injected: list[BugInfo] = []

        if category:
            categories = [category] * num_bugs
        else:
            categories = self._pick_categories(num_bugs, difficulty)

        # Track which lines we've already mutated to avoid double-mutation
        mutated_lines: set[int] = set()

        for cat in categories:
            bug = self._inject_one(lines, cat, mutated_lines)
            if bug:
                bugs_injected.append(bug)
                mutated_lines.add(bug.line_num)

        return "\n".join(lines), bugs_injected

    def _pick_categories(self, num_bugs: int, difficulty: str) -> list[str]:
        if difficulty == "easy":
            pool = ["syntax", "syntax", "wrong_arg"]
        elif difficulty == "medium":
            pool = ["wrong_arg", "wrong_col", "logic_bug", "wrong_arg"]
        else:  # hard
            pool = ["wrong_col", "join_bug", "logic_bug", "logic_bug"]

        return [self.rng.choice(pool) for _ in range(num_bugs)]

    def _inject_one(self, lines: list[str], category: str, mutated: set[int]) -> BugInfo | None:
        """Inject a single bug. Modifies lines in place."""
        injectors = {
            "syntax": self._inject_syntax,
            "wrong_arg": self._inject_wrong_arg,
            "wrong_col": self._inject_wrong_col,
            "join_bug": self._inject_join_bug,
            "logic_bug": self._inject_logic_bug,
        }
        injector = injectors.get(category)
        if injector is None:
            return None
        return injector(lines, mutated)

    # --- Syntax bugs ---

    def _inject_syntax(self, lines: list[str], mutated: set[int]) -> BugInfo | None:
        pipe_lines = _find_pipe_lines(lines)
        candidates = [(i, l) for i, l in pipe_lines if i not in mutated]
        if not candidates:
            return None

        idx, original = self.rng.choice(candidates)
        bug_type = self.rng.choice(["remove_pipe", "typo_keyword"])

        if bug_type == "typo_keyword":
            for kw, typos in _SYNTAX_KEYWORDS.items():
                if kw + "(" in original:
                    typo = self.rng.choice(typos)
                    mutated_line = original.replace(kw, typo, 1)
                    _replace_line(lines, idx, mutated_line)
                    return BugInfo(
                        category="syntax",
                        description=f"Typo in keyword: '{kw}' -> '{typo}'",
                        line_num=idx + 1,
                        original=original,
                        mutated=mutated_line,
                    )

        # remove_pipe (also fallback if no keyword matched)
        mutated_line = original.replace("|>", "", 1).strip()
        _replace_line(lines, idx, mutated_line)
        return BugInfo(
            category="syntax",
            description="Missing pipe operator '|>'",
            line_num=idx + 1,
            original=original,
            mutated=mutated_line,
        )

    # --- Wrong argument bugs ---

    def _inject_wrong_arg(self, lines: list[str], mutated: set[int]) -> BugInfo | None:
        pipe_lines = _find_pipe_lines(lines)
        candidates = [(i, l) for i, l in pipe_lines if i not in mutated]
        self.rng.shuffle(candidates)

        for idx, original in candidates:
            # Try swapping aggregate function
            agg_match = re.search(r"(sum|avg|min|max|count)\(", original)
            if agg_match:
                old_func = agg_match.group(1)
                alternatives = [f for f in ["sum", "avg", "min", "max", "count"] if f != old_func]
                new_func = self.rng.choice(alternatives)
                mutated_line = original.replace(old_func + "(", new_func + "(", 1)
                _replace_line(lines, idx, mutated_line)
                return BugInfo(
                    category="wrong_arg",
                    description=f"Wrong aggregate function: '{old_func}' -> '{new_func}'",
                    line_num=idx + 1,
                    original=original,
                    mutated=mutated_line,
                )

            # Try swapping sort order
            sort_match = re.search(r"sort_by\(\s*\w+\s*,\s*(asc|desc)\s*\)", original)
            if sort_match:
                old_order = sort_match.group(1)
                new_order = "desc" if old_order == "asc" else "asc"
                mutated_line = original.replace(old_order, new_order, 1)
                _replace_line(lines, idx, mutated_line)
                return BugInfo(
                    category="wrong_arg",
                    description=f"Wrong sort order: '{old_order}' -> '{new_order}'",
                    line_num=idx + 1,
                    original=original,
                    mutated=mutated_line,
                )

            # Try swapping comparison operator
            cmp_match = re.search(r"filter\(\s*\w+\s*(==|!=|>|<|>=|<=)", original)
            if cmp_match:
                old_op = cmp_match.group(1)
                alt_ops = [op for op in ["==", "!=", ">", "<", ">=", "<="] if op != old_op]
                new_op = self.rng.choice(alt_ops)
                mutated_line = original.replace(old_op, new_op, 1)
                _replace_line(lines, idx, mutated_line)
                return BugInfo(
                    category="wrong_arg",
                    description=f"Wrong comparison operator: '{old_op}' -> '{new_op}'",
                    line_num=idx + 1,
                    original=original,
                    mutated=mutated_line,
                )

        return None

    # --- Wrong column bugs ---

    def _inject_wrong_col(self, lines: list[str], mutated: set[int]) -> BugInfo | None:
        # Find all column names referenced in the code
        all_cols: set[str] = set()
        for line in lines:
            for m in re.finditer(r"(?:filter|select|group_by|sort_by|aggregate|compute)\(([^)]+)\)", line):
                for word in re.findall(r"\b(\w+)\b", m.group(1)):
                    if word not in {"sum", "avg", "min", "max", "count", "asc", "desc",
                                    "filter", "select", "group_by", "sort_by", "aggregate",
                                    "compute", "take", "on"}:
                        all_cols.add(word)

        if len(all_cols) < 2:
            return None

        pipe_lines = _find_pipe_lines(lines)
        candidates = [(i, l) for i, l in pipe_lines if i not in mutated]
        self.rng.shuffle(candidates)

        for idx, original in candidates:
            cols_in_line = []
            for col in all_cols:
                if re.search(rf"\b{re.escape(col)}\b", original):
                    cols_in_line.append(col)

            if not cols_in_line:
                continue

            old_col = self.rng.choice(cols_in_line)
            alternatives = [c for c in all_cols if c != old_col]
            if not alternatives:
                continue

            new_col = self.rng.choice(list(alternatives))
            mutated_line = re.sub(rf"\b{re.escape(old_col)}\b", new_col, original, count=1)

            if mutated_line == original:
                continue

            _replace_line(lines, idx, mutated_line)
            return BugInfo(
                category="wrong_col",
                description=f"Wrong column reference: '{old_col}' -> '{new_col}'",
                line_num=idx + 1,
                original=original,
                mutated=mutated_line,
            )

        return None

    # --- Join bugs ---

    def _inject_join_bug(self, lines: list[str], mutated: set[int]) -> BugInfo | None:
        for i, line in enumerate(lines):
            if i in mutated:
                continue
            join_match = re.search(
                r"join\(\s*(\w+)\s*,\s*(\w+)\s*,\s*on:\s*(\w+)\s*(==)\s*(\w+)\s*\)",
                line.strip(),
            )
            if join_match:
                original = line.strip()
                left_key = join_match.group(3)
                right_key = join_match.group(5)

                bug_type = self.rng.choice(["swap_keys", "wrong_key"])
                if bug_type == "swap_keys":
                    mutated_line = original.replace(
                        f"on: {left_key} == {right_key}",
                        f"on: {right_key} == {left_key}",
                    )
                    _replace_line(lines, i, mutated_line)
                    return BugInfo(
                        category="join_bug",
                        description=f"Swapped join keys: '{left_key} == {right_key}' -> '{right_key} == {left_key}'",
                        line_num=i + 1,
                        original=original,
                        mutated=mutated_line,
                    )
                else:
                    all_cols: set[str] = set()
                    for ln in lines:
                        for m in re.finditer(r"\b(\w+)\b", ln):
                            all_cols.add(m.group(1))
                    skip = {"load", "join", "filter", "select", "group_by", "aggregate",
                            "compute", "sort_by", "take", "emit", "on", "asc", "desc",
                            "sum", "avg", "min", "max", "count"}
                    all_cols -= skip
                    alternatives = [c for c in all_cols if c != left_key and c != right_key]
                    if alternatives:
                        new_key = self.rng.choice(alternatives)
                        target_key = self.rng.choice([left_key, right_key])
                        mutated_line = original.replace(target_key, new_key, 1)
                        _replace_line(lines, i, mutated_line)
                        return BugInfo(
                            category="join_bug",
                            description=f"Wrong join key: '{target_key}' -> '{new_key}'",
                            line_num=i + 1,
                            original=original,
                            mutated=mutated_line,
                        )

        return None

    # --- Logic bugs ---

    def _inject_logic_bug(self, lines: list[str], mutated: set[int]) -> BugInfo | None:
        pipe_lines = _find_pipe_lines(lines)
        candidates = [(i, l) for i, l in pipe_lines if i not in mutated]
        self.rng.shuffle(candidates)

        for idx, original in candidates:
            # Off-by-one in take()
            take_match = re.search(r"take\(\s*(\d+)\s*\)", original)
            if take_match:
                old_n = int(take_match.group(1))
                delta = self.rng.choice([-1, 1, 2, -2])
                new_n = max(1, old_n + delta)
                if new_n == old_n:
                    new_n = old_n + 1
                mutated_line = original.replace(f"take({old_n})", f"take({new_n})")
                _replace_line(lines, idx, mutated_line)
                return BugInfo(
                    category="logic_bug",
                    description=f"Off-by-{'one' if abs(new_n - old_n) == 1 else str(abs(new_n - old_n))} in take: {old_n} -> {new_n}",
                    line_num=idx + 1,
                    original=original,
                    mutated=mutated_line,
                )

            # Wrong filter value (string)
            filter_str_match = re.search(r'filter\(\s*\w+\s*==\s*"([^"]+)"\s*\)', original)
            if filter_str_match:
                old_val = filter_str_match.group(1)
                mutations = [
                    old_val.upper() if old_val.islower() else old_val.lower(),
                    old_val + "s",
                    old_val[:-1] if len(old_val) > 1 else old_val + "x",
                ]
                new_val = self.rng.choice([m for m in mutations if m != old_val])
                mutated_line = original.replace(f'"{old_val}"', f'"{new_val}"')
                _replace_line(lines, idx, mutated_line)
                return BugInfo(
                    category="logic_bug",
                    description=f"Wrong filter value: '{old_val}' -> '{new_val}'",
                    line_num=idx + 1,
                    original=original,
                    mutated=mutated_line,
                )

            # Wrong filter value (number)
            filter_num_match = re.search(r"filter\(\s*\w+\s*(?:>|<|>=|<=|==|!=)\s*(\d+)\s*\)", original)
            if filter_num_match:
                old_val = int(filter_num_match.group(1))
                delta = self.rng.choice([-10, -5, -1, 1, 5, 10])
                new_val = max(0, old_val + delta)
                if new_val == old_val:
                    new_val = old_val + 1
                mutated_line = re.sub(
                    rf"(filter\(\s*\w+\s*(?:>|<|>=|<=|==|!=)\s*){old_val}",
                    rf"\g<1>{new_val}",
                    original,
                )
                _replace_line(lines, idx, mutated_line)
                return BugInfo(
                    category="logic_bug",
                    description=f"Wrong filter threshold: {old_val} -> {new_val}",
                    line_num=idx + 1,
                    original=original,
                    mutated=mutated_line,
                )

        return None


def verify_bug(
    correct_code: str,
    buggy_code: str,
    datasets: dict[str, list[dict]],
) -> bool:
    """Verify that the buggy code produces different output than the correct code.
    Returns True if the bug is effective (outputs differ or buggy code errors)."""
    interp = Interpreter()
    correct_result = interp.run(correct_code, datasets)
    buggy_result = interp.run(buggy_code, datasets)

    # If buggy code errors, the bug is effective
    if not buggy_result.success:
        return True

    # If correct code errors, something is wrong with the test
    if not correct_result.success:
        return False

    # Compare outputs
    return correct_result.data != buggy_result.data
