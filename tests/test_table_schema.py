"""Tests for the weekly-scan table schema.

Two things are pinned here.

**The DDL and ``TABLE_COLUMNS`` agree, in order.** ``ClickHouseResultSink``
buffers each row as a positional list and passes ``column_names=TABLE_COLUMNS``
to a bulk insert, so a mismatch against the CREATE TABLE body does not raise —
it writes values into the wrong columns. Three comments in the source used to
warn about keeping them in sync; this asserts it instead.

**``table_schema`` stays a leaf.** Importing it must not pull in
``clickhouse_connect`` or ``dotenv``, because ``csv_sink`` imports it and
``--write-to-csv`` is meant to work on a host with neither installed.

Run with ``pytest --noconftest`` (no torch needed) — see
``test_weekly_prefilter.py`` for the same pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.spyre.weekly_generation.table_schema import (
    DATABASE,
    EMBEDDING_CREATE_TABLE_SQL,
    EMBEDDING_TABLE_NAME,
    GENERATIVE_CREATE_TABLE_SQL,
    GENERATIVE_TABLE_NAME,
    TABLE_COLUMNS,
)


def _columns_in_ddl(ddl: str) -> list[str]:
    """The column names from a CREATE TABLE body, in declaration order.

    Takes the parenthesised block that opens after the table name and closes
    before ENGINE — matched by paren depth, since the ``ORDER BY (...)`` tuple
    later in the statement also contains parentheses.
    """
    start = ddl.index("(")
    depth = 0
    for index in range(start, len(ddl)):
        if ddl[index] == "(":
            depth += 1
        elif ddl[index] == ")":
            depth -= 1
            if depth == 0:
                body = ddl[start + 1 : index]
                break
    else:  # pragma: no cover - unbalanced DDL is a syntax error, not a test case
        raise AssertionError("unbalanced parentheses in the DDL")

    names: list[str] = []
    for line in body.splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped:
            continue
        match = re.match(r"^(\w+)\s+\S", stripped)
        assert match, f"could not parse a column name from {stripped!r}"
        names.append(match.group(1))
    return names


class TestDdlMatchesTableColumns:
    """A positional bulk insert makes any drift here a silent data-corruption bug."""

    def test_embedding_ddl_columns_match_in_order(self) -> None:
        assert _columns_in_ddl(EMBEDDING_CREATE_TABLE_SQL) == list(TABLE_COLUMNS)

    def test_generative_ddl_columns_match_in_order(self) -> None:
        assert _columns_in_ddl(GENERATIVE_CREATE_TABLE_SQL) == list(TABLE_COLUMNS)

    def test_the_two_tables_share_one_shape(self) -> None:
        """Both model types write the same 14 columns; only the table name differs."""
        assert _columns_in_ddl(EMBEDDING_CREATE_TABLE_SQL) == _columns_in_ddl(
            GENERATIVE_CREATE_TABLE_SQL
        )

    def test_column_names_are_unique(self) -> None:
        assert len(set(TABLE_COLUMNS)) == len(TABLE_COLUMNS)

    def test_the_parser_would_notice_a_reordering(self) -> None:
        """Guard the guard: a swapped pair in the DDL must fail the comparison.

        Without this, a parser that silently returned [] or a sorted list would
        make the tests above pass against any DDL at all.
        """
        swapped = EMBEDDING_CREATE_TABLE_SQL.replace(
            "    model_name        String,\n    config_class      String,",
            "    config_class      String,\n    model_name        String,",
        )
        assert swapped != EMBEDDING_CREATE_TABLE_SQL, "the replace found no match"
        assert _columns_in_ddl(swapped) != list(TABLE_COLUMNS)


class TestDdlShape:
    def test_replacing_merge_tree_on_snapshot_date(self) -> None:
        """Same-day duplicates must collapse on merge.

        The weekly scan deliberately prefers writing a possible duplicate over
        dropping a result it was asked to produce; that trade-off is only safe
        because of this engine choice.
        """
        for ddl in (EMBEDDING_CREATE_TABLE_SQL, GENERATIVE_CREATE_TABLE_SQL):
            assert "ReplacingMergeTree(snapshot_date)" in ddl
            assert "ORDER BY (model_name, snapshot_date)" in ddl

    def test_is_idempotent(self) -> None:
        """The sink runs this whenever the table is missing."""
        for ddl in (EMBEDDING_CREATE_TABLE_SQL, GENERATIVE_CREATE_TABLE_SQL):
            assert "CREATE TABLE IF NOT EXISTS" in ddl

    def test_each_ddl_targets_its_own_qualified_table(self) -> None:
        assert f"{DATABASE}.{EMBEDDING_TABLE_NAME}" in EMBEDDING_CREATE_TABLE_SQL
        assert f"{DATABASE}.{GENERATIVE_TABLE_NAME}" in GENERATIVE_CREATE_TABLE_SQL
        assert EMBEDDING_TABLE_NAME != GENERATIVE_TABLE_NAME

    def test_nullable_columns_are_the_optional_ones(self) -> None:
        """added_date/failure_category/error are None for some rows; the rest never are.

        ``ClickHouseResultSink`` coerces failure_category/error to '' before
        inserting, so this documents the DDL's intent rather than the sink's
        behaviour — but added_date genuinely arrives as None.
        """
        nullable = set(
            re.findall(r"^\s*(\w+)\s+Nullable\(", EMBEDDING_CREATE_TABLE_SQL, re.M)
        )
        assert nullable == {"added_date", "failure_category", "error"}


class TestTableSchemaIsALeafModule:
    """``--write-to-csv`` must not need the ClickHouse driver or a .env file."""

    def test_imports_nothing_outside_the_standard_library(self) -> None:
        import ast

        import tests.spyre.weekly_generation.table_schema as ts

        # Located via the module's own __file__ so the test does not depend on
        # pytest's cwd.
        src = Path(ts.__file__).read_text()
        roots: set[str] = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        # __future__ only; anything else means the leaf has grown a dependency.
        assert roots <= {"__future__"}, f"table_schema gained imports: {roots}"

    def test_importing_it_does_not_load_the_driver(self) -> None:
        """Fresh subprocess: neither module may appear in sys.modules afterwards."""
        import subprocess
        import sys

        program = (
            "import sys; "
            "import tests.spyre.weekly_generation.table_schema as ts; "
            "assert ts.TABLE_COLUMNS, 'schema did not load'; "
            "leaked = [m for m in ('clickhouse_connect', 'dotenv') "
            "          if m in sys.modules]; "
            "assert not leaked, f'table_schema pulled in {leaked}'; "
            "print('clean')"
        )
        proc = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert proc.returncode == 0, proc.stderr
        assert "clean" in proc.stdout


class TestClickhouseDbReExports:
    """Existing importers of clickhouse_db keep working after the split."""

    def test_re_exports_are_the_same_objects(self) -> None:
        import tests.spyre.weekly_generation.clickhouse_db as db
        import tests.spyre.weekly_generation.table_schema as ts

        for name in (
            "DATABASE",
            "EMBEDDING_TABLE_NAME",
            "GENERATIVE_TABLE_NAME",
            "TABLE_COLUMNS",
            "EMBEDDING_CREATE_TABLE_SQL",
            "GENERATIVE_CREATE_TABLE_SQL",
        ):
            assert getattr(db, name) is getattr(ts, name), name

    def test_still_exposes_the_client_helpers(self) -> None:
        import tests.spyre.weekly_generation.clickhouse_db as db

        assert callable(db.get_client)
        assert callable(db.table_exists)
