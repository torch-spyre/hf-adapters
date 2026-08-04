"""Build the result sink a run should write to.

The single place that decides between the two backends, so no caller has to
import a concrete sink — or know that importing one has a cost.
"""

from __future__ import annotations

from pathlib import Path

from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink
from tests.spyre.weekly_generation.sink.result_sink import ResultSink


def csv_path_for(base: Path, model_type: ModelType) -> Path:
    """Per-model-type CSV path derived from a ``--write-to-csv`` argument.

    A single invocation can cover both model types and each needs its own file,
    so the stem gets a ``-<model_type>`` suffix. Exposed rather than inlined
    because callers also report the path they wrote to, and computing it twice is
    how the log line and the actual file get to disagree.
    """
    return base.with_name(f"{base.stem}-{model_type}{base.suffix}")


def create_sink(
    model_type: ModelType,
    write_to_csv: Path | None,
) -> ResultSink:
    """Return the sink for a *model_type* run, keyed on whether a CSV was asked for.

    One sink instance binds one destination — a single ClickHouse table, or a
    single file — so a run covering both model types needs one per type rather
    than one overall.

    With *write_to_csv* the verdicts go to a new CSV instead of ClickHouse, which
    needs no credentials. Its path gets a ``-<model_type>`` suffix, since a single
    invocation can cover both types and each needs its own file.

    Otherwise a ``ClickHouseResultSink`` is built, which connects during
    construction and creates its table if missing.
    """
    if write_to_csv is not None:
        path = csv_path_for(write_to_csv, model_type)
        print(f"{model_type}: recording verdicts in '{path}' (no DB access)")
        return CsvResultSink(path=path)

    # Deferred, so reaching the CSV branch neither imports clickhouse_connect nor
    # reads .env. Together with csv_sink taking its column list from the
    # dependency-free table_schema, this is what lets --write-to-csv run on a host
    # with no driver installed at all.
    from tests.spyre.weekly_generation.sink.clickhouse_sink import ClickHouseResultSink

    return ClickHouseResultSink(model_type=model_type)
