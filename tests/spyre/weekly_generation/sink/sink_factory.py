from __future__ import annotations

from datetime import date
from pathlib import Path

from tests.spyre.weekly_generation.model_type import ModelType
from tests.spyre.weekly_generation.result_sink import ResultSink
from tests.spyre.weekly_generation.sink.clickhouse_sink import ClickHouseResultSink
from tests.spyre.weekly_generation.sink.csv_sink import CsvResultSink


def create_sink(model_type: ModelType, snapshot_date: date, write_to_csv: Path | None) -> ResultSink:
    sink: ResultSink
    if write_to_csv is not None:
        # One file per mode, since a single run covers both.
        path = write_to_csv.with_name(
            f"{write_to_csv.stem}-{model_type.value}{write_to_csv.suffix}"
        )
        sink = CsvResultSink(path=path, today=snapshot_date)
        print(f"{model_type.value}: recording verdicts in '{path}' (no DB access)")
    else:
        sink = ClickHouseResultSink(
            model_type=model_type, today=snapshot_date
        )
    return sink
