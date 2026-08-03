from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

from tests.spyre.weekly_generation.clickhouse_db import TABLE_COLUMNS
from tests.spyre.weekly_generation.result_sink import ResultSink, _require_non_empty


class CsvResultSink(ResultSink):
    """Write rows to a fresh CSV file. Write-only, for no-database test runs.

    Nothing is read back and the file must not already exist, so there is no prior
    history to consult: ``should_insert_row`` is unconditionally True.
    """

    def __init__(self, path: Path, today: date | None = None) -> None:
        """Open *path* for writing. Raises if it already exists and is non-empty."""
        super().__init__(today=today)
        self._path: Path = path
        if path.exists() and path.stat().st_size > 0:
            raise FileExistsError(
                f"'{path}' already exists and is not empty. This sink writes a "
                f"single run's results to a new file and never reads one back; "
                f"choose a different path or remove the existing file."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(TABLE_COLUMNS))
        self._writer.writeheader()
        self._fh.flush()

    def should_insert_row(self, model_name: str) -> bool:
        """Always True — a fresh output file has no prior rows to block on."""
        _require_non_empty(model_name, "model_name")
        return True

    def _insert_entry(
        self,
        *,
        model_name: str,
        config_class: str,
        adapter_name: str,
        added_date: date | None,
        snapshot_date: date,
        verified_on_cpu: bool,
        verified_on_gpu: bool,
        verified_on_spyre: bool,
        num_downloads: int,
        family: str,
        architecture: str,
        parameters_number: int,
        failure_category: str | None,
        error: str | None,
    ) -> None:
        rec: dict[str, Any] = {
            "model_name": model_name,
            "config_class": config_class,
            "adapter_name": adapter_name,
            "added_date": added_date,
            "snapshot_date": snapshot_date,
            "verified_on_cpu": verified_on_cpu,
            "verified_on_gpu": verified_on_gpu,
            "verified_on_spyre": verified_on_spyre,
            "num_downloads": num_downloads,
            "family": family,
            "architecture": architecture,
            "parameters_number": parameters_number,
            "failure_category": failure_category,
            "error": error,
        }
        self._writer.writerow(rec)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
