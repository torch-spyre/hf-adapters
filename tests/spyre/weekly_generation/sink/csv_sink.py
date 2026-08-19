"""CSV result sink: one run's rows to one new file, no database.

What ``--write-to-csv`` selects, so the weekly pipeline can be exercised on a
host with no ClickHouse credentials — and, because ``TABLE_COLUMNS`` comes from
the dependency-free ``table_schema`` rather than from ``clickhouse_db``, with no
ClickHouse driver installed either. The header stays column-for-column identical
to the live table, since the point of this sink is to show what *would* be
inserted.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any

from tests.spyre.weekly_generation.sink.result_sink import ResultSink
from tests.spyre.weekly_generation.table_schema import TABLE_COLUMNS


class CsvResultSink(ResultSink):
    """Write rows to a fresh CSV file, for no-database test runs.

    The file must not already exist — so one file holds exactly one run's rows.
    Reading back via ``get_models_at_snapshot_date`` is supported: the CSV is
    re-opened for reading and filtered by ``snapshot_date``.
    """

    def __init__(self, path: Path) -> None:
        """Open *path* for writing. Raises if it already exists and is non-empty."""
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
        curated: bool,
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
            "curated": curated,
        }
        self._writer.writerow(rec)
        self._fh.flush()

    def get_models_at_snapshot_date(self, *, snapshot_date: date) -> set[str]:
        """Return every distinct model name recorded for *snapshot_date*.

        Re-opens the CSV for reading (independent of the write handle, which
        stays open and flushed-per-row) and returns the unique ``model_name``
        values whose ``snapshot_date`` column parses to *snapshot_date*.

        Row values are parsed as dates rather than compared as strings, so a
        CSV that mixes ISO and ``DD/MM/YYYY`` snapshot_date formats (e.g. one
        assembled by ``clickhouse_db.import_csv``, which accepts either) still
        resumes correctly. Rows with an unparseable snapshot_date, or an
        empty model_name, are ignored — the base class rejects empty names at
        ``add_entry`` time, so a blank one is by definition external noise.
        """
        seen: set[str] = set()
        with open(self._path, newline="") as fh:
            for row in csv.DictReader(fh):
                name: str = (row.get("model_name") or "").strip()
                if not name:
                    continue
                raw: str = (row.get("snapshot_date") or "").strip()
                try:
                    parsed: date = date.fromisoformat(raw)
                except ValueError:
                    # DD/MM/YYYY is the only other shape the sibling importer
                    # writes; anything else is skipped.
                    try:
                        parsed = datetime.strptime(raw, "%d/%m/%Y").date()
                    except ValueError:
                        continue
                if parsed == snapshot_date:
                    seen.add(name)
        return seen

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
