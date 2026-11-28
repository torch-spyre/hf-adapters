"""Read stable adapter dates without requiring git history (issue #372)."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

_LOG = logging.getLogger(__name__)


def load_added_dates(adapter_dir: Path, dates_json: Path) -> dict[str, str | None]:
    """Return committed ISO dates for installed adapters, or None for missing dates.

    A shallow checkout cannot recover real add-dates from git. Missing or
    invalid metadata therefore stays unknown rather than being guessed.
    """
    raw: dict[str, object] = {}
    try:
        loaded: object = json.loads(dates_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("expected a JSON object")
        raw = loaded
    except (OSError, ValueError) as error:
        _LOG.warning("Could not read adapter dates from %s: %s", dates_json, error)

    dates: dict[str, str | None] = {}
    for adapter_path in sorted(adapter_dir.glob("hf_*.py")):
        value: object = raw.get(adapter_path.stem)
        iso_date: str | None = None
        if isinstance(value, str):
            try:
                parsed: date = date.fromisoformat(value)
                if parsed.isoformat() == value:
                    iso_date = value
            except ValueError:
                pass
        if value is not None and iso_date is None:
            _LOG.warning("Invalid add-date for %s: %r", adapter_path.stem, value)
        dates[adapter_path.stem] = iso_date
    return dates
