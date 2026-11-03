import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT: Path = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "weekly_scan_deadline.py"
)
_SPEC = importlib.util.spec_from_file_location("weekly_scan_deadline", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
weekly_scan_deadline: ModuleType = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(weekly_scan_deadline)


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("start", "expected"),
    [
        # Scheduled Saturday 22:00 UTC start, winter (EST): Monday 14:00 UTC.
        (_utc(2026, 11, 7, 22), _utc(2026, 11, 9, 14)),
        # Same start in summer (EDT): Monday 13:00 UTC, i.e. still 09:00 NY.
        (_utc(2026, 7, 4, 22), _utc(2026, 7, 6, 13)),
        # A late-starting run (cron delay) still targets the same Monday.
        (_utc(2026, 11, 8, 3, 30), _utc(2026, 11, 9, 14)),
        # DST ends (Nov 1 2026) between Saturday start and Monday deadline.
        (_utc(2026, 10, 31, 22), _utc(2026, 11, 2, 14)),
        # DST starts (Mar 8 2026) between Saturday start and Monday deadline.
        (_utc(2026, 3, 7, 22), _utc(2026, 3, 9, 13)),
        # Monday before 09:00 NY: the deadline is later that same day.
        (_utc(2026, 11, 9, 13, 59), _utc(2026, 11, 9, 14)),
        # Exactly at / after the deadline: roll over to the next Monday.
        (_utc(2026, 11, 9, 14), _utc(2026, 11, 16, 14)),
    ],
)
def test_next_deadline(start: datetime, expected: datetime) -> None:
    assert weekly_scan_deadline.next_deadline(start) == expected


def test_naive_start_is_rejected() -> None:
    with pytest.raises(ValueError):
        weekly_scan_deadline.next_deadline(datetime(2026, 11, 7, 22))


def test_cli_prints_utc_iso(capsys: pytest.CaptureFixture[str]) -> None:
    weekly_scan_deadline.main(["--start", "2026-11-07T22:00:00Z"])
    assert capsys.readouterr().out.strip() == "2026-11-09T14:00:00Z"
