#!/usr/bin/env python3
"""Compute the hard stop for the weekly push-to-clickhouse scan (issue #745).

The scheduled scan starts Saturday 22:00 UTC and must not be running anymore
when the US workweek starts, so its deadline is the first **Monday 09:00
America/New_York** after the scan started. The wall-clock time is computed in
New York local time, so the deadline follows DST:

* EST (roughly Nov-Mar): Monday 14:00 UTC
* EDT (roughly Mar-Nov): Monday 13:00 UTC

Prints the deadline as an ISO-8601 UTC timestamp (``2026-11-09T14:00:00Z``),
which ``weekly_test.py --deadline`` accepts.

Usage::

    python .github/scripts/weekly_scan_deadline.py                     # from now
    python .github/scripts/weekly_scan_deadline.py --start 2026-11-07T22:00:00Z
"""

from __future__ import annotations

import argparse
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

DEADLINE_TZ: ZoneInfo = ZoneInfo("America/New_York")
DEADLINE_WEEKDAY: int = 0  # Monday (datetime.weekday())
DEADLINE_LOCAL_TIME: time = time(hour=9)


def next_deadline(start: datetime) -> datetime:
    """Return the first Monday 09:00 New York time strictly after *start*, in UTC.

    Args:
        start: When the scan started. Must be timezone-aware.

    Raises:
        ValueError: *start* is naive, so it cannot be placed on the UTC timeline.
    """
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    local_start: datetime = start.astimezone(DEADLINE_TZ)
    days_ahead: int = (DEADLINE_WEEKDAY - local_start.weekday()) % 7
    candidate_day = local_start.date() + timedelta(days=days_ahead)
    # Build from the local wall-clock time rather than adding a timedelta to an
    # aware datetime, so a DST transition in between does not shift the hour.
    candidate: datetime = datetime.combine(
        candidate_day, DEADLINE_LOCAL_TIME, tzinfo=DEADLINE_TZ
    )
    if candidate <= local_start:
        candidate = datetime.combine(
            candidate_day + timedelta(days=7), DEADLINE_LOCAL_TIME, tzinfo=DEADLINE_TZ
        )
    return candidate.astimezone(timezone.utc)


def format_utc(moment: datetime) -> str:
    """Format an aware datetime as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        default=None,
        metavar="ISO8601",
        help="Scan start time (timezone-aware ISO-8601). Defaults to now (UTC).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args: argparse.Namespace = _parse_args(argv)
    start: datetime = args.start or datetime.now(timezone.utc)
    print(format_utc(next_deadline(start)))


if __name__ == "__main__":
    main()
