from datetime import datetime


def ts() -> str:
    """Local-time timestamp prefix, e.g. '[2026-07-17 14:32:05]'."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
