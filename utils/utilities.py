from datetime import datetime


def ts() -> str:
    """Local-time timestamp prefix, e.g. '[2026-07-17 14:32:05]'."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def human_bytes(n: float) -> str:
    """Format a byte count for logs, e.g. 1536 -> '1.5KB'.

    Binary units (1 KB = 1024 B), one decimal place, no space before the unit.
    Saturates at TB rather than continuing to PB — the caller is reporting freed
    HuggingFace cache space, which never reaches that scale.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
    raise AssertionError("unreachable: the TB branch always returns")
