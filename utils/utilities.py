from datetime import datetime


def ts() -> str:
    """Local-time timestamp prefix, e.g. '[2026-07-17 14:32:05]'."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def human_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}"
        n /= 1024
