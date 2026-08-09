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


def concat_and_dedup_dicts(
    first: list[dict], second: list[dict], key: str = "model_id"
) -> list[dict]:
    """Concatenate two lists of dicts, dropping later duplicates of *key*.

    First occurrence wins, so passing the higher-precedence list as *first* is how
    the caller chooses which copy of a duplicate survives. Relative order within
    each list is preserved — the weekly pipeline's descending-downloads contract
    depends on the fetched list not being reshuffled here.

    Dicts missing *key* are passed through unchanged rather than raising: they
    cannot be compared for identity, so they are never treated as duplicates.
    """
    seen: set = set()
    result: list[dict] = []

    for d in (*first, *second):
        if key not in d:
            result.append(d)
            continue
        item_id = d[key]
        if item_id not in seen:
            seen.add(item_id)
            result.append(d)

    return result
