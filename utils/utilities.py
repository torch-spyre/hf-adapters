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


def concat_and_dedup_dicts(l1: list[dict], l2: list[dict]) -> list[dict]:
    """
    Concatenates two lists of dicts, removing duplicates by a unique ID key.
    Keeps the first instance encountered.
    """
    seen = set()
    result = []

    # Iterate through both lists sequentially
    for d in l1 + l2:
        # Get the ID value, skip or handle safely if the key is missing
        item_id = d["model_id"]
        if item_id not in seen:
            seen.add(item_id)
            result.append(d)

    return result
