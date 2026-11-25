from datetime import datetime


def ts() -> str:
    """Local-time timestamp prefix, e.g. '[2026-07-17 14:32:05]'."""
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


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
