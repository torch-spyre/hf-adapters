#!/usr/bin/env python3
# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Download one HF snapshot in a child process, for ``warm_top_models_cache.py``.

``snapshot_download`` cannot be interrupted from its own thread, so the warmer
runs each attempt here, in a child it can kill when the time budget runs out.
A killed download leaves ``.incomplete`` blobs, which the next run resumes.

Reads ``{"repo_id": ..., "ignore_patterns": [...]}`` as the only argument
(``HF_TOKEN`` and ``HF_HOME`` come from the environment) and prints one
``RESULT:`` JSON line describing the outcome, already classified, so the parent
never has to unpickle a Hub exception.

Imports only ``huggingface_hub`` and ``httpx``, which keeps each attempt's
start-up cheap and lets the error classifiers be tested on their own.
"""

from __future__ import annotations

import errno
import json
import os
import sys
from collections.abc import Iterator

import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

RESULT_PREFIX: str = "RESULT:"
HTTP_TOO_MANY_REQUESTS: int = 429
_HTTP_TRANSIENT_STATUSES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


def exception_chain(error: BaseException) -> Iterator[BaseException]:
    """*error*, then each exception it was raised from or during, without cycles."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def http_status(error: BaseException) -> int | None:
    """The HTTP status of the first Hub response anywhere in the cause chain.

    ``snapshot_download`` wraps a failed metadata request, a 429 included, in a
    ``LocalEntryNotFoundError`` when no usable local snapshot exists, so the
    status is usually on the cause rather than on the raised exception.
    """
    for e in exception_chain(error):
        if isinstance(e, HfHubHTTPError) and e.response is not None:
            return e.response.status_code
    return None


def is_disk_full(error: BaseException) -> bool:
    return any(
        isinstance(e, OSError) and e.errno == errno.ENOSPC
        for e in exception_chain(error)
    )


def is_transient(error: BaseException) -> bool:
    """True for errors a later attempt can fix: rate limits, 5xx, network drops."""
    chain: list[BaseException] = list(exception_chain(error))
    # Also covers GatedRepoError, a RepositoryNotFoundError.
    if any(
        isinstance(e, (RepositoryNotFoundError, RevisionNotFoundError)) for e in chain
    ):
        return False
    status: int | None = http_status(error)
    if status is not None:
        return status in _HTTP_TRANSIENT_STATUSES
    # Raised when the Hub cannot be reached at all. Checked before its parent
    # class EntryNotFoundError, which means the file really is not there.
    if isinstance(error, LocalEntryNotFoundError):
        return True
    if isinstance(error, EntryNotFoundError):
        return False
    return any(
        isinstance(e, (httpx.TransportError, TimeoutError, ConnectionError))
        for e in chain
    )


def describe(error: BaseException) -> dict[str, object]:
    """The classified outcome of a failed attempt, as the parent reads it."""
    return {
        "ok": False,
        "path": None,
        "error": f"{type(error).__name__}: {error}",
        "status": http_status(error),
        "transient": is_transient(error),
        "disk_full": is_disk_full(error),
    }


def download(repo_id: str, ignore_patterns: list[str]) -> dict[str, object]:
    try:
        path: str = snapshot_download(
            repo_id=repo_id,
            token=os.environ.get("HF_TOKEN") or None,
            ignore_patterns=ignore_patterns,
            max_workers=1,
        )
    except Exception as e:
        return describe(e)
    return {"ok": True, "path": path}


def main(argv: list[str]) -> int:
    request: dict[str, object] = json.loads(argv[1])
    ignore: object = request.get("ignore_patterns") or []
    outcome: dict[str, object] = download(
        str(request["repo_id"]),
        [str(p) for p in ignore] if isinstance(ignore, list) else [],
    )
    print(RESULT_PREFIX + json.dumps(outcome), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
