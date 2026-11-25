#!/usr/bin/env bash
# Pre-commit hook: keep resources/adapter_added_dates.json in step with the
# adapter files (issue #372).
#
# The weekly scan reads that committed JSON for each adapter's `added_date`
# instead of deriving it from git at scan time, because the scan runs inside a
# `git clone --depth=1` checkout where `git log` reports the wrong (tip) date.
# The JSON is authoritative. Add a new adapter's UTC introduction date in the
# same commit; preserve existing dates across squash merges and renames.
#
# This wrapper only scopes the work to relevant commits and delegates the actual
# logic to `regenerate_adapter_added_dates.py --check-hook`, which is unit-tested
# (tests/test_adapter_added_dates.py). That check NEVER writes; it enforces:
#   * every adapter, including new files, has a valid ISO date; and
#   * removed adapter names no longer appear in the map.
# No git history is read or compared with dates.

set -euo pipefail

REGEN="tests/spyre/weekly_generation/regenerate_adapter_added_dates.py"
DATES_JSON="resources/adapter_added_dates.json"

# Only care about commits that stage an adapter file or the JSON itself. Any
# other commit is date-neutral and passes through untouched.
STAGED_FILES="$(git diff --cached --name-only)"
if ! grep -Eq "^hf_adapters/hf_.*\.py$|^${DATES_JSON//./\\.}$" <<< "$STAGED_FILES"; then
    exit 0
fi

# Only stdlib + git are needed; do not resolve or sync project dependencies.
exec python3 "$REGEN" --check-hook
