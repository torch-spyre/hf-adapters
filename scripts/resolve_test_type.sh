#!/usr/bin/env bash
# Single source of truth for defaulting an empty/unset TEST_TYPE, shared by
# the Makefile `tests` target and _test_matrix.yaml's resolve-test-type job
# so both entry points apply the same default. unit/integration/regression/
# trunk are literal, first-class tier values everywhere downstream (this
# repo's Makefile suite-selection case, and every suite job's `if:` in
# _test_matrix.yaml) -- there is no alias-resolution step here anymore.
# "smoke" is a suite key, not a tier (a bare TEST_TYPE=smoke is rejected by
# the Makefile's case statement; use TEST_TYPE=integration for that suite
# alone), but it still passes through this script unchanged like any other
# suite key.
#
# Usage: resolve_test_type.sh [TEST_TYPE...]
# Each argument is passed through unchanged (empty args resolve to
# "regression"); multiple arguments are printed space-separated, so a
# space-separated suite-key combo (e.g. "smoke load") still works.

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo regression
    exit 0
fi

resolved=""
for t in "$@"; do
    if [[ -z "$t" ]]; then
        resolved="$resolved regression"
    else
        resolved="$resolved $t"
    fi
done
echo "${resolved# }"
