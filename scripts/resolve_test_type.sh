#!/usr/bin/env bash
# Single source of truth for the unit/integration/regression tier aliases,
# shared by the Makefile `tests` target and _test_matrix.yaml's
# resolve-test-type job so both entry points apply the same mapping.
#
# Usage: resolve_test_type.sh [TEST_TYPE...]
# Each argument is resolved independently and printed space-separated.
# Empty/no args resolves to "full" (this repo's own default label; callers
# that want the user-facing "regression" default pass it explicitly).

set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo full
    exit 0
fi

resolved=""
for t in "$@"; do
    case "$t" in
        unit)        resolved="$resolved core" ;;
        integration) resolved="$resolved smoke" ;;
        regression)  resolved="$resolved full" ;;
        '')          resolved="$resolved full" ;;
        *)           resolved="$resolved $t" ;;
    esac
done
echo "${resolved# }"
