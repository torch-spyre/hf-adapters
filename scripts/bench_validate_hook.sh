#!/usr/bin/env bash
# Pre-commit hook: benchmarks/ is current with tests/model_lists/perf.yaml, validates and keeps its ingest ids.
#
# The generated-files check needs only PyYAML, so it runs unconditionally and FAILS the commit: it is
# the guard that makes tests/model_lists/perf.yaml the single source of truth, and gating it behind an
# installed engine would switch it off for everyone who has not installed one.
# Only the engine-dependent steps are advisory: with no spyre-bench on PATH they print a notice and
# pass, unless BENCH_REQUIRED=1.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if python3 -c 'import yaml' 2>/dev/null; then
    python3 scripts/generate_bench_definitions.py --check
else
    echo "bench-validate: PyYAML not available, skipping the generated-files check" >&2
fi

if ! command -v "${BENCH:-spyre-bench}" >/dev/null 2>&1; then
    echo "bench-validate: spyre-bench is not installed; skipping validation of benchmarks/" >&2
    if [[ "${BENCH_REQUIRED:-0}" == 1 ]]; then
        exit 2
    fi
    exit 0
fi

make -s bench-validate BENCH_OPTIONAL=1
BENCH_OPTIONAL=1 exec python3 benchmarks/tools/check_identity_ids.py
