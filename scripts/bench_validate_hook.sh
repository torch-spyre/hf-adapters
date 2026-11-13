#!/usr/bin/env bash
# Pre-commit hook: benchmarks/ is current with tests/model_lists/perf.yaml and validates.
# Advisory: with no spyre-bench engine installed it prints a notice and passes, unless
# BENCH_REQUIRED=1.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if ! command -v "${BENCH:-spyre-bench}" >/dev/null 2>&1; then
    echo "bench-validate: spyre-bench is not installed; skipping validation of benchmarks/" >&2
    [[ "${BENCH_REQUIRED:-0}" == 1 ]] && exit 2
    exit 0
fi

if python3 -c 'import yaml' 2>/dev/null; then
    python3 scripts/generate_bench_definitions.py --check
else
    echo "bench-validate: PyYAML not available, skipping the generated-files check" >&2
fi

exec make -s bench-validate
