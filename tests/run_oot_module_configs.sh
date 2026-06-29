#!/usr/bin/env bash
# Run OOT module config tests.
#
# Usage (from any directory):
#   bash run_module_tests.sh [config.yaml | configs/dir/] [extra pytest args...]
#
# Examples:
#   bash run_module_tests.sh tests/configs/module_tests/granite_3_3_8b_instruct_spyre.yaml -v
#   bash run_module_tests.sh tests/configs/module_tests/ -v
#   bash run_module_tests.sh tests/configs/module_tests/ --junit-xml=report.xml

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml | configs/dir/> [extra pytest args...]" >&2
    exit 1
fi

# Resolve run_test.sh from the installed oot_framework package.
_run_test=$(python3 -c "
import oot_framework, os
p = os.path.join(os.path.dirname(oot_framework.__file__), 'run_test.sh')
if not os.path.isfile(p):
    raise FileNotFoundError(f'run_test.sh not found at {p}')
print(p)
") || {
    echo "ERROR: oot_framework is not installed." >&2
    echo "       Run: uv sync --group oot" >&2
    exit 1
}

cd "$REPO_ROOT"
TORCH_DEVICE_ROOT="$REPO_ROOT" bash "$_run_test" "$@"
