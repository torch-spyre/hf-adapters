#!/usr/bin/env bash
# Pre-commit hook: if pyproject.toml is staged, uv.lock must be staged too.
# Run as: USE_SPYRE_CCL=0 uv lock   then git add uv.lock

set -euo pipefail

if git diff --cached --name-only | grep -q "^pyproject\.toml$"; then
    if ! git diff --cached --name-only | grep -q "^uv\.lock$"; then
        echo ""
        echo "  ✗ pyproject.toml was modified but uv.lock was not updated."
        echo ""
        echo "  Run: uv lock"
        echo "  Then: git add uv.lock"
        echo ""
        exit 1
    fi
fi
