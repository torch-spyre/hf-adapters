#!/usr/bin/env bash
# Pre-commit hook: if pyproject.toml is staged AND the change actually affects
# dependency resolution, uv.lock must be staged too.
#
# The naive "any pyproject.toml change requires a staged uv.lock" rule fires
# on tool-only edits (e.g. [tool.mypy], [tool.ruff]) that uv.lock doesn't
# track, forcing a spurious lock touch. We defer to `uv lock --check` — the
# only source of truth for whether the lockfile is stale — and only fail
# when it actually is.
#
# Run as: USE_SPYRE_CCL=0 uv lock   then git add uv.lock

set -euo pipefail

# Only care about commits that touch pyproject.toml. Other paths pass through.
if ! git diff --cached --name-only | grep -q "^pyproject\.toml$"; then
    exit 0
fi

# If the lockfile is already staged, trust the developer ran `uv lock` and let
# the commit through. (`uv lock --check` would run against the working-tree
# copy, not the staged one, and could disagree if the working tree drifted
# after `git add uv.lock`.)
if git diff --cached --name-only | grep -q "^uv\.lock$"; then
    exit 0
fi

# pyproject.toml staged, uv.lock not staged — is uv.lock actually stale?
if uv lock --check >/dev/null 2>&1; then
    # Lockfile is up-to-date w.r.t. pyproject.toml; the staged pyproject.toml
    # change is dep-neutral (tool config, metadata, etc.) so nothing to do.
    exit 0
fi

echo ""
echo "  ✗ pyproject.toml was modified but uv.lock was not updated."
echo ""
echo "  Run: uv lock"
echo "  Then: git add uv.lock"
echo ""
exit 1
