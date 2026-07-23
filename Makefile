SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# TEST_TYPE selects which subset of the on-pod Spyre tests `make test` runs
# (uniform knob across the product repos: torch-spyre, spyre-inference,
# hf-adapters). Here it maps to a model subset. Local runs can't reproduce the
# full GHA fan-out matrix; this is a best-effort local proxy.
#   smoke — 3 fixed causal models, smoke e2e file only (~10 min target)
#   core  — same as full for now (no distinct middle tier yet)
#   full  — every tests/spyre test over the representative model set (default)
TEST_TYPE ?= full

# Flags passed verbatim to pytest (matches CI verbosity). Override e.g.
#   make test PYTEST_ARGS="-x -q"
PYTEST_ARGS ?= -s -vvv

# The 3 smoke models, as pytest -k path substrings (tests parametrize over the
# HF model_path, so the test id is the path). Keep in sync with
# tests/model_registry.py SMOKE_CAUSAL_KEYS.
SMOKE_K := Qwen3-0.6B or Ministral-3B-Instruct or granite-4.0-1b-base

# pyproject sets addopts = --ignore=tests/spyre, so every target must pass an
# explicit tests/spyre path to override the ignore.
ifeq ($(TEST_TYPE),smoke)
TEST_SELECTION := tests/spyre/test_e2e_smoke_spyre.py -k "$(SMOKE_K)"
else ifeq ($(TEST_TYPE),core)
TEST_SELECTION := tests/spyre
else ifeq ($(TEST_TYPE),full)
TEST_SELECTION := tests/spyre
else
$(error Invalid TEST_TYPE '$(TEST_TYPE)'. Valid values: smoke | core | full)
endif

.PHONY: help test tests

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: TEST_TYPE=smoke|core|full (default full), PYTEST_ARGS (default '$(PYTEST_ARGS)')"

test: ## Run Spyre tests on-pod. Narrow with TEST_TYPE=smoke|core|full (default full).
	# Port of the CI env setup: ibm-aiu-setup.sh ends with a chmod of root-owned
	# /tmp/etc that fails on the Spyre image; env vars are already exported by
	# then, so tolerate that failure.
	unset _IBM_AIU_SETUP
	rm -f /tmp/etc/ibm/spyre/topo.json
	set +e
	source "$$HOME/.bashrc"
	source /etc/profile.d/ibm-aiu-setup.sh
	set -e
	echo "Running Spyre tests for TEST_TYPE=$(TEST_TYPE)..."
	uv run pytest $(PYTEST_ARGS) $(TEST_SELECTION)

tests: test  ## Alias for `test`.
