SHELL := /bin/bash
.DEFAULT_GOAL := help

# TEST_TYPE selects which subset of tests to run (uniform knob across the
# product repos: torch-spyre, hf-adapters, spyre-inference). These tier names
# are literal, first-class values -- there is no alias-resolution layer:
#   unit        — all spyre-native tests (excludes the heavy upstream suites)
#   integration — the smoke suite. This is the ONLY valid top-level tier for
#                 that suite -- TEST_TYPE=smoke by itself is rejected (see the
#                 `tests` target below); "smoke" is just the individual suite
#                 key "integration" maps to, still usable inside a
#                 multi-suite combo (e.g. TEST_TYPE="smoke load").
#   regression  — everything
#   trunk       — same coverage as regression; push-to-main CI label (see
#                 resolve_test_type.sh)
#   perf        — SCAFFOLD ONLY: no benchmark harness yet, writes a placeholder
#                 empty JUnit XML (no .benchmark classname, so ingest reads it
#                 as 0 rows). A real producer (like torch-spyre's
#                 spyre-perf-suite) is a follow-up.
# Also accepts a space-separated list of individual suite keys (matches
# _test_matrix.yaml's `test_type` semantics), e.g. TEST_TYPE="smoke load".
# Empty / unset defaults to "regression" (every suite).
TEST_TYPE ?= regression

# MODEL_KEY narrows a suite to one model via pytest's -k substring filter
# (local dev use, e.g. `make tests MODEL_KEY=granite` to match several paths
# at once); empty = run every model in the suite.
MODEL_KEY ?=

# MODEL_PATH narrows a suite to exactly one model via pytest's --model-path
# (see tests/conftest.py's pytest_generate_tests), which replaces the
# registry-derived parametrization outright rather than filtering it -- so it
# works for any model path, including ones that lost the smallest-per-adapter
# CAUSAL_PATHS/EMBED_PATHS/VISION_PATHS representative slot. Matrix-style
# per-model CI jobs pass this (see _test_matrix.yaml); empty = no override.
MODEL_PATH ?=

# Flags passed verbatim to pytest, mirroring _test_matrix.yaml's extra_test_flags.
PYTEST_ARGS ?= -s -vvv

# Pytest invocation. Override e.g. `make adapter-coverage-tests PYTEST="python -m pytest"`
# for callers without a uv-managed venv (the adapter-coverage job runs on a bare
# ubuntu-latest runner with only `pip install pytest`, no uv/project venv).
# --active --no-sync targets the prebaked image venv ($VIRTUAL_ENV) and skips
# re-resolution: the lockfile pins torch to a +cpu build that has no ppc64le wheel,
# so any resolve fails there even though the venv already has a local torch build.
PYTEST ?= uv run --active --no-sync pytest

# When set, write JUnit XML here. Unset = no JUnit file (plain local run).
JUNIT_XML ?=

# Root all suite targets' JUnit output under one directory so a caller can glob
# it in one shot (ingest_xml.py globs non-recursively).
RESULTS_DIR ?= .

ifneq ($(MODEL_KEY),)
K_ARGS := -k "$(MODEL_KEY)"
else
K_ARGS :=
endif

ifneq ($(MODEL_PATH),)
MODEL_PATH_ARGS := --model-path "$(MODEL_PATH)"
else
MODEL_PATH_ARGS :=
endif

.PHONY: help test tests adapter-coverage-tests smoke-tests load-tests \
        token-compare-tests embed-compare-tests vlm-tests reranker-tests model-module-tests \
        masked-lm-compare-tests question-answering-compare-tests

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: TEST_TYPE=unit|integration|regression|trunk|<space-separated suite keys, e.g. smoke> (default regression),"
	@echo "  MODEL_KEY (pytest -k filter, default all), MODEL_PATH (exact model path via --model-path, overrides registry parametrization),"
	@echo "  PYTEST_ARGS (default '$(PYTEST_ARGS)'), JUNIT_XML (single-suite targets only), RESULTS_DIR (default '$(RESULTS_DIR)')"

# Suite keys, one target each -- same vocabulary and test_types membership as
# _test_matrix.yaml. Each is independently runnable with its own JUNIT_XML.
adapter-coverage-tests: ## Run adapter registry coverage check (suite key: adapter_coverage)
	$(PYTEST) -v --noconftest tests/test_adapter_coverage.py $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

smoke-tests: ## Run e2e smoke tests (suite key: smoke)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_smoke_spyre.py $(K_ARGS) $(MODEL_PATH_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

load-tests: ## Run load tests (suite key: load)
	# test_load_spyre.py is the one suite file with FOUR model_path-parametrized
	# functions (causal/embed/masked-LM/QA) sharing that fixture name. conftest's
	# --model-path override reparametrizes every function with a model_path
	# fixture, not just the one whose registry the model belongs to -- so it
	# would force the other three to load the model through the wrong auto-class
	# and fail. Filter via -k (substring on the test ID) instead, which only
	# selects the one matching parametrization, same as every other suite target
	# gets from --model-path (safe here because MODEL_PATH is always one exact
	# registry path, never an attacker-controlled or ambiguous substring).
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_load_spyre.py $(if $(MODEL_PATH),-k "$(MODEL_PATH)",$(K_ARGS)) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

token-compare-tests: ## Run token-compare tests (suite key: token_compare)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_token_compare_spyre.py $(K_ARGS) $(MODEL_PATH_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

embed-compare-tests: ## Run embed-compare tests (suite key: embed_compare)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_embed_compare_spyre.py $(K_ARGS) $(MODEL_PATH_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

vlm-tests: ## Run VLM e2e tests (suite key: vlm)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_vlm_e2e_spyre.py $(K_ARGS) $(MODEL_PATH_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

reranker-tests: ## Run reranker compare tests (suite key: reranker_compare)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_reranker_compare_spyre.py $(K_ARGS) $(MODEL_PATH_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

masked-lm-compare-tests: ## Run masked-LM compare tests (suite key: masked_lm_compare)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_masked_lm_compare_spyre.py $(K_ARGS) $(MODEL_PATH_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

question-answering-compare-tests: ## Run question-answering compare tests (suite key: question_answering_compare)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_question_answering_compare_spyre.py $(K_ARGS) $(MODEL_PATH_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

# MODULE_CONFIG narrows model-module-tests to one YAML config (matrix-style
# per-config CI jobs pass this); empty = run every config in tests/configs/module_tests.
MODULE_CONFIG ?=
# --junit-xml is resolved to an absolute path: run_test.sh cd's into each
# test file's own directory before invoking pytest, so a relative path
# would land under that directory instead of RESULTS_DIR.
model-module-tests: ## Run oot_framework module tests (suite key: model_module; MODULE_CONFIG=<file>.yaml narrows to one)
	# Env setup mirrors _test_matrix.yaml's "Run module tests" step: ibm-aiu-setup.sh
	# ends with a chmod of root-owned /tmp/etc that fails on the Spyre image; env vars
	# are already exported by then, so tolerate that failure. One logical shell line
	# (no .ONESHELL, for portability across make versions) via `\` continuations.
	set +e; \
	source "$$HOME/.bashrc"; \
	source /etc/profile.d/ibm-aiu-setup.sh; \
	set -e; \
	_run_test=$$(uv run --active --no-sync python3 -c \
	  "import oot_framework, os; print(os.path.join(os.path.dirname(oot_framework.__file__), 'run_test.sh'))") || { \
	  echo "ERROR: oot_framework is not installed in the active venv. Run 'uv sync --group oot' (see CLAUDE.md) and retry."; \
	  exit 1; \
	}; \
	configs="$(MODULE_CONFIG)"; \
	if [[ -z "$$configs" ]]; then \
	  configs=$$(cd tests/configs/module_tests && ls *.yaml); \
	fi; \
	rc=0; \
	for cfg in $$configs; do \
	  junit_arg=""; \
	  if [[ -n "$(JUNIT_XML)" ]]; then \
	    junit_arg="--junit-xml=$$(cd "$(RESULTS_DIR)" && pwd)/model-module-$${cfg}.xml"; \
	  fi; \
	  TORCH_DEVICE_ROOT="$$PWD" bash "$$_run_test" \
	    "tests/configs/module_tests/$${cfg}" $(PYTEST_ARGS) $${junit_arg} || rc=1; \
	done; \
	exit $$rc

# Aggregate target: every suite named in TEST_TYPE (unit|integration|
# regression|trunk|space-separated suite keys), each writing its own flat
# JUnit file into RESULTS_DIR so a caller can glob the whole directory in one
# ClickHouse push. One failing suite doesn't skip the rest; the aggregate's
# exit code still reflects any failure.
tests: ## Run the suites selected by TEST_TYPE into RESULTS_DIR (JUnit per suite)
	@# Apply the shared default (empty -> regression) and pass literal tier
	@# names / suite keys through unchanged -- same source of truth as
	@# _test_matrix.yaml's resolve-test-type job, so `make tests TEST_TYPE=unit`
	@# matches what CI runs for the "unit" tier via GHA.
	resolved="$$(scripts/resolve_test_type.sh $(TEST_TYPE))"; \
	case " $$resolved " in \
	  *" regression "*|*" trunk "*) suites="adapter_coverage smoke load token_compare embed_compare vlm reranker_compare masked_lm_compare question_answering_compare model_module" ;; \
	  *" unit "*) suites="adapter_coverage load token_compare embed_compare vlm reranker_compare masked_lm_compare question_answering_compare model_module" ;; \
	  " integration ") suites="smoke" ;; \
	  " perf ") suites="perf" ;; \
	  " smoke ") echo "TEST_TYPE=smoke is not a valid tier -- use TEST_TYPE=integration to run the smoke suite alone, or include 'smoke' in a multi-suite combo (e.g. TEST_TYPE=\"smoke load\")."; exit 1 ;; \
	  *) suites="$$resolved" ;; \
	esac; \
	mkdir -p "$(RESULTS_DIR)"; \
	rc=0; \
	for suite in $$suites; do \
	  echo "=== running suite: $$suite ==="; \
	  case "$$suite" in \
	    adapter_coverage) $(MAKE) adapter-coverage-tests JUNIT_XML="$(RESULTS_DIR)/adapter-coverage.xml" || rc=1 ;; \
	    smoke)            $(MAKE) smoke-tests            JUNIT_XML="$(RESULTS_DIR)/spyre-smoke-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    load)             $(MAKE) load-tests             JUNIT_XML="$(RESULTS_DIR)/spyre-load-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    token_compare)    $(MAKE) token-compare-tests     JUNIT_XML="$(RESULTS_DIR)/spyre-token-compare-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    embed_compare)    $(MAKE) embed-compare-tests     JUNIT_XML="$(RESULTS_DIR)/spyre-embed-compare-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    vlm)              $(MAKE) vlm-tests               JUNIT_XML="$(RESULTS_DIR)/spyre-vlm-e2e-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    reranker_compare) $(MAKE) reranker-tests          JUNIT_XML="$(RESULTS_DIR)/spyre-reranker-compare-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    masked_lm_compare) $(MAKE) masked-lm-compare-tests JUNIT_XML="$(RESULTS_DIR)/spyre-masked-lm-compare-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    question_answering_compare) $(MAKE) question-answering-compare-tests JUNIT_XML="$(RESULTS_DIR)/spyre-question-answering-compare-tests.xml" MODEL_KEY="$(MODEL_KEY)" || rc=1 ;; \
	    model_module)     $(MAKE) model-module-tests      JUNIT_XML=1 RESULTS_DIR="$(RESULTS_DIR)" MODULE_CONFIG="$(MODULE_CONFIG)" || rc=1 ;; \
	    perf)             printf '%s\n' \
	                        '<?xml version="1.0" encoding="utf-8"?>' \
	                        '<testsuites name="hf-adapters-perf">' \
	                        '  <testsuite name="hf-adapters-perf" tests="0" skipped="0" failures="0" errors="0"/>' \
	                        '</testsuites>' > "$(RESULTS_DIR)/report.xml"; \
	                      echo "hf-adapters has no perf harness yet (scaffold stub): wrote placeholder $(RESULTS_DIR)/report.xml" ;; \
	    *) echo "Unknown suite key '$$suite'. Valid: adapter_coverage smoke load token_compare embed_compare vlm reranker_compare masked_lm_compare question_answering_compare model_module perf"; rc=1 ;; \
	  esac; \
	done; \
	exit $$rc

test: tests  ## Alias for `tests`, matching torch-spyre's Makefile target name.
