SHELL := /bin/bash
.DEFAULT_GOAL := help

# TEST_TYPE selects which subset of tests to run (uniform knob across the
# product repos: torch-spyre, hf-adapters, spyre-inference):
#   smoke — fast per-op unit tests only
#   core  — all spyre-native tests (excludes the heavy upstream suites)
#   full  — everything (default)
# Also accepts a space-separated list of individual suite keys (matches
# _test_matrix.yaml's `test_type` semantics), e.g. TEST_TYPE="smoke load".
TEST_TYPE ?= full

# MODEL_KEY narrows a suite to one model via pytest's -k filter (matrix-style
# per-model CI jobs pass this); empty = run every model in the suite.
MODEL_KEY ?=

# Flags passed verbatim to pytest, mirroring _test_matrix.yaml's extra_test_flags.
PYTEST_ARGS ?= -s -vvv

# Pytest invocation. Override e.g. `make adapter-coverage-tests PYTEST="python -m pytest"`
# for callers without a uv-managed venv (the adapter-coverage job runs on a bare
# ubuntu-latest runner with only `pip install pytest`, no uv/project venv).
PYTEST ?= uv run pytest

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

.PHONY: help test tests adapter-coverage-tests smoke-tests load-tests \
        token-compare-tests embed-compare-tests vlm-tests model-module-tests

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[0-9a-zA-Z_-]+:.*?## / {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Variables: TEST_TYPE=smoke|core|full|<space-separated suite keys> (default full),"
	@echo "  MODEL_KEY (pytest -k filter, default all), PYTEST_ARGS (default '$(PYTEST_ARGS)'),"
	@echo "  JUNIT_XML (single-suite targets only), RESULTS_DIR (default '$(RESULTS_DIR)')"

# Suite keys, one target each -- same vocabulary and test_types membership as
# _test_matrix.yaml. Each is independently runnable with its own JUNIT_XML.
adapter-coverage-tests: ## Run adapter registry coverage check (suite key: adapter_coverage)
	$(PYTEST) -v --noconftest tests/test_adapter_coverage.py $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

smoke-tests: ## Run e2e smoke tests (suite key: smoke)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_smoke_spyre.py $(K_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

load-tests: ## Run load tests (suite key: load)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_load_spyre.py $(K_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

token-compare-tests: ## Run token-compare tests (suite key: token_compare)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_token_compare_spyre.py $(K_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

embed-compare-tests: ## Run embed-compare tests (suite key: embed_compare)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_e2e_embed_compare_spyre.py $(K_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

vlm-tests: ## Run VLM e2e tests (suite key: vlm)
	$(PYTEST) $(PYTEST_ARGS) tests/spyre/test_vlm_e2e_spyre.py $(K_ARGS) $(if $(JUNIT_XML),--junitxml=$(JUNIT_XML))

# MODULE_CONFIG narrows model-module-tests to one YAML config (matrix-style
# per-config CI jobs pass this); empty = run every config in tests/configs/module_tests.
MODULE_CONFIG ?=
model-module-tests: ## Run oot_framework module tests (suite key: model_module; MODULE_CONFIG=<file>.yaml narrows to one)
	# Env setup mirrors _test_matrix.yaml's "Run module tests" step: ibm-aiu-setup.sh
	# ends with a chmod of root-owned /tmp/etc that fails on the Spyre image; env vars
	# are already exported by then, so tolerate that failure. One logical shell line
	# (no .ONESHELL, for portability across make versions) via `\` continuations.
	set +e; \
	source "$$HOME/.bashrc"; \
	source /etc/profile.d/ibm-aiu-setup.sh; \
	set -e; \
	_run_test=$$(uv run python3 -c \
	  "import oot_framework, os; print(os.path.join(os.path.dirname(oot_framework.__file__), 'run_test.sh'))"); \
	configs="$(MODULE_CONFIG)"; \
	if [[ -z "$$configs" ]]; then \
	  configs=$$(cd tests/configs/module_tests && ls *.yaml); \
	fi; \
	rc=0; \
	for cfg in $$configs; do \
	  junit_arg=""; \
	  if [[ -n "$(JUNIT_XML)" ]]; then \
	    junit_arg="--junit-xml=$(RESULTS_DIR)/model-module-$${cfg}.xml"; \
	  fi; \
	  TORCH_DEVICE_ROOT="$$PWD" bash "$$_run_test" \
	    "tests/configs/module_tests/$${cfg}" $(PYTEST_ARGS) $${junit_arg} || rc=1; \
	done; \
	exit $$rc

# Aggregate target: every suite named in TEST_TYPE (smoke|core|full|space-separated
# keys), each writing its own flat JUnit file into RESULTS_DIR so a caller can glob
# the whole directory in one ClickHouse push. One failing suite doesn't skip the
# rest; the aggregate's exit code still reflects any failure.
test: ## Run the suites selected by TEST_TYPE into RESULTS_DIR (JUnit per suite)
	case " $(TEST_TYPE) " in \
	  *" full "*) suites="adapter_coverage smoke load token_compare embed_compare vlm model_module" ;; \
	  *" core "*) suites="adapter_coverage load token_compare embed_compare vlm model_module" ;; \
	  " smoke ") suites="smoke" ;; \
	  *) suites="$(TEST_TYPE)" ;; \
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
	    model_module)     $(MAKE) model-module-tests      JUNIT_XML=1 RESULTS_DIR="$(RESULTS_DIR)" MODULE_CONFIG="$(MODULE_CONFIG)" || rc=1 ;; \
	    *) echo "Unknown suite key '$$suite'. Valid: adapter_coverage smoke load token_compare embed_compare vlm model_module"; rc=1 ;; \
	  esac; \
	done; \
	exit $$rc

tests: test  ## Alias for `test`.
