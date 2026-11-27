# Thin wrappers over the spyre-bench CLI. Parameters are variables, exported as BENCH_*
# for the CLI; there is no per-architecture logic here.
# Needs the engine on PATH (override with BENCH=...). When the engine, or the verb in the installed
# version, is missing, a target says how to install it and exits 2; BENCH_OPTIONAL=1 turns that into
# a notice and exit 0 (for callers that treat the engine as optional). BENCH_* are exported to
# bench-* recipes only.
# CPU pinning is requested at plan time (BENCH_CPU_PINNING, BENCH_CPU_SET, BENCH_PIN_OPTS, BENCH_REQUIRE_PINNING)
# and applied by the engine when it runs the plan; BENCH_PREFIX must not be used for it.
# BENCH_MODE (off|side-by-side|replace) only affects `bench-perf`, which `make tests TEST_TYPE=perf` calls.
#
# benchmarks/ is GENERATED from tests/model_lists/perf.yaml, which stays the single source of truth
# (bench-generate writes it, bench-check-catalog fails when it is stale), and every case runs through
# bench-model-case / bench-eval-case below -- so each runner's path is written down once, here.

BENCH              ?= spyre-bench
BENCH_DIR          ?= benchmarks
BENCH_CENTRAL      ?=
BENCH_PACK         ?=
BENCH_PREFIX       ?=
BENCH_BASE_DIR     ?=
BENCH_PURPOSE      ?= on-demand
# From the engine's shared catalog, not redeclared here.
BENCH_ENVIRONMENT  ?= spyre-1card
BENCH_SELECTION    ?=
BENCH_CASES        ?=
BENCH_TIER         ?=
BENCH_TYPE         ?=
BENCH_SUITES       ?=
BENCH_PLAN         ?=
BENCH_SET          ?=
BENCH_REUSE        ?= never
BENCH_INGEST       ?= 0
BENCH_INVOKER      ?= shell
BENCH_ARCH         ?=
BENCH_SUBJECT_ID   ?=
BENCH_RUN_ID       ?=
BENCH_QUIET        ?= $(if $(CI),1,0)
BENCH_PASS_ENV     ?=
BENCH_CPU_PINNING  ?=
BENCH_CPU_SET      ?=
BENCH_PIN_OPTS     ?=
BENCH_REQUIRE_PINNING ?=
BENCH_OUT          ?= results/bench
BENCH_RESULTS_DIR  ?= $(BENCH_OUT)
BENCH_PLAN_FILE    ?= $(BENCH_RESULTS_DIR)/plan/plan.json
BENCH_MODE         ?= off
BENCH_PERF_RESULTS ?=
CASE               ?=
BENCH_OPTIONAL     ?=
BENCH_REQUIRED     ?=
BENCH_ENGINE_REF   ?= main

# Per-case variables a run.command sets; see bench-model-case and bench-eval-case.
BENCH_CASE_MODEL   ?=
BENCH_CASE_NAME    ?=
BENCH_CASE_CONFIG  ?=
BENCH_CASE_MODE    ?=
BENCH_SUITE_DIR    ?=
BENCH_RAW_DIR      ?=
BENCH_REPEAT_DIR   ?=
BENCH_EVAL_TASK    ?=
BENCH_EVAL_MODEL   ?=
BENCH_EVAL_LIMIT   ?=

_bench_env = env BENCH_PURPOSE='$(BENCH_PURPOSE)' BENCH_ENVIRONMENT='$(BENCH_ENVIRONMENT)' BENCH_SELECTION='$(BENCH_SELECTION)' BENCH_CASES='$(BENCH_CASES)' BENCH_TIER='$(BENCH_TIER)' BENCH_TYPE='$(BENCH_TYPE)' BENCH_SUITES='$(BENCH_SUITES)' BENCH_PLAN='$(BENCH_PLAN)' BENCH_SET='$(BENCH_SET)' BENCH_REUSE='$(BENCH_REUSE)' BENCH_INGEST='$(BENCH_INGEST)' BENCH_INVOKER='$(BENCH_INVOKER)' BENCH_ARCH='$(BENCH_ARCH)' BENCH_SUBJECT_ID='$(BENCH_SUBJECT_ID)' BENCH_RUN_ID='$(BENCH_RUN_ID)' BENCH_QUIET='$(BENCH_QUIET)' BENCH_PASS_ENV='$(_bench_pass)' BENCH_PERF_RESULTS='$(BENCH_PERF_RESULTS)' BENCH_PACK='$(BENCH_PACK)' BENCH_CPU_PINNING='$(BENCH_CPU_PINNING)' BENCH_CPU_SET='$(BENCH_CPU_SET)' BENCH_PIN_OPTS='$(BENCH_PIN_OPTS)' BENCH_REQUIRE_PINNING='$(BENCH_REQUIRE_PINNING)'
_comma := ,
_bench_pass = $(BENCH_PASS_ENV)$(if $(BENCH_PERF_RESULTS),$(if $(BENCH_PASS_ENV),$(_comma))BENCH_PERF_RESULTS)
_bench_install = uv pip install 'spyre-bench[ingest] @ git+https://github.com/torch-spyre/torch-spyre@$(BENCH_ENGINE_REF)\#subdirectory=extensions/spyre-bench'

# The card-count Environments this catalog refers to ship inside the engine (spyre_bench/catalogs/),
# so a plain `make bench-validate` resolves with no extra arguments. `--central` is repeatable and
# BENCH_CENTRAL arrives on the command line from a CI leg, so the two COMPOSE: dropping the shared
# one when a leg supplies its own would leave every Environment reference dangling.
BENCH_SHARED_CATALOG ?= $(shell $(BENCH) catalogs --path llm 2>/dev/null)
_bench_central = $(if $(BENCH_SHARED_CATALOG), --central '$(BENCH_SHARED_CATALOG)')$(if $(BENCH_CENTRAL), --central '$(BENCH_CENTRAL)')
_bench_cat = --catalog '$(BENCH_DIR)'$(_bench_central)

# Without a shared catalog the Environment this catalog names is unresolvable, and the engine
# reports one error per reference whose actual cause is that the installed engine predates
# spyre_bench/catalogs/. Say the cause once instead. An operator supplying BENCH_CENTRAL has
# provided the resources themselves, so the check only fires when both are empty. It is part of
# _bench_do rather than a line of its own, so one BENCH_OPTIONAL skip covers the whole recipe.
_bench_no_central = \
	echo "bench: no shared catalog, so every Environment reference would dangle." >&2; \
	echo "       '$(BENCH) catalogs --path llm' returned nothing: the installed engine predates" >&2; \
	echo "       spyre_bench/catalogs/. Install an engine that ships it:" >&2; \
	echo "         $(_bench_install)" >&2; \
	echo "       or pass BENCH_CENTRAL=<dir> holding the shared resources." >&2
# $(call _bench_do,<verb>,<command>): run the command when the engine provides the verb and the
# shared resources resolve; otherwise fail with the cause and the install command, or skip with a
# notice when BENCH_OPTIONAL=1 (BENCH_REQUIRED=1 wins).
_bench_soft = $(if $(BENCH_REQUIRED),,$(BENCH_OPTIONAL))
_bench_do = if ! command -v $(firstword $(BENCH)) >/dev/null 2>&1 || ! $(BENCH) --help 2>&1 | grep -q '^  $(1) '; then echo "bench: '$(BENCH)' with a '$(1)' verb is not available. Install spyre-bench (and the ingest, from a torch-spyre checkout): $(_bench_install)" >&2; if [ "$(_bench_soft)" = 1 ]; then echo "bench: skipped (BENCH_OPTIONAL=1)" >&2; exit 0; fi; exit 2; elif [ -z "$(strip $(BENCH_SHARED_CATALOG))" ] && [ -z "$(strip $(BENCH_CENTRAL))" ]; then $(_bench_no_central); if [ "$(_bench_soft)" = 1 ]; then echo "bench: skipped (BENCH_OPTIONAL=1)" >&2; exit 0; fi; exit 2; else $(2); fi
_bench_plan_cmd = $(_bench_env) $(BENCH) plan $(_bench_cat) --out '$(BENCH_PLAN_FILE)'

.PHONY: bench-validate bench-list bench-explain bench-plan bench-run bench-verdict bench-check-ids bench-collect bench-smoke bench-perf bench-quality \
	bench-generate bench-check-catalog bench-model-case bench-eval-case


bench-validate: ## Validate benchmark definitions (no hardware, no network); BENCH_BASE_DIR=<dir> reports series changes.
	@$(call _bench_do,validate,$(_bench_env) $(BENCH) validate '$(BENCH_DIR)'$(_bench_central)$(if $(BENCH_BASE_DIR), --base-dir '$(BENCH_BASE_DIR)'))

bench-check-ids: ## Check the identity blocks derive the ids the ingest writes (needs the engine, the ingest, lxml, pyyaml).
	@python3 $(BENCH_DIR)/tools/check_identity_ids.py

bench-generate: ## Regenerate benchmarks/hf-models/ and selections.yaml from tests/model_lists/perf.yaml. Hardware-free, no engine.
	@python3 scripts/generate_bench_definitions.py

bench-check-catalog: ## Fail if benchmarks/ is stale against tests/model_lists/perf.yaml. Hardware-free, no engine.
	@python3 scripts/generate_bench_definitions.py --check

# `ids` is the verb that lists the cases with their stored ids; the engine has no `list`.
bench-list: ## List benchmark cases with the stored id of each.
	@$(call _bench_do,ids,$(_bench_env) $(BENCH) ids $(_bench_cat))

bench-explain: ## Show one case with every reference resolved and its ids (CASE=<case name>).
	@test -n "$(CASE)" || { echo "CASE=<case name> is required" >&2; exit 2; }
	@$(call _bench_do,explain,$(_bench_env) $(BENCH) explain '$(CASE)' $(_bench_cat)$(if $(BENCH_ENVIRONMENT), --environment '$(BENCH_ENVIRONMENT)'))

bench-plan: ## Write BENCH_PLAN_FILE (default BENCH_RESULTS_DIR/plan/plan.json) for the selected cases; runs nothing.
	@$(call _bench_do,plan,mkdir -p '$(dir $(BENCH_PLAN_FILE))' && $(_bench_plan_cmd))

bench-run: ## Plan (unless BENCH_PLAN=<plan.json>), then run on this machine's devices; writes BENCH_RESULTS_DIR.
	@$(call _bench_do,run,$(if $(BENCH_PLAN),:,mkdir -p '$(dir $(BENCH_PLAN_FILE))' && $(_bench_plan_cmd)) && mkdir -p '$(BENCH_RESULTS_DIR)' && $(BENCH_PREFIX) $(_bench_env) $(BENCH) run $(_bench_cat) --plan '$(or $(BENCH_PLAN),$(BENCH_PLAN_FILE))' --out '$(BENCH_RESULTS_DIR)')

bench-collect: ## Gather the ingest input (report XML) of the cases that ran ok into BENCH_RESULTS_DIR/ingest-xml.
	@python3 $(BENCH_DIR)/tools/collect_ingest_xml.py '$(BENCH_RESULTS_DIR)' '$(BENCH_RESULTS_DIR)/ingest-xml'

bench-verdict: ## Verdict words for the run (BENCH_RUN_ID, else run.json stored_run_ids); fails when no stored id is known or no rows exist.
	@$(call _bench_do,verdict,id='$(BENCH_RUN_ID)'; [ -n "$$id" ] || id=$$(python3 $(BENCH_DIR)/tools/stored_run_id.py '$(BENCH_RESULTS_DIR)') || exit 2; $(BENCH) verdict --run-id "$$id" --json > '$(BENCH_RESULTS_DIR)/verdict.json' && python3 $(BENCH_DIR)/tools/stored_run_id.py --check-verdict '$(BENCH_RESULTS_DIR)/verdict.json')

# No BENCH_TYPE here: the model-correctness Selection declares `types: [quality]` itself, so a
# selector that stopped selecting would fail this target loudly instead of being papered over.
bench-quality: ## Score the evals/ question samples on the device as quality results (Selection model-correctness).
	@$(MAKE) bench-run BENCH_SELECTION=model-correctness BENCH_CASES=

bench-smoke: ## Run the `smoke` selection once, as an on-demand check of the harness.
	@$(MAKE) bench-run BENCH_SELECTION=smoke BENCH_CASES= BENCH_PURPOSE=on-demand BENCH_SET=run.repeats=1 BENCH_INGEST=0

# No bench-report target: the engine has no `report` verb. A run's own summary is in
# BENCH_RESULTS_DIR/run.json, and `bench-verdict` reads the stored rows.

# The two places a runner's location is written down: every Benchmark's `run.command` calls one of
# these targets, so moving a script is a one-line change here rather than an edit to 28 definitions
# (26 of which are generated). Not called by hand -- the engine invokes them per case.
bench-model-case: ## Run one hf-models perf case (used by a generated case's run.command).
	@test -n "$(BENCH_CASE_NAME)" || { echo "usage: make bench-model-case BENCH_CASE_MODEL=... BENCH_CASE_NAME=... BENCH_CASE_CONFIG=... BENCH_CASE_MODE=... BENCH_SUITE_DIR=... BENCH_RAW_DIR=..." >&2; exit 2; }
	@python3 $(BENCH_DIR)/tools/run_model_case.py \
		--model '$(BENCH_CASE_MODEL)' \
		--name '$(BENCH_CASE_NAME)' \
		--config '$(BENCH_CASE_CONFIG)' \
		--mode '$(BENCH_CASE_MODE)' \
		--suite-dir '$(BENCH_SUITE_DIR)' \
		--raw-dir '$(BENCH_RAW_DIR)' \
		$(if $(BENCH_REPEAT_DIR),--repeat-dir '$(BENCH_REPEAT_DIR)')

bench-eval-case: ## Score one evals/ question set (used by a model-correctness case's run.command).
	@test -n "$(BENCH_EVAL_TASK)" || { echo "usage: make bench-eval-case BENCH_EVAL_TASK=gsm8k|mmlu CASE=... BENCH_EVAL_MODEL=... BENCH_EVAL_LIMIT=... BENCH_REPEAT_DIR=... BENCH_RAW_DIR=..." >&2; exit 2; }
	@python3 $(BENCH_DIR)/tools/run_eval_case.py \
		--task '$(BENCH_EVAL_TASK)' \
		--case '$(CASE)' \
		--model '$(BENCH_EVAL_MODEL)' \
		--limit '$(BENCH_EVAL_LIMIT)' \
		--repeat-dir '$(BENCH_REPEAT_DIR)' \
		--raw-dir '$(BENCH_RAW_DIR)'

# What `make tests TEST_TYPE=perf` calls after (side-by-side) or instead of (replace) perf-tests, for MODEL_PATH.
# Advisory in side-by-side (a failure never fails the legacy path unless BENCH_REQUIRED=1); in replace it is the only path, so it fails the run. Reads the legacy reports in RESULTS_DIR, so nothing is measured twice;
# the rows reach the warehouse only through the legacy ingest of RESULTS_DIR (BENCH_INGEST stays 0).
bench-perf: ## BENCH_MODE=side-by-side|replace: run MODEL_PATH's cases through the engine; off does nothing.
	@case "$(BENCH_MODE)" in \
	  off|"") exit 0 ;; \
	  side-by-side|replace) ;; \
	  *) echo "ERROR: BENCH_MODE must be off, side-by-side or replace (got '$(BENCH_MODE)')" >&2; exit 2 ;; \
	esac; \
	[ "$(RESULTS_DIR)" != . ] || { echo "ERROR: bench-perf needs RESULTS_DIR (the directory perf-tests wrote)" >&2; exit 2; }; \
	models="$$($(PYTHON) -c "import sys; sys.path.insert(0, 'tests'); import model_registry as mr; print(' '.join(mr._exclude(sys.argv[1:])))" $(MODEL_PATH))" || { echo "bench: could not apply tests/model_lists/exclude.yaml" >&2; exit 1; }; \
	cases="$$($(PYTHON) scripts/bench_model_cases.py $$models)" || { echo "bench: could not list the cases of $$models" >&2; exit 1; }; \
	[ -n "$$cases" ] || { echo "bench: no declared case for $(MODEL_PATH)" >&2; [ "$(BENCH_MODE)" = side-by-side ]; exit $$?; }; \
	$(MAKE) bench-run BENCH_CASES="$$cases" BENCH_SELECTION= BENCH_INGEST=0 \
	  BENCH_RESULTS_DIR='$(abspath $(RESULTS_DIR))-bench' $(if $(filter side-by-side,$(BENCH_MODE)),BENCH_PERF_RESULTS='$(abspath $(RESULTS_DIR))')
