# Thin wrappers over the spyre-bench CLI. Parameters are variables, exported as BENCH_*
# for the CLI; there is no per-architecture logic here.
# Needs the engine on PATH (override with BENCH=...). When the engine, or the verb in the installed
# version, is missing, a target says how to install it and exits 2; BENCH_OPTIONAL=1 turns that into
# a notice and exit 0 (for callers that treat the engine as optional). BENCH_* are exported to
# bench-* recipes only.
# CPU pinning is requested at plan time (BENCH_CPU_PINNING, BENCH_CPU_SET, BENCH_PIN_OPTS, BENCH_REQUIRE_PINNING)
# and applied by the engine when it runs the plan; BENCH_PREFIX must not be used for it.
# BENCH_MODE (off|side-by-side|replace) only affects `bench-perf`, which `make tests TEST_TYPE=perf` calls.

BENCH              ?= spyre-bench
BENCH_DIR          ?= benchmarks
BENCH_CENTRAL      ?=
BENCH_PACK         ?=
BENCH_PREFIX       ?=
BENCH_BASE_DIR     ?=
BENCH_PURPOSE      ?= on-demand
BENCH_ENVIRONMENT  ?= one-device
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
BENCH_ENGINE_REF   ?= main

_bench_env = env BENCH_PURPOSE='$(BENCH_PURPOSE)' BENCH_ENVIRONMENT='$(BENCH_ENVIRONMENT)' BENCH_SELECTION='$(BENCH_SELECTION)' BENCH_CASES='$(BENCH_CASES)' BENCH_TIER='$(BENCH_TIER)' BENCH_TYPE='$(BENCH_TYPE)' BENCH_SUITES='$(BENCH_SUITES)' BENCH_PLAN='$(BENCH_PLAN)' BENCH_SET='$(BENCH_SET)' BENCH_REUSE='$(BENCH_REUSE)' BENCH_INGEST='$(BENCH_INGEST)' BENCH_INVOKER='$(BENCH_INVOKER)' BENCH_ARCH='$(BENCH_ARCH)' BENCH_SUBJECT_ID='$(BENCH_SUBJECT_ID)' BENCH_RUN_ID='$(BENCH_RUN_ID)' BENCH_QUIET='$(BENCH_QUIET)' BENCH_PASS_ENV='$(_bench_pass)' BENCH_PERF_RESULTS='$(BENCH_PERF_RESULTS)' BENCH_PACK='$(BENCH_PACK)' BENCH_CPU_PINNING='$(BENCH_CPU_PINNING)' BENCH_CPU_SET='$(BENCH_CPU_SET)' BENCH_PIN_OPTS='$(BENCH_PIN_OPTS)' BENCH_REQUIRE_PINNING='$(BENCH_REQUIRE_PINNING)'
_comma := ,
_bench_pass = $(BENCH_PASS_ENV)$(if $(BENCH_PERF_RESULTS),$(if $(BENCH_PASS_ENV),$(_comma))BENCH_PERF_RESULTS)
_bench_central = $(if $(BENCH_CENTRAL), --central '$(BENCH_CENTRAL)')
_bench_cat = --catalog '$(BENCH_DIR)'$(_bench_central)

_bench_install = uv pip install 'spyre-bench[ingest] @ git+https://github.com/torch-spyre/torch-spyre@$(BENCH_ENGINE_REF)\#subdirectory=extensions/spyre-bench'
# $(call _bench_do,<verb>,<command>): run the command when the engine provides the verb; otherwise fail
# with the install command, or skip with a notice when BENCH_OPTIONAL=1.
_bench_do = if command -v $(firstword $(BENCH)) >/dev/null 2>&1 && $(BENCH) --help 2>&1 | grep -q '^  $(1) '; then $(2); else echo "bench: '$(BENCH)' with a '$(1)' verb is not available. Install spyre-bench (and the ingest, from a torch-spyre checkout): $(_bench_install)" >&2; if [ "$(BENCH_OPTIONAL)" = 1 ]; then echo "bench: skipped (BENCH_OPTIONAL=1)" >&2; else exit 2; fi; fi
_bench_plan_cmd = $(_bench_env) $(BENCH) plan $(_bench_cat) --out '$(BENCH_PLAN_FILE)'

.PHONY: bench-validate bench-list bench-explain bench-plan bench-run bench-verdict bench-check-ids bench-collect bench-smoke bench-report bench-perf bench-quality


bench-validate: ## Validate benchmark definitions (no hardware, no network); BENCH_BASE_DIR=<dir> reports series changes.
	@$(call _bench_do,validate,$(_bench_env) $(BENCH) validate '$(BENCH_DIR)'$(_bench_central)$(if $(BENCH_BASE_DIR), --base-dir '$(BENCH_BASE_DIR)'))

bench-check-ids: ## Check the identity blocks derive the ids the ingest writes (needs the engine, the ingest, lxml, pyyaml).
	@python3 $(BENCH_DIR)/tools/check_identity_ids.py

bench-list: ## List benchmark cases and named selections.
	@$(call _bench_do,list,$(_bench_env) $(BENCH) list $(_bench_cat))

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

bench-quality: ## Score the evals/ question samples on the device as quality results (Selection model-correctness).
	@$(MAKE) bench-run BENCH_TYPE=quality BENCH_SELECTION=model-correctness BENCH_CASES=

bench-smoke: ## Run the `smoke` selection once, as an on-demand check of the harness.
	@$(MAKE) bench-run BENCH_SELECTION=smoke BENCH_CASES= BENCH_PURPOSE=on-demand BENCH_SET=run.repeats=1 BENCH_INGEST=0

bench-report: ## Summarise BENCH_RESULTS_DIR.
	@$(call _bench_do,report,$(_bench_env) $(BENCH) report '$(BENCH_RESULTS_DIR)')

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
