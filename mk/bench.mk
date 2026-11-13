# Thin wrappers over the spyre-bench CLI. Parameters are variables, exported as BENCH_*
# for the CLI; there is no per-architecture logic here.
# Needs the engine on PATH (override with BENCH=...). Advisory: when the engine, or the
# verb in the installed version, is missing, a target prints a notice and exits 0; with
# BENCH_REQUIRED=1 it exits 2. BENCH_* are exported to bench-* recipes only.

BENCH              ?= spyre-bench
BENCH_DIR          ?= benchmarks
BENCH_CENTRAL      ?=
BENCH_PREFIX       ?=
BENCH_BASE_DIR     ?=
BENCH_PURPOSE      ?= on-demand
BENCH_ENVIRONMENT  ?=
BENCH_SELECTION    ?=
BENCH_CASES        ?=
BENCH_TIER         ?=
BENCH_SUITES       ?=
BENCH_PLAN         ?=
BENCH_SET          ?=
BENCH_OUT          ?= results/bench
CASE               ?=
BENCH_REQUIRED     ?=

_bench_env = env BENCH_PURPOSE='$(BENCH_PURPOSE)' BENCH_ENVIRONMENT='$(BENCH_ENVIRONMENT)' BENCH_SELECTION='$(BENCH_SELECTION)' BENCH_CASES='$(BENCH_CASES)' BENCH_TIER='$(BENCH_TIER)' BENCH_SUITES='$(BENCH_SUITES)' BENCH_PLAN='$(BENCH_PLAN)' BENCH_SET='$(BENCH_SET)'
_bench_central = $(if $(BENCH_CENTRAL), --central '$(BENCH_CENTRAL)')
_bench_cat = --catalog '$(BENCH_DIR)'$(_bench_central)

# $(call _bench_do,<verb>,<command>): run the command when the engine provides the verb.
_bench_do = if command -v $(firstword $(BENCH)) >/dev/null 2>&1 && $(BENCH) --help 2>&1 | grep -q '^  $(1) '; then $(2); else echo "bench: spyre-bench with a '$(1)' verb is not available; skipping" >&2; [ "$(BENCH_REQUIRED)" != 1 ]; fi

.PHONY: bench-validate bench-list bench-explain bench-plan bench-run bench-smoke bench-report


bench-validate: ## Validate benchmark definitions (no hardware, no network); BENCH_BASE_DIR=<dir> reports series changes.
	@$(call _bench_do,validate,$(_bench_env) $(BENCH) validate '$(BENCH_DIR)'$(_bench_central)$(if $(BENCH_BASE_DIR), --base-dir '$(BENCH_BASE_DIR)'))

bench-list: ## List benchmark cases and named selections.
	@$(call _bench_do,list,$(_bench_env) $(BENCH) list $(_bench_cat))

bench-explain: ## Show one case with every reference resolved and its ids (CASE=<case name>).
	@test -n "$(CASE)" || { echo "CASE=<case name> is required" >&2; exit 2; }
	@$(call _bench_do,explain,$(_bench_env) $(BENCH) explain '$(CASE)' $(_bench_cat)$(if $(BENCH_ENVIRONMENT), --environment '$(BENCH_ENVIRONMENT)'))

bench-plan: ## Write BENCH_OUT/plan.json for the selected cases; runs nothing.
	@$(call _bench_do,plan,mkdir -p '$(BENCH_OUT)' && $(_bench_env) $(BENCH) plan $(_bench_cat) --out '$(BENCH_OUT)/plan.json')

bench-run: ## Run the plan (or BENCH_PLAN=<plan.json>) on this machine's devices; writes BENCH_OUT.
	@$(call _bench_do,run,mkdir -p '$(BENCH_OUT)' && $(BENCH_PREFIX) $(_bench_env) $(BENCH) run $(_bench_cat) --out '$(BENCH_OUT)')

bench-smoke: ## Run the `smoke` selection once, as an on-demand check of the harness.
	@$(MAKE) bench-run BENCH_SELECTION=smoke BENCH_CASES= BENCH_PURPOSE=on-demand BENCH_SET=run.repeats=1

bench-report: ## Summarise BENCH_OUT.
	@$(call _bench_do,report,$(_bench_env) $(BENCH) report '$(BENCH_OUT)')
