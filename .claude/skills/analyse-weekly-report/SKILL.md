---
name: analyse-weekly-report
description: >
  Run the weekly-scan week-over-week report (weekly_report.py) for the generative
  or embedding table, then append a per-hf_adapter analysis explaining why each
  adapter whose pass rate DROPPED largely between the two snapshots regressed —
  grounded in the failure_category shift and error clusters — reusing the exact
  rows the report already downloaded. Improvements are not analysed; only
  regressions. The analysis is appended as a "7. PER-ADAPTER ANALYSIS"
  section in the script's own banner format. The final answer is the script's full
  report followed by that section, and can be saved to a file with
  --output-file. Use when the user asks to run/analyse the weekly report, explain
  adapter rate changes, or say why an adapter started failing this week, for either
  the generative or embedding models.
argument-hint: "[--mode generative|embedding] [--prev YYYY-MM-DD --curr YYYY-MM-DD] [--min-delta PP] [--min-n N] [--output-file PATH]"
allowed-tools: Bash(python3:*), Read, Write
---

# Weekly Spyre scan — report + per-adapter analysis

Two things in one run, over the **generative** table (default) or the
**embedding** table (`--mode`):

1. **The report** — the six-section week-over-week comparison that
   `tests/spyre/weekly_generation/weekly_report.py` produces (coverage, adapter
   coverage, verified_on_spyre delta, failure_category distribution, error
   patterns, adapter breakdown).
2. **The analysis** — a focused, per-`hf_adapter` follow-up: which adapters
   **regressed** a lot week over week (pass rate dropped), and the one-sentence
   reason for each. Improvements are deliberately **not** analysed — we only care
   what broke.

The analysis **reuses the exact rows the report downloaded** — both are built from
a single fetch of each snapshot, so they can never describe different data or
different dates. It is rendered as a seventh section, **`7. PER-ADAPTER
ANALYSIS`**, in the script's own banner format (a `═`×72 rule, the title, a `─`×72
rule) so the report and the analysis read as one continuous document. The final
answer presents the report as a plain fenced text block, followed by that `7.`
section, and — when `--output-file` is given — writes the whole thing
(report + section 7, plain text) to that path.

The output is text only. Do **not** write a standalone Python script or add files
to the repo — call the report's own functions from short inline `python3 -c`
snippets, exactly as the steps below do. The two tables have the identical column
shape (`adapter_name`, `verified_on_spyre`, `failure_category`, `error`, …), so
the whole procedure is one table constant — `GENERATIVE_TABLE_NAME` /
`EMBEDDING_TABLE_NAME` — chosen once from `--mode` and used in every snippet.

## Usage — print the parameter hints instead of running

Before doing anything else, look at `$ARGUMENTS`. If — and only if — the user is
asking *how to use the skill* rather than to run it, print the usage block below
and **stop**: do not touch ClickHouse, do not run any `python3 -c` snippet, do not
write any file. Treat it as a help request when `$ARGUMENTS`, after trimming
whitespace, is any of:

- empty (no arguments at all), or
- exactly `help`, `-h`, `--help`, `?`, or `usage` (case-insensitive), or
- prose that plainly asks about the parameters rather than for a run — e.g.
  "what are the options", "what flags does this take", "hints on the parameters",
  "how do I use this".

Anything that carries a real run instruction — a `--mode`, a date, a threshold, an
output path, or prose like "run the embedding report" — is **not** a help request:
run the scan normally (Step 0 onward). When in doubt between "asking about a
parameter" and "asking for a run", prefer printing the help; a wasted ClickHouse
fetch is worse than an extra hint.

Print the hints as a single ```text``` fenced block, in this shape (this is the
canonical source for what each flag means — keep it in sync with `## Parameters`
below if either changes):

```text
/analyse-weekly-report — weekly Spyre scan report + per-adapter regression analysis

USAGE
  /analyse-weekly-report [--mode generative|embedding]
                         [--prev YYYY-MM-DD --curr YYYY-MM-DD]
                         [--min-delta PP] [--min-n N]
                         [--output-file PATH]
  /analyse-weekly-report help        show this help and exit (no query runs)

PARAMETERS
  --mode generative|embedding  Which table to scan. Default: generative.
  --prev / --curr YYYY-MM-DD   The two snapshots to compare (older=prev,
                               newer=curr). Given together or not at all;
                               omitted → the two most recent snapshots of the
                               chosen table.
  --min-delta PP               Regression threshold in percentage points, for
                               the analysis section only. Default: 10.
  --min-n N                    Minimum denominator in BOTH snapshots for an
                               adapter to be eligible for analysis (suppresses
                               tiny-sample noise). Default: 10.
  --output-file PATH           Also write the full report + section 7 as plain
                               text to PATH (overwrites if it exists). Default:
                               print only.

NOTES
  • Read-only: queries ClickHouse once; never writes the DB, never commits.
  • Only regressions (pass rate DROPPED) are analysed; improvements are not.
  • Dates/thresholds/paths named in prose are mapped onto these flags.
```

## What "the rate" means

For one adapter, over the rows of one snapshot:

- **Denominator**: rows for that `adapter_name`, **excluding infrastructure
  failures** (`hardware_exception` / `worker_crashed` / `worker_timeout` — use
  `_is_infra_failure(row)`). Their verdict is about the infra, not the model, so
  they must not count for or against the adapter.
- **Numerator**: those rows with `verified_on_spyre = True`.
- **Rate** = numerator / denominator, as a percentage. A change is measured in
  **percentage points (pp)**, `curr_rate − prev_rate`.

Rows with no adapter bucket under `(no adapter)` — report it if it moves, but it
is not an adapter, so never write a per-adapter explanation sentence for it.

## Parameters

If `$ARGUMENTS` is a help request rather than a run instruction, do **not** read
this section as run flags — print the usage block from *Usage — print the
parameter hints instead of running* above and stop. Otherwise, `$ARGUMENTS` may
carry:

- `--mode generative|embedding` — which table to analyse. Default **generative**.
  Maps to the table constant: `generative` → `GENERATIVE_TABLE_NAME`,
  `embedding` → `EMBEDDING_TABLE_NAME`. If the user names the model kind in prose
  ("the embedding report", "for embeddings"), map it here.
- `--prev YYYY-MM-DD --curr YYYY-MM-DD` — the two snapshots to compare. If
  omitted, use the two most recent snapshots **of the chosen table** (older =
  prev, newer = curr). `--prev` and `--curr` go together or not at all. The two
  tables are scanned on the same cadence but do not have to share dates.
- `--min-delta PP` — the "largely changed" threshold in pp, for the analysis
  section only. Default **10**.
- `--min-n N` — minimum denominator, in **both** snapshots, for an adapter to be
  eligible for the analysis's "largely changed" list. Default **10**. This
  suppresses tiny-sample noise (17/26 → 16/21 swings double digits on a handful of
  models and means nothing).
- `--output-file PATH` — also write the combined plain-text output (the six-section
  report **plus** `7. PER-ADAPTER ANALYSIS`) to `PATH`. Default: unset (print
  only). See Step 5 for exactly what gets written. The chat reply is unchanged
  whether or not this is given.

If the user names dates, a threshold, a sample floor, or an output path in prose,
map them onto these; otherwise use the defaults and say so in the output.

## Step 0 — Run the report and cache the downloaded rows (one fetch)

This single snippet does points 1 and 2 together: it resolves the two snapshot
dates, fetches each snapshot **once**, builds the script's full six-section report
from *those* rows (the report is generated by `weekly_report.py`'s own section
functions — no reimplementation), and **pickles the rows to `/tmp`** so every
later step reads the same downloaded data instead of re-querying ClickHouse.

The module is used as functions, not run as a script (running the file directly
hits a known `ModuleNotFoundError: No module named 'tests.spyre'` packaging quirk;
importing its functions is fine). Set `MODE`; if the user gave `--prev`/`--curr`,
replace the `_two_latest_dates(...)` line with those two dates instead (and if
either is missing from the table, stop and say which). From the repo root:

```bash
python3 -c "
import pickle
from datetime import date
from tests.spyre.weekly_generation import weekly_report as wr
from tests.spyre.weekly_generation.clickhouse_db import get_client
from tests.spyre.weekly_generation.table_schema import GENERATIVE_TABLE_NAME, EMBEDDING_TABLE_NAME

MODE = 'generative'  # or 'embedding'
TABLE = GENERATIVE_TABLE_NAME if MODE == 'generative' else EMBEDDING_TABLE_NAME
c = get_client()
prev_d, curr_d = wr._two_latest_dates(c, TABLE)   # or: date.fromisoformat('PREV'), date.fromisoformat('CURR')

# ONE download of each snapshot, shared by the report and the analysis.
prev_rows = wr._fetch_snapshot(c, TABLE, prev_d)
curr_rows = wr._fetch_snapshot(c, TABLE, curr_d)
with open('/tmp/weekly_rows.pkl', 'wb') as fh:
    pickle.dump({'mode': MODE, 'prev_d': prev_d, 'curr_d': curr_d,
                 'prev_rows': prev_rows, 'curr_rows': curr_rows}, fh)

# The script's own six-section report, built from the cached rows.
mt = wr.ModelType(MODE)
report = '\n'.join([
    wr._hr('='),
    f'WEEKLY SPYRE SCAN — {mt.value.upper()} REPORT',
    f'  Previous snapshot : {prev_d}',
    f'  Current snapshot  : {curr_d}',
    wr._section_coverage(prev_rows, curr_rows),
    wr._section_adapter_coverage(prev_rows, curr_rows),
    wr._section_verified_on_spyre(prev_rows, curr_rows),
    wr._section_failure_categories(prev_rows, curr_rows),
    wr._section_error_patterns(prev_rows, curr_rows),
    wr._section_family_breakdown(prev_rows, curr_rows),
    wr._hr('='),
])
print(report)
"
```

Keep this snippet's full stdout verbatim — it is the report text you will collapse
in Step 5. `/tmp/weekly_rows.pkl` now holds the downloaded rows and the resolved
dates; Steps 2 and 4 load it rather than touching ClickHouse again.

> This composition is exactly what `weekly_report.main()` prints (same header,
> same six sections, same order); it is split out only so the fetch can be shared.
> If the two ever diverge, `weekly_report.py`'s section list changed — re-sync the
> `_section_*` calls above with its `main()`.

## Step 2 — Per-adapter pass-rate table (from the cached rows)

No new query — load the pickle from Step 0:

```bash
python3 -c "
import pickle
from tests.spyre.weekly_generation.weekly_report import _is_infra_failure

with open('/tmp/weekly_rows.pkl', 'rb') as fh:
    d = pickle.load(fh)
prev, curr = d['prev_rows'], d['curr_rows']

def rates(rows: list[dict]) -> dict[str, tuple[int, int]]:
    out: dict[str, list[int]] = {}
    for r in rows:
        if _is_infra_failure(r):
            continue
        a: str = r.get('adapter_name') or '(no adapter)'
        t = out.setdefault(a, [0, 0])
        t[1] += 1
        if r['verified_on_spyre']:
            t[0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}

pr, cr = rates(prev), rates(curr)
print(f'{\"adapter\":<28} {\"prev\":>13} {\"curr\":>13} {\"dpp\":>7}')
rows = []
for a in sorted(set(pr) | set(cr)):
    pp, pt = pr.get(a, (0, 0)); cp, ct = cr.get(a, (0, 0))
    ppct = pp / pt * 100 if pt else None
    cpct = cp / ct * 100 if ct else None
    dpp = (cpct - ppct) if (ppct is not None and cpct is not None) else None
    rows.append((a, pp, pt, cp, ct, dpp))
for a, pp, pt, cp, ct, dpp in sorted(rows, key=lambda x: (abs(x[5]) if x[5] is not None else -1), reverse=True):
    ps = f'{pp}/{pt}={pp/pt*100:.0f}%' if pt else 'absent'
    cs = f'{cp}/{ct}={cp/ct*100:.0f}%' if ct else 'absent'
    ds = f'{dpp:+.0f}pp' if dpp is not None else 'new/gone'
    print(f'{a:<28} {ps:>13} {cs:>13} {ds:>7}')
"
```

## Step 3 — Select the adapters that regressed "largely"

We analyse **regressions only** — adapters whose pass rate **dropped**. From the
Step 2 table, an adapter is a **largely-regressed** adapter when **both** hold:

- `dpp <= -(--min-delta)` (default: a drop of 10pp or more — note the sign, this
  is a one-sided test, not `abs(dpp)`), and
- `prev_n >= --min-n` **and** `curr_n >= --min-n` (default 10).

Adapters that *rose* by ≥ `--min-delta` are **not** selected and are **not**
analysed — ignore them entirely (no per-adapter sentence, no "improvements" list).

Handle the edge buckets explicitly, do not force them through the pp test:

- **New adapter** (absent in prev, present in curr): there is no prior rate to
  compare, so it cannot be a regression — leave it to the report's `2. ADAPTER
  COVERAGE`, do not analyse it.
- **Gone adapter** (present in prev, absent in curr): likewise the report's
  coverage section covers it; not a per-adapter regression to explain.
- **Dropped but below `--min-n`**: an adapter that fell ≥ `--min-delta` but has
  fewer than `--min-n` rows in either snapshot is too small to trust — mention it
  in one short "small-sample regressions (not analysed)" line so it is not silently
  dropped, but do not write it a full explanation.

## Step 4 — Explain each regressed adapter (from the cached rows)

For each regressed adapter in the Step 3 list, find the driver by diffing what
changed between prev and curr among its **non-infra failing** rows. Loads the same
pickle; set `ADAPTERS` to the Step 3 list (regressions only):

```bash
python3 -c "
import pickle
from collections import Counter
from tests.spyre.weekly_generation.weekly_report import _is_infra_failure, _signature_for, ClusterMethod

with open('/tmp/weekly_rows.pkl', 'rb') as fh:
    d = pickle.load(fh)
prev, curr = d['prev_rows'], d['curr_rows']
ADAPTERS = ['ADAPTER1', 'ADAPTER2']  # from Step 3

for ADAPTER in ADAPTERS:
    def fails(rows: list[dict]) -> list[dict]:
        return [r for r in rows
                if (r.get('adapter_name') or '(no adapter)') == ADAPTER
                and not r['verified_on_spyre'] and not _is_infra_failure(r)]
    pf, cf = fails(prev), fails(curr)
    pc = Counter((r.get('failure_category') or 'none') for r in pf)
    cc = Counter((r.get('failure_category') or 'none') for r in cf)
    print(f'=== {ADAPTER}: non-infra failures {len(pf)} -> {len(cf)} ===')
    for k in sorted(set(pc) | set(cc), key=lambda k: cc.get(k, 0) - pc.get(k, 0)):
        print(f'    {k:<28} {pc.get(k,0):>4} -> {cc.get(k,0):>4}  ({cc.get(k,0)-pc.get(k,0):+d})')
    sig = Counter(_signature_for(r.get('error') or '', ClusterMethod.NORMALIZED) for r in cf if r.get('error'))
    print('  current top error clusters (normalized):')
    for s, n in sig.most_common(5):
        print(f'    [{n:>4}x] {s[:110]}')
    print()
"
```

Read the two diffs together:

- The **failure_category** with the largest **positive** delta is almost always
  the driver of a regression — a new wave of failures in one category. Watch for
  the subtle case where the rate dropped for a *denominator* reason (the model set
  grew, adding failing checkpoints) rather than an existing model breaking — say
  so.
- The **error clusters** name the concrete fault behind that category. Use the
  normalized method (it collapses model-specific tokens so the recurring fault is
  legible); mention the literal detail — a version, a package name — only when it
  is the point.
- A category that is a pre-filter verdict (`quantized_model`, `moe`,
  `not-implemented-adapter`, `model_too_large`) moving usually means the *set of
  models* shifted, not that the adapter's behaviour changed — say so rather than
  implying a code regression.

Then write **one sentence** per regressed adapter, of the shape:

> `hf_gemma3` fell −33pp (33% → 0%) because `test_execution_exception` exploded
> 3 → 338 — a new `InductorError: Unsupported modular coordinate expression`
> compile fault; every Gemma-3 checkpoint that used to pass now fails to compile.

Order the regressions worst-first (largest drop at the top). If a regression has
no clear single driver (the deltas are spread across categories), say that
honestly rather than forcing a cause.

## Step 5 — Format section 7 and present the result

First build the analysis as **section 7**, in the script's exact banner format —
a `═`×72 rule, the title `7. PER-ADAPTER ANALYSIS`, a `─`×72 rule — so it reads as
a continuation of the six-section report. Both rules are literally
`_hr('═')` / `_hr()` from `weekly_report.py` (72 chars each); copy the widths from
the `═`/`─` rules already in the Step 0 output rather than counting by hand.

The body of section 7 is a tight briefing, indented two spaces like the script's
sections, with these parts and **no Coverage part** (the report's `1. COVERAGE`
and `2. ADAPTER COVERAGE` already give new/gone adapters and counts — do not repeat
them here):

- One line naming the thresholds in force (`--min-delta`, `--min-n`) and whether
  they were defaults or user-supplied.
- **Regressions** (adapters that dropped ≥ `--min-delta` with n ≥ `--min-n` in
  both snapshots), worst-first: one sentence each, per Step 4. If none, write
  `Regressions (≥Npp): none`.
- Optionally one **small-sample regressions (not analysed)** line naming any
  adapter that dropped ≥ `--min-delta` but fell below `--min-n`. Skip the line if
  there are none.
- Optionally one closing line if there is a single cross-cutting takeaway (e.g. a
  shared root-cause fault). Skip it if there is nothing to add.

Do **not** include improvements — this section is regressions only.

Section 7 looks like this (fill in real content; keep the two rules exactly as
long as the report's):

```text
════════════════════════════════════════════════════════════════════════
7. PER-ADAPTER ANALYSIS
────────────────────────────────────────────────────────────────────────
  Thresholds: min-delta 10pp, min-n 10 (defaults)
  Regressions (≥10pp):
    • hf_gemma3 −33pp (33% → 0%) — test_execution_exception exploded 3 → 338
      (InductorError: Unsupported modular coordinate expression); a compile-time
      break, not a model-set change.
    • hf_llama −11pp (35% → 24%) — same InductorError fault; +5 test_execution_exception.
  Note: both regressions share one Spyre backend lowering fault.
```

When nothing regressed, section 7 is simply:

```text
════════════════════════════════════════════════════════════════════════
7. PER-ADAPTER ANALYSIS
────────────────────────────────────────────────────────────────────────
  Thresholds: min-delta 10pp, min-n 10 (defaults)
  Regressions (≥10pp): none
```

Then present the result to the user as **two plain ```text``` fenced blocks**, in
this order — no `<details>`/`<summary>` wrapper:

1. The **verbatim** Step 0 report stdout, in its own ```text``` block. (Do not
   wrap it in `<details>`: that HTML does not render as a collapse in the terminal,
   and its closing `</details>` tag collides with the report's trailing `=` rule
   and leaks out as literal text. A plain fenced block is what renders cleanly.)
2. Section 7, in its own ```text``` fenced block, so its banner and alignment
   render monospaced.

Put a blank line between the two blocks. Keep section 7 tight — it is a briefing,
not a table dump. The report block is the supporting evidence; do not also paste
the Step 2 table unless the user asks.

### If `--output-file PATH` was given

Also write the **combined plain-text document** to `PATH` with the Write tool: the
verbatim Step 0 report followed by a single blank line and then section 7 — the
same text that is inside the two ```text``` blocks above, but with **no Markdown
fences** (it is a `.txt`-style artifact, not a chat message). In other words,
`PATH` should contain exactly what `weekly_report.py` would print if it had a
seventh section. After writing, tell the user the path and byte/line count on one
line. Writing the file does **not** change the chat reply above — the user always
sees the report block + section 7.

## Notes and guardrails

- **Read-only against the data.** This skill only *queries* ClickHouse (once, in
  Step 0); it never writes to the database and never commits. On disk it writes
  only the throwaway `/tmp/weekly_rows.pkl` cache and, when `--output-file PATH` is
  given, the report+section-7 text at `PATH`. Write `PATH` where the user asked; if
  it already exists, it is overwritten — mention that in the confirmation line.
- **One fetch.** Step 0 is the only ClickHouse round-trip; Steps 2 and 4 read the
  pickle. This is what "uses the downloaded data the script made" means here — the
  analysis is provably on the same rows the report was built from. If
  `/tmp/weekly_rows.pkl` is missing when Step 2/4 runs (e.g. a fresh shell), re-run
  Step 0 rather than adding a new query.
- **Mode.** `--mode generative` (default) or `--mode embedding` selects the table
  via the `MODE`/`TABLE` line in Step 0; the cached pickle carries the mode
  forward, so Steps 2 and 4 need no table constant. Everything else is identical
  between the two tables. The embedding side is dominated by a few large adapters
  (`hf_bert`, `hf_xlm_roberta`, `hf_mpnet`, `hf_modernbert`); its many single-model
  adapters usually fall below `--min-n` and land in the small-sample note — that is
  expected, not a bug.
- Credentials come from `.env` via `clickhouse_db.get_client()`, like the rest of
  the weekly pipeline. If `get_client()` raises a missing-env / connection error,
  stop and report it plainly — do not fabricate numbers.
- Never invent a cause. Every explanation sentence must trace to a
  failure_category delta or an error cluster you actually saw in Step 4's output.
