# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the multicard smoke test and print a human-readable summary.

For real tensor-parallel multi-card runs, launch with torchrun (one process
per card).  For single-card, plain python works.

Requirements:
  - transformers==5.15.x  (uv pip install "transformers==5.15.0")

SPYRE_DEVICES tells the Flex runtime which card indices to use per rank.
Index-to-physical-card mapping is handled internally by Flex and is not
independently verifiable from this script.

2-card example (valid indices are node-specific — check yours first)::

    export SPYRE_DEVICES=0,1
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=2 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

4-card example::

    export SPYRE_DEVICES=0,1,2,3
    export PYTHONPATH=/path/to/hf-adapters
    torchrun --nproc-per-node=4 --master-port=29500 \\
        scripts/run_multicard_smoke.py --dtype float16

Single-card::

    python scripts/run_multicard_smoke.py

The --model argument accepts any HuggingFace repo ID or local path
(default: ibm-granite/granite-3.3-8b-instruct).

The script exits with code 0 on PASS and code 1 on FAIL or ERROR. A FAIL is
reported by that exit code alone, not by an exception, so under torchrun the
only thing printed after the summary is the elastic agent's teardown:

    Sending process <pid> closing signal SIGTERM
    failed (exitcode: 1) local_rank: 0 ...
    torch.distributed.elastic.multiprocessing.errors.ChildFailedError

That SIGTERM goes to the ranks that were still running; torchrun stops the
whole gang once any rank exits nonzero. It is the result of the failure, never
the cause -- read the RESULTS SUMMARY above it for the real verdict.

Note: a ``corrupted double-linked list`` / SIGABRT crash may appear after the
RESULTS SUMMARY prints.  This is a known shutdown bug in the Spyre runtime
(libsenlib-dd2.so destructors) and does not affect correctness — if the
summary shows ``Status: PASS`` the test passed.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

# Ensure the project root (parent of scripts/) is on sys.path so that
# tests.spyre.test_multicard_spyre can be imported when running directly
# from anywhere inside the repo.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.spyre.test_multicard_spyre import run_multicard_smoke_test  # noqa: E402

DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MODEL = "ibm-granite/granite-3.3-8b-instruct"
DEFAULT_MAX_NEW_TOKENS = 8


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model path or local dir (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Token generation budget (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help=(
            "Torch dtype to pass to AutoSpyreModelForCausalLM.from_pretrained "
            "(e.g. float16, bfloat16, float32).  Omit to let the model decide."
        ),
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        metavar="N",
        help="Number of identical prompts to batch together (default: 1).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help=f"Read prompt from the file (default: {DEFAULT_PROMPT})",
    )
    return parser


def _fmt(value: float | None, unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}{unit}"


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dtype is not None:
        if args.dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unknown --dtype {args.dtype!r}. "
                f"Choose from: {', '.join(_DTYPE_MAP)}"
            )
    dtype = _DTYPE_MAP.get(args.dtype) if args.dtype is not None else None

    aiu_ids = os.environ.get("AIU_IDS")
    spyre_devices = os.environ.get("SPYRE_DEVICES")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_rank0 = local_rank == 0

    # Trim AIU_IDS to WORLD_SIZE entries so a 4-address env var doesn't get
    # printed when only 1 or 2 ranks are actually launched.
    if aiu_ids is not None:
        _ids = [c.strip() for c in aiu_ids.split(",") if c.strip()]
        if len(_ids) > world_size:
            aiu_ids = ",".join(_ids[:world_size])

    if is_rank0:
        print()
        print("=" * 70)
        print("  run_multicard_smoke")
        print("=" * 70)
        print(f"  SPYRE_DEVICES    : {spyre_devices!r}")
        print(f"  AIU_IDS          : {aiu_ids!r}  (fallback; SPYRE_DEVICES is primary)")
        print(f"  Model            : {args.model}")
        print(f"  max_new_tokens   : {args.max_new_tokens}")
        print(f"  batch            : {args.batch}")
        print(f"  dtype            : {args.dtype or '(not set — model default)'}")
        print(f"  Prompt File      : {args.prompt_file}")
        print("=" * 70)

    if args.prompt_file is not None:
        try:
            with open(args.prompt_file, "r") as _f:
                prompt = _f.read()
        except OSError as e:
            raise RuntimeError(
                f"Cannot read prompt file {args.prompt_file!r}: {e}"
            ) from e
    else:
        prompt = DEFAULT_PROMPT

    result = run_multicard_smoke_test(
        args.model,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        batch_size=args.batch,
        prompt=prompt,
    )

    # ── Summary table ──────────────────────────────────────────────────────
    # Each rank builds its summary as one string and writes it atomically.
    # A brief sleep stagger keeps blocks from interleaving in the terminal.
    import time

    def _rank_summary(r: dict) -> str:
        lines = [
            "",
            "=" * 70,
            f"  RESULTS SUMMARY  [rank {r['local_rank']}/{r['world_size']}]",
            "=" * 70,
            f"  SPYRE_DEVICES : {r['spyre_devices_env']!r}",
        ]
        if r["resolved_cards"]:
            lines.append(
                f"  PCI hint      : {', '.join(r['resolved_cards'])}"
                f"  (AIU_WORLD_RANK_* guess, unconfirmed)"
            )
        lines += [
            f"  AIU_IDS       : {r['aiu_ids_env']!r}",
            f"  Model         : {r['model']}",
            f"  Batch      : {r['batch_size']}",
            f"  Status     : {r['status']}",
            f"  Load       : {_fmt(r['load_s'], 's')}",
            f"  Generate   : {_fmt(r['gen_s'], 's')}",
            f"  TTFT       : {_fmt(r['ttft_ms'], ' ms')}",
            f"  Decode avg : {_fmt(r['decode_ms'], ' ms')}",
            f"  Steady ITL : {_fmt(r['steady_itl_ms'], ' ms')}  (outliers excluded)",
            f"  Output     : {r['output']!r}",
        ]
        if r["error"]:
            lines += ["", "  ERROR DETAIL:"]
            lines += [f"    {line}" for line in r["error"].splitlines()]
        lines += ["=" * 70, ""]
        return "\n".join(lines)

    time.sleep(local_rank * 0.3)
    sys.stdout.write(_rank_summary(result))
    sys.stdout.flush()

    failed = result["status"] != "PASS"

    # A FAIL is reported by exiting nonzero, not by raising, so torchrun has no
    # traceback to show. Say so explicitly: otherwise the only thing in the log
    # is the agent's SIGTERM/ChildFailedError teardown, which reads like an
    # abort or a timeout rather than a verdict this script already reached.
    if failed and local_rank == 0:
        sys.stdout.write(
            "\n"
            f"multicard smoke FAILED (status={result['status']!r}); "
            "exiting rank 0 with code 1.\n"
            "The torchrun 'SIGTERM' / ChildFailedError below is the elastic "
            "agent tearing down the remaining ranks after this exit. It is the "
            "consequence of this failure, not its cause; the real reason is in "
            "the RESULTS SUMMARY above.\n"
        )
        sys.stdout.flush()

    # Ranks reach their verdict a few ms apart. Whichever exits first leaves its
    # siblings alive, and torchrun SIGTERMs them mid-print -- truncating their
    # summaries and making one rank's failure look like an unrelated signal
    # kill. Sync first so every rank prints fully and they exit together.
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:  # pragma: no cover - teardown best effort
            sys.stdout.write(f"\n[rank {local_rank}] exit barrier skipped: {e}\n")
            sys.stdout.flush()

    # os._exit (not sys.exit) is deliberate: it skips interpreter teardown and
    # so avoids the libsenlib-dd2.so destructor SIGABRT described in the module
    # docstring. It also skips atexit/buffer flushing, hence the flushes above.
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
