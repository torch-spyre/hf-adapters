"""The child-process half of the weekly scan: evaluate models, report rows.

Everything here runs in a ``multiprocessing`` "spawn" child started by
``weekly_test.main``, one child per batch. Two consequences shape the module:

* **Torch stays out of module scope.** The heavy imports (``hf_adapters``, the
  Spyre test entry points, ``tests.conftest``) happen inside the functions that
  need them, so the parent — which imports this module only to name
  ``_process_batch`` as the process target — never pays for them.
* **A failure is a row, not an exception.** One bad model must not cost the
  other N-1 in its batch, so every model is evaluated inside its own
  ``try``/``except`` and errors are recorded in the returned dict's
  ``failure_category``/``error`` fields. The one exception is
  ``hardware_exception``, which ends the batch early because the accelerator
  itself is gone.

The child never touches the sink. It returns plain dicts over the queue and the
parent does all the writing, which keeps database credentials and connection
state in one process.
"""

import os
import sys
import traceback as _traceback
from datetime import date
from multiprocessing.queues import SimpleQueue

from huggingface_hub.errors import HfHubHTTPError

from tests.spyre.weekly_generation.failure_categories import (
    FAILURE_CATEGORY_CPU_GENERATE_FAILED,
    FAILURE_CATEGORY_CPU_LOAD_FAILED,
    FAILURE_CATEGORY_HARDWARE_EXCEPTION,
    FAILURE_CATEGORY_MISFORMED_HF_FAILED,
    FAILURE_CATEGORY_MODEL_TOO_LARGE,
    FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
    FAILURE_CATEGORY_QUANTIZED_MODEL,
    FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION,
    FAILURE_CATEGORY_VERIFICATION_FAILED,
)
from tests.spyre.weekly_generation.model_type import ModelType
from utils.utilities import ts


def _process_batch(
    batch: list[dict],
    adapter_dates: dict[str, str | None],
    result_queue: SimpleQueue,
    model_type: ModelType,
    snapshot_date: date,
) -> None:
    """Worker target: evaluate up to one batch of models in a single spawned child.

    Amortizes the per-child fixed cost (spawn + module imports + kernel
    teardown on exit) across the batch; ``weekly_test``'s
    ``{GENERATIVE,EMBEDDING}_NUMBER_OF_MODEL_PER_PROCESS`` set how many. Puts a
    ``list[dict]`` on the queue — one full result dict per row, in the same order
    as *batch*. If a single model errors, its ``error`` field is populated and
    the loop continues to the next model; the child does NOT abort.

    The queue is the ``multiprocessing.SimpleQueue`` the parent created via its
    spawn context, not an ``asyncio`` one — nothing here is coroutine-based.

    Exits via ``os._exit(0)`` rather than returning; see the comment at the end
    for why skipping interpreter shutdown is what actually frees the card.

    Each returned dict has the same shape ``main`` expects for a rec plus an
    ``error`` field (str or None):

        {
            "model_name":       ...,
            "config_class":     ...,
            "adapter_name":     ...,
            "added_date":       ...,   # ISO 8601 str or None
            "snapshot_date":    ...,   # date object
            "verified_on_cpu":  bool,
            "verified_on_gpu":  False,
            "verified_on_spyre": bool,
            "curated":          bool,
            "num_downloads":    int,
            "family":           str,
            "architecture":     str,
            "parameters_number": int,
            "error":            None or str,
            "failure_category": None or str,
        }
    """
    import time as _t

    _child_entered: float = _t.monotonic()
    print(
        f"{ts()}       child[{os.getpid()}] entered _process_batch with {len(batch)} model(s)",
        flush=True,
    )

    from tests.conftest import resolve_adapter_module_for_test

    results: list[dict] = []
    for row in batch:
        model_path: str = str(row["model_id"])
        rec: dict = {
            "model_name": model_path,
            "config_class": row.get("config_class"),
            "adapter_name": "",
            "added_date": None,
            "snapshot_date": snapshot_date,
            "verified_on_cpu": False,
            "verified_on_gpu": False,
            "verified_on_spyre": False,
            "curated": bool(row["curated"]),
            "num_downloads": int(row.get("downloads") or 0),
            "family": str(row.get("model_type") or ""),
            "architecture": str(row.get("architectures") or ""),
            "parameters_number": int(row.get("parameters") or 0),
            "error": None,
            "failure_category": None,
        }
        print(f"{ts()} - {model_path} " + ("\t(curated)" if row["curated"] else ""))
        try:
            try:
                adapter_module = resolve_adapter_module_for_test(model_path)
            except Exception:
                rec["failure_category"] = FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER
                raise
            adapter_name: str = os.path.splitext(
                os.path.basename(adapter_module.__file__)
            )[0]
            rec["adapter_name"] = adapter_name
            rec["added_date"] = adapter_dates.get(adapter_name)

            metrics = eval_model(model_path, adapter_module, model_type)
            rec["verified_on_cpu"] = bool(metrics.get("load", False))
            rec["verified_on_spyre"] = bool(metrics.get("correct", False))
            rec["error"] = metrics.get("error") or None
            rec["failure_category"] = metrics.get("failure_category") or None
            if not rec["verified_on_cpu"] and rec["failure_category"] is None:
                rec["failure_category"] = _classify_failure(
                    rec["error"] or "", FAILURE_CATEGORY_CPU_LOAD_FAILED
                )
        except Exception as e:
            # Skip the error/traceback for shallow failure categories where the
            # failure_category itself is fully self-describing.
            if rec["failure_category"] not in (
                FAILURE_CATEGORY_NOT_IMPLEMENTED_ADAPTER,
                FAILURE_CATEGORY_MODEL_TOO_LARGE,
            ):
                rec["error"] = (
                    f"{type(e).__name__}: {e}\n"
                    f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
                )
            if rec["failure_category"] is None:
                rec["failure_category"] = FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION
        results.append(rec)
        print(
            f"{ts()}       child[{os.getpid()}] finished model "
            f"{len(results)}/{len(batch)}: {model_path!r}  "
            f"(verified_on_cpu={rec['verified_on_cpu']}, "
            f"verified_on_spyre={rec['verified_on_spyre']}, "
            f"failure_category={rec['failure_category']}, "
            f"error={rec['error']})",
            flush=True,
        )
        # Bail out of the batch immediately on a hardware exception — the
        # Spyre device is unreachable, so every remaining model in this
        # batch would hit the same wall. The parent picks up the signal
        # from the returned results and aborts the outer loop.
        if rec["failure_category"] == FAILURE_CATEGORY_HARDWARE_EXCEPTION:
            print(
                f"{ts()}       child[{os.getpid()}] aborting batch — "
                f"hardware_exception detected; "
                f"{len(batch) - len(results)} model(s) not attempted",
                flush=True,
            )
            break

    result_queue.put(results)
    print(
        f"{ts()}       child[{os.getpid()}] done in "
        f"{_t.monotonic() - _child_entered:.2f}s ({len(results)} results)",
        flush=True,
    )

    # Skip Python's graceful shutdown: no atexit handlers, no thread
    # finalization, no torch/torch_spyre destructors walking the tensor graph
    # that the kernel is about to reclaim in bulk anyway. Closing the Spyre
    # device FD on _exit(2) triggers the driver's own release path (VFIO
    # unmap-all + IOMMU teardown), which is what actually returns the
    # accelerator memory. Prior measurements: leaving Python's graceful
    # shutdown in place cost ~30 s per child; running gc.collect() here on
    # top of that added another ~20 s.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _classify_failure(err: str, default: str) -> str:
    """Bucket a raw error/traceback string into a failure_category.

    Signals in order of specificity:

    * ``"Failed to open the IBM Spyre VFIO device"`` — the accelerator itself
      is unreachable (driver, permissions, another process holding it, …);
      the model under test is not to blame, so tag as hardware_exception.
    * ``"quantiz"`` / ``"optimum"`` — bitsandbytes / AWQ / GPTQ error text
      almost always contains ``quantiz``, and ``optimum`` catches the
      optimum-quanto / optimum-neuron loaders.

    Anything unrecognised falls through to *default* (usually the surrounding
    context's fallback: cpu_load_failed at load time, test_execution_exception
    at eval time).
    """
    if not err:
        return default
    if "Failed to open the IBM Spyre VFIO device" in err or "Replace card" in err:
        return FAILURE_CATEGORY_HARDWARE_EXCEPTION
    if "does not appear to have files named ('model" in err:
        return FAILURE_CATEGORY_MISFORMED_HF_FAILED
    lowered: str = err.lower()
    if "quantiz" in lowered or "optimum" in lowered:
        return FAILURE_CATEGORY_QUANTIZED_MODEL
    return default


def eval_model(model_id: str, adapter, model_type: ModelType) -> dict:
    """Load *model_id* on CPU then run the mode's verification pipeline.

    Generative mode: CPU-load → CPU-generate (single-prompt HF forward pass)
    → Spyre smoke + token-compare. The intermediate CPU-generate step catches
    lazy shape errors, tokenizer/config mismatches, and custom-code bugs that
    don't surface at ``from_pretrained`` time; on failure the row is tagged
    ``cpu_generate_failed`` and the Spyre steps are skipped.

    Embedding mode: CPU-load → Spyre cosine-compare (no generate step —
    embedders don't have a ``.generate()`` method).

    Returns a metrics dict with keys ``load``, ``correct``, ``error``,
    ``failure_category``. ``correct`` is ``smoke_passed and not mismatches``
    — in embedding mode there is no smoke step, so ``smoke_passed`` is
    treated as True and the outcome reduces to ``not mismatches``.
    """
    load_on_cpu = False
    smoke_passed = model_type == ModelType.EMBEDDING
    mismatches = True
    result: dict = {"error": "", "failure_category": None}

    try:
        if adapter is not None:
            load_on_cpu, load_error = _load_on_cpu(
                model_path=model_id, model_type=model_type
            )
            if load_error and not result["error"]:
                result["error"] = load_error
            if load_on_cpu:
                if model_type == ModelType.GENERATIVE:
                    # Extra CPU-generate step — a load that succeeds but crashes
                    # here means the checkpoint is malformed in a way that only
                    # surfaces during forward. Stop before we waste Spyre time.
                    generate_ok, generate_error = _cpu_generate(model_path=model_id)
                    if not generate_ok:
                        if generate_error and not result["error"]:
                            result["error"] = generate_error
                        result["failure_category"] = _classify_failure(
                            generate_error or "",
                            FAILURE_CATEGORY_CPU_GENERATE_FAILED,
                        )
                    else:
                        from tests.spyre.test_e2e_smoke_spyre import run_smoke_test
                        from tests.spyre.test_e2e_token_compare_spyre import (
                            token_compare_spyre,
                        )

                        smoke_passed = (
                            run_smoke_test(model_path=model_id)["status"] == "PASS"
                        )
                        mismatches, _ = token_compare_spyre(model_id)
                else:
                    from tests.spyre.test_e2e_embed_compare_spyre import (
                        embed_compare_spyre,
                    )

                    mismatches, _ = embed_compare_spyre(model_id)
    except Exception as e:
        err: str = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        result["error"] = err
        result["failure_category"] = _classify_failure(
            err, FAILURE_CATEGORY_TEST_EXECUTION_EXCEPTION
        )
    finally:
        result["correct"] = smoke_passed and not mismatches
        result["load"] = load_on_cpu
        if result["failure_category"] is None and load_on_cpu and not result["correct"]:
            result["failure_category"] = FAILURE_CATEGORY_VERIFICATION_FAILED
        return result


def _load_on_cpu(
    model_path: str,
    model_type: ModelType,
) -> tuple[bool, str | None]:
    """Try to load *model_path* on CPU. Returns ``(loaded, error_message)``.

    ``error_message`` is ``None`` on success. On failure, it carries a
    ``"ExcType: message\\n<tail traceback>"`` string that the caller can
    stash into the row's ``error`` field. Transient HF 5xx propagate — the
    driver retries at a higher level.
    """
    import hf_adapters.hf_common as _hf_common
    from hf_adapters import AutoSpyreModelForCausalLM
    from hf_adapters.auto_spyre_model import AutoSpyreModel
    from tests.conftest import get_dtype_for_cpu

    _orig_device = _hf_common.DEVICE  # save
    _hf_common.DEVICE = "cpu"  # patch
    try:
        dtype = get_dtype_for_cpu(model_path)
        model = None
        match model_type:
            case ModelType.EMBEDDING:
                model = AutoSpyreModel.from_pretrained(model_path, dtype=dtype)
            case ModelType.GENERATIVE:
                model = AutoSpyreModelForCausalLM.from_pretrained(
                    model_path, dtype=dtype
                )

        return model is not None, None
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code >= 500:
            raise
        err: str = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_load_on_cpu exception - {e}")
        return False, err
    except Exception as e:
        err = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_load_on_cpu exception - {e}")
        return False, err
    finally:
        _hf_common.DEVICE = _orig_device  # restore


def _cpu_generate(model_path: str) -> tuple[bool, str | None]:
    """Run a single-prompt HF ``generate()`` on CPU for *model_path*.

    Separate from ``_load_on_cpu``: load succeeding tells us the checkpoint
    is well-formed, generate succeeding tells us the forward pass runs
    end-to-end (catches lazy shape errors, tokenizer/config mismatches, and
    custom-code bugs that don't surface at ``from_pretrained`` time).

    Returns ``(ok, error_message)`` with the same convention as
    ``_load_on_cpu``.
    """
    import hf_adapters.hf_common as _hf_common
    from tests.cpu._generate_helpers import simple_generate

    _orig_device = _hf_common.DEVICE
    _hf_common.DEVICE = "cpu"
    try:
        simple_generate(model_path=model_path)
        return True, None
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code >= 500:
            raise
        err: str = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_cpu_generate exception - {e}")
        return False, err
    except Exception as e:
        err = (
            f"{type(e).__name__}: {e}\n"
            f"{''.join(_traceback.format_exc().splitlines(keepends=True)[-6:])}"
        )
        print(f"_cpu_generate exception - {e}")
        return False, err
    finally:
        _hf_common.DEVICE = _orig_device
