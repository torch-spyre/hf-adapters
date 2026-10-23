"""Spyre lm-eval-harness backend and an ad-hoc task runner.

For repeatable suites, native GPU references, and comparisons, use
``python -m evals.lm_eval.run_benchmarks``; see this directory's README.

On a Spyre pod, activate the runtime environment and install ``lm_eval`` first.
Run from the project root so hf_adapters is on PYTHONPATH::

    python -m evals.lm_eval.run_spyre_lm_eval
    python -m evals.lm_eval.run_spyre_lm_eval --model Qwen/Qwen3-0.6B \
        --tasks mmlu --limit 1

``--limit`` applies per task (so ``mmlu`` runs all subjects). This is a wiring
probe, not a meaningful benchmark score. MMLU exercises loglikelihood rather
than generation; additional harness adaptations may still be needed.
Exceptions are deliberately left uncaught to expose the first incompatibility.
"""

from __future__ import annotations

import argparse
import inspect

import torch
import torch.nn.functional as F
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM, get_dtype
from lm_eval.models.utils import DEFAULT_MAX_LENGTH

from hf_adapters import AutoSpyreModelForCausalLM
from hf_adapters.auto_spyre_model import dtype_for_model_path, resolve_adapter_module
from hf_adapters.hf_common import BLOCK_SIZE, prefill_logits, text_config


@register_model("spyre_hf")
class SpyreHFLM(HFLM):
    """Reuse HFLM with the Spyre causal-LM loader (experimental)."""

    AUTO_MODEL_CLASS = AutoSpyreModelForCausalLM
    _DEFAULT_MAX_LENGTH = DEFAULT_MAX_LENGTH

    def __init__(self, pretrained: str, **kwargs):
        """Validate Spyre options before HFLM loads configs, tokens or weights."""
        if not isinstance(pretrained, str):
            raise ValueError("spyre_hf requires a model path, not a preloaded model")
        # Use the parent signature to distinguish harness options from extra
        # from_pretrained kwargs, which our narrower loader cannot accept.
        bound = inspect.signature(HFLM.__init__).bind(self, pretrained, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        unsupported = {
            "backend": options["backend"] not in ("default", "causal"),
            "mixed_precision_dtype": options["mixed_precision_dtype"] is not None,
            "revision": options["revision"] not in (None, "main"),
            "subfolder": bool(options["subfolder"]),
            "parallelize": bool(options["parallelize"]),
            "tp_plan": options["tp_plan"] is not None,
            "max_memory_per_gpu": options["max_memory_per_gpu"] is not None,
            "max_cpu_memory": options["max_cpu_memory"] is not None,
            "offload_folder": options["offload_folder"] not in (None, "./offload"),
            "peft": options["peft"] is not None,
            "delta": options["delta"] is not None,
            "autogptq": bool(options["autogptq"]),
            "gptqmodel": bool(options["gptqmodel"]),
            "gguf_file": options["gguf_file"] is not None,
            "device": kwargs.get("device") not in (None, "cpu", "spyre", "spyre:0"),
        }
        rejected = [name for name, requested in unsupported.items() if requested]
        rejected.extend(options["kwargs"])
        if rejected:
            raise ValueError(
                "spyre_hf does not support these options yet: " + ", ".join(rejected)
            )
        # HFLM owns CPU token handling/scoring; the adapter loader owns Spyre
        # placement. This also makes the parent's generation autocast context
        # a disabled CPU context rather than an unsupported Spyre autocast.
        kwargs["device"] = "cpu"
        kwargs["backend"] = "causal"
        super().__init__(pretrained=pretrained, **kwargs)

    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        tp_plan: str | dict | None = None,
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        quantization_config=None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        """Bridge HFLM loading options to the narrower Spyre loader API."""
        if self.backend != "causal":
            raise ValueError("spyre_hf only supports causal language models")

        # HFLM discovers quantization metadata in the checkpoint config. Leave
        # its interpretation to the normal adapter loader: e.g. Ministral 3's
        # FP8 checkpoint is dequantized on CPU before moving to Spyre. Do not
        # forward HFLM's quantization_config or erase the checkpoint metadata.
        # Explicit user quantization overrides are rejected by __init__.

        spyre_dtype = (
            dtype_for_model_path(
                pretrained,
                target_device="spyre",
                trust_remote_code=trust_remote_code,
            )
            if dtype is None or dtype == "auto"
            else get_dtype(dtype)
        )
        self._model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            dtype=spyre_dtype,
            tp_plan=tp_plan,
            trust_remote_code=trust_remote_code,
        )
        self._spyre_adapter = resolve_adapter_module(
            pretrained, trust_remote_code=trust_remote_code
        )
        # The harness handles tokenization, log_softmax, gather and comparisons
        # on CPU. Only _model_call's adapter forward runs on the model's device.
        self._device = torch.device("cpu")

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare HFLM inputs and convert raw adapter logits for CPU scoring."""
        if self.backend != "causal" or attn_mask is not None or labels is not None:
            raise ValueError(
                "spyre_hf scoring supports causal inputs without labels/masks"
            )
        if not isinstance(inps, torch.Tensor):
            raise TypeError("inps must be a torch.Tensor")
        if inps.ndim != 2 or min(inps.shape) == 0:
            raise ValueError("inps must have nonempty shape [batch, sequence]")
        if inps.dtype not in (torch.int32, torch.int64):
            raise TypeError("inps must contain integer token IDs")

        batch_size, input_length = inps.shape
        pad_multiple = (
            getattr(self.model, "_spyre_prefill_chunk_size", None) or BLOCK_SIZE
        )
        padded_len = ((input_length + pad_multiple - 1) // pad_multiple) * pad_multiple
        # HFLM right-pads batches without a padding mask. Preserve all supplied
        # positions and IDs (including zero); trailing pads cannot affect earlier
        # logits under causal attention. Add only right-side padding, keeping
        # every query chunk full-sized when the adapter requires chunking.
        input_ids = F.pad(inps.detach().cpu(), (0, padded_len - input_length))
        position_ids = torch.arange(padded_len, dtype=torch.int32).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1).contiguous()
        prompt_offsets = torch.zeros(batch_size, dtype=torch.long)
        device = self.model.device
        logits = prefill_logits(
            self._spyre_adapter._run_forward,
            self.model,
            input_ids.to(device),
            position_ids.to(device),
            prompt_offsets,
        )
        # Crop on CPU before softmax: padded vocabulary entries must not enter
        # the scoring denominator. Return one logit row per original input token.
        return logits.cpu()[
            :, :input_length, : text_config(self.model.config).vocab_size
        ].float()


def main(argv: list[str] | None = None) -> None:
    import lm_eval
    from lm_eval.utils import make_table

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tasks", nargs="+", default=["mmlu_abstract_algebra"])
    parser.add_argument(
        "--limit", type=int, default=1, help="Questions per task (default: 1)."
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-gen-toks",
        type=int,
        default=None,
        help="Override generation-task token budget (e.g. 128 for a GSM8K probe).",
    )
    args = parser.parse_args(argv)
    if args.limit < 1 or args.max_length < 1 or args.num_fewshot < 0:
        parser.error("limit/max-length must be positive and num-fewshot nonnegative")
    if args.max_gen_toks is not None and not 0 < args.max_gen_toks < args.max_length:
        parser.error("max-gen-toks must be positive and less than max-length")

    # The decorator above registers spyre_hf before simple_evaluate resolves it.
    # Loading and forward are adapted above; reuse inherited CPU scoring.
    results = lm_eval.simple_evaluate(
        model="spyre_hf",
        model_args={
            "pretrained": args.model,
            "backend": "causal",
            "dtype": "auto",
            "max_length": args.max_length,
        },
        tasks=args.tasks,
        device="spyre",
        batch_size=1,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        gen_kwargs=(
            {"max_gen_toks": args.max_gen_toks}
            if args.max_gen_toks is not None
            else None
        ),
    )
    if results is not None:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    main()
