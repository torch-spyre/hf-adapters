"""
Automatic module configuration generator for **vLLM (v1)** models.

This is the vLLM analog of ``auto_generate_module_config.py`` (which targets
HuggingFace ``transformers`` models). It:

1. Loads the model with ``LLM(model=..., enforce_eager=True)`` (vLLM v1, 1 GPU,
   tensor_parallel_size=1).
2. Registers forward pre-hooks on the *target* modules via ``llm.apply_model()``
   (which runs the function inside the worker process where the model lives).
3. Runs a single **prefill** pass (``generate`` with ``max_tokens=1``) so the
   hooks observe the real forward inputs (shape / dtype / structure).
4. Emits a unified YAML config so each captured module can be re-run standalone
   by ``tests/test_modules_custom.py::test_vllm`` (which does NOT use ``LLM()``;
   it rebuilds the module from ``AutoConfig`` under a vLLM config + distributed
   init, see that file).

Scope of this version:
- **prefill only** — KV cache / decode phase is deferred to a later version;
  ``past_key_values`` kwargs are dropped during capture.
- **layer 0 only** — one representative decoder layer, including its submodules
  (e.g. ``model.layers.0.self_attn.rotary_emb``).

Model name embedding: the model id is recorded inside a *config-type constructor
arg* (``{config_path, model_id, config_overrides}``), the same convention used by
the ``config_source="pretrained"`` path of ``auto_generate_module_config.py`` and
consumed by the OOT framework's ``_build_hf_config``. ``test_vllm`` reads it back
to build the module's config via ``AutoConfig.from_pretrained(model_id)``.

Usage (on a GPU box with vLLM):
    python auto_generate_module_config_vllm.py \
        --model ibm-granite/granite-3.3-2b-instruct --seq-len 128 --dtype bfloat16
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import yaml

# Reuse the device/backend-independent helpers from the HF generator.
from auto_generate_module_config import (
    PrettyDumper,
    _build_module_entry_dict,
    _process_pytree_structure,
)

logger = logging.getLogger(__name__)


def _invocation_signature(invocation_inputs: List[Dict[str, Any]]) -> str:
    """Signature of an invocation's input pattern (shape/dtype/structure only).

    Inlined from ``ModuleInfoCapture._create_invocation_signature`` (which is a
    method, not importable) so identical prefill invocations dedup the same way
    the HF generator dedups them.
    """

    def _extract_pattern(input_info: Dict[str, Any]) -> Dict[str, Any]:
        if "type" in input_info and "items" in input_info:
            return {
                "type": input_info["type"],
                "items": [
                    {
                        "shape": item.get("shape"),
                        "dtype": str(item.get("dtype")),
                        "init": item.get("init"),
                    }
                    for item in input_info["items"]
                ],
            }
        if "shape" in input_info:
            return {
                "type": "tensor",
                "shape": input_info.get("shape"),
                "dtype": str(input_info.get("dtype")),
                "init": input_info.get("init"),
            }
        return {"type": "unknown"}

    patterns = [_extract_pattern(info) for info in invocation_inputs]
    pattern_str = json.dumps(patterns, sort_keys=True)
    return hashlib.sha256(pattern_str.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Target module selection
# ---------------------------------------------------------------------------
#
# Capture, by name:
#   (1) root-direct children (no dot in the name), except "model"
#       -> e.g. "lm_head", "logits_processor"
#   (2) "model"-direct children ("model.<x>" with no further dot), except
#       "model.layers"
#       -> e.g. "model.embed_tokens", "model.norm"
#   (3) everything at / under "model.layers.0" INCLUDING submodules
#       -> e.g. "model.layers.0", "model.layers.0.self_attn",
#          "model.layers.0.self_attn.rotary_emb", "model.layers.0.mlp.act_fn"
#
# Only layer 0 is captured (one representative layer). Excluded outright: the
# root ("") and the pure containers "model" and "model.layers".
#
# The user-named tail modules "model.norm", "lm_head", and "logits_processor"
# are covered by rules (2)/(1). If a backend registers "logits_processor" at a
# different path, extend TARGET_EXTRA below.

_LAYER0_PREFIX = "model.layers.0"
_EXCLUDE_EXACT = {"", "model", "model.layers"}
# Names that must always be captured even if the rules above miss them (e.g. a
# backend nests logits_processor differently). Matched by exact name or as the
# last dotted segment.
TARGET_EXTRA = ("logits_processor",)


def is_target(name: str) -> bool:
    """Return True if ``name`` is a module we want to capture (see module doc)."""
    if name in _EXCLUDE_EXACT:
        return False

    # (3) model.layers.0 and all its submodules
    if name == _LAYER0_PREFIX or name.startswith(_LAYER0_PREFIX + "."):
        return True

    # Any other module under model.layers.* (i.e. layers 1..N) is skipped —
    # layer 0 is the representative layer.
    if name.startswith("model.layers."):
        return False

    # (1) root-direct: no dot
    if "." not in name:
        return True

    # (2) model-direct: "model.<x>" with no further dot
    if name.startswith("model.") and "." not in name[len("model.") :]:
        return True

    # Safety net for explicitly-requested tail modules registered elsewhere.
    if name in TARGET_EXTRA or name.rsplit(".", 1)[-1] in TARGET_EXTRA:
        return True

    return False


# ---------------------------------------------------------------------------
# Worker-side functions (run inside the vLLM worker via llm.apply_model)
# ---------------------------------------------------------------------------
#
# These must return plain, picklable data only (no tensors): apply_model returns
# results to the main process over RPC.

# Forward kwargs we never record — KV cache is out of scope for this version.
_SKIP_KWARGS = ("past_key_values", "past_key_value")


def _pytree_meta(value: Any, name: str) -> Dict[str, Any] | None:
    """Thin wrapper over the HF generator's pytree extractor.

    Returns tensor metadata (shape/dtype/is_random/container structure) for a
    forward arg/kwarg, or ``None`` if it holds no tensors.
    """
    return _process_pytree_structure(value, name)


def register_hooks(model) -> Dict[str, str]:
    """Attach input-capturing forward pre-hooks to every target module.

    Runs inside the worker. Records, per invocation, the pytree metadata of the
    positional args and keyword args (skipping KV-cache kwargs). Tensor *values*
    are not kept — only shape/dtype/structure.

    Returns ``{module_name: "<module>.<ClassName>"}`` (plain data).
    """
    resolved: Dict[str, str] = {}
    for name, module in model.named_modules():
        if not is_target(name):
            continue

        module._cap_invocations = []

        def make_hook(store):
            def pre_hook(mod, args, kwargs):
                invocation_inputs: List[Dict[str, Any]] = []
                for i, arg in enumerate(args):
                    info = _pytree_meta(arg, f"arg_{i}")
                    if info:
                        invocation_inputs.append(info)
                for key, val in kwargs.items():
                    if key in _SKIP_KWARGS:
                        continue
                    info = _pytree_meta(val, key)
                    if info:
                        invocation_inputs.append(info)
                store.append(invocation_inputs)

            return pre_hook

        module._cap_handle = module.register_forward_pre_hook(
            make_hook(module._cap_invocations), with_kwargs=True
        )
        resolved[name] = f"{module.__class__.__module__}.{module.__class__.__name__}"

    return resolved


# A module is an atomic vLLM library layer (constructor args recoverable from
# instance attributes) vs. a model-specific composite module (built from a
# PretrainedConfig -- handled via the config arg / Option C in the test).
_ATOMIC_PREFIX = "vllm.model_executor.layers."
_COMPOSITE_PREFIX = "vllm.model_executor.models."

# Constructor params that are never captured as values: they are supplied by the
# test at construction time (quant_config=None, a prefix, dtype/config/cache
# come from the vLLM context) rather than recorded in the YAML.
_CTOR_SKIP_PARAMS = frozenset(
    {"self", "quant_config", "prefix", "params_dtype", "config", "cache_config"}
)

# __init__ param name -> instance attribute name(s) to read it back from, for
# the few vLLM layers where the attribute name differs from the param name.
_CTOR_ATTR_ALIASES = {
    "bias": ("has_bias",),
    "eps": ("variance_epsilon",),
    "output_sizes": ("output_sizes",),
    "org_num_embeddings": ("org_vocab_size",),
}


def _capture_ctor_args(module) -> Dict[str, Any] | None:
    """Read an atomic module's constructor kwargs back from instance attributes.

    Runs inside the worker on a constructed module. Uses the class __init__
    signature: for each non-skipped param, read the same-named attribute (or an
    alias). A param that has no default and cannot be read back makes capture
    fail (return None) so the caller falls back to the config arg. Returns a
    dict of ``{param_name: scalar_value}`` (only plain/picklable values are
    kept). ``SiluAndMul`` and similar zero-arg layers yield ``{}``.
    """
    import inspect

    try:
        sig = inspect.signature(type(module).__init__)
    except (ValueError, TypeError):
        return None

    kwargs: Dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname in _CTOR_SKIP_PARAMS:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        value = None
        found = False
        for attr in _CTOR_ATTR_ALIASES.get(pname, (pname,)):
            if hasattr(module, attr):
                value = getattr(module, attr)
                found = True
                break

        if not found:
            if param.default is inspect.Parameter.empty:
                # A required param we cannot recover -> capture unusable.
                return None
            continue  # optional and absent: rely on the default

        # Only keep plain, picklable scalar/list values (skip tensors, modules,
        # dtypes, config objects, etc. -- they are not YAML-representable here).
        if isinstance(value, bool) or isinstance(value, (int, float, str)) or value is None:
            kwargs[pname] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float, str, bool)) for v in value
        ):
            kwargs[pname] = list(value)
        else:
            # Non-scalar (e.g. torch.dtype) required value we cannot represent.
            if param.default is inspect.Parameter.empty:
                return None
            # optional: drop it, the default applies
    return kwargs


def collect_captures(model) -> Dict[str, Dict[str, Any]]:
    """Collect captured invocations + constructor args, then remove hooks.

    Runs inside the worker. Returns, per target module,
    ``{module_path, invocations, ctor_kwargs}`` (plain data). ``ctor_kwargs`` is
    a dict for atomic ``vllm.model_executor.layers.*`` modules whose constructor
    args were recovered from instance attributes, or ``None`` for composite
    ``vllm.model_executor.models.*`` modules (built from the model config in the
    test) and for any atomic module whose args could not be fully recovered.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for name, module in model.named_modules():
        if not is_target(name):
            continue
        module_path = f"{module.__class__.__module__}.{module.__class__.__name__}"
        ctor_kwargs = None
        if module_path.startswith(_ATOMIC_PREFIX):
            ctor_kwargs = _capture_ctor_args(module)
        out[name] = {
            "module_path": module_path,
            "invocations": list(getattr(module, "_cap_invocations", [])),
            "ctor_kwargs": ctor_kwargs,
        }
        handle = getattr(module, "_cap_handle", None)
        if handle is not None:
            handle.remove()
    return out


# ---------------------------------------------------------------------------
# Capture driver (main process)
# ---------------------------------------------------------------------------


def capture_via_llm(
    model_id: str,
    seq_len: int,
    dtype: str,
    model_impl: str | None,
) -> Dict[str, Dict[str, Any]]:
    """Load the model with vLLM, run one prefill, return per-module captures.

    ``enforce_eager=True`` is required: v1 otherwise wraps the model in
    torch.compile + piecewise CUDA graphs, and graph replay bypasses the
    Python-level submodule hooks (so nothing would be captured).
    """
    from vllm import LLM, SamplingParams

    llm_kwargs: Dict[str, Any] = dict(
        model=model_id,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype=dtype,
        trust_remote_code=True,
    )
    if model_impl:
        llm_kwargs["model_impl"] = model_impl

    logger.info("Loading model via vLLM: %s (%s)", model_id, llm_kwargs)
    llm = LLM(**llm_kwargs)

    resolved = llm.apply_model(register_hooks)
    resolved = resolved[0] if isinstance(resolved, (list, tuple)) else resolved
    logger.info("Registered capture hooks on %d target modules", len(resolved))

    # Drive a single prefill. Feed token ids directly so the seqlen is exactly
    # ``seq_len``; max_tokens=1 means prefill only (no decode this version).
    prompt_token_ids = list(range(seq_len))
    prompts = [{"prompt_token_ids": prompt_token_ids}]
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1))

    captures = llm.apply_model(collect_captures)
    captures = captures[0] if isinstance(captures, (list, tuple)) else captures
    return captures


# ---------------------------------------------------------------------------
# YAML assembly
# ---------------------------------------------------------------------------


def _config_constructor_arg(
    config_path: str, model_id: str, config_overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Build the constructor-arg spec that carries the model id (Decision A).

    Shape matches ``auto_generate_module_config.py``'s ``config_source="pretrained"``
    output so ``_convert_constructor_arg_to_sample_input`` emits
    ``{config_path, model_id, config_overrides}`` and the OOT framework resolves
    it to a live HF config via ``AutoConfig.from_pretrained(model_id)``.
    """
    spec: Dict[str, Any] = {"type": "config", "config_path": config_path, "model_id": model_id}
    if config_overrides:
        spec["config_overrides"] = config_overrides
    return spec


def _dedup_invocations(invocations: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    """Drop invocations whose input pattern (shape/dtype) repeats."""
    seen = set()
    unique = []
    for inv in invocations:
        sig = _invocation_signature(inv)
        if sig not in seen:
            seen.add(sig)
            unique.append(inv)
    return unique


def build_captured_modules(
    captures: Dict[str, Dict[str, Any]],
    model_id: str,
    config_path: str,
    config_overrides: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Turn worker captures into ``_build_module_entry_dict`` input dicts.

    Constructor-arg strategy per module:

    - **Atomic** ``vllm.model_executor.layers.*`` modules whose ctor args were
      recovered from instance attributes (``ctor_kwargs`` is a dict): emit those
      as explicit constructor kwargs (plain int/float/str/bool/list values). The
      module is then built generically as ``module_cls(**kwargs)`` with no HF
      config -- no per-class adapter needed.
    - **Composite** ``vllm.model_executor.models.*`` modules (and any atomic
      module whose args could not be recovered): fall back to a config-type
      constructor arg carrying ``model_id`` (Option C), from which the test
      derives the constructor args via the model config.

    A ``_vllm_ctor_kwargs`` key carries the captured kwargs through to the YAML
    emitter (which writes them into ``constructor_inputs.kwargs``); it is not
    consumed by ``_build_module_entry_dict``. Modules that captured no
    invocation (never hit during the prefill) are skipped.
    """
    captured_modules: List[Dict[str, Any]] = []
    for name, data in sorted(captures.items()):
        invocations = _dedup_invocations(data.get("invocations", []))
        if not invocations:
            logger.info("Skipping %s — no forward invocation captured", name)
            continue

        class_name = data["module_path"].rsplit(".", 1)[-1]
        # Name each entry <ClassName>_<8-hex-hash>, matching the HF generator's
        # convention. The OOT framework reduces an include name to the module
        # class name via _extract_base_module_name, which only strips a trailing
        # _<hexhash> / _<digits> / _layer<N> suffix; a <ClassName>_<hash> name
        # reduces to <ClassName> and matches ModuleInfo.name (= the class name),
        # so the module is injected into the @modules variant list. A descriptive
        # suffix like "__model_layers_0_mlp" would NOT reduce and the module would
        # be silently filtered out. The hash of the captured instance path keeps
        # entries unique (layer 0 only, so no real collisions).
        name_hash = hashlib.sha256(name.encode()).hexdigest()[:8]

        ctor_kwargs = data.get("ctor_kwargs")
        if ctor_kwargs is not None:
            # Atomic module: explicit captured constructor kwargs, no config arg.
            constructor_args: List[Dict[str, Any]] = []
            vllm_ctor_kwargs = dict(ctor_kwargs)
        else:
            # Composite / unrecoverable: Option C -- carry the model config arg.
            constructor_args = [
                _config_constructor_arg(config_path, model_id, config_overrides)
            ]
            vllm_ctor_kwargs = {}

        captured_modules.append(
            {
                "name": f"{class_name}_{name_hash}",
                "module_path": data["module_path"],
                "example_instance": name,
                "constructor_args": constructor_args,
                "constructor_kwargs": {},
                "_vllm_ctor_kwargs": vllm_ctor_kwargs,
                "invocations": invocations,
            }
        )
    return captured_modules


# Per-dtype comparison tolerances for the generated ``supported_dtypes`` block.
# Low-precision types use a looser tolerance; float32 is tight.
_DTYPE_PRECISION = {
    "bfloat16": {"atol": 0.005, "rtol": 0.005},
    "float16": {"atol": 0.005, "rtol": 0.005},
    "float32": {"atol": 0.001, "rtol": 0.001},
}


def generate_unified_yaml_config_vllm(
    captured_modules: List[Dict[str, Any]], model_name: str, dtype: str = "bfloat16"
) -> str:
    """Emit the unified YAML for the vLLM ``test_vllm`` runner.

    One file block pointing at ``test_modules_custom.py`` with a single
    ``*TestModuleCustom*::test_vllm`` test entry. Mirrors the structure of
    ``auto_generate_module_config.generate_unified_yaml_config`` but with the
    vLLM test name/tags. ``dtype`` selects the single ``supported_dtypes`` entry
    (and its comparison tolerance) the test runs at; it should match the dtype
    the model was captured with.
    """
    precision = _DTYPE_PRECISION.get(dtype, {"atol": 0.005, "rtol": 0.005})
    module_entries = []
    for m in captured_modules:
        entry = _build_module_entry_dict(m)
        # Append the captured module path (e.g. "model.layers.0.mlp") to the
        # description so the YAML records which named_module each entry came
        # from. _build_module_entry_dict only records the class path, and the
        # entry `name` is a hashed class name, so this is the only place the
        # human-readable instance path is preserved.
        instance = m.get("example_instance")
        if instance:
            entry["description"] = f"{entry['description']} (path: {instance})"
        # Inject captured constructor kwargs (atomic modules). _build_module_entry_dict
        # only emits int-typed constructor_kwargs, so write the full captured dict
        # (int/float/str/bool/list values) directly -- the OOT resolved_kwargs path
        # passes these plain values through unchanged.
        vllm_ctor_kwargs = m.get("_vllm_ctor_kwargs") or {}
        if vllm_ctor_kwargs:
            entry["constructor_inputs"]["kwargs"] = dict(vllm_ctor_kwargs)
        module_entries.append(entry)

    config = {
        "test_suite_config": {
            "files": [
                {
                    "path": "${TORCH_DEVICE_ROOT}/tests/test_modules_custom.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": ["*TestModuleCustom*::test_vllm"],
                            "mode": "mandatory_success",
                            "tags": [f"model__{model_name}", "vllm"],
                            # Spyre custom ops have no autograd formula; build the
                            # modules under no_grad so AOTAutograd does not trace a
                            # backward graph at compile time.
                            "no_grad": True,
                            "edits": {"modules": {"include": module_entries}},
                        }
                    ],
                },
            ],
            "global": {
                "supported_dtypes": [
                    {"name": dtype, "precision": precision},
                ],
                "input_config": {"seed": 123},
            },
        }
    }

    header = (
        f"# Auto-generated vLLM module test configuration for {model_name}\n"
        f"# Generated by auto_generate_module_config_vllm.py\n"
        f"# Consumed by tests/test_modules_custom.py::test_vllm\n\n"
    )
    return header + yaml.dump(
        config,
        Dumper=PrettyDumper,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=float("inf"),
    )


def write_module_config(
    yaml_content: str, model_id: str, output: str | None = None
) -> Path:
    """Write the YAML to ``output`` (or the default module_tests path)."""
    model_name = model_id.rstrip("/").split("/")[-1]
    model_name_normalized = model_name.replace("-", "_").replace(".", "_")

    if output:
        output_path = output
    else:
        output_path = f"./tests/configs/module_tests/{model_name_normalized}_vllm_spyre.yaml"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(yaml_content)
    logger.info("\n✓ Generated vLLM module configuration: %s", output_file)
    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-generate a vLLM module-test YAML by capturing prefill inputs."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model path/id (e.g. ibm-granite/granite-3.3-2b-instruct)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=128, help="Prefill sequence length (default: 128)"
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model load dtype for LLM(dtype=...) and the generated "
        "supported_dtypes (default: bfloat16)",
    )
    parser.add_argument(
        "--model-impl",
        default="native",
        choices=["native", "transformers"],
        help="vLLM model backend (default: native)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output YAML path (default: ./tests/configs/module_tests/<model>_vllm_spyre.yaml)",
    )
    return parser.parse_args()


def _resolve_hf_config_path(model_id: str) -> str:
    """Best-effort ``module.ClassName`` of the model's HF config class.

    Used only as a human-readable hint in the YAML; the OOT framework resolves
    the config from ``model_id`` via ``AutoConfig.from_pretrained``, so this is
    not load-bearing. Falls back to a plain ``AutoConfig`` label if lookup fails.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        return f"{type(cfg).__module__}.{type(cfg).__name__}"
    except Exception as exc:  # pragma: no cover - hint only
        logger.warning("Could not resolve HF config class for %s: %s", model_id, exc)
        return "transformers.AutoConfig"


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    captures = capture_via_llm(
        model_id=args.model,
        seq_len=args.seq_len,
        dtype=args.dtype,
        model_impl=args.model_impl,
    )

    config_path = _resolve_hf_config_path(args.model)
    captured_modules = build_captured_modules(
        captures,
        model_id=args.model,
        config_path=config_path,
        config_overrides={},
    )

    model_name = args.model.rstrip("/").split("/")[-1].replace("-", "_").replace(".", "_")
    yaml_content = generate_unified_yaml_config_vllm(
        captured_modules, model_name, dtype=args.dtype
    )
    output_file = write_module_config(yaml_content, args.model, args.output)

    logger.info("\n  Module summary: %d modules captured", len(captured_modules))
    for m in captured_modules:
        logger.info("    - %s (%s)", m["name"], m["module_path"])
    logger.info("  YAML: %s", output_file)


if __name__ == "__main__":
    main()
