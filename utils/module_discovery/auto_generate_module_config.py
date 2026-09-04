"""
Automatic module configuration generator using forward hooks.

This script automatically generates YAML configuration for all unique modules
in a model by:
1. Loading the model
2. Registering forward hooks on all modules
3. Running a forward pass to capture module inputs
4. Analyzing captured data to generate YAML config

Usage:
    python auto_generate_module_config.py --model_path ibm-granite/granite-3.3-8b-instruct --seq_len 128
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml
from torch.utils._pytree import tree_flatten
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, StaticCache

logger = logging.getLogger(__name__)


# Get existing modules from PyTorch's module_db to avoid duplicates
try:
    from torch.testing._internal.common_modules import module_db

    # Extract just the class name from module_db names (e.g., "nn.Linear" -> "Linear")
    existing_modules = set()
    for m in module_db:
        # module_db names are like "nn.Linear", "nn.Conv2d", etc.
        if "." in m.name:
            class_name = m.name.split(".")[-1]
            existing_modules.add(class_name)
        else:
            existing_modules.add(m.name)
    logger.info(
        f"Found {len(existing_modules)} existing modules in PyTorch's module_db"
    )
except ImportError:
    existing_modules = set()
    logger.warning("could not import module_db, will not filter duplicates")


class PrettyDumper(yaml.SafeDumper):
    """Custom YAML dumper with consistent 2-space indentation."""

    def increase_indent(self, flow=False, indentless=False):
        """Ensure consistent indentation (no indentless sequences)."""
        return super().increase_indent(flow, False)

    def represent_data(self, data):
        """Override to handle shape lists specially."""
        # Check if this is a list that should be inline (shape values)
        if isinstance(data, list) and len(data) > 0:
            # Check if all elements are integers (shape lists are all ints)
            if all(isinstance(x, int) for x in data):
                # This is likely a shape list - use flow style
                return self.represent_sequence(
                    "tag:yaml.org,2002:seq", data, flow_style=True
                )

        # For everything else, use default representation
        return super().represent_data(data)


def _is_special_tensor(name: str) -> bool:
    """Check if tensor name indicates it should not be random."""
    return "position_embeddings" not in name.lower() and any(
        keyword in name.lower() for keyword in ["position", "mask", "ids"]
    )


# Extracted from the loaded config so a standalone module rebuilt from the YAML
# dispatches to the same attention path used at capture time. ``from_pretrained``
# leaves ``config._attn_implementation`` as ``None`` on some models, and a ``None``
# value makes ``AttentionInterface.get_interface`` emit the "standalone Module"
# warning and fall back to eager. Writing the resolved value keeps the generated
# config faithful to the runtime implementation.
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"

# The dtype Spyre actually runs in. ``from_pretrained`` defaults to float32, but
# Spyre executes in bfloat16, so both the capture path (``load_model_only``) and
# the YAML emit path (``_tensor_info_to_spec``) default floating-point tensors to
# bfloat16. This keeps the generated config faithful to the runtime dtype
# regardless of the checkpoint's stored precision. Only floating-point dtypes are
# remapped; integer/bool tensors (ids, masks, positions) keep their own dtype.
DEFAULT_FLOAT_DTYPE = torch.bfloat16
_FLOAT_DTYPE_ALIASES = ("float16", "float32", "float64", "float", "half", "double")

# Special tensors (position/mask/ids -- see ``_is_special_tensor``) carry indices
# rather than activations, so they are forced to this integer dtype regardless of
# the dtype they were captured under. This makes their ``randint`` init consistent
# (randint on a floating-point dtype is meaningless).
DEFAULT_INT_DTYPE = torch.int64


def _resolve_attn_implementation(config: Any) -> str:
    """Return the attention implementation the model actually used.

    Prefers the concrete value set on the loaded config; falls back to
    ``DEFAULT_ATTN_IMPLEMENTATION`` (the ``from_pretrained`` default) when the
    config still reports ``None``, so the generated YAML never carries a null
    that would trigger the standalone-module warning.
    """
    impl = getattr(config, "_attn_implementation", None)
    if impl is None:
        return DEFAULT_ATTN_IMPLEMENTATION
    return impl


def _layer_config(config: Any, layer_index: Optional[int]) -> Any:
    """Return the config to read a layer's dimensions from.

    A heterogeneous config (``per_layer_config``, e.g. a model whose layers differ
    in ``num_key_value_heads``) refuses to serve per-layer attributes off the
    global object: ``HeterogeneousConfigMixin.__getattribute__`` raises
    ``AmbiguousGlobalPerLayerAttributeError`` and directs the caller to
    ``config.per_layer_config[i].<attr>``. That exception is a ``RuntimeError``,
    NOT an ``AttributeError``, so even ``hasattr(config, attr)`` propagates it --
    a plain global read crashes rather than silently returning a wrong value.

    Returns the per-layer config when this config is heterogeneous and the layer is
    known, otherwise ``config`` unchanged. The per-layer entries are themselves
    homogeneous instances of the same config class, so callers can read them
    normally.
    """
    if layer_index is None:
        return config
    if not getattr(config, "is_heterogeneous", False):
        return config
    per_layer = getattr(config, "per_layer_config", None)
    if not per_layer or layer_index >= len(per_layer):
        logger.warning(
            "Config is heterogeneous but has no per_layer_config entry for layer "
            "%s; falling back to the global config, whose per-layer attributes may "
            "raise or be wrong.",
            layer_index,
        )
        return config
    return per_layer[layer_index]


# Sentinel distinguishing "attribute absent" from a legitimate ``None`` value.
_CONFIG_ATTR_MISSING = object()


def _read_config_attr(config: Any, attr: str) -> Any:
    """Read one config attribute, tolerating heterogeneous per-layer attributes.

    Returns :data:`_CONFIG_ATTR_MISSING` when the attribute cannot be read.

    ``hasattr``/``getattr`` are not enough on a heterogeneous config. Reading a
    per-layer attribute (e.g. ``num_key_value_heads`` on Gemma 4, whose layers mix
    full and sliding attention) raises ``AmbiguousGlobalPerLayerAttributeError``,
    and because that is a ``RuntimeError`` rather than an ``AttributeError`` even
    ``hasattr`` propagates it.

    :func:`_layer_config` handles the case where the layer is known. It cannot help
    here: a module may legitimately own the heterogeneous config without belonging
    to one layer -- Gemma 4's ``language_model`` is the whole decoder stack, so it
    has no layer index, yet its config carries the per-layer attributes. For those,
    the only correct answer is "there is no single value", so the attribute is
    dropped and the framework falls back to the config's own default when
    rebuilding. Dropping is safer than recording one layer's value as if it were
    global.
    """
    try:
        return getattr(config, attr)
    except AttributeError:
        return _CONFIG_ATTR_MISSING
    except RuntimeError as exc:
        # AmbiguousGlobalPerLayerAttributeError and anything else the config raises
        # to say "this value is not well defined globally".
        logger.debug(
            "Skipping config attribute %r: not readable off this config (%s)",
            attr,
            type(exc).__name__,
        )
        return _CONFIG_ATTR_MISSING


def _extract_config_kwargs(
    config: Any, layer_index: Optional[int] = None
) -> Dict[str, Any]:
    """Extract the config parameters the framework needs to rebuild a module.

    ``_attn_implementation`` is resolved to a concrete implementation (never
    ``None``) so a module reconstructed from the YAML dispatches attention the
    same way it did during capture.

    ``layer_index`` selects which layer's values to read on a heterogeneous config
    (see :func:`_layer_config`). Without it, reading a per-layer attribute such as
    ``num_key_value_heads`` off the global config raises
    ``AmbiguousGlobalPerLayerAttributeError``. It is optional so the out-of-layer
    callers (and homogeneous models, i.e. every model today) are unaffected.
    """
    config = _layer_config(config, layer_index)

    config_kwargs: Dict[str, Any] = {}
    for attr in [
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
    ]:
        value = _read_config_attr(config, attr)
        if value is not _CONFIG_ATTR_MISSING:
            config_kwargs[attr] = value

    if _read_config_attr(config, "_attn_implementation") is not _CONFIG_ATTR_MISSING:
        config_kwargs["_attn_implementation"] = _resolve_attn_implementation(config)

    return config_kwargs


def _spyre_config_kwargs(config: Any, model: Any) -> Dict[str, Any]:
    """``_extract_config_kwargs`` plus the head_dim the Spyre path actually runs.

    ``pad_attention_heads`` lifts head_dim to a Spyre stick boundary and rewrites
    the Q/K/V/O projections to that padded width, recording the result on
    ``model._spyre_head_dim``. Emitting that value means a module rebuilt from the
    generated YAML gets the PADDED projection shapes -- the ones the adapter
    actually runs -- rather than the checkpoint's unpadded ones.

    Only the shape is reproduced, not the padded weights: padding writes zeros
    into the new positions, and a module test builds fresh weights anyway. A model
    needing no padding (head_dim already >= 2 * BLOCK_SIZE) carries no
    ``_spyre_head_dim``, and its config value is already correct.
    """
    config_kwargs = _extract_config_kwargs(config)
    spyre_head_dim = getattr(model, "_spyre_head_dim", None)
    if spyre_head_dim is not None:
        config_kwargs["head_dim"] = spyre_head_dim
    return config_kwargs


# ---------------------------------------------------------------------------
# Spyre adapter capture (--loader spyre)
# ---------------------------------------------------------------------------

# The device the Spyre execution path runs on. ``hf_common.DEVICE`` is pinned to
# this; the loader below can patch it to "cpu" for an off-pod dry run.
SPYRE_DEVICE = "spyre"

# Spyre-introduced wrapper modules, i.e. the classes ``prepare_for_spyre()`` puts
# into the model tree in place of the HF ones. Named here rather than detected
# structurally because each needs different handling.
SPYRE_ATTENTION_TYPE = "StandardGQAAttention"
SPYRE_BLOCK_TYPE = "StandardGQABlock"

# Module types excluded from a --loader spyre YAML.
#
# StandardGQABlock: deliberately not emitted, to keep the test framework simple.
# Its ctor takes a live HF DecoderLayer *and* an is_res_mul flag, and it owns
# submodules (mlp, both norms, self_attn) that are already captured as standalone
# entries in their own right. Emitting it would need a second module-arg nesting
# level for no extra module-level coverage. The block's own arithmetic (residual
# order, residual_multiplier, norm placement) is covered end-to-end by
# tests/spyre/test_e2e_*.
#
# PrecomputedRotaryEmbedding / InvFreqShim: Spyre-internal RoPE plumbing wrapping
# the HF rotary embedding; the HF module itself is captured instead.
SPYRE_EXCLUDED_MODULE_TYPES = frozenset(
    {
        SPYRE_BLOCK_TYPE,
        "PrecomputedRotaryEmbedding",
        "InvFreqShim",
    }
)


def _ensure_hf_adapters_importable() -> None:
    """Make ``import hf_adapters`` work when run as a script from this directory.

    ``hf_adapters`` is not pip-installed on the Spyre pod (see CLAUDE.md), and
    running this file directly puts ``utils/module_discovery/`` on sys.path rather
    than the repo root. Every ``hf_adapters`` import in this module is lazy (inside
    a function) precisely so the HF loader path never needs this; the Spyre path
    calls it first. A pre-existing installed/PYTHONPATH copy wins -- this only adds
    a fallback.
    """
    import importlib.util
    import sys

    if importlib.util.find_spec("hf_adapters") is not None:
        return
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        logger.info("Adding repo root to sys.path for hf_adapters: %s", repo_root)
        sys.path.insert(0, repo_root)


# ---------------------------------------------------------------------------
# Execution-phase split (both loaders)
# ---------------------------------------------------------------------------
#
# Upstream builds the generated test name from ``module_info.formatted_name``
# (common_modules.py), so a YAML entry carrying several invocations becomes several
# ModuleInputs under ONE test id: a failure cannot be attributed to a phase, and a
# failing early invocation stops the later ones from running at all. Emitting one
# entry per phase -- with the phase in the name -- gives each its own test id.
#
# The OOT framework already makes the YAML ``name`` authoritative
# (_make_named_module_info_cls), so this needs no framework change.

# Both loaders produce exactly two forward shapes: the prompt pass and the
# per-token pass. The same two labels are used for both, so one ``-k prefill`` /
# ``-k decode`` selects the same phase either way, even though the shapes differ
# (hf sees 128/1; the Spyre adapter sees a block-padded prompt then 1).
#
# There is deliberately no third label. The Spyre path used to walk a 64-slot block
# -- an ``expansion`` forward claiming a block, then ``is_filling=True`` writes into
# it -- but hf-adapters#330 replaced that with an indirect scatter and one token per
# step, so ``is_filling`` no longer exists and those two phases are gone with it.
PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"


# Integer dtypes mark a 2-D tensor as token ids / indices (``[batch, seq_len]``)
# rather than a flattened activation (``[tokens, hidden]``). Recorded by
# ``_extract_tensor_info`` as a bare name (no "torch." prefix).
_INT_DTYPE_NAMES = frozenset(
    {
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "long",
        "int",
        "short",
        "bool",
    }
)


def _is_int_dtype_name(dtype: Any) -> bool:
    """True when a recorded dtype string names an integer/bool dtype."""
    if not isinstance(dtype, str):
        return False
    return dtype.replace("torch.", "") in _INT_DTYPE_NAMES


def _invocation_seq_len(invocation_inputs: List[Dict[str, Any]]) -> Optional[int]:
    """Sequence length of an invocation, read from its first 3-D or 2-D tensor.

    A 3-D activation is ``[batch, seq_len, hidden]``, so dim 1 is the sequence
    length. 3-D is preferred over 2-D: where both are present the 3-D one is the
    activation, while a 2-D tensor could be an unrelated 2-D argument.

    2-D needs the dtype to be read correctly, because two different layouts arrive
    at that rank and dim 1 means something different in each:

    - ``[batch, seq_len]`` token ids / positions -- an INTEGER dtype. seq_len is
      dim 1.
    - ``[tokens, hidden]`` a flattened activation -- a FLOAT dtype. Some modules are
      called with the batch and sequence axes already folded together, so the
      sequence axis is dim 0 and dim 1 is the hidden size. ``GptOssMLP`` does
      exactly this before calling its router and experts
      (``hidden_states.reshape(-1, hidden_dim)``); reading dim 1 there returns
      hidden_size (2880) for BOTH the prompt and the per-token pass, which then
      compare equal and label both invocations ``prefill`` -- so the decode entry
      is lost and the two collide under one name.

    Returns ``None`` when neither rank is present (e.g. a 4-D-only module), leaving
    that module unsplit rather than risking a wrong label.
    """

    def _seq_len_of(shape: List[int], dtype: Any, rank: int) -> Optional[int]:
        if len(shape) != rank:
            return None
        if rank == 2 and not _is_int_dtype_name(dtype):
            # Flattened activation [tokens, hidden]: the sequence axis is dim 0.
            return shape[0]
        return shape[1]

    def _scan(rank: int) -> Optional[int]:
        for inp in invocation_inputs:
            shape = inp.get("shape")
            if isinstance(shape, list):
                found = _seq_len_of(shape, inp.get("dtype"), rank)
                if found is not None:
                    return found
            for item in inp.get("items", []) or []:
                shape = item.get("shape")
                if isinstance(shape, list):
                    found = _seq_len_of(shape, item.get("dtype"), rank)
                    if found is not None:
                        return found
        return None

    return _scan(3) if _scan(3) is not None else _scan(2)


def _invocation_feature_width(
    invocation_inputs: List[Dict[str, Any]],
) -> Optional[int]:
    """Trailing (feature) dimension of an invocation's first 3-D tensor.

    Used only to disambiguate two invocations that share a phase label, where the
    sequence length is by definition equal and the feature width is what differs --
    e.g. a gated MLP's ``nn.Linear`` called at both 4096 and 12800.

    Returns ``None`` when no 3-D tensor is present, in which case the caller leaves
    the collision in place rather than inventing a suffix.
    """
    for inp in invocation_inputs:
        shape = inp.get("shape")
        if isinstance(shape, list) and len(shape) == 3:
            return shape[-1]
        for item in inp.get("items", []) or []:
            shape = item.get("shape")
            if isinstance(shape, list) and len(shape) == 3:
                return shape[-1]
    return None


def _invocation_scalar_values(
    invocation_inputs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Scalar (non-tensor) forward args of one invocation, as ``{arg name: value}``.

    ``_extract_scalar_info`` records these alongside the tensors in the same
    ``invocation_inputs`` list, tagged ``type: "value"``, so they only need reading
    back -- see :func:`_scalar_disambiguator` for why they are useful.
    """
    return {
        inp["name"]: inp.get("value")
        for inp in invocation_inputs
        if isinstance(inp, dict) and inp.get("type") == "value" and "name" in inp
    }


def _scalar_token(value: Any) -> str:
    """Render a scalar value as a short, filename/test-id-safe name fragment."""
    text = "None" if value is None else str(value)
    safe = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_")
    return safe[:32] or "value"


def _scalar_disambiguator(
    invocation_inputs: List[Dict[str, Any]],
    differing_keys: List[str],
) -> Optional[str]:
    """Name fragment built from the scalar args that differ across a colliding group.

    Some modules are called several times with identical tensor shapes and differ
    only in a scalar argument, so neither the phase label nor the feature width can
    separate them. Gemma 4's ``Gemma4UnifiedTextRotaryEmbedding`` is the case in
    hand: one instance lives outside the decoder stack
    (``model.rotary_emb``) and is called once per layer *type*
    (``rotary_emb(hidden_states, position_ids, layer_type)``), where ``layer_type``
    selects a different frequency table (``<layer_type>_inv_freq``). The two calls
    therefore compute different values from the same shapes, and a layer index
    cannot tell them apart -- there is only one instance, outside any layer.

    Only keys whose value actually VARIES within the colliding group are used
    (``differing_keys``, computed by the caller). Scalars that are constant across
    the group -- ``use_cache=True``, ``attention_mask=None``, ``return_dict=True``,
    present in nearly every captured config -- carry no information here and would
    rename entries for every model while separating nothing.

    Returns ``None`` when the group has no differing scalar, leaving the caller to
    keep its existing behaviour (one entry plus a warning).
    """
    if not differing_keys:
        return None
    scalars = _invocation_scalar_values(invocation_inputs)
    parts = [_scalar_token(scalars[key]) for key in differing_keys if key in scalars]
    return "_".join(parts) if parts else None


def _differing_scalar_keys(
    invocations: List[List[Dict[str, Any]]],
) -> List[str]:
    """Scalar arg names whose value is not the same across every invocation given.

    Sorted for a deterministic suffix order, so a name does not change between runs.
    """
    per_invocation = [_invocation_scalar_values(inv) for inv in invocations]
    keys = {key for scalars in per_invocation for key in scalars}
    differing = [
        key
        for key in keys
        if len({repr(scalars.get(key)) for scalars in per_invocation}) > 1
    ]
    return sorted(differing)


def _phase_label(
    invocation_inputs: List[Dict[str, Any]],
    prompt_seq_len: Optional[int],
) -> Optional[str]:
    """Label one invocation ``prefill`` or ``decode`` by its sequence length.

    Both loaders produce exactly two forward shapes -- the prompt pass and the
    per-token pass -- so the same two labels cover both, and one ``-k decode``
    selects the same phase either way.

    The prefill test is relative (``seq_len == prompt_seq_len``), never a fixed
    length: the two loaders run different shapes (hf sees 128/1; the Spyre adapter
    block-pads the prompt), so any hardcoded value would mislabel one of them.

    Returns ``None`` when no sequence length can be read, leaving the entry unsplit.
    """
    if prompt_seq_len is None:
        return None
    seq_len = _invocation_seq_len(invocation_inputs)
    if seq_len is None:
        return None
    return PHASE_PREFILL if seq_len == prompt_seq_len else PHASE_DECODE


def split_module_data_by_phase(capture: "ModuleInfoCapture") -> None:
    """Rewrite ``capture.module_data`` so each entry holds exactly one invocation.

    Runs after capture rather than inside the hooks because the prefill label is
    defined relative to the longest sequence a module saw, which is only known once
    every invocation has been observed.

    Each phase becomes its own entry named ``<original name>_<phase>``, keeping the
    original identifier (a dim, a layer index, a config hash) so modules that differ
    by config stay distinguishable. A module whose invocations cannot be labelled
    (no sequence length readable from any argument) is left exactly as it was -- an
    unsplit entry is better than a mislabelled one.

    A phase can still hold several invocations when one class is used at more than
    one width under the same config -- ``nn.Linear`` serves both the 4096->12800 and
    12800->4096 halves of a gated MLP, and both are captured under one config
    signature. Those get the feature width appended (``..._decode_h4096``) so each
    still lands in its own entry with its own test id.

    Where the shapes are identical too, a differing scalar argument is appended
    instead (``..._prefill_sliding_attention``); see
    :func:`_scalar_disambiguator`. Only scalars that actually vary within the
    colliding group are used, so the constant ones every model records
    (``use_cache``, ``attention_mask=None``) never enter a name.
    """
    split: Dict[str, Dict[str, Any]] = {}

    for name, data in capture.module_data.items():
        invocations = data.get("invocations", [])

        # The prompt pass is the longest sequence this module saw.
        seq_lens = [_invocation_seq_len(inv) for inv in invocations]
        known = [s for s in seq_lens if s is not None]
        prompt_seq_len = max(known) if known else None

        labels = [_phase_label(inv, prompt_seq_len) for inv in invocations]
        if len(invocations) < 2 or any(lbl is None for lbl in labels):
            if len(invocations) > 1:
                logger.info(
                    "%s: keeping %d invocations in one entry (no phase label "
                    "available for every invocation).",
                    name,
                    len(invocations),
                )
            split[name] = data
            continue

        # Disambiguate only where a label is genuinely reused, so the common case
        # keeps the short ``<name>_<phase>`` form.
        duplicated = {lbl for lbl in labels if labels.count(lbl) > 1}

        # Scalars that vary within a reused label are the last axis available when
        # the shapes match as well. Computed per label group, not over the whole
        # module: a scalar that differs only between prefill and decode is already
        # covered by the phase label and must not also enter the name.
        differing_scalars = {
            label: _differing_scalar_keys(
                [inv for lbl, inv in zip(labels, invocations) if lbl == label]
            )
            for label in duplicated
        }

        for label, inv in zip(labels, invocations):
            phase_name = f"{name}_{label}"
            if label in duplicated:
                width = _invocation_feature_width(inv)
                if width is not None:
                    phase_name = f"{phase_name}_h{width}"
                # Append a scalar fragment only where it is needed to separate this
                # group -- appending it unconditionally would rename entries whose
                # width already distinguishes them.
                scalar_suffix = _scalar_disambiguator(
                    inv, differing_scalars.get(label, [])
                )
                if scalar_suffix is not None:
                    phase_name = f"{phase_name}_{scalar_suffix}"
            if phase_name in split:
                # Still colliding: two invocations share a label AND a width. Keep
                # both rather than dropping one, and say so -- the test id cannot
                # then tell them apart.
                logger.warning(
                    "%s: more than one invocation labelled %r with the same width; "
                    "leaving them in one entry, so a failing test id will not say "
                    "which one broke.",
                    phase_name,
                    label,
                )
                split[phase_name]["invocations"].append(inv)
                continue
            entry = data.copy()
            entry["name"] = phase_name
            entry["invocations"] = [inv]
            entry.pop("invocation_signatures", None)
            entry["phase"] = label
            split[phase_name] = entry

    capture.module_data = split


def _snapshot_hf_attention_classes(model) -> Dict[str, Tuple[str, Any, int]]:
    """Record each decoder layer's HF attention class BEFORE prepare_for_spyre().

    ``StandardGQAAttention.__init__(attn)`` adopts an HF attention's projections
    and keeps none of its provenance -- no ``config``, no ``layer_idx``. Once
    ``prepare_standard_gqa_blocks`` has replaced ``layers[i]``, the original class
    is unrecoverable from the live tree, so the Spyre loader path must snapshot it
    while the plain HF model is still intact.

    Returns ``{layer_path: (attn_class_path, config, layer_idx)}`` keyed by the
    decoder layer's ``named_modules`` path, which survives the replacement (only
    the object at that path changes).
    """
    from hf_adapters.hf_common import get_backbone

    snapshot: Dict[str, Tuple[str, Any, int]] = {}
    backbone = get_backbone(model)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        logger.warning(
            "Model has no backbone .layers; no HF attention classes recorded. "
            "Spyre attention entries will not be rebuildable."
        )
        return snapshot

    # Resolve the backbone's own path so keys match named_modules() on `model`.
    backbone_path = ""
    for name, mod in model.named_modules():
        if mod is backbone:
            backbone_path = name
            break

    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        cls = type(attn)
        config = getattr(attn, "config", None)
        layer_idx = getattr(attn, "layer_idx", None)
        prefix = f"{backbone_path}.layers.{i}" if backbone_path else f"layers.{i}"
        snapshot[prefix] = (
            f"{cls.__module__}.{cls.__name__}",
            config,
            i if layer_idx is None else layer_idx,
        )
    return snapshot


def _extract_tensor_info(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Extract information from a single tensor."""
    return {
        "type": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "is_random": not _is_special_tensor(name),
        "requires_grad": tensor.requires_grad,
    }


# Scalar types recorded verbatim as a forward arg. A module's forward may take plain
# Python scalars alongside tensors (a flag, an index, a length). Recording them is
# not optional for POSITIONAL args: dropping one shifts every later arg a slot left,
# so the replayed forward is called with the wrong arity.
# ``bool`` precedes ``int`` only for clarity; ``isinstance`` covers both since
# bool subclasses int.
_SCALAR_ARG_TYPES = (bool, int, float, str)


def _extract_scalar_info(value: Any, name: str) -> Dict[str, Any] | None:
    """Describe a non-tensor scalar forward arg, or ``None`` if not a scalar.

    Tensor-free args carry no pytree leaves, so :func:`_process_pytree_structure`
    returns ``None`` for them. Dropping a *positional* arg is not merely a missing
    input: it shifts every later arg one slot left, so the replayed forward is
    called with the wrong arity and the wrong values. Recording the scalar keeps
    the positional sequence intact.

    ``None`` is recorded too (a genuine "pass None here" argument). It is checked
    before the scalar types because ``None`` is not an instance of any of them.
    """
    if value is None:
        return {"name": name, "type": "value", "value": None}
    if isinstance(value, _SCALAR_ARG_TYPES):
        # bool before int in the emitted type name for readability only; the YAML
        # carries the value itself, which round-trips with its Python type.
        return {"name": name, "type": "value", "value": value}
    return None


def _process_pytree_structure(value: Any, name: str) -> Dict[str, Any] | None:
    """
    Process a pytree structure (nested tensors/lists/tuples/dicts) and extract info.

    Uses PyTorch's tree_flatten to handle arbitrary nesting uniformly.
    """
    # Check if this is a tensor or contains tensors
    if isinstance(value, torch.Tensor):
        # Single tensor - simple case
        return {"name": name, **_extract_tensor_info(value, name)}

    # Use tree_flatten to extract all tensor leaves regardless of nesting.
    # We intentionally do not reconstruct the original structure since only
    # tensor metadata is needed for config generation.
    flat_values, _ = tree_flatten(value)

    # Extract info from all tensors in the flattened structure
    # Single source of truth: pytree handles all container types uniformly
    tensor_infos = []
    for item in flat_values:
        if isinstance(item, torch.Tensor):
            tensor_infos.append(_extract_tensor_info(item, name))

    # Post-process: enrich dict tensors with their keys
    if isinstance(value, dict) and tensor_infos:
        dict_keys = [k for k, v in value.items() if isinstance(v, torch.Tensor)]
        for i, key in enumerate(dict_keys):
            if i < len(tensor_infos):
                tensor_infos[i]["dict_key"] = key

    # If we found tensors, return with structure info
    if tensor_infos:
        # Determine container type from the original value
        if isinstance(value, tuple):
            container_type = "tuple"
        elif isinstance(value, list):
            container_type = "list"
        elif isinstance(value, dict):
            container_type = "dict"
        else:
            container_type = "pytree"

        return {
            "name": name,
            "type": container_type,
            "items": tensor_infos,
        }

    return None


def _resolve_layer_idx(module: Any) -> int | None:
    """Find a module's decoder layer index for indexing into the KV cache.

    In transformers >=5 the ``layer_idx`` lives on the attention submodule, not
    on the DecoderLayer itself, so we check the layer first (older layouts /
    other archs) and fall back to ``self_attn.layer_idx``. Returns ``None`` when
    neither exists (e.g. a norm/MLP module that never touches the KV cache).
    """
    layer_idx = getattr(module, "layer_idx", None)
    if layer_idx is not None:
        return layer_idx
    self_attn = getattr(module, "self_attn", None)
    if self_attn is not None:
        return getattr(self_attn, "layer_idx", None)
    return None


# Decoder-stack path segments across HF architectures: ``model.layers.N``
# (Llama/Granite/Qwen/Mistral), ``transformer.h.N`` (GPT-2), ``blocks.N``,
# ``encoder.layer.N`` (BERT). The captured group is the layer index.
_LAYER_PATH_RE = re.compile(
    r"(?:^|\.)(?:layers|h|block|blocks|encoder\.layer)\.(\d+)(?:\.|$)"
)


def _layer_index_from_path(module_name: str) -> Optional[int]:
    """Return the decoder-layer index a module lives under, or ``None`` if outside one.

    Derived from the ``named_modules()`` path rather than a ``layer_idx``
    attribute, because only the attention module carries that attribute --
    verified on Granite 3.3, where ``mlp`` / ``input_layernorm`` / the
    ``DecoderLayer`` itself have none. An attribute-based approach therefore
    cannot attribute most modules to a layer.

    Returning ``None`` for a module outside any layer is load-bearing: an RMSNorm
    exists both inside a layer and as the backbone's final ``model.norm``, and only
    the former should be named per layer. Note the converse also occurs -- a
    ``RotaryEmbedding`` sits outside the stack in Granite (``model.rotary_emb``) but
    inside it in others (RecurrentGemma's ``layers.N.temporal_block.rotary_emb``),
    so no module type may be assumed to be on one side or the other.
    """
    match = _LAYER_PATH_RE.search(module_name)
    return None if match is None else int(match.group(1))


def _compose_module_name(base: str, *suffixes: Optional[str]) -> str:
    """Join a module base name with the suffixes that distinguish this entry.

    Suffixes are appended in a fixed order so a name stays stable as new axes are
    added (layer today; execution phase is appended later by
    :func:`split_module_data_by_phase`). ``None`` suffixes are dropped, so a module
    outside any decoder layer keeps exactly the name it has today.
    """
    return "_".join([base, *(s for s in suffixes if s)])


def _class_source_location(cls: type) -> Tuple[Optional[str], Optional[int]]:
    """Return (source file, first line of the class definition) for ``cls``.

    Resolved from the live class object while the hook still holds the module
    instance. ``module_path`` alone is not enough: a model loaded with
    ``trust_remote_code`` lives in a dynamically created module that cannot be
    re-imported by name later. Returns ``(None, None)`` when no source is
    retrievable (C extension, class synthesized at runtime).
    """
    try:
        source_file = inspect.getsourcefile(cls)
        _, lineno = inspect.getsourcelines(cls)
    except (OSError, TypeError):
        return None, None
    return source_file, lineno


def _shorten_source_path(path: str) -> str:
    """Trim an absolute source path down to something environment-independent.

    A ``site-packages``/``dist-packages`` install becomes ``<pkg>/...`` so the
    generated YAML does not hard-code the generating machine's venv layout.
    Paths outside a site install are returned unchanged.
    """
    parts = Path(path).parts
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            return str(Path(*parts[parts.index(marker) + 1 :]))
    return path


def _get_transformers_ref() -> str:
    """Git ref used in generated transformers source URLs.

    Mirrors ``utils/model_ops/utils/torchop_yaml.py``: ``TRANSFORMERS_VERSION``
    overrides, otherwise the installed version becomes a ``vX.Y.Z`` release tag.
    A dev/editable install ("5.0.0.dev0") has no such tag, so it falls back to
    ``main`` rather than emitting a dead link.
    """
    version = os.getenv("TRANSFORMERS_VERSION")
    if version:
        return version
    try:
        import transformers
    except ImportError:
        return "main"
    return (
        "main" if "dev" in transformers.__version__ else f"v{transformers.__version__}"
    )


_TRANSFORMERS_BLOB_URL = "https://github.com/huggingface/transformers/blob"


def _source_reference(
    source_file: Optional[str], lineno: Optional[int]
) -> Optional[str]:
    """Render a captured source location as a human-followable reference.

    A file inside an installed ``transformers`` package becomes a GitHub blob
    URL pinned to the installed version, matching the scheme
    ``torchop_yaml._convert_transformers_path_to_url`` uses. Any other package
    (torch, vLLM, a trust_remote_code module) degrades to a venv-relative
    ``path:line``, since there is no single upstream repo to point at.
    """
    if not source_file:
        return None
    rel = _shorten_source_path(source_file)
    # rel differing from the input means the file came from a site install, so
    # a leading "transformers/" component is the installed transformers package
    # (and maps onto src/transformers/... in the upstream repo layout).
    if rel != source_file and rel.startswith("transformers/"):
        anchor = f"#L{lineno}" if lineno else ""
        return f"{_TRANSFORMERS_BLOB_URL}/{_get_transformers_ref()}/src/{rel}{anchor}"
    return f"{rel}:{lineno}" if lineno else rel


def _cache_max_length(past_key_values: Any) -> int | None:
    """Return a ``Cache``'s fixed allocation length, or ``None`` if it has none.

    transformers 5.15 deprecated the ``Cache.max_cache_len`` property in favour
    of ``get_max_length()`` (removal in 5.16), and merely reading the old
    property emits a warning. Prefer the new API and fall back to the property
    for older transformers, which lack ``get_max_length``.
    """
    get_max_length = getattr(past_key_values, "get_max_length", None)
    if callable(get_max_length):
        return get_max_length()
    return getattr(past_key_values, "max_cache_len", None)


def _extract_cache_info(
    past_key_values: Any, name: str, layer_idx: int, config: Any = None
) -> Dict[str, Any] | None:
    """Snapshot one layer's populated K/V from a ``Cache`` for a decode step.

    A DecoderLayer receives ``past_key_values`` as a live
    :class:`~transformers.cache_utils.Cache`, not raw tensors. During prefill
    the layer's slot is empty (``keys is None``), so there is nothing to record
    and this returns ``None`` — the layer then runs its
    ``if past_key_values is not None: past_key_values.update(...)`` branch on
    freshly computed K/V only, which is the correct prefill behaviour. During
    decode the slot already holds ``past_len`` tokens; we snapshot that layer's
    ``keys``/``values`` so the module test can rebuild an equivalent Cache and
    exercise the same "attend over past + new token" path. Without this, the
    decode invocation would replay with ``past_key_values=None`` and silently
    degrade to a 1-token self-attention (the ``update`` branch never runs).

    Transformers >=5 stores per-layer K/V under ``.layers[i].keys/.values``
    (the older flat ``key_cache``/``value_cache`` lists are gone).

    Only :class:`~transformers.cache_utils.StaticCache` is recorded. A
    fixed-size StaticCache has a fully specified per-layer K/V shape that the
    test side can reconstruct deterministically; a growable ``DynamicCache``
    (the default when no cache is passed in) has no such fixed shape, so we warn
    and skip it rather than emit a cache the test cannot faithfully rebuild.
    Drive the generator with an explicit StaticCache to capture decode state.

    Args:
        past_key_values: The live cache passed to the DecoderLayer.
        name: The kwarg name (``"past_key_values"``).
        layer_idx: The layer whose K/V slot to snapshot.
        config: The model config the cache was built from. In transformers >=5
            a StaticCache no longer exposes ``.config``, but its ``__init__``
            requires one, so the test side needs ``config_path`` +
            ``config_kwargs`` to reconstruct it. Pass the DecoderLayer's config
            (e.g. ``module.self_attn.config``).

    Returns:
        A cache spec dict, or ``None`` when the cache is not a StaticCache, this
        layer's slot is empty (prefill), or it exposes no usable per-layer K/V.
    """
    cache_cls = type(past_key_values)
    if cache_cls.__name__ != "StaticCache":
        logger.warning(
            "past_key_values is a %s, not StaticCache; skipping cache capture. "
            "Drive the generator with an explicit StaticCache to record the "
            "decode KV state (see generate_gpt_oss_20b_config.py).",
            cache_cls.__name__,
        )
        return None

    layers = getattr(past_key_values, "layers", None)
    if layers is None or layer_idx >= len(layers):
        return None

    layer_cache = layers[layer_idx]
    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)

    # Empty slot -> prefill call; nothing populated to record.
    if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
        return None

    # StaticCache allocates the full max_cache_len up front, so keys/values are
    # [B, num_kv_heads, max_cache_len, head_dim] with only the first past_len
    # positions populated. Record just that populated slice so the shape means
    # "the real past" and the test side can prime a cache by a single update()
    # of past_len tokens (not the whole fixed allocation of mostly-zeros).
    #
    # Use the PER-LAYER length, not past_key_values.get_seq_length(): at a
    # decode step the whole-cache length reflects layers already updated this
    # pass, so for layer i>0 (whose slot hasn't been updated yet at pre-hook
    # time) it reads one token too long. layer_cache.get_seq_length() reports
    # just this layer's populated past, which is the same across layers.
    try:
        past_len = int(layer_cache.get_seq_length())
    except Exception:
        try:
            past_len = int(past_key_values.get_seq_length())
        except Exception:
            past_len = keys.shape[-2]
    if past_len <= 0:
        return None
    keys = keys[:, :, :past_len, :]
    values = values[:, :, :past_len, :]

    cache_info: Dict[str, Any] = {
        "name": name,
        "type": "cache",
        "cache_path": f"{cache_cls.__module__}.{cache_cls.__name__}",
        "layer_idx": layer_idx,
        # StaticCache.__init__ needs max_cache_len (the fixed allocation), which
        # is not derivable from the (sliced) K/V shape. Record it so the test
        # side rebuilds a cache of the same allocation before priming it.
        "max_cache_len": _cache_max_length(past_key_values),
        # keys/values carry real past tokens; the test rebuilds a cache of the
        # same seq length via update(). "key"/"value" are not special tensor
        # names, so they default to random init (see _is_special_tensor).
        "key": _extract_tensor_info(keys, f"{name}_key"),
        "value": _extract_tensor_info(values, f"{name}_value"),
    }

    # Snapshot the config so the test side can construct the concrete Cache with
    # matching dimensions (num_kv_heads, head_dim, ...). transformers >=5 no
    # longer exposes StaticCache.config, so we take the config passed in from
    # the DecoderLayer; fall back to the cache's own attribute for older builds.
    if config is None:
        config = getattr(past_key_values, "config", None)
    if config is not None:
        config_cls = type(config)
        # Read the dimensions off this layer's config: a heterogeneous config
        # raises AmbiguousGlobalPerLayerAttributeError for per-layer attributes
        # such as num_key_value_heads, and that exception is a RuntimeError, so
        # even hasattr() below would propagate it (see _layer_config).
        dim_config = _layer_config(config, layer_idx)
        config_kwargs = {}
        for attr in [
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "max_position_embeddings",
        ]:
            value = _read_config_attr(dim_config, attr)
            if value is not _CONFIG_ATTR_MISSING:
                config_kwargs[attr] = value
        # num_hidden_layers must cover layer_idx: the framework primes the cache
        # with cache.update(key, value, layer_idx), so a cache sized to 1 layer
        # raises IndexError for any layer_idx > 0 (e.g. --capture_layers 0,5).
        # Size it to layer_idx + 1 -- the layers below the captured one stay empty
        # and are never read, so this costs allocation only, not correctness.
        config_kwargs["num_hidden_layers"] = layer_idx + 1
        cache_info["config_path"] = f"{config_cls.__module__}.{config_cls.__name__}"
        cache_info["config_kwargs"] = config_kwargs

    return cache_info


# Sentinel for "record every layer" -- distinct from ``None``, which selects the
# default of layer 0 only. An empty frozenset reads as "no filtering".
CAPTURE_ALL_LAYERS: frozenset = frozenset()


class ModuleInfoCapture:
    """Captures module information during forward pass using hooks."""

    def __init__(
        self,
        spyre_attn_snapshot=None,
        spyre_model=None,
        capture_layers: Optional[Set[int]] = None,
    ):
        self.module_data: Dict[str, Dict[str, Any]] = {}
        self.seen_module_configs: Set[str] = (
            set()
        )  # Track unique configs, not just types
        # Track model-level context (KV cache, execution mode)
        self.current_model_context: Dict[str, Any] = {}
        # Set only on the --loader spyre path: maps a decoder layer path to the HF
        # attention class/config it held before prepare_for_spyre() replaced it
        # (see _snapshot_hf_attention_classes). Left empty on the HF path, which
        # keeps capture_constructor_info's behaviour bit-identical there.
        self.spyre_attn_snapshot = spyre_attn_snapshot or {}
        # The prepared model, read for _spyre_head_dim (the padded head_dim the
        # adapter actually runs). None on the HF path.
        self.spyre_model = spyre_model
        # Layers to record, by index. ``None`` -> {0}: a decoder stack repeats the
        # same module shapes per layer, so recording all 40 of an 8B model would
        # multiply the YAML -- and, on Spyre, the number of compiled binaries in
        # the module test -- for little extra coverage. Pass CAPTURE_ALL_LAYERS to
        # record every layer.
        self.capture_layers: Set[int] = (
            {0} if capture_layers is None else set(capture_layers)
        )

    def _resolve_spyre_attn_source(self, module_name: str):
        """Look up the pre-Spyre HF attention for a wrapper's ``named_modules`` path.

        ``module_name`` is the wrapper's path (e.g. ``model.layers.7.self_attn``)
        while the snapshot is keyed by the decoder layer's path
        (``model.layers.7``), so match on the longest recorded prefix rather than
        assuming a fixed suffix -- the attribute the wrapper sits under differs
        across adapters.
        """
        best = None
        for layer_path, entry in self.spyre_attn_snapshot.items():
            if module_name == layer_path or module_name.startswith(layer_path + "."):
                if best is None or len(layer_path) > len(best[0]):
                    best = (layer_path, entry)
        return None if best is None else best[1]

    def capture_constructor_info(
        self,
        module,
        module_name: str,
        module_type: str,
        layer_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Capture constructor information from an instantiated module.

        ``layer_index`` selects which layer's dimensions to read on a heterogeneous
        config (see :func:`_layer_config`); it is optional so out-of-layer modules
        and homogeneous models behave exactly as before.

        This inspects the module to infer what constructor args were used.
        For Transformers modules, we look for config objects and layer_idx.
        """
        constructor_args = []
        constructor_kwargs = {}

        # A Spyre attention wrapper adopts an HF attention's projections and keeps
        # no config/layer_idx of its own, so none of the heuristics below can
        # describe it (they would fall through to the 1-D-weight branch and emit a
        # wrong ctor). Emit a *module* arg instead: the test side rebuilds the
        # inner HF attention from its config, then wraps it -- exactly what
        # prepare_for_spyre did. Needs the OOT framework's InputArgModule.
        if module_type == SPYRE_ATTENTION_TYPE:
            entry = self._resolve_spyre_attn_source(module_name)
            if entry is not None:
                attn_path, attn_config, layer_idx = entry
                constructor_args.append(
                    {
                        "type": "module",
                        "module_path": attn_path,
                        "config_path": (
                            f"{type(attn_config).__module__}."
                            f"{type(attn_config).__name__}"
                        ),
                        # head_dim reflects _spyre_head_dim when the adapter padded
                        # it, so the rebuilt projections get the padded widths the
                        # adapter actually runs.
                        "config_kwargs": _spyre_config_kwargs(
                            attn_config, self.spyre_model
                        ),
                        "module_kwargs": {"layer_idx": layer_idx},
                    }
                )
                return {
                    "constructor_args": constructor_args,
                    "constructor_kwargs": constructor_kwargs,
                }
            logger.warning(
                "%s at %s: no pre-Spyre HF attention recorded; emitting no "
                "constructor args (this entry will not be rebuildable).",
                module_type,
                module_name,
            )

        # Special handling for decoder layers that don't expose config attribute
        # but require it as constructor arg (e.g., GraniteDecoderLayer)
        if "decoder" in module_type.lower() and "layer" in module_type.lower():
            # Try to get config from parent model or infer from module structure
            # For now, we'll look for self_attn or mlp submodules that might have config
            if hasattr(module, "self_attn") and hasattr(module.self_attn, "config"):
                config = module.self_attn.config
            elif hasattr(module, "mlp") and hasattr(module.mlp, "config"):
                config = module.mlp.config
            else:
                config = None

            if config is not None:
                config_class = type(config).__name__
                config_module = type(config).__module__

                # Extract key config parameters
                config_kwargs = _extract_config_kwargs(config, layer_index)

                constructor_args.append(
                    {
                        "type": "config",
                        "config_path": f"{config_module}.{config_class}",
                        "config_kwargs": config_kwargs,
                    }
                )

                # Decoder layers typically need layer_idx as kwarg
                # Always add it for decoder layers, even if not found as attribute.
                #
                # A DecoderLayer carries no layer_idx attribute of its own (only
                # its attention submodule does), so fall back to the index in the
                # module path before defaulting to 0. Without the path fallback,
                # every captured layer would be rebuilt as layer 0 -- silently
                # wrong once more than one layer is captured.
                layer_idx_value = 0  # Default to 0
                if hasattr(module, "layer_idx") and module.layer_idx is not None:
                    layer_idx_value = module.layer_idx
                else:
                    path_layer_idx = _layer_index_from_path(module_name)
                    if path_layer_idx is not None:
                        layer_idx_value = path_layer_idx
                constructor_kwargs["layer_idx"] = {
                    "type": "int",
                    "value": layer_idx_value,
                }
        # Check if module has a config attribute (common in Transformers)
        elif hasattr(module, "config"):
            config = module.config
            config_class = type(config).__name__
            config_module = type(config).__module__

            # Extract key config parameters
            config_kwargs = _extract_config_kwargs(config, layer_index)

            constructor_args.append(
                {
                    "type": "config",
                    "config_path": f"{config_module}.{config_class}",
                    "config_kwargs": config_kwargs,
                }
            )

            # Check for layer_idx (common in decoder layers with config)
            # Note: layer_idx can be 0, so check for attribute existence, not truthiness
            if hasattr(module, "layer_idx"):
                layer_idx_value = module.layer_idx
                if layer_idx_value is None:
                    # Present but unset: prefer the index in the module path over a
                    # blanket 0, so a captured layer is rebuilt as itself.
                    path_layer_idx = _layer_index_from_path(module_name)
                    layer_idx_value = 0 if path_layer_idx is None else path_layer_idx
                constructor_kwargs["layer_idx"] = {
                    "type": "int",
                    "value": layer_idx_value,
                }
        else:
            # No config - check for direct constructor parameters
            # RMSNorm: hidden_size or dim
            if hasattr(module, "weight") and hasattr(module.weight, "shape"):
                # Normalization layers typically have weight with shape (hidden_size,)
                if len(module.weight.shape) == 1:
                    hidden_size = module.weight.shape[0]
                    constructor_args.append({"type": "int", "value": hidden_size})
            elif hasattr(module, "normalized_shape"):
                # LayerNorm-style
                if isinstance(module.normalized_shape, tuple):
                    hidden_size = module.normalized_shape[0]
                else:
                    hidden_size = module.normalized_shape
                constructor_args.append({"type": "int", "value": hidden_size})

        return {
            "constructor_args": constructor_args,
            "constructor_kwargs": constructor_kwargs,
        }

    def create_model_hook(self):
        """Create a model-level hook to detect execution mode (prefill vs decode)..

        This hook runs BEFORE module-level hooks and sets context that module hooks can use.
        """

        def model_hook(model, args, kwargs):
            # Capture model-level context
            past_key_values = kwargs.get("past_key_values", None)
            attention_mask = kwargs.get("attention_mask", None)

            # Detect execution mode. A pre-allocated but empty cache (e.g. a
            # freshly constructed StaticCache passed into prefill) is not None,
            # so fall back to its sequence length to distinguish prefill from
            # decode.
            if past_key_values is None:
                mode = "prefill"
            elif (
                hasattr(past_key_values, "get_seq_length")
                and past_key_values.get_seq_length() == 0
            ):
                mode = "prefill"
            else:
                mode = "decode"

            # Store context for module hooks to access
            self.current_model_context = {
                "mode": mode,
                "attention_mask": attention_mask,
            }

        return model_hook

    def create_hook(self, module_name: str, module_type: str, module_instance):
        """Create a forward hook that captures module input information.

        This hook captures unique invocations of the module, deduplicating by input pattern.
        This allows testing with multiple input configurations (e.g., prefill + decode)
        without storing redundant identical invocations.
        """

        def hook(module, args, kwargs):
            # Attribute this module to a decoder layer, and skip layers the caller
            # did not ask for. A module outside any layer (model.norm, lm_head,
            # rotary_emb on most architectures) is always recorded: it exists
            # once, not once per layer.
            layer_index = _layer_index_from_path(module_name)
            if (
                layer_index is not None
                and self.capture_layers
                and layer_index not in self.capture_layers
            ):
                return

            # Capture constructor information to create unique config identifier
            constructor_info = self.capture_constructor_info(
                module, module_name, module_type, layer_index
            )

            # Create a unique identifier based on module type + constructor args
            # This allows us to capture multiple variants of the same module type
            config_signature = self._create_config_signature(
                module_type, constructor_info, layer_index
            )

            # Create unique module name for this variant
            unique_module_name = self._create_unique_module_name(
                module_type, constructor_info, config_signature, layer_index
            )

            # Initialize module_info if this is the first invocation
            if unique_module_name not in self.module_data:
                self.seen_module_configs.add(config_signature)

                cls = module.__class__
                source_file, source_lineno = _class_source_location(cls)
                self.module_data[unique_module_name] = {
                    "name": unique_module_name,
                    "module_type": module_type,
                    "module_path": f"{cls.__module__}.{cls.__name__}",
                    "source_file": source_file,
                    "source_lineno": source_lineno,
                    "example_instance": module_name,
                    "layer_index": layer_index,
                    "constructor_args": constructor_info["constructor_args"],
                    "constructor_kwargs": constructor_info["constructor_kwargs"],
                    "invocations": [],  # List of unique invocations
                    "invocation_signatures": set(),  # Track seen invocation patterns
                }

            # Capture this invocation's inputs
            invocation_inputs = []

            # Analyze positional arguments using pytree.
            #
            # A positional arg MUST always yield an entry: these are replayed in
            # order, so silently skipping one (a tensor-free scalar, for which
            # _process_pytree_structure returns None) shifts every later arg into
            # the wrong slot and the rebuilt forward is called with the wrong
            # arity. Scalars are recorded as plain values; anything genuinely
            # undescribable is logged rather than dropped in silence.
            for i, arg in enumerate(args):
                input_info = _process_pytree_structure(arg, f"arg_{i}")
                if input_info is None:
                    input_info = _extract_scalar_info(arg, f"arg_{i}")
                if input_info is None:
                    logger.warning(
                        "%s: positional arg_%d of type %s is neither tensor-bearing "
                        "nor a scalar; the generated entry will be missing it and "
                        "every later positional arg will shift one slot left.",
                        module_type,
                        i,
                        type(arg).__name__,
                    )
                    continue
                invocation_inputs.append(input_info)

            # Analyze keyword arguments using pytree
            for key, value in kwargs.items():
                if key in ("past_key_values", "past_key_value"):
                    # A live Cache object can't go through the tensor-spec
                    # pytree path. For a decode step we snapshot this layer's
                    # populated K/V so the module test can rebuild an equivalent
                    # cache and drive the real "attend over past + new token"
                    # path; for prefill the slot is empty and this records
                    # nothing (equivalent to past_key_values=None).
                    layer_idx = _resolve_layer_idx(module)
                    if layer_idx is not None and value is not None:
                        layer_config = getattr(
                            getattr(module, "self_attn", None), "config", None
                        ) or getattr(module, "config", None)
                        cache_info = _extract_cache_info(
                            value, "past_key_values", layer_idx, config=layer_config
                        )
                        if cache_info is not None:
                            invocation_inputs.append(cache_info)
                    continue
                input_info = _process_pytree_structure(value, key)
                if input_info is None:
                    # Scalar kwargs matter for the same reason as scalar
                    # positionals (a forward may branch on them), though omitting
                    # one here only loses that kwarg -- it cannot misalign others.
                    input_info = _extract_scalar_info(value, key)
                if input_info:
                    invocation_inputs.append(input_info)

            # Create signature for this invocation to detect duplicates
            invocation_sig = self._create_invocation_signature(invocation_inputs)

            # Only add if this is a new unique invocation pattern
            if (
                invocation_sig
                not in self.module_data[unique_module_name]["invocation_signatures"]
            ):
                self.module_data[unique_module_name]["invocation_signatures"].add(
                    invocation_sig
                )
                self.module_data[unique_module_name]["invocations"].append(
                    invocation_inputs
                )

        return hook

    def _create_config_signature(
        self,
        module_type: str,
        constructor_info: Dict[str, Any],
        layer_index: Optional[int] = None,
    ) -> str:
        """Create a unique signature for a module configuration.

        This signature is used to detect duplicate configurations.

        The layer index IS part of the signature: how many layers get recorded is
        bounded by ``capture_layers``, so distinct layers must not collapse here.
        ``layer_index=None`` (a module outside any layer) reproduces the previous
        signature exactly, keeping those entries unchanged.
        """
        # Build signature from constructor args
        sig_parts = [module_type]

        for arg in constructor_info.get("constructor_args", []):
            if arg["type"] == "int":
                sig_parts.append(f"int_{arg['value']}")
            elif arg["type"] == "config":
                sig_parts.append(f"config_{arg['config_path']}")
            else:
                sig_parts.append(f"{arg['type']}")

        # layer_idx is skipped here because the layer is represented by
        # layer_index below, taken from the module path -- which covers the
        # modules that have no layer_idx attribute (mlp, norms, the layer itself).
        for key, kwarg in constructor_info.get("constructor_kwargs", {}).items():
            if key == "layer_idx":
                continue
            if kwarg["type"] == "int":
                sig_parts.append(f"{key}_{kwarg['value']}")

        if layer_index is not None:
            sig_parts.append(f"layer_{layer_index}")

        return "__".join(sig_parts)

    def _create_unique_module_name(
        self,
        module_type: str,
        constructor_info: Dict[str, Any],
        config_signature: str,
        layer_index: Optional[int] = None,
    ) -> str:
        """Create a unique, human-readable name for a module variant.

        A module inside a decoder layer is suffixed with its layer index, so each
        captured layer gets its own entry. A module outside any layer keeps the
        name it has today.

        The layer suffix replaces the old hash fallback for in-layer modules. That
        hash carried almost no information -- ``_create_config_signature`` records
        only the config CLASS path, not its dimensions, so ``SiLUActivation``'s
        hash was ``sha256("SiLUActivation")[:8]``, a constant -- and ``_layer0``
        is both meaningful and readable. The hash remains for out-of-layer modules
        that still need disambiguating.

        Examples:
            MyRMSNorm with dim=4096 -> MyRMSNorm_4096
            MyRMSNorm with dim=4096, inside layer 3 -> MyRMSNorm_4096_layer3
            GraniteMLP inside layer 0 -> GraniteMLP_layer0
            GraniteDecoderLayer inside layer 3 -> GraniteDecoderLayer_layer3
        """
        layer_suffix = None if layer_index is None else f"layer{layer_index}"

        # Check if there's a simple int arg (common for norm layers)
        args = constructor_info.get("constructor_args", [])
        if len(args) == 1 and args[0]["type"] == "int":
            return _compose_module_name(
                module_type, str(args[0]["value"]), layer_suffix
            )

        # Inside a layer the index alone disambiguates; no hash needed.
        if layer_suffix is not None:
            return _compose_module_name(module_type, layer_suffix)

        # Outside any layer, fall back to a hash of the config signature.
        sig_hash = hashlib.sha256(config_signature.encode()).hexdigest()[:8]
        return _compose_module_name(module_type, sig_hash)

    def _create_invocation_signature(
        self, invocation_inputs: List[Dict[str, Any]]
    ) -> str:
        """Create a signature for an invocation based on input patterns.

        This signature captures the structure of inputs (shapes, dtypes, types)
        but not the actual values, allowing us to deduplicate identical invocations.

        Args:
            invocation_inputs: List of input info dicts from _process_pytree_structure

        Returns:
            A string signature representing this invocation pattern
        """

        def _extract_pattern(input_info: Dict[str, Any]) -> Dict[str, Any]:
            """Extract the pattern from an input, removing variable data.

            input_info structure from _process_pytree_structure:
            - Single tensor: {"name": "arg_0", "shape": [...], "dtype": ..., ...}
            - Container: {"name": "arg_0", "type": "list/tuple/dict/pytree", "items": [...]}
            """
            # A scalar: include the VALUE, not just the type. Two invocations can
            # differ only in a scalar while their tensor shapes are identical, so
            # ignoring the value would collapse them into one and silently drop a
            # pattern the module is really called with.
            if input_info.get("type") == "value":
                return {"type": "value", "value": input_info.get("value")}
            # A KV cache: distinct pattern so prefill (no cache) and decode
            # (cache present) never collapse into one invocation signature.
            if input_info.get("type") == "cache":
                return {
                    "type": "cache",
                    "cache_path": input_info.get("cache_path"),
                    "key_shape": input_info.get("key", {}).get("shape"),
                    "value_shape": input_info.get("value", {}).get("shape"),
                }
            # Check if this is a container with items
            if "type" in input_info and "items" in input_info:
                # Container (list, tuple, dict, pytree)
                pattern = {
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
                return pattern
            elif "shape" in input_info:
                # Single tensor
                return {
                    "type": "tensor",
                    "shape": input_info.get("shape"),
                    "dtype": str(input_info.get("dtype")),
                    "init": input_info.get("init"),
                }
            else:
                # Unknown structure
                return {"type": "unknown"}

        # Build pattern for all inputs
        patterns = []
        for input_info in invocation_inputs:
            # input_info is already a dict with structure like:
            # {"name": "arg_0", "tensor": {...}} or {"name": "x", "type": "list", "items": [...]}
            # We want to extract the pattern from the whole input_info
            patterns.append(_extract_pattern(input_info))

        # Convert to JSON for consistent string representation
        pattern_str = json.dumps(patterns, sort_keys=True)
        return hashlib.sha256(pattern_str.encode()).hexdigest()

    def get_captured_modules(self) -> List[Dict[str, Any]]:
        """Return list of captured module information."""
        # Remove invocation_signatures before returning (internal tracking only)
        result = []
        for module_data in self.module_data.values():
            module_copy = module_data.copy()
            module_copy.pop("invocation_signatures", None)
            result.append(module_copy)
        return result


def get_all_custom_modules(
    model,
    excluded_types: frozenset = frozenset(),
    capture_layers: Optional[Set[int]] = None,
) -> List[Tuple[str, str, Any]]:
    """
    Get ALL custom module instances from the model (not just unique types).

    Args:
        model: The model to walk.
        excluded_types: Class names to skip entirely. Used by the Spyre loader
            path to drop wrapper modules not worth emitting as test entries (see
            SPYRE_EXCLUDED_MODULE_TYPES). Empty by default, so the HF path is
            unchanged. Excluding a container does NOT exclude its children: they
            appear in named_modules() in their own right and are still captured.
        capture_layers: When given, in-layer modules outside these indices are
            skipped, so hooks are registered on one layer's worth of modules
            instead of all 40. Modules outside any layer are always kept. The hook
            itself re-checks, so this is purely an optimization.

    Returns:
        List of (module_name, module_type, module_instance) tuples
    """
    custom_modules = []
    for name, module in model.named_modules():
        if name == "":  # Skip root
            continue

        module_type = type(module).__name__

        if module_type in excluded_types:
            continue

        # Skip if already in upstream module_db
        if module_type in existing_modules:
            continue

        if capture_layers:
            layer_index = _layer_index_from_path(name)
            if layer_index is not None and layer_index not in capture_layers:
                continue

        # Keep ALL instances (not just first of each type)
        custom_modules.append((name, module_type, module))

    return custom_modules


def _convert_constructor_arg_to_sample_input(
    arg_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert constructor arg spec to sample_inputs_func format."""
    if arg_spec["type"] == "config":
        # Emit a structured config arg carrying the captured model dimensions so
        # the framework can rebuild the config (config_path + config_kwargs) with
        # the right shapes instead of library defaults. Resolved by
        # InputsEdits.build_cpu_args -> InputArgConfig in the OOT framework.
        return {
            "config_path": arg_spec["config_path"],
            "config_kwargs": arg_spec.get("config_kwargs", {}),
        }
    elif arg_spec["type"] == "module":
        # Resolved by the OOT framework's InputArgModule -> _build_inner_module:
        # import module_path, build the config, instantiate it, then hand the live
        # module to the outer wrapper's constructor.
        return {
            "module_path": arg_spec["module_path"],
            "config_path": arg_spec["config_path"],
            "config_kwargs": arg_spec.get("config_kwargs", {}),
            "module_kwargs": arg_spec.get("module_kwargs", {}),
        }
    elif arg_spec["type"] == "int":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "float":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "str":
        return {"value": arg_spec["value"]}
    elif arg_spec["type"] == "bool":
        return {"value": arg_spec["value"]}
    else:
        return {"value": None}


def _tensor_info_to_spec(tensor_info: Dict[str, Any], name: str) -> Dict[str, Any]:
    """
    Convert a single tensor info dict to sample_inputs tensor spec format.

    This function can be used with tree_map to transform entire structures.
    """
    dtype = tensor_info["dtype"]
    if not dtype.startswith("torch."):
        dtype = f"torch.{dtype}"

    # Default every floating-point tensor to bfloat16 (the dtype Spyre runs in),
    # regardless of the precision the checkpoint was captured in. A model loaded
    # in float32 would otherwise emit float32 specs; normalizing here guarantees
    # the "default is bfloat16" contract even when the capture path did not (or
    # could not) load the model in bfloat16. Integer/bool tensors are left alone.
    bare_dtype = dtype.replace("torch.", "")
    if bare_dtype in _FLOAT_DTYPE_ALIASES:
        dtype = str(DEFAULT_FLOAT_DTYPE)

    # Determine init strategy based on tensor characteristics
    is_random = tensor_info.get("is_random", True)
    init = "randn" if is_random else "zeros"
    init_args = {}

    # An integer tensor (e.g. token ids for an embedding, position ids, masks)
    # must not use randn -- torch.randn ("normal_kernel_cpu") is float-only and
    # raises NotImplementedError for integer dtypes. Use randint for any integer
    # dtype, and also for the name-based special tensors (position/mask/ids),
    # which may be captured under a generic name like "arg_0".
    is_int_dtype = any(t in dtype for t in ("int", "uint", "long", "short", "bool"))
    if is_int_dtype or _is_special_tensor(name):
        init = "randint"
        # A special tensor (position/mask/ids) holds indices, not activations,
        # so force it to an integer dtype. This keeps the randint init consistent
        # even when the tensor was captured under a floating-point dtype
        # tensor captured as bfloat16): randint on a float
        # dtype is meaningless, so it becomes torch.int64 here.
        if _is_special_tensor(name):
            dtype = str(DEFAULT_INT_DTYPE)
        # Use the smallest dimension of the tensor's own shape as the exclusive
        # upper bound (e.g. shape (64, 32, 128) -> high=32). This keeps generated
        # index/position values in range for that tensor rather than using a
        # fixed, possibly out-of-range constant. Guard against empty shapes and
        # zero/one-sized dims (randint needs high >= 1).
        shape = tensor_info.get("shape") or []
        high = min(shape) if shape else 1
        init_args = {"high": max(int(high), 1)}
    elif init in ("randn", "rand"):
        # Float random tensors use xavier init. xavier is undefined for <2-D
        # shapes (the OOT framework rejects it), so 1-D float tensors fall back
        # to randn.
        shape = tensor_info.get("shape") or []
        init = "xavier" if len(shape) >= 2 else "randn"

    tensor_spec = {
        "shape": tensor_info["shape"],
        "stride": None,  # Let PyTorch compute default stride
        "storage_offset": 0,
        "dtype": dtype,
        "device": "spyre",
        "init": init,
    }

    if init_args:
        tensor_spec["init_args"] = init_args

    return tensor_spec


def _convert_captured_input_to_sample_input(inp_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert captured input spec to sample_inputs_func format.

    Uses pytree utilities to handle single tensors and nested collections uniformly.
    The key insight: pytree lets us treat single tensors and collections the same way.
    """
    inp_name = inp_spec["name"]
    inp_type = inp_spec["type"]

    if inp_type == "value":
        # A plain Python scalar (or None) recorded verbatim. Resolved by the OOT
        # framework's InputArgValue, which passes it through unchanged.
        return {"value": inp_spec["value"]}

    if inp_type == "tensor":
        # Single tensor - wrap in standard format
        return {"tensor": _tensor_info_to_spec(inp_spec, inp_name)}

    elif inp_type in ("tuple", "list", "dict", "pytree"):
        # Collection of tensors - pytree handles all container types uniformly
        # Convert each tensor in the flattened structure
        tensor_list = [
            _tensor_info_to_spec(item, inp_name) for item in inp_spec.get("items", [])
        ]

        return {"tensor_list": tensor_list}

    elif inp_type == "cache":
        # A KV cache: emit cache_path + per-layer key/value tensor specs so the
        # test side can rebuild a concrete Cache and prime it via update(),
        # reproducing the decode path (attend over past + new token).
        cache_spec: Dict[str, Any] = {
            "cache_path": inp_spec["cache_path"],
            "layer_idx": inp_spec["layer_idx"],
            "key": _tensor_info_to_spec(inp_spec["key"], f"{inp_name}_key"),
            "value": _tensor_info_to_spec(inp_spec["value"], f"{inp_name}_value"),
        }
        if inp_spec.get("max_cache_len") is not None:
            cache_spec["max_cache_len"] = inp_spec["max_cache_len"]
        if "config_path" in inp_spec:
            cache_spec["config_path"] = inp_spec["config_path"]
            cache_spec["config_kwargs"] = inp_spec.get("config_kwargs", {})
        return {"cache": cache_spec}

    else:
        return {"value": None}


def _validate_cache_mask_consistency(
    invocation_inputs: List[Dict[str, Any]], module_name: str
) -> None:
    """Warn if a cached (decode) invocation lacks a mask that can cover the past.

    When an invocation carries a KV cache, the test side rebuilds a Cache primed
    with ``past_len`` tokens and drives a decode forward. That forward also needs
    an ``attention_mask`` whose key/value axis is at least ``past_len`` (the
    cache's populated length) — otherwise the mask and the cache disagree about
    how many past tokens exist and the replayed decode attends over the wrong
    span. This is a generation-time sanity check (logged, not fatal) so a
    malformed invocation is visible rather than silently emitted.

    K/V key shape is ``[B, num_kv_heads, head_dim, past_len]`` (past_len last).
    A 4-D ``attention_mask`` is ``[B, 1, q_len, kv_len]`` (kv_len last). We only
    require ``past_len <= kv_len`` since a fixed-length cache (e.g. StaticCache)
    reports its allocation, not its populated length, in the mask.
    """
    cache_spec = None
    mask_spec = None
    for inp in invocation_inputs:
        if inp.get("type") == "cache":
            cache_spec = inp
        elif inp.get("name") == "attention_mask":
            mask_spec = inp

    if cache_spec is None:
        return  # prefill invocation — nothing to check

    if mask_spec is None:
        logger.warning(
            "%s: decode invocation has a KV cache but no attention_mask; "
            "the replayed decode cannot mask the cached past correctly.",
            module_name,
        )
        return

    key_shape = cache_spec.get("key", {}).get("shape")
    mask_shape = mask_spec.get("shape")
    if not key_shape or not mask_shape:
        return
    past_len = key_shape[-1]
    kv_len = mask_shape[-1]
    if kv_len < past_len:
        logger.warning(
            "%s: attention_mask kv_len=%d < cached past_len=%d; mask cannot "
            "cover the cached past for the decode step.",
            module_name,
            kv_len,
            past_len,
        )


def _build_module_entry_dict(module_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a module entry dictionary for YAML generation.

    Args:
        module_info: Captured module information with multiple invocations

    Returns:
        Dictionary representing a module entry for YAML
    """
    # Build constructor_inputs
    constructor_args = []
    constructor_kwargs = {}

    for arg_spec in module_info.get("constructor_args", []):
        constructor_args.append(_convert_constructor_arg_to_sample_input(arg_spec))

    for key, kwarg_spec in module_info.get("constructor_kwargs", {}).items():
        if kwarg_spec["type"] == "int":
            constructor_kwargs[key] = kwarg_spec["value"]

    # Build forward_inputs from all invocations
    # NEW: Handle multiple invocations - each invocation becomes a separate input set
    invocations = module_info.get("invocations", [])

    if not invocations:
        # Fallback for old format (backward compatibility)
        invocations = [module_info.get("inputs", [])]

    # Process each invocation
    forward_inputs_list = []
    for invocation_inputs in invocations:
        forward_args = []
        forward_kwargs = {}

        for inp_spec in invocation_inputs:
            # Validate inp_spec has required fields
            if "name" not in inp_spec:
                logger.error(f"inp_spec missing 'name' field: {inp_spec}")
                continue  # Skip malformed entries

            inp_name = inp_spec["name"]
            converted = _convert_captured_input_to_sample_input(inp_spec)

            if inp_name.startswith("arg_"):
                forward_args.append(converted)
            else:
                forward_kwargs[inp_name] = converted

        # A decode invocation carries a KV cache; verify it also carries an
        # attention_mask whose key/value length can cover the cached past, so
        # the test side rebuilds a self-consistent (mask, cache) pair rather
        # than a decode step that silently attends over the wrong span.
        _validate_cache_mask_consistency(
            invocation_inputs, module_info.get("name", "<unknown>")
        )

        forward_inputs_list.append(
            {
                "args": forward_args if forward_args else [],
                "kwargs": forward_kwargs if forward_kwargs else {},
            }
        )

    # One invocation per entry is the point of the phase split: it is what makes a
    # test id identify exactly one phase. Warn rather than fail so an unexpected
    # shape is visible in the generated YAML instead of aborting the capture.
    if module_info.get("phase") and len(forward_inputs_list) > 1:
        logger.warning(
            "%s: %d invocations in a phase-split entry (expected 1), so a failing "
            "test id will not say which phase broke.",
            module_info.get("name", "<unknown>"),
            len(forward_inputs_list),
        )

    forward_inputs = forward_inputs_list

    # Record where the class is defined so a reader of the generated YAML can
    # jump straight to the source. Appended to the free-text description rather
    # than emitted as its own key, so the entry stays within the shape the OOT
    # framework's include schema accepts. Absent for captures that carry no
    # source location (the vLLM generator builds module_info dicts by hand).
    description = f"Module: {module_info['module_path']}"
    location = _source_reference(
        module_info.get("source_file"), module_info.get("source_lineno")
    )
    if location:
        description = f"{description} (defined at {location})"

    # Build module entry
    entry = {
        "name": module_info["name"],
        "module_path": module_info["module_path"],
        "description": description,
        "constructor_inputs": {
            "args": constructor_args if constructor_args else [],
            "kwargs": constructor_kwargs if constructor_kwargs else {},
        },
        "forward_inputs": forward_inputs,
    }

    # Only the Spyre-loader capture path records a device-side parameter layout:
    # its modules were captured from a model already moved to Spyre by
    # move_model_to_spyre. HF-loader entries omit the key entirely so existing
    # configs regenerate byte-identically.
    if module_info.get("apply_device_layout"):
        entry["apply_device_layout"] = True

    return entry


def generate_unified_yaml_config(
    captured_modules: List[Dict[str, Any]], model_name: str
) -> str:
    """Generate unified YAML configuration using yaml.dump().

    This creates a single YAML file with edits.modules.include that contains:
    - Module name and path
    - constructor_inputs: Args/kwargs for module.__init__()
    - forward_inputs: Args/kwargs for module.forward()
    """
    # Build module entries
    module_entries = [_build_module_entry_dict(m) for m in captured_modules]

    # Build the complete configuration dictionary
    config = {
        "test_suite_config": {
            "files": [
                {
                    "path": "${TORCH_ROOT}/test/test_modules.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": ["*TestModule*::test_forward"],
                            "mode": "xfail",
                            "tags": [f"model__{model_name}"],
                            # Spyre's custom ops have no registered autograd
                            # formula, so upstream's test_forward (which builds
                            # modules with ordinary requires_grad=True
                            # parameters) must run under torch.no_grad() to
                            # avoid AOTAutograd tracing a backward graph at
                            # compile time.
                            "no_grad": True,
                            "edits": {"modules": {"include": module_entries}},
                        }
                    ],
                },
                {
                    "path": "${TORCH_DEVICE_ROOT}/tests/test_modules_custom.py",
                    "unlisted_test_mode": "skip",
                    "tests": [
                        {
                            "names": [
                                "*TestModuleCustom*::test_with_cpu",
                                "*TestModuleCustom*::test_eager_vs_compile",
                                "*TestModuleCustom*::test_layout_stride",
                            ],
                            "mode": "xfail",
                            "tags": [f"model__{model_name}", "custom_tests"],
                            # Same AOTAutograd/no_grad issue as test_forward
                            # above: these custom tests also build modules
                            # with requires_grad=True parameters and compile
                            # them for Spyre.
                            "no_grad": True,
                            "edits": {"modules": {"include": module_entries}},
                        }
                    ],
                },
            ],
            "global": {
                "supported_dtypes": [
                    {"name": "float16", "precision": {"atol": 0.005, "rtol": 0.005}},
                    {"name": "float32", "precision": {"atol": 0.001, "rtol": 0.001}},
                    {"name": "bfloat16", "precision": {"atol": 0.005, "rtol": 0.005}},
                ],
                "input_config": {"seed": 123},
            },
        }
    }

    # Generate YAML string with header comments and consistent 2-space indentation
    header = f"""# Auto-generated unified test configuration for {model_name}
# Generated by auto_generate_module_config.py
# Format compatible with PyTorch's test_modules.py (using edits.modules.include)

"""

    # Use custom Dumper with 2-space indentation for consistency
    yaml_str = header + yaml.dump(
        config,
        Dumper=PrettyDumper,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        width=float("inf"),  # Prevent line wrapping
    )
    return yaml_str


def run_prefill(model, inputs) -> Any:
    """Run a prefill forward pass under ``torch.no_grad()``.

    Wraps a single ``model(**inputs, use_cache=True)`` call so the same
    error-handling can be reused anywhere a prefill forward is needed.

    Args:
        model: The model to invoke.
        inputs: Mapping of forward kwargs (e.g. tokenizer output).

    Returns:
        The model outputs, or ``None`` if the forward pass raised.
    """
    logger.info(f"  Input shape: {inputs['input_ids'].shape}")
    try:
        with torch.no_grad():
            return model(**inputs, use_cache=True)
    except Exception as e:
        logger.exception(f"  ERROR during prefill: {e}")
        return None


def _build_decode_inputs(inputs, past_key_values) -> Dict[str, Any]:
    """Build the forward kwargs for a single decode step from prefill state.

    The new tensors are created on the same device as the prefill inputs, so a
    capture running on cuda/spyre does not mix an on-device prefill with CPU decode
    tensors (``torch.cat`` and the forward would both raise).
    """
    batch_size = inputs["input_ids"].shape[0]
    device = inputs["input_ids"].device
    # Single new token for decode
    next_token = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    return {
        "input_ids": next_token,  # Shape: [B, 1]
        "attention_mask": torch.cat(
            [
                inputs["attention_mask"],
                torch.ones((batch_size, 1), dtype=torch.long, device=device),
            ],
            dim=1,
        ),
        "past_key_values": past_key_values,  # Use cached KV
        "use_cache": True,
    }


def run_decode(model, inputs, prefill_outputs) -> Any:
    """Run a single decode forward pass using the KV cache from prefill.

    No-op (returns ``None``) when ``prefill_outputs`` carries no usable KV
    cache. Wraps the ``model(**decode_inputs)`` call with the same
    error-handling so it can be reused wherever a decode forward is needed.

    Args:
        model: The model to invoke.
        inputs: The original prefill forward kwargs (for shapes / masks).
        prefill_outputs: The outputs returned by :func:`run_prefill`.

    Returns:
        The decode outputs, or ``None`` if skipped or the forward raised.
    """
    if (
        prefill_outputs is None
        or not hasattr(prefill_outputs, "past_key_values")
        or prefill_outputs.past_key_values is None
    ):
        logger.info("\n  Skipping decode pass - no KV cache available")
        return None

    decode_inputs = _build_decode_inputs(inputs, prefill_outputs.past_key_values)
    logger.info(f"Decode input_ids shape: {decode_inputs['input_ids'].shape}")
    logger.info(f"Decode attention_mask shape: {decode_inputs['attention_mask'].shape}")
    logger.info(
        f"Decode past_key_values layers: {len(decode_inputs['past_key_values'])}"
    )
    try:
        with torch.no_grad():
            decode_outputs = model(**decode_inputs)
        logger.info(
            f"Decode complete. Output shape: "
            f"{decode_outputs.logits.shape if hasattr(decode_outputs, 'logits') else 'N/A'}"
        )
        return decode_outputs
    except Exception:
        logger.exception("ERROR during decode")
        return None


def register_capture_hooks(
    model,
    capture: "ModuleInfoCapture",
    excluded_types: frozenset = frozenset(),
) -> List[Any]:
    """Register the model-level + per-module capture hooks; return their handles.

    Split out of :func:`capture_module_invocations` so an alternative driver -- the
    Spyre loader path, which cannot call ``model(**inputs)`` -- reuses the exact
    same instrumentation instead of a divergent copy. The caller owns removal and
    must do it in a ``finally``.
    """
    all_custom_modules = get_all_custom_modules(
        model,
        excluded_types=excluded_types,
        capture_layers=capture.capture_layers,
    )
    logger.info(f"Found {len(all_custom_modules)} custom module instances")

    # This hook sets context that module-level hooks will read
    model_hook = capture.create_model_hook()
    handles = [model.register_forward_pre_hook(model_hook, with_kwargs=True)]

    # Register hooks on ALL custom module instances (not just unique types)
    for module_name, module_type, module_instance in all_custom_modules:
        hook = capture.create_hook(module_name, module_type, module_instance)
        handles.append(
            module_instance.register_forward_pre_hook(hook, with_kwargs=True)
        )
    return handles


def capture_module_invocations(model, capture: ModuleInfoCapture, inputs) -> None:
    """Register capture hooks, run prefill + decode, then remove the hooks.

    This drives the model through both execution modes so that
    ``capture`` observes every unique module invocation pattern. Hooks are
    always removed, even if a forward pass raises.

    Args:
        model: The model to instrument and run.
        capture: The :class:`ModuleInfoCapture` to populate.
        inputs: Forward kwargs for the prefill pass (e.g. tokenizer output).
    """
    handles = register_capture_hooks(model, capture)
    try:
        prefill_outputs = run_prefill(model, inputs)
        run_decode(model, inputs, prefill_outputs)
    finally:
        # Remove hooks even if a forward pass raised
        for handle in handles:
            handle.remove()


def resolve_capture_device(device: Optional[str]) -> Optional[str]:
    """Validate the device the HF capture forward pass will run on.

    ``None`` keeps the historical behaviour: ``from_pretrained`` leaves the model on
    CPU. Any other value is checked for availability up front, so an unavailable
    accelerator fails with a clear message here rather than deep inside the forward
    pass. ``"spyre"`` additionally needs ``torch_spyre`` imported to register the
    backend.

    This is the ``--loader hf`` capture device (``--capture_device``), distinct from
    ``--device``, which the ``--loader spyre`` path uses to patch
    ``hf_common.DEVICE``.
    """
    if device is None:
        return None

    # "spyre" is handled BEFORE torch.device(): torch does not know the backend
    # until torch_spyre is imported, so torch.device("spyre") would raise
    # "Expected one of cpu, cuda, ..." and mask the real cause.
    if device.split(":")[0] == "spyre":
        try:
            import torch_spyre  # noqa: F401  (registers the "spyre" backend)
        except ImportError as exc:
            raise RuntimeError(
                f"--capture_device {device!r} requested but torch_spyre is not "
                f"installed ({exc}). Run on the Spyre pod or use "
                f"--capture_device cpu."
            ) from exc
        # Importing is not enough: the source tree imports fine without its
        # compiled extension, leaving the backend unregistered. Allocate a tensor
        # to confirm the device actually works before loading a whole model onto it.
        try:
            torch.zeros(1, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"--capture_device {device!r} requested and torch_spyre imported, "
                f"but the device is not usable ({exc}). The backend extension is "
                f"probably missing; run on the Spyre pod or use "
                f"--capture_device cpu."
            ) from exc
        return device

    device_type = torch.device(device).type
    if device_type == "cpu":
        return device
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--capture_device {device!r} requested but "
                f"torch.cuda.is_available() is False. Run on a CUDA host or use "
                f"--capture_device cpu."
            )
        return device

    # Anything else is passed through: torch may know a backend we do not.
    logger.warning(
        "Capture device %r is not one of cpu/cuda/spyre; passing it through "
        "unchecked.",
        device,
    )
    return device


def load_model_only(
    model_path: str,
    model_cls=AutoModel,
    **from_pretrained_kwargs: Any,
):
    """Load an eval-mode model (no tokenizer).

    Split out from :func:`load_model_and_tokenizer` so callers whose tokenizer
    is not an ``AutoTokenizer`` (e.g. ``mistral_common``'s ``MistralTokenizer``,
    or a VLM processor) can still reuse the model-loading path.

    Args:
        model_path: HuggingFace model path or local directory.
        model_cls: The class to load with. Defaults to :class:`AutoModel`;
            pass :class:`AutoModelForCausalLM` for causal LMs whose bare
            backbone lacks ``past_key_values`` / logits (e.g. ``gpt_oss``), or
            an explicit architecture class such as
            ``Mistral3ForConditionalGeneration`` for VLMs.
        **from_pretrained_kwargs: Extra kwargs forwarded to
            ``from_pretrained`` (e.g. ``dtype``, ``device_map``,
            ``quantization_config``, ``trust_remote_code``). ``dtype``
            defaults to :data:`DEFAULT_FLOAT_DTYPE` (bfloat16, the dtype Spyre
            runs in) rather than ``from_pretrained``'s float32; pass it
            explicitly to override.

    Returns:
        The loaded, ``.eval()``-mode model.
    """
    # Capture in bfloat16 by default so the recorded floating-point tensors match
    # the dtype Spyre executes in. Callers may still override dtype.
    from_pretrained_kwargs.setdefault("dtype", DEFAULT_FLOAT_DTYPE)
    logger.info(f"Loading model: {model_path} via {model_cls.__name__}")
    return model_cls.from_pretrained(model_path, **from_pretrained_kwargs).eval()


def load_model_and_tokenizer(
    model_path: str,
    model_cls=AutoModel,
    **from_pretrained_kwargs: Any,
):
    """Load an eval-mode model and its ``AutoTokenizer``, fixing a missing pad token.

    Convenience wrapper around :func:`load_model_only` for the common case
    where the tokenizer is a standard HF ``AutoTokenizer``. See
    :func:`load_model_only` for the argument semantics.

    Returns:
        ``(model, tokenizer)``.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Fix missing pad_token for Mistral tokenizers
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model_only(model_path, model_cls=model_cls, **from_pretrained_kwargs)
    return model, tokenizer


def build_dummy_inputs(
    tokenizer, seq_len: int, device: Optional[str] = None
) -> Dict[str, Any]:
    """Tokenize placeholder text padded/truncated to ``seq_len``.

    ``device`` relocates the tokenized tensors, which must sit on the same device
    as the model or the forward pass raises a device mismatch. ``None`` leaves them
    on CPU (the tokenizer's default).
    """
    # Generate enough text to reach desired seq_len
    text = "This is a test input for capturing module information. " * (
        seq_len // 10 + 1
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    if device is not None:
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }
    return encoded


def load_spyre_model_and_tokenizer(
    model_path: str,
    dtype: torch.dtype = torch.float16,
    device: str = SPYRE_DEVICE,
):
    """Load via ``AutoSpyreModelForCausalLM``; return (model, tokenizer, module, snapshot).

    Four values, because the Spyre execution path needs more than the model:

    - ``module``: the resolved adapter module. Spyre runs
      ``module._run_forward(...)``, NOT ``model.forward``, so the driver needs it.
    - ``snapshot``: each layer's pre-Spyre HF attention class/config, recorded
      before ``prepare_for_spyre()`` discards that information (see
      :func:`_snapshot_hf_attention_classes`).

    ``dtype`` defaults to float16 to match ``AutoSpyreModelForCausalLM``, NOT to
    this module's bfloat16 :data:`DEFAULT_FLOAT_DTYPE`.

    ``device`` exists for off-pod dry runs: ``hf_common.DEVICE`` is a module
    constant pinned to "spyre", and the CPU test lane patches it in place. It must
    be patched BEFORE ``auto_spyre_model`` is imported, because
    ``move_model_to_spyre`` reads the module global at call time.
    """
    _ensure_hf_adapters_importable()

    if device != SPYRE_DEVICE:
        from hf_adapters import hf_common

        logger.info("Patching hf_common.DEVICE -> %s (off-pod dry run)", device)
        hf_common.DEVICE = device

    from hf_adapters.auto_spyre_model import (
        AutoSpyreModelForCausalLM,
        resolve_adapter_module,
    )
    from hf_adapters.hf_common import SpyreUnsupportedModelError

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve the adapter first, so an unsupported model fails before the (slow)
    # load + compile, and so the driver has _run_forward.
    adapter_module = resolve_adapter_module(model_path)
    if getattr(adapter_module, "_is_encoder_only", False):
        raise SpyreUnsupportedModelError(
            f"{model_path} is encoder-only; the Spyre loader path drives a "
            f"causal-LM generate() and cannot capture it. Use --loader hf."
        )

    # Snapshot the HF attention classes from a plain load first:
    # AutoSpyreModelForCausalLM.from_pretrained does load + prepare + move in one
    # call with no hook in between, and prepare_for_spyre() destroys the
    # provenance the Spyre attention entries need. Freed immediately to bound the
    # peak of holding two copies of the weights.
    probe = load_model_only(model_path, model_cls=AutoModelForCausalLM, dtype=dtype)
    snapshot = _snapshot_hf_attention_classes(probe)
    del probe

    logger.info(
        "Loading %s via AutoSpyreModelForCausalLM (dtype=%s)", model_path, dtype
    )
    model = AutoSpyreModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    return model, tokenizer, adapter_module, snapshot


def run_spyre_capture_forward(
    model,
    tokenizer,
    adapter_module,
    seq_len: int,
    max_new_tokens: int = 3,
    device: Optional[str] = None,
):
    """Drive the Spyre execution path so the capture hooks see every forward shape.

    Delegates to the adapter's own generate loop rather than re-implementing it: that
    loop already builds the block-padded input_ids, the KV caches, the position_ids
    and the prefill/decode masks exactly as production does, so the captured shapes
    are the real ones.

    ``max_new_tokens`` only has to reach the second of the two forward shapes the
    loop produces -- the padded prompt pass, then the per-token pass -- so anything
    >= 2 suffices and the default leaves one step of margin. Extra steps add no new
    shape: since hf-adapters#330 the cache is written by an indirect scatter with a
    *tensor* index, so one compiled binary serves every write position and every
    decode step looks identical to the capture.

    ``seq_len`` is only a target: generate() block-pads the prompt to a multiple of
    BLOCK_SIZE, so the captured sequence length is the padded one. That is the
    correct thing to record -- it is what Spyre actually runs.

    Returns the generated text, or ``None`` if the forward raised.
    """
    try:
        with torch.no_grad():
            inputs = build_dummy_inputs(tokenizer, seq_len, device=device)

            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                timing=True,
            )
    except Exception:
        logger.exception("ERROR during Spyre forward")
        return None


def generate_spyre_module_config(
    model_path: str,
    seq_len: int = 128,
    output: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
    device: str = SPYRE_DEVICE,
    max_new_tokens: int = 3,
    excluded_types: frozenset = SPYRE_EXCLUDED_MODULE_TYPES,
    capture_layers: Optional[Set[int]] = None,
):
    """Capture a module-test YAML from an ``AutoSpyreModelForCausalLM`` model.

    The programmatic entry point for the Spyre loader path; :func:`main` is a thin
    CLI wrapper over it, so ``--loader spyre`` and a direct call share one
    implementation.

    Emitted entries name the Spyre wrapper as ``module_path`` with the HF module it
    wrapped as a nested module arg, and carry ``apply_device_layout: true`` so the
    module test allocates parameters with the adapter's device layout.

    ``capture_layers`` selects which decoder layers to record, by index, exactly as
    on the HF path -- the adapter replaces ``layers[i]`` in place, so the
    ``named_modules()`` paths stay ``…layers.N…`` and the same filter applies.
    ``None`` records only layer 0. Note this bounds only what the YAML lists:
    ``prepare_for_spyre()`` has already compiled every layer by the time the hooks
    are registered, so narrowing it shrinks the emitted config (and the module
    test's own compiles), never this capture run.

    NOTE: importing hf_adapters patches the RMSNorm class globally
    (``patch_rmsnorm``), so a single process must not also run the HF loader path.

    Returns:
        The written YAML path.
    """
    model, tokenizer, adapter_module, snapshot = load_spyre_model_and_tokenizer(
        model_path, dtype=dtype, device=device
    )

    capture = ModuleInfoCapture(
        spyre_attn_snapshot=snapshot,
        spyre_model=model,
        capture_layers=capture_layers,
    )
    handles = register_capture_hooks(model, capture, excluded_types=excluded_types)
    try:
        run_spyre_capture_forward(
            model, tokenizer, adapter_module, seq_len, max_new_tokens, device
        )
    finally:
        for handle in handles:
            handle.remove()

    # Mark every captured entry so the emitted YAML requests the adapter's
    # device-side parameter layout (see _build_module_entry_dict). Set here rather
    # than per-hook because it is a property of the capture path, not of any one
    # module.
    for module_data in capture.module_data.values():
        module_data["apply_device_layout"] = True

    return write_module_config(capture, model_path, output, filename_suffix="_adapter")


def write_module_config(
    capture: ModuleInfoCapture,
    model_path: str,
    output: str = None,
    filename_suffix: str = "",
):
    """Generate the unified YAML config from captured modules and write it out.

    Splits every captured module into one entry per execution phase first (see
    :func:`split_module_data_by_phase`), so each phase gets its own test id. Done
    here rather than in each capture driver so both loaders get it from one place.
    """
    split_module_data_by_phase(capture)

    # Extract model name from path (handle both local paths and HuggingFace paths)
    model_path_parts = model_path.rstrip("/").split("/")
    model_name = model_path_parts[
        -1
    ]  # e.g., "granite-3.3-8b-instruct" or "granite-3.0-2b-instruct"

    # For the YAML content, no normalization
    model_name_normalized = model_name

    # Generate unified YAML config (new format)
    unified_yaml_content = generate_unified_yaml_config(
        capture.get_captured_modules(), model_name_normalized
    )

    # Determine output path
    if output:
        output_path = output
    else:
        # filename_suffix keeps the two loader paths from overwriting each other:
        # the HF path owns <model>_spyre.yaml, the Spyre-adapter path writes
        # <model>_spyre_adapter.yaml.
        output_path = (
            f"./tests/configs/module_tests/"
            f"{model_name_normalized}{filename_suffix}.yaml"
        )

    # Write unified YAML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(unified_yaml_content)

    logger.info(f"\n✓ Generated unified configuration: {output_file}")

    # Print module summary
    captured_modules = capture.get_captured_modules()
    logger.info("\n  Module Summary:")
    logger.info(f"    Total modules captured: {len(captured_modules)}")
    for module_info in captured_modules:
        logger.info(f"      - {module_info['name']}")

    return output_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-generate module configuration YAML using forward hooks"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="HuggingFace model path (e.g., ibm-granite/granite-3.3-8b-instruct)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=128,
        help="Sequence length for forward pass (default: 128)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML file path (default: "
        "./tests/configs/module_tests/<model>.yaml for --loader hf, "
        "<model>_adapter.yaml for --loader spyre)",
    )
    parser.add_argument(
        "--loader",
        choices=["hf", "spyre"],
        default="hf",
        help="hf (default): AutoModel/AutoModelForCausalLM driven through a "
        "StaticCache prefill+decode. spyre: AutoSpyreModelForCausalLM (patched + "
        "compiled blocks) driven through hf_common.generate(). Mutually exclusive "
        "-- patch_rmsnorm rewrites the RMSNorm class globally, so one process "
        "does one loader.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=None,
        help="Load dtype. Default: bfloat16 for --loader hf, float16 for "
        "--loader spyre (matching AutoSpyreModelForCausalLM). 'auto' consults the "
        "adapter registry's per-model dtype.",
    )
    parser.add_argument(
        "--device",
        default=SPYRE_DEVICE,
        help="--loader spyre only: patches hf_common.DEVICE. Use 'cpu' for an "
        "off-pod dry run (no torch_spyre required).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=3,
        help="--loader spyre only: decode steps to run. The generate loop produces "
        "two forward shapes -- the padded prompt pass and the per-token pass -- so "
        "anything >= 2 reaches both; the default leaves one step of margin. Extra "
        "steps add no new shape, because the KV cache is written by an indirect "
        "scatter whose index is a tensor, so every decode step looks identical.",
    )
    parser.add_argument(
        "--no_static_cache",
        action="store_true",
        help="--loader hf only. Disable the StaticCache used for the forward pass "
        "(default: enabled). "
        "When set, the model uses its default dynamic KV cache instead.",
    )
    parser.add_argument(
        "--max_cache_len",
        type=int,
        default=2048,
        help="--loader hf only. max_cache_len for the StaticCache (default: 2048). "
        "Ignored when --no_static_cache is set.",
    )
    parser.add_argument(
        "--capture_layers",
        type=str,
        default="0",
        help="Comma-separated decoder layer indices to capture, or 'all' "
        "(default: 0). Applies to both loaders. Each captured layer emits its own "
        "entries, named <Module>_layer<N>. A decoder stack repeats the same shapes "
        "per layer, so capturing many layers multiplies the YAML and the number of "
        "compiled binaries in the module test -- prefer a couple of "
        "representatives.",
    )
    parser.add_argument(
        "--capture_device",
        type=str,
        default=None,
        help="--loader hf only. Device to RUN the capture forward pass on, e.g. "
        "cpu / cuda / cuda:1 / spyre (default: leave the model on CPU). This is "
        "where the model executes while its inputs are recorded; it does not "
        "change the 'device' written into the generated tensor specs, which stays "
        "spyre because that is where the module test runs. Distinct from --device, "
        "which applies to --loader spyre.",
    )
    return parser.parse_args()


def _parse_capture_layers(spec: str) -> Optional[Set[int]]:
    """Parse ``--capture_layers`` into a set of indices.

    ``'all'`` maps to :data:`CAPTURE_ALL_LAYERS` (no filtering), NOT to ``None``:
    ``None`` means "use the default", which is layer 0 only.
    """
    if spec.strip().lower() == "all":
        return CAPTURE_ALL_LAYERS
    layers = {int(x) for x in spec.split(",") if x.strip()}
    if not layers:
        raise ValueError(
            f"--capture_layers={spec!r} selected no layers; pass indices "
            f"(e.g. '0,1') or 'all'."
        )
    return layers


def generate_module_config(
    model_path: str,
    seq_len: int = 128,
    output: Optional[str] = None,
    capture_layers: Optional[Set[int]] = None,
    use_static_cache: bool = True,
    max_cache_len: int = 2048,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Capture a module-test YAML from a HuggingFace model (the ``--loader hf`` path).

    The programmatic entry point; :func:`main` is a thin CLI wrapper over it, so
    both routes share one implementation.

    ``capture_layers`` selects which decoder layers to record, by index. ``None``
    records only layer 0; pass :data:`CAPTURE_ALL_LAYERS` to record every layer.

    ``device`` is where the capture forward pass RUNS (e.g. ``"cpu"``, ``"cuda"``,
    ``"spyre"``); ``None`` leaves the model on CPU, as before. This is independent
    of the ``device`` recorded in the generated tensor specs, which stays ``spyre``
    because that is where the module test will run.

    Returns:
        The written YAML path.
    """
    device = resolve_capture_device(device)
    from_pretrained_kwargs: Dict[str, Any] = {}
    if dtype is not None:
        from_pretrained_kwargs["dtype"] = dtype
    if device is not None:
        # device_map places the weights at load time, avoiding a CPU copy that a
        # post-hoc .to(device) would make.
        from_pretrained_kwargs["device_map"] = device
        logger.info("Running capture forward pass on device: %s", device)

    model, tokenizer = load_model_and_tokenizer(model_path, **from_pretrained_kwargs)
    inputs = build_dummy_inputs(tokenizer, seq_len, device=device)

    # Use a StaticCache by default; disabling it falls back to the model's default
    # dynamic cache. The (empty) StaticCache is passed into the prefill forward
    # and reused for decode.
    if use_static_cache:
        inputs["past_key_values"] = StaticCache(
            config=model.config, max_cache_len=max_cache_len
        )

    capture = ModuleInfoCapture(capture_layers=capture_layers)
    capture_module_invocations(model, capture, inputs)

    return write_module_config(capture, model_path, output)


def _resolve_dtype(name: Optional[str], model_path: str, loader: str) -> torch.dtype:
    """Resolve ``--dtype`` to a torch dtype, defaulting per loader.

    The two loaders have different natural defaults: the HF capture path records
    bfloat16 (:data:`DEFAULT_FLOAT_DTYPE`), while ``AutoSpyreModelForCausalLM``
    loads float16.
    """
    if name == "auto":
        _ensure_hf_adapters_importable()
        from hf_adapters.auto_spyre_model import torch_dtype_for_model_path

        return torch_dtype_for_model_path(model_path)
    if name is not None:
        return getattr(torch, name)
    return torch.float16 if loader == "spyre" else DEFAULT_FLOAT_DTYPE


def main():
    args = parse_args()
    dtype = _resolve_dtype(args.dtype, args.model_path, args.loader)
    print("### START for", args.model_path, "###")

    if args.loader == "spyre":
        # These only shape the HF path's StaticCache; the Spyre path's KV caches
        # come from generate(). Warn rather than fail so a mixed command line is
        # visible instead of silently ignored.
        if args.no_static_cache:
            logger.warning("--no_static_cache is ignored with --loader spyre.")
        if args.max_cache_len != 2048:
            logger.warning("--max_cache_len is ignored with --loader spyre.")
        if args.capture_device is not None:
            logger.warning(
                "--capture_device is ignored with --loader spyre; use --device."
            )
        generate_spyre_module_config(
            args.model_path,
            seq_len=args.seq_len,
            output=args.output,
            dtype=dtype,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            capture_layers=_parse_capture_layers(args.capture_layers),
        )
        print("### FINISHED ###")
        return

    generate_module_config(
        args.model_path,
        seq_len=args.seq_len,
        output=args.output,
        capture_layers=_parse_capture_layers(args.capture_layers),
        use_static_cache=not args.no_static_cache,
        max_cache_len=args.max_cache_len,
        device=args.capture_device,
        dtype=dtype,
    )
    print("### FINISHED ###")


if __name__ == "__main__":
    main()
