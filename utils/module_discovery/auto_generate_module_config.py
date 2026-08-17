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
from transformers import AutoModel, AutoTokenizer, StaticCache

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


def _extract_config_kwargs(config: Any) -> Dict[str, Any]:
    """Extract the config parameters the framework needs to rebuild a module.

    ``_attn_implementation`` is resolved to a concrete implementation (never
    ``None``) so a module reconstructed from the YAML dispatches attention the
    same way it did during capture.
    """
    config_kwargs: Dict[str, Any] = {}
    for attr in [
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
    ]:
        if hasattr(config, attr):
            config_kwargs[attr] = getattr(config, attr)

    if hasattr(config, "_attn_implementation"):
        config_kwargs["_attn_implementation"] = _resolve_attn_implementation(config)

    return config_kwargs


def _extract_tensor_info(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    """Extract information from a single tensor."""
    return {
        "type": "tensor",
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "is_random": not _is_special_tensor(name),
        "requires_grad": tensor.requires_grad,
    }


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
    added (layer today; execution phase is the expected next one). ``None``
    suffixes are dropped, so a module outside any decoder layer keeps exactly the
    name it has today.
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
        "max_cache_len": getattr(past_key_values, "max_cache_len", None),
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
        config_kwargs = {}
        for attr in [
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "num_hidden_layers",
            "max_position_embeddings",
        ]:
            if hasattr(config, attr):
                config_kwargs[attr] = getattr(config, attr)
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

    def __init__(self, capture_layers: Optional[Set[int]] = None):
        self.module_data: Dict[str, Dict[str, Any]] = {}
        self.seen_module_configs: Set[str] = (
            set()
        )  # Track unique configs, not just types
        # Track model-level context (KV cache, execution mode)
        self.current_model_context: Dict[str, Any] = {}
        # Layers to record, by index. ``None`` -> {0}: a decoder stack repeats the
        # same module shapes per layer, so recording all 40 of an 8B model would
        # multiply the YAML -- and, on Spyre, the number of compiled binaries in
        # the module test -- for little extra coverage. Pass CAPTURE_ALL_LAYERS to
        # record every layer.
        self.capture_layers: Set[int] = (
            {0} if capture_layers is None else set(capture_layers)
        )

    def capture_constructor_info(
        self, module, module_name: str, module_type: str
    ) -> Dict[str, Any]:
        """
        Capture constructor information from an instantiated module.

        This inspects the module to infer what constructor args were used.
        For Transformers modules, we look for config objects and layer_idx.
        """
        constructor_args = []
        constructor_kwargs = {}

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
                config_kwargs = _extract_config_kwargs(config)

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
            config_kwargs = _extract_config_kwargs(config)

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
                module, module_name, module_type
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

            # Analyze positional arguments using pytree
            for i, arg in enumerate(args):
                input_info = _process_pytree_structure(arg, f"arg_{i}")
                if input_info:
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
    model, capture_layers: Optional[Set[int]] = None
) -> List[Tuple[str, str, Any]]:
    """
    Get ALL custom module instances from the model (not just unique types).

    Args:
        model: The model to walk.
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
    all_custom_modules = get_all_custom_modules(
        model, capture_layers=capture.capture_layers
    )
    logger.info(f"Found {len(all_custom_modules)} custom module instances")

    # This hook sets context that module-level hooks will read
    model_hook = capture.create_model_hook()
    model_handle = model.register_forward_pre_hook(model_hook, with_kwargs=True)
    handles = [model_handle]

    # Register hooks on ALL custom module instances (not just unique types)
    for module_name, module_type, module_instance in all_custom_modules:
        hook = capture.create_hook(module_name, module_type, module_instance)
        handle = module_instance.register_forward_pre_hook(hook, with_kwargs=True)
        handles.append(handle)

    try:
        prefill_outputs = run_prefill(model, inputs)
        run_decode(model, inputs, prefill_outputs)
    finally:
        # Remove hooks even if a forward pass raised
        for handle in handles:
            handle.remove()


def resolve_capture_device(device: Optional[str]) -> Optional[str]:
    """Validate the device the capture forward pass will run on.

    ``None`` keeps the historical behaviour: ``from_pretrained`` leaves the model on
    CPU. Any other value is checked for availability up front, so an unavailable
    accelerator fails with a clear message here rather than deep inside the forward
    pass. ``"spyre"`` additionally needs ``torch_spyre`` imported to register the
    backend.
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
                f"--device {device!r} requested but torch_spyre is not installed "
                f"({exc}). Run on the Spyre pod or use --device cpu."
            ) from exc
        # Importing is not enough: the source tree imports fine without its
        # compiled extension, leaving the backend unregistered. Allocate a tensor
        # to confirm the device actually works before loading a whole model onto it.
        try:
            torch.zeros(1, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"--device {device!r} requested and torch_spyre imported, but the "
                f"device is not usable ({exc}). The backend extension is probably "
                f"missing; run on the Spyre pod or use --device cpu."
            ) from exc
        return device

    device_type = torch.device(device).type
    if device_type == "cpu":
        return device
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"--device {device!r} requested but torch.cuda.is_available() is "
                f"False. Run on a CUDA host or use --device cpu."
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
            ``from_pretrained`` (e.g. ``torch_dtype``, ``device_map``,
            ``quantization_config``, ``trust_remote_code``). ``torch_dtype``
            defaults to :data:`DEFAULT_FLOAT_DTYPE` (bfloat16, the dtype Spyre
            runs in) rather than ``from_pretrained``'s float32; pass it
            explicitly to override.

    Returns:
        The loaded, ``.eval()``-mode model.
    """
    # Capture in bfloat16 by default so the recorded floating-point tensors match
    # the dtype Spyre executes in. Callers may still override torch_dtype.
    from_pretrained_kwargs.setdefault("torch_dtype", DEFAULT_FLOAT_DTYPE)
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


def write_module_config(
    capture: ModuleInfoCapture, model_path: str, output: str = None
):
    """Generate the unified YAML config from captured modules and write it out."""
    # Extract model name from path (handle both local paths and HuggingFace paths)
    model_path_parts = model_path.rstrip("/").split("/")
    model_name = model_path_parts[
        -1
    ]  # e.g., "granite-3.3-8b-instruct" or "granite-3.0-2b-instruct"

    # For the YAML content, use underscores for the model_name field
    model_name_normalized = model_name.replace("-", "_").replace(".", "_")

    # Generate unified YAML config (new format)
    unified_yaml_content = generate_unified_yaml_config(
        capture.get_captured_modules(), model_name_normalized
    )

    # Determine output path
    if output:
        output_path = output
    else:
        # Use tests/configs directory for unified format
        output_path = f"./tests/configs/module_tests/{model_name_normalized}_spyre.yaml"

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
        help="Output YAML file path (default: ./tests/configs/<model>_spyre.yaml)",
    )
    parser.add_argument(
        "--no_static_cache",
        action="store_true",
        help="Disable the StaticCache used for the forward pass (default: enabled). "
        "When set, the model uses its default dynamic KV cache instead.",
    )
    parser.add_argument(
        "--max_cache_len",
        type=int,
        default=2048,
        help="max_cache_len for the StaticCache (default: 2048). "
        "Ignored when --no_static_cache is set.",
    )
    parser.add_argument(
        "--capture_layers",
        type=str,
        default="0",
        help="Comma-separated decoder layer indices to capture, or 'all' "
        "(default: 0). Each captured layer emits its own entries, named "
        "<Module>_layer<N>. A decoder stack repeats the same shapes per layer, so "
        "capturing many layers multiplies the YAML and the number of compiled "
        "binaries in the module test -- prefer a couple of representatives.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to RUN the capture forward pass on, e.g. cpu / cuda / "
        "cuda:1 / spyre (default: leave the model on CPU). This is where the "
        "model executes while its inputs are recorded; it does not change the "
        "'device' written into the generated tensor specs, which stays spyre "
        "because that is where the module test runs.",
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
):
    """Capture a module-test YAML from a HuggingFace model.

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


def main():
    args = parse_args()
    generate_module_config(
        args.model_path,
        seq_len=args.seq_len,
        output=args.output,
        capture_layers=_parse_capture_layers(args.capture_layers),
        use_static_cache=not args.no_static_cache,
        max_cache_len=args.max_cache_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
