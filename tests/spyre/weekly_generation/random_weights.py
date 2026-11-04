"""Construct the weekly scan's models without reading pretrained weights.

The context is confined to a spawned evaluation worker. Intercept the concrete
Transformers class *after* AutoModel has resolved it, so adapter-specific model
classes and SentenceTransformer pipelines still use their normal dispatch.
Tokenizers, configs and any required model code are still loaded normally.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch
from sentence_transformers.base.modules.module import Module
from transformers import AutoConfig, GenerationConfig, PretrainedConfig, PreTrainedModel


def _model_seed(seed: int, model_id: str, subfolder: str, class_name: str) -> int:
    """Keep initialization independent of load order, backend and batch size."""
    key = f"{seed}:{model_id}:{subfolder}:{class_name}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63)


@contextmanager
def random_weights(seed: int = 42) -> Iterator[None]:
    """Replace standard HF/ST weight loading with reproducible CPU initialization.

    Always initialize in float32, then convert, so separate CPU/Spyre loaders
    start with the same values even when their requested dtypes differ. Full
    model memory is still required. The original loaders and CPU RNG state are
    restored on exit. Custom loaders that bypass these HF/ST APIs are unsupported.
    """

    def from_config(
        cls: type[PreTrainedModel],
        pretrained_model_name_or_path: str | os.PathLike[str],
        *model_args: Any,
        **kwargs: Any,
    ) -> PreTrainedModel:
        if model_args:
            raise ValueError("Random weights do not support positional model arguments")
        if (
            kwargs.get("tp_plan") is not None
            or kwargs.get("distributed_config") is not None
        ):
            raise ValueError(
                "Random weights do not yet support tensor-parallel loading"
            )
        model_id = os.fspath(pretrained_model_name_or_path)
        subfolder = str(kwargs.get("subfolder") or "")
        hub_args: dict[str, Any] = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "revision",
                "token",
                "local_files_only",
                "trust_remote_code",
                "subfolder",
            )
            if key in kwargs
        }
        config: PretrainedConfig | None = kwargs.get("config")
        if config is None:
            config = AutoConfig.from_pretrained(model_id, **hub_args)
        if not isinstance(config, PretrainedConfig):
            raise TypeError("Random weights require a Transformers PretrainedConfig")
        config = copy.deepcopy(config)
        requested_dtype: torch.dtype | str | None = kwargs.get(
            "dtype", kwargs.get("torch_dtype", config.dtype)
        )
        if requested_dtype == "auto":
            requested_dtype = config.dtype
        if isinstance(requested_dtype, str):
            requested_dtype = getattr(torch, requested_dtype)
        constructor_keys = set(inspect.signature(cls.__init__).parameters) - {
            "self",
            "config",
        }
        constructor_keys.update(("attn_implementation", "experts_implementation"))
        constructor_args: dict[str, Any] = {
            key: value for key, value in kwargs.items() if key in constructor_keys
        }
        with torch.random.fork_rng(devices=[]):
            # torch.manual_seed() also seeds/initializes accelerator devices.
            torch.random.default_generator.manual_seed(
                _model_seed(seed, model_id, subfolder, cls.__name__)
            )
            model: PreTrainedModel = cls._from_config(
                config, dtype=torch.float32, **constructor_args
            )
        if requested_dtype is not None:
            if not isinstance(requested_dtype, torch.dtype):
                raise TypeError(f"Unsupported random-weight dtype: {requested_dtype!r}")
            module: torch.nn.Module = model
            module.to(dtype=requested_dtype)
            model.config.dtype = requested_dtype
        model.eval()
        if model.can_generate():
            try:
                model.generation_config = GenerationConfig.from_pretrained(
                    model_id, **hub_args
                )
            except OSError:
                # Repositories may omit this optional file. The constructor's
                # generation config already carries defaults from config.json.
                pass
        return model

    def initialize_st_module(
        cls: type[Module],
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        model: Module | None = None,
    ) -> Module:
        # Standard learned ST heads (e.g. Dense) construct from their own config
        # before calling this hook. Pooling/Normalize have no weights to load.
        if model is None:
            raise ValueError(
                "Random ST weights require a module constructed from config"
            )
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                _model_seed(seed, model_name_or_path, subfolder, cls.__name__)
            )
            for child in model.modules():
                reset: Callable[[], None] | None = getattr(
                    child, "reset_parameters", None
                )
                if callable(reset):
                    reset()
                elif any(True for _ in child.parameters(recurse=False)):
                    raise ValueError(
                        f"Random initialization is unsupported for ST module {type(child).__name__}"
                    )
        return model

    with (
        torch.random.fork_rng(devices=[]),
        patch.object(PreTrainedModel, "from_pretrained", classmethod(from_config)),
        patch.object(Module, "load_torch_weights", classmethod(initialize_st_module)),
    ):
        yield
