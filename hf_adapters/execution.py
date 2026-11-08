# Copyright 2026 The Torch-Spyre Authors.
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

"""Runtime bindings for existing adapter blocks, independent of serving engines."""

import importlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionSpec:
    """Attention geometry and optional model-owned additive position bias.

    ``position_bias()`` returns ``[1, heads or 1, extent, extent]``. The extent must
    cover the runtime's padded request lengths; kernels slice it independently
    for each request group before adding it to the runtime-owned mask.
    """

    layer_idx: int
    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float
    is_causal: bool = True
    uses_cache: bool = True
    position_bias: torch.nn.Module | None = None


class ExternalAttention:
    def __init__(self, spec):
        self.spec = spec
        self.kernel = None

    def bind(self, kernel):
        if self.kernel is not None:
            raise RuntimeError("Attention executor has already been bound")
        self.kernel = kernel

    def __call__(self, q, k, v, mask, key_cache, value_cache, cache_index):
        if self.kernel is None:
            raise RuntimeError("Bind external attention before running adapter blocks")
        if any(x is not None for x in (mask, key_cache, value_cache, cache_index)):
            raise ValueError("External attention owns masks and KV caches")
        return self.kernel(q, k, v), None, None


class _DeferredBlock:
    def __init__(self, function):
        self.function = function
        self.compiled = None

    def __call__(self, *args, **kwargs):
        function = self.function if self.compiled is None else self.compiled
        return function(*args, **kwargs)


class ExternalRuntime:
    """Bind attention and compile existing blocks after registration/placement."""

    def __init__(self, *, max_model_len, task="generate", is_causal=True):
        self.max_model_len = max_model_len
        self.task = task
        self.is_causal = is_causal
        self.attentions = []
        self.blocks = []
        self.logits_scale = None
        self._compiled = False

    def attention(self, attn, *, head_size=None, scale=None):
        spec = AttentionSpec(
            attn.layer_idx,
            attn.config.num_attention_heads,
            getattr(
                attn.config, "num_key_value_heads", attn.config.num_attention_heads
            ),
            attn.head_dim if head_size is None else head_size,
            float(attn.scaling if scale is None else scale),
            is_causal=self.is_causal,
            uses_cache=self.task == "generate",
        )
        return self.register_attention(spec)

    def register_attention(self, spec):
        executor = ExternalAttention(spec)
        self.attentions.append(executor)
        return executor

    def wrap(self, function):
        block = _DeferredBlock(function)
        self.blocks.append(block)
        return block

    def compile_body(self, model):
        if self._compiled:
            raise RuntimeError("Adapter body compilation has already been installed")
        if any(attention.kernel is None for attention in self.attentions):
            raise RuntimeError("Bind attention before compiling adapter blocks")
        for block in self.blocks:
            block.compiled = torch.compile(
                block.function, dynamic=False, fullgraph=True
            )
        self._compiled = True


@dataclass
class PreparedExecution:
    model: object
    runtime: ExternalRuntime
    driver: object
    encoder_only: bool = False

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if self.encoder_only:
            return self.driver(
                self.model, input_ids, None, position_ids, token_type_ids
            )
        if token_type_ids is not None:
            raise ValueError("Decoder adapter execution does not accept token_type_ids")
        layers = len(self.runtime.attentions)
        return self.driver(
            self.model,
            input_ids,
            position_ids,
            None,
            [None] * layers,
            [None] * layers,
            None,
        )


def prepare_adapter_execution(model, *, max_model_len, task="generate"):
    """Prepare existing adapter blocks on CPU, without their dense caches or head."""
    from hf_adapters.hf_common import text_config

    if type(max_model_len) is not int or max_model_len <= 0:
        raise ValueError("max_model_len must be a positive integer")
    if task not in ("generate", "embed"):
        raise ValueError("Adapter execution supports generate or embed tasks")
    if model.training or next(model.parameters()).device.type != "cpu":
        raise ValueError("Prepare an eval-mode model on CPU before device placement")
    if hasattr(model, "_spyre_rope") or hasattr(model, "_spyre_compiled_blocks"):
        raise ValueError("Model has already been prepared for Spyre")
    cfg = text_config(model.config)
    adapters = {
        "bert": "Bert",
        "distilbert": "DistilBert",
        "mpnet": "MPNet",
        "modernbert": "ModernBert",
        "roberta": "Roberta",
        "xlm-roberta": "XLMRoberta",
        "gpt2": "GPT2",
        "gpt_neo": "GPTNeo",
        "gpt_neox": "GPTNeoX",
        "granite": "Granite",
        "llama": "Llama",
        "mistral": "Mistral",
        "ministral": "Ministral",
        "olmo": "Olmo",
        "olmo2": "Olmo2",
        "opt": "OPT",
        "qwen2": "Qwen2",
        "qwen3": "Qwen3",
        "phi3": "Phi3",
        "smollm3": "SmolLM3",
    }
    if cfg.model_type not in adapters or type(model).__name__ not in (
        adapters.get(cfg.model_type, "") + "Model",
        adapters.get(cfg.model_type, "")
        + ("LMHeadModel" if cfg.model_type == "gpt2" else "ForCausalLM"),
    ):
        raise ValueError(f"Adapter execution does not support {type(model).__name__}")
    module_name = (
        "xlm_roberta"
        if cfg.model_type in ("roberta", "xlm-roberta")
        else cfg.model_type
    )
    adapter = importlib.import_module(f"hf_adapters.hf_{module_name}")
    encoder_only = getattr(adapter, "_is_encoder_only", False)
    if encoder_only and task != "embed":
        raise ValueError("Encoder adapter execution requires task='embed'")
    is_causal = (
        not encoder_only
        and getattr(cfg, "is_causal", True)
        and not getattr(cfg, "use_bidirectional_attention", False)
    )
    if task == "generate" and not is_causal:
        raise ValueError("Generation requires causal attention")
    rope = getattr(cfg, "rope_parameters", None) or {}
    rope_configs = (
        rope.values()
        if rope and all(isinstance(value, dict) for value in rope.values())
        else (rope,)
    )
    for rope_config in rope_configs:
        rope_type = rope_config.get("rope_type", "default")
        if rope_type == "longrope":
            limit = getattr(cfg, "original_max_position_embeddings", None)
            if limit is None or max_model_len > limit:
                raise ValueError(
                    "Frozen LongRoPE requires the original short-context limit"
                )
        elif rope_type != "default":
            raise ValueError(
                f"Adapter execution does not support frozen {rope_type} RoPE"
            )
    window = getattr(cfg, "sliding_window", None)
    full_attention_types = {"attention", "full_attention"}
    external_sliding = getattr(adapter, "_supports_external_sliding_attention", False)
    if external_sliding or (window is not None and window >= max_model_len):
        full_attention_types.add("sliding_attention")
    if (
        not external_sliding and window is not None and 0 < window < max_model_len
    ) or any(
        kind not in full_attention_types
        for kind in (getattr(cfg, "layer_types", None) or [])
    ):
        raise ValueError(
            "Adapter execution requires full attention within max_model_len"
        )
    if getattr(cfg, "attn_logit_softcapping", None) is not None:
        raise ValueError("Adapter execution does not yet support attention softcapping")
    runtime = ExternalRuntime(
        max_model_len=max_model_len, task=task, is_causal=is_causal
    )
    adapter.prepare_for_spyre(model, runtime=runtime)
    if hasattr(model, "_spyre_rope"):
        rope = model._spyre_rope
        ropes = (
            rope.values() if isinstance(rope, (dict, torch.nn.ModuleDict)) else (rope,)
        )
        for rotary in ropes:
            rotary.freeze(max_model_len, next(model.parameters()).dtype)
    return PreparedExecution(
        model, runtime, adapter._run_backbone_forward, encoder_only
    )
