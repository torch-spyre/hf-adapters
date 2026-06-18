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

"""Per-layer CPU vs Spyre compiled block comparison.

Creates tiny random-weight models (3–4 layers each, no weight download needed),
runs each decoder block on CPU (uncompiled) and Spyre (compiled), and asserts
the per-layer hidden-state max-abs-diff stays under threshold with no NaN.

Marked ``@pytest.mark.spyre_block`` and **deselected by default**. CI does not
run this until the senlib/torch interpreter-shutdown SIGABRT is root-caused
(see the long comment block in .github/workflows/_spyre_tests.yaml). To run
locally on a Spyre pod:

    pytest -s -vvv tests/spyre/test_block_cpu_vs_spyre.py -m spyre_block
    pytest -s -vvv tests/spyre/test_block_cpu_vs_spyre.py -m spyre_block -k qwen3
"""

import importlib

import pytest
import torch

DEVICE = "spyre"
SEQ_LEN = 64

# Per-model max-abs-diff threshold (random-weight tiny models — diffs are not
# directly comparable across architectures, so the bar is loose).
DEFAULT_MAX_DIFF = 0.5
PER_MODEL_MAX_DIFF: dict = {
    # Decode mode is known to drift further on random-weight blocks; we use
    # the same default for now and tighten as calibration data accumulates.
}


# ---------------------------------------------------------------------------
# Tiny config factories (one per architecture)
# ---------------------------------------------------------------------------


def _make_qwen3_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception:
        from transformers import Qwen3Config

        cfg = Qwen3Config(
            hidden_size=1024,
            num_attention_heads=16,
            num_key_value_heads=2,
            intermediate_size=2560,
            num_hidden_layers=3,
            vocab_size=151936,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_granite_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("ibm-granite/granite-3.3-8b-instruct")
    except Exception:
        from transformers import GraniteConfig

        cfg = GraniteConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=10944,
            num_hidden_layers=3,
            vocab_size=49152,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            embedding_multiplier=12.0,
            residual_multiplier=0.22,
            attention_multiplier=0.0625,
            logits_scaling=13.0,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_granite4_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("ibm-granite/granite-4.0-1b-base")
    except Exception:
        from transformers import GraniteMoeHybridConfig

        cfg = GraniteMoeHybridConfig(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=4,
            shared_intermediate_size=5632,
            num_hidden_layers=3,
            vocab_size=49152,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
            embedding_multiplier=12.0,
            residual_multiplier=0.22,
            attention_multiplier=0.0625,
            logits_scaling=13.0,
            num_local_experts=0,
            layers_block_type=["attention"] * 3,
        )
    cfg.num_hidden_layers = 3
    if hasattr(cfg, "layers_block_type"):
        cfg.layers_block_type = ["attention"] * cfg.num_hidden_layers
    if hasattr(cfg, "num_local_experts"):
        cfg.num_local_experts = 0
    cfg._attn_implementation = "eager"
    return cfg


def _make_smollm3_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM3-3B-Base")
    except Exception:
        from transformers import SmolLM3Config

        cfg = SmolLM3Config(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=4,
            intermediate_size=11008,
            num_hidden_layers=4,
            vocab_size=128256,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
            no_rope_layer_interval=4,
        )
    cfg.num_hidden_layers = 4  # keep 4 so NoPE pattern is exercised
    cfg.pad_token_id = None
    interval = getattr(cfg, "no_rope_layer_interval", 4)
    cfg.no_rope_layers = [
        int((i + 1) % interval != 0) for i in range(cfg.num_hidden_layers)
    ]
    if hasattr(cfg, "layer_types"):
        cfg.layer_types = ["full_attention"] * cfg.num_hidden_layers
    cfg._attn_implementation = "eager"
    return cfg


def _make_llama_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("meta-llama/Llama-3.2-3B")
    except Exception:
        from transformers import LlamaConfig

        cfg = LlamaConfig(
            hidden_size=3072,
            num_attention_heads=24,
            num_key_value_heads=8,
            intermediate_size=8192,
            num_hidden_layers=3,
            vocab_size=128256,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_qwen2_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
    except Exception:
        from transformers import Qwen2Config

        cfg = Qwen2Config(
            hidden_size=3584,
            num_attention_heads=28,
            num_key_value_heads=4,
            intermediate_size=18944,
            num_hidden_layers=3,
            vocab_size=152064,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_mistral_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("mistralai/Mistral-7B-v0.3")
    except Exception:
        from transformers import MistralConfig

        cfg = MistralConfig(
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=14336,
            num_hidden_layers=3,
            vocab_size=32768,
            rms_norm_eps=1e-5,
            max_position_embeddings=4096,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_olmo_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("allenai/OLMo-1B-hf")
    except Exception:
        from transformers import OlmoConfig

        cfg = OlmoConfig(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=8192,
            num_hidden_layers=3,
            vocab_size=50304,
            max_position_embeddings=2048,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_olmo2_config():
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained("allenai/OLMo-2-0425-1B")
    except Exception:
        from transformers import Olmo2Config

        cfg = Olmo2Config(
            hidden_size=2048,
            num_attention_heads=16,
            num_key_value_heads=16,
            intermediate_size=8192,
            num_hidden_layers=3,
            vocab_size=100352,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
        )
    cfg.num_hidden_layers = 3
    cfg._attn_implementation = "eager"
    return cfg


def _make_granite_vision_config():
    from transformers import GraniteConfig

    cfg = GraniteConfig(
        hidden_size=2560,
        num_attention_heads=40,
        num_key_value_heads=8,
        intermediate_size=8192,
        num_hidden_layers=3,
        vocab_size=100353,
        rms_norm_eps=1e-5,
        max_position_embeddings=131072,
        embedding_multiplier=12.0,
        residual_multiplier=0.22,
        attention_multiplier=0.0625,
        logits_scaling=10.0,
        rope_theta=10000000,
    )
    cfg._attn_implementation = "eager"
    return cfg


MODEL_REGISTRY = {
    "qwen3": {
        "name": "Qwen3 0.6B",
        "config_fn": _make_qwen3_config,
        "adapter": "hf_adapters.hf_qwen3",
    },
    "granite": {
        "name": "Granite 3.3 8B",
        "config_fn": _make_granite_config,
        "adapter": "hf_adapters.hf_granite",
    },
    "granite4": {
        "name": "Granite 4.0",
        "config_fn": _make_granite4_config,
        "adapter": "hf_adapters.hf_granitemoehybrid",
    },
    "smollm3": {
        "name": "SmolLM3",
        "config_fn": _make_smollm3_config,
        "adapter": "hf_adapters.hf_smollm3",
    },
    "llama": {
        "name": "Llama 3.2 3B",
        "config_fn": _make_llama_config,
        "adapter": "hf_adapters.hf_llama",
    },
    "qwen2": {
        "name": "Qwen2.5 7B",
        "config_fn": _make_qwen2_config,
        "adapter": "hf_adapters.hf_qwen2",
    },
    "mistral": {
        "name": "Mistral 7B",
        "config_fn": _make_mistral_config,
        "adapter": "hf_adapters.hf_mistral",
    },
    "olmo": {
        "name": "OLMo 1B",
        "config_fn": _make_olmo_config,
        "adapter": "hf_adapters.hf_olmo",
    },
    "olmo2": {
        "name": "OLMo2 1B",
        "config_fn": _make_olmo2_config,
        "adapter": "hf_adapters.hf_olmo2",
    },
    "granite-vision": {
        "name": "Granite Vision 4.1",
        "config_fn": _make_granite_vision_config,
        "adapter": "hf_adapters.hf_granite_vision",
    },
}


def _make_inputs(
    config, mode, seed, cache_len=64, device="cpu", head_dim_override=None
):
    """Deterministic random inputs for a block_forward call."""
    torch.manual_seed(seed)
    H = config.hidden_size
    head_dim = head_dim_override or (
        getattr(config, "head_dim", None) or H // config.num_attention_heads
    )
    num_kv_heads = config.num_key_value_heads
    half_dim = head_dim // 2

    if mode == "prefill":
        L = SEQ_LEN
        max_cache_len = L
        hidden = torch.randn(1, L, H, dtype=torch.float16).to(device)
        freqs = torch.randn(1, L, 2, 2, half_dim, dtype=torch.float16).to(device)
        mask = torch.zeros(1, 1, L, max_cache_len, dtype=torch.float16)
        for i in range(L):
            mask[:, :, i, i + 1 :] = -torch.inf
        mask = mask.to(device)
        kc = torch.zeros(
            1, num_kv_heads, max_cache_len, head_dim, dtype=torch.float16, device=device
        )
        vc = torch.zeros(
            1, num_kv_heads, max_cache_len, head_dim, dtype=torch.float16, device=device
        )
        cache_pos = 0
    else:
        L = 1
        max_cache_len = cache_len + L
        hidden = torch.randn(1, L, H, dtype=torch.float16).to(device)
        freqs = torch.randn(1, L, 2, 2, half_dim, dtype=torch.float16).to(device)
        mask = torch.zeros(1, 1, L, max_cache_len, dtype=torch.float16)
        mask[:, :, :, cache_len + L :] = -torch.inf
        mask = mask.to(device)
        kc = torch.zeros(
            1, num_kv_heads, max_cache_len, head_dim, dtype=torch.float16, device=device
        )
        kc[:, :, :cache_len, :] = torch.randn(
            1, num_kv_heads, cache_len, head_dim, dtype=torch.float16
        ).to(device)
        vc = torch.zeros(
            1, num_kv_heads, max_cache_len, head_dim, dtype=torch.float16, device=device
        )
        vc[:, :, :cache_len, :] = torch.randn(
            1, num_kv_heads, cache_len, head_dim, dtype=torch.float16
        ).to(device)
        cache_pos = cache_len

    return {
        "hidden_states": hidden,
        "selected_freqs": freqs,
        "attn_mask": mask,
        "key_cache": kc,
        "value_cache": vc,
        "is_filling": False,
        "token_index": 0,
        "cache_position": cache_pos,
    }


@pytest.mark.spyre_block
@pytest.mark.parametrize("model_key", list(MODEL_REGISTRY.keys()))
def test_block_compare(model_key):
    """Per-layer compiled block: CPU (uncompiled) vs Spyre (compiled)."""
    from transformers import AutoModelForCausalLM

    from hf_adapters.hf_common import (
        _move_to_spyre_with_layout,
        _untie_embedding_and_lm_head,
    )

    info = MODEL_REGISTRY[model_key]
    threshold = PER_MODEL_MAX_DIFF.get(model_key, DEFAULT_MAX_DIFF)
    print(
        f"\n  {info['name']}: tiny random-weight model, max_diff threshold {threshold}"
    )

    config = info["config_fn"]()
    print(
        f"  Config: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
        f"heads={config.num_attention_heads}/{config.num_key_value_heads}"
    )

    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_config(config).to(torch.float16)
    model.eval()
    model.requires_grad_(False)

    adapter = importlib.import_module(info["adapter"])
    _untie_embedding_and_lm_head(model)
    adapter.prepare_for_spyre(model)

    num_blocks = len(model._spyre_compiled_blocks)
    padded_hd = getattr(model, "_spyre_head_dim", None)

    # Phase A: CPU runs (uncompiled fn from torch.compile wrapper).
    cpu_results = {}
    for layer_idx in range(num_blocks):
        compiled_block = model._spyre_compiled_blocks[layer_idx]
        uncompiled = getattr(compiled_block, "_orig_mod", compiled_block)
        for mode in ("prefill", "decode"):
            seed = 42 + layer_idx * 100 + (0 if mode == "prefill" else 1)
            inputs = _make_inputs(
                config, mode, seed, device="cpu", head_dim_override=padded_hd
            )
            with torch.no_grad():
                h, _, _ = uncompiled(
                    inputs["hidden_states"],
                    inputs["selected_freqs"],
                    inputs["attn_mask"],
                    inputs["key_cache"],
                    inputs["value_cache"],
                    inputs["is_filling"],
                    inputs["token_index"],
                    inputs["cache_position"],
                )
            cpu_results[(layer_idx, mode)] = h.clone()

    # Phase B: move to Spyre, run compiled blocks.
    _move_to_spyre_with_layout(model, torch.float16)

    failures = []
    for layer_idx in range(num_blocks):
        compiled_block = model._spyre_compiled_blocks[layer_idx]
        for mode in ("prefill", "decode"):
            seed = 42 + layer_idx * 100 + (0 if mode == "prefill" else 1)
            spyre_inputs = _make_inputs(
                config, mode, seed, device=DEVICE, head_dim_override=padded_hd
            )
            try:
                with torch.no_grad():
                    spyre_h, _, _ = compiled_block(
                        spyre_inputs["hidden_states"],
                        spyre_inputs["selected_freqs"],
                        spyre_inputs["attn_mask"],
                        spyre_inputs["key_cache"],
                        spyre_inputs["value_cache"],
                        spyre_inputs["is_filling"],
                        spyre_inputs["token_index"],
                        spyre_inputs["cache_position"],
                    )
            except Exception as e:
                failures.append(f"layer {layer_idx} {mode}: forward raised {e!r}")
                continue

            spyre_h_cpu = spyre_h.to("cpu")
            cpu_h = cpu_results[(layer_idx, mode)]
            diff = (cpu_h - spyre_h_cpu).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            cpu_nan = bool(cpu_h.isnan().any().item())
            sp_nan = bool(spyre_h_cpu.isnan().any().item())
            print(
                f"  layer={layer_idx} mode={mode:7s} max_diff={max_diff:.4f} "
                f"mean_diff={mean_diff:.6f} cpu_nan={cpu_nan} spyre_nan={sp_nan}"
            )
            if cpu_nan:
                failures.append(f"layer {layer_idx} {mode}: CPU output contains NaN")
            if sp_nan:
                failures.append(f"layer {layer_idx} {mode}: Spyre output contains NaN")
            if max_diff >= threshold:
                failures.append(
                    f"layer {layer_idx} {mode}: max_diff {max_diff:.4f} >= "
                    f"threshold {threshold}"
                )

    if failures:
        pytest.fail(f"{model_key}: " + "; ".join(failures))
