# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generation prefill projects only the final prompt row.

``hf_common.generate`` asks every adapter's prefill backbone for one row
(``rows_to_keep=1``); supported norms select it inside their graph, while legacy
norms normalize all rows and copy the kept row. The LM head gets its own
``[B, 1, H]`` buffer, the same input as in a decode step. The
image-text-to-text prefills (``_prefill_forward(logits_to_keep=1)``) ask their
text backbones the same way.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, GraniteConfig, GraniteForCausalLM

from hf_adapters import auto_spyre_model, hf_common, hf_granite

_HIDDEN = 128
_VOCAB = 129


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("wrapped", [False, True])
@torch.no_grad()
def test_row_request_with_real_compiled_norm(batch_size, wrapped):
    """Both legacy and row-selecting norms execute real full-graph AOT compile."""
    from torch._dynamo.backends.registry import lookup_backend

    torch.manual_seed(23)
    norm = torch.nn.RMSNorm(64, eps=1e-5).eval()
    h = torch.randn(batch_size, 8, 64)
    expected = norm(h)[:, -1:, :]
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return lookup_backend("aot_eager")(graph, inputs)

    fn = hf_common.row_selecting_norm(norm) if wrapped else norm
    compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
    assert getattr(compiled, "_spyre_selects_rows", False) is wrapped
    actual = hf_common.run_final_norm(compiled, h, rows_to_keep=1)
    assert graphs  # No compile identity mock in this regression.
    assert actual.shape == (batch_size, 1, 64)
    assert actual.storage_offset() == 0 and actual.is_contiguous()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hf_common.run_final_norm(compiled, h), norm(h))
    # The wrapped path specializes a one-row graph and a full-row graph. A
    # silently lost capability would use the same full-row graph for both.
    assert len(graphs) == (2 if wrapped else 1)


def _load_through_auto_class(monkeypatch, adapter, model):
    monkeypatch.setattr(
        auto_spyre_model, "resolve_adapter_module", lambda *args, **kwargs: adapter
    )
    monkeypatch.setattr(
        auto_spyre_model.AutoSpyreModel,
        "from_pretrained",
        classmethod(lambda cls, *args, **kwargs: model),
    )
    return auto_spyre_model.AutoSpyreModelForCausalLM.from_pretrained("tiny-model")


def _tiny_granite(monkeypatch):
    """Two-layer Granite with its Spyre adaptation, loaded through the auto class."""
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    config = GraniteConfig(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=2048,
        logits_scaling=2.0,
        pad_token_id=0,
        eos_token_id=None,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = GraniteForCausalLM(config).eval()
    hf_granite.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, torch.float32)
    # 64-position prefill chunks, configured on the model itself.
    model._spyre_prefill_chunk_size = 64
    return _load_through_auto_class(monkeypatch, hf_granite, model)


@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_granite_prefill_head_input_is_its_own_buffer(monkeypatch, batch_size):
    """The kept prefill row reaches the head as its own buffer, as in decode.

    Sliced off the full-row norm output, the batch-1 row is a view at a storage
    offset (``.contiguous()`` keeps it). Torch-Spyre keeps a matmul that reads
    an offset slice on its fixed work division, so that head would not get the
    decode head's plan.
    """
    model = _tiny_granite(monkeypatch)
    torch.manual_seed(0)
    ids = torch.randint(1, _VOCAB, (batch_size, 139))
    head_inputs = []
    model.lm_head.register_forward_pre_hook(
        lambda _module, args: head_inputs.append(
            (tuple(args[0].shape), args[0].storage_offset())
        )
    )
    model.generate(ids, max_new_tokens=3, do_sample=False, eos_token_id=None)
    # One prefill projection, then two decode steps.
    assert head_inputs == [((batch_size, 1, _HIDDEN), 0)] * 3


@torch.no_grad()
def test_granite_generation_normalizes_only_the_kept_row(monkeypatch):
    """Each prefill chunk normalizes one row; tokens and logits are unchanged.

    The full-position forward (``hf_granite._run_forward``) still normalizes
    and projects every row.
    """
    model = _tiny_granite(monkeypatch)
    torch.manual_seed(0)
    ids = torch.randint(1, _VOCAB, (2, 139))
    mask = torch.ones_like(ids)
    mask[0, 71:] = 0  # Mixed-length, right-padded prompts.
    ids[0, 71:] = 0
    options = dict(
        attention_mask=mask,
        max_new_tokens=3,
        do_sample=False,
        eos_token_id=None,
        return_dict_in_generate=True,
        output_logits=True,
    )
    head_rows, norm_rows = [], []
    model.lm_head.register_forward_pre_hook(
        lambda _module, args: head_rows.append(args[0].shape[1])
    )
    model.model.norm.register_forward_pre_hook(
        lambda _module, args: norm_rows.append(args[0].shape[1])
    )
    full = hf_common.generate(hf_granite._run_forward, model, ids, **options)
    # Three 64-position prefill chunks, then two decode steps.
    assert norm_rows == head_rows == [64, 64, 64, 1, 1]
    head_rows.clear()
    norm_rows.clear()
    actual = model.generate(ids, **options)
    assert norm_rows == [1, 1, 1, 1, 1]
    assert head_rows == [1, 1, 1]
    torch.testing.assert_close(actual.sequences, full.sequences)
    assert len(actual.logits) == len(full.logits) == 3
    for got, want in zip(actual.logits, full.logits, strict=True):
        torch.testing.assert_close(got, want)


# One tiny model per adapter module whose prefill backbone or final-norm compile
# gained the row selection (every module on ``generate``'s prefill-backbone
# path that owns one of the two): (config, final-norm spy). A spy is the norm
# module's path, or a function that records the rows a functional norm gets.
def _spy_qwen3_5_norm(monkeypatch, model, rows):
    from hf_adapters import hf_qwen3_5

    final_norm = model.model.norm
    rms_norm = hf_qwen3_5._rms_norm

    def recording_rms_norm(x, norm):
        if norm is final_norm:
            rows.append(x.shape[1])
        return rms_norm(x, norm)

    monkeypatch.setattr(hf_qwen3_5, "_rms_norm", recording_rms_norm)


def _spy_gemma4_norm(monkeypatch, model, rows):
    from hf_adapters import hf_gemma4

    def recording_rms_norm(hidden_states, weight, eps):
        rows.append(hidden_states.shape[1])
        return hf_gemma4._gemma4_rms_norm(hidden_states, weight, eps)

    # Preserve the bare production norm's full-row fallback contract.
    monkeypatch.setattr(
        hf_gemma4,
        "_compiled_gemma4_rms_norm",
        recording_rms_norm,
    )


_ATTN = dict(
    hidden_size=_HIDDEN,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    max_position_embeddings=2048,
    vocab_size=_VOCAB,
    pad_token_id=0,
    eos_token_id=None,
)
_WIDE = dict(_ATTN, hidden_size=256, intermediate_size=512)
_FAMILIES = {
    "hf_granite": (lambda: GraniteConfig(logits_scaling=2.0, **_ATTN), "model.norm"),
    # hf_granite_vision prepares the Granite text decoder of its VLM.
    "hf_granite_vision": (
        lambda: GraniteConfig(logits_scaling=2.0, **_ATTN),
        "model.norm",
    ),
    "hf_granite_swa": (
        lambda: transformers.GraniteSWAConfig(
            logits_scaling=2.0, sliding_window=64, **_ATTN
        ),
        "model.norm",
    ),
    "hf_granitemoehybrid": (
        lambda: transformers.GraniteMoeHybridConfig(
            layer_types=["attention", "attention"],
            position_embedding_type="rope",
            logits_scaling=2.0,
            **_ATTN,
        ),
        "model.norm",
    ),
    # standard_gqa_backbone_forward with prepare_standard_gqa's norm (Llama,
    # Mistral, Ministral, Mistral 3, Qwen2, BharatGen) or the adapter's own.
    "hf_llama": (lambda: transformers.LlamaConfig(**_ATTN), "model.norm"),
    "hf_qwen3": (lambda: transformers.Qwen3Config(head_dim=64, **_ATTN), "model.norm"),
    "hf_olmo": (lambda: transformers.OlmoConfig(**_ATTN), "model.norm"),
    "hf_olmo2": (
        lambda: transformers.Olmo2Config(**dict(_WIDE, num_key_value_heads=2)),
        "model.norm",
    ),
    "hf_olmoe": (
        lambda: transformers.OlmoeConfig(
            num_experts=4,
            num_experts_per_tok=2,
            **dict(_WIDE, num_key_value_heads=2),
        ),
        "model.norm",
    ),
    "hf_smollm3": (lambda: transformers.SmolLM3Config(**_ATTN), "model.norm"),
    "hf_phi3": (lambda: transformers.Phi3Config(**_ATTN), "model.norm"),
    "hf_lfm2": (
        lambda: transformers.Lfm2Config(
            layer_types=["conv", "full_attention"], **_ATTN
        ),
        "model.embedding_norm",
    ),
    "hf_qwen3_5": (
        lambda: transformers.Qwen3_5TextConfig(
            head_dim=256, layer_types=["linear_attention", "full_attention"], **_ATTN
        ),
        _spy_qwen3_5_norm,
    ),
    "hf_gemma2": (
        lambda: transformers.Gemma2Config(head_dim=128, sliding_window=128, **_WIDE),
        "model.norm",
    ),
    "hf_gemma3": (
        lambda: transformers.Gemma3TextConfig(
            head_dim=128, sliding_window=128, **_WIDE
        ),
        "model.norm",
    ),
    "hf_gemma4": (
        lambda: transformers.Gemma4TextConfig(head_dim=128, **_WIDE),
        _spy_gemma4_norm,
    ),
    "hf_gpt2": (
        lambda: transformers.GPT2Config(
            vocab_size=_VOCAB,
            n_embd=_HIDDEN,
            n_head=2,
            n_layer=2,
            n_positions=2048,
            pad_token_id=0,
            eos_token_id=None,
        ),
        "transformer.ln_f",
    ),
    "hf_gpt_neo": (
        lambda: transformers.GPTNeoConfig(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            num_layers=2,
            num_heads=2,
            attention_types=[[["global", "local"], 1]],
            max_position_embeddings=2048,
            window_size=256,
            pad_token_id=0,
            eos_token_id=None,
        ),
        "transformer.ln_f",
    ),
    "hf_gpt_neox": (
        lambda: transformers.GPTNeoXConfig(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=None,
        ),
        "gpt_neox.final_layer_norm",
    ),
    "hf_opt": (
        lambda: transformers.OPTConfig(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            ffn_dim=256,
            word_embed_proj_dim=_HIDDEN,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=2048,
            pad_token_id=0,
            eos_token_id=None,
        ),
        "model.decoder.final_layer_norm",
    ),
}


@pytest.mark.parametrize(
    "adapter_name,dtype",
    [(name, torch.float32) for name in sorted(_FAMILIES)]
    + [
        (name, dtype)
        for name in ("hf_gemma2", "hf_gemma3")
        for dtype in (torch.float16, torch.bfloat16)
    ],
)
@torch.no_grad()
def test_prefill_backbone_returns_the_projected_row(monkeypatch, adapter_name, dtype):
    """Each family's prefill backbone returns the last row from its final norm.

    LayerNorm families and Gemma 2/3/4 retain their full-row norm; other
    families normalize only the requested row. The LM head reads its own buffer
    (storage offset 0), and tokens and logits equal the
    full-position forward's, at batch 1 and for mixed-length batch-2 prompts.
    FP16/BF16 cases check the CPU row/buffer contract, not the Spyre norm branch.
    """
    make_config, norm_spy = _FAMILIES[adapter_name]
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    adapter = importlib.import_module(f"hf_adapters.{adapter_name}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = AutoModelForCausalLM.from_config(make_config()).eval().to(dtype)
    adapter.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, dtype)
    model._spyre_prefill_chunk_size = 64
    model = _load_through_auto_class(monkeypatch, adapter, model)

    head_inputs, norm_rows = [], []
    model.lm_head.register_forward_pre_hook(
        lambda _module, args: head_inputs.append(
            (tuple(args[0].shape), args[0].storage_offset())
        )
    )
    if isinstance(norm_spy, str):
        model.get_submodule(norm_spy).register_forward_pre_hook(
            lambda _module, args: norm_rows.append(args[0].shape[1])
        )
    else:
        norm_spy(monkeypatch, model, norm_rows)
    hidden_size = hf_common.text_config(model.config).hidden_size
    torch.manual_seed(0)
    for batch_size in (1, 2):
        ids = torch.randint(1, _VOCAB, (batch_size, 139))
        mask = torch.ones_like(ids)
        if batch_size > 1:
            mask[0, 71:] = 0  # Mixed-length, right-padded prompts.
            ids[0, 71:] = 0
        options = dict(
            attention_mask=mask,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        full = hf_common.generate(adapter._run_forward, model, ids, **options)
        norm_rows.clear()
        head_inputs.clear()
        actual = model.generate(ids, **options)
        # Three prefill chunks, then two decode steps. Legacy norms keep the
        # full-row graph, with selection/copy afterward in run_final_norm.
        expected_norm_rows = (
            [64, 64, 64, 1, 1]
            if adapter_name
            in (
                "hf_gpt2",
                "hf_gpt_neo",
                "hf_gpt_neox",
                "hf_opt",
                "hf_olmo",
                "hf_gemma2",
                "hf_gemma3",
                "hf_gemma4",
            )
            else [1, 1, 1, 1, 1]
        )
        assert norm_rows == expected_norm_rows
        assert head_inputs == [((batch_size, 1, hidden_size), 0)] * 3
        torch.testing.assert_close(actual.sequences, full.sequences)
        for got, want in zip(actual.logits, full.logits, strict=True):
            torch.testing.assert_close(got, want)


@pytest.mark.parametrize(
    "adapter_name", ["hf_granite", "hf_lfm2", "hf_llama", "hf_phi3"]
)
@torch.no_grad()
def test_full_position_forward_accepts_a_plain_final_norm(monkeypatch, adapter_name):
    """Only the generation prefill passes a row count to the final norm.

    Decode and the full-position forward call it exactly as before, so any
    final-norm callable still works there, e.g. the bare norm module installed
    by a caller that builds its own model namespace.
    """
    make_config, norm_path = _FAMILIES[adapter_name]
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    adapter = importlib.import_module(f"hf_adapters.{adapter_name}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        model = AutoModelForCausalLM.from_config(make_config()).eval()
    adapter.prepare_for_spyre(model)
    hf_common.set_rope_dtype(model, torch.float32)
    ids = torch.randint(1, _VOCAB, (1, 70))
    options = dict(max_new_tokens=2, do_sample=False, eos_token_id=None)
    expected = hf_common.generate(adapter._run_forward, model, ids, **options)
    model._spyre_compiled_norm = model.get_submodule(norm_path)
    actual = hf_common.generate(adapter._run_forward, model, ids, **options)
    torch.testing.assert_close(actual, expected)


# #600 also gave the image-text-to-text prefills a one-row head: auto_spyre_model
# calls each adapter's ``_prefill_forward`` with ``logits_to_keep=1``. Their text
# backbones take the same row request, from ``_logits_from_embeds``.
_VLM_ADAPTERS = sorted(
    set(auto_spyre_model.IMAGE_TEXT_TO_TEXT_CONFIG_TO_ADAPTER_MODULE_MAPPING.values()),
    key=lambda module: module.__name__,
)


@pytest.mark.parametrize("adapter", _VLM_ADAPTERS, ids=lambda module: module.__name__)
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_multimodal_prefill_asks_the_backbone_for_the_projected_row(
    monkeypatch, adapter, batch_size
):
    """``logits_to_keep=1`` asks the text backbone for one row, so the LM head
    reads its own one-row buffer; the default still projects every row."""
    torch.manual_seed(5)
    ids = torch.arange(1, 9).expand(batch_size, -1).clone()
    hidden = torch.randn(batch_size, ids.shape[1], 64)
    final_norm = hf_common.row_selecting_norm(torch.nn.RMSNorm(64, eps=1e-5))
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(64, 129, bias=False)
    model.config = SimpleNamespace(
        enable_moe_block=True, use_bidirectional_attention=None
    )
    model._spyre_has_ple = False
    requests, head_inputs = [], []

    def backbone(model, inputs_embeds, *args, rows_to_keep=0, **kwargs):
        # The backbone contract: every position runs through the blocks, and
        # the final norm returns only the requested trailing rows.
        requests.append(rows_to_keep)
        return hf_common.run_final_norm(
            final_norm, inputs_embeds * 2, rows_to_keep=rows_to_keep
        )

    def head(value):
        head_inputs.append((tuple(value.shape), value.storage_offset()))
        return model.lm_head(value)

    model._spyre_lm_head_forward = head
    kwargs = dict(
        model=model,
        input_ids=ids,
        position_ids=ids - 1,
        attention_mask=torch.zeros(batch_size, 1, 8, 8),
        key_caches=[],
        value_caches=[],
        cache_index=torch.arange(8),
        pixel_values=None,
    )
    if adapter.__name__.endswith("hf_gemma4_mm"):
        monkeypatch.setattr(adapter, "_image_features", lambda *args: None)
        monkeypatch.setattr(
            adapter, "_embed_and_scatter", lambda *args: (hidden, None, ids)
        )
        monkeypatch.setattr(adapter.hf_gemma4, "_run_blocks_over_embeds", backbone)
        kwargs.update(image_position_ids=None, mm_token_type_ids=torch.zeros_like(ids))
    else:
        monkeypatch.setattr(adapter, "embed_text_tokens", lambda *args: hidden)
        mask_shape = (
            (batch_size, 8, 1)
            if adapter.__name__.endswith("hf_granite_vision_mm")
            else (batch_size, 8)
        )
        monkeypatch.setattr(
            adapter,
            "_vision_mask",
            lambda *args: torch.zeros(mask_shape, dtype=torch.bool),
        )
        feature_fn = (
            "_deepstack_features"
            if adapter.__name__.endswith("hf_granite_vision_mm")
            else "_image_features"
        )
        monkeypatch.setattr(adapter, feature_fn, lambda *args: None)
        monkeypatch.setattr(adapter, "_run_text_backbone", backbone)
        kwargs["image_sizes"] = None

    reference = adapter._prefill_forward(**kwargs)
    actual = adapter._prefill_forward(**kwargs, logits_to_keep=1)
    assert requests == [0, 1]
    assert head_inputs == [((batch_size, 8, 64), 0), ((batch_size, 1, 64), 0)]
    torch.testing.assert_close(actual, reference[:, -1:, :])


@pytest.mark.parametrize(
    "adapter_name, vision_tower",
    [
        ("hf_granite_vision_mm", "hf_siglip_vision"),
        ("hf_mistral3_vision_mm", "hf_pixtral_vision"),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2])
@torch.no_grad()
def test_multimodal_text_backbone_normalizes_only_the_kept_row(
    monkeypatch, adapter_name, vision_tower, batch_size
):
    """The VLM's prepared final norm selects the kept rows inside its graph.

    ``prepare_for_spyre`` compiles ``row_selecting_norm(norm)``, so the text
    backbone normalizes one row into its own buffer when asked, and every row
    otherwise. Without a row request (decode, full-sequence logits), a plain
    norm callable still works.
    """
    adapter = importlib.import_module(f"hf_adapters.{adapter_name}")
    monkeypatch.setattr(torch, "compile", lambda fn, **_kwargs: fn)
    # Only the final-norm preparation is under test.
    monkeypatch.setattr(
        getattr(adapter, vision_tower), "prepare_for_spyre", lambda model: None
    )
    monkeypatch.setattr(adapter, "prepare_rope_and_heads", lambda model: None)
    monkeypatch.setattr(
        adapter, "prepare_lm_head_for_spyre", lambda model, **kwargs: None
    )
    monkeypatch.setattr(
        adapter, "prepare_standard_gqa_blocks", lambda layers, *args: []
    )
    torch.manual_seed(3)
    norm = torch.nn.RMSNorm(64, eps=1e-5)
    torch.nn.init.normal_(norm.weight)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.norm = norm
    model.model.language_model.layers = torch.nn.ModuleList()
    model.config = SimpleNamespace(text_config=SimpleNamespace(logits_scaling=2.0))
    adapter.prepare_for_spyre(model)

    model._spyre_rope = lambda h, position_ids: None
    model._spyre_text_blocks = [
        lambda h, freqs, mask, k, v, index: (h * 1.5 + 0.25, k, v)
    ]
    norm_rows = []
    norm.register_forward_pre_hook(
        lambda _module, args: norm_rows.append(args[0].shape[1])
    )
    h = torch.randn(batch_size, 8, 64)
    args = (model, h, None, None, [None], [None], None)

    full = adapter._run_text_backbone(*args)
    kept = adapter._run_text_backbone(*args, rows_to_keep=1)
    assert norm_rows == [8, 1]
    assert kept.shape == (batch_size, 1, 64)
    assert kept.storage_offset() == 0 and kept.is_contiguous()
    torch.testing.assert_close(kept, full[:, -1:, :])

    model._spyre_compiled_norm = norm
    torch.testing.assert_close(adapter._run_text_backbone(*args), full)
