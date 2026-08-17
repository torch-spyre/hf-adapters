import torch
from transformers import AutoConfig, AutoModelForCausalLM

from hf_adapters import hf_gemma4

E2B = "google/gemma-4-E2B-it"


def test_producer_map_e2b():
    cfg = AutoConfig.from_pretrained(E2B)
    tc = hf_gemma4.text_config(cfg)
    first, producer_of = hf_gemma4._shared_producer_map(tc)
    assert first == tc.num_hidden_layers - tc.num_kv_shared_layers  # 35-20=15
    # Non-shared layers map to None.
    assert all(producer_of[i] is None for i in range(first))
    # Each shared layer maps to a real earlier layer of the SAME type.
    for i in range(first, tc.num_hidden_layers):
        p = producer_of[i]
        assert p is not None and p < first
        assert tc.layer_types[p] == tc.layer_types[i]


def test_shared_block_runs_and_is_finite():
    # Structural check on the lean KV-share path: a shared block has no k_proj,
    # reads the producer's cache with the resolved per-layer head_dim, and
    # returns a well-shaped finite output. The end-to-end numerical equivalence
    # of KV-sharing is covered by the E2B token-parity test
    # (tests/cpu/test_adapter_cpu_accuracy.py), not here.
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    backbone = hf_gemma4._gemma4_backbone(model)
    cfg = hf_gemma4.text_config(model.config)
    first, producer_of = hf_gemma4._shared_producer_map(cfg)
    shared_idx = first  # first shared layer
    layer = backbone.layers[shared_idx]
    hf_gemma4._patch_gemma4_rmsnorm(type(layer.input_layernorm))

    # hasattr must be checked first: the real model doesn't set k_proj at all
    # on shared layers (rather than setting it to None), so accessing the
    # attribute directly raises AttributeError before an `or` could fall back.
    assert not hasattr(layer.self_attn, "k_proj") or layer.self_attn.k_proj is None

    # cfg.head_dim is ambiguous on this heterogeneous config (global layers
    # override head_dim=512 vs the sliding default 256); read the resolved
    # per-layer value instead, as the AmbiguousGlobalPerLayerAttributeError
    # message itself directs (config.per_layer_config[i].head_dim).
    layer_head_dim = cfg.per_layer_config[shared_idx].head_dim

    # Instantiate the shared-block module directly (uncompiled) for a pure-math
    # check. has_ple=False so no per_layer_input is read.
    block = hf_gemma4.Gemma4SharedBlock(
        layer, cfg.num_attention_heads, layer_head_dim, has_ple=False
    )

    B, S, H = 1, 3, cfg.hidden_size
    hd, nkv = layer_head_dim, cfg.num_key_value_heads
    Lc = 8
    h = torch.randn(B, S, H)
    # apply_rope_matmul's real contract is [B, L, 2, 2, D/2] (see
    # hf_common.apply_rope_matmul / PrecomputedRotaryEmbedding docstrings),
    # not the [1, S, D/2, 2] shape the brief sketched.
    freqs = torch.randn(B, S, 2, 2, hd // 2)
    mask = torch.zeros(B, 1, S, Lc)
    kcache = torch.randn(B, nkv, Lc, hd)
    vcache = torch.randn(B, nkv, Lc, hd)
    scalar = block.layer_scalar
    out = block(h, freqs, mask, kcache, vcache, scalar, per_layer_input=None)
    assert out.shape == (B, S, H)
    assert torch.isfinite(out).all()


def test_e2b_forward_runs_and_is_finite():
    from hf_adapters.hf_common import allocate_kv_caches, make_cache_index

    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float32)
    hf_gemma4.prepare_for_spyre(model)  # DEVICE is patched to "cpu" by tests/conftest.py
    assert hasattr(model, "_spyre_producer_of")
    assert hasattr(model, "_spyre_has_ple") and model._spyre_has_ple

    input_ids = torch.tensor([[2, 10, 20, 30, 40]])
    S = input_ids.shape[1]
    Lc = S + 2
    kc, vc = allocate_kv_caches(model, 1, Lc, torch.float32, device="cpu")
    pos = torch.arange(S).unsqueeze(0)
    mask = torch.zeros(1, 1, S, Lc)
    for i in range(S):
        mask[:, :, i, i + 1 :] = -torch.inf
    # Prefill writes the S query rows into cache slots [0, S).
    cache_index = make_cache_index(0, S, "cpu")
    with torch.no_grad():
        logits = hf_gemma4._run_forward(
            model, input_ids, pos, mask, kc, vc, cache_index,
        )
    assert logits.shape[1] == S
    assert torch.isfinite(logits[0, -1]).all()


def test_e2b_block_padded_prefill_logits_finite():
    """Regression: block-padded (left-pad) prefill must yield finite real-token
    logits in fp16.

    ``generate`` left-pads a short prompt to a BLOCK_SIZE multiple, so the
    leading pad rows attend to no key (their whole mask row is ``-inf``). Before
    the fix, the ``<pad>`` embedding overflowed the fp16 MLP to ``+inf`` and the
    sandwich ``post_feedforward_layernorm`` turned it into ``NaN``, which then
    poisoned the KV cache at the pad slots and spread to every real-token logit
    (``NaN + (-inf) = NaN``) — ``generate`` emitted all ``<pad>``. fp16 is
    required to reproduce the overflow (fp32 has the range to avoid it). See
    ``hf_gemma4._zero_fully_masked_rows``.
    """
    import hf_adapters.hf_common as hf_common

    # fp16 is load-bearing here: the overflow that seeds the NaN only happens in
    # the half-precision MLP; fp32 would silently pass.
    model = AutoModelForCausalLM.from_pretrained(E2B, dtype=torch.float16)
    hf_gemma4.prepare_for_spyre(model)  # DEVICE patched to "cpu" by conftest

    # Mirror hf_common.generate's prefill: left block-pad a 5-token prompt to 64.
    input_ids = torch.tensor([[818, 5279, 529, 7001, 563]])
    actual_lengths = torch.tensor([input_ids.shape[1]])
    max_cache_len = hf_common.generation_cache_len(input_ids.shape[1], 4)
    padded_ids, padded_len, prompt_offsets, position_ids = hf_common.pad_and_position(
        input_ids, actual_lengths
    )
    assert padded_len == 64 and prompt_offsets.tolist() == [59]  # 59 pad rows
    prefill_mask = hf_common.build_prefill_mask(
        1, padded_len, max_cache_len, prompt_offsets, dtype=torch.float16
    )
    kc, vc = hf_common.allocate_kv_caches(
        model, 1, max_cache_len, torch.float16, device="cpu"
    )
    # Prefill writes the whole padded block into cache slots [0, padded_len).
    cache_index = hf_common.make_cache_index(0, padded_len, "cpu")
    with torch.no_grad():
        logits = hf_gemma4._run_forward(
            model, padded_ids, position_ids, prefill_mask, kc, vc, cache_index,
        )
    # The only rows that matter are the real tokens (offset 59..63); their
    # logits must be finite for argmax/generation to work.
    real_logits = logits[0, prompt_offsets[0].item():]
    assert torch.isfinite(real_logits).all(), "block-padded prefill produced non-finite real-token logits"
