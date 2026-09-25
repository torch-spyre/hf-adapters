"""prepare_for_spyre must attach a compiled whole-forward callable and region blocks."""

import types


def test_prepare_attaches_compiled_run_forward(monkeypatch):
    import hf_adapters.hf_granite as g

    created = {}

    def fake_region_blocks(layers, is_res_mul=None):
        created["region_blocks"] = True
        return ["blk0", "blk1"]

    def fake_compile(fn, **kw):
        created["compiled"] = True
        created["dynamic"] = kw.get("dynamic")
        return fn

    # Stub the heavy prep pieces so this stays a CPU wiring test.
    monkeypatch.setattr(
        g, "prepare_standard_gqa_region_blocks", fake_region_blocks, raising=False
    )
    monkeypatch.setattr(g, "prepare_rope_and_heads", lambda m: None, raising=False)
    monkeypatch.setattr(g, "pad_lm_head", lambda m: None, raising=False)
    monkeypatch.setattr(g.torch, "compile", fake_compile, raising=False)

    model = types.SimpleNamespace()
    model.config = types.SimpleNamespace()
    # get_backbone(model).layers and .norm must exist; stub via monkeypatch.
    backbone = types.SimpleNamespace(layers=[object(), object()], norm=object())
    monkeypatch.setattr(g, "get_backbone", lambda m: backbone, raising=False)

    g.prepare_for_spyre(model)

    assert created.get("region_blocks") is True
    assert created.get("compiled") is True
    assert created.get("dynamic") is False
    assert model._spyre_compiled_blocks == ["blk0", "blk1"]
    assert callable(model._spyre_run_forward)


def test_compiled_forward_consumes_freqs_not_position_ids(monkeypatch):
    """The compiled whole-forward must take ``selected_freqs`` as an input and
    NOT invoke the host-side RoPE gather (``model._spyre_rope``) inside the graph.
    That gather (``.item()`` + CPU fancy-index) is not traceable on Spyre."""
    import hf_adapters.hf_granite as g

    # Capture the fn passed to torch.compile so we can inspect its parameters and
    # call it, verifying it consumes freqs and never touches model._spyre_rope.
    captured = {}

    def fake_compile(fn, **kw):
        captured["fn"] = fn
        return fn

    monkeypatch.setattr(g.torch, "compile", fake_compile, raising=False)

    # Stub the backbone/head/norm so _run_forward_freqs runs on plain Python objects.
    def embed_tokens(input_ids):
        return input_ids  # identity — h just flows through

    backbone = types.SimpleNamespace(
        embed_tokens=embed_tokens,
        embedding_multiplier=1,
        norm=lambda h: h,
    )
    monkeypatch.setattr(g, "get_backbone", lambda m: backbone, raising=False)
    monkeypatch.setattr(
        g,
        "text_config",
        lambda cfg: types.SimpleNamespace(logits_scaling=1),
        raising=False,
    )

    def rope_should_not_be_called(*a, **k):
        raise AssertionError("model._spyre_rope was called INSIDE the compiled forward")

    seen_block_freqs = {}

    # _spyre_compiled_blocks entries are region wrappers (nested_region_block),
    # which mutate KV in place and return ONLY h (the aliasing fix).
    def block(h, selected_freqs, attn_mask, kc, vc, cache_index):
        seen_block_freqs["freqs"] = selected_freqs
        return h

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(),
        lm_head=lambda h: 42,  # numeric so the /logits_scaling epilogue is valid
        _spyre_rope=rope_should_not_be_called,
        _spyre_compiled_blocks=[block],
        _spyre_compiled_norm=lambda h: h,
    )

    compiled = g._make_compiled_run_forward(model)
    fn = captured["fn"]

    # Second positional parameter of the compiled callable is selected_freqs, not position_ids.
    import inspect

    params = list(inspect.signature(fn).parameters)
    assert params[0] == "input_ids"
    assert params[1] == "selected_freqs"
    assert "position_ids" not in params

    # Calling it with a freqs sentinel must reach the block WITHOUT calling _spyre_rope.
    sentinel_freqs = object()
    key_caches = [None]
    value_caches = [None]
    compiled(
        "ids",
        sentinel_freqs,
        "mask",
        key_caches,
        value_caches,
        0,
    )
    assert seen_block_freqs["freqs"] is sentinel_freqs


def test_eager_run_forward_host_gathers_freqs_once(monkeypatch):
    """The eager (position_ids based) entry must call ``model._spyre_rope`` exactly
    once on the host and delegate to the freqs-based body."""
    import hf_adapters.hf_granite as g

    backbone = types.SimpleNamespace(
        embed_tokens=lambda input_ids: input_ids,
        embedding_multiplier=1,
        norm=lambda h: h,
    )
    monkeypatch.setattr(g, "get_backbone", lambda m: backbone, raising=False)
    monkeypatch.setattr(
        g,
        "text_config",
        lambda cfg: types.SimpleNamespace(logits_scaling=1),
        raising=False,
    )

    rope_calls = []

    def rope(first_arg, position_ids):
        rope_calls.append((first_arg, position_ids))
        return "FREQS"

    seen = {}

    # region-wrapper contract: mutate KV in place, return only h
    def block(h, selected_freqs, attn_mask, kc, vc, cache_index):
        seen["freqs"] = selected_freqs
        return h

    model = types.SimpleNamespace(
        config=types.SimpleNamespace(),
        lm_head=lambda h: 42,  # numeric so the /logits_scaling epilogue is valid
        _spyre_rope=rope,
        _spyre_compiled_blocks=[block],
        _spyre_compiled_norm=lambda h: h,
    )

    key_caches = [None]
    value_caches = [None]
    g._run_forward(
        model,
        "ids",
        "posids",
        "mask",
        key_caches,
        value_caches,
        0,
    )

    assert len(rope_calls) == 1, "host RoPE gather must run exactly once"
    assert rope_calls[0][1] == "posids"
    assert seen["freqs"] == "FREQS", "gathered freqs must flow into the blocks"
