"""The causal-LM generate path must prefer self._spyre_run_forward when present.

Tests the free helper _resolve_run_forward_fn(model, module_run_forward) that the
model_generate closure will use to choose the forward callable.
"""

import types


def test_resolve_prefers_compiled_forward_and_host_gathers_freqs():
    import hf_adapters.auto_spyre_model as asm

    called = {}

    # The compiled forward now consumes selected_freqs in place of position_ids.
    def compiled_fw(input_ids, selected_freqs, attn_mask, kc, vc, cache_index):
        called["freqs"] = selected_freqs
        called["input_ids"] = input_ids
        return "logits"

    rope_calls = []

    def rope(first_arg, position_ids):
        rope_calls.append((first_arg, position_ids))
        return "FREQS"

    model = types.SimpleNamespace(_spyre_run_forward=compiled_fw, _spyre_rope=rope)

    def eager(*a, **k):
        raise AssertionError("eager path used despite compiled forward present")

    fn = asm._resolve_run_forward_fn(model, eager)
    # generate() calls fn(model, ...args...): the shim must drop the model arg,
    # host-gather freqs via model._spyre_rope, and pass freqs (not position_ids)
    # to the compiled forward.
    out = fn(model, "ids", "pos", "mask", [], [], 0)

    assert out == "logits"
    assert len(rope_calls) == 1, "host RoPE gather must run exactly once"
    assert rope_calls[0][1] == "pos"
    assert called.get("freqs") == "FREQS", "compiled fw must receive gathered freqs"
    assert called.get("input_ids") == "ids"


def test_resolve_falls_back_to_eager_when_no_compiled():
    import hf_adapters.auto_spyre_model as asm

    model = types.SimpleNamespace()  # no _spyre_run_forward

    def eager(m, *a, **k):
        return ("eager", m)

    fn = asm._resolve_run_forward_fn(model, eager)
    assert fn is eager
