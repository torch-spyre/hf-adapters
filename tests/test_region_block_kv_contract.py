"""nested_region_block must mutate KV in place and drop the cache return.

The whole-forward compile turns each region block into an ``invoke_subgraph``
HOP call. That HOP rejects a subgraph output that aliases a subgraph input, so
the region wrapper must NOT return the KV cache buffers it received — it relies
on in-place mutation instead. The underlying StandardGQABlock keeps its 3-tuple
return for eager (non-region) adapters.
"""


def test_region_wrapper_returns_only_h_and_mutates_kv_in_place():
    import hf_adapters.hf_common as c

    calls = {}

    class FakeBlock:
        # Mimics StandardGQABlock.forward (the method the region dispatches
        # to): mutates the caches in place, returns a 3-tuple.
        def forward(self, h, selected_freqs, attn_mask, kc, vc, cache_index):
            calls["kc_id"] = id(kc)
            calls["vc_id"] = id(vc)
            kc["written"] = True  # in-place mutation of the passed buffer
            vc["written"] = True
            return h + 1, kc, vc

    region = c.nested_region_block(FakeBlock())

    kc = {}
    vc = {}
    out = region(10, "freqs", "mask", kc, vc, 0)

    # Region wrapper exposes only h (NOT the 3-tuple) — no aliasing return.
    assert out == 11, "region wrapper must return only h, not the cache tuple"

    # The in-place mutation on the SAME buffers is visible to the caller.
    assert kc == {"written": True}
    assert vc == {"written": True}
    assert calls["kc_id"] == id(kc), "block must receive the caller's kc buffer"
    assert calls["vc_id"] == id(vc), "block must receive the caller's vc buffer"


def test_standard_gqa_block_still_returns_three_tuple_for_eager_callers():
    """The shared StandardGQABlock contract is untouched: eager adapters
    (olmo, granite_vision_mm, mistral3_vision_mm, standard_gqa_backbone_forward)
    still unpack ``h, key_cache, value_cache``."""
    import inspect

    import hf_adapters.hf_common as c

    src = inspect.getsource(c.StandardGQABlock.forward)
    assert (
        "return h, key_cache, value_cache" in src
    ), "StandardGQABlock must keep its 3-tuple return for eager callers"
    attn_src = inspect.getsource(c.StandardGQAAttention.forward)
    assert (
        "key_cache, value_cache" in attn_src.rsplit("return", 1)[-1]
    ), "StandardGQAAttention must keep its 3-tuple return for eager callers"
