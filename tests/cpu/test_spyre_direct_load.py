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

"""Host-only tests for ``hf_adapters.spyre_direct_load``.

The DMA helpers need hardware, so these substitute a recorder for
``_copy_linear`` / ``_copy_embedding`` / ``_copy_default`` and assert on which
one each parameter was routed to, which is the decision this module owns. The
end-to-end check that the weights land correctly lives in
``tests/spyre/test_direct_load_compare_spyre.py``.
"""

import os
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import core_model_loading

from hf_adapters import hf_granite, hf_llama
from hf_adapters.spyre_direct_load import (
    _host_module_prefixes,
    _placement_copy,
    _should_place,
    _stays_on_host,
    _tied_to_host,
    spyre_direct_placement,
    supports_direct_load,
)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(64, 16)
        self.model.norm = nn.LayerNorm(16)
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].q_proj = nn.Linear(16, 32, bias=True)
        self.lm_head = nn.Linear(16, 64, bias=False)
        self.register_buffer("token_type_ids", torch.zeros(8, dtype=torch.long))


@pytest.fixture
def record_copies(monkeypatch):
    """Replace the three DMA helpers with recorders.

    Returns the list of ``(helper, shape, dtype)`` triples, in call order.
    """
    calls = []

    def _recorder(name):
        def _copy(shard, *, device, dtype):
            calls.append((name, tuple(shard.shape), dtype))
            # Stand in for a device tensor: the caller only forwards it on.
            return shard.to(dtype) if dtype is not None else shard

        return _copy

    import hf_adapters.spyre_tensor_parallel as tp

    monkeypatch.setattr(tp, "_copy_linear", _recorder("linear"))
    monkeypatch.setattr(tp, "_copy_embedding", _recorder("embedding"))
    monkeypatch.setattr(tp, "_copy_default", _recorder("default"))
    return calls


# ── layout decision ────────────────────────────────────────────────────


def test_linear_weight_routes_to_linear_copy(record_copies):
    module = nn.Linear(16, 32, bias=True)
    _placement_copy(
        module,
        "weight",
        torch.randn(32, 16),
        device=torch.device("spyre", 0),
        dtype=torch.float16,
    )
    assert record_copies == [("linear", (32, 16), torch.float16)]


def test_linear_bias_routes_to_default_copy(record_copies):
    module = nn.Linear(16, 32, bias=True)
    _placement_copy(
        module,
        "bias",
        torch.randn(32),
        device=torch.device("spyre", 0),
        dtype=torch.float16,
    )
    assert record_copies == [("default", (32,), torch.float16)]


def test_embedding_weight_routes_to_embedding_copy(record_copies):
    module = nn.Embedding(64, 16)
    _placement_copy(
        module,
        "weight",
        torch.randn(64, 16),
        device=torch.device("spyre", 0),
        dtype=torch.float16,
    )
    assert record_copies == [("embedding", (64, 16), torch.float16)]


def test_layernorm_weight_routes_to_default_copy(record_copies):
    """``weight`` alone must not mean Linear: norms take the default layout."""
    module = nn.LayerNorm(16)
    _placement_copy(
        module,
        "weight",
        torch.randn(16),
        device=torch.device("spyre", 0),
        dtype=torch.float16,
    )
    assert record_copies == [("default", (16,), torch.float16)]


def test_integer_buffer_keeps_its_dtype(record_copies):
    """A non-floating tensor must not be handed the model's float dtype."""
    module = nn.Module()
    _placement_copy(
        module,
        "token_type_ids",
        torch.zeros(8, dtype=torch.long),
        device=torch.device("spyre", 0),
        dtype=torch.float16,
    )
    assert record_copies == [("default", (8,), None)]


# ── when the wrapper declines ───────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs, reason",
    [
        ({"hf_quantizer": object()}, "a quantizer owns the placement"),
        ({"param_value": torch.zeros(4, device="meta")}, "not on the host"),
    ],
)
def test_should_place_declines(kwargs, reason):
    args = {
        "target_name": "model.layers.0.q_proj.weight",
        "param_value": torch.zeros(4),
        "hf_quantizer": None,
        "host_prefixes": (),
    }
    args.update(kwargs)
    assert not _should_place(**args), reason


def test_should_place_accepts_a_plain_host_tensor():
    assert _should_place(
        target_name="model.layers.0.q_proj.weight",
        param_value=torch.zeros(4),
        hf_quantizer=None,
        host_prefixes=(),
    )


@pytest.mark.parametrize(
    "target_name, expected",
    [
        ("lm_head.weight", True),
        ("lm_head", True),
        ("lm_head_extra.weight", False),
        ("model.lm_head.weight", False),
        ("model.layers.0.q_proj.weight", False),
    ],
)
def test_stays_on_host_matches_on_path_boundaries(target_name, expected):
    assert _stays_on_host(target_name, ("lm_head",)) is expected


# ── weights tied to a host-resident module ─────────────────────────────


def _tiny_config(*, tied):
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        tie_word_embeddings=tied,
    )


def _tiny_causal_lm(*, tied, on_meta=False):
    """A tiny Llama, optionally built the way ``from_pretrained`` builds one.

    ``on_meta=True`` matters for the tie tests: the loader constructs the
    skeleton on ``meta``, and a tied head is still its meta placeholder while
    the funnel is live.
    """
    from transformers import AutoModelForCausalLM

    config = _tiny_config(tied=tied)
    torch.manual_seed(0)
    if on_meta:
        with torch.device("meta"):
            return AutoModelForCausalLM.from_config(config)
    return AutoModelForCausalLM.from_config(config)


def test_a_tied_head_holds_back_its_embedding():
    """The source of a tie to a host module must not be placed either.

    ``lm_head.weight`` is not a checkpoint key on a tied model, so it never
    reaches the funnel and ``_stays_on_host`` is never asked about it. Only
    ``model.embed_tokens.weight`` is offered -- and it becomes the head weight.
    """
    model = _tiny_causal_lm(tied=True, on_meta=True)
    assert _tied_to_host(model, "model.embed_tokens.weight", ("lm_head",))


def test_the_tie_is_not_yet_observable_by_identity_during_loading():
    """Why the check cannot simply compare the two parameters.

    Transformers ties the head *after* the weights land. While the funnel is
    live the tied parameter is still a meta placeholder, so an identity test
    would answer False for every tied model and the guard would never fire.
    """
    model = _tiny_causal_lm(tied=True, on_meta=False)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    loading = _tiny_causal_lm(tied=True, on_meta=True)
    assert loading.lm_head.weight.device.type == "meta"


def test_an_untied_head_does_not_hold_back_its_embedding():
    """``_tied_weights_keys`` advertises the pair even when untied.

    So the mapping alone cannot decide it, or every untied model would lose
    its embedding from the direct path.
    """
    model = _tiny_causal_lm(tied=False, on_meta=True)
    assert "lm_head.weight" in model._tied_weights_keys
    assert not _tied_to_host(model, "model.embed_tokens.weight", ("lm_head",))


def test_no_host_modules_means_no_holding_back():
    model = _tiny_causal_lm(tied=True, on_meta=True)
    assert not _tied_to_host(model, "model.embed_tokens.weight", ())


def test_a_tied_embedding_survives_a_real_load_on_the_host(record_copies, tmp_path):
    """End to end: a tied checkpoint must leave the shared weight on the host.

    Without this, ``untie_embedding_and_lm_head`` reads the head back off the
    device -- a round trip fp16 does not survive intact.
    """
    from transformers import AutoModelForCausalLM

    path = tmp_path / "tied-llama"
    _tiny_causal_lm(tied=True).save_pretrained(path)

    with spyre_direct_placement(
        device="cpu", dtype=torch.float16, host_modules=("lm_head",)
    ):
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, device_map="cpu"
        )

    assert record_copies, "the loader never reached the placement wrapper"
    helpers = [name for name, _shape, _dtype in record_copies]
    assert (
        "embedding" not in helpers
    ), "the tied embedding was placed on the device; untying would read it back"
    # The rest of the model still loads directly.
    assert "linear" in helpers


def test_an_untied_embedding_is_still_loaded_directly(record_copies, tmp_path):
    """The guard must not cost untied models their direct embedding load."""
    from transformers import AutoModelForCausalLM

    path = tmp_path / "untied-llama"
    _tiny_causal_lm(tied=False).save_pretrained(path)

    with spyre_direct_placement(
        device="cpu", dtype=torch.float16, host_modules=("lm_head",)
    ):
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, device_map="cpu"
        )

    assert "embedding" in [name for name, _s, _d in record_copies]


# ── adapter opt-in ─────────────────────────────────────────────────────


def test_llama_and_granite_opt_in_and_keep_the_head_on_host():
    for module in (hf_llama, hf_granite):
        assert supports_direct_load(module)
        assert _host_module_prefixes(module) == ("lm_head",)


def test_an_adapter_without_the_flag_does_not_opt_in():
    from hf_adapters import hf_bert

    assert not supports_direct_load(hf_bert)


# ── attention that prep pads on the host ───────────────────────────────


def test_padded_attention_stays_on_host_only_when_prep_will_pad_it():
    from transformers import LlamaConfig

    from hf_adapters.hf_common import padded_attention_stays_on_host

    padded = SimpleNamespace(config=_tiny_config(tied=False))  # head_dim 4
    aligned = SimpleNamespace(
        config=LlamaConfig(
            hidden_size=512, num_attention_heads=4, num_key_value_heads=2
        )
    )  # head_dim 128

    assert padded_attention_stays_on_host(
        padded, "model.layers.0.self_attn.q_proj.weight"
    )
    assert padded_attention_stays_on_host(
        padded, "model.layers.1.self_attn.o_proj.weight"
    )
    assert not padded_attention_stays_on_host(
        padded, "model.layers.0.mlp.gate_proj.weight"
    )
    assert not padded_attention_stays_on_host(
        padded, "model.layers.0.self_attn.rotary_emb.inv_freq"
    )
    assert not padded_attention_stays_on_host(
        aligned, "model.layers.0.self_attn.q_proj.weight"
    )


def test_llama_and_granite_hold_back_attention_that_prep_pads():
    from hf_adapters.hf_common import padded_attention_stays_on_host
    from hf_adapters.spyre_direct_load import _host_filter

    for module in (hf_llama, hf_granite):
        assert _host_filter(module) is padded_attention_stays_on_host


# ── the context manager ────────────────────────────────────────────────


def test_placement_wrapper_routes_by_target_module(record_copies, monkeypatch):
    """Each parameter reaches the helper its *target module* implies."""
    model = _ToyModel()
    seen = []

    def _fake_original(model_, target_name, param_value, *args, **kwargs):
        seen.append((target_name, param_value.dtype))

    monkeypatch.setattr(core_model_loading, "set_param_for_module", _fake_original)

    with spyre_direct_placement(
        device="spyre:0", dtype=torch.float16, host_modules=("lm_head",)
    ):
        for target_name, tensor in [
            ("model.embed_tokens.weight", torch.randn(64, 16)),
            ("model.layers.0.q_proj.weight", torch.randn(32, 16)),
            ("model.layers.0.q_proj.bias", torch.randn(32)),
            ("model.norm.weight", torch.randn(16)),
            ("lm_head.weight", torch.randn(64, 16)),
        ]:
            core_model_loading.set_param_for_module(
                model, target_name, tensor, None, None
            )

    helpers = [name for name, _shape, _dtype in record_copies]
    assert helpers == ["embedding", "linear", "default", "default"]
    # lm_head was declared host-resident, so it reached HF untouched.
    assert seen[-1] == ("lm_head.weight", torch.float32)


def test_context_manager_restores_the_original():
    original = core_model_loading.set_param_for_module
    with spyre_direct_placement(device="spyre:0", dtype=torch.float16):
        assert core_model_loading.set_param_for_module is not original
    assert core_model_loading.set_param_for_module is original


def test_nested_entry_restores_once():
    original = core_model_loading.set_param_for_module
    with spyre_direct_placement(device="spyre:0", dtype=torch.float16):
        patched = core_model_loading.set_param_for_module
        with spyre_direct_placement(device="spyre:0", dtype=torch.float16):
            assert core_model_loading.set_param_for_module is patched
        # The inner exit must not restore while the outer block is live.
        assert core_model_loading.set_param_for_module is patched
    assert core_model_loading.set_param_for_module is original


def test_concurrent_loads_restore_the_original():
    """Two threads entering and leaving must not strand the patch."""
    original = core_model_loading.set_param_for_module
    start = threading.Barrier(2)

    def _load():
        start.wait()
        with spyre_direct_placement(device="spyre:0", dtype=torch.float16):
            assert core_model_loading.set_param_for_module is not original

    threads = [threading.Thread(target=_load) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert core_model_loading.set_param_for_module is original


def test_each_thread_keeps_its_own_device(monkeypatch):
    """Two concurrent loads must not inherit each other's rank.

    The installed wrapper is shared, so the device has to come from per-thread
    state rather than a closure over whichever load installed it first.
    """
    import hf_adapters.spyre_tensor_parallel as tp

    seen = {}

    def _capture(shard, *, device, dtype):
        seen[threading.current_thread().name] = device
        return shard

    monkeypatch.setattr(tp, "_copy_linear", _capture)
    monkeypatch.setattr(
        core_model_loading, "set_param_for_module", lambda *a, **kw: None
    )

    model = _ToyModel()
    ready = threading.Barrier(2)

    def _load(rank):
        with spyre_direct_placement(device=f"spyre:{rank}", dtype=torch.float16):
            # Both threads are inside their own block before either places a
            # tensor, so a shared closure would be observable here.
            ready.wait()
            core_model_loading.set_param_for_module(
                model,
                "model.layers.0.q_proj.weight",
                torch.randn(32, 16),
                None,
                None,
            )

    threads = [
        threading.Thread(target=_load, args=(rank,), name=f"rank{rank}")
        for rank in (0, 1)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen["rank0"] == torch.device("spyre:0")
    assert seen["rank1"] == torch.device("spyre:1")


def test_a_thread_outside_any_block_is_passed_through(monkeypatch):
    """A concurrent plain load must not be placed on the device."""
    import hf_adapters.spyre_tensor_parallel as tp

    placed = []
    monkeypatch.setattr(
        tp, "_copy_linear", lambda shard, **kw: placed.append(shard) or shard
    )
    monkeypatch.setattr(
        core_model_loading, "set_param_for_module", lambda *a, **kw: None
    )

    model = _ToyModel()
    with spyre_direct_placement(device="spyre:0", dtype=torch.float16):
        done = threading.Event()

        def _plain_load():
            core_model_loading.set_param_for_module(
                model,
                "model.layers.0.q_proj.weight",
                torch.randn(32, 16),
                None,
                None,
            )
            done.set()

        thread = threading.Thread(target=_plain_load)
        thread.start()
        thread.join()
        assert done.is_set()

    assert placed == [], "a thread outside the block had its tensor placed"


def test_exception_inside_the_block_still_restores():
    original = core_model_loading.set_param_for_module
    with pytest.raises(RuntimeError):
        with spyre_direct_placement(device="spyre:0", dtype=torch.float16):
            raise RuntimeError("load failed")
    assert core_model_loading.set_param_for_module is original


def test_read_ahead_is_turned_off_inside_the_block(monkeypatch):
    """Workers must not hold converted copies while the card is still transferring."""
    key = "HF_DEACTIVATE_ASYNC_LOAD"
    monkeypatch.delenv(key, raising=False)
    with spyre_direct_placement(device="cpu", dtype=torch.float16):
        assert os.environ[key] == "1"
    assert key not in os.environ


def test_read_ahead_env_is_restored_when_the_caller_had_set_it(monkeypatch):
    key = "HF_DEACTIVATE_ASYNC_LOAD"
    monkeypatch.setenv(key, "0")
    with spyre_direct_placement(device="cpu", dtype=torch.float16):
        assert os.environ[key] == "1"
    assert os.environ[key] == "0"


def test_nested_blocks_restore_read_ahead_once(monkeypatch):
    key = "HF_DEACTIVATE_ASYNC_LOAD"
    monkeypatch.delenv(key, raising=False)
    with spyre_direct_placement(device="cpu", dtype=torch.float16):
        with spyre_direct_placement(device="cpu", dtype=torch.float16):
            assert os.environ[key] == "1"
        assert os.environ[key] == "1"
    assert key not in os.environ


def test_exception_inside_the_block_still_restores_read_ahead(monkeypatch):
    key = "HF_DEACTIVATE_ASYNC_LOAD"
    monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError):
        with spyre_direct_placement(device="cpu", dtype=torch.float16):
            raise RuntimeError("load failed")
    assert key not in os.environ


# ── wiring in load_model_common ────────────────────────────────────────


def test_direct_to_device_is_refused_with_tp_plan():
    from hf_adapters.hf_common import load_model_common

    with pytest.raises(ValueError, match="not supported with tp_plan"):
        load_model_common(
            "unused",
            hf_llama,
            tp_plan="auto",
            direct_to_device=True,
        )


def test_every_auto_class_forwards_direct_to_device():
    """No ``AutoSpyreModelFor*`` may silently drop the flag.

    ``AutoSpyreModel.from_pretrained`` grew ``direct_to_device``, and each
    subclass overrides that method and calls ``super()``. A subclass that omits
    the parameter would accept the call and load via the host anyway, which is
    a slower load with no warning — so require it on all of them. mypy reports
    the same thing as an ``[override]`` error; this keeps it true for subclasses
    added later.
    """
    import inspect

    from hf_adapters.auto_spyre_model import AutoSpyreModel

    auto_classes = [AutoSpyreModel, *AutoSpyreModel.__subclasses__()]
    assert len(auto_classes) >= 7, auto_classes

    missing = [
        cls.__name__
        for cls in auto_classes
        if "direct_to_device" not in inspect.signature(cls.from_pretrained).parameters
    ]
    assert not missing, f"from_pretrained drops direct_to_device: {missing}"


def test_direct_to_device_warns_for_an_adapter_that_has_not_opted_in():
    from hf_adapters.hf_common import _maybe_direct_placement

    adapter = SimpleNamespace(__name__="hf_adapters.hf_notyet")
    with pytest.warns(UserWarning, match="SPYRE_SUPPORTS_DIRECT_LOAD"):
        with _maybe_direct_placement(adapter, torch.float16, True):
            pass


def test_the_device_index_survives(monkeypatch):
    """``spyre:1`` must not collapse to ``spyre``: a rank loads its own card."""
    captured = {}

    import hf_adapters.spyre_tensor_parallel as tp

    def _capture(shard, *, device, dtype):
        captured["device"] = device
        return shard

    monkeypatch.setattr(tp, "_copy_linear", _capture)
    _placement_copy(
        nn.Linear(16, 32),
        "weight",
        torch.randn(32, 16),
        device=torch.device("spyre:1"),
        dtype=torch.float16,
    )
    assert captured["device"] == torch.device("spyre:1")
    assert captured["device"].index == 1


def test_no_fake_safetensors_custom_device_registry_is_installed():
    """A ``hasattr`` probe for safetensors#804 must still report absence.

    This path never stubs the upstream registry, so code that feature-detects
    #804 is not misled and no transfer hook is silently discarded.
    """
    import safetensors

    with spyre_direct_placement(device="spyre:0", dtype=torch.float16):
        for attr in ("_is_custom_device", "_register_device_transfer_hook"):
            installed = getattr(safetensors, attr, None)
            assert installed is None or not getattr(
                installed, "_spyre_monkeypatched", False
            )


def test_a_quantized_tensor_passes_through_the_installed_wrapper(
    record_copies, monkeypatch
):
    """A tensor a quantizer owns must reach Transformers exactly as it arrived.

    ``load_model_common`` refuses ``direct_to_device`` together with
    ``tp_plan``, and since Transformers 5.23 a TP style is applied before this
    funnel is reached, so a quantizer is the remaining owner the wrapper has to
    decline for: its packed representation is not ours to re-lay-out.
    """
    model = _ToyModel()
    forwarded = []

    def _fake_original(model_, target_name, param_value, *args, **kwargs):
        forwarded.append((target_name, param_value))

    monkeypatch.setattr(core_model_loading, "set_param_for_module", _fake_original)

    packed = torch.randn(32, 16)
    # ``device`` is immaterial here -- the tensor must not be touched at all --
    # and "cpu" keeps this runnable without torch_spyre's PrivateUse1 rename.
    with spyre_direct_placement(device="cpu", dtype=torch.float16):
        core_model_loading.set_param_for_module(
            model,
            "model.layers.0.q_proj.weight",
            packed,
            None,
            object(),  # a quantizer owns this tensor
        )

    assert record_copies == [], "a quantizer-owned tensor must not be re-laid-out here"
    (_target_name, passed), *rest = forwarded
    assert not rest
    assert passed is packed, "the tensor must reach Transformers unchanged"


# ── against a real from_pretrained ─────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    """A real, two-layer Llama checkpoint on disk."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    config = LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
    )
    path = tmp_path_factory.mktemp("tiny-llama")
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(config).save_pretrained(path)
    return path


def test_the_wrapper_binds_the_real_set_param_for_module_signature():
    """Pin the signature the wrapper forwards positionally.

    ``_spyre_set_param_for_module`` takes five positional parameters and passes
    them straight through. If Transformers reorders or renames them, that
    forwarding silently hands arguments to the wrong parameters, so assert the
    order here rather than discovering it in a model's weights.
    """
    import inspect

    expected = [
        "model",
        "target_name",
        "param_value",
        "loading_info",
        "hf_quantizer",
    ]
    actual = list(inspect.signature(core_model_loading.set_param_for_module).parameters)
    assert actual == expected, (
        "transformers changed set_param_for_module's signature; "
        f"spyre_direct_load forwards {expected} positionally but found {actual}"
    )


def test_a_real_load_routes_every_parameter_and_preserves_values(
    record_copies, tiny_checkpoint
):
    """Drive a genuine ``from_pretrained`` through the placement block.

    Every other test here calls the wrapper directly, which leaves the
    assumption that Transformers invokes it with five positional arguments
    unexercised — the same gap that was called out on torch-spyre#4906. This
    runs the real loader, so the routing, the signature binding and the fact
    that the settings are visible from the thread that calls the funnel are all
    exercised at once. The DMA helpers are still recorders, so no hardware is
    needed; placement is asserted through which helper each parameter reached.
    """
    from transformers import AutoModelForCausalLM

    reference = dict(
        AutoModelForCausalLM.from_pretrained(
            tiny_checkpoint, dtype=torch.float16, device_map="cpu"
        ).named_parameters()
    )

    with spyre_direct_placement(
        device="cpu", dtype=torch.float16, host_modules=("lm_head",)
    ):
        loaded = AutoModelForCausalLM.from_pretrained(
            tiny_checkpoint, dtype=torch.float16, device_map="cpu"
        )

    assert record_copies, "the real loader never reached the placement wrapper"

    routed = {}
    for name, shape, _dtype in record_copies:
        routed.setdefault(name, []).append(shape)

    assert "embedding" in routed, routed
    assert "linear" in routed, routed
    assert "default" in routed, routed
    # The token table is an nn.Embedding; the projections are nn.Linear.
    assert (64, 16) in routed["embedding"]
    assert all(len(shape) == 2 for shape in routed["linear"])
    # Declared host-resident, so it must not have been placed.
    assert (64, 16) not in routed["linear"]

    # Routing must not have perturbed the weights themselves.
    actual = dict(loaded.named_parameters())
    assert set(actual) == set(reference)
    differing = [
        name for name in reference if not torch.equal(actual[name], reference[name])
    ]
    assert not differing, differing


def test_padded_attention_is_left_on_the_host_in_a_real_load(
    record_copies, tiny_checkpoint
):
    """q/k/v/o that prep will pad must reach Transformers as host tensors.

    The tiny checkpoint's head dim is 4, which ``prepare_rope_and_heads`` pads
    by slice-copying into new host tensors, so only the MLP projections may be
    placed directly.
    """
    from transformers import AutoModelForCausalLM

    with spyre_direct_placement(
        device="cpu",
        dtype=torch.float16,
        host_modules=("lm_head",),
        host_filter=hf_llama.spyre_direct_stays_on_host,
    ):
        AutoModelForCausalLM.from_pretrained(
            tiny_checkpoint, dtype=torch.float16, device_map="cpu"
        )

    linear_shapes = [shape for name, shape, _dtype in record_copies if name == "linear"]
    # Two layers of gate (32, 16), up (32, 16) and down (16, 32); no attention.
    assert sorted(linear_shapes) == sorted([(32, 16), (32, 16), (16, 32)] * 2)


# ── upload threads ─────────────────────────────────────────────────────


@pytest.fixture
def record_threaded_copies(monkeypatch):
    """Recorders that return a new tensor and note the thread they ran on.

    Returns a list of ``(helper, shape, thread name, result)``. A new tensor,
    unlike ``record_copies``' pass-through, shows whether the result was
    swapped into the module or the host tensor was left in place.
    """
    calls = []
    lock = threading.Lock()

    def _recorder(name):
        def _copy(shard, *, device, dtype):
            result = (shard.to(dtype) if dtype is not None else shard).clone()
            with lock:
                calls.append(
                    (name, tuple(shard.shape), threading.current_thread().name, result)
                )
            return result

        return _copy

    import hf_adapters.spyre_tensor_parallel as tp

    for helper in ("linear", "embedding", "default"):
        monkeypatch.setattr(tp, f"_copy_{helper}", _recorder(helper))
    return calls


def _load_tiny(path):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float16, device_map="cpu"
    )


def test_upload_threads_place_every_weight_in_a_real_load(
    record_threaded_copies, tiny_checkpoint
):
    """Uploads run on the block's threads and every result is swapped in.

    Each placed parameter must end up holding the tensor its upload returned,
    so nothing still holds the host copy, and the values must match a plain
    load.
    """
    reference = dict(_load_tiny(tiny_checkpoint).named_parameters())

    with spyre_direct_placement(
        device="cpu",
        dtype=torch.float16,
        host_modules=("lm_head",),
        upload_threads=4,
    ):
        loaded = _load_tiny(tiny_checkpoint)

    assert record_threaded_copies, "the real loader never reached the wrapper"
    threads = {thread for _, _, thread, _ in record_threaded_copies}
    assert all(name.startswith("spyre-direct-upload") for name in threads), threads

    uploaded = {result.data_ptr() for *_, result in record_threaded_copies}
    actual = dict(loaded.named_parameters())
    swapped_in = sorted(name for name, p in actual.items() if p.data_ptr() in uploaded)
    assert swapped_in == sorted(name for name in actual if name != "lm_head.weight")

    differing = [n for n in reference if not torch.equal(actual[n], reference[n])]
    assert not differing, differing


def test_a_swapped_in_upload_keeps_the_parameter_object():
    """Whatever holds the parameter -- a tie, made before or after -- sees it."""
    from hf_adapters.spyre_direct_load import _InFlight, _swap_in

    linear = nn.Linear(4, 2)
    held = linear.weight
    held._is_hf_initialized = True
    uploaded = torch.full((2, 4), 7.0)

    _swap_in(_InFlight(None, linear, "weight", held, 0), uploaded)

    assert linear.weight is held
    assert torch.equal(held, uploaded)
    assert held._is_hf_initialized
    assert held.requires_grad


def test_an_uploaded_buffer_replaces_the_host_buffer():
    from hf_adapters.spyre_direct_load import _InFlight, _swap_in

    module = nn.Module()
    module.register_buffer("ids", torch.zeros(3, dtype=torch.long))
    uploaded = torch.ones(3, dtype=torch.long)

    _swap_in(_InFlight(None, module, "ids", module.ids, 0), uploaded)

    assert module.ids is uploaded


def test_an_upload_is_dropped_if_the_attribute_was_replaced():
    """Something else installed a new tensor meanwhile; it must not be undone."""
    from hf_adapters.spyre_direct_load import _InFlight, _swap_in

    linear = nn.Linear(4, 2)
    old = linear.weight
    replacement = nn.Parameter(torch.zeros(2, 4))
    linear.weight = replacement

    _swap_in(_InFlight(None, linear, "weight", old, 0), torch.ones(2, 4))

    assert linear.weight is replacement
    assert torch.equal(replacement, torch.zeros(2, 4))


def test_in_flight_host_bytes_stay_within_the_limit(
    record_threaded_copies, tiny_checkpoint, monkeypatch
):
    """After every hand-off the bytes awaiting upload are at most the limit.

    The limit is what bounds the host memory the threads add. A tensor larger
    than the limit is still uploaded, alone: the funnel waits for it.
    """
    import hf_adapters.spyre_direct_load as direct_load

    in_flight_after = []
    original_drain = direct_load._drain

    def _drain(block, limit):
        original_drain(block, limit)
        if limit >= 0:
            in_flight_after.append(block.in_flight_bytes)

    monkeypatch.setattr(direct_load, "_drain", _drain)
    # Each MLP projection is 32 x 16 fp16 = 1 KiB; the embedding is 2 KiB.
    limit = 1024
    with spyre_direct_placement(
        device="cpu",
        dtype=torch.float16,
        upload_threads=4,
        max_in_flight_bytes=limit,
    ):
        _load_tiny(tiny_checkpoint)

    assert in_flight_after
    assert max(in_flight_after) <= limit


def test_a_failed_upload_fails_the_load_and_restores_everything(
    monkeypatch, tiny_checkpoint
):
    import hf_adapters.spyre_tensor_parallel as tp

    def _broken(shard, *, device, dtype):
        raise RuntimeError("DMA failed")

    for helper in ("linear", "embedding", "default"):
        monkeypatch.setattr(tp, f"_copy_{helper}", _broken)
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    original = core_model_loading.set_param_for_module

    with pytest.raises(RuntimeError, match="DMA failed"):
        with spyre_direct_placement(
            device="cpu", dtype=torch.float16, upload_threads=2
        ):
            _load_tiny(tiny_checkpoint)

    assert core_model_loading.set_param_for_module is original
    assert "HF_DEACTIVATE_ASYNC_LOAD" not in os.environ


def test_only_an_indexed_device_gets_upload_threads_by_default():
    from hf_adapters.spyre_direct_load import default_upload_threads

    assert default_upload_threads("cpu") == 1
    # Each upload thread has to be told its device, so an index is required.
    assert default_upload_threads("spyre") == 1


def test_load_model_common_uploads_on_as_many_threads_as_model_to(monkeypatch):
    import hf_adapters.spyre_direct_load as direct_load
    from hf_adapters.hf_common import _maybe_direct_placement

    captured = {}

    @contextmanager
    def _capture(**kwargs):
        captured.update(kwargs)
        yield

    monkeypatch.setattr(direct_load, "default_upload_threads", lambda device: 6)
    monkeypatch.setattr(direct_load, "spyre_direct_placement", _capture)

    with _maybe_direct_placement(hf_llama, torch.float16, True):
        pass

    assert captured["upload_threads"] == 6


# ── review fixes ───────────────────────────────────────────────────────


def test_a_tied_head_holds_back_its_embedding_once_it_is_loaded_too():
    """The head leaving ``meta`` must not release the embedding.

    A checkpoint that stores both keys fills the head first; Transformers
    still ties the pair afterwards when their values match.
    """
    model = _tiny_causal_lm(tied=True, on_meta=False)
    assert model.lm_head.weight.device.type != "meta"
    assert _tied_to_host(model, "model.embed_tokens.weight", ("lm_head",))


def test_a_tied_checkpoint_that_also_stores_the_head_stays_on_the_host(
    record_copies, tmp_path
):
    """``.bin`` checkpoints store both keys of a tied pair.

    The embedding must stay on the host there too: Transformers ties the head
    to it after loading, so placing it would place the head.
    """
    from safetensors.torch import load_file, save_file
    from transformers import AutoModelForCausalLM

    path = tmp_path / "tied-llama-both-keys"
    _tiny_causal_lm(tied=True).save_pretrained(path)
    weights = path / "model.safetensors"
    state = load_file(weights)
    assert "lm_head.weight" not in state
    state["lm_head.weight"] = state["model.embed_tokens.weight"].clone()
    save_file(state, weights, metadata={"format": "pt"})

    with spyre_direct_placement(
        device="cpu", dtype=torch.float16, host_modules=("lm_head",)
    ):
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16, device_map="cpu"
        )

    assert model.lm_head.weight is model.model.embed_tokens.weight
    helpers = [name for name, _shape, _dtype in record_copies]
    assert "embedding" not in helpers, "the tied embedding was placed on the device"
    assert "linear" in helpers


def test_a_wrapper_looked_up_before_the_last_block_exits_still_installs():
    """A plain load on another thread may call the wrapper after the restore."""
    model = _ToyModel()
    info = SimpleNamespace(
        missing_keys=set(), unexpected_keys=set(), mismatched_keys=set()
    )
    with spyre_direct_placement(device="cpu", dtype=torch.float16):
        looked_up = core_model_loading.set_param_for_module
    assert core_model_loading.set_param_for_module is not looked_up

    value = torch.full((32, 16), 3.0)
    looked_up(model, "model.layers.0.q_proj.weight", value, info, None)

    assert torch.equal(model.model.layers[0].q_proj.weight, value)


@pytest.mark.parametrize("upload_threads", [1, 4])
def test_the_block_counts_the_tensors_it_placed(
    record_threaded_copies, tiny_checkpoint, upload_threads
):
    with spyre_direct_placement(
        device="cpu",
        dtype=torch.float16,
        host_modules=("lm_head",),
        upload_threads=upload_threads,
    ) as block:
        assert block.placed == 0
        _load_tiny(tiny_checkpoint)

    assert block.placed == len(record_threaded_copies) > 0


def test_a_direct_load_that_places_nothing_warns(monkeypatch):
    """The model is still right, but the caller asked for something else."""
    import hf_adapters.spyre_direct_load as direct_load
    from hf_adapters.hf_common import _maybe_direct_placement

    @contextmanager
    def _places_nothing(**kwargs):
        yield SimpleNamespace(placed=0)

    monkeypatch.setattr(direct_load, "default_upload_threads", lambda device: 1)
    monkeypatch.setattr(direct_load, "spyre_direct_placement", _places_nothing)

    with pytest.warns(UserWarning, match="had no effect"):
        with _maybe_direct_placement(hf_llama, torch.float16, True):
            pass


def test_load_model_common_records_how_many_tensors_were_placed(
    record_copies, tiny_checkpoint, monkeypatch
):
    import warnings

    from transformers import AutoModelForCausalLM

    from hf_adapters import hf_common

    monkeypatch.setattr(hf_common, "DEVICE", "cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = hf_common.load_model_common(
            tiny_checkpoint,
            hf_llama,
            dtype=torch.float16,
            auto_model_cls=AutoModelForCausalLM,
            direct_to_device=True,
        )

    assert model._spyre_direct_placed == len(record_copies) > 0


def test_a_staged_load_records_no_placement_count(tiny_checkpoint):
    from transformers import AutoModelForCausalLM

    from hf_adapters import hf_common

    model = hf_common.load_model_common(
        tiny_checkpoint,
        hf_llama,
        dtype=torch.float16,
        auto_model_cls=AutoModelForCausalLM,
    )

    assert not hasattr(model, "_spyre_direct_placed")


def test_direct_to_device_warns_for_an_adapter_with_its_own_loader():
    from hf_adapters.hf_common import load_model_common

    adapter = SimpleNamespace(
        __name__="hf_adapters.hf_custom",
        load_hf_model=lambda model_path, dtype: nn.Linear(2, 2),
    )
    with pytest.warns(UserWarning, match="load_hf_model"):
        load_model_common("unused", adapter, direct_to_device=True)
