"""``TorchOpCollector`` must forward ``compile_fx`` arguments unchanged.

torch-spyre wraps ``torch._inductor.compile_fx.compile_fx`` with
``(gm, example_inputs, *args, **kwargs)`` and does
``kwargs.setdefault("decompositions", ...)``. A collector that forwards
``decompositions`` positionally makes that raise ``TypeError: compile_fx() got
multiple values for argument 'decompositions'``. The stand-in below reproduces
that wrapper shape, so this runs on CPU without ``torch_spyre``:

    pytest --noconftest tests/test_torchop_collector_compile_fx.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch._inductor.compile_fx as cfx

_TORCHOP_YAML = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "model_ops"
    / "utils"
    / "torchop_yaml.py"
)


@pytest.fixture(scope="module")
def collector():
    spec = importlib.util.spec_from_file_location("_torchop_yaml_cfx", _TORCHOP_YAML)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TorchOpCollector


def _spyre_like_wrapper(received):
    """Mimic torch-spyre's wrapper around the real ``compile_fx``.

    It ``setdefault``s the keywords it injects, then forwards to an inner
    function with ``compile_fx``'s named parameters — that is where a
    positional ``decompositions`` from the caller collides.
    """

    def inner(
        gm,
        example_inputs,
        inner_compile=None,
        config_patches=None,
        decompositions=None,
        **kwargs,
    ):
        received.append(
            {"inner_compile": inner_compile, "decompositions": decompositions}
        )
        return gm.forward

    def wrapper(gm, example_inputs, *args, **kwargs):
        kwargs.setdefault("decompositions", {})
        return inner(gm, example_inputs, *args, **kwargs)

    return wrapper


@pytest.fixture
def spyre_like_compile_fx(monkeypatch):
    received = []
    monkeypatch.setattr(cfx, "compile_fx", _spyre_like_wrapper(received))
    return received


def _trace(collector):
    def fn(x):
        return x * 2 + 1

    torch._dynamo.reset()
    with collector() as ctx:
        torch.compile(fn, backend="inductor")(torch.ones(4, 4))
    return ctx


def test_collector_does_not_collide_with_setdefault_wrapper(
    collector, spyre_like_compile_fx
):
    ctx = _trace(collector)

    assert "torch.mul" in ctx.ops_list
    assert "torch.add" in ctx.ops_list
    assert len(spyre_like_compile_fx) == 1


def test_collector_forwards_caller_arguments_unchanged(
    collector, spyre_like_compile_fx
):
    _trace(collector)

    (seen,) = spyre_like_compile_fx
    # The wrapper's injected default reaches the real compile_fx; the collector
    # did not inject an ``inner_compile`` of its own.
    assert seen["decompositions"] == {}
    assert seen["inner_compile"] is None


def test_collector_restores_compile_fx_on_exit(collector):
    before = cfx.compile_fx
    with collector():
        assert cfx.compile_fx is not before
    assert cfx.compile_fx is before


def test_extract_meta_info_accepts_opaque_script_objects(collector):
    """Opaque graph inputs carry no tensor metadata and must not abort tracing.

    vLLM (torch >= 2.11) passes the encoded layer name to
    ``unified_attention_with_output`` as a ``FakeScriptObject`` graph input.
    ``_extract_meta_info`` used to fall through to ``raise RuntimeError``. The
    ``FakeScriptObject`` below is built directly (it deep-copies the wrapped
    object), so this checks the type handling, not vLLM's real graph; the
    end-to-end failure was reproduced on a Spyre pod.
    """
    from torch._library.fake_class_registry import FakeScriptObject

    fake = FakeScriptObject(object(), "test.OpaqueLayerName", object())
    real = torch.jit.script(torch.nn.Linear(1, 1))._c

    for opaque in (fake, real):
        assert collector._extract_meta_info(opaque) == (None,) * 5


def test_extract_meta_info_still_rejects_unknown_types(collector):
    with pytest.raises(RuntimeError):
        collector._extract_meta_info(object())


@pytest.fixture(scope="module")
def spyre_dummy_op():
    """A custom op in a ``spyre`` namespace, standing in for torch-spyre's own ops.

    A dedicated ``Library`` handle keeps the registration scoped to this module.
    """
    lib = torch.library.Library("spyre", "FRAGMENT")
    lib.define("dummy_copy(Tensor x) -> Tensor")
    lib.impl("dummy_copy", lambda x: x.clone(), "CompositeExplicitAutograd")
    lib.impl("dummy_copy", lambda x: torch.empty_like(x), "Meta")
    yield torch.ops.spyre.dummy_copy
    lib._destroy()


def test_torch_spyre_internal_ops_get_no_test_case(
    collector, spyre_like_compile_fx, spyre_dummy_op
):
    """torch-spyre's own device-copy / dtype ops are not model ops.

    They are skipped before ``ops_set`` and ``test_gen_ops_set`` are touched, so
    they appear in neither list, while ordinary ops in the same graph still do.
    """

    def fn(x):
        return spyre_dummy_op(x) * 2

    # Test cases are only emitted for non-CPU float16 tensors; "meta" gives the
    # collector that shape without a device.
    x = torch.ones(4, 4, dtype=torch.float16, device="meta")
    torch._dynamo.reset()
    with collector() as ctx:
        torch.compile(fn, backend="inductor")(x)

    assert "torch.mul" in ctx.ops_list
    assert "torch.mul" in ctx.test_gen_ops
    assert not any(op.startswith("torch.ops.spyre.") for op in ctx.ops_list)
    assert not any(op.startswith("torch.ops.spyre.") for op in ctx.test_gen_ops)
