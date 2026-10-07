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
