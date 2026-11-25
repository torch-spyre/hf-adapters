"""Op names the model_ops generator emits for ``&`` / ``|`` captures.

``TorchOpCollector._extract_op_name`` turns a dynamo node into the ``op:``
string a torch-spyre op-test YAML carries, and torch-spyre's ``op_registry``
reads a trailing ``_`` as **in-place** (``torch.add`` vs ``torch.add_``). The
trailing underscore in ``operator.and_`` / ``or_`` is only Python
keyword-avoidance — ``operator.and_.__doc__`` is "Same as a & b." — so spelling
those ``"torch." + name`` gave the *out-of-place* capture the in-place name, and
the registry duly implemented it in place. An in-place write cannot express a
capture whose operands broadcast, which is how it surfaced: the encoder mask
chain (``result = result & mask(...)``) failed inside the test harness before
reaching the device. See torch-spyre/hf-adapters#546.

Pinned here against real dynamo graphs rather than by reading the mapping table,
because the names depend on which ``_operator`` function dynamo picks for ``&``
vs ``&=``. CPU-only, no ``torch_spyre`` — and nothing here needs the root
conftest's adapter scaffolding, so ``--noconftest`` works on a host without a
matching transformers:

    pytest --noconftest tests/test_model_ops_op_names.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_TORCHOP_YAML = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "model_ops"
    / "utils"
    / "torchop_yaml.py"
)


@pytest.fixture(scope="module")
def collector():
    """``TorchOpCollector``, loaded by file path.

    ``utils/model_ops/`` is not a package (the generator scripts run with it as
    cwd and import ``utils.torchop_yaml``), so there is no import path to spell
    from the repo root.
    """
    spec = importlib.util.spec_from_file_location("_torchop_yaml", _TORCHOP_YAML)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TorchOpCollector


def _op_names(collector, fn):
    """``{operator function name: emitted op name}`` for ``fn``'s ``_operator`` nodes.

    Traces ``fn`` with a backend that keeps the graph and returns it unchanged,
    so nothing is compiled or run on a device.
    """
    graphs = []

    def keep(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    lhs = torch.ones(2, 2, dtype=torch.bool)
    rhs = torch.zeros(2, 2, dtype=torch.bool)
    torch._dynamo.reset()
    torch.compile(fn, backend=keep, dynamic=False)(lhs, rhs)
    assert graphs, f"dynamo captured no graph for {fn.__name__}"

    return {
        node.target.__name__: collector._extract_op_name(node)
        for gm in graphs
        for node in gm.graph.nodes
        if node.op == "call_function"
        and getattr(node.target, "__module__", "") == "_operator"
    }


def _and(lhs, rhs):
    return lhs & rhs


def _or(lhs, rhs):
    return lhs | rhs


def _iand(lhs, rhs):
    lhs &= rhs
    return lhs


def _ior(lhs, rhs):
    lhs |= rhs
    return lhs


@pytest.mark.parametrize(
    "fn, operator_name, expected",
    [
        (_and, "and_", "torch.bitwise_and"),
        (_or, "or_", "torch.bitwise_or"),
        (_iand, "iand", "torch.Tensor.bitwise_and_"),
        (_ior, "ior", "torch.Tensor.bitwise_or_"),
    ],
    ids=["and", "or", "iand", "ior"],
)
def test_bitwise_op_names(collector, fn, operator_name, expected):
    names = _op_names(collector, fn)
    assert operator_name in names, f"no _operator.{operator_name} node: {names}"
    assert names[operator_name] == expected


@pytest.mark.parametrize(
    "fn, inplace",
    [(_and, False), (_or, False), (_iand, True), (_ior, True)],
    ids=["and", "or", "iand", "ior"],
)
def test_trailing_underscore_means_inplace(collector, fn, inplace):
    """The invariant behind the names above, stated on its own.

    Every emitted name must be a real ``torch`` spelling whose trailing ``_``
    agrees with whether the capture mutates its first argument — that agreement
    is what torch-spyre's registry relies on when it sets ``is_inplace``. The
    in-place ops exist only as Tensor methods (there is no ``torch.bitwise_and_``),
    so both prefixes resolve against ``torch.Tensor``.
    """
    for name in _op_names(collector, fn).values():
        stem = name.removeprefix("torch.").removeprefix("Tensor.")
        assert stem.endswith("_") is inplace, f"{name}: is_inplace should be {inplace}"
        assert hasattr(torch.Tensor, stem), f"{name} is not a torch spelling"
