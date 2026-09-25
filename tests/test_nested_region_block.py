"""CPU-only unit tests for the nested_compile_region block wrapper."""

import torch
from torch import nn


def test_nested_region_block_drops_cache_return():
    # The wrapper drops the block's (key_cache, value_cache) return and exposes
    # only ``h`` (see nested_region_block / _shared_region_block docstrings and
    # test_region_block_kv_contract.py). The block itself returns (h, kc, vc).
    from hf_adapters.hf_common import nested_region_block

    class Dummy(nn.Module):
        # The region dispatches to forward, so the fake must provide it.
        def forward(self, h, freqs, mask, kc, vc, cache_index):
            return h + 1.0, kc, vc

    wrapped = nested_region_block(Dummy())
    h = torch.zeros(1, 4, 8)
    out = wrapped(h, None, None, torch.zeros(1), torch.zeros(1), 0)
    assert isinstance(out, torch.Tensor)
    assert torch.allclose(out, h + 1.0)


def test_nested_region_block_is_callable_region():
    # The wrapper must be callable and dispatch to the shared compile region,
    # so torch.compile treats it as an invoke_subgraph boundary.
    # Importing this IS part of the assertion: the wrapper is only a real compile
    # region if torch exposes nested_compile_region, so an ImportError here is a
    # genuine failure. Hence noqa rather than deleting the "unused" import.
    from torch.compiler import nested_compile_region  # noqa: F401

    from hf_adapters.hf_common import nested_region_block

    class Dummy(nn.Module):
        def forward(self, h):
            return h, None, None

    wrapped = nested_region_block(Dummy())
    assert callable(wrapped)
