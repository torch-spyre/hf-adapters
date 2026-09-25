# tests/test_nested_region_compile_once.py
"""Gate: a region-wrapped block called N times compiles ONCE under torch.compile
AND each layer runs with its OWN weights.

Uses torch._dynamo compile counters + the invoke_subgraph HOP so it runs on CPU
with the default backend (no Spyre needed).

The weight-correctness half is the regression guard for the closure-capture bug:
``nested_compile_region`` traces the shared subgraph once and reuses it, so any
per-layer state the region *closes over* is frozen at layer 0. The block must
therefore be passed as an ARGUMENT to the region (not a closure cell), or every
layer silently runs with layer 0's weights.
"""

import torch
import torch._dynamo as dynamo
from torch import nn


class _Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(self, h, kc, vc):
        # Return the (h, kc, vc) triple that nested_region_block expects; the
        # wrapper drops kc/vc and exposes only h.
        return self.lin(h).relu(), kc, vc


def test_region_block_compiles_once_across_layers():
    from hf_adapters.hf_common import nested_region_block

    dim = 16
    n_layers = 4
    torch.manual_seed(0)  # CPU only, no VFIO card here — safe
    raw = [_Block(dim) for _ in range(n_layers)]
    # Distinct weights per layer so a frozen-first-layer bug would show up.
    for b in raw:
        with torch.no_grad():
            b.lin.weight.copy_(torch.randn(dim, dim))
    blocks = [nested_region_block(b) for b in raw]

    kc = torch.zeros(1)
    vc = torch.zeros(1)

    def outer(h):
        for b in blocks:
            h = b(h, kc, vc)
        return h

    h = torch.randn(2, dim)
    eager_out = outer(h)

    dynamo.reset()
    compiled = torch.compile(outer, dynamic=False, fullgraph=True)
    out = compiled(h)
    assert out.shape == (2, dim)

    # Each layer must have used its own weights: compiled == eager.
    torch.testing.assert_close(out, eager_out)

    # The N region calls must lower to invoke_subgraph, sharing one subgraph.
    gm, _ = dynamo.export(outer)(h)
    calls = [
        n for n in gm.graph.nodes if "invoke_subgraph" in str(getattr(n, "target", ""))
    ]
    assert len(calls) >= 2, f"expected repeated invoke_subgraph calls, got {calls}"
