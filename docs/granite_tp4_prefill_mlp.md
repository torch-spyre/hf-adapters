# Experimental Granite TP4 prefill MLP work splits

This candidate is parked. Padding is off by default. Set
`SPYRE_GRANITE_TP4_MLP_PADDING=1` before loading to opt in. With this option,
eligible padded prefill calls request eight row groups and four column groups
across gate, activation, up, product and down. The old
`SPYRE_GRANITE_TP4_PREFILL_MLP` option does not override the padding opt-in.

The hint path requires the standard Granite inference MLP, SiLU activation,
hidden width 4096, intermediate width 12800 and 40 layers. Each projection must
have FP16 weights, no bias and the supported Spyre TP hooks. All three styles
must share the same one-dimensional four-rank mesh. Before padding, gate and up
weights must each be 3200 by 4096; down must be 4096 by 3200. The padding guard
also rejects custom forwards and unsupported hooks. Gate and up are staged on
the CPU before padding and normal device transfer.

Gate and up storage grows to 3328 channels. Down reads the original 3200
channels. A hinted input must be a Spyre FP16 tensor with shape 1 by 512 by
4096. Decode and other input shapes retain the padded module's normal forward.
Unsupported models, meshes and weights do not take the hint path. With batch
one, the guard sees each local 512-row chunk. This does not qualify later
chunks or full-model generation.

The existing `spyre_hint` API requests work divisions. It does not reserve LX
memory or guarantee the requested emitted plan. Check actual gate, up and down
divisions, memory placement and transfer plans for each compiler pin before
timing. Keep merged-main tests separate from tests that include unmerged
compiler PRs.

## Current full-model device result

On 2026-10-30, merged torch-spyre main `6ef8efd1` included compiler PR #5644.
The adapter baseline was main `3b54811e` plus #613 and #697. This candidate
added only the guarded hint path to that baseline. The production Python
sources match the measured full-stack tree `275a4824`.

The test directly ran all 40 layers of trained Granite-3.3-8B-Instruct, FP16,
batch one and 512 prompt tokens. It used rank-zero TP4 weight shards on one
card, with identity collectives. This is not distributed TP4 or serving timing.

| Metric | Padding alone, ms | Padding and hints, ms | Change |
| --- | ---: | ---: | ---: |
| 40-layer device kernels | 51.654212 | 51.655210 | +0.002% |
| All device kernels | 55.399414 | 55.374874 | -0.044% |
| Synchronized wall | 63.274120 | 63.255647 | -0.029% |

Both configurations emitted the same captured program summaries. Gate, up
and down already used eight row groups and four column groups without the
hints. These results show parity, not a useful extra hint gain. Keep this
candidate parked.

Each configuration had one private cold compile, two warmups, 15 device
profiles and 15 separate wall samples. No timed call compiled. Required finite
and numerical screens passed, and the final top token matched the CPU
reference. This does not replace broad model CI. Repeated calls do not measure
variation across independent compiler draws.

The compiler used native module SHA-256
`dc427d5efb0548f54e728a39e47179318ec6319226ba441c4079ef63b67d5bd5`,
built at `aa5a41e8`. All 63 native build inputs matched `6ef8efd1` exactly.
The isolated SDK used deeptools 291, Flex 140 and libaiupti 19. CPU settings
were OMP/MKL 16 with `SPYRE_NUM_CPUS` unset; the planner resolved 96 workers.
The checkpoint revision was `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`.
The wall boundary included mask and input preparation, all layers, the compact
final-token head, logits D2H and synchronization. Compilation was excluded.

The full-stack 40-layer device time is 0.96% above the 51.166543 ms
[golden reference](https://github.com/torch-spyre/torch-spyre/issues/5575).
This is parity within the accepted 2–3% range. The historical reference used
different source/runtime pins and a different card; this comparison does not
isolate a compiler change. The related
[SI reference](https://github.com/torch-spyre/spyre-inference/issues/1333)
has its own application boundary.

## Earlier evidence

The 2026-10-28 comparison used trained Granite layer-zero MLP weights,
TP4 rank-zero shards, one card and identity collectives. Prefill median
synchronized wall times were 1482.217 microseconds for the original MLP,
2158.893 for padding alone and 1960.162 for padding with these hints. The hints
reduced time by 9.2% against padding alone, but the combined change was 32.2%
slower than the original MLP. All twelve paired blocks lost against the
original. Prefill outputs matched across the three arms. Device-only timing
was unavailable in that run.

Those results used a frozen optional compiler stack on 2026-10-28. They are
not results for current merged compiler main.

The 2026-10-29 comparison used merged torch-spyre main `aa5a41e8` and
hf-adapters main `3b54811e`, plus the complete current padding change and this
local hint update. The native module was built from that compiler source.
Unmerged compiler PR #5644 was excluded. The workload was a trained layer-zero
`StandardGQABlock`, with batch one, FP16, 512 rows, TP4 rank-zero shards and
identity collectives on one card. It used a private cold compile cache per
process, five warmups, 31 wall samples and seven device profiles. Two processes
ran per arm. All output screens passed; timing contained no compilation.

Process median wall times were 2.154 and 2.286 ms for main, 1.646 and 1.609 ms
for padding alone, and 1.602 and 1.610 ms for padding with hints. The O-projection
divisions varied across compiles. Pairing samples with the same non-MLP matmul
divisions leaves padding 25.3% and 28.0% faster. The corresponding hint changes
against padding alone are -0.44% and +0.05%. These do not establish extra hint
benefit. Main already selected 8-by-4 for all three padded MLP matmuls without
these hints. The unpadded gate and up used 16 cores; the padded projections
used 32. Other allocation choices can still vary.

This block result and the earlier standalone-MLP result have different scopes
and compiler pins. They cannot measure a compiler-only improvement. Neither
establishes serving or distributed TP4 performance. Keep padding off by default.
Keep this hint candidate parked until it has a separate useful effect.
