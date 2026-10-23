# Automatic Granite TP4 prefill MLP work split

Matching inputs use the experimental work split automatically. For example,
512 rows are divided into eight groups of 64 rows. The hints also request
four output-column groups for gate, activation, up, product and down.
No environment flag is needed. The former `SPYRE_GRANITE_TP4_PREFILL_MLP`
variable has no effect, including when it is set to `0`.

The model must use the standard Granite MLP for inference, SiLU activation,
hidden width 4096, intermediate width 12800, and 40 layers. Each projection
must have FP16 weights, no bias, and the supported Spyre projection hooks.
All three projections must share the same one-dimensional four-rank group.
Before padding, gate/up weights must each be 3200 by 4096 and down weights
4096 by 3200. Gate/up storage grows to 3328 channels; down still reads only
the original 3200 channels. The input at each MLP call must be a Spyre FP16
tensor of shape 1 by 512 by 4096. Other cases keep the existing path,
including decode, training, TP1, other models, and unsupported shapes or hooks.

The check sees the local input chunk, not the original prompt length.
With batch 1, every matching 512-row chunk takes this path. Shorter prompts
are padded to one chunk. Each chunk of a longer prompt also takes this path.
Only a single 512-token chunk was validated; shorter prompts and later chunks
were not compiled or checked in that validation.

Use the frontend LX transfers and MLP prefix transfer described in
[issue #5575](https://github.com/torch-spyre/torch-spyre/issues/5575), plus the
existing `spyre_hint` interface from
[PR #5057](https://github.com/torch-spyre/torch-spyre/pull/5057).
The saved compiler `6070e77` supports the needed hint form. The hints request
work splits; they do not reserve LX memory. The fixed 80% frontend / 20%
backend LX split is unchanged. On another compiler version, inspect the
generated splits and memory placement again.

The saved checks ran each logical rank separately on one card, with rank
communication replaced by calls that return the input unchanged. The baseline
already used LX. No device-time gain was shown. Ranks 0, 1 and 3 passed the
required output checks. Rank 2 had the same hidden-output failure without
and with hints; its cause remains unresolved. Saved outputs matched byte for
byte in both cases on every rank. This does not prove full correctness.

Automatic enablement keeps the prior hinted computation and all guards.
Focused CPU checks cover default selection, ignored old flag values, fallback
cases, and the two unchanged hint scopes. Timing was not repeated. These
checks do not add device or real distributed evidence. The reported regression
remains unreproduced, the observed plan variation does not prove nondeterminism,
and this work has no production qualification or real distributed TP4
correctness or time-to-first-token result.
