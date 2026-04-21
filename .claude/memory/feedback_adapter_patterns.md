---
name: Adapter Development Patterns
description: "Non-obvious pitfalls learned from building HF Spyre adapters (rules in CLAUDE.md are not repeated here)"
type: feedback
---

## Granite 4.0 (granitemoehybrid) needs special test handling
- Use `granite-4.0-1b-base` (all-attention), NOT `granite-4.0-tiny-preview` (has Mamba)
- Use `float32` on CPU (fp16 overflows due to embedding_multiplier)
- Use `model.generate()` fallback for HF reference (DynamicCache incompatible)
**Why:** Three separate issues that all cause test failures. Discovered the hard way.
**How to apply:** The test registry has `"dtype": "float32"` for granite4. Keep it.

## FMS eager_spyre ran Granite in eager mode, not compiled
The FMS inference script uses `--compile=False` by default. `block.compile()`
is called during model init but the graph is only traced on first forward call.
Don't assume FMS results mean compiled mode works.
**Why:** Led to incorrect assumption that 32-head Granite compiled fine on Spyre.
**How to apply:** When comparing with FMS results, check their compile flags.

## SmolLM3 no_rope_layers: 1 = use RoPE, 0 = skip
Despite the confusing name, `config.no_rope_layers[i] = 1` means USE RoPE.
The field is auto-generated from `no_rope_layer_interval` in the config class.
**Why:** Original adapter had `not no_rope[idx]` which inverted the meaning.
**How to apply:** Use `bool(no_rope[idx])`, never `not no_rope[idx]`.
