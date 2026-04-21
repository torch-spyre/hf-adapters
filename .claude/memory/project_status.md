---
name: HF Adapters Project Status
description: "Current adapter status and what's in progress — see ARCHITECTURE.md for the full compatibility matrix"
type: project
---

The full model compatibility matrix lives in `ARCHITECTURE.md` (single source of truth).

## Notable status (2026-04-21)

- Granite 3.3 2B works on Spyre after head-dim padding (64→128) via `pad_attention_heads()` — added in PR #5
- Phi-4 mini adapter code is complete but blocked on Spyre by sub-stick `head_dim=96` — needs compiler fix
- 5 models fully working on Spyre: Qwen3 0.6B, Granite 3.3 8B, Granite 3.3 2B, Granite 4.0 1B, SmolLM3 3B

**Why:** Track which models work so new adapter work starts from the right baseline.
**How to apply:** Update `ARCHITECTURE.md` when a new adapter is verified on CPU or Spyre.
