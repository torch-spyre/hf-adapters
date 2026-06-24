# Comparison with FMS `eager_spyre` Approach

How the HF adapter approach differs from the FMS (Foundation Model
Stack) `eager_spyre` compilation path.

| Aspect | FMS `eager_spyre` | HF Adapter |
|---|---|---|
| Model source | FMS (custom codebase) | HuggingFace Transformers (standard) |
| RoPE | Matmul with `selected_freqs` | Same approach, applied via monkey-patch |
| RMSNorm | `torch.nn.RMSNorm` | `patch_rmsnorm(cls)` patches HF's RMSNorm in-place |
| KV cache | Tuple of tensors through FMS attention | List of tensors passed to compiled block function |
| GQA | Manual expand/flatten | `F.scaled_dot_product_attention(enable_gqa=True)` |
| Compilation | `block.compile(dynamic=False)` | `torch.compile(block_forward, dynamic=False)` |
| Generation | `fms.utils.generation` | `hf_common.generate()` |
| Weight loading | FMS serialization | HF `from_pretrained` then `.to("spyre")` |
| Partial RoPE | Not supported | Identity-padded rotation matrices (`PartialPrecomputedRotaryEmbedding`) |
| Fused weights | N/A | Split at prepare time (QKV, gate+up) |
| Large vocab | N/A | `pad_lm_head()` pads to a smooth stick count (single head fits the 256 MB EAR limit; `chunk_lm_head` is an unused fallback) |
| Sub-stick head_dim | Not supported | `pad_attention_heads()` zero-pads Q/K/V/O and RoPE freqs |
| Maintenance | Requires FMS fork | No fork; runtime monkey-patches on stock HF |
