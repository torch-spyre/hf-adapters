"""
Minimal repro: deeptools L3 scheduler failure on vision encoder layer.

Reproduces the full SiglipEncoderLayer pattern (LayerNorm + attention +
residual + LayerNorm + MLP + residual) that fails to compile on Spyre.

The simpler attention-only pattern compiles fine — the failure requires
the full layer including LayerNorm and GELU MLP.

Error:
  DtException: There must be at least one valid candidate.,
  file .../deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 1058

Usage:
    python tests/test_vision_sdpa_repro.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "spyre"

# Exact shapes from SiglipVisionModel encoder layer
BATCH = 1
SEQ_LEN = 576       # 24x24 image patches (384/16)^2
HIDDEN = 1152       # vision hidden size
NUM_HEADS = 16
HEAD_DIM = 128      # padded from 72 for stick alignment (72 -> 128)
MLP_DIM = 4304      # intermediate_size


def make_full_vision_layer():
    """Full SiglipEncoderLayer pattern with MLP padded to stick alignment."""
    padded_mlp = ((MLP_DIM + 63) // 64) * 64  # 4304 → 4352
    ln1 = nn.LayerNorm(HIDDEN, eps=1e-6)
    q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=True)
    k_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=True)
    v_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=True)
    out_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=True)
    ln2 = nn.LayerNorm(HIDDEN, eps=1e-6)
    fc1 = nn.Linear(HIDDEN, padded_mlp, bias=True)
    fc2 = nn.Linear(padded_mlp, HIDDEN, bias=True)

    all_modules = [ln1, q_proj, k_proj, v_proj, out_proj, ln2, fc1, fc2]

    def forward(hidden_states):
        bsz, seq_len, _ = hidden_states.shape

        # Pre-attention LayerNorm + residual
        residual = hidden_states
        h = ln1(hidden_states)

        # Self-attention (bidirectional, no mask)
        q = q_proj(h).view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k_proj(h).view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v_proj(h).view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, NUM_HEADS * HEAD_DIM)
        attn_out = out_proj(attn_out)

        h = residual + attn_out

        # Post-attention LayerNorm + MLP + residual
        residual = h
        h = ln2(h)
        h = fc1(h)
        h = F.gelu(h, approximate="tanh")
        h = fc2(h)
        h = residual + h

        return h

    return forward, all_modules


def make_attn_only():
    """Attention-only (this compiles fine — included for comparison)."""
    q_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=True)
    k_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=True)
    v_proj = nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, bias=True)
    out_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, bias=True)

    all_modules = [q_proj, k_proj, v_proj, out_proj]

    def forward(hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        q = q_proj(hidden_states).view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k_proj(hidden_states).view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v_proj(hidden_states).view(bsz, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seq_len, NUM_HEADS * HEAD_DIM)
        return out_proj(attn_out)

    return forward, all_modules


def run_test(name, make_fn):
    print(f"\n--- {name} ---")
    print(f"  B={BATCH}, S={SEQ_LEN}, H={HIDDEN}, heads={NUM_HEADS}, "
          f"head_dim={HEAD_DIM}, mlp={MLP_DIM}")

    forward_fn, modules = make_fn()

    for m in modules:
        m.to(dtype=torch.float16)
        m.eval()
        m.requires_grad_(False)
    for m in modules:
        m.to(device=DEVICE)

    compiled = torch.compile(forward_fn, dynamic=False)
    x = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.float16, device=DEVICE)

    print("  Compiling and running...")
    with torch.no_grad():
        out = compiled(x)

    out_cpu = out.cpu()
    print(f"  Output shape: {list(out_cpu.shape)}")
    print(f"  Has NaN: {out_cpu.isnan().any().item()}")
    print(f"  PASS")
    return True


def make_mlp_only():
    """MLP only: fc1 → gelu → fc2 + residual. Padded to stick-aligned."""
    padded_mlp = ((MLP_DIM + 63) // 64) * 64  # 4304 → 4352
    fc1 = nn.Linear(HIDDEN, padded_mlp, bias=True)
    fc2 = nn.Linear(padded_mlp, HIDDEN, bias=True)

    def forward(hidden_states):
        residual = hidden_states
        h = fc1(hidden_states)
        h = F.gelu(h, approximate="tanh")
        h = fc2(h)
        h = residual + h
        return h

    return forward, [fc1, fc2]


def make_ln_mlp():
    """LayerNorm + MLP: ln → fc1 → gelu → fc2 + residual. Padded MLP."""
    padded_mlp = ((MLP_DIM + 63) // 64) * 64
    ln = nn.LayerNorm(HIDDEN, eps=1e-6)
    fc1 = nn.Linear(HIDDEN, padded_mlp, bias=True)
    fc2 = nn.Linear(padded_mlp, HIDDEN, bias=True)

    def forward(hidden_states):
        residual = hidden_states
        h = ln(hidden_states)
        h = fc1(h)
        h = F.gelu(h, approximate="tanh")
        h = fc2(h)
        h = residual + h
        return h

    return forward, [ln, fc1, fc2]


if __name__ == "__main__":
    import sys
    tests = sys.argv[1:] if len(sys.argv) > 1 else ["attn", "full"]

    for t in tests:
        try:
            if t == "attn":
                run_test("Attention only", make_attn_only)
            elif t == "mlp":
                run_test("MLP only (fc1 + gelu + fc2 + residual)", make_mlp_only)
            elif t == "ln_mlp":
                run_test("LayerNorm + MLP", make_ln_mlp)
            elif t == "full":
                run_test("Full layer (LayerNorm + Attn + MLP)", make_full_vision_layer)
        except Exception as e:
            print(f"  FAILED: {e}")

