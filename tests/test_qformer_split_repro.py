"""
Test: split cross-attention into 3 compiled blocks:
  1. LN + Q projection (single input)
  2. K/V projection (single input)
  3. SDPA + output projection (Q, K, V inputs — like LLM decode pattern)

Hypothesis: the L3 scheduler chokes on fused LN+SDPA with two input streams,
but SDPA alone with multiple pre-computed inputs works (like LLM decode).

Usage:
    python tests/test_qformer_split_repro.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "spyre"
HIDDEN = 1152
NUM_HEADS = 18
HEAD_DIM = 128
ALL_HEAD_SIZE = NUM_HEADS * HEAD_DIM  # 2304
QUERY_LEN = 16
ENCODER_LEN = 64
BATCH = 9


class QProjBlock(nn.Module):
    """LN + Q projection: single input → Q in attention shape."""
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(HIDDEN)
        self.q_proj = nn.Linear(HIDDEN, ALL_HEAD_SIZE)

    def forward(self, query_input):
        h = self.ln(query_input)
        B, S, _ = h.shape
        Q = self.q_proj(h).view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        return Q


class KVProjBlock(nn.Module):
    """K/V projection: single input → K, V in attention shape."""
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(HIDDEN, ALL_HEAD_SIZE)
        self.v_proj = nn.Linear(HIDDEN, ALL_HEAD_SIZE)

    def forward(self, encoder_input):
        B = encoder_input.shape[0]
        K = self.k_proj(encoder_input).view(B, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        V = self.v_proj(encoder_input).view(B, -1, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        return K, V


class SDPAOutBlock(nn.Module):
    """SDPA + output projection: takes Q, K, V (like LLM decode takes Q + KV cache)."""
    def __init__(self):
        super().__init__()
        self.out_proj = nn.Linear(ALL_HEAD_SIZE, HIDDEN)

    def forward(self, Q, K, V):
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(Q.shape[0], Q.shape[2], ALL_HEAD_SIZE)
        return self.out_proj(out)


if __name__ == "__main__":
    print(f"QFormer 3-block split: B={BATCH}, Q_len={QUERY_LEN}, "
          f"KV_len={ENCODER_LEN}, H={HIDDEN}, heads={NUM_HEADS}, head_dim={HEAD_DIM}")
    print()

    q_proj_block = QProjBlock().to(dtype=torch.float16)
    q_proj_block.eval()
    q_proj_block.requires_grad_(False)
    q_proj_block.to(device=DEVICE)

    kv_proj_block = KVProjBlock().to(dtype=torch.float16)
    kv_proj_block.eval()
    kv_proj_block.requires_grad_(False)
    kv_proj_block.to(device=DEVICE)

    sdpa_block = SDPAOutBlock().to(dtype=torch.float16)
    sdpa_block.eval()
    sdpa_block.requires_grad_(False)
    sdpa_block.to(device=DEVICE)

    q_input = torch.randn(BATCH, QUERY_LEN, HIDDEN, dtype=torch.float16, device=DEVICE)
    kv_input = torch.randn(BATCH, ENCODER_LEN, HIDDEN, dtype=torch.float16, device=DEVICE)

    compiled_q = torch.compile(q_proj_block.forward, dynamic=False)
    compiled_kv = torch.compile(kv_proj_block.forward, dynamic=False)
    compiled_sdpa = torch.compile(sdpa_block.forward, dynamic=False)

    # Step 1: Q projection (single input)
    print("Step 1: Compiling Q block (LN + Q_proj, single input)...")
    with torch.no_grad():
        Q = compiled_q(q_input)
    print(f"  Q shape: {list(Q.cpu().shape)}")
    print("  PASS")
    print()

    # Step 2: KV projection (single input)
    print("Step 2: Compiling KV block (K_proj + V_proj, single input)...")
    with torch.no_grad():
        K, V = compiled_kv(kv_input)
    print(f"  K shape: {list(K.cpu().shape)}, V shape: {list(V.cpu().shape)}")
    print("  PASS")
    print()

    # Step 3: SDPA + out_proj (Q, K, V as inputs)
    print("Step 3: Compiling SDPA block (Q + K + V → attention + out_proj)...")
    with torch.no_grad():
        out = compiled_sdpa(Q, K, V)
    print(f"  Output shape: {list(out.cpu().shape)}")
    print("  PASS")
    print()

    print("ALL PASS")
