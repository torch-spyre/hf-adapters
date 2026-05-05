"""
Minimal repro: deeptools L3 scheduler failure on QFormer cross-attention.

LayerNorm fused with cross-attention SDPA (Q from one source, K/V from another)
produces an SDSC with multiple input data sources that the scheduler rejects.

Error:
  DtException: Expect at most one LabeledDs with DsType INPUT.,
  file .../deeptools/dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp line 3828

Usage:
    python tests/test_qformer_repro.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "spyre"
HIDDEN = 1152
NUM_HEADS = 18
HEAD_DIM = 128  # padded from 64
ALL_HEAD_SIZE = NUM_HEADS * HEAD_DIM  # 2304
QUERY_LEN = 16
ENCODER_LEN = 64
BATCH = 9  # B*n*n (windowed)


class SimpleQFormerAttention(nn.Module):
    """Minimal cross-attention that reproduces the failure."""
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, ALL_HEAD_SIZE)
        self.k_proj = nn.Linear(HIDDEN, ALL_HEAD_SIZE)
        self.v_proj = nn.Linear(HIDDEN, ALL_HEAD_SIZE)
        self.out_proj = nn.Linear(ALL_HEAD_SIZE, HIDDEN)
        self.ln = nn.LayerNorm(HIDDEN)
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM

    def forward(self, query_input, encoder_input):
        # LayerNorm on query
        h = self.ln(query_input)

        # Cross-attention: Q from query, K/V from encoder
        B, S, _ = h.shape
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_input).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_input).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, S, ALL_HEAD_SIZE)
        return self.out_proj(out)


if __name__ == "__main__":
    print(f"QFormer cross-attention repro: B={BATCH}, Q_len={QUERY_LEN}, "
          f"KV_len={ENCODER_LEN}, H={HIDDEN}, heads={NUM_HEADS}, head_dim={HEAD_DIM}")

    model = SimpleQFormerAttention()
    model.to(dtype=torch.float16)
    model.eval()
    model.requires_grad_(False)
    model.to(device=DEVICE)

    compiled = torch.compile(model.forward, dynamic=False)

    q_input = torch.randn(BATCH, QUERY_LEN, HIDDEN, dtype=torch.float16, device=DEVICE)
    kv_input = torch.randn(BATCH, ENCODER_LEN, HIDDEN, dtype=torch.float16, device=DEVICE)

    print("Compiling...")
    with torch.no_grad():
        out = compiled(q_input, kv_input)

    print(f"Output shape: {list(out.cpu().shape)}")
    print("PASS")
