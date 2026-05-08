"""Minimal repro: F.interpolate(mode='area') on Spyre with accuracy check."""
import torch
import torch.nn.functional as F
import torch_spyre

def area_downsample(x):
    B, _, C = x.shape
    x = x.view(B, 24, 24, C).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(6, 6), mode="area")
    return x.permute(0, 2, 3, 1).flatten(1, 2)

# Test 1: all ones — avg of 4x4 block of 1.0 should be 1.0
print("=== Test 1: all ones ===")
x_ones = torch.ones(1, 576, 1152, dtype=torch.float16)
ref_ones = area_downsample(x_ones)
compiled = torch.compile(area_downsample, dynamic=False)
x_spyre = x_ones.to("spyre")
with torch.no_grad():
    out_ones = compiled(x_spyre).cpu()
print(f"Expected: all 1.0")
print(f"Spyre first 4: {out_ones[0, 0, :4].tolist()}")
print(f"CPU first 4:   {ref_ones[0, 0, :4].tolist()}")
diff = (out_ones - ref_ones).abs()
print(f"max_diff: {diff.max().item():.6e}")
print()

# Test 2: sequential values — each 4x4 block has known average
print("=== Test 2: sequential integers ===")
torch._dynamo.reset()
x_seq = torch.arange(576, dtype=torch.float16).unsqueeze(0).unsqueeze(-1).expand(1, 576, 1152).contiguous()
ref_seq = area_downsample(x_seq)
compiled2 = torch.compile(area_downsample, dynamic=False)
x_spyre2 = x_seq.to("spyre")
with torch.no_grad():
    out_seq = compiled2(x_spyre2).cpu()
print(f"Spyre first 4 (ch0): {out_seq[0, :4, 0].tolist()}")
print(f"CPU first 4 (ch0):   {ref_seq[0, :4, 0].tolist()}")
diff2 = (out_seq - ref_seq).abs()
print(f"max_diff: {diff2.max().item():.6e}")
print()

# Test 3: random (original test)
print("=== Test 3: random ===")
torch._dynamo.reset()
torch.manual_seed(42)
x_rand = torch.randn(1, 576, 1152, dtype=torch.float16)
ref_rand = area_downsample(x_rand)
compiled3 = torch.compile(area_downsample, dynamic=False)
x_spyre3 = x_rand.to("spyre")
with torch.no_grad():
    out_rand = compiled3(x_spyre3).cpu()
diff3 = (out_rand - ref_rand).abs()
print(f"max_diff: {diff3.max().item():.6e}, mean_diff: {diff3.mean().item():.6e}")
print(f"Spyre first 4: {out_rand[0, 0, :4].tolist()}")
print(f"CPU first 4:   {ref_rand[0, 0, :4].tolist()}")
