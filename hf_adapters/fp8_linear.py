# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 (E4M3) linear layer for compressed-tensors checkpoints on Spyre.

Per-output-channel weight scale, per-token dynamic activation scale, no bias.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max
# Floor on the per-token activation scale; matches torch-spyre's
# QUANTSCALEPERTOKENFP8_CLIP_MIN so the CPU path agrees with the device op.
SCALE_CLIP_MIN = 2.0**-23

# TODO: o_proj and down_proj fail torch-spyre FP8 codegen (DtException on
# o_proj, ReStickifyOpHBM on SEN143_FP8 for down_proj); kept fp16 until fixed.
DEFAULT_FP8_EXCLUDE = ("o_proj", "down_proj")


class FP8Linear(nn.Module):
    """FP8 replacement for a bias-free ``nn.Linear``.

    ``weight`` is stored ``[in_features, out_features]``: transposing an E4M3
    tensor on device needs a restickify Spyre cannot emit for FP8.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype
        # Set by prequantize_fp8_weights once `weight` holds E4M3/QFP8WT.
        self.prequantized = False

        self.register_buffer(
            "weight", torch.empty(in_features, out_features, dtype=dtype)
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)

        if x.device.type == "spyre":
            # Fused per-token scale (torch-spyre#3457). clone() gives x_scale a
            # fresh layout; inherited rank-reducing ancestry otherwise breaks
            # the multi-arg rescale in layout propagation.
            x_scale = torch.ops.spyre.quantscalepertokenfp8(x, FP8_MAX).clone()
            wq = (
                self.weight
                if self.prequantized
                else torch.ops.spyre.quantize_weight_fp8_with_scale(
                    self.weight, self.weight_scale
                )
            )
            xq = torch.ops.spyre.quantize_fp8_with_scale(x, x_scale)
            y: torch.Tensor = torch.ops.spyre.scaled_mm(
                xq.reshape(-1, xq.shape[-1]), wq, out_dtype=self.compute_dtype
            )
            y = y.reshape(out_shape) * x_scale * self.weight_scale
            return y.to(x.dtype)

        # CPU reference path, in fp32: the reciprocal of a small scale
        # (SCALE_CLIP_MIN, or a tiny weight_scale) overflows fp16, and fp32
        # accumulation avoids fp16 overflow.
        xf = x.to(torch.float32)
        w_scale = self.weight_scale.to(torch.float32)
        x_scale = (xf.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(
            min=SCALE_CLIP_MIN
        )
        wq = (self.weight.to(torch.float32) / w_scale).clamp(-FP8_MAX, FP8_MAX)
        xq = (xf / x_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        acc = xq.reshape(-1, xq.shape[-1]).to(torch.float32) @ wq.to(FP8_DTYPE).to(
            torch.float32
        )
        return (acc.reshape(out_shape) * x_scale * w_scale).to(x.dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"prequantized={self.prequantized}"
        )


def is_compressed_fp8_config(config: object) -> bool:
    """True for a compressed-tensors checkpoint with 8-bit float weights."""
    quant = getattr(config, "quantization_config", None)
    to_dict = getattr(quant, "to_dict", None)
    if callable(to_dict):  # HF may wrap the dict in a QuantizationConfig
        quant = to_dict()
    if not isinstance(quant, dict):
        return False
    if quant.get("quant_method") != "compressed-tensors":
        return False
    if quant.get("format") != "float-quantized":
        return False
    for group in (quant.get("config_groups") or {}).values():
        weights = group.get("weights") or {}
        if weights.get("type") == "float" and weights.get("num_bits") == 8:
            return True
    return False


def _checkpoint_weight_scale(name: str, linear: nn.Linear) -> torch.Tensor:
    """Checkpoint ``weight_scale`` as fp32 ``[out, 1]``.

    Accepts per-tensor (scalar) and per-channel (``[out]`` / ``[out, 1]``)
    scales; block- and group-wise scales are rejected rather than broadcast.
    """
    scale = getattr(linear, "weight_scale", None)
    if not isinstance(scale, torch.Tensor):
        raise ValueError(f"{name}: E4M3 weight has no weight_scale")
    out_features = linear.out_features
    if scale.numel() == 1:
        return scale.to(torch.float32).reshape(1, 1).expand(out_features, 1)
    if tuple(scale.shape) not in ((out_features,), (out_features, 1)):
        raise NotImplementedError(
            f"{name}: unsupported weight_scale shape {tuple(scale.shape)} for "
            f"weight {tuple(linear.weight.shape)}; only per-tensor and "
            "per-channel FP8 checkpoints are supported (block/group-wise are not)"
        )
    return scale.to(torch.float32).reshape(out_features, 1)


def _dequantize_checkpoint_weight(name: str, linear: nn.Linear) -> torch.Tensor:
    """E4M3 ``[out, in]`` * ``weight_scale [out, 1]`` -> fp16 ``[out, in]``."""
    scale = _checkpoint_weight_scale(name, linear)
    return (linear.weight.to(torch.float32) * scale).to(torch.float16)


def _check_swap_supported(model: nn.Module, targets: list[str]) -> None:
    """Reject checkpoints the swap would silently mishandle, before mutating."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        # HF's TP loader shards weights and installs hooks before the swap;
        # replacement modules would not inherit them.
        raise NotImplementedError(
            "FP8Linear does not support tensor parallelism; load FP8 "
            "checkpoints without tp_plan"
        )
    # FP8Linear has no bias and excluded layers are rebuilt bias-free.
    biased = [name for name in targets if model.get_submodule(name).bias is not None]
    if biased:
        raise NotImplementedError(
            f"swap_linears_to_fp8 does not support biased Linears, got "
            f"{len(biased)}: {', '.join(biased[:4])}"
            f"{', ...' if len(biased) > 4 else ''} (attention_bias/mlp_bias=True?)"
        )
    for name in targets:
        submodule = model.get_submodule(name)
        assert isinstance(submodule, nn.Linear)
        _checkpoint_weight_scale(name, submodule)


def swap_linears_to_fp8(
    model: nn.Module,
    *,
    exclude: tuple[str, ...] | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[int, int]:
    """Replace a compressed-tensors checkpoint's E4M3 Linears with FP8Linear.

    Must run before the first forward: compressed-tensors decompresses E4M3
    weights in place on first use. Returns ``(n_swapped, n_excluded)``.
    """
    if exclude is None:
        exclude = DEFAULT_FP8_EXCLUDE

    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.dtype == FP8_DTYPE
    ]
    if targets:
        _check_swap_supported(model, targets)

    n_swapped = n_excluded = 0
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)
        weight_fp16 = _dequantize_checkpoint_weight(name, old)  # [out, in]

        # Excluded projections are still dequantized: casting untouched E4M3
        # weights to fp16 later would drop weight_scale.
        if attr in exclude or any(part in exclude for part in name.split(".")):
            plain = nn.Linear(
                old.in_features, old.out_features, bias=False, dtype=dtype
            )
            plain.weight.data.copy_(weight_fp16)
            setattr(parent, attr, plain)
            n_excluded += 1
            continue

        fp8 = FP8Linear(old.in_features, old.out_features, dtype=dtype)
        fp8.weight.copy_(weight_fp16.t().contiguous())  # -> [in, out]
        fp8.weight_scale.copy_(
            _checkpoint_weight_scale(name, old).reshape(-1).to(dtype)
        )
        setattr(parent, attr, fp8)
        n_swapped += 1

    return n_swapped, n_excluded


def prequantize_fp8_weights(model: nn.Module) -> int:
    """Quantize each FP8Linear weight to E4M3/QFP8WT once, in place.

    Run after the device move and before the first forward (compile is lazy).
    Frees the fp16 copy and removes the per-call weight quantize from forward.
    Returns the number of modules quantized.
    """
    # TODO: transfer E4M3 directly into QFP8WT once supported; drops the CPU
    # dequantize and this on-device pass.
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, FP8Linear) or m.prequantized:
            continue
        if m.weight.device.type != "spyre":  # QFP8WT is a device-only layout
            continue

        wq = torch.ops.spyre.quantize_weight_fp8_with_scale(m.weight, m.weight_scale)
        if not isinstance(wq, torch.Tensor) or wq.dtype != FP8_DTYPE:
            raise RuntimeError(
                f"quantize_weight_fp8_with_scale returned "
                f"{getattr(wq, 'dtype', type(wq).__name__)} for {name!r}: "
                f"FP8 weight prequantization requires eager FP8 op support in "
                f"torch-spyre (torch-spyre#3172)."
            )

        m.weight = wq  # rebinding drops the fp16 buffer
        m.prequantized = True
        n += 1

    if n:
        print(f"FP8: {n} weight(s) prequantized to E4M3/QFP8WT")
    return n


def fp8_status(model: nn.Module) -> dict:
    """Counts used to verify the FP8 swap and prequantization took effect."""
    status = {
        "n_fp8": 0,
        "n_prequantized": 0,
        "n_weight_fp8": 0,
        "n_unswapped_e4m3": 0,
        "orientation_ok": True,
    }
    for _, m in model.named_modules():
        if isinstance(m, FP8Linear):
            status["n_fp8"] += 1
            status["n_prequantized"] += int(m.prequantized)
            status["n_weight_fp8"] += int(m.weight.dtype == FP8_DTYPE)
            if tuple(m.weight.shape) != (m.in_features, m.out_features):
                status["orientation_ok"] = False
        elif isinstance(m, nn.Linear) and m.weight.dtype == FP8_DTYPE:
            status["n_unswapped_e4m3"] += 1
    return status
