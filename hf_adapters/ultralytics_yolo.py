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

"""
Spyre adapter for Ultralytics YOLO detection and segmentation models
(verified: yolov5nu, yolov8n-seg).

YOLO is **not** a ``transformers`` architecture (no ``config.json`` /
``AutoModel``); it loads only via ``from ultralytics import YOLO``. That is why
this module is named ``ultralytics_yolo`` rather than ``hf_*``: the ``hf_*.py``
files are the Hugging Face adapters that ``AutoSpyreModel``,
``tests/model_registry.py`` and the weekly report enumerate, and none of that
machinery applies here. Import it directly. It follows the same shape as the
``hf_*`` adapters (``patch_*`` functions that rebind ``forward`` on module
instances, guarded on ``device.type == "spyre"`` with a CPU-equivalent fallback,
plus a ``load_model`` entry that applies them) so the split lives in the model
definition rather than in every driver.

What this adapter handles:

    image [1, 3, 640, 640]
      → CSP-Darknet backbone + PAN neck                     (Spyre, compiled)
          except: SPPF max-pools + concat, nn.Upsample      (CPU, graph break)
      → Detect head per-scale convolutions (cv2 / cv3)      (Spyre, compiled)
      → per-scale flatten + concat, box decode, DFL         (CPU)
      → [1, 84, 8400]

Needs only one torch-spyre patch: "route non-stick-aligned conv reshape through
CPU fallback" (the conv decomposition's final reshape). Max-pool, upsample and
the head flatten are handled here instead of as compiler fallbacks.

Spyre adaptations:
- **SPPF max-pools** (``patch_sppf_pool_on_cpu``) and **nearest upsample**
  (``patch_upsample_on_cpu``) run on CPU behind graph breaks: Spyre has neither
  op, and Inductor's decompositions (masked max reduction, ``index_expr``
  gather) cannot be layout-solved.
- The head's **per-scale flatten + concat** (``patch_detect_flatten_on_cpu``)
  runs on CPU: the merged view of a sub-stick-wide cell output has no feasible
  device layout for the dense ``cat``. The head convs still run on device.
- The **box decode** (``Detect._inference``) is moved to CPU behind a Dynamo
  graph break. The decode is a thin, irregular, non-convolutional tail whose
  anchor-grid and ``dist2bbox`` ops keep producing sub-stick layouts the backend
  cannot place (RFC walls 9-12). Moving it wholesale is cheaper than a per-op
  CPU-fallback and cleanly isolates the on-device convolutional network. The
  learned convolutions of the head (``cv2``/``cv3``) stay in ``Detect.forward``,
  on device and in the compiled graph.
- ``model.fuse()`` folds BatchNorm into the preceding conv (removes the
  channel-broadcast affine add the layout solver cannot place, RFC wall 1), and
  fuse + ``.half()`` happen on CPU *before* ``.to("spyre")`` (a fp32 eager
  multiply inside ``fuse`` would otherwise hit fp32 restickify, RFC wall 2).

- Segment heads (e.g. ``yolov8n-seg``) work too: the decode wrapper carries the
  ``mask_coefficient`` payload to CPU with the rest, and the mask-prototype
  net's ``ConvTranspose2d`` (``proto.upsample``) runs on CPU behind a graph break
  (torch-spyre does not lower transposed conv). The other proto convs and the
  ``cv4`` mask-coefficient convs stay on device.
- The input image is cast to fp16 **on the host** before upload
  (``prepare_input``). ``x.to("spyre").half()`` casts on device, and the
  on-device FP32->DL16 cast stamps a staggered element arrangement that the
  compiled graph reads as a permuted stem input (0 detections). Same root cause
  as the read-side ``.cpu().float()`` rule in ``_to_cpu_f32``.

Like the hf-adapters patches, the patch is a no-op on CPU: when the head's
inputs are not Spyre tensors it calls the original ``_inference`` unchanged, so
CPU outputs stay identical to stock ultralytics.

Do not run with ``SENCORES=1``: single-core LX planning loses LX-resident
intermediates read by ``identity`` ops (every computed ``cat`` input and every
LX dump around a CPU-fallback call), which corrupts all C3 blocks. Any core count
>= 2 is correct; ``LX_PLANNING=0`` also avoids it.

Usage::

    from hf_adapters.ultralytics_yolo import load_model, prepare_input
    model = load_model("yolov5nu.pt", device="spyre")   # patched + fused + fp16
    compiled = torch.compile(model, dynamic=False)
    with torch.no_grad():
        out = compiled(prepare_input(x, "spyre"))       # (1, 84, 8400)
"""

import copy
import types

import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import SPPF
from ultralytics.nn.modules.head import Detect


def _first_tensor(x):
    """The head's input is a dict {boxes, scores, feats:[...]} (fused output) or a
    list of per-scale feature maps. Return any leaf tensor to sniff its device."""
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(x, torch.Tensor):
        return x
    return None


def _to_cpu_f32(x):
    """Move the head's pre-decode payload to fp32 CPU, preserving container shape.
    Decode is numerically sensitive (anchor arithmetic, sigmoid), so fp32 on CPU
    both matches the stock reference and keeps the compiled region from having to
    place any of it."""
    if isinstance(x, dict):
        return {k: _to_cpu_f32(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_to_cpu_f32(v) for v in x)
    if isinstance(x, torch.Tensor):
        # fp16-cast-stagger fix: transfer fp16 to host FIRST, cast on host.
        # A device-side .float() stamps a staggered EA (DL16_TO_FP32) the
        # EA-blind D2H copy-out reads back permuted; .cpu().float() is faithful.
        return x.cpu().float()
    return x


def patch_detect_decode_on_cpu(*detects):
    """Patch ``Detect`` *instances* so box decode runs on CPU on Spyre.

    Binds a new ``_inference`` that, when the head's inputs are Spyre tensors,
    moves the pre-decode payload to fp32 CPU and runs the *original* decode there
    behind ``@torch._dynamo.disable`` — a graph break that keeps decode out of
    the compiled graph. When the inputs are not Spyre tensors (a plain CPU run)
    it calls the original ``_inference`` unchanged, so CPU behaviour is identical
    to stock ultralytics.

    Patches the given **instances** (binds a new ``_inference``), not the
    ``Detect`` class, mirroring ``patch_layernorm`` — so it never perturbs a
    ``Detect`` head in another model loaded in the same process.

    Decode reads ``self.stride`` (a small device buffer); it is swapped to CPU
    transiently around the original call and restored after. The fixed ``dfl``
    conv is moved to CPU **permanently** by ``load_model`` before compile —
    NOT save/restored here. Restoring a device buffer that the traced graph
    captured injects a stray sub-stick device-to-device copy that trips the SDSC
    scheduler (``sdsc_fused_copy_from_d2d`` DtException); the working manual
    split proves only the transient ``stride`` swap is needed, and ``anchors``/
    ``strides`` need not be touched at all — ``_inference`` rebuilds anchors from
    the CPU feats it is handed.

    Args:
        *detects: ``Detect`` instances to patch (skips ``None``).
    """
    for head in detects:
        if head is None:
            continue
        assert isinstance(
            head, Detect
        ), f"patch_detect_decode_on_cpu expects Detect instances, got {type(head)}"

        orig_inference = head._inference.__func__

        @torch._dynamo.disable
        def _inference_cpu(self, x, _orig=orig_inference):
            # A Spyre head is recognised by its stride buffer: with
            # patch_detect_flatten_on_cpu the payload already arrives on CPU.
            t = _first_tensor(x)
            stride = getattr(self, "stride", None)
            on_spyre = (t is not None and t.device.type == "spyre") or (
                isinstance(stride, torch.Tensor) and stride.device.type == "spyre"
            )
            if not on_spyre:
                # CPU (or any non-Spyre) run: stock decode, bit-identical to HF.
                return _orig(self, x)

            # Move every payload key: Detect's is {boxes, scores, feats:[...]};
            # Segment adds mask_coefficient, which its _inference concatenates
            # onto the decoded predictions.
            # fp16-cast-stagger fix: .cpu().float() (host cast), not
            # .float().cpu() (device cast -> staggered EA -> scrambled copy-out).
            x_cpu = _to_cpu_f32(x)

            # Swap ONLY self.stride to CPU for the call, then restore. Do not
            # touch anchors/strides/dfl (see docstring): restoring extra device
            # buffers the graph captured trips the SDSC copy_from_d2d wall.
            saved_stride = getattr(self, "stride", None)
            if isinstance(saved_stride, torch.Tensor):
                self.stride = saved_stride.cpu()
            try:
                return _orig(self, x_cpu)
            finally:
                if isinstance(saved_stride, torch.Tensor):
                    self.stride = saved_stride

        head._inference = types.MethodType(_inference_cpu, head)


def _find_detect(model):
    return [m for m in model.modules() if isinstance(m, Detect)]


class _Precomputed:
    """Stands in for a head ``ModuleList``: ``[i](x)`` returns the i-th output
    already computed on device and moved to CPU, ignoring ``x``."""

    def __init__(self, outs):
        self.outs = outs

    def __getitem__(self, i):
        return lambda _x, _t=self.outs[i]: _t

    def __len__(self):
        return len(self.outs)


def patch_detect_flatten_on_cpu(*detects):
    """Run the head's per-scale flatten + concat on CPU; keep its convs on device.

    Stock ``forward_head`` does ``cat([cell[i](x[i]).view(bs, c, -1) ...], -1)``.
    Each cell output has a sub-stick innermost width (80/40/20), and the merged
    view that feeds the dense ``cat`` has no feasible device layout. The patched
    ``forward_head`` runs every head cell on device, then crosses one graph break
    to CPU and calls the *stock* ``forward_head`` with proxy heads that return
    those outputs. That reuses ultralytics' own view/cat code, which covers
    ``Segment``'s ``mask_coefficient`` too.

    Args:
        *detects: ``Detect`` (or subclass) instances to patch.
    """
    for head in detects:
        orig_forward_head = type(head).forward_head

        @torch._dynamo.disable
        def _flatten_cpu(self, x, outs, _orig=orig_forward_head):
            proxies = {
                k: None if v is None else _Precomputed(_to_cpu_f32(v))
                for k, v in outs.items()
            }
            return _orig(self, _to_cpu_f32(x), **proxies)

        def _forward_head(self, x, **heads):
            outs = {
                k: None if h is None else [h[i](x[i]) for i in range(self.nl)]
                for k, h in heads.items()
            }
            return _flatten_cpu(self, x, outs)

        head.forward_head = types.MethodType(_forward_head, head)


@torch._dynamo.disable
def _sppf_pools_cpu(pool, n, y0):
    """SPPF's ``n`` chained max-pools and channel concat, on CPU in one trip."""
    y = [y0.cpu().float()]
    y.extend(pool(y[-1]) for _ in range(n))
    return torch.cat(y, 1).half().to(y0.device)


def patch_sppf_pool_on_cpu(model):
    """Run each ``SPPF`` block's max-pools on CPU; keep its two convs on device.

    Spyre has no max-pool, and Inductor's padded max-pool decomposition (border
    mask + ``where(-inf)`` + max reduction) cannot be layout-solved. The patched
    ``forward`` mirrors stock ``SPPF.forward``: ``cv1`` on device, the pools and
    concat on CPU (one transfer each way rather than one per pool), then ``cv2``
    on device.
    """
    for mod in model.modules():
        if not isinstance(mod, SPPF):
            continue

        def _forward(self, x):
            y = self.cv2(_sppf_pools_cpu(self.m, getattr(self, "n", 3), self.cv1(x)))
            return y + x if getattr(self, "add", False) else y

        mod.forward = types.MethodType(_forward, mod)


def patch_upsample_on_cpu(model):
    """Run every ``nn.Upsample`` on CPU behind a graph break.

    Spyre has no upsample op, and Inductor lowers nearest upsample to an
    ``index_expr`` gather the layout solver cannot place. ``forward`` is rebound
    on the instance (rather than the module replaced) so the ultralytics routing
    attributes (``.f``, ``.i``) stay intact.
    """
    for mod in model.modules():
        if not isinstance(mod, torch.nn.Upsample):
            continue

        @torch._dynamo.disable
        def _forward(self, x, _orig=torch.nn.Upsample.forward):
            return _orig(self, x.cpu().float()).half().to(x.device)

        mod.forward = types.MethodType(_forward, mod)


class _ConvTransposeOnCPU(torch.nn.Module):
    """Runs a ``ConvTranspose2d`` on CPU in fp32 behind a Dynamo graph break.

    torch-spyre's conv decomposition rejects transposed convolution, and the
    equivalent matmul + depth-to-space needs a stride-2 scatter across the stick
    dim that the layout solver cannot place. The Segment head's ``Proto`` has one
    (k2 s2 upsample of the 80x80 P3 map), so it runs here instead. Both
    crossings cast on the host (``.cpu().float()`` out, ``.half().to()`` back)
    to avoid the staggered-EA device casts.
    """

    def __init__(self, convt):
        super().__init__()
        self.convt = convt.float().cpu()

    @torch._dynamo.disable
    def forward(self, x):
        y = self.convt(x.cpu().float())
        return y.half().to(x.device)


def patch_proto_upsample_on_cpu(*heads):
    """Swap each Segment head's ``proto.upsample`` for a CPU ConvTranspose2d.

    Takes fp32 CPU copies captured before the model moved to device, so no
    device parameter is moved back to CPU.

    Args:
        *heads: ``(head, cpu_convt)`` pairs; ``cpu_convt`` is ``None`` for heads
            without a ``proto`` (plain ``Detect``).
    """
    for head, cpu_convt in heads:
        if cpu_convt is not None:
            head.proto.upsample = _ConvTransposeOnCPU(cpu_convt)


def prepare_input(x, device="spyre"):
    """Move a CPU image batch to ``device`` in the dtype ``load_model`` expects.

    On Spyre the cast to fp16 happens on the host, then the fp16 tensor is
    uploaded (``x.half().to(device)``). Casting after the upload
    (``x.to(device).half()``) runs an on-device FP32->DL16 cast whose staggered
    element arrangement the compiled stem conv reads permuted. On CPU the input
    stays fp32, matching the unpatched reference model.
    """
    if str(device) == "cpu":
        return x.float().cpu()
    return x.half().to(device)


def load_model(weights="yolov5nu.pt", device="spyre"):
    """Load YOLOv5n as a raw ``nn.Module``, fused, with decode patched to CPU.

    Ordering matters (RFC walls 1-2): ``fuse()`` folds BatchNorm into conv and
    does an eager ``weight * bn_scale`` multiply, and ``.half()`` must precede
    ``.to(device)`` so that multiply never runs as fp32 on Spyre. The decode
    patch (and the dfl->CPU pin) are applied **after** ``.to(device)``, matching
    the proven manual split runner exactly. For a CPU device the model stays
    fp32 and is returned unpatched (stock ``_inference`` already runs on CPU),
    so ``load_model(..., device="cpu")`` gives a bit-identical stock reference.

    Args:
        weights: ultralytics weights path (e.g. ``yolov5nu.pt``).
        device: target device string (``"spyre"`` or ``"cpu"``).

    Returns:
        The fused, patched ``nn.Module`` on ``device``. Wrap in
        ``torch.compile(model, dynamic=False)`` before running on Spyre.
    """
    m = YOLO(weights).model.eval()
    m.fuse(verbose=False)
    heads = _find_detect(m)

    # CPU device: stock model, no patch. Stock _inference already runs on CPU,
    # so wrapping it is unnecessary -- and, more importantly, installing the
    # @torch._dynamo.disable wrapper on a CPU instance first would register that
    # disabled code object in Dynamo's cache before the Spyre instance is traced,
    # perturbing how the Spyre graph breaks. Keep the CPU reference pristine.
    if str(device) == "cpu":
        return m.to(device)

    # Segment heads: keep an fp32 CPU copy of proto.upsample (ConvTranspose2d)
    # taken before the move, so the CPU fallback never pulls device params back.
    cpu_convts = [
        (
            head,
            (
                copy.deepcopy(head.proto.upsample).float()
                if isinstance(getattr(head, "proto", None), torch.nn.Module)
                and isinstance(head.proto.upsample, torch.nn.ConvTranspose2d)
                else None
            ),
        )
        for head in heads
    ]

    m = m.half()
    m = m.to(device)

    # Order matches the proven manual split runner EXACTLY: move to device
    # first, THEN pin dfl to CPU fp32, THEN install the decode-on-CPU wrapper.
    # (The earlier ordering -- patch before .to(device) -- was the one remaining
    # construction delta from the working baseline; a back-to-back same-cache run
    # showed the manual seam compiling the identical copy_from_d2d kernel clean
    # while the adapter crashed it, so the difference is real and here.)
    for head in heads:
        dfl = getattr(head, "dfl", None)
        if dfl is not None:
            head.dfl = dfl.float().cpu()
    patch_detect_decode_on_cpu(*heads)
    patch_detect_flatten_on_cpu(*heads)
    patch_proto_upsample_on_cpu(*cpu_convts)
    patch_sppf_pool_on_cpu(m)
    patch_upsample_on_cpu(m)

    return m
