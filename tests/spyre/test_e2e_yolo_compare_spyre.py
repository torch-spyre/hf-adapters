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
E2E Ultralytics YOLO accuracy: stock fp32 CPU model vs ``ultralytics_yolo`` on Spyre.

For each checkpoint the Spyre model is compiled once, then every test image runs
through both it and the unpatched CPU model. Per image, post-NMS detections are
matched by class and box IoU > 0.5; for segmentation checkpoints the final
640x640 binary mask of each matched object is compared too.

Pass criteria:

- every CPU detection is found on Spyre and Spyre adds none, except detections
  whose score is within ``CONF_EDGE`` of the confidence threshold. Those can
  legitimately flip across it under fp16 rounding;
- matched scores agree within ``MAX_SCORE_DIFF``;
- segmentation: every matched mask IoU >= ``MIN_MASK_IOU`` and the mean
  >= ``MIN_MEAN_MASK_IOU``.

Images are the two bundled with the ultralytics package (``bus.jpg``,
``zidane.jpg``), so no image download is needed. Set ``YOLO_TEST_IMAGES`` to a
directory of ``.jpg`` files to test others as well. Weights are resolved by
ultralytics (downloaded on first use) unless ``YOLO_WEIGHTS_DIR`` points at a
directory that already holds them.

Requires ``ultralytics`` and ``opencv``. Do not run with ``SENCORES=1``: a
torch-spyre single-core LX-planning bug silently corrupts results.

Usage (on Spyre pod)::

    pytest -s -vvv tests/spyre/test_e2e_yolo_compare_spyre.py
    pytest -s -vvv tests/spyre/test_e2e_yolo_compare_spyre.py -k seg
"""

import glob
import os

import pytest
import torch

ultralytics = pytest.importorskip("ultralytics")
cv2 = pytest.importorskip("cv2")

import numpy as np  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402
from ultralytics.utils.ops import process_mask  # noqa: E402

from hf_adapters.ultralytics_yolo import load_model, prepare_input  # noqa: E402

WEIGHTS = ["yolov5nu.pt", "yolov8n-seg.pt"]

IMG_SIZE = 640
CONF = 0.25
IOU_NMS = 0.45
BOX_MATCH_IOU = 0.5
CONF_EDGE = 0.01
MAX_SCORE_DIFF = 0.05
MIN_MASK_IOU = 0.9
MIN_MEAN_MASK_IOU = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_images() -> list[str]:
    assets = os.path.join(os.path.dirname(ultralytics.__file__), "assets")
    images = [os.path.join(assets, name) for name in ("bus.jpg", "zidane.jpg")]
    extra = os.environ.get("YOLO_TEST_IMAGES")
    if extra:
        images += sorted(glob.glob(os.path.join(extra, "*.jpg")))
    return images


def _weights_path(name: str) -> str:
    weights_dir = os.environ.get("YOLO_WEIGHTS_DIR")
    if weights_dir and os.path.exists(os.path.join(weights_dir, name)):
        return os.path.join(weights_dir, name)
    return name


def _letterbox(path: str) -> torch.Tensor:
    """Ultralytics-style letterbox to a (1, 3, IMG_SIZE, IMG_SIZE) fp32 tensor."""
    im = cv2.imread(path)
    h, w = im.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
    top, left = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    out[top : top + nh, left : left + nw] = resized
    chw = np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(chw / 255.0).unsqueeze(0)


def _split_output(out) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Detect -> (pred, None); Segment -> (pred, proto). Both as fp32 CPU."""
    first = out[0]
    if isinstance(first, (tuple, list)):
        return first[0].cpu().float(), first[1].cpu().float()
    return first.cpu().float(), None


def _postprocess(pred: torch.Tensor, proto: torch.Tensor | None):
    """Post-NMS rows (x1, y1, x2, y2, score, cls[, mask coeffs]) and binary masks."""
    nc = 80 if proto is not None else 0  # nc=0 lets NMS infer it for detect
    det = non_max_suppression(pred, CONF, IOU_NMS, nc=nc)[0]
    if proto is None:
        return det, None
    if len(det) == 0:
        return det, torch.zeros(0, IMG_SIZE, IMG_SIZE, dtype=torch.bool)
    masks = process_mask(
        proto[0], det[:, 6:], det[:, :4], (IMG_SIZE, IMG_SIZE), upsample=True
    )
    return det, masks > 0.5


def _box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _compare_image(ref, ref_masks, dev, dev_masks, names) -> tuple[list[str], dict]:
    """Greedy class + box-IoU match of Spyre detections to CPU ones.

    Returns (problems, stats). Unmatched detections within CONF_EDGE of CONF are
    reported in stats but are not problems.
    """
    used: set[int] = set()
    problems: list[str] = []
    score_diffs: list[float] = []
    mask_ious: list[float] = []
    edge: list[str] = []
    for i in range(len(ref)):
        best, best_j = BOX_MATCH_IOU, None
        for j in range(len(dev)):
            if j in used or int(dev[j, 5]) != int(ref[i, 5]):
                continue
            iou = _box_iou(ref[i, :4].tolist(), dev[j, :4].tolist())
            if iou > best:
                best, best_j = iou, j
        label = f"{names[int(ref[i, 5])]}@{float(ref[i, 4]):.3f}"
        if best_j is None:
            if float(ref[i, 4]) < CONF + CONF_EDGE:
                edge.append(f"missed {label}")
            else:
                problems.append(f"Spyre missed CPU detection {label}")
            continue
        used.add(best_j)
        diff = abs(float(ref[i, 4]) - float(dev[best_j, 4]))
        score_diffs.append(diff)
        if diff > MAX_SCORE_DIFF:
            problems.append(f"{label}: score differs by {diff:.3f}")
        if ref_masks is not None:
            inter = (ref_masks[i] & dev_masks[best_j]).sum().item()
            union = (ref_masks[i] | dev_masks[best_j]).sum().item()
            mask_iou = inter / union if union else 1.0
            mask_ious.append(mask_iou)
            if mask_iou < MIN_MASK_IOU:
                problems.append(f"{label}: mask IoU {mask_iou:.3f} < {MIN_MASK_IOU}")
    for j in range(len(dev)):
        if j in used:
            continue
        label = f"{names[int(dev[j, 5])]}@{float(dev[j, 4]):.3f}"
        if float(dev[j, 4]) < CONF + CONF_EDGE:
            edge.append(f"extra {label}")
        else:
            problems.append(f"Spyre added detection {label}")
    stats = {
        "cpu": len(ref),
        "spyre": len(dev),
        "max_score_diff": max(score_diffs, default=0.0),
        "mask_ious": mask_ious,
        "edge": edge,
    }
    return problems, stats


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "weights", WEIGHTS, ids=[w.removesuffix(".pt") for w in WEIGHTS]
)
def test_yolo_compare_spyre(weights: str) -> None:
    if os.environ.get("SENCORES") == "1":
        pytest.skip("SENCORES=1 hits a torch-spyre LX-planning bug; use >= 2 cores")

    path = _weights_path(weights)
    cpu_model = load_model(path, device="cpu")
    names = cpu_model.names
    spyre_model = torch.compile(load_model(path, device="spyre"), dynamic=False)

    problems: list[str] = []
    all_mask_ious: list[float] = []
    for image in _test_images():
        x = _letterbox(image)
        with torch.no_grad():
            ref_pred, ref_proto = _split_output(cpu_model(prepare_input(x, "cpu")))
            dev_pred, dev_proto = _split_output(spyre_model(prepare_input(x, "spyre")))
        assert (ref_proto is None) == (dev_proto is None)
        assert torch.isfinite(dev_pred).all(), f"{image}: non-finite Spyre output"
        ref, ref_masks = _postprocess(ref_pred, ref_proto)
        dev, dev_masks = _postprocess(dev_pred, dev_proto)
        image_problems, stats = _compare_image(ref, ref_masks, dev, dev_masks, names)
        problems += [f"{os.path.basename(image)}: {p}" for p in image_problems]
        all_mask_ious += stats["mask_ious"]
        line = (
            f"[{weights}] {os.path.basename(image)}: cpu={stats['cpu']} "
            f"spyre={stats['spyre']} max|dscore|={stats['max_score_diff']:.3f}"
        )
        if ref_proto is not None:
            line += f" min mask IoU {min(stats['mask_ious'], default=1.0):.3f}"
        if stats["edge"]:
            line += f" (threshold edge: {', '.join(stats['edge'])})"
        print(line)

    if all_mask_ious:
        mean_iou = sum(all_mask_ious) / len(all_mask_ious)
        n = len(all_mask_ious)
        print(f"[{weights}] mean mask IoU {mean_iou:.4f} over {n} objects")
        if mean_iou < MIN_MEAN_MASK_IOU:
            problems.append(f"mean mask IoU {mean_iou:.4f} < {MIN_MEAN_MASK_IOU}")
    assert not problems, f"{weights}: Spyre differs from CPU:\n" + "\n".join(problems)
