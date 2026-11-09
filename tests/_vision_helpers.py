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

"""Shared helpers for the vision/multimodal CPU accuracy tests.

The reference for a vision-tower test is the stock-HF tower run on a
deterministic ``pixel_values`` input. We synthesize ``pixel_values`` directly
(seeded ``torch.randn`` at the tower's native resolution) rather than running an
image through the processor: the tower test certifies the tower, and a fixed
canonical tensor keeps the test fast (no image file, no processor tiling) and
fully deterministic across runs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoProcessor
from transformers import __version__ as transformers_version

from tests.conftest import load_ref_model
from tests.model_registry import REMOTE_CODE_PATHS

# ── VLM (image→text) end-to-end helpers ──────────────────────────────────────
#
# These drive a full multimodal adapter (both towers) the way an application
# would: processor → bound model.generate → decoded text, compared against stock's
# real ``model.generate``. They are model-agnostic given a model path — the only
# convention they bake in is the modern single-call chat-template path, which
# every current HF VLM processor supports and which (for anyres VLMs like Granite
# Vision) produces correct tiling + image-token expansion. Per-model inputs
# (e.g. ``image_sizes``, ``image_grid_thw``) ride along in the returned ``batch``
# dict, so a new VLM needs no change here.

SAMPLE_IMAGE = {
    "repo_id": "huggingface/documentation-images",
    "filename": "pipeline-cat-chonk.jpeg",
    "repo_type": "dataset",
}

VLM_REFERENCE_CACHE_ENV = "HF_ADAPTERS_VLM_REF_CACHE"
_VLM_REFERENCE_CACHE_SCHEMA = 1
# Bump whenever the stock-reference generation algorithm changes in a way that
# is not already represented in the cache metadata below.
_VLM_REFERENCE_ALGORITHM_VERSION = 1


@dataclass(frozen=True)
class VLMReference:
    """The stock-HF outputs needed by the Spyre VLM comparison."""

    logits: list[torch.Tensor]
    token_ids: list[int]
    text: str


# Registry of diverse sample images for multi-image smoke tests.
# All sourced from the public ``huggingface/documentation-images`` dataset —
# no local files required; each is downloaded on first use and cached by
# ``huggingface_hub``.  Each entry carries a short ``label`` (used in test
# output) and a ``prompt`` suited to the scene.
SMOKE_TEST_IMAGES: list[dict] = [
    {
        "label": "cat",
        "repo_id": "huggingface/documentation-images",
        "filename": "pipeline-cat-chonk.jpeg",
        "repo_type": "dataset",
        "prompt": "Describe what you see in this image.",
    },
    {
        "label": "bee",
        "repo_id": "huggingface/documentation-images",
        "filename": "bee.jpg",
        "repo_type": "dataset",
        "prompt": "What type of insect is shown in the image?",
    },
    {
        "label": "car",
        "repo_id": "huggingface/documentation-images",
        "filename": "transformers/tasks/car.jpg",
        "repo_type": "dataset",
        "prompt": "What type of vehicle is shown in this image?",
    },
    {
        "label": "rabbit",
        "repo_id": "huggingface/documentation-images",
        "filename": "transformers/rabbit.png",
        "repo_type": "dataset",
        "prompt": "Describe the animal in this image.",
    },
    {
        "label": "owl",
        "repo_id": "huggingface/documentation-images",
        "filename": "transformers/tasks/owl.jpg",
        "repo_type": "dataset",
        "prompt": "What do you see in this image?",
    },
]


def _load_sample_image() -> Image.Image:
    """A real, recognizable hub image (a chonky cat) so a caption is judgeable.

    Downloaded at test time — no committed fixture. Human-eyeballable output is
    a deliberate secondary signal on top of the token-exact assertion.
    """
    path = hf_hub_download(**SAMPLE_IMAGE)
    return Image.open(path).convert("RGB")


def load_smoke_test_images() -> list[tuple[str, str, Image.Image]]:
    """Download and return all SMOKE_TEST_IMAGES as (label, prompt, image) tuples.

    Each image is downloaded from the HF hub on first call and cached locally
    by ``huggingface_hub`` — subsequent calls are instant (no re-download).
    """
    results = []
    for entry in SMOKE_TEST_IMAGES:
        path = hf_hub_download(
            repo_id=entry["repo_id"],
            filename=entry["filename"],
            repo_type=entry["repo_type"],
        )
        image = Image.open(path).convert("RGB")
        results.append((entry["label"], entry["prompt"], image))
    return results


def extra_image_inputs(fn, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Batch tensors, beyond the standard three, that ``fn`` declares as params.

    VLM adapters take ``input_ids, attention_mask, pixel_values`` and then
    whatever extra image inputs their model needs — Granite Vision / Mistral 3:
    ``image_sizes``; Gemma 4 unified: ``image_position_ids`` +
    ``mm_token_type_ids``. Matching a processor ``batch`` against the callee's
    signature (``adapter.generate`` / ``adapter._prefill_forward``) keeps the CPU
    and Spyre e2e harnesses signature-agnostic across those adapters; the extra
    tensors are forwarded by keyword.
    """
    accepted = set(inspect.signature(fn).parameters)
    standard = {"input_ids", "attention_mask", "pixel_values"}
    return {
        k: v
        for k, v in batch.items()
        if k in accepted and k not in standard and isinstance(v, torch.Tensor)
    }


def build_vlm_batch(
    model_path: str,
    prompt: str,
    image: Image.Image | None = None,
    trust_remote_code: bool | None = None,
) -> tuple[AutoProcessor, dict[str, torch.Tensor]]:
    """Processor + tokenized (image + prompt) batch, the official VLM way.

    Embeds the image in the conversation and lets ``apply_chat_template`` tokenize
    and expand image tokens in one call (``tokenize=True, return_dict=True``). The
    two-step ``processor(text=..., images=...)`` path under-tiles anyres images and
    mis-aligns image tokens, so the documented single-call path is used instead.

    Sets ``padding_side='left'`` to match the adapters' right-aligned decode
    convention. Returns ``(processor, batch)``; ``batch`` carries whatever image
    inputs the model needs (``pixel_values``, ``image_sizes``, …).
    """
    if trust_remote_code is None:
        trust_remote_code = model_path in REMOTE_CODE_PATHS
    if "mistral" in model_path.lower():
        processor = AutoProcessor.from_pretrained(
            model_path, fix_mistral_regex=True, trust_remote_code=trust_remote_code
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    processor.tokenizer.padding_side = "left"

    if image is None:
        image = _load_sample_image()
    conv = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    batch = processor.apply_chat_template(
        conv,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return processor, batch


def resolve_model_revision(
    model_path: str, trust_remote_code: bool | None = None
) -> str | None:
    """Resolve the immutable Hub commit used by ``model_path``.

    Persistent reference caching is disabled when Transformers cannot provide
    a commit hash (for example, for a mutable local model directory). Reusing a
    reference without an immutable model identity would risk hiding a model
    update behind a stale cache hit.
    """
    try:
        config = AutoConfig.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
    except Exception as exc:
        warnings.warn(
            f"Could not resolve {model_path!r} for VLM reference caching: {exc}",
            stacklevel=2,
        )
        return None
    return getattr(config, "_commit_hash", None)


def _batch_digest(batch: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and bytes without dtype conversions."""
    digest = hashlib.sha256()
    for name, tensor in sorted(batch.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"VLM batch value {name!r} is not a tensor")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _reference_cache_key(
    *,
    model_path: str,
    model_revision: str,
    model_dtype: torch.dtype,
    trust_remote_code: bool,
    prompt: str,
    batch: Mapping[str, torch.Tensor],
    max_new_tokens: int,
    num_compare_steps: int,
) -> str:
    metadata = {
        "schema": _VLM_REFERENCE_CACHE_SCHEMA,
        "algorithm": _VLM_REFERENCE_ALGORITHM_VERSION,
        "model_path": model_path,
        "model_revision": model_revision,
        "model_dtype": str(model_dtype),
        "trust_remote_code": trust_remote_code,
        "transformers_version": transformers_version,
        "torch_version": torch.__version__,
        "prompt": prompt,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "num_compare_steps": num_compare_steps,
            "do_sample": False,
            "use_cache": True,
        },
        "batch_sha256": _batch_digest(batch),
    }
    serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _load_cached_vlm_reference(
    path: Path, cache_key: str, num_compare_steps: int
) -> VLMReference | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("cache_key") != cache_key:
            raise ValueError("cache key mismatch")
        logits = payload.get("logits")
        token_ids = payload.get("token_ids")
        text = payload.get("text")
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("logits must be a rank-2 tensor")
        if logits.shape[0] != num_compare_steps or not logits.is_floating_point():
            raise ValueError("unexpected logits shape or dtype")
        if not isinstance(token_ids, torch.Tensor) or token_ids.ndim != 1:
            raise ValueError("token_ids must be a rank-1 tensor")
        if token_ids.shape[0] != num_compare_steps:
            raise ValueError("unexpected token_ids shape")
        if not isinstance(text, str):
            raise ValueError("decoded text must be a string")
        return VLMReference(
            logits=[step.clone() for step in logits.float().unbind()],
            token_ids=[int(token) for token in token_ids.tolist()],
            text=text,
        )
    except Exception as exc:
        warnings.warn(
            f"Ignoring invalid VLM reference cache entry {path}: {exc}",
            stacklevel=2,
        )
        return None


def _save_cached_vlm_reference(
    path: Path, cache_key: str, reference: VLMReference
) -> None:
    """Publish a complete cache file atomically for concurrent CI jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers need directory search permission as well as readable cache files.
    # Override restrictive umasks while preserving shared write and special bits.
    directory_mode = stat.S_IMODE(path.parent.stat().st_mode)
    if directory_mode & 0o055 != 0o055:
        path.parent.chmod(directory_mode | 0o055)
    payload = {
        "cache_key": cache_key,
        "logits": torch.stack(reference.logits).float().cpu(),
        "token_ids": torch.tensor(reference.token_ids, dtype=torch.long),
        "text": reference.text,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def get_or_create_vlm_reference(
    *,
    cache_dir: str | Path | None,
    model_path: str,
    model_revision: str | None,
    model_dtype: torch.dtype,
    trust_remote_code: bool,
    prompt: str,
    batch: Mapping[str, torch.Tensor],
    max_new_tokens: int,
    num_compare_steps: int,
    compute: Callable[[], VLMReference],
) -> tuple[VLMReference, Literal["hit", "saved", "disabled", "write_failed"]]:
    """Load a stock VLM reference, or compute and persist it on a cache miss.

    Returns ``(reference, cache_status)`` with status ``hit``, ``saved``,
    ``disabled``, or ``write_failed``. Caching is deliberately opt-in via
    ``cache_dir`` and requires an immutable model revision. A write failure
    still returns the computed reference.
    """
    if cache_dir is None or model_revision is None:
        return compute(), "disabled"

    cache_key = _reference_cache_key(
        model_path=model_path,
        model_revision=model_revision,
        model_dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        prompt=prompt,
        batch=batch,
        max_new_tokens=max_new_tokens,
        num_compare_steps=num_compare_steps,
    )
    path = Path(cache_dir).expanduser() / f"vlm-reference-{cache_key}.pt"
    if path.is_file():
        cached = _load_cached_vlm_reference(path, cache_key, num_compare_steps)
        if cached is not None:
            return cached, "hit"

    reference = compute()
    if len(reference.logits) != num_compare_steps:
        raise ValueError(
            f"expected {num_compare_steps} reference logit vectors, "
            f"got {len(reference.logits)}"
        )
    if len(reference.token_ids) != num_compare_steps:
        raise ValueError(
            f"expected {num_compare_steps} reference token IDs, "
            f"got {len(reference.token_ids)}"
        )
    try:
        _save_cached_vlm_reference(path, cache_key, reference)
    except OSError as exc:
        warnings.warn(
            f"Could not write VLM reference cache entry {path}: {exc}",
            stacklevel=2,
        )
        return reference, "write_failed"
    return reference, "saved"


def stock_vlm_reference(
    model,
    processor: AutoProcessor,
    batch: dict[str, torch.Tensor],
    max_new_tokens: int,
    num_compare_steps: int,
) -> VLMReference:
    """Generate one stock-HF run for both comparison logits and caption."""
    prompt_len = batch["input_ids"].shape[1]
    with torch.no_grad():
        generation = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True,
        )
    if len(generation.logits) < num_compare_steps:
        raise ValueError(
            f"stock generation produced {len(generation.logits)} steps; "
            f"{num_compare_steps} are required for comparison"
        )
    generated_tokens = generation.sequences[0, prompt_len:]
    return VLMReference(
        logits=[
            generation.logits[step][0].float().cpu().clone()
            for step in range(num_compare_steps)
        ],
        token_ids=[int(token) for token in generated_tokens[:num_compare_steps]],
        text=processor.tokenizer.decode(generated_tokens, skip_special_tokens=True),
    )


def stock_vlm_generate(
    model_path: str,
    processor: AutoProcessor,
    batch: dict[str, torch.Tensor],
    adapter_mod,
    max_new_tokens: int,
    ref_model=None,
    trust_remote_code: bool | None = None,
) -> str:
    """Reference: stock ``AutoModelForImageTextToText.generate`` on ``batch``.

    Loaded via stock HF directly so the reference stays independent of the code
    under test. Returns the decoded **new** text (prompt tokens sliced off).

    Pass ``ref_model`` to reuse an already-loaded stock model.
    """
    from transformers import AutoModelForImageTextToText

    if ref_model is None:
        ref_model = load_ref_model(
            model_path=model_path,
            adapter_mod=adapter_mod,
            trust_remote_code=trust_remote_code,
            auto_model_cls=AutoModelForImageTextToText,
        )

    prompt_len = batch["input_ids"].shape[1]
    with torch.no_grad():
        gen = ref_model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    text = processor.tokenizer.decode(gen[0, prompt_len:], skip_special_tokens=True)
    return text
