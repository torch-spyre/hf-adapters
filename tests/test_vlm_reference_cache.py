# Copyright 2026 The Torch-Spyre Authors.
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

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests._vision_helpers import (
    VLMReference,
    get_or_create_vlm_reference,
    stock_vlm_reference,
)


def _reference(value: float = 1.0) -> VLMReference:
    return VLMReference(
        logits=[torch.full((4,), value), torch.full((4,), value + 1)],
        token_ids=[1, 2],
        text=f"caption {value}",
    )


def _get_reference(
    cache_dir: Path,
    compute,
    *,
    batch: dict[str, torch.Tensor] | None = None,
    max_new_tokens: int = 4,
):
    return get_or_create_vlm_reference(
        cache_dir=cache_dir,
        model_path="org/model",
        model_revision="0123456789abcdef",
        model_dtype=torch.float16,
        trust_remote_code=False,
        prompt="describe",
        batch=batch or {"input_ids": torch.tensor([[1, 2]])},
        max_new_tokens=max_new_tokens,
        num_compare_steps=2,
        compute=compute,
    )


def test_cache_miss_saves_and_hit_skips_compute(tmp_path: Path) -> None:
    calls = 0

    def compute() -> VLMReference:
        nonlocal calls
        calls += 1
        return _reference()

    first, first_status = _get_reference(tmp_path, compute)
    second, second_status = _get_reference(tmp_path, compute)

    assert first_status == "saved"
    assert second_status == "hit"
    assert calls == 1
    assert len(list(tmp_path.glob("vlm-reference-*.pt"))) == 1
    assert torch.equal(first.logits[0], second.logits[0])
    assert first.token_ids == second.token_ids
    assert first.text == second.text


@pytest.mark.parametrize("existing_mode", [None, 0o700, 0o770, 0o2775])
def test_cache_permissions_allow_shared_reads(
    tmp_path: Path, existing_mode: int | None
) -> None:
    cache_dir = tmp_path / "references"
    if existing_mode is not None:
        cache_dir.mkdir()
        cache_dir.chmod(existing_mode)

    previous_umask = os.umask(0o077)
    try:
        _get_reference(cache_dir, _reference)
    finally:
        os.umask(previous_umask)

    directory_mode = stat.S_IMODE(cache_dir.stat().st_mode)
    assert directory_mode & 0o055 == 0o055
    if existing_mode is not None:
        assert directory_mode & existing_mode == existing_mode
    cache_path = next(cache_dir.glob("vlm-reference-*.pt"))
    assert stat.S_IMODE(cache_path.stat().st_mode) == 0o644


def test_shared_cache_directory_does_not_require_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "references"
    cache_dir.mkdir()
    cache_dir.chmod(0o775)
    original_chmod = Path.chmod

    def chmod(path, mode, **kwargs):
        if path == cache_dir:
            raise PermissionError("shared cache directory is owned by another user")
        return original_chmod(path, mode, **kwargs)

    monkeypatch.setattr(Path, "chmod", chmod)
    _, status = _get_reference(cache_dir, _reference)

    assert status == "saved"
    assert len(list(cache_dir.glob("vlm-reference-*.pt"))) == 1


def test_cache_directory_failure_returns_computed_reference(tmp_path: Path) -> None:
    cache_dir = tmp_path / "references"
    cache_dir.write_text("not a directory")
    expected = _reference()

    with pytest.warns(UserWarning, match="Could not write VLM reference cache entry"):
        reference, status = _get_reference(cache_dir, lambda: expected)

    assert reference is expected
    assert status == "write_failed"
    assert cache_dir.read_text() == "not a directory"


def test_cache_publish_failure_returns_computed_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_publish(source, destination):
        raise PermissionError("cache is read-only")

    monkeypatch.setattr("tests._vision_helpers.os.replace", fail_publish)
    expected = _reference()

    with pytest.warns(UserWarning, match="Could not write VLM reference cache entry"):
        reference, status = _get_reference(tmp_path, lambda: expected)

    assert reference is expected
    assert status == "write_failed"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("changed_batch", "max_new_tokens"),
    [
        ({"input_ids": torch.tensor([[1, 3]])}, 4),
        ({"input_ids": torch.tensor([[1, 2]])}, 5),
    ],
)
def test_input_or_generation_change_invalidates_cache(
    tmp_path: Path,
    changed_batch: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> None:
    calls = 0

    def compute() -> VLMReference:
        nonlocal calls
        calls += 1
        return _reference(float(calls))

    _get_reference(tmp_path, compute)
    changed, cache_status = _get_reference(
        tmp_path,
        compute,
        batch=changed_batch,
        max_new_tokens=max_new_tokens,
    )

    assert cache_status == "saved"
    assert calls == 2
    assert changed.text == "caption 2.0"
    assert len(list(tmp_path.glob("vlm-reference-*.pt"))) == 2


def test_corrupt_cache_entry_is_recomputed(tmp_path: Path) -> None:
    calls = 0

    def compute() -> VLMReference:
        nonlocal calls
        calls += 1
        return _reference(float(calls))

    _get_reference(tmp_path, compute)
    cache_path = next(tmp_path.glob("vlm-reference-*.pt"))
    cache_path.write_bytes(b"not a torch checkpoint")

    with pytest.warns(UserWarning, match="Ignoring invalid VLM reference cache"):
        recovered, cache_status = _get_reference(tmp_path, compute)

    assert cache_status == "saved"
    assert calls == 2
    assert recovered.text == "caption 2.0"


def test_unresolved_model_revision_disables_cache(tmp_path: Path) -> None:
    calls = 0

    def compute() -> VLMReference:
        nonlocal calls
        calls += 1
        return _reference()

    kwargs = {
        "cache_dir": tmp_path,
        "model_path": "local/model",
        "model_revision": None,
        "model_dtype": torch.float16,
        "trust_remote_code": False,
        "prompt": "describe",
        "batch": {"input_ids": torch.tensor([[1, 2]])},
        "max_new_tokens": 4,
        "num_compare_steps": 2,
        "compute": compute,
    }
    _, first_status = get_or_create_vlm_reference(**kwargs)
    _, second_status = get_or_create_vlm_reference(**kwargs)

    assert first_status == "disabled"
    assert second_status == "disabled"
    assert calls == 2
    assert not list(tmp_path.iterdir())


def test_stock_reference_uses_one_generation_for_logits_and_caption() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                sequences=torch.tensor([[7, 8, 10, 11, 12, 13]]),
                logits=tuple(
                    torch.tensor([[float(step), float(step + 1)]]) for step in range(4)
                ),
            )

    class FakeTokenizer:
        def decode(self, tokens, **kwargs) -> str:
            assert kwargs == {"skip_special_tokens": True}
            return "decoded " + " ".join(str(int(token)) for token in tokens)

    model = FakeModel()
    processor = SimpleNamespace(tokenizer=FakeTokenizer())
    batch = {"input_ids": torch.tensor([[7, 8]])}

    reference = stock_vlm_reference(
        model=model,
        processor=processor,
        batch=batch,
        max_new_tokens=4,
        num_compare_steps=2,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["max_new_tokens"] == 4
    assert model.calls[0]["output_logits"] is True
    assert model.calls[0]["return_dict_in_generate"] is True
    assert reference.token_ids == [10, 11]
    assert reference.text == "decoded 10 11 12 13"
    assert torch.equal(reference.logits[1], torch.tensor([1.0, 2.0]))
