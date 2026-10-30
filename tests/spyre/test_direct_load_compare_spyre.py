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

"""Direct-to-device loading must match the CPU-staged path exactly.

``direct_to_device=True`` changes *when* a weight is transferred, never what
lands on the device. This loads the same checkpoint both ways and asserts the
device tensors, their device layouts, and the greedy tokens all agree.

This file does not measure host RSS. ``ru_maxrss`` is a per-process high-water
mark that never falls, so the second load in a process can never read below
the first, and both loads have to happen here to be compared. Measure the
saving with one load per process instead.

Run on a Spyre pod::

    pytest -s -vvv tests/spyre/test_direct_load_compare_spyre.py
    pytest -s -vvv tests/spyre/test_direct_load_compare_spyre.py -k granite

Only adapters that set ``SPYRE_SUPPORTS_DIRECT_LOAD`` are exercised; the rest
are skipped rather than failed, since the flag tracks whether that adapter's
``prepare_for_spyre`` has been verified against on-device weights.
"""

import gc

import pytest
import torch

from hf_adapters.auto_spyre_model import AutoSpyreModelForCausalLM
from hf_adapters.spyre_direct_load import supports_direct_load
from tests.conftest import (
    encode_generation_inputs,
    resolve_adapter_module_for_test,
)
from tests.model_registry import CAUSAL_PATHS

pytestmark = pytest.mark.model_harness("causal")

_PROMPTS = ["The capital of France is"]
_NUM_NEW_TOKENS = 8


def _load(model_path, *, direct, trust_remote_code):
    return AutoSpyreModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        direct_to_device=direct,
    )


def _device_state(model):
    """Snapshot every parameter as ``(host values, device layout)``.

    Reading back through ``.cpu()`` compares what the device actually holds,
    which is what a layout bug would change.
    """
    from torch_spyre._C import get_spyre_tensor_layout

    snapshot = {}
    for name, param in model.named_parameters():
        assert (
            param.device.type == "spyre"
        ), f"{name} is on {param.device}, not Spyre, after loading"
        layout = get_spyre_tensor_layout(param)
        snapshot[name] = (param.detach().cpu(), tuple(layout.device_size))
    return snapshot


def _greedy_tokens(model, model_path, trust_remote_code):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    encoded = encode_generation_inputs(tokenizer, _PROMPTS)
    with torch.no_grad():
        return model.generate(
            **encoded,
            max_new_tokens=_NUM_NEW_TOKENS,
            do_sample=False,
        )


def _opted_in(model_path, trust_remote_code):
    adapter = resolve_adapter_module_for_test(
        model_path, trust_remote_code=trust_remote_code
    )
    return supports_direct_load(adapter)


@pytest.mark.parametrize("model_path", CAUSAL_PATHS)
def test_direct_load_matches_cpu_staged_load(
    model_path: str, trust_remote_code: bool | None
) -> None:
    if not _opted_in(model_path, trust_remote_code):
        pytest.skip(f"{model_path}: adapter has not set SPYRE_SUPPORTS_DIRECT_LOAD")

    staged = _load(model_path, direct=False, trust_remote_code=trust_remote_code)
    staged_state = _device_state(staged)
    staged_tokens = _greedy_tokens(staged, model_path, trust_remote_code)
    del staged
    gc.collect()

    direct = _load(model_path, direct=True, trust_remote_code=trust_remote_code)
    direct_state = _device_state(direct)
    direct_tokens = _greedy_tokens(direct, model_path, trust_remote_code)

    assert (
        direct_state.keys() == staged_state.keys()
    ), "direct loading produced a different set of parameters"
    for name in staged_state:
        staged_values, staged_layout = staged_state[name]
        direct_values, direct_layout = direct_state[name]
        assert (
            direct_layout == staged_layout
        ), f"{name}: device layout {direct_layout} != staged {staged_layout}"
        assert torch.equal(
            direct_values, staged_values
        ), f"{name}: device contents differ between load paths"

    assert torch.equal(
        direct_tokens, staged_tokens
    ), f"greedy tokens differ: direct={direct_tokens} staged={staged_tokens}"

    print(f"\n{model_path}: {len(staged_state)} parameters match, tokens match")
