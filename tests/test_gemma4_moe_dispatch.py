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

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from hf_adapters import auto_spyre_model as asm


def test_moe_config_routes_to_moe_adapter(monkeypatch):
    cfg = Gemma4TextConfig(
        enable_moe_block=True,
        num_experts=8,
        top_k_experts=4,
        moe_intermediate_size=8,
    )
    cfg.architectures = ["Gemma4ForConditionalGeneration"]
    monkeypatch.setattr(
        asm.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: cfg),
    )
    monkeypatch.setattr(asm, "assert_spyre_dimensions", lambda *a, **k: None)
    assert asm.resolve_adapter_module("dummy").__name__.endswith("hf_gemma4_moe")


def test_dense_config_still_routes_to_dense_adapter(monkeypatch):
    cfg = Gemma4TextConfig()
    cfg.architectures = ["Gemma4ForConditionalGeneration"]
    monkeypatch.setattr(
        asm.AutoConfig,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: cfg),
    )
    monkeypatch.setattr(asm, "assert_spyre_dimensions", lambda *a, **k: None)
    assert asm.resolve_adapter_module("dummy").__name__.endswith("hf_gemma4")
