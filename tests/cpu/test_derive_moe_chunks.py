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

"""Chunk-count selection for the gathered MoE decode form."""

import pytest

from hf_adapters.hf_common import derive_moe_chunks


def test_gemma4_dimensions_select_four_chunks_on_32_cores():
    # Gemma 4 26B-A4B: 8 selected experts x 4 hidden chunks fills 32 cores.
    assert derive_moe_chunks(2816, 8, 64, 32) == 4


def test_non_power_of_two_chunk_counts_are_allowed():
    # Five hidden sticks can use five equal chunks.
    assert derive_moe_chunks(320, 4, 64, 32) == 5


def test_fewer_cores_or_larger_top_k_need_fewer_chunks():
    assert derive_moe_chunks(2816, 8, 64, 16) == 2
    assert derive_moe_chunks(2816, 16, 64, 32) == 2


def test_override_must_split_hidden_into_whole_sticks():
    assert derive_moe_chunks(2816, 8, 64, 32, override=2) == 2
    with pytest.raises(
        ValueError, match="SPYRE_MOE_CHUNKS=3 requires hidden size 2816"
    ):
        derive_moe_chunks(2816, 8, 64, 32, override=3)
    with pytest.raises(ValueError, match="SPYRE_MOE_CHUNKS must be positive"):
        derive_moe_chunks(2816, 8, 64, 32, override=0)


def test_hidden_size_must_be_stick_aligned():
    with pytest.raises(ValueError, match="must be a multiple of"):
        derive_moe_chunks(2815, 8, 64, 32)


def test_env_override_is_read(monkeypatch):
    monkeypatch.setenv("SPYRE_MOE_CHUNKS", "2")
    assert derive_moe_chunks(2816, 8, 64, 32) == 2
