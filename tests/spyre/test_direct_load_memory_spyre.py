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

"""Direct-to-device loading must not keep converted weights on the host.

``tests/spyre/test_direct_load_compare_spyre.py`` checks that both load paths
put the same tensors on the device. This checks what direct loading is for:
host memory. Each load runs in a process of its own (see
``_load_memory_helpers``), and the judged number is the *anonymous* memory the
load added, which is what decides whether a load runs out of memory.

Two cases, each one staged load and one direct load:

* A **dtype-converting** load (the other 16-bit dtype from the checkpoint's).
  The staged path holds a converted copy of every weight until
  ``model.to``; the direct path transfers each weight as it is converted.
  Measured as fp16 from the bf16 checkpoints, with the upload threads: on
  Granite 3.3 8B the direct path added about 14% of what the staged path
  did, and about 93% with Transformers' read-ahead left on. Granite 3.3 2B
  keeps its attention projections on the host as well
  (``spyre_direct_stays_on_host``) and added about 38%. A direct load may
  add ``_MAX_DIRECT_SHARE`` of what the staged path added, plus
  ``max_in_flight_bytes``, the most the upload threads may hold on the host
  waiting for the card. Both models pass that with at least 1 GiB to spare,
  and the 8B read-ahead case fails it.
* A load in the checkpoint's **own** dtype. Transformers then hands out views
  into the memory-mapped file, the staged path is already cheap, and direct
  loading must not make it worse.

Run on a Spyre pod (each case loads the model twice, so allow a few minutes)::

    pytest -s -v tests/spyre/test_direct_load_memory_spyre.py \\
        --model-path ibm-granite/granite-3.3-8b-instruct

For the full table, including the read-ahead diagnostic, use
``scripts/measure_direct_load_memory.py``.
"""

import sys

import pytest

from hf_adapters.spyre_direct_load import (
    _DEFAULT_MAX_IN_FLIGHT_BYTES,
    supports_direct_load,
)
from tests.conftest import resolve_adapter_module_for_test
from tests.model_registry import CAUSAL_PATHS
from tests.spyre._load_memory_helpers import (
    GIB,
    cast_dtype_name,
    checkpoint_dtype_name,
    format_table,
    measure_load,
)

pytestmark = [
    pytest.mark.model_harness("causal"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="load memory is read from /proc/<pid>/status",
    ),
]

#: Most of the staged path's load memory a direct dtype-converting load may add.
_MAX_DIRECT_SHARE = 0.4
#: Least of the model's bytes the staged path must hold for a converting load.
#: Below this the staged path was not holding a converted copy, so the
#: comparison above would pass without measuring anything.
_MIN_STAGED_SHARE_OF_PARAMS = 0.5
#: Extra anonymous memory direct loading may add when no dtype conversion runs.
_MATCHING_DTYPE_SLACK = int(0.5 * GIB)


def _skip_unless_opted_in(model_path, trust_remote_code):
    adapter = resolve_adapter_module_for_test(
        model_path, trust_remote_code=trust_remote_code
    )
    if not supports_direct_load(adapter):
        pytest.skip(f"{model_path}: adapter has not set SPYRE_SUPPORTS_DIRECT_LOAD")


def _measure_both_paths(model_path, dtype, trust_remote_code):
    staged = measure_load(
        model_path, direct=False, dtype=dtype, trust_remote_code=trust_remote_code
    )
    direct = measure_load(
        model_path, direct=True, dtype=dtype, trust_remote_code=trust_remote_code
    )
    print("\n" + format_table([staged, direct]))
    for measured in (staged, direct):
        assert measured.params_off_spyre == 0, (
            f"{measured.path} load left {measured.params_off_spyre} of "
            f"{measured.params} parameters off Spyre"
        )
    return staged, direct


@pytest.mark.parametrize("model_path", CAUSAL_PATHS)
def test_direct_load_keeps_converted_weights_off_the_host(
    model_path: str, trust_remote_code: bool | None
) -> None:
    _skip_unless_opted_in(model_path, trust_remote_code)
    checkpoint_dtype = checkpoint_dtype_name(model_path, trust_remote_code)
    cast_dtype = cast_dtype_name(checkpoint_dtype)
    if cast_dtype is None:
        pytest.skip(
            f"{model_path}: no 16-bit cast from a {checkpoint_dtype} checkpoint"
        )

    staged, direct = _measure_both_paths(model_path, cast_dtype, trust_remote_code)

    assert staged.load_anon >= _MIN_STAGED_SHARE_OF_PARAMS * staged.param_bytes, (
        f"the staged {cast_dtype} load added only {staged.load_anon / GIB:.2f} GiB "
        f"for {staged.param_bytes / GIB:.2f} GiB of weights, so it did not hold a "
        "converted copy and this comparison cannot tell the paths apart"
    )
    allowed = _MAX_DIRECT_SHARE * staged.load_anon + _DEFAULT_MAX_IN_FLIGHT_BYTES
    assert direct.load_anon <= allowed, (
        f"direct {cast_dtype} load added {direct.load_anon / GIB:.2f} GiB anonymous, "
        f"more than {_MAX_DIRECT_SHARE:.0%} of the staged path's "
        f"{staged.load_anon / GIB:.2f} GiB plus "
        f"{_DEFAULT_MAX_IN_FLIGHT_BYTES / GIB:.2f} GiB for uploads in flight"
    )


@pytest.mark.parametrize("model_path", CAUSAL_PATHS)
def test_direct_load_adds_no_host_memory_in_the_checkpoint_dtype(
    model_path: str, trust_remote_code: bool | None
) -> None:
    _skip_unless_opted_in(model_path, trust_remote_code)
    checkpoint_dtype = checkpoint_dtype_name(model_path, trust_remote_code)
    if cast_dtype_name(checkpoint_dtype) is None:
        pytest.skip(f"{model_path}: a {checkpoint_dtype} checkpoint is cast on load")

    staged, direct = _measure_both_paths(
        model_path, checkpoint_dtype, trust_remote_code
    )

    assert direct.load_anon <= staged.load_anon + _MATCHING_DTYPE_SLACK, (
        f"direct {checkpoint_dtype} load added {direct.load_anon / GIB:.2f} GiB "
        f"anonymous against the staged path's {staged.load_anon / GIB:.2f} GiB"
    )
