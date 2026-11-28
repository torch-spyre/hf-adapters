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

"""The load-memory measurement itself, without a device.

``tests/spyre/test_direct_load_memory_spyre.py`` judges direct-to-device
loading by these numbers, so the sampling has to be right independently: a
sampler that read the wrong field, or missed a peak, would let that test pass
or fail for the wrong reason.
"""

import sys
import textwrap
from pathlib import Path

import pytest

from tests.spyre._load_memory_helpers import (
    GIB,
    LoadMemory,
    PeakTracker,
    cast_dtype_name,
    format_table,
    parse_proc_status,
    sample_process,
)

_STATUS = textwrap.dedent("""\
    Name:\tpython
    VmPeak:\t 9000000 kB
    VmHWM:\t 3000000 kB
    VmRSS:\t 2500000 kB
    RssAnon:\t 2000000 kB
    RssFile:\t  400000 kB
    RssShmem:\t  100000 kB
    Threads:\t8
    """)


def test_parse_proc_status_reads_resident_fields_in_bytes():
    assert parse_proc_status(_STATUS) == {
        "RssAnon": 2_000_000 * 1024,
        "RssFile": 400_000 * 1024,
        "RssShmem": 100_000 * 1024,
    }


def test_parse_proc_status_of_an_exited_process_is_empty():
    # A zombie's status has no Rss* lines; it must not read as zero memory.
    assert parse_proc_status("Name:\tpython\nState:\tZ (zombie)\n") == {}


def test_parse_proc_status_rejects_an_unknown_unit():
    with pytest.raises(ValueError, match="unexpected unit"):
        parse_proc_status("RssAnon:\t 12 MB\n")


def test_peak_tracker_keeps_each_maximum_and_the_largest_single_total():
    peaks = PeakTracker()
    peaks.add({"RssAnon": 10, "RssFile": 1, "RssShmem": 0})
    peaks.add({"RssAnon": 2, "RssFile": 7, "RssShmem": 0})
    peaks.add({})  # an exited child: ignored, not counted as a sample

    assert (peaks.anon, peaks.file, peaks.samples) == (10, 7, 2)
    # 10 + 7 never existed at the same moment.
    assert peaks.total == 11


def test_cast_dtype_pairs_the_two_16_bit_dtypes():
    assert cast_dtype_name("bfloat16") == "float16"
    assert cast_dtype_name("float16") == "bfloat16"
    assert cast_dtype_name("float32") is None


def test_load_anon_subtracts_the_imported_baseline():
    row = LoadMemory(
        model_path="org/model",
        path="direct",
        dtype="float16",
        read_ahead_kept=False,
        load_seconds=1.0,
        baseline_anon=1 * GIB,
        peak_anon=3 * GIB,
        peak_file=0,
        peak_total=3 * GIB,
        max_rss=3 * GIB,
        param_bytes=4 * GIB,
        params=2,
        params_off_spyre=0,
        samples=5,
    )
    assert row.load_anon == 2 * GIB
    assert "2.00" in format_table([row]).splitlines()[1]


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs Linux /proc")
def test_sampling_catches_a_short_lived_allocation(tmp_path):
    """A child that holds 256 MiB for half a second must be seen holding it.

    ``b"x" * n`` writes every byte, so the pages are resident; a zero-filled
    ``bytearray`` could stay unbacked and read as no memory at all.
    """
    code = "import time; b = b'x' * (256 * 2**20); time.sleep(0.5)"
    returncode, peaks = sample_process(
        [sys.executable, "-c", code], log_path=tmp_path / "child.log"
    )

    assert returncode == 0
    assert peaks.samples > 0
    assert peaks.anon >= 256 * 2**20
