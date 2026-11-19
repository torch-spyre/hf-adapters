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

"""Measure the host memory one model load uses, in a process of its own.

Peak memory only ever rises within a process, so two loads in one process
cannot be compared: the second can never read below the first. Each load here
runs in a fresh child (``python -m tests.spyre._load_memory_helpers --child``)
and the parent samples the child's ``/proc/<pid>/status`` while it runs.
Sampling from outside also keeps a long native copy that holds the GIL from
hiding a peak, which an in-process sampling thread could miss.

The number that matters is anonymous memory (``RssAnon``): private pages such
as dtype-converted weight copies, which the kernel cannot drop. File-backed
pages (``RssFile``) of the memory-mapped checkpoint are reclaimable and are
reported, not judged. The child records ``RssAnon`` after importing torch,
torch-spyre and Transformers but before loading, so ``load_anon`` is the
memory the load itself added.

Linux only: everything here reads ``/proc``.

Used by ``tests/spyre/test_direct_load_memory_spyre.py`` and
``scripts/measure_direct_load_memory.py``. Only the child imports torch, so the
sampling code is importable (and tested) without it.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GIB = 1024**3

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATUS_FIELDS = ("RssAnon", "RssFile", "RssShmem")

# ``--keep-read-ahead`` points the env switch ``spyre_direct_placement`` sets
# at a name Transformers never reads, so the load runs with read-ahead on: the
# behaviour before that switch existed. A diagnostic, not a supported mode.
_UNUSED_READ_AHEAD_ENV = "HF_ADAPTERS_MEASURE_UNUSED_READ_AHEAD_SWITCH"


def parse_proc_status(text: str) -> dict[str, int]:
    """Return the resident-memory fields of a ``/proc/<pid>/status`` dump in bytes.

    A process that has exited but not been reaped reports no ``Rss*`` lines,
    so an empty result means there is nothing to sample.
    """
    values = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        if not sep or key not in _STATUS_FIELDS:
            continue
        amount, _, unit = rest.strip().partition(" ")
        if unit != "kB":
            raise ValueError(f"unexpected unit in /proc status line: {line!r}")
        values[key] = int(amount) * 1024
    return values


def read_status(pid: int | str) -> dict[str, int]:
    """Resident-memory fields of process ``pid`` (or ``"self"``), or ``{}``."""
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return {}
    return parse_proc_status(text)


@dataclasses.dataclass
class PeakTracker:
    """Per-field maxima over a series of ``/proc`` samples, in bytes.

    ``total`` is the largest anon + file + shmem seen in a *single* sample, so
    it is a real moment's footprint rather than a sum of separate peaks.
    """

    anon: int = 0
    file: int = 0
    total: int = 0
    samples: int = 0

    def add(self, fields: dict[str, int]) -> None:
        if not fields:
            return
        anon = fields.get("RssAnon", 0)
        file = fields.get("RssFile", 0)
        shmem = fields.get("RssShmem", 0)
        self.anon = max(self.anon, anon)
        self.file = max(self.file, file)
        self.total = max(self.total, anon + file + shmem)
        self.samples += 1


def sample_process(
    cmd: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    poll_interval: float = 0.02,
    timeout: float = 3600.0,
) -> tuple[int, PeakTracker]:
    """Run ``cmd``, sampling its memory until it exits; return (exit code, peaks).

    The child's output goes to ``log_path`` rather than a pipe, so a chatty
    child cannot block on a full pipe while the parent is busy sampling.
    """
    peaks = PeakTracker()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        deadline = time.monotonic() + timeout
        # Until it is reaped the child's pid cannot be reused, so every read
        # below is of this child even after it has exited.
        while proc.poll() is None:
            peaks.add(read_status(proc.pid))
            if time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                raise TimeoutError(f"{cmd[:4]}... still running after {timeout}s")
            time.sleep(poll_interval)
    return proc.returncode, peaks


@dataclasses.dataclass
class LoadMemory:
    """Host memory for one ``AutoSpyreModelForCausalLM.from_pretrained`` load."""

    model_path: str
    path: str  # "staged" or "direct"
    dtype: str
    read_ahead_kept: bool
    load_seconds: float
    baseline_anon: int  # after imports, before the load
    peak_anon: int
    peak_file: int
    peak_total: int
    max_rss: int  # the child's own ru_maxrss, as a cross-check on peak_total
    param_bytes: int
    params: int
    params_off_spyre: int
    samples: int

    @property
    def load_anon(self) -> int:
        """Anonymous memory the load added on top of the imported baseline."""
        return self.peak_anon - self.baseline_anon


def checkpoint_dtype_name(model_path: str, trust_remote_code: bool | None) -> str:
    """The dtype the checkpoint declares (``float16``, ``bfloat16``, ...)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=bool(trust_remote_code)
    )
    text_config = getattr(config, "text_config", None)
    dtype = getattr(text_config, "dtype", None) or getattr(config, "dtype", None)
    return str(dtype).removeprefix("torch.") if dtype is not None else "unknown"


def cast_dtype_name(checkpoint_dtype: str) -> str | None:
    """The other 16-bit dtype, i.e. a load that converts every weight."""
    return {"bfloat16": "float16", "float16": "bfloat16"}.get(checkpoint_dtype)


def measure_load(
    model_path: str,
    *,
    direct: bool,
    dtype: str,
    trust_remote_code: bool | None = None,
    keep_read_ahead: bool = False,
    poll_interval: float = 0.02,
    timeout: float = 3600.0,
) -> LoadMemory:
    """Load ``model_path`` once in a fresh process and report its host memory."""
    if not Path("/proc/self/status").exists():
        raise RuntimeError("measuring load memory needs Linux /proc")
    if keep_read_ahead and not direct:
        raise ValueError("keep_read_ahead only changes the direct path")

    cmd = [
        sys.executable,
        "-m",
        "tests.spyre._load_memory_helpers",
        "--child",
        "--model-path",
        model_path,
        "--dtype",
        dtype,
    ]
    if direct:
        cmd.append("--direct")
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if keep_read_ahead:
        cmd.append("--keep-read-ahead")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        log_path = Path(tmp) / "child.log"
        cmd += ["--result", str(result_path)]
        returncode, peaks = sample_process(
            cmd,
            log_path=log_path,
            cwd=_REPO_ROOT,
            env=env,
            poll_interval=poll_interval,
            timeout=timeout,
        )
        if returncode != 0:
            tail = log_path.read_text(errors="replace")[-4000:]
            raise RuntimeError(
                f"{'direct' if direct else 'staged'} load of {model_path} "
                f"exited with {returncode}:\n{tail}"
            )
        child = json.loads(result_path.read_text())

    return LoadMemory(
        model_path=model_path,
        path="direct" if direct else "staged",
        dtype=dtype,
        read_ahead_kept=keep_read_ahead,
        peak_anon=peaks.anon,
        peak_file=peaks.file,
        peak_total=peaks.total,
        samples=peaks.samples,
        **child,
    )


_TABLE_COLUMNS = (
    ("model", 38),
    ("dtype", 9),
    ("path", 15),
    ("load s", 7),
    ("base anon", 10),
    ("peak anon", 10),
    ("load anon", 10),
    ("peak file", 10),
    ("peak total", 11),
    ("off spyre", 9),
)


def format_table(rows: list[LoadMemory]) -> str:
    """Render measurements as a fixed-width table, memory in GiB."""

    def gib(value: int) -> str:
        return f"{value / GIB:.2f}"

    lines = [" ".join(name.rjust(width) for name, width in _TABLE_COLUMNS)]
    for row in rows:
        path = row.path + (" +read-ahead" if row.read_ahead_kept else "")
        cells = (
            row.model_path[-38:],
            row.dtype,
            path,
            f"{row.load_seconds:.1f}",
            gib(row.baseline_anon),
            gib(row.peak_anon),
            gib(row.load_anon),
            gib(row.peak_file),
            gib(row.peak_total),
            str(row.params_off_spyre),
        )
        lines.append(
            " ".join(
                cell.rjust(width)
                for cell, (_, width) in zip(cells, _TABLE_COLUMNS, strict=True)
            )
        )
    return "\n".join(lines)


def _child(args: argparse.Namespace) -> None:
    """Load once and write what only the loading process can see as JSON."""
    import resource

    import torch
    import torch_spyre  # noqa: F401  (part of the baseline, like the others)
    import transformers  # noqa: F401

    import hf_adapters.spyre_direct_load as spyre_direct_load
    from hf_adapters.auto_spyre_model import AutoSpyreModelForCausalLM

    if args.keep_read_ahead:
        spyre_direct_load._ASYNC_LOAD_ENV = _UNUSED_READ_AHEAD_ENV

    baseline_anon = read_status("self")["RssAnon"]
    start = time.perf_counter()
    model = AutoSpyreModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=getattr(torch, args.dtype),
        trust_remote_code=True if args.trust_remote_code else None,
        direct_to_device=args.direct,
    )
    load_seconds = time.perf_counter() - start

    params = list(model.parameters())
    result = {
        "load_seconds": load_seconds,
        "baseline_anon": baseline_anon,
        "max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "param_bytes": sum(p.numel() * p.element_size() for p in params),
        "params": len(params),
        "params_off_spyre": sum(p.device.type != "spyre" for p in params),
    }
    Path(args.result).write_text(json.dumps(result))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dtype", required=True, choices=("float16", "bfloat16"))
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-read-ahead", action="store_true")
    parser.add_argument("--result", required=True)
    _child(parser.parse_args())


if __name__ == "__main__":
    _main()
