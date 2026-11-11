# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real Transformers checkpoint loader across Gloo processes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is required for CPU TP")
def test_distributed_checkpoint_loading(tmp_path: Path, world_size: int) -> None:
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(root), env.get("PYTHONPATH", "")))
    env["HF_HUB_OFFLINE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world_size}",
            str(Path(__file__).with_name("_tensor_parallel_worker.py")),
            str(tmp_path),
        ],
        env=env,
        cwd=root,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(world_size):
        assert (
            f"TP{world_size} rank {rank}: all loading, placement and numerical checks passed"
            in result.stdout
        )
