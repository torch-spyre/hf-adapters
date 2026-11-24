"""Exercise the build action's environment selection without Spyre hardware or torch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_ACTION = (
    Path(__file__).resolve().parents[1]
    / ".github/actions/build-hf-adapters/action.yaml"
)


# Stand-in for the parts of torch the action's backend probe touches.
_FAKE_TORCH: str = """\
import types

_device_types: set[str] = {"cpu"}


def device(name: str) -> str:
    if name not in _device_types:
        raise RuntimeError(
            "Expected one of cpu, cuda, ... device type at start of device string: "
            + name
        )
    return name


def _rename_privateuse1_backend(name: str) -> None:
    _device_types.add(name)


utils = types.SimpleNamespace(rename_privateuse1_backend=_rename_privateuse1_backend)
"""


def _build_script() -> str:
    action: dict[str, Any] = yaml.safe_load(_ACTION.read_text())
    step: dict[str, Any] = next(
        step for step in action["runs"]["steps"] if step["name"] == "Build hf-adapters"
    )
    script: str = step["run"]
    # Skip runner shell initialization and diagnostics, which require the image.
    return script[script.index("# Build group flags") :]


def _run_build(
    tmp_path: Path,
    *,
    matching_pin: bool = True,
    pinned: bool = True,
    prebaked: bool = False,
    cached: bool = False,
    broken_install: bool = False,
    custom_project_env: bool = False,
) -> subprocess.CompletedProcess[str]:
    image_env = tmp_path / "image-env"
    project_env = tmp_path / ("custom-env" if custom_project_env else ".venv")
    action_path = tmp_path / "action"
    action_path.mkdir()
    # Each simulated environment has a stand-in torch that, like PyTorch,
    # rejects "spyre" until a backend registers it, so the test runs without
    # torch. The stub torch_spyre that registers it on import is importable only
    # once the environment's backend marker is installed.
    for environment in (image_env, project_env):
        (environment / "bin").mkdir(parents=True)
        (environment / "lib").mkdir()
        (environment / "lib/torch.py").write_text(_FAKE_TORCH)
        (environment / "site-packages").mkdir()
        (environment / "site-packages/torch_spyre.py").write_text(
            'import torch\ntorch.utils.rename_privateuse1_backend("spyre")\n'
        )
        python = environment / "bin/python"
        python.write_text(
            f"#!{sys.executable}\n"
            "import sys\n"
            "from pathlib import Path\n"
            "here = Path(__file__).resolve().parent\n"
            'sys.path.insert(0, str(here.parent / "lib"))\n'
            'if (here / "torch-spyre-installed").exists():\n'
            '    sys.path.insert(0, str(here.parent / "site-packages"))\n'
            "exec(sys.argv[2])\n"
        )
        python.chmod(0o755)
    (image_env / "bin/torch-spyre-installed").touch()
    if cached:
        (action_path / "torch_spyre_wheel_cache.sh").write_text(
            'uv sync --frozen "$@" --no-install-package torch-spyre\n'
            'touch "$PROJECT_ENV/bin/torch-spyre-installed"\n'
            'echo hit > "$TS_WHEEL_CACHE_STATUS_FILE"\n'
        )
    env: dict[str, str] = {
        **os.environ,
        "VIRTUAL_ENV": str(image_env),
        "PROJECT_ENV": str(project_env),
        "CLONED_HF_ADAPTERS_DIR": str(tmp_path),
        "GITHUB_ACTION_PATH": str(action_path),
        "TORCH_SPYRE_SHA": "pin" if pinned else "",
        "BAKED_SHA_FOR_TEST": "pin" if matching_pin else "old",
        "PREBAKED_IMAGE": str(prebaked).lower(),
        "UV_GROUPS": "dev spyre test",
        "BROKEN_INSTALL": str(broken_install).lower(),
        "COMMAND_LOG": str(tmp_path / "commands.log"),
        "UV_NO_SYNC": "1",
    }
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    if custom_project_env:
        env["UV_PROJECT_ENVIRONMENT"] = str(project_env)
    fake_commands = r"""
cd() {
  if [[ "$1" == /home/senuser/torch-spyre ]]; then
    builtin cd "$CLONED_HF_ADAPTERS_DIR"
  else
    builtin cd "$@"
  fi
}
git() {
  echo "git $*" >> "$COMMAND_LOG"
  if [[ "$1 $2" == 'rev-parse HEAD' ]]; then echo "$BAKED_SHA_FOR_TEST"; fi
}
uv() {
  echo "uv $*" >> "$COMMAND_LOG"
  if [[ "$1" == sync && "$BROKEN_INSTALL" == false ]]; then
    if [[ " $* " == *' --active '* ]]; then
      touch "$VIRTUAL_ENV/bin/torch-spyre-installed"
    elif [[ " $* " != *' --no-install-package torch-spyre '* ]]; then
      touch "$PROJECT_ENV/bin/torch-spyre-installed"
    fi
  fi
}
export -f uv
"""
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", fake_commands + _build_script()],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("custom_project_env", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_matching_pin_installs_backend_in_project_environment(
    tmp_path: Path, cached: bool, custom_project_env: bool
) -> None:
    result = _run_build(tmp_path, cached=cached, custom_project_env=custom_project_env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Spyre backend registered: spyre" in result.stdout
    commands = (tmp_path / "commands.log").read_text()
    assert "git fetch" not in commands
    project_env = tmp_path / ("custom-env" if custom_project_env else ".venv")
    assert (project_env / "bin/torch-spyre-installed").exists()
    expected_python = (
        str(project_env / "bin/python") if custom_project_env else ".venv/bin/python"
    )
    assert f"uv pip show --python {expected_python} torch torch-spyre" in commands


def test_nonmatching_pin_fetches_and_installs_backend(tmp_path: Path) -> None:
    result = _run_build(tmp_path, matching_pin=False)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = (tmp_path / "commands.log").read_text()
    assert "git fetch --tags --force origin pin" in commands
    assert "git checkout --force pin" in commands
    assert "Spyre backend registered: spyre" in result.stdout


def test_unpinned_mode_installs_and_checks_project_backend(tmp_path: Path) -> None:
    result = _run_build(tmp_path, pinned=False)
    assert result.returncode == 0, result.stdout + result.stderr
    commands = (tmp_path / "commands.log").read_text()
    assert "uv lock --upgrade-package torch-spyre" in commands
    assert "Spyre backend registered: spyre" in result.stdout


def test_prebaked_mode_preserves_and_checks_image_environment(tmp_path: Path) -> None:
    result = _run_build(tmp_path, prebaked=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / ".venv/bin/torch-spyre-installed").exists()
    assert "Spyre backend registered: spyre" in result.stdout
    commands = (tmp_path / "commands.log").read_text()
    assert "uv add" not in commands
    assert "uv sync --active" in commands
    assert (
        f"uv pip show --python {tmp_path}/image-env/bin/python torch torch-spyre"
        in commands
    )


def test_missing_project_backend_fails_setup(tmp_path: Path) -> None:
    result = _run_build(tmp_path, broken_install=True)
    assert result.returncode != 0
    assert "No module named 'torch_spyre'" in result.stderr
    assert "uv pip freeze" not in (tmp_path / "commands.log").read_text()
