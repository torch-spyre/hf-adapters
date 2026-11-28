"""``TorchOpCollector.write_yaml`` annotates the bfloat16 ``rtol`` with a comment.

PyYAML cannot emit comments, so the writer post-processes the dumped text. These
tests check the comment lands directly above the bfloat16 ``rtol`` (and nowhere
else) in both the old (``USE_OLDFORMAT``) and new layouts, and that the output
still parses to the same data. Runs on CPU:

    pytest --noconftest tests/test_torchop_yaml_bf16_comment.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_TORCHOP_YAML = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "model_ops"
    / "utils"
    / "torchop_yaml.py"
)

_COMMENT = [
    "# rtol 0.008 (> 2^-7 = 0.0078125, the max relative size of one bf16 step):",
    "# device results, computed via DLFloat16, can differ from the CPU reference",
    "# by one bf16 step.",
]

_CASE = {"name": "torch.add", "inputs": [{"shape": [2, 3], "dtype": "torch.float16"}]}


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_torchop_yaml_bf16", _TORCHOP_YAML)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(mod, monkeypatch, tmp_path, old_format, supress_spyre=True):
    monkeypatch.setattr(mod, "USE_OLDFORMAT", old_format)
    monkeypatch.setattr(mod.TorchOpCollector, "test_cases_yaml", [dict(_CASE)])
    monkeypatch.setattr(mod.TorchOpCollector, "test_cases_norm_yaml", [dict(_CASE)])
    mod.TorchOpCollector().write_yaml(
        "m", output_dir=str(tmp_path), supress_spyre=supress_spyre
    )


def _comment_block(lines, indent):
    pad = " " * indent
    return [pad + c for c in _COMMENT]


def _assert_comment_above_bf16_rtol(text, rtol_indent):
    lines = text.splitlines()
    rtol_line = " " * rtol_indent + "rtol: 0.008"
    idx = lines.index(rtol_line)
    assert lines[idx - 3 : idx] == _comment_block(lines, rtol_indent)
    # Comment appears exactly once.
    assert sum(1 for ln in lines if ln.strip() == _COMMENT[0]) == 1


def test_old_format(mod, monkeypatch, tmp_path):
    _write(mod, monkeypatch, tmp_path, old_format=True)
    text = (tmp_path / "m.yaml").read_text()
    expected = "\n".join(
        [
            "dtypes:",
            "  - name: bfloat16",
            "    precision:",
            "      atol: 0.005",
            *("      " + c for c in _COMMENT),
            "      rtol: 0.008",
        ]
    )
    assert expected in text
    _assert_comment_above_bf16_rtol(text, rtol_indent=6)
    assert yaml.safe_load(text)["dtypes"] == [
        {"name": "bfloat16", "precision": {"atol": 0.005, "rtol": 0.008}}
    ]


def test_new_format(mod, monkeypatch, tmp_path):
    _write(mod, monkeypatch, tmp_path, old_format=False)
    text = (tmp_path / "m.yaml").read_text()
    expected = "\n".join(
        [
            "      - name: bfloat16",
            "        precision:",
            "          atol: 0.005",
            *("          " + c for c in _COMMENT),
            "          rtol: 0.008",
        ]
    )
    assert expected in text
    assert "  global:\n    supported_dtypes:\n" in text
    _assert_comment_above_bf16_rtol(text, rtol_indent=10)
    dtypes = yaml.safe_load(text)["test_suite_config"]["global"]["supported_dtypes"]
    bf16 = [d for d in dtypes if d["name"] == "bfloat16"]
    assert bf16 == [{"name": "bfloat16", "precision": {"atol": 0.005, "rtol": 0.008}}]
    # Other dtypes are untouched and keep their own precision.
    assert all(
        d["precision"] == {"atol": 0.005, "rtol": 0.005}
        for d in dtypes
        if d["name"] != "bfloat16"
    )


@pytest.mark.parametrize("old_format", [True, False])
def test_spyre_yaml_also_commented(mod, monkeypatch, tmp_path, old_format):
    _write(mod, monkeypatch, tmp_path, old_format=old_format, supress_spyre=False)
    text = (tmp_path / "m_spyre.yaml").read_text()
    _assert_comment_above_bf16_rtol(text, rtol_indent=6 if old_format else 10)


def test_insert_skips_non_bf16_entries(mod):
    text = (
        "dtypes:\n"
        "  - name: float16\n"
        "    precision:\n"
        "      rtol: 0.005\n"
        "  - name: bfloat16\n"
        "    precision:\n"
        "      atol: 0.005\n"
        "      rtol: 0.008\n"
        "  - name: float32\n"
        "    precision:\n"
        "      rtol: 0.005\n"
    )
    out = mod._insert_bf16_rtol_comment(text)
    assert out.count(_COMMENT[0]) == 1
    _assert_comment_above_bf16_rtol(out, rtol_indent=6)
    assert yaml.safe_load(out) == yaml.safe_load(text)


def test_insert_no_rtol_in_bf16_item(mod):
    # A bf16 item without rtol must not leak the comment into the next item.
    text = (
        "dtypes:\n"
        "  - name: bfloat16\n"
        "    precision:\n"
        "      atol: 0.005\n"
        "  - name: float32\n"
        "    precision:\n"
        "      rtol: 0.005\n"
    )
    assert mod._insert_bf16_rtol_comment(text) == text
