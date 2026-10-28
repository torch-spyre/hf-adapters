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

"""The `capability.*` properties a model-module test declares."""

import xml.etree.ElementTree as ET

from tests.module_capability_plugin import (
    capability_properties,
    fallback_op,
    split_test_name,
    subject_for,
)

pytest_plugins = ["pytester"]


def test_split_test_name():
    assert split_test_name(
        "test_with_cpu_GraniteAttention_layer0_prefill_spyre_bfloat16"
    ) == (
        "test_with_cpu",
        "GraniteAttention_layer0_prefill",
        "torch.bfloat16",
    )
    assert split_test_name(
        "test_forward_Embedding_49159_4096_prefill_spyre_float16"
    ) == (
        "test_forward",
        "Embedding_49159_4096_prefill",
        "torch.float16",
    )
    assert split_test_name("test_something_else") is None


def test_subject_is_the_single_model_tag(tmp_path):
    one = tmp_path / "granite-3.3-8b-instruct_adapter.yaml"
    one.write_text(
        "tags:\n  - model__granite-3.3-8b-instruct\n  - model__granite-3.3-8b-instruct\n"
    )
    assert subject_for(str(one)) == "granite-3.3-8b-instruct"
    two = tmp_path / "mixed.yaml"
    two.write_text("tags: [model__a, model__b]\n")
    assert subject_for(str(two)) == "mixed"


def test_fallback_op():
    assert (
        fallback_op("aten.arange.default is falling back to cpu")
        == "aten.arange.default"
    )
    assert (
        fallback_op(
            "conversion from torch.float32 to torch.int64 is falling back to cpu"
        )
        == "conversion from torch.float32 to torch.int64"
    )


def test_properties(tmp_path):
    cfg = tmp_path / "gpt-oss-20b.yaml"
    cfg.write_text(
        "tags: [model__gpt-oss-20b]\n"
        "include:\n"
        "  - name: GptOssMLP_layer1_decode\n"
        "    module_path: transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP\n"
        "  - name: GptOssMLP_layer1_prefill\n"
        "    module_path: hf_adapters.hf_gpt_oss.GptOssMLP\n"
        "    apply_device_layout: true\n"
    )
    name = "test_eager_vs_compile_GptOssMLP_layer1_decode_spyre_bfloat16"
    props = dict(capability_properties(name, str(cfg), []))
    assert props == {
        "capability.test_type": "model_modules",
        "capability.subject": "gpt-oss-20b",
        "capability.name": "GptOssMLP_layer1_decode",
        "capability.sig.test": "test_eager_vs_compile",
        "capability.sig.dtype": "torch.bfloat16",
        "capability.sig.device_layout": "false",
        "capability.backend": "spyre",
        "capability.prop.module_path": "transformers.models.gpt_oss.modeling_gpt_oss.GptOssMLP",
    }
    adapter = dict(
        capability_properties(name.replace("decode", "prefill"), str(cfg), [])
    )
    assert adapter["capability.sig.device_layout"] == "true"
    assert adapter["capability.prop.module_path"] == "hf_adapters.hf_gpt_oss.GptOssMLP"
    fb = dict(capability_properties(name, str(cfg), ["aten.x", "aten.x", "aten.y"]))
    assert fb["capability.backend"] == "cpu"
    assert fb["capability.prop.fallback_ops"] == '["aten.x", "aten.y"]'
    assert capability_properties("test_other", str(cfg), []) == []
    assert capability_properties(name, "", []) == []


def test_junit_carries_properties(pytester, tmp_path, monkeypatch):
    cfg = tmp_path / "granite.yaml"
    cfg.write_text("tags: [model__granite]\n")
    monkeypatch.setenv("PYTORCH_TEST_CONFIG", str(cfg))
    pytester.makepyfile(test_modules_custom="""
        import unittest
        class TestModuleCustom(unittest.TestCase):
            def test_with_cpu_GraniteMLP_spyre_bfloat16(self):
                pass
            def test_unrelated(self):
                pass
        """)
    xml = tmp_path / "out.xml"
    result = pytester.runpytest(
        "-p", "tests.module_capability_plugin", f"--junitxml={xml}"
    )
    result.assert_outcomes(passed=2)
    props = {
        tc.get("name"): {p.get("name"): p.get("value") for p in tc.iter("property")}
        for tc in ET.parse(xml).iter("testcase")
    }
    assert (
        props["test_with_cpu_GraniteMLP_spyre_bfloat16"]["capability.name"]
        == "GraniteMLP"
    )
    assert (
        props["test_with_cpu_GraniteMLP_spyre_bfloat16"]["capability.subject"]
        == "granite"
    )
    assert props["test_unrelated"] == {}


def test_fallback_seen_per_test_despite_once_filter(pytester, tmp_path, monkeypatch):
    cfg = tmp_path / "granite.yaml"
    cfg.write_text("tags: [model__granite]\n")
    monkeypatch.setenv("PYTORCH_TEST_CONFIG", str(cfg))
    pytester.mkpydir("torch_spyre")
    pytester.mkpydir("torch_spyre/ops")
    pytester.makepyfile(**{"torch_spyre/ops/fallbacks": """
            import warnings
            class FallbackWarning(UserWarning):
                pass
            warnings.simplefilter("once", FallbackWarning)
            """})
    pytester.makepyfile(test_modules_custom="""
        import unittest, warnings
        from torch_spyre.ops.fallbacks import FallbackWarning
        class TestModuleCustom(unittest.TestCase):
            def test_with_cpu_A_spyre_bfloat16(self):
                warnings.warn("aten.arange.default is falling back to cpu", FallbackWarning)
            def test_with_cpu_B_spyre_bfloat16(self):
                warnings.warn("aten.arange.default is falling back to cpu", FallbackWarning)
            def test_with_cpu_C_spyre_bfloat16(self):
                pass
        """)
    xml = tmp_path / "out.xml"
    pytester.syspathinsert()
    result = pytester.runpytest(
        "-p", "tests.module_capability_plugin", f"--junitxml={xml}"
    )
    result.assert_outcomes(passed=3)
    backend = {
        tc.get("name"): {
            p.get("name"): p.get("value") for p in tc.iter("property")
        }.get("capability.backend")
        for tc in ET.parse(xml).iter("testcase")
    }
    assert backend == {
        "test_with_cpu_A_spyre_bfloat16": "cpu",
        "test_with_cpu_B_spyre_bfloat16": "cpu",
        "test_with_cpu_C_spyre_bfloat16": "spyre",
    }
