# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline tests for the size estimator: no Hub requests are made."""

import unittest
from unittest.mock import MagicMock

import estimate_top_models_cache_size as estimate
from huggingface_hub.hf_api import ModelInfo


def _info(
    safetensors: dict[str, int] | None = None, files: list[str] | None = None
) -> ModelInfo:
    kwargs: dict[str, object] = {"id": "org/model"}
    if safetensors is not None:
        kwargs["safetensors"] = {
            "parameters": safetensors,
            "total": sum(safetensors.values()),
        }
    if files is not None:
        kwargs["siblings"] = [{"rfilename": name} for name in files]
    return ModelInfo(**kwargs)


def _row(model_info: ModelInfo | None, parameters: object = None) -> dict[str, object]:
    return {
        "model_id": "org/model",
        "mode": "generative",
        "curated": False,
        "downloads": 7,
        "parameters": parameters,
        "model_info": model_info,
    }


class SafetensorsBytesTests(unittest.TestCase):
    def test_sums_each_dtype_at_its_own_width(self) -> None:
        info: ModelInfo = _info({"F32": 10, "BF16": 10, "I8": 10, "F4": 10})
        self.assertEqual(estimate.safetensors_bytes(info), 40 + 20 + 10 + 5)

    def test_fp32_model_is_sized_at_four_bytes_per_parameter(self) -> None:
        info: ModelInfo = _info({"F32": 20_000_000_000})
        self.assertEqual(estimate.safetensors_bytes(info), 80_000_000_000)

    def test_unknown_dtype_falls_back_to_two_bytes(self) -> None:
        self.assertEqual(estimate.safetensors_bytes(_info({"NEW_DTYPE": 3})), 6)

    def test_no_metadata_gives_none(self) -> None:
        self.assertIsNone(estimate.safetensors_bytes(None))
        self.assertIsNone(estimate.safetensors_bytes(_info()))


class SizeModelTests(unittest.TestCase):
    def test_prefers_safetensors_over_parameter_count(self) -> None:
        sized = estimate.size_model(_row(_info({"F32": 100}), 100), 2.0, None)
        self.assertEqual((sized.estimated_bytes, sized.method), (400, "safetensors"))

    def test_parameter_fallback_uses_the_given_width(self) -> None:
        sized = estimate.size_model(_row(None, "1000"), 4.0, None)
        self.assertEqual((sized.estimated_bytes, sized.method), (4000, "parameters"))

    def test_unsizable_model_counts_zero_and_is_flagged(self) -> None:
        sized = estimate.size_model(_row(None, ""), 2.0, None)
        self.assertEqual((sized.estimated_bytes, sized.method), (0, "unknown"))

    def test_exact_counts_only_the_files_the_warmer_downloads(self) -> None:
        sibling = MagicMock
        api = MagicMock()
        api.model_info.return_value = MagicMock(
            siblings=[
                sibling(
                    rfilename="model.safetensors", size=None, lfs=MagicMock(size=500)
                ),
                sibling(
                    rfilename="pytorch_model.bin", size=None, lfs=MagicMock(size=500)
                ),
                sibling(
                    rfilename="onnx/model.onnx", size=None, lfs=MagicMock(size=900)
                ),
                sibling(rfilename="config.json", size=3, lfs=None),
            ]
        )
        info: ModelInfo = _info(files=["model.safetensors", "pytorch_model.bin"])
        sized = estimate.size_model(_row(info, 100), 2.0, api)
        self.assertEqual((sized.estimated_bytes, sized.method), (503, "exact"))

    def test_exact_failure_falls_back_to_metadata(self) -> None:
        api = MagicMock()
        api.model_info.side_effect = RuntimeError("hub down")
        sized = estimate.size_model(_row(_info({"BF16": 10}), 10), 2.0, api)
        self.assertEqual((sized.estimated_bytes, sized.method), (20, "safetensors"))


class SummaryTests(unittest.TestCase):
    def test_totals_split_by_type_and_method(self) -> None:
        models = [
            estimate.SizedModel(
                "a/x", "generative", False, 1, 1, 1024**3, "safetensors"
            ),
            estimate.SizedModel("b/y", "embedding", True, 1, None, 0, "unknown"),
        ]
        text: str = "\n".join(estimate._summary_lines(models))
        self.assertIn("Estimated download for 2 models: 1.0GB", text)
        self.assertIn("embedding", text)
        self.assertIn("unknown", text)


if __name__ == "__main__":
    unittest.main()
