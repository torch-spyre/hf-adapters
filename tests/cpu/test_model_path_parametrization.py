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

"""Checkpoint selection must preserve narrowly scoped pytest parameter marks."""

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("override", [None, "known/model", "new/model"])
def test_model_override_preserves_only_matching_marks(
    pytester: pytest.Pytester, override: str | None
) -> None:
    pytester.makeconftest(
        "from tests.conftest import pytest_addoption, pytest_configure, pytest_generate_tests"
    )
    pytester.makepyfile("""
        import pytest

        class AccuracyError(AssertionError):
            pass

        pytestmark = pytest.mark.model_harness("multimodal_embedding")
        affected = pytest.param(
            "known/model", marks=pytest.mark.xfail(
                reason="known numerical limit", raises=AccuracyError, strict=False,
            ),
        )

        @pytest.mark.parametrize("case", ["scalar", "batch"])
        @pytest.mark.parametrize("model_path", [affected])
        def test_accuracy(model_path: str, case: str) -> None:
            raise AccuracyError(case)

        @pytest.mark.parametrize("model_path", [affected])
        def test_structure(model_path: str) -> None:
            raise AssertionError("missing tower output")

        @pytest.mark.parametrize("model_path", ["known/model"])
        def test_other_check(model_path: str, request: pytest.FixtureRequest) -> None:
            assert request.node.get_closest_marker("xfail") is None
    """)
    args: list[str] = ["-q"]
    if override is not None:
        args.extend(["--model-path", override])
    result = pytester.runpytest(*args)
    if override == "new/model":
        result.assert_outcomes(passed=1, failed=3)
    else:
        result.assert_outcomes(passed=1, failed=1, xfailed=2)
