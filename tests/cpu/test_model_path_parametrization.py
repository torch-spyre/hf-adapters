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

"""Model overrides preserve scenarios and use the selected harness's failure policy."""

import pytest

pytest_plugins = ["pytester"]


@pytest.mark.parametrize("harness", ["multimodal_embedding", "causal"])
@pytest.mark.parametrize("override", [None, "known/model", "new/model"])
def test_model_override_uses_harness_policy(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    override: str | None,
    harness: str,
) -> None:
    import model_registry

    monkeypatch.setattr(
        model_registry, "NON_BLOCKING_CAUSAL_MODELS", {"known/model": "known failure"}
    )
    pytester.makeconftest(
        "from tests.conftest import pytest_addoption, pytest_configure, pytest_generate_tests"
    )
    pytester.makepyfile(f"""
        import pytest
        from model_registry import NON_BLOCKING_CAUSAL_MODELS, xfail_non_blocking

        pytestmark = pytest.mark.model_harness({harness!r})
        table = NON_BLOCKING_CAUSAL_MODELS if {harness!r} == "causal" else {{}}

        @pytest.mark.parametrize("case", ["scalar", "batch"])
        @pytest.mark.parametrize(
            "model_path", xfail_non_blocking(["known/model"], table=table)
        )
        def test_accuracy(model_path: str, case: str) -> None:
            assert model_path != "known/model", case
    """)
    args: list[str] = ["-q"]
    if override is not None:
        args.extend(["--model-path", override])
    result = pytester.runpytest(*args)
    if override == "new/model":
        result.assert_outcomes(passed=2)
    elif harness == "causal":
        result.assert_outcomes(xfailed=2)
    else:
        result.assert_outcomes(failed=2)
