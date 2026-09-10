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

"""Which test tiers each suite belongs to, for the JUnit `testtype__<tier>` tags.

WHY THE TAGS EXIST. The CI/CD warehouse answers "has this artifact already been tested
at tier X" by reading these tags off the ingested JUnit XML. A run therefore has to record
every tier its tests BELONG to, not the one tier that happened to invoke it: a `regression`
run of a suite that is also a `unit` and `integration` member must say so, or a later
`integration` run finds no coverage and re-executes identical work.

WHY A TABLE HERE. A suite target (`make smoke-tests`) does not know its tiers -- the
mapping lives in the CI job gates, and CI calls the per-suite targets directly, never
`make tests`. So the tier cannot be recovered at test time from TEST_TYPE; it has to be
declared. `--suite <key>` names the suite, and this table turns that into the tier set.

MEMBERSHIP IS DECLARED, NEVER INFERRED FROM A LADDER. It is tempting to treat the tiers as
nested (unit < integration < regression < trunk) and expand upward. Do not: 11 of the 15
suites below are `[regression, trunk, unit]` with NO `integration`, so closing the ladder
would claim integration coverage for suites that never ran under it and silently skip real
tests. torch-spyre learned the same lesson -- see the "never inferred from a ladder" note
in its filter_configs.py.

SOURCE OF TRUTH is each suite job's `if:` gate in .github/workflows/_test_matrix.yaml,
because that is what actually decides whether a suite runs. The Makefile's `tests` target
carries a second, HAND-MAINTAINED copy in its `case` block, and the two have already
drifted (see tests/test_tier_tags.py, which pins the difference): the Makefile omits `clip`
from every suites= list, has no `multicard_smoke` target, and puts `model_components` only
in regression/trunk even though Makefile:8 documents it as an integration suite. Keep this
table in step with the WORKFLOW; the test reports drift rather than guessing.
"""

from __future__ import annotations

# suite key -> the tiers whose runs include that suite.
# Mirrors the `if:` gate of each suite job in .github/workflows/_test_matrix.yaml.
# Five suites are deliberately absent. tier_tags() returns [] for any of them, so their
# cases still carry a model tag -- they simply contribute no tier-coverage data.
#   perf             -- a scaffold that echoes an empty JUnit file with no <testcase>
#                       elements (Makefile `tests` target): nothing to tag.
#   edge_cases       -- gated on `inputs.edge_cases_only`, never on a tier, and absent
#                       from every suites= list. Its target still passes --suite so the
#                       key is declared in one place if it ever joins a tier.
#   multicard_smoke  -- CI runs scripts/run_multicard_smoke.py under torchrun, not
#                       pytest, so it emits no JUnit XML at all.
#   adapter_coverage -- runs with --noconftest (Makefile), so the autouse fixture never
#                       binds; passing --suite there would be an unknown-option error.
#   model_module     -- delegates to the oot_framework run_test.sh, which does its own
#                       marker-based tagging (torch-spyre's mechanism), not this one.
SUITE_TIERS: dict[str, tuple[str, ...]] = {
    "clip": ("regression", "trunk", "unit"),
    "embed_compare": ("regression", "trunk", "unit"),
    "load": ("regression", "trunk", "unit"),
    "masked_lm_compare": ("regression", "trunk", "unit"),
    "model_components": ("integration", "regression", "trunk", "unit"),
    "question_answering_compare": ("regression", "trunk", "unit"),
    "reranker_compare": ("regression", "trunk", "unit"),
    "seq_classification_compare": ("regression", "trunk", "unit"),
    "smoke": ("regression", "trunk"),
    "token_classification_compare": ("regression", "trunk", "unit"),
    "token_compare": ("integration", "regression", "trunk", "unit"),
    "vlm": ("regression", "trunk", "unit"),
}

# Parametrize argnames whose value names the model under test. `model_path` covers almost
# every parametrized test here (tests/conftest.py's pytest_generate_tests owns that axis
# and can rewrite it from --model-path); the rest are for the few suites using other names.
MODEL_PARAM_NAMES = ("model_path", "model", "model_key", "model_info")


def tier_tags(suite: str) -> list[str]:
    """`testtype__<tier>` for every tier `suite` belongs to; empty for an unknown suite.

    Empty rather than raising: an untagged case is a gap in reuse data, while a raise
    would fail a test run over a reporting concern.
    """
    return [f"testtype__{tier}" for tier in SUITE_TIERS.get(suite, ())]


def model_tag(params) -> str | None:
    """`model__<id>` from a test's parametrization, or None when no model param is bound."""
    for name in MODEL_PARAM_NAMES:
        if name not in params:
            continue
        value = params[name]
        if value is None:
            continue
        # (model_id, ...) tuple: the id is the first element.
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        # vLLM-style model-info objects carry the id on .name.
        name_attr = getattr(value, "name", None)
        text = str(name_attr if name_attr is not None else value).strip()
        return f"model__{text}" if text else None
    return None


def result_tags(suite: str, params) -> list[tuple[str, str]]:
    """The (name, value) JUnit property pairs for one test case.

    Emitted as `<property name="tag" value="namespace__value"/>`, the shape the ClickHouse
    ingest reads (see .github/scripts/ingest_xml_hf_adapters.py extract_properties).
    """
    tags: list[tuple[str, str]] = []
    model = model_tag(params)
    if model:
        tags.append(("tag", model))
    for tag in tier_tags(suite):
        tags.append(("tag", tag))
    return tags
