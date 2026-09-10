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

"""CPU-only tests for the JUnit tier tags (no hardware, no models needed).

Two invariants matter here:

1. `SUITE_TIERS` matches the workflow's `if:` gates, which are what actually decide
   whether a suite runs. A drifted table silently mis-reports coverage.
2. Membership is never expanded up a tier ladder. 11 of the gated suites are
   `[regression, trunk, unit]` with NO `integration`, so closing the ladder would claim
   integration coverage for suites that never ran under it and skip real tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from tests._tier_tags import (
    SUITE_TIERS,
    model_tag,
    result_tags,
    tier_tags,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "_test_matrix.yaml"
_MAKEFILE = _REPO_ROOT / "Makefile"

_TIERS = ("smoke", "unit", "integration", "regression", "trunk")

# Suites that intentionally carry no tier; see the module docstring in _tier_tags.py for
# why each one is out.
_UNTIERED = {
    "perf",
    "edge_cases",
    "multicard_smoke",
    "adapter_coverage",
    "model_module",
}


def _workflow_gates() -> dict[str, set[str]]:
    """suite key -> tiers, parsed from each suite job's `if:` gate."""
    doc = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    gates: dict[str, set[str]] = {}
    for spec in doc["jobs"].values():
        cond = spec.get("if") or ""
        if "resolve-test-type" not in cond:
            continue
        suite = re.search(r"contains\(format\(' \{0\} ',[^)]*\), ' ([a-z_]+) '\)", cond)
        if not suite:
            continue
        gates[suite.group(1)] = {
            t for t in _TIERS if re.search(rf"outputs\.test_type == '{t}'", cond)
        }
    return gates


# ── the table tracks the workflow ───────────────────────────────────────────────


def test_workflow_gates_are_parseable():
    """Guards the parser itself: if the `if:` shape changes, the drift test below would
    silently compare against an empty dict and pass."""
    gates = _workflow_gates()
    assert len(gates) >= 12, f"only parsed {len(gates)} suite gates"
    assert all(tiers for tiers in gates.values()), "a gate parsed with no tiers"


def test_table_matches_the_workflow_gates():
    """The workflow is the source of truth -- it is what decides whether a suite runs."""
    gates = _workflow_gates()
    mismatches = {
        suite: (sorted(tiers), sorted(SUITE_TIERS.get(suite, ())))
        for suite, tiers in gates.items()
        if suite not in _UNTIERED and tiers != set(SUITE_TIERS.get(suite, ()))
    }
    assert not mismatches, f"SUITE_TIERS drifted from the workflow: {mismatches}"


def test_every_tiered_table_entry_is_a_real_workflow_suite():
    """The other direction: no invented suite keys."""
    gates = _workflow_gates()
    unknown = sorted(set(SUITE_TIERS) - set(gates))
    msg = f"SUITE_TIERS names suites the workflow does not gate: {unknown}"
    assert not unknown, msg


def test_untiered_suites_are_absent_from_the_table():
    assert not (_UNTIERED & set(SUITE_TIERS))


# ── the Makefile carries a SECOND, already-drifted copy ─────────────────────────


def _makefile_suites_by_tier() -> dict[str, set[str]]:
    """suite key -> tiers, from the `tests` target's `case` block."""
    mk = _MAKEFILE.read_text(encoding="utf-8")
    patterns = {
        (
            "regression",
            "trunk",
        ): r'\*" regression "\*\|\*" trunk "\*\) suites="([^"]+)"',
        ("unit",): r'\*" unit "\*\) suites="([^"]+)"',
        ("integration",): r'" integration "\) suites="([^"]+)"',
    }
    out: dict[str, set[str]] = {}
    for tiers, pat in patterns.items():
        m = re.search(pat, mk)
        assert m, f"could not find the {tiers} branch of the Makefile case block"
        for suite in m.group(1).split():
            out.setdefault(suite, set()).update(tiers)
    return out


def test_the_makefile_copy_is_known_to_disagree_with_the_workflow():
    """Documents a PRE-EXISTING divergence rather than asserting the two agree.

    `make tests` and CI maintain the tier->suite mapping separately, and they have
    already drifted. This test pins the known set so a NEW divergence shows up as a
    failure, and so nobody "fixes" _tier_tags.py against the wrong copy. Reconciling
    them is a separate change with real behaviour impact -- it would alter which suites
    `make tests` runs.
    """
    workflow = _workflow_gates()
    makefile = _makefile_suites_by_tier()
    diffs = {
        suite: (
            sorted(workflow.get(suite, set())) or None,
            sorted(makefile.get(suite, set())) or None,
        )
        for suite in set(workflow) | set(makefile)
        if workflow.get(suite, set()) != makefile.get(suite, set())
    }
    assert diffs == {
        # CI has no tier gate for it (runs unconditionally); the Makefile tiers it.
        "adapter_coverage": (None, ["regression", "trunk", "unit"]),
        # Has a Makefile target but appears in no suites= list, so `make tests` never
        # runs it.
        "clip": (["regression", "trunk", "unit"], None),
        # CI-only: no Makefile target at all (CI drives torchrun directly).
        "multicard_smoke": (["regression", "trunk", "unit"], None),
        # Makefile:8 documents this as an integration suite and CI agrees; the
        # Makefile's own case block omits it from unit AND integration.
        "model_components": (
            ["integration", "regression", "trunk", "unit"],
            ["regression", "trunk"],
        ),
    }, f"the workflow/Makefile divergence changed: {diffs}"


# ── no ladder inference ────────────────────────────────────────────────────────


def test_a_ladder_would_over_claim_this_repos_suites():
    """The reason membership is declared and not inferred, re-derived from the table.

    If this ever finds zero over-claims, the ladder has become safe FOR THESE SUITES --
    a property of the current gates, not a licence to start inferring it.
    """
    ladder = ("unit", "integration", "regression", "trunk")
    over = []
    for suite, declared in SUITE_TIERS.items():
        idx = [ladder.index(t) for t in declared if t in ladder]
        if not idx:
            continue
        missing = set(ladder[min(idx) :]) - set(declared)
        if missing:
            over.append((suite, sorted(missing)))
    assert over, "a ladder no longer over-claims; do not start inferring it"


def test_tags_are_the_declared_set_not_the_ladder_closure():
    """A `[regression, trunk, unit]` suite must NOT be tagged integration, even though
    integration sits between unit and regression in the ladder."""
    tags = tier_tags("load")
    assert "testtype__integration" not in tags
    assert sorted(tags) == [
        "testtype__regression",
        "testtype__trunk",
        "testtype__unit",
    ]


def test_an_integration_suite_is_tagged_integration():
    assert "testtype__integration" in tier_tags("token_compare")


# ── behaviour at the edges ─────────────────────────────────────────────────────


def test_unknown_suite_tags_nothing():
    """Empty, not an exception: a reporting gap must not fail a test run."""
    assert tier_tags("no_such_suite") == []
    assert tier_tags("") == []


def test_untiered_suite_tags_no_tier():
    assert tier_tags("perf") == []
    assert tier_tags("multicard_smoke") == []


# ── the model tag ─────────────────────────────────────────────────────────────


def test_model_tag_from_model_path():
    assert model_tag({"model_path": "ibm-granite/granite-3.3-8b"}) == (
        "model__ibm-granite/granite-3.3-8b"
    )


def test_model_tag_takes_the_first_element_of_a_tuple():
    assert model_tag({"model_path": ("ibm/granite", "extra")}) == "model__ibm/granite"


def test_model_tag_prefers_a_name_attribute():
    class Info:
        name = "ibm/granite"

    assert model_tag({"model_info": Info()}) == "model__ibm/granite"


def test_model_tag_absent_when_no_model_param():
    assert model_tag({}) is None
    assert model_tag({"dtype": "float16"}) is None


def test_model_tag_absent_for_none_or_blank():
    assert model_tag({"model_path": None}) is None
    assert model_tag({"model_path": "  "}) is None


# ── the emitted pairs ─────────────────────────────────────────────────────────


def test_result_tags_emits_model_then_tiers():
    pairs = result_tags("token_compare", {"model_path": "ibm/granite"})
    assert [n for n, _ in pairs] == ["tag"] * len(pairs)
    assert [v for _, v in pairs] == [
        "model__ibm/granite",
        "testtype__integration",
        "testtype__regression",
        "testtype__trunk",
        "testtype__unit",
    ]


def test_result_tags_is_empty_when_there_is_nothing_to_say():
    assert result_tags("", {}) == []


def test_tag_values_match_the_ingest_namespace_form():
    """`namespace__value`, the shape ingest_xml_hf_adapters.extract_properties reads off
    `<property name="tag" value="..."/>`."""
    for name, value in result_tags("smoke", {"model_path": "ibm/granite"}):
        assert name == "tag"
        assert re.fullmatch(r"[a-z_]+__\S+", value), value
