"""Unit tests for the weekly scan's schema-v2 capability rows.

Run with ``pytest --noconftest``, like ``test_weekly_prefilter.py``: ``capability_results`` is
pure, so no torch, database driver or ingest library is needed.
"""

from __future__ import annotations

from tests.spyre.weekly_generation.sink.capability_write import capability_results


def _row(**overrides):
    row = {
        "model_name": "org/model",
        "adapter_name": "causal_lm",
        "verified_on_cpu": True,
        "verified_on_gpu": False,
        "verified_on_spyre": False,
        "failure_category": "not-implemented-adapter",
    }
    row.update(overrides)
    return row


def test_no_gpu_verdict():
    backends = {r["backend"] for r in capability_results([_row()])}
    assert backends == {"cpu", "spyre"}


def test_status_and_fail_reason_per_backend():
    by_backend = {r["backend"]: r for r in capability_results([_row()])}
    assert by_backend["cpu"]["status"] == "passed"
    assert by_backend["cpu"]["fail_reason"] == ""
    assert by_backend["spyre"]["status"] == "failed"
    assert by_backend["spyre"]["fail_reason"] == "not-implemented-adapter"


def test_prefilter_skip_fails_tested_backends_only():
    rows = capability_results(
        [
            _row(
                adapter_name="",
                verified_on_cpu=False,
                failure_category="model_too_large",
            )
        ]
    )
    assert [(r["backend"], r["status"], r["name"]) for r in rows] == [
        ("cpu", "failed", "unknown"),
        ("spyre", "failed", "unknown"),
    ]
