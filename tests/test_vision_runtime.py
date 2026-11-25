"""Retained exception graphs must not retain callee-owned resources."""

from __future__ import annotations

import gc
import traceback
import weakref
from builtins import ExceptionGroup

import pytest

from tests._vision_runtime import clear_exception_frames


class _Resource:
    pass


@pytest.mark.parametrize("cycle", [False, True])
def test_grouped_exception_cleanup_releases_resources_and_keeps_tracebacks(
    cycle: bool,
) -> None:
    resource_refs: list[weakref.ReferenceType[_Resource]] = []

    def fail() -> None:
        resource = _Resource()
        resource_refs.append(weakref.ref(resource))
        raise RuntimeError("retained resource")

    def capture() -> RuntimeError:
        try:
            fail()
        except RuntimeError as error:
            return error
        raise AssertionError("failure was not raised")

    causes = [capture(), capture()]
    grouped = ExceptionGroup("grouped failures", causes)
    if cycle:
        causes[0].__cause__ = grouped
    clear_exception_frames(grouped)
    gc.collect()
    assert all(resource_ref() is None for resource_ref in resource_refs)
    assert all(cause.__traceback__ is not None for cause in causes)
    formatted = "".join(traceback.format_exception(grouped))
    assert "fail" in formatted and "retained resource" in formatted
