"""Reference-stage reporting and resource cleanup shared by vision verifiers."""

import traceback
from builtins import BaseExceptionGroup
from typing import Literal

ReferenceStage = Literal["load", "forward"]


def clear_exception_frames(error: BaseException) -> None:
    """Release callee locals while preserving traceback locations and error chains.

    Active frames cannot be cleared; their resource owners still use finally to
    delete model references. Unwound callees can hold models and device caches,
    including in chained compiler errors or grouped exceptions.
    """
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
