from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

import utils.hf_model_catalog as catalog


def _model(model_id: str = "org/model") -> SimpleNamespace:
    return SimpleNamespace(id=model_id)


def _hub_error(status: int, *, retry_after: str | None = None) -> HfHubHTTPError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        status,
        headers=headers,
        request=httpx.Request("GET", "https://huggingface.co/org/model"),
    )
    return HfHubHTTPError(f"HTTP {status}", response=response)


def test_contains_remote_code_accepts_native_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        catalog.AutoConfig, "from_pretrained", lambda *args, **kwargs: object()
    )

    assert catalog.contains_remote_code(_model()) is False


def test_contains_remote_code_only_classifies_explicit_remote_code_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        catalog.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("custom code")),
    )

    assert catalog.contains_remote_code(_model()) is True


def test_contains_remote_code_retries_429_and_keeps_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise _hub_error(429, retry_after="7")

    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", fail)
    monkeypatch.setattr(catalog.time, "sleep", sleeps.append)

    assert catalog.contains_remote_code(_model()) is False
    assert calls == catalog._MAX_FETCH_ATTEMPTS
    assert sleeps == [7.0] * (catalog._MAX_FETCH_ATTEMPTS - 1)


def test_contains_remote_code_keeps_permission_errors_for_downstream_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        catalog.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            PermissionError("read-only cache")
        ),
    )

    assert catalog.contains_remote_code(_model()) is False
