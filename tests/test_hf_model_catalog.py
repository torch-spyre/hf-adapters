from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

import utils.hf_model_catalog as catalog
from utils.fetch_top_embedding_models import keep as keep_embedding
from utils.fetch_top_generative_models import keep as keep_generative


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
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Please pass `trust_remote_code=True` to allow custom code")
        ),
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


def test_unrelated_value_error_is_kept_and_logged(monkeypatch, caplog) -> None:
    def fail(*args, **kwargs):
        raise ValueError("unknown model type")

    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", fail)
    assert catalog.contains_remote_code(_model()) is False
    assert "unknown model type" in caplog.text


def test_retry_after_http_date_then_success(monkeypatch) -> None:
    calls = iter([_hub_error(429, retry_after="Wed, 21 Oct 2015 07:28:00 GMT"), None])
    sleeps = []

    def load(*args, **kwargs):
        error = next(calls)
        if error:
            raise error
        return object()

    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", load)
    monkeypatch.setattr(catalog.time, "sleep", sleeps.append)
    assert catalog.contains_remote_code(_model()) is False
    assert sleeps == [0.0]


@pytest.mark.parametrize("header", ["-1", "nan", "invalid"])
def test_invalid_retry_after_uses_backoff(monkeypatch, header) -> None:
    calls = iter([_hub_error(429, retry_after=header), None])
    sleeps = []

    def load(*args, **kwargs):
        error = next(calls)
        if error:
            raise error
        return object()

    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", load)
    monkeypatch.setattr(catalog.time, "sleep", sleeps.append)
    assert catalog.contains_remote_code(_model()) is False
    assert sleeps == [2.0]


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


@pytest.mark.parametrize("retry_after", ["120", "date"])
def test_503_recovery_respects_server_delay(monkeypatch, retry_after) -> None:
    if retry_after == "date":
        retry_after = format_datetime(
            datetime.now(timezone.utc) + timedelta(seconds=120)
        )
    calls = iter([_hub_error(503, retry_after=retry_after), None])
    sleeps = []

    def load(*args, **kwargs):
        error = next(calls)
        if error:
            raise error
        return object()

    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", load)
    monkeypatch.setattr(catalog.time, "sleep", sleeps.append)
    assert catalog.contains_remote_code(_model()) is False
    assert len(sleeps) == 1
    assert 118 <= sleeps[0] <= 120


@pytest.mark.parametrize("keep", [keep_generative, keep_embedding])
@pytest.mark.parametrize("remote_code", [False, True])
def test_fetcher_keep_handles_config_failures(monkeypatch, keep, remote_code) -> None:
    model = SimpleNamespace(
        id="org/model",
        library_name="sentence-transformers",
        config={"model_type": "bert"},
        tags=[],
        gated=False,
        siblings=[SimpleNamespace(rfilename="model.safetensors")],
    )

    def fail(*args, **kwargs):
        if remote_code:
            raise ValueError("Please pass `trust_remote_code=True`")
        raise _hub_error(503)

    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", fail)
    monkeypatch.setattr(catalog.time, "sleep", lambda delay: None)
    assert keep(model, False) == (
        (False, "requires trust_remote_code") if remote_code else (True, "")
    )
