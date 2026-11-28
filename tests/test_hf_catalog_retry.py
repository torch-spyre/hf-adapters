"""Offline regression tests for discovery throttling; run with --noconftest."""

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from threading import Event
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub import HfApi, ModelInfo
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import BertConfig

from utils import hf_model_catalog as catalog
from utils.fetch_curated_models_metadata import _create_fetch_metadata
from utils.fetch_top_embedding_models import keep as keep_embedding
from utils.fetch_top_generative_models import keep as keep_generative

_NOW = datetime(2026, 11, 22, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record delays without sleeping and reject any accidental live HTTP call."""
    delays: list[float] = []
    monkeypatch.setattr(catalog.time, "sleep", delays.append)
    monkeypatch.setattr(catalog.time, "time", lambda: _NOW.timestamp())

    def no_network(
        self: httpx.Client, request: httpx.Request, **kwargs: object
    ) -> httpx.Response:
        raise AssertionError(f"Unexpected HTTP request: {request.url}")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    return delays


def _http_error(
    status: int = 429, headers: dict[str, str] | None = None
) -> HfHubHTTPError:
    response = httpx.Response(
        status,
        headers=headers,
        request=httpx.Request("GET", "https://huggingface.co/api/models"),
    )
    return HfHubHTTPError("Simulated Hub failure", response=response)


def _model() -> ModelInfo:
    return ModelInfo(
        id="org/model",
        config={"model_type": "bert"},
        library_name="sentence-transformers",
        siblings=[{"rfilename": "model.safetensors"}],
    )


def _build_catalog(kind: str) -> list[dict[str, object]]:
    keeper: Callable[[ModelInfo, str | bool], tuple[bool, str]] = (
        keep_generative if kind == "generative" else keep_embedding
    )
    return catalog.build_catalog(
        fetch_fn=lambda limit: [_model()],
        filter_fn=lambda model: keeper(model, False),
        limit=1,
        output_csv=None,
        label=kind,
        token=False,
    )


@pytest.mark.parametrize(
    ("headers", "expected_wait"),
    [
        ({"Retry-After": "134"}, 135.0),
        ({"RateLimit": '"api";r=0;t=134'}, 135.0),
        ({"Retry-After": "2", "RateLimit": '"api";r=0;t=134'}, 135.0),
        ({"Retry-After": "180", "RateLimit": '"api";r=0;t=134'}, 181.0),
        (
            {
                "Retry-After": format_datetime(
                    _NOW + timedelta(seconds=134), usegmt=True
                )
            },
            135.0,
        ),
        ({"Retry-After": "0"}, 1.0),
        ({}, 301.0),
        ({"Retry-After": "invalid", "RateLimit": "invalid"}, 301.0),
        ({"Retry-After": "-1"}, 301.0),
        ({"RateLimit": '"api";r=10;t=134'}, 301.0),
    ],
)
def test_listing_recovers_after_advertised_wait(
    headers: dict[str, str], expected_wait: float, waits: list[float]
) -> None:
    request = Mock(side_effect=[_http_error(headers=headers), ["org/model"]])

    assert catalog.with_transient_retry(request, "listing") == ["org/model"]
    assert request.call_count == 2
    assert waits == [expected_wait]


def test_later_page_failure_restarts_without_duplicates(waits: list[float]) -> None:
    attempts = 0

    def listing() -> Iterator[str]:
        nonlocal attempts
        attempts += 1
        yield "org/first"
        if attempts == 1:
            raise _http_error(headers={"Retry-After": "134"})
        yield "org/second"

    assert catalog.with_transient_retry(listing, "listing") == [
        "org/first",
        "org/second",
    ]
    assert attempts == 2
    assert waits == [135.0]


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_gateway_errors_keep_exponential_backoff(
    status: int, waits: list[float]
) -> None:
    request = Mock(side_effect=[_http_error(status), _http_error(status), ["ok"]])

    assert catalog.with_transient_retry(request, "listing") == ["ok"]
    assert waits == [2.0, 4.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_permanent_listing_errors_are_not_retried(
    status: int, waits: list[float]
) -> None:
    error = _http_error(status)
    request = Mock(side_effect=error)

    with pytest.raises(HfHubHTTPError) as caught:
        catalog.with_transient_retry(request, "listing")

    assert caught.value is error
    assert request.call_count == 1
    assert waits == []


@pytest.mark.parametrize("lookup", ["remote_code", "config_type"])
@pytest.mark.parametrize("wrapping", ["direct", "cause", "context"])
def test_config_lookups_recover_from_wrapped_throttling(
    monkeypatch: pytest.MonkeyPatch,
    lookup: str,
    wrapping: str,
    waits: list[float],
) -> None:
    http_error = _http_error(headers={"RateLimit": '"resolvers";r=0;t=134'})
    error: Exception = http_error
    if wrapping != "direct":
        error = OSError("Unable to load config")
        if wrapping == "cause":
            error.__cause__ = http_error
        else:
            error.__context__ = http_error
    request = Mock(side_effect=[error, BertConfig()])
    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", request)

    if lookup == "remote_code":
        assert catalog.contains_remote_code(_model()) is False
    else:
        assert catalog.get_config_type("org/model", False) == "BertConfig"

    assert request.call_count == 2
    assert waits == [135.0]


@pytest.mark.parametrize("kind", ["generative", "embedding"])
def test_catalog_keeps_model_after_throttled_config_recovers(
    monkeypatch: pytest.MonkeyPatch, kind: str, waits: list[float]
) -> None:
    request = Mock(
        side_effect=[
            _http_error(headers={"Retry-After": "134"}),
            BertConfig(),
            BertConfig(),
        ]
    )
    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", request)

    rows: list[dict[str, object]] = _build_catalog(kind)

    assert [row["model_id"] for row in rows] == ["org/model"]
    assert rows[0]["config_class"] == "BertConfig"
    assert waits == [135.0]


@pytest.mark.parametrize(
    "phase", ["listing", "curated", "generative", "embedding", "config_type"]
)
@pytest.mark.parametrize("status", [429, 503])
def test_exhaustion_fails_visibly_instead_of_dropping_models(
    monkeypatch: pytest.MonkeyPatch, phase: str, status: int, waits: list[float]
) -> None:
    error = _http_error(status, {"Retry-After": "134"})
    request = Mock(side_effect=error)
    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", request)
    api = HfApi(token=False)
    monkeypatch.setattr(api, "model_info", request)

    with pytest.raises(
        catalog.HubRetryExhaustedError, match=f"{status} after 5 attempts"
    ) as caught:
        if phase == "listing":
            catalog.with_transient_retry(request, "listing")
        elif phase == "curated":
            _create_fetch_metadata(["org/model"])(api, 1)
        elif phase == "config_type":
            catalog.get_config_type("org/model", False)
        else:
            _build_catalog(phase)

    assert caught.value.__cause__ is error
    assert request.call_count == 5
    assert waits == ([135.0] * 4 if status == 429 else [2.0, 4.0, 8.0, 16.0])


def test_curated_metadata_recovers_after_throttling(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    api = HfApi(token=False)
    request = Mock(side_effect=[_http_error(headers={"Retry-After": "134"}), _model()])
    monkeypatch.setattr(api, "model_info", request)

    rows: list[ModelInfo] = _create_fetch_metadata(["org/model"])(api, 1)

    assert [row.id for row in rows] == ["org/model"]
    assert waits == [135.0]


def test_missing_curated_model_still_skips_without_retry(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    api = HfApi(token=False)
    request = Mock(side_effect=[_http_error(404), _model()])
    monkeypatch.setattr(api, "model_info", request)

    rows: list[ModelInfo] = _create_fetch_metadata(["org/missing", "org/model"])(api, 2)

    assert [row.id for row in rows] == ["org/model"]
    assert request.call_count == 2
    assert waits == []


def _wrapped_network_error(
    error: httpx.TransportError | None = None,
) -> OSError:
    """Mirror the chain Transformers raises when the Hub cannot be reached."""
    network_error: httpx.TransportError = error or httpx.ConnectError(
        "Connection reset"
    )
    hub_error = LocalEntryNotFoundError("Cannot find the requested files")
    hub_error.__cause__ = network_error
    wrapped = OSError("We couldn't connect to 'https://huggingface.co'")
    wrapped.__cause__ = hub_error
    return wrapped


def _wrapped_http_error(status: int) -> OSError:
    wrapped = OSError("There was a specific connection error")
    wrapped.__cause__ = _http_error(status)
    return wrapped


_REMOTE_CODE_ERROR = ValueError(
    "The repository org/model contains custom code which must be executed"
)


def test_remote_code_value_error_is_still_rejected(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    request = Mock(side_effect=_REMOTE_CODE_ERROR)
    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", request)

    assert catalog.contains_remote_code(_model()) is True
    assert keep_generative(_model(), False) == (False, "requires trust_remote_code")
    assert keep_embedding(_model(), False) == (False, "requires trust_remote_code")
    assert catalog.get_config_type("org/model", False) is None
    assert request.call_count == 4
    assert waits == []


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("Read-only HF_HOME"),
        OSError("Missing config"),
        _wrapped_http_error(401),
        _wrapped_http_error(403),
        _wrapped_http_error(404),
    ],
    ids=["permission", "missing-config", "http-401", "http-403", "http-404"],
)
def test_non_remote_code_failures_are_not_reclassified(
    monkeypatch: pytest.MonkeyPatch, error: OSError, waits: list[float]
) -> None:
    request = Mock(side_effect=error)
    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", request)
    reason: str = f"exception during contains_remote_code ({type(error).__name__})"

    with pytest.raises(type(error)) as caught:
        catalog.contains_remote_code(_model())
    assert caught.value is error
    assert keep_generative(_model(), False) == (False, reason)
    assert keep_embedding(_model(), False) == (False, reason)
    # A degraded config_class is still the right outcome for the enrichment step.
    assert catalog.get_config_type("org/model", False) is None
    assert request.call_count == 4
    assert waits == []


@pytest.mark.parametrize("lookup", ["remote_code", "config_type", "listing"])
@pytest.mark.parametrize(
    "network_error",
    [
        httpx.ConnectError("Connection reset"),
        httpx.ReadError("Connection reset while reading"),
        httpx.WriteError("Connection reset while writing"),
        httpx.ReadTimeout("Read timed out"),
        httpx.RemoteProtocolError("Server disconnected"),
    ],
    ids=["connect", "read", "write", "timeout", "protocol"],
)
def test_network_errors_are_retried(
    monkeypatch: pytest.MonkeyPatch,
    lookup: str,
    network_error: httpx.TransportError,
    waits: list[float],
) -> None:
    error: Exception = _wrapped_network_error(network_error)
    if lookup == "listing":
        request = Mock(side_effect=[error, ["org/model"]])
        assert catalog.with_transient_retry(request, "listing") == ["org/model"]
    else:
        request = Mock(side_effect=[error, BertConfig()])
        monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", request)
        if lookup == "remote_code":
            assert catalog.contains_remote_code(_model()) is False
        else:
            assert catalog.get_config_type("org/model", False) == "BertConfig"

    assert request.call_count == 2
    assert waits == [2.0]


@pytest.mark.parametrize("kind", ["generative", "embedding"])
@pytest.mark.parametrize(
    "network_error_type", [httpx.ConnectError, httpx.ReadError, httpx.WriteError]
)
def test_persistent_network_errors_fail_visibly(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    network_error_type: type[httpx.NetworkError],
    waits: list[float],
) -> None:
    error: OSError = _wrapped_network_error(network_error_type("Connection reset"))
    request = Mock(side_effect=error)
    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", request)

    with pytest.raises(
        catalog.HubRetryExhaustedError,
        match=f"{network_error_type.__name__} after 5 attempts",
    ) as caught:
        _build_catalog(kind)

    assert caught.value.__cause__ is error
    assert request.call_count == 5
    assert waits == [2.0, 4.0, 8.0, 16.0]


@pytest.mark.parametrize("phase", ["filter", "enrichment"])
def test_exhaustion_cancels_queued_config_lookups(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    monkeypatch.setattr(catalog, "_MAX_FETCH_ATTEMPTS", 1)
    monkeypatch.setenv("HF_CATALOG_MAX_WORKERS", "1")
    models: list[ModelInfo] = [_model() for _ in range(20)]
    for index, model in enumerate(models):
        model.id = f"org/model-{index}"
    requested: list[str] = []

    def load_config(model_id: str, **kwargs: object) -> BertConfig:
        requested.append(model_id)
        if model_id == "org/model-0":
            raise _http_error()
        # Give the main thread time to observe the failure and cancel the queue.
        Event().wait(0.01)
        return BertConfig()

    monkeypatch.setattr(catalog.AutoConfig, "from_pretrained", load_config)

    def keep(model: ModelInfo) -> tuple[bool, str]:
        if phase == "filter":
            return keep_generative(model, False)
        return True, ""

    with pytest.raises(catalog.HubRetryExhaustedError):
        catalog.build_catalog(
            fetch_fn=lambda limit: models,
            filter_fn=keep,
            limit=len(models),
            output_csv=None,
            label="generative",
            token=False,
        )

    assert requested[0] == "org/model-0"
    assert len(requested) < len(models)
