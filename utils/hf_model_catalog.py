"""Shared building blocks for the Hugging Face model-catalog fetchers.

Both ``fetch_top_generative_models.py`` and ``fetch_top_embedding_models.py``
pull models from the Hub, enrich them with config/param metadata, and write a
ranked CSV. Everything they have in common lives here; each script only has to
supply how it *sources* candidates, how it *filters* them, and any *extra
columns* it wants on top of the shared schema.
"""

import csv
import logging
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal, TypeVar

import httpx
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.hf_api import ExpandModelProperty_T, ModelInfo
from huggingface_hub.utils import RateLimitInfo, parse_ratelimit_headers
from tqdm import tqdm
from transformers import AutoConfig, PretrainedConfig

logging.getLogger("transformers").setLevel(logging.ERROR)


# Get the resources directory (parent of resources/__init__.py)
RESOURCES_DIR: Path = Path(__file__).resolve().parent.parent / "resources"

# Manually-maintained model-id lists (one id per line; '#' comments and blank
# lines ignored). See load_curated_model_ids().
CURATED_GENERATIVE_MODELS_FILE: Path = RESOURCES_DIR / "generative_models_curated.txt"
CURATED_EMBEDDING_MODELS_FILE: Path = RESOURCES_DIR / "embedding_models_curated.txt"


def load_curated_model_ids(path: Path) -> list[str]:
    """Load Hugging Face model ids from a curated list file.

    The file holds one model id per line. Blank lines and lines whose first
    non-whitespace character is ``#`` are ignored, as is any inline ``#``
    comment following an id. Surrounding whitespace is stripped. Order is
    preserved and duplicates are dropped (first occurrence wins).
    """
    seen: set[str] = set()
    model_ids: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line in seen:
            continue
        seen.add(line)
        model_ids.append(line)
    return model_ids


def load_curated_generative_models() -> list[str]:
    return load_curated_model_ids(CURATED_GENERATIVE_MODELS_FILE)


def load_curated_embedding_models() -> list[str]:
    return load_curated_model_ids(CURATED_EMBEDDING_MODELS_FILE)


# Metadata fields requested from list_models/model_info for every fetcher.
# Typed as the hub's own ExpandModelProperty_T literal rather than list[str], so a
# typo here is a type error instead of a runtime 400 from the API.
EXPAND_FIELDS: list[ExpandModelProperty_T] = [
    "config",
    "safetensors",
    "gated",
    "likes",
    "downloads",
    "trendingScore",
    "createdAt",
    "library_name",
    "tags",
    "siblings",
]

# How a ranked fetcher orders the Hub before taking its top-K. A subset of the hub's
# ModelSort_T: "downloads" is the last-30-days download count (all-time is the
# separate downloads_all_time field), and "trending_score" is the Hub's
# recent-activity ranking -- the closest thing to "likes in the last 30 days", since
# ``likes`` itself is an all-time total and the API exposes no per-period likes.
Ranking = Literal["downloads", "trending_score"]
RANKINGS: tuple[Ranking, ...] = ("downloads", "trending_score")

# Threads build_catalog uses for its per-model Hub calls (the keep() filter and the
# AutoConfig config-class lookup). Both are network-bound, so this bounds how many
# requests hit the Hub at once. 8 (down from a hard-coded 32) eases rate-limit
# pressure; the slower filter is affordable at the weekly scan's 1000-per-ranking
# sizes. Override with HF_CATALOG_MAX_WORKERS.
DEFAULT_CATALOG_MAX_WORKERS: int = 8


def catalog_max_workers() -> int:
    """Thread count for build_catalog: HF_CATALOG_MAX_WORKERS, else the default."""
    raw: str = os.environ.get("HF_CATALOG_MAX_WORKERS", "").strip()
    if not raw:
        return DEFAULT_CATALOG_MAX_WORKERS
    try:
        value: int = int(raw)
    except ValueError:
        raise ValueError(
            f"HF_CATALOG_MAX_WORKERS must be a positive integer, got {raw!r}"
        ) from None
    if value < 1:
        raise ValueError(f"HF_CATALOG_MAX_WORKERS must be >= 1, got {value}")
    return value


# Transport, throttling and gateway failures can recover; other HTTP errors are
# not retried.
_TRANSIENT_HTTP_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MAX_FETCH_ATTEMPTS: int = 5
_MAX_BACKOFF_SECONDS: float = 60.0
# Without usable response headers, wait a full HF quota window before retrying.
_RATE_LIMIT_FALLBACK_SECONDS: float = 300.0
_RATE_LIMIT_BUFFER_SECONDS: float = 1.0

_T = TypeVar("_T")


class HubRetryExhaustedError(RuntimeError):
    """Discovery cannot continue without silently losing models or metadata."""


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    """Walk failures wrapped by Transformers or fsspec without following cycles."""
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _hub_http_error(error: BaseException) -> HfHubHTTPError | None:
    """Find HTTP failures that Transformers may wrap in an OSError."""
    for current in _exception_chain(error):
        if isinstance(current, HfHubHTTPError):
            return current
    return None


def _hub_transport_error(
    error: BaseException, retry_os_errors: bool
) -> BaseException | None:
    """Find transport failures, optionally including ambiguous remote-file I/O."""
    os_error: OSError | None = None
    for current in _exception_chain(error):
        if isinstance(current, (httpx.TransportError, ConnectionError, TimeoutError)):
            return current
        if isinstance(current, (FileNotFoundError, PermissionError)):
            return None
        if retry_os_errors and isinstance(current, OSError):
            os_error = current
    return os_error


def _rate_limit_delay(error: HfHubHTTPError) -> float:
    """Honor both server hints without applying the short gateway-backoff cap."""
    delays: list[float] = []
    if error.response is not None:
        headers: Mapping[str, str] = error.response.headers
        retry_after: str = headers.get("Retry-After", "").strip()
        if retry_after.isdigit():
            delays.append(float(retry_after))
        elif retry_after:
            try:
                retry_at: datetime = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delays.append(max(0.0, retry_at.timestamp() - time.time()))
            except (TypeError, ValueError, OverflowError):
                pass

        rate_limit: RateLimitInfo | None = parse_ratelimit_headers(headers)
        if rate_limit is not None and rate_limit.remaining == 0:
            delays.append(float(rate_limit.reset_in_seconds))

    return (
        max(delays, default=_RATE_LIMIT_FALLBACK_SECONDS) + _RATE_LIMIT_BUFFER_SECONDS
    )


def _retry_hub_call(
    call: Callable[[], _T], description: str, *, retry_os_errors: bool = False
) -> _T:
    """Retry transport, throttling and gateway errors, including wrapped failures.

    Exhaustion has a distinct exception so model-level guards cannot turn a
    temporary Hub outage into rejected models or missing config metadata.
    Remote checkpoint inspection enables retry_os_errors for fsspec I/O;
    missing files and permission errors remain permanent failures.
    """
    for attempt in range(1, _MAX_FETCH_ATTEMPTS + 1):
        try:
            return call()
        except Exception as error:
            http_error: HfHubHTTPError | None = _hub_http_error(error)
            status: int | None = (
                http_error.response.status_code
                if http_error is not None and http_error.response is not None
                else None
            )
            transport_error: BaseException | None = (
                _hub_transport_error(error, retry_os_errors)
                if http_error is None
                else None
            )
            failure: str
            if http_error is not None and status in _TRANSIENT_HTTP_STATUSES:
                failure = f"HF API returned {status}"
            elif transport_error is not None:
                failure = f"HF transport failed with {type(transport_error).__name__}"
            else:
                raise
            if attempt == _MAX_FETCH_ATTEMPTS:
                raise HubRetryExhaustedError(
                    f"{description}: {failure} after {attempt} attempts"
                ) from error
            backoff: float = (
                _rate_limit_delay(http_error)
                if http_error is not None and status == 429
                else min(_MAX_BACKOFF_SECONDS, 2.0**attempt)
            )
            print(
                f"    {description}: {failure} "
                f"(attempt {attempt}/{_MAX_FETCH_ATTEMPTS}); "
                f"retrying in {backoff:.1f}s...",
                flush=True,
            )
            time.sleep(backoff)
    raise AssertionError("HF retry loop requires at least one attempt")


def with_transient_retry(
    call: Callable[[], Iterator[_T] | Iterable[_T]],
    description: str,
) -> list[_T]:
    """Materialize an HF listing, retrying transport, HTTP 429 and 5xx failures.

    Consume the iterable inside the retry boundary so a failure on a later page
    restarts the traversal without returning partial or duplicate results.
    """
    return _retry_hub_call(lambda: list(call()), description)


MOE_MODEL_TYPES: set[str] = {
    "mixtral",
    "qwen2_moe",
    "qwen3_moe",
    "dbrx",
    "jamba",
    "arctic",
    "olmoe",
    "gpt_oss",
}

MOE_MODEL_TYPE_PREFIXES: tuple[str, ...] = ("deepseek_v2", "deepseek_v3", "deepseek_v4")

MOE_ARCH_SUBSTRINGS: list[str] = [
    "mixtral",
    "moe",
    "dbrx",
    "jamba",
    "arctic",
    "olmoe",
    "deepseek",
    "gptoss",
]


def tags(model: ModelInfo) -> set[str]:
    """Lower-cased set of a model's tags (empty set if none)."""
    return {t.lower() for t in (getattr(model, "tags", None) or [])}


def _is_moe(model: ModelInfo) -> bool:
    if any("moe" in t for t in tags(model)):
        return True

    config: dict = model.config or {}
    model_type: str = (config.get("model_type") or "").lower()
    if model_type in MOE_MODEL_TYPES:
        return True
    if model_type.startswith(MOE_MODEL_TYPE_PREFIXES):
        return True

    architectures: list[str] = config.get("architectures") or []
    arch_lower: str = " ".join(architectures).lower()
    return any(sub in arch_lower for sub in MOE_ARCH_SUBSTRINGS)


def is_custom_code(model: ModelInfo) -> bool:
    if "custom_code" in tags(model):
        return True
    config: dict = model.config or {}
    return bool(config.get("auto_map"))


def is_nsfw(model: ModelInfo) -> bool:
    if "nsfw" in tags(model):
        return True
    return False


# Repo-id substrings marking non-native conversions (ONNX/GGUF/MLX), dropped.
NON_NATIVE_ID_SUBSTRINGS: tuple[str, ...] = ("onnx", "gguf", "mlx")


def is_baseline_keep(model: ModelInfo) -> tuple[bool, str]:
    """Shared inclusion gate: drop config-less, and ONNX/GGUF/MLX id checkpoints.

    Returns (keep, reason) where reason describes why the model was rejected
    (empty string when kept).
    """
    failure_constant = "failed baseline keep: "
    if model.library_name in NON_NATIVE_ID_SUBSTRINGS:
        return False, failure_constant + "non-native library (ONNX/GGUF/MLX)"
    model_id_lower: str = model.id.lower()
    if any(sub in model_id_lower for sub in NON_NATIVE_ID_SUBSTRINGS):
        return False, failure_constant + "non-native format in model id (ONNX/GGUF/MLX)"
    if "nsfw" in tags(model):
        return False, failure_constant + "NSFW tag"
    if not model.config:
        return False, failure_constant + "no config"
    return True, ""


def _guarded(fn: Callable[[], bool], name: str) -> tuple[bool, str]:
    """Return model-level failure details; propagate exhausted Hub retries."""
    try:
        return fn(), ""
    except HubRetryExhaustedError:
        raise
    except Exception:
        return False, f"exception during {name}"


def contains_remote_code(model: ModelInfo) -> bool:
    """Check whether config loading fails with remote code disabled."""
    try:
        _retry_hub_call(
            lambda: AutoConfig.from_pretrained(model.id, trust_remote_code=False),
            description=f"config[{model.id}]",
        )
        return False
    except (ValueError, OSError):
        return True


# Filenames transformers' AutoModel.from_pretrained recognizes as native
# weights (single-file or sharded via the matching index.json).
_NATIVE_WEIGHT_FILES: frozenset[str] = frozenset(
    {
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }
)


def has_loadable_weights(model: ModelInfo) -> bool:
    """True if the repo ships weights AutoModel.from_pretrained can consume.

    Detects three unloadable classes from the sibling metadata included in the
    bulk ``list_models`` response, without issuing a request per repository:

    * adapter-only repos (LoRA/PEFT, `adapter_config.json` but no full model),
    * GGUF/MLX/ONNX-only repos that slipped past the id-substring filter,
    * abandoned uploads with a config but no weight files at all.
    """
    lower_files: set[str] = {
        sibling.rfilename.lower() for sibling in (model.siblings or [])
    }

    # Adapter-only repos ship adapter_config.json + adapter_model.safetensors
    # and expect PeftModel.from_pretrained(base, ...), not AutoModel directly.
    if "adapter_config.json" in lower_files:
        return False

    return any(name in lower_files for name in _NATIVE_WEIGHT_FILES)


def format_number_to_billions_smart(num: int | float) -> str:
    """Smart formatting that adjusts precision based on magnitude."""
    billions: float = num / 1_000_000_000

    if billions >= 10:
        # For numbers >= 10B, round to nearest integer
        result: int | float = round(billions)
        return f"{result}B"
    elif billions >= 1:
        # For 1B-10B, show 1 decimal place
        result = round(billions, 1)
        return f"{result}B" if result != int(result) else f"{int(result)}B"
    else:
        # For < 1B, show 1-2 decimal places
        result = round(billions, 2)
        return f"{result}B"


def parse_number_suffix(value: str) -> int:
    value = value.strip().upper()

    multipliers: dict[str, int] = {
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "T": 1_000_000_000_000,
    }

    suffix: str = value[-1]

    if suffix in multipliers:
        number: float = float(value[:-1])
        return int(number * multipliers[suffix])

    # No suffix → return as integer
    return int(float(value))


def extract_model_size_from_model_name(
    model_name: str, allow_millions: bool = False
) -> str | None:
    """Pull a parameter-size token (e.g. "7B", "33M") out of a model id.

    ``allow_millions`` also matches an ``M`` suffix — useful for embedding
    models, which are frequently sized in the tens/hundreds of millions.
    Returns the token only if exactly one match is found (avoids ambiguity).
    """
    units: str = "MBmb" if allow_millions else "Bb"
    pattern: str = rf"\b\d+(?:\.\d+)?[{units}]\b"
    matches: list[str] = re.findall(pattern, model_name)
    return matches[0] if len(matches) == 1 else None


def get_param_count(model: ModelInfo) -> int | None:
    if model.safetensors and model.safetensors.parameters:
        return sum(model.safetensors.parameters.values())
    return None


def get_config_type(model_id: str, token: str | bool) -> str | None:
    try:
        model_config: PretrainedConfig = _retry_hub_call(
            lambda: AutoConfig.from_pretrained(
                model_id, token=token, trust_remote_code=False
            ),
            description=f"config[{model_id}]",
        )
        return type(model_config).__name__
    except HubRetryExhaustedError:
        raise
    except Exception:
        return None


# Columns written for every catalog, in order. Each entry is
# (header, value_fn) where value_fn(model, config_class) -> cell value. The
# rank/config-class/param columns are interleaved by build_catalog because they
# depend on per-row computed state.
def _resolve_param_columns(
    model: ModelInfo, allow_millions: bool
) -> tuple[str | None, int | None]:
    """Return (param_str, param_int) for a model, name first then safetensors."""
    param_str: str | None = extract_model_size_from_model_name(model.id, allow_millions)
    param_int: int | None
    if param_str is None:
        param_int = get_param_count(model)
        if param_int is not None:
            param_str = format_number_to_billions_smart(param_int)
    else:
        param_int = parse_number_suffix(param_str)
    return param_str, param_int


def build_catalog(
    *,
    fetch_fn: Callable[[int], Iterable[ModelInfo]],
    filter_fn: Callable[[ModelInfo], tuple[bool, str]],
    limit: int,
    output_csv: Path | str | None,
    label: str,
    extra_columns: (
        list[tuple[str, Callable[[ModelInfo, str | None], object]]] | None
    ) = None,
    allow_millions: bool = False,
    token: str | bool,
) -> list[dict[str, object]]:
    """Fetch → filter → enrich → return a ranked model catalog as a list of dicts.

    Args:
        fetch_fn: callable(limit) -> list of raw model objects (over-fetched).
        filter_fn: callable(model) -> (keep, reason), keep the model if keep is True.
        limit: number of rows to write after filtering.
        output_csv: destination path, or None to skip writing.
        label: human-readable noun for log lines (e.g. "generative").
        extra_columns: optional list of (header, value_fn) where
            value_fn(model, config_class) -> cell. Inserted before config_class.
        allow_millions: pass-through to the size-name extractor.
        token: HF token (or True) for AutoConfig downloads.

    Returns:
        List of dicts, one per model, keyed by column name.
    """
    extra_columns = extra_columns or []

    def _safe_filter(model: ModelInfo) -> tuple[bool, str]:
        try:
            return filter_fn(model)
        except HubRetryExhaustedError:
            raise
        except Exception:
            return False, "_safe_filter raised an exception"

    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    max_workers: int = catalog_max_workers()
    print(f"Using {max_workers} worker thread(s) for per-model Hub calls.")

    t0 = time.perf_counter()
    candidates: list[ModelInfo] = list(fetch_fn(limit))
    timings["fetch (HF list_models)"] = time.perf_counter() - t0
    print(f"Retrieved {len(candidates)} raw {label} candidates.")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        try:
            keep_flags: list[tuple[bool, str]] = list(
                tqdm(
                    ex.map(_safe_filter, candidates),
                    total=len(candidates),
                    desc="Filtering candidates",
                    miniters=1000,
                    mininterval=10,
                )
            )
        except HubRetryExhaustedError:
            # Do not spend another retry budget on every queued candidate.
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    models: list[ModelInfo] = [
        m for m, (keep, _) in zip(candidates, keep_flags) if keep
    ]
    timings["filter (filter_fn)"] = time.perf_counter() - t0
    print(f"Kept {len(models)} {label} models after filtering.")

    reason_counts: dict[str, int] = {}
    for keep, reason in keep_flags:
        if not keep:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        total_filtered_out: int = sum(reason_counts.values())
        print(f"Filtered out {total_filtered_out} {label} models by reason:")
        count_width: int = len(f"{limit:,}")
        for reason, count in sorted(
            reason_counts.items(), key=lambda kv: kv[1], reverse=True
        ):
            print(f"\t{count:>{count_width},} - {reason}")

    models = models[:limit]

    base_head: list[str] = [
        "rank",
        "model_id",
        "downloads",
        "likes",
        "model_type",
        "architectures",
        "parameters (str)",
        "parameters",
        "library",
    ]
    extra_head: list[str] = [h for h, _ in extra_columns]
    tail_head: list[str] = ["is_custom_code", "config_class", "Year"]
    header: list[str] = base_head + extra_head + tail_head

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        try:
            config_classes: list[str | None] = list(
                tqdm(
                    ex.map(lambda m: get_config_type(m.id, token), models),
                    total=len(models),
                    desc="Fetching config classes",
                    miniters=1000,
                    mininterval=10,
                )
            )
        except HubRetryExhaustedError:
            ex.shutdown(wait=False, cancel_futures=True)
            raise
    timings["config classes (AutoConfig)"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rows: list[dict[str, object]] = []
    for rank, (m, config_class) in enumerate(zip(models, config_classes), start=1):
        architectures: list[str] | None = (m.config or {}).get("architectures")
        arch_str: str | None = ";".join(architectures) if architectures else None
        param_str, param_int = _resolve_param_columns(m, allow_millions)
        extra_vals: list[object] = [fn(m, config_class) for _, fn in extra_columns]
        rows.append(
            dict(
                zip(
                    header,
                    [
                        rank,
                        m.id,
                        m.downloads,
                        m.likes,
                        (m.config or {}).get("model_type"),
                        arch_str,
                        param_str,
                        param_int,
                        m.library_name,
                        *extra_vals,
                        is_custom_code(m),
                        config_class,
                        m.created_at.year if m.created_at else None,
                    ],
                )
            )
        )

    timings["build rows"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if output_csv is not None:
        print(f"Writing top {len(rows)} to {output_csv}")
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    timings["write CSV"] = time.perf_counter() - t0

    # Attach the source ModelInfo to each row AFTER the CSV write. It is a
    # runtime-only field (not serializable, and never part of the schema),
    # useful for callers that need metadata the row dict does not expose —
    # e.g. safetensors.parameters, gated, sha, siblings.
    #
    # is_moe is precomputed here (a pure function of data already fetched —
    # tags, config.model_type, config.architectures) so callers that need it
    # don't have to carry the non-serializable ModelInfo object forward.
    t0 = time.perf_counter()
    for row, m in zip(rows, models):
        row["model_info"] = m
        row["is_moe"] = _is_moe(m)
    timings["attach model_info / is_moe"] = time.perf_counter() - t0

    timings["other"] = (time.perf_counter() - t_total) - sum(timings.values())

    total = sum(timings.values())
    print(f"\nTiming breakdown for {label} catalog ({total:.1f}s total):")
    width = max(len(name) for name in timings)
    for name, secs in timings.items():
        share = secs / total if total else 0.0
        print(f"  {name:<{width}}  {secs:7.2f}s  {share:5.1%}")

    return rows
