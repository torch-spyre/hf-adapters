"""Offline checkpoint-discovery regressions; run with --noconftest."""

import io
import json
from unittest.mock import Mock

import httpx
import pytest
import torch
from huggingface_hub import HfFileSystem, ModelInfo
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import save
from typing_extensions import Buffer

from utils import embedding_checkpoint_validation as validation
from utils import fetch_top_embedding_models as fetcher
from utils import hf_model_catalog as catalog


class _TrackedFile(io.BytesIO):
    def __init__(self: "_TrackedFile", payload: bytes, reads: list[int]) -> None:
        super().__init__(payload)
        self.reads = reads

    def read(self: "_TrackedFile", size: int | None = -1) -> bytes:
        payload: bytes = super().read(size)
        self.reads.append(len(payload))
        return payload

    def readinto(self: "_TrackedFile", buffer: Buffer) -> int:
        count: int = super().readinto(buffer)
        self.reads.append(count)
        return count


@pytest.fixture
def repository(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    fs = Mock(spec=HfFileSystem)

    def open_file(path: str, mode: str, **kwargs: object) -> io.BytesIO:
        assert path.startswith("org/model@revision/")
        filename: str = path.removeprefix("org/model@revision/")
        if filename not in files:
            raise FileNotFoundError(filename)
        return io.BytesIO(files[filename])

    def send(
        self: httpx.Client, request: httpx.Request, **kwargs: object
    ) -> httpx.Response:
        assert request.url.host == "huggingface.co"
        assert "/resolve/revision/" in request.url.path
        assert "range" in request.headers
        filename: str = request.url.path.split("/resolve/revision/", 1)[1]
        payload: bytes = files[filename]
        start, end = request.headers["range"].removeprefix("bytes=").split("-")
        return httpx.Response(
            206, content=payload[int(start) : int(end) + 1], request=request
        )

    fs.open.side_effect = open_file
    monkeypatch.setattr(validation, "HfFileSystem", Mock(return_value=fs))
    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(fetcher, "contains_remote_code", lambda model: False)
    return files


def _model(files: dict[str, bytes], tags: list[str] | None = None) -> ModelInfo:
    return ModelInfo(
        id="org/model",
        sha="revision",
        config={"model_type": "bert"},
        library_name="sentence-transformers",
        tags=tags or [],
        siblings=[{"rfilename": filename} for filename in files],
    )


def _torch_checkpoint(
    state_dict: object, protocol: int = 2, legacy: bool = False
) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        state_dict,
        buffer,
        pickle_protocol=protocol,
        _use_new_zipfile_serialization=not legacy,
    )
    return buffer.getvalue()


def test_missing_weights_are_filtered(repository: dict[str, bytes]) -> None:
    repository["config.json"] = b"{}"
    assert fetcher.keep(_model(repository), False) == (False, "no loadable weights")


@pytest.mark.parametrize("suffix", ["safetensors", "bin"])
def test_missing_shards_are_filtered(repository: dict[str, bytes], suffix: str) -> None:
    prefix: str = "model" if suffix == "safetensors" else "pytorch_model"
    repository[f"{prefix}.{suffix}.index.json"] = json.dumps(
        {"weight_map": {"weight": f"missing.{suffix}"}}
    ).encode()
    kept, reason = fetcher.keep(_model(repository), False)
    assert not kept
    assert f"missing checkpoint shard: missing.{suffix}" in reason


@pytest.mark.parametrize("weight_map", [None, {}, [], {"weight": 42}])
def test_malformed_index_is_filtered(
    repository: dict[str, bytes], weight_map: object
) -> None:
    repository["model.safetensors.index.json"] = json.dumps(
        {"weight_map": weight_map}
    ).encode()
    assert fetcher.keep(_model(repository), False)[0] is False


@pytest.mark.parametrize("sharded", [False, True])
def test_valid_safetensors_are_kept(
    repository: dict[str, bytes], sharded: bool
) -> None:
    filename: str = "shard.safetensors" if sharded else "model.safetensors"
    repository[filename] = save({"weight": torch.ones(2, 3)})
    if sharded:
        repository["model.safetensors.index.json"] = json.dumps(
            {"weight_map": {"weight": filename}}
        ).encode()
    assert fetcher.keep(_model(repository), False) == (True, "")


@pytest.mark.parametrize(
    "payload",
    [
        b"version https://git-lfs.github.com/spec/v1\n",
        (100_000_001).to_bytes(8, "little"),
        (8).to_bytes(8, "little") + b"not json",
        save({"weight": torch.ones(3)})[:-1],
    ],
)
def test_bad_safetensors_are_filtered(
    repository: dict[str, bytes], payload: bytes
) -> None:
    repository["model.safetensors"] = payload
    kept, reason = fetcher.keep(_model(repository), False)
    assert not kept
    assert reason.startswith("invalid embedding checkpoint:")


def test_indexed_tensor_must_exist_in_its_shard(repository: dict[str, bytes]) -> None:
    repository["shard.safetensors"] = save({"other": torch.ones(2)})
    repository["model.safetensors.index.json"] = json.dumps(
        {"weight_map": {"missing": "shard.safetensors"}}
    ).encode()
    kept, reason = fetcher.keep(_model(repository), False)
    assert not kept
    assert "missing indexed tensors" in reason


@pytest.mark.parametrize("legacy", [False, True])
def test_pytorch_validation_skips_tensor_payloads(legacy: bool) -> None:
    payload: bytes = _torch_checkpoint(
        {"weight": torch.ones(1024, 1024)}, legacy=legacy
    )
    reads: list[int] = []
    with _TrackedFile(payload, reads) as file:
        assert validation._pytorch_keys(file) == {"weight"}
    assert sum(reads) < len(payload) // 10


def test_valid_pytorch_checkpoint_is_kept(repository: dict[str, bytes]) -> None:
    repository["pytorch_model.bin"] = _torch_checkpoint({"weight": torch.ones(2)})
    assert fetcher.keep(_model(repository), False) == (True, "")


@pytest.mark.parametrize(
    "payload",
    [
        _torch_checkpoint({"weight": torch.ones(2)})[:-25],
        _torch_checkpoint({"weight": torch.ones(2)}, protocol=4),
        _torch_checkpoint(
            {"weight": torch.ones(2), "config": Exception("custom global")}
        ),
        b"version https://git-lfs.github.com/spec/v1\n",
    ],
)
def test_bad_pytorch_checkpoints_are_filtered(
    repository: dict[str, bytes], payload: bytes
) -> None:
    repository["pytorch_model.bin"] = payload
    assert fetcher.keep(_model(repository), False)[0] is False


def test_safetensors_take_precedence_over_bad_pickle(
    repository: dict[str, bytes],
) -> None:
    repository["model.safetensors"] = save({"weight": torch.ones(2)})
    repository["pytorch_model.bin"] = b"broken"
    assert fetcher.keep(_model(repository), False) == (True, "")


@pytest.mark.parametrize("tag", ["sparse-encoder", "splade"])
def test_sparse_tags_are_filtered(repository: dict[str, bytes], tag: str) -> None:
    repository["model.safetensors"] = b"not inspected"
    assert fetcher.keep(_model(repository, [tag]), False) == (
        False,
        "model is a sparse encoder",
    )


@pytest.mark.parametrize(
    "tag",
    [
        "sparse",
        "sparse-pruned",
        "dataset:org/sparse-training-data",
        "dataset:org/splade-distillation",
        "base_model:org/sparse-encoder",
        "base_model:finetune:org/splade",
        "license:sparse-license",
    ],
)
def test_dense_models_with_sparse_metadata_tags_are_kept(
    repository: dict[str, bytes], tag: str
) -> None:
    repository["model.safetensors"] = save({"weight": torch.ones(2)})
    assert fetcher.keep(_model(repository, [tag]), False) == (True, "")


@pytest.mark.parametrize("sparse", [False, True])
def test_sentence_transformer_module_types_are_checked(
    repository: dict[str, bytes], sparse: bool
) -> None:
    module_type: str = (
        "sentence_transformers.sparse_encoder.models.MLMTransformer"
        if sparse
        else "sentence_transformers.models.Transformer"
    )
    repository["modules.json"] = json.dumps([{"type": module_type}]).encode()
    repository["model.safetensors"] = save({"weight": torch.ones(2)})
    assert fetcher.keep(_model(repository), False) == (
        (False, "model is a sparse encoder") if sparse else (True, "")
    )


@pytest.mark.parametrize("exhausted", [False, True])
def test_checkpoint_hub_outages_are_retried_not_filtered(
    monkeypatch: pytest.MonkeyPatch, exhausted: bool
) -> None:
    response = httpx.Response(
        503, request=httpx.Request("GET", "https://huggingface.co/weights")
    )
    error = HfHubHTTPError("temporary outage", response=response)
    request = Mock(side_effect=error if exhausted else [error, (True, "")])
    monkeypatch.setattr(validation, "_inspect_checkpoint", request)
    monkeypatch.setattr(catalog.time, "sleep", lambda seconds: None)
    if exhausted:
        with pytest.raises(catalog.HubRetryExhaustedError):
            validation.validate_embedding_checkpoint(_model({}), False)
        assert request.call_count == 5
    else:
        assert validation.validate_embedding_checkpoint(_model({}), False) == (True, "")
        assert request.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectError("DNS lookup failed"),
        httpx.ReadError("connection reset"),
        OSError("remote filesystem read failed"),
    ],
)
@pytest.mark.parametrize("exhausted", [False, True])
def test_checkpoint_transport_failures_recover_or_abort_catalog(
    repository: dict[str, bytes],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    exhausted: bool,
) -> None:
    repository["model.safetensors"] = save({"weight": torch.ones(2)})
    repository["modules.json"] = b"[]"
    read_json = Mock(side_effect=error if exhausted else [error, []])
    monkeypatch.setattr(validation, "_read_json", read_json)
    delays: list[float] = []
    monkeypatch.setattr(catalog.time, "sleep", delays.append)
    monkeypatch.setattr(
        catalog, "get_config_type", lambda model_id, token: "BertConfig"
    )

    def build_catalog() -> list[dict[str, object]]:
        return catalog.build_catalog(
            fetch_fn=lambda limit: [_model(repository)],
            filter_fn=lambda model: fetcher.keep(model, False),
            limit=1,
            output_csv=None,
            label="embedding",
            token=False,
        )

    if exhausted:
        with pytest.raises(
            catalog.HubRetryExhaustedError, match="after 5 attempts"
        ) as caught:
            build_catalog()
        assert caught.value.__cause__ is error
        assert read_json.call_count == 5
        assert delays == [2.0, 4.0, 8.0, 16.0]
    else:
        assert [row["model_id"] for row in build_catalog()] == ["org/model"]
        assert read_json.call_count == 2
        assert delays == [2.0]


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing shard"), PermissionError("access denied")]
)
def test_permanent_checkpoint_io_failures_are_filtered_without_retry(
    repository: dict[str, bytes], monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    repository["modules.json"] = b"[]"
    repository["model.safetensors"] = save({"weight": torch.ones(2)})
    read_json = Mock(side_effect=error)
    monkeypatch.setattr(validation, "_read_json", read_json)
    kept, reason = fetcher.keep(_model(repository), False)
    assert not kept
    assert str(error) in reason
    assert read_json.call_count == 1


def test_ranked_catalog_refills_after_checkpoint_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models: list[ModelInfo] = [_model({"model.safetensors": b""}) for _ in range(2)]
    models[0].id = "org/broken"
    models[1].id = "org/valid"
    monkeypatch.setattr(fetcher, "contains_remote_code", lambda model: False)
    monkeypatch.setattr(
        catalog, "get_config_type", lambda model_id, token: "BertConfig"
    )

    def validate(model: ModelInfo, token: str | bool) -> tuple[bool, str]:
        return (False, "invalid checkpoint") if model.id == "org/broken" else (True, "")

    monkeypatch.setattr(fetcher, "validate_embedding_checkpoint", validate)
    rows: list[dict[str, object]] = catalog.build_catalog(
        fetch_fn=lambda limit: models,
        filter_fn=lambda model: fetcher.keep(model, False),
        limit=1,
        output_csv=None,
        label="embedding",
        token=False,
    )
    assert [row["model_id"] for row in rows] == ["org/valid"]
