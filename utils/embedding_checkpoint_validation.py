"""Inspect embedding checkpoints without downloading their tensor payloads."""

import json
from collections.abc import Mapping
from typing import BinaryIO

import torch
from huggingface_hub import HfApi, HfFileSystem, ModelInfo
from huggingface_hub.hf_api import SafetensorsFileMetadata
from torch._subclasses.fake_tensor import FakeTensorMode

from utils.hf_model_catalog import HubRetryExhaustedError, _retry_hub_call

# Bound reads of repository JSON files; weights are accessed through range reads.
_MAX_JSON_BYTES: int = 10_000_000


def _read_json(fs: HfFileSystem, path: str) -> object:
    with fs.open(path, "rb", block_size=64 * 1024) as file:
        payload: bytes = file.read(_MAX_JSON_BYTES + 1)
    if len(payload) > _MAX_JSON_BYTES:
        raise ValueError("repository JSON exceeds the inspection size limit")
    return json.loads(payload)


def _checkpoint_files(
    fs: HfFileSystem, root: str, filenames: set[str]
) -> tuple[list[str], dict[str, str]]:
    # Match Transformers' preference for safetensors over pickle checkpoints.
    for filename in (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        if filename not in filenames:
            continue
        if not filename.endswith(".index.json"):
            return [filename], {}
        index: object = _read_json(fs, f"{root}/{filename}")
        if not isinstance(index, dict):
            raise ValueError("checkpoint index must be an object")
        weight_map: object = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
        expected_suffix: str = (
            ".safetensors" if filename.startswith("model.") else ".bin"
        )
        validated_map: dict[str, str] = {}
        for name, shard in weight_map.items():
            if not isinstance(name, str) or not isinstance(shard, str):
                raise ValueError("checkpoint index entries must be strings")
            if shard not in filenames:
                raise ValueError(f"missing checkpoint shard: {shard}")
            if not shard.endswith(expected_suffix):
                raise ValueError(f"unexpected checkpoint shard format: {shard}")
            validated_map[name] = shard
        return sorted(set(validated_map.values())), validated_map
    raise ValueError("no loadable weights")


def _safetensors_keys(
    api: HfApi, fs: HfFileSystem, model: ModelInfo, root: str, filename: str
) -> set[str]:
    metadata: SafetensorsFileMetadata = api.parse_safetensors_file_metadata(
        model.id, filename, revision=model.sha, token=api.token
    )
    # The Hub parser checks the header length/JSON. Also check offsets against
    # the actual file length so truncated uploads cannot pass a header probe.
    with fs.open(f"{root}/{filename}", "rb", block_size=64 * 1024) as file:
        prefix: bytes = file.read(8)
        file.seek(0, 2)
        file_size: int = file.tell()
    header_size: int = int.from_bytes(prefix, "little")
    data_size: int = file_size - 8 - header_size
    offset = 0
    for tensor in sorted(metadata.tensors.values(), key=lambda t: t.data_offsets):
        begin, end = tensor.data_offsets
        if begin != offset or end < begin or end > data_size:
            raise ValueError(f"invalid tensor offsets in {filename}")
        offset = end
    if offset != data_size or not metadata.tensors:
        raise ValueError(f"incomplete safetensors checkpoint: {filename}")
    return set(metadata.tensors)


def _pytorch_keys(file: BinaryIO) -> set[str]:
    # FakeTensorMode skips storage reads, including on older supported PyTorch
    # versions. The real weights-only unpickler still rejects unsupported
    # protocols/globals; never enable arbitrary pickle execution for discovery.
    with FakeTensorMode():
        state_dict: object = torch.load(file, map_location="meta", weights_only=True)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("checkpoint does not contain a state dict")
    if any(
        not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
        for name, tensor in state_dict.items()
    ):
        raise ValueError("checkpoint contains non-tensor state dict entries")
    return set(state_dict)


def _inspect_checkpoint(model: ModelInfo, token: str | bool) -> tuple[bool, str]:
    filenames: set[str] = {s.rfilename for s in (model.siblings or [])}
    root: str = f"{model.id}@{model.sha or 'main'}"
    fs = HfFileSystem(token=token, skip_instance_cache=True)
    if "modules.json" in filenames:
        modules: object = _read_json(fs, f"{root}/modules.json")
        if not isinstance(modules, list):
            raise ValueError("sentence-transformers modules.json must be a list")
        for module in modules:
            if not isinstance(module, dict) or not isinstance(module.get("type"), str):
                raise ValueError("invalid sentence-transformers module")
            if module["type"].startswith("sentence_transformers.sparse_encoder."):
                return False, "model is a sparse encoder"

    checkpoints, weight_map = _checkpoint_files(fs, root, filenames)
    api = HfApi(token=token)
    for filename in checkpoints:
        keys: set[str]
        if filename.endswith(".safetensors"):
            keys = _safetensors_keys(api, fs, model, root, filename)
        else:
            with fs.open(f"{root}/{filename}", "rb", block_size=64 * 1024) as file:
                keys = _pytorch_keys(file)
        if any(
            name not in keys for name, shard in weight_map.items() if shard == filename
        ):
            raise ValueError(f"checkpoint shard is missing indexed tensors: {filename}")
    return True, ""


def validate_embedding_checkpoint(
    model: ModelInfo, token: str | bool
) -> tuple[bool, str]:
    """Reject broken/sparse ranked candidates; propagate transient Hub outages.

    Only serialization metadata is inspected, not tensor values or model
    execution. Curated fetches deliberately bypass ranked discovery gates.
    """
    try:
        return _retry_hub_call(
            lambda: _inspect_checkpoint(model, token),
            description=f"embedding checkpoint[{model.id}]",
            retry_os_errors=True,
        )
    except HubRetryExhaustedError:
        raise
    except Exception as error:
        detail: str = str(error).splitlines()[0] if str(error) else type(error).__name__
        return False, f"invalid embedding checkpoint: {detail}"
