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

"""Shared repository layout discovery for Hugging Face model loaders."""

import json
import os

from transformers import AutoConfig, PretrainedConfig
from transformers.utils.hub import cached_file


def model_subfolder(
    model_path: str | os.PathLike[str], *, token: str | bool | None = None
) -> str:
    """Find the HF module declared by SentenceTransformers, locally or on the Hub.

    A recognized HF module's path takes precedence over any configuration at
    the root. Missing metadata or other module types fall back to the root,
    preserving native HF loading for repositories such as sparse encoders.
    Configurations, weights, and processors must all use this same location.
    """
    modules_file = cached_file(
        os.fspath(model_path),
        "modules.json",
        token=token,
        _raise_exceptions_for_missing_entries=False,
    )
    if modules_file is None:
        return ""
    with open(modules_file, encoding="utf-8") as handle:
        modules: object = json.load(handle)
    if not isinstance(modules, list):
        raise ValueError(f"{model_path}: modules.json must contain a list")
    for module in modules:
        if not isinstance(module, dict):
            raise ValueError(f"{model_path}: modules.json entries must be objects")
        module_type: object = module.get("type")
        if isinstance(module_type, str) and module_type.rsplit(".", 1)[-1] in {
            "CLIPModel",
            "Transformer",
        }:
            subfolder: object = module.get("path")
            if not isinstance(subfolder, str):
                raise ValueError(f"{model_path}: HF module path must be a string")
            return subfolder
    return ""


def load_model_config(
    model_path: str | os.PathLike[str],
    *,
    token: str | bool | None = None,
    trust_remote_code: bool | None = None,
    subfolder: str | None = None,
) -> PretrainedConfig:
    """Load the configuration from the same repository location as the weights.

    ``subfolder=None`` discovers the location through ``model_subfolder``.
    Callers that already know the layout, such as catalog filtering with Hub
    sibling metadata, pass it explicitly to skip the ``modules.json`` lookup.
    Errors propagate so callers can apply their own retry and error policies.
    """
    resolved_subfolder: str = (
        model_subfolder(model_path, token=token) if subfolder is None else subfolder
    )
    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_path,
        subfolder=resolved_subfolder,
        token=token,
        trust_remote_code=trust_remote_code,
    )
    return config
