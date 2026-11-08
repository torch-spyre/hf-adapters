# Copyright 2025 The Torch-Spyre Authors.
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


try:
    from hf_adapters._version import __version__
except ImportError:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("hf-adapters-spyre")
    except PackageNotFoundError:
        __version__ = "0.0.0"

__all__ = [
    "__version__",
    "AutoSpyreModel",
    "AutoSpyreModelForCausalLM",
    "AutoSpyreModelForImageTextToText",
    "AutoSpyreModelForMaskedLM",
    "AutoSpyreModelForQuestionAnswering",
    "AutoSpyreModelForSequenceClassification",
    "AutoSpyreModelForTokenClassification",
    "encode_prompts",
]


def __getattr__(name):
    # Keep package initialization independent of hf_common and model-family imports.
    if name in __all__ and name.startswith("AutoSpyreModel"):
        from hf_adapters import auto_spyre_model

        value = getattr(auto_spyre_model, name)
    elif name == "encode_prompts":
        from hf_adapters.hf_common import encode_prompts

        value = encode_prompts
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
