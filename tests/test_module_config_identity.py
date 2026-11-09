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

"""The v2 case identity hashes (classname, name, tags), and a module test's name carries
only the module name. Two configs that test a same-named module must therefore differ
in their tags, or their cases collapse into one identity."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

_MODULE_CONFIGS = Path(__file__).resolve().parent / "configs" / "module_tests"


def test_configs_sharing_a_module_differ_in_tags():
    owners = defaultdict(list)
    for path in sorted(_MODULE_CONFIGS.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        for file_entry in doc["test_suite_config"]["files"]:
            for test in file_entry.get("tests", []):
                tags = frozenset(test.get("tags") or [])
                modules = test.get("edits", {}).get("modules", {}).get("include", [])
                for name in test["names"]:
                    for module in modules:
                        key = (file_entry["path"], name, module["name"], tags)
                        owners[key].append(path.name)
    assert owners, f"no module tests found under {_MODULE_CONFIGS}"
    clashes = {k[1:3]: v for k, v in owners.items() if len(v) > 1}
    assert not clashes, f"same test, module and tags in several configs: {clashes}"
