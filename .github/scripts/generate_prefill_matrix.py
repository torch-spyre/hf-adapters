#!/usr/bin/env python3
"""Read nightly.models from perf.yaml and emit a JSON model list for Actions."""

import argparse
import json
from pathlib import Path

import yaml

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "tests" / "model_lists" / "perf.yaml"
)


def read_nightly_models(path: Path) -> list[str]:
    """Reject invalid lists instead of silently skipping every benchmark job."""
    data: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    nightly: object = data.get("nightly")
    if not isinstance(nightly, dict):
        raise ValueError(f"{path}: nightly must be a mapping")
    raw_models: object = nightly.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"{path}: nightly.models must be a non-empty list")

    models: list[str] = []
    for model in raw_models:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"{path}: nightly.models entries must be non-empty strings"
            )
        if model != model.strip():
            raise ValueError(
                f"{path}: model path has surrounding whitespace: {model!r}"
            )
        if model in models:
            raise ValueError(f"{path}: duplicate nightly model: {model}")
        models.append(model)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        models = read_nightly_models(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(json.dumps(models))


if __name__ == "__main__":
    main()
