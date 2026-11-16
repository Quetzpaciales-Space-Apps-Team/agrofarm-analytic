"""Manifest validation against `manifest.schema.json` (packaged data)."""

import json
from collections.abc import Mapping
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

SCHEMA_FILE = "manifest.schema.json"


class ManifestError(ValueError):
    """The manifest does not follow the schema. `errors` has one `path: message` per problem."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("\n".join(errors))
        self.errors = errors


@cache
def load_schema() -> dict[str, Any]:
    text = resources.files("agrofarm_data").joinpath(SCHEMA_FILE).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def _path(error: ValidationError) -> str:
    return ".".join(str(part) for part in error.absolute_path) or "(manifest)"


def _message(error: ValidationError, parent: ValidationError | None = None) -> str:
    # `anyOf` alone says "not valid under any of the given schemas": list the alternatives.
    if error.validator == "anyOf" and error.context:
        return "needs one of: " + "; or ".join(_message(sub, error) for sub in error.context)
    if parent is not None and error.absolute_path != parent.absolute_path:
        return f"{_path(error)}: {error.message}"
    return error.message


def _load(source: Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError([f"(manifest): {source} cannot be read: {exc.strerror}"]) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ManifestError([f"(manifest): {source} is not valid JSON: {exc}"]) from exc
    if not isinstance(data, dict):
        raise ManifestError([f"(manifest): {source} must be a JSON object"])
    return data


def validate_manifest(source: Path | Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest as a dict, or raise `ManifestError` with readable messages."""
    manifest = dict(_load(source))
    validator = Draft202012Validator(load_schema(), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(manifest), key=lambda e: (_path(e), e.message))
    if errors:
        raise ManifestError([f"{_path(e)}: {_message(e)}" for e in errors])
    return manifest
