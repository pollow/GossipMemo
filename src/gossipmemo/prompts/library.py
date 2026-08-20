"""The prompt fragments a deployment may override, as one typed value.

Every static fragment is a field, so a use site reads `prompts.extraction_system`
and a typo is a type error rather than a silently missing prompt. Defaults are
the shipped wording in `prompts.defaults`; `from_file` replaces individual
fields from a flat TOML table and rejects anything it does not recognize --
running on a default because a key was misspelled is exactly the failure this
seam exists to prevent.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..config import ConfigurationError
from . import defaults

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found]


# Read-only views so a frozen library cannot be mutated through its tables --
# and so the dataclass accepts them as defaults at all.
_VIEWPOINTS: Mapping[str, str] = MappingProxyType(defaults.COVERAGE_ROOT_VIEWPOINTS)
_BLIND_SPOTS: Mapping[str, str] = MappingProxyType(defaults.COVERAGE_ROOT_BLIND_SPOTS)


@dataclass(frozen=True, slots=True)
class PromptLibrary:
    """Every static prompt fragment, defaulting to the shipped wording."""

    extraction_system: str = defaults.EXTRACTION_SYSTEM_PROMPT
    person_reasoning_system: str = defaults.PERSON_REASONING_SYSTEM_PROMPT
    relationship_reasoning_system: str = defaults.RELATIONSHIP_REASONING_SYSTEM_PROMPT
    user_model_reasoning_system: str = defaults.USER_MODEL_REASONING_SYSTEM_PROMPT
    continuity_system: str = defaults.CONTINUITY_SYSTEM_PROMPT
    coverage_audit_system: str = defaults.COVERAGE_AUDIT_SYSTEM_PROMPT
    goal_planning_system: str = defaults.GOAL_PLANNING_SYSTEM_PROMPT
    query_synthesis_system: str = defaults.QUERY_SYNTHESIS_SYSTEM_PROMPT
    coverage_method: str = defaults.COVERAGE_METHOD
    coverage_root_viewpoints: Mapping[str, str] = _VIEWPOINTS
    coverage_root_blind_spots: Mapping[str, str] = _BLIND_SPOTS
    projection_stage: str = defaults.PROJECTION_STAGE_PROMPT
    actions_stage: str = defaults.ACTIONS_STAGE_PROMPT

    @classmethod
    def from_file(cls, path: Path) -> PromptLibrary:
        """Build a library from a flat TOML table of `field = "text"` overrides."""

        try:
            with open(path, "rb") as handle:
                document = tomllib.load(handle)
        except OSError as error:
            raise ConfigurationError(f"prompts file cannot be read: {path}") from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigurationError(f"prompts file is not valid TOML: {path}: {error}") from error
        return cls.from_overrides(document)

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any]) -> PromptLibrary:
        """Validate a mapping of field-name overrides and apply it over the defaults."""

        tables = {"coverage_root_viewpoints", "coverage_root_blind_spots"}
        known = {field.name for field in fields(cls)}
        values: dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in known:
                raise ConfigurationError(f"unknown prompt override key: {key}")
            if key in tables:
                values[key] = _merged_table(cls, key, value)
            elif not isinstance(value, str):
                raise ConfigurationError(f"prompt override must be a string: {key}")
            else:
                values[key] = value
        return cls(**values)


def _merged_table(
    cls: type[PromptLibrary], key: str, value: Any
) -> Mapping[str, str]:
    """Merge a per-root override over the shipped table, keeping untouched roots."""

    if not isinstance(value, dict):
        raise ConfigurationError(f"prompt override must be a table of roots: {key}")
    base: Mapping[str, str] = getattr(cls(), key)
    merged = dict(base)
    for root, text in value.items():
        if root not in merged:
            raise ConfigurationError(f"unknown coverage root in {key}: {root}")
        if not isinstance(text, str):
            raise ConfigurationError(f"prompt override must be a string: {key}.{root}")
        merged[root] = text
    return MappingProxyType(merged)


__all__ = ["PromptLibrary"]
