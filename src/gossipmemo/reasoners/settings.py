"""Server configuration the reasoner prompts need.

This is the counterpart to transport configuration: `LlmTransport` decides
how a request is shaped and sent, while `ReasoningSettings` carries what
the prompts themselves say. Factories take one explicitly, so every
reasoner reads the same values without reaching past the transport seam.

`prompts` carries the static prompt text, so a deployment can replace any
fragment without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..prompts import PromptLibrary

# Module-level singleton: a mutable default would be rebuilt per instance, and
# every reasoner that takes the shipped wording can share one immutable value.
DEFAULT_PROMPTS = PromptLibrary()


@dataclass(frozen=True, slots=True)
class ReasoningSettings:
    """Prompt-facing server configuration handed to every reasoner factory."""

    user_name: str = "CurrentUser"
    prompts: PromptLibrary = DEFAULT_PROMPTS
    # Observation-only probe: ask extraction to also report what it could not
    # safely interpret, and log it. Nothing reads the answers back, so this
    # stays off unless a deployment is deliberately collecting that data.
    extraction_clarification_probe: bool = False


__all__ = ["DEFAULT_PROMPTS", "ReasoningSettings"]
