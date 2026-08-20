"""Server configuration the reasoner prompts need.

This is the counterpart to transport configuration: `LlmTransport` decides
how a request is shaped and sent, while `ReasoningSettings` carries what
the prompts themselves say. Factories take one explicitly, so every
reasoner reads the same values without reaching past the transport seam.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningSettings:
    """Prompt-facing server configuration handed to every reasoner factory."""

    user_name: str = "CurrentUser"


__all__ = ["ReasoningSettings"]
