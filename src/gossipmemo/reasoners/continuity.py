"""Rolling continuity reasoner.

Deliberately not part of `DEFAULT_REASONING_PIPELINE`: continuity keeps its
own message-count trigger (`SocialMemoryWorld._schedule_continuity_reason`)
instead of running on the daily/startup induction sweep.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..priority import TIER_FRESHNESS
from ..models import ContinuityView, ModelMessage
from ..prompts import _json
from ..store import WorldStore
from .base import DescriptorReasoner

if TYPE_CHECKING:
    from ..llm import LlmModel

logger = logging.getLogger(__name__)

CONTINUITY_SYSTEM_PROMPT = """Rebuild compact cross-session continuity.
Return only the supplied JSON schema. Keep ongoing threads, recent decisions,
pending actions, and context useful for the next conversation. Do not make
long-term personality inferences or copy person/user profiles; the current user
is not a Person. Use the language that best matches supplied messages and prior
continuity; keep IDs and enum values unchanged.
"""


def continuity_prompt(
    continuity: ContinuityView | None, messages: list[ModelMessage]
) -> str:
    return (
        "Rebuild continuity from the prior summary and newer raw messages. "
        "Choose the last supplied message as through_message_id.\n\nPrior continuity:\n"
        + _json(continuity)
        + "\n\nNew messages:\n"
        + _json(messages)
    )


def build_continuity_reasoner(store: WorldStore, model: LlmModel) -> DescriptorReasoner:
    def load_context(space_id: str):
        context = store.continuity_context(space_id, limit=None)
        if not context:
            return None
        _, messages = context
        if not messages:
            return None
        return context

    def call(space_id: str, context):
        continuity, messages = context
        return "reason-continuity", model.reason_continuity, (continuity, messages)

    def apply(space_id: str, context, result) -> bool:
        continuity, messages = context
        logger.info("continuity_extracted", extra={"space_id": space_id, "message_count": len(messages)})
        expected = continuity.through_message_id if continuity else None
        return store.apply_continuity_reasoning(space_id, expected, result)

    def continue_when(context, result, applied: bool) -> bool:
        # Success means more newer messages may remain uncovered; a conflict
        # (someone else updated continuity concurrently) means give up.
        return applied

    return DescriptorReasoner(
        "continuity", load_context, call, apply, continue_when, tier=TIER_FRESHNESS,
    )


__all__ = ["CONTINUITY_SYSTEM_PROMPT", "build_continuity_reasoner", "continuity_prompt"]
