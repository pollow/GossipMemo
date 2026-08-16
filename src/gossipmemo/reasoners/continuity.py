"""Rolling continuity reasoner.

Deliberately not part of `DEFAULT_REASONING_PIPELINE`: continuity keeps its
own message-count trigger (`SocialMemoryWorld._schedule_continuity_reason`)
instead of running on the daily/startup induction sweep.
"""

from __future__ import annotations

import logging

from ..llm import LlmModel
from ..queue import ReasonerCallQueue
from ..store import WorldStore
from .base import DescriptorReasoner

logger = logging.getLogger(__name__)


def build_continuity_reasoner(store: WorldStore, model: LlmModel, queue: ReasonerCallQueue) -> DescriptorReasoner:
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

    return DescriptorReasoner("continuity", queue, load_context, call, apply, continue_when)


__all__ = ["build_continuity_reasoner"]
