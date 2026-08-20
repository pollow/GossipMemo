"""UserModel profile-card reasoner and its prompt."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

from ..models import (
    HypothesisView,
    MemoryView,
    PersonProjectionResult,
    UserModelReasoningResult,
    UserModelView,
    UserReasoningActionsResult,
)
from ..store import WorldStore
from .base import DescriptorReasoner
from .owner import owner_reasoning
from .settings import ReasoningSettings

if TYPE_CHECKING:
    from ..llm import LlmTransport


async def _reason_user_model(
    transport: LlmTransport, settings: ReasoningSettings, memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
) -> UserModelReasoningResult:
    projection, actions = await owner_reasoning(
        transport, settings, settings.prompts.user_model_reasoning_system,
        UserModelView(space_id="current"),
        memories, inferred_memories, hypotheses, PersonProjectionResult,
        UserReasoningActionsResult,
    )
    return UserModelReasoningResult(
        profile_card=projection.profile_card, hypothesis_actions=actions.hypothesis_actions,
    )


def build_user_model_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings
) -> DescriptorReasoner:
    reason_user_model = partial(_reason_user_model, model, settings)

    def load_context(space_id: str):
        _, _, user_models = store.stale_entities()
        if space_id not in user_models:
            return None
        context = store.user_model_context(space_id)
        if not context:
            return None
        user_model, memories, watermark = context
        if not user_model.stale:
            return None
        evidence = [memory for memory in memories if memory.basis != "inferred"]
        inferred, hypotheses = store.owner_review_context(space_id, "user", None)
        return watermark, evidence, inferred, hypotheses

    def call(space_id: str, context):
        _, evidence, inferred, hypotheses = context
        return "reason-user-model", reason_user_model, (evidence, inferred, hypotheses)

    def apply(space_id: str, context, result) -> bool:
        watermark, _, _, hypotheses = context
        return store.apply_user_model_reasoning(
            space_id, watermark, result, {hypothesis.id for hypothesis in hypotheses}
        )

    return DescriptorReasoner("user_model", load_context, call, apply)


__all__ = ["build_user_model_reasoner"]
