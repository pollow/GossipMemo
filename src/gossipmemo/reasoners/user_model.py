"""UserModel profile-card reasoner."""

from __future__ import annotations

from ..llm import LlmModel
from ..queue import ReasonerCallQueue
from ..store import WorldStore
from .base import DescriptorReasoner


def build_user_model_reasoner(store: WorldStore, model: LlmModel, queue: ReasonerCallQueue) -> DescriptorReasoner:
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
        return "reason-user-model", model.reason_user_model, (evidence, inferred, hypotheses)

    def apply(space_id: str, context, result) -> bool:
        watermark, _, _, hypotheses = context
        return store.apply_user_model_reasoning(
            space_id, watermark, result, {hypothesis.id for hypothesis in hypotheses}
        )

    return DescriptorReasoner("user_model", queue, load_context, call, apply)


__all__ = ["build_user_model_reasoner"]
