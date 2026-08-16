"""UserModel profile-card reasoner and its prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import MemoryView
from ..prompts import _json
from ..queue import ReasonerCallQueue
from ..store import WorldStore
from .base import DescriptorReasoner

if TYPE_CHECKING:
    from ..llm import LlmModel

USER_MODEL_REASONING_SYSTEM_PROMPT = """Reason carefully about the fixed current user from
active memories marked about_user. Capture preferences, communication preferences, goals, current
situations, and practical interaction guidance. Generalize recurring patterns
when supported, but do not turn a one-off event into a stable trait or include
another person's identity. Use valid_from and valid_to to separate current
conditions from historical events. Do not use projections, inferred memories, or
hypotheses as evidence. Use the language that best matches supplied memories; keep IDs and enum values unchanged.
"""


def user_model_reasoning_prompt(
    memories: list[MemoryView], user_name: str = "CurrentUser"
) -> str:
    return (
        f"Rebuild the compact profile card for {user_name} from these active "
        "memories only. Keep it bounded and useful for interaction.\n\nActive "
        "about-user memories:\n" + _json(memories)
    )


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


__all__ = [
    "USER_MODEL_REASONING_SYSTEM_PROMPT",
    "build_user_model_reasoner",
    "user_model_reasoning_prompt",
]
