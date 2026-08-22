"""UserModel profile-card reasoner and its prompt."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING

from ..embedding import DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS, EmbeddingClient
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
    transport: LlmTransport, settings: ReasoningSettings, user_model: UserModelView,
    memories: Sequence[MemoryView],
    inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    *,
    store: WorldStore | None = None,
    space_id: str | None = None,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> UserModelReasoningResult:
    projection, actions = await owner_reasoning(
        transport, settings, settings.prompts.user_model_reasoning_system,
        user_model,
        memories, inferred_memories, hypotheses, PersonProjectionResult,
        UserReasoningActionsResult,
        store=store, space_id=space_id, embedding_client_getter=embedding_client_getter,
        embedding_query_timeout_seconds=embedding_query_timeout_seconds,
    )
    return UserModelReasoningResult(
        profile_card=projection.profile_card, hypothesis_actions=actions.hypothesis_actions,
    )


def build_user_model_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings,
    embedding_client_getter: Callable[[], EmbeddingClient | None] | None = None,
    embedding_query_timeout_seconds: float = DEFAULT_EMBEDDING_QUERY_TIMEOUT_SECONDS,
) -> DescriptorReasoner:
    """Fold the space's UserModel card forward per attempt.

    Same `delta_only` read as the person and relationship reasoners: the
    card plus its `about_user` memories changed since the card's own
    watermark. The full `about_user` history is folded only when the card
    has no watermark yet -- the case `overwrite_user_model` seeds from
    `gossipmemo import --user-md`.
    """

    reason_user_model = partial(
        _reason_user_model, model, settings,
        store=store, embedding_client_getter=embedding_client_getter,
        embedding_query_timeout_seconds=embedding_query_timeout_seconds,
    )

    def load_context(space_id: str):
        _, _, user_models = store.stale_entities()
        if space_id not in user_models:
            return None
        context = store.user_model_context(space_id, delta_only=True)
        if not context:
            return None
        user_model, memories, watermark = context
        if not user_model.stale:
            return None
        evidence = [memory for memory in memories if memory.basis != "inferred"]
        inferred, hypotheses = store.owner_review_context(space_id, "user", None)
        return watermark, user_model, evidence, inferred, hypotheses

    def call(space_id: str, context):
        _, user_model, evidence, inferred, hypotheses = context
        return (
            "reason-user-model",
            partial(reason_user_model, space_id=space_id),
            (user_model, evidence, inferred, hypotheses),
        )

    def apply(space_id: str, context, result) -> bool:
        watermark, _, _, _, hypotheses = context
        return store.apply_user_model_reasoning(
            space_id, watermark, result, {hypothesis.id for hypothesis in hypotheses}
        )

    return DescriptorReasoner("user_model", load_context, call, apply)


__all__ = ["build_user_model_reasoner"]
