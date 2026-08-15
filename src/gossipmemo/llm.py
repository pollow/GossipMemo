"""LLM interface and an OpenAI-compatible chat-completions adapter.

The rest of GossipMemo depends on the small :class:`LlmModel` interface in
this module.  The concrete adapter is deliberately responsible for all HTTP
details and for validating model output against the domain Pydantic models;
callers never need to parse a chat-completions response themselves.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, Literal, Protocol, TypeVar, cast, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .context_budget import ContextBudget
from .models import (
    ExtractionResult,
    MemoryView,
    ModelMessage,
    PersonReasoningResult,
    PersonProjectionResult,
    RelationshipProjectionResult,
    ReasoningActionsResult,
    UserReasoningActionsResult,
    HypothesisView,
    PersonView,
    QueryContext,
    RelationshipReasoningResult,
    RelationshipView,
    UserModelReasoningResult,
    UserModelView,
    ExtractedOwnerEvidenceDigest,
    OwnerEvidenceDigestView,
    ContinuityReasoningResult,
    ContinuityView,
    CoverageAuditPatch,
    CoverageMapView,
    GoalPlanningResult,
    GoalPlanningCandidates,
    LearningGoalCandidate,
    LearningGoalView,
)
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    PERSON_REASONING_SYSTEM_PROMPT,
    QUERY_SYNTHESIS_SYSTEM_PROMPT,
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    USER_MODEL_REASONING_SYSTEM_PROMPT,
    extraction_prompt,
    person_reasoning_prompt,
    owner_reasoning_prefix,
    owner_evidence_digest_prompt,
    projection_stage_prompt,
    actions_stage_prompt,
    query_synthesis_prompt,
    relationship_reasoning_prompt,
    user_model_reasoning_prompt,
    CONTINUITY_SYSTEM_PROMPT,
    COVERAGE_AUDIT_SYSTEM_PROMPT,
    GOAL_PLANNING_SYSTEM_PROMPT,
    continuity_prompt,
    coverage_audit_prompt,
    goal_planning_prompt,
    goal_candidate_prompt,
    goal_reconciliation_prompt,
    goal_candidate_reduction_prompt,
    schema_instruction,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class LlmModel(Protocol):
    """The application-facing asynchronous LLM seam.

    Implementations may be deterministic fakes or a hosted model adapter.
    Server configuration is validated before this interface is constructed.
    """

    @property
    def configured(self) -> bool: ...

    async def extract(
        self,
        messages: Sequence[ModelMessage],
        context: Sequence[ModelMessage] = (),
        known_people: Sequence[dict[str, Any]] = (),
        comparison_memories: Sequence[MemoryView] = (),
    ) -> ExtractionResult: ...

    async def reason_person(
        self, person: PersonView, memories: Sequence[MemoryView],
        inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    ) -> PersonReasoningResult: ...

    async def reason_relationship(
        self, relationship: RelationshipView, memories: Sequence[MemoryView],
        inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    ) -> RelationshipReasoningResult: ...

    async def reason_user_model(
        self, memories: Sequence[MemoryView],
        inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    ) -> UserModelReasoningResult: ...

    async def reason_continuity(
        self, continuity: ContinuityView | None, messages: Sequence[ModelMessage]
    ) -> ContinuityReasoningResult: ...

    async def audit_coverage(
        self, coverage: CoverageMapView, memories: Sequence[MemoryView],
        hypotheses: Sequence[HypothesisView] = (),
    ) -> CoverageAuditPatch: ...

    async def plan_learning_goals(
        self, coverage: CoverageMapView, hypotheses: Sequence[HypothesisView],
        open_goals: Sequence[LearningGoalView], recent_closed_goals: Sequence[LearningGoalView],
    ) -> GoalPlanningResult: ...

    async def synthesize(self, question: str, context: QueryContext) -> str: ...


LLMModel = LlmModel


class LLMError(RuntimeError):
    """Base class for model configuration, transport, and output failures."""


class LLMRequestError(LLMError):
    """Raised when the chat-completions HTTP request fails."""


class LLMProtocolError(LLMError):
    """Raised when a provider response is not a valid chat completion."""


class LLMOutputError(LLMError):
    """Raised when model content cannot validate as the requested result."""


class ChatMessage(BaseModel):
    """Minimal OpenAI-compatible chat message used in request payloads."""

    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class ChatCompletionRequest(BaseModel):
    """Pydantic representation of the request sent to a compatible server."""

    model_config = ConfigDict(extra="ignore")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    response_format: dict[str, Any] | None = None
    max_tokens: int | None = Field(default=None, ge=1)


class ChatCompletionMessage(BaseModel):
    """Relevant subset of an OpenAI-compatible response message."""

    model_config = ConfigDict(extra="ignore")

    role: str = "assistant"
    content: str | list[Any] | None = None


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: ChatCompletionMessage


class ChatCompletionResponse(BaseModel):
    """Relevant subset of a chat-completions response."""

    model_config = ConfigDict(extra="ignore")

    choices: list[ChatCompletionChoice] = Field(min_length=1)


class OpenAICompatibleAdapter(AbstractAsyncContextManager["OpenAICompatibleAdapter"]):
    """Adapter for servers exposing an OpenAI-compatible ``/chat/completions``.

    ``base_url`` may be either the configured API root or a complete
    chat-completions endpoint. GossipMemo supplies no provider URL default.
    Supplying an ``httpx.AsyncClient`` is supported for tests and for callers
    that manage connection pooling themselves; when omitted, this adapter
    owns a lazily-created client and closes it in :meth:`aclose`.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
        headers: Mapping[str, str] | None = None,
        extraction_policy: str = "balanced",
        user_name: str = "CurrentUser",
        max_retries: int = 5,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        context_budget: ContextBudget | None = None,
    ) -> None:
        normalized_base = base_url.strip().rstrip("/")
        if not normalized_base:
            raise ValueError("LLM base_url must not be empty")
        if not model.strip():
            raise ValueError("LLM model must not be empty")
        if not api_key.strip():
            raise ValueError("LLM api_key must not be empty")
        if timeout <= 0:
            raise ValueError("LLM timeout must be greater than zero")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("LLM max_tokens must be greater than zero")
        if extraction_policy not in {"conservative", "balanced", "comprehensive"}:
            raise ValueError(
                "extraction_policy must be conservative, balanced, or comprehensive"
            )
        if not user_name.strip():
            raise ValueError("LLM user_name must not be empty")
        if max_retries < 0:
            raise ValueError("LLM max_retries must not be negative")
        if retry_base_seconds <= 0:
            raise ValueError("LLM retry_base_seconds must be greater than zero")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError(
                "LLM retry_max_seconds must be at least retry_base_seconds"
            )

        self.base_url = normalized_base
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extraction_policy = extraction_policy
        self.user_name = user_name.strip()
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.context_budget = context_budget or ContextBudget()
        self._client = client
        self._owns_client = client is None
        self._headers = dict(headers or {})

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAICompatibleAdapter":
        """Build the Adapter from the already-validated global settings."""

        return cls(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            extraction_policy=settings.extraction_policy,
            user_name=settings.user_name,
            max_retries=settings.llm_max_retries,
            retry_base_seconds=settings.llm_retry_base_seconds,
            retry_max_seconds=settings.llm_retry_max_seconds,
            context_budget=ContextBudget(
                settings.llm_context_window_tokens,
                settings.llm_output_reserve_tokens,
                settings.llm_context_safety_tokens,
            ),
        )

    @property
    def configured(self) -> bool:
        return True

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return self.base_url + "/chat/completions"

    async def extract(
        self,
        messages: Sequence[ModelMessage],
        context: Sequence[ModelMessage] = (),
        known_people: Sequence[dict[str, Any]] = (),
        comparison_memories: Sequence[MemoryView] = (),
    ) -> ExtractionResult:
        content = await self._structured_call(
            EXTRACTION_SYSTEM_PROMPT,
            extraction_prompt(
                list(messages),
                self.extraction_policy,
                list(context),
                list(known_people),
                self.user_name,
                list(comparison_memories),
            ),
            ExtractionResult,
        )
        return cast(ExtractionResult, content)

    async def reason_person(
        self, person: PersonView, memories: Sequence[MemoryView],
        inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    ) -> PersonReasoningResult:
        projection, actions = await self._owner_reasoning(
            PERSON_REASONING_SYSTEM_PROMPT, person, memories, inferred_memories, hypotheses, PersonProjectionResult,
        )
        return PersonReasoningResult(profile_card=projection.profile_card, **actions.model_dump(exclude_none=True))

    async def reason_relationship(
        self, relationship: RelationshipView, memories: Sequence[MemoryView],
        inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    ) -> RelationshipReasoningResult:
        projection, actions = await self._owner_reasoning(
            RELATIONSHIP_REASONING_SYSTEM_PROMPT, relationship, memories, inferred_memories, hypotheses, RelationshipProjectionResult,
        )
        return RelationshipReasoningResult(**projection.model_dump(), **actions.model_dump(exclude_none=True))

    async def reason_user_model(
        self, memories: Sequence[MemoryView],
        inferred_memories: Sequence[MemoryView] = (), hypotheses: Sequence[HypothesisView] = (),
    ) -> UserModelReasoningResult:
        projection, actions = await self._owner_reasoning(
            USER_MODEL_REASONING_SYSTEM_PROMPT, UserModelView(space_id="current"), memories, inferred_memories, hypotheses, PersonProjectionResult, UserReasoningActionsResult,
        )
        return UserModelReasoningResult(profile_card=projection.profile_card, hypothesis_actions=actions.hypothesis_actions)

    async def _owner_reasoning(self, system_prompt: str, target: BaseModel, memories: Sequence[MemoryView], inferred_memories: Sequence[MemoryView], hypotheses: Sequence[HypothesisView], projection_type: type[BaseModel], actions_type: type[BaseModel] = ReasoningActionsResult) -> tuple[BaseModel, BaseModel]:
        bounded_inferred, bounded_hypotheses = self._bounded_owner_comparisons(
            system_prompt, target, inferred_memories, hypotheses,
            projection_type, actions_type,
        )
        prefix = owner_reasoning_prefix(target, list(memories), bounded_inferred, bounded_hypotheses, user_name=self.user_name)
        common = [ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=prefix)]
        first_messages = common + [ChatMessage(role="user", content=projection_stage_prompt() + "\n" + schema_instruction(projection_type))]
        if not self._owner_stage2_fits(first_messages, actions_type):
            digest = await self._digest_owner_evidence(
                target, memories, bounded_inferred, bounded_hypotheses,
                system_prompt, projection_type, actions_type,
            )
            prefix = owner_reasoning_prefix(target, digest, bounded_inferred, bounded_hypotheses, user_name=self.user_name)
            common = [ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=prefix)]
            first_messages = common + [ChatMessage(role="user", content=projection_stage_prompt() + "\n" + schema_instruction(projection_type))]
            if not self._owner_stage2_fits(first_messages, actions_type):
                raise ValueError("owner evidence digest did not fit context budget")
        first, projection = await self._structured_messages(
            first_messages, projection_type,
        )
        _, actions = await self._structured_messages(
            first_messages + [
                ChatMessage(role="assistant", content=first),
                ChatMessage(
                    role="user",
                    content=actions_stage_prompt() + "\n" + schema_instruction(actions_type),
                ),
            ],
            actions_type,
        )
        return projection, actions

    def _owner_first_messages(
        self, system_prompt: str, target: BaseModel, evidence: Sequence[Any],
        inferred: Sequence[MemoryView], hypotheses: Sequence[HypothesisView],
        projection_type: type[BaseModel],
    ) -> list[ChatMessage]:
        prefix = owner_reasoning_prefix(
            target, list(evidence), list(inferred), list(hypotheses),
            user_name=self.user_name,
        )
        return [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=prefix),
            ChatMessage(
                role="user",
                content=projection_stage_prompt() + "\n" + schema_instruction(projection_type),
            ),
        ]

    def _owner_stage2_estimate(
        self, first_messages: list[ChatMessage], actions_type: type[BaseModel],
    ) -> int:
        request = ChatCompletionRequest(
            model=self.model,
            messages=first_messages + [
                ChatMessage(role="assistant", content=""),
                ChatMessage(
                    role="user",
                    content=actions_stage_prompt() + "\n" + schema_instruction(actions_type),
                ),
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
        )
        # The second request contains the first completion as an assistant
        # message. Reserve its configured maximum in addition to the normal
        # output reserve already excluded by ContextBudget.
        return (
            self.context_budget.estimate_request(request)
            + self.context_budget.output_reserve_tokens
        )

    def _owner_stage2_fits(self, first_messages: list[ChatMessage], actions_type: type[BaseModel]) -> bool:
        return self.context_budget.report(
            self._owner_stage2_estimate(first_messages, actions_type)
        ).fits

    def _bounded_owner_comparisons(
        self, system_prompt: str, target: BaseModel,
        inferred: Sequence[MemoryView], hypotheses: Sequence[HypothesisView],
        projection_type: type[BaseModel], actions_type: type[BaseModel],
    ) -> tuple[list[MemoryView], list[HypothesisView]]:
        """Bound comparison-only state without turning it into evidence.

        IDs are retained before prose. If even every ID skeleton cannot fit,
        the oldest tail is omitted; omission remains a no-op by contract.
        Comparison state gets at most one third of the usable input budget so
        evidence always has room to be represented or digested.
        """
        empty_first = self._owner_first_messages(
            system_prompt, target, [], [], [], projection_type,
        )
        base = self._owner_stage2_estimate(empty_first, actions_type)
        ceiling = min(
            self.context_budget.usable_input_tokens,
            base + self.context_budget.usable_input_tokens // 3,
        )
        if base > ceiling:
            raise ValueError("owner prompt scaffolding exceeds context budget")

        chosen_inferred: list[MemoryView] = []
        chosen_hypotheses: list[HypothesisView] = []

        def fits(
            candidate_inferred: Sequence[MemoryView],
            candidate_hypotheses: Sequence[HypothesisView],
        ) -> bool:
            first = self._owner_first_messages(
                system_prompt, target, [], candidate_inferred,
                candidate_hypotheses, projection_type,
            )
            return self._owner_stage2_estimate(first, actions_type) <= ceiling

        # Preserve actionable IDs first with empty prose, in the stable store
        # order (newest first). Then spend remaining budget on bounded prose.
        for item in inferred:
            skeleton = item.model_copy(update={"content": ""})
            if fits([*chosen_inferred, skeleton], chosen_hypotheses):
                chosen_inferred.append(skeleton)
        for item in hypotheses:
            skeleton = item.model_copy(update={"content": ""})
            if fits(chosen_inferred, [*chosen_hypotheses, skeleton]):
                chosen_hypotheses.append(skeleton)

        inferred_by_id = {item.id: item for item in inferred}
        for index, skeleton in enumerate(tuple(chosen_inferred)):
            original = inferred_by_id[skeleton.id]
            expanded = original.model_copy(update={"content": original.content[:1200]})
            candidate = [*chosen_inferred]
            candidate[index] = expanded
            if fits(candidate, chosen_hypotheses):
                chosen_inferred = candidate

        hypotheses_by_id = {item.id: item for item in hypotheses}
        for index, skeleton in enumerate(tuple(chosen_hypotheses)):
            original = hypotheses_by_id[skeleton.id]
            expanded = original.model_copy(update={"content": original.content[:1200]})
            candidate = [*chosen_hypotheses]
            candidate[index] = expanded
            if fits(chosen_inferred, candidate):
                chosen_hypotheses = candidate
        return chosen_inferred, chosen_hypotheses

    def _digest_request(self, chunk: list[Any]) -> ChatCompletionRequest:
        return ChatCompletionRequest(model=self.model, messages=[ChatMessage(role="system", content="Compress evidence only; do not infer people or actions.\n\n" + schema_instruction(ExtractedOwnerEvidenceDigest)), ChatMessage(role="user", content=owner_evidence_digest_prompt(chunk, self.user_name))], temperature=self.temperature, response_format={"type":"json_object"}, max_tokens=self.max_tokens)

    async def _digest_owner_evidence(self, target: BaseModel, memories: Sequence[MemoryView], inferred_memories: Sequence[MemoryView], hypotheses: Sequence[HypothesisView], system_prompt: str, projection_type: type[BaseModel], actions_type: type[BaseModel]) -> list[OwnerEvidenceDigestView]:
        source: list[MemoryView | OwnerEvidenceDigestView] = list(memories)
        original_ids = {memory.id for memory in source}
        previous_size: int | None = None
        for _ in range(3):
            chunks: list[list[MemoryView | OwnerEvidenceDigestView]] = []
            current: list[MemoryView | OwnerEvidenceDigestView] = []
            for memory in source:
                # Segment oversized content so every digest request is valid.
                pieces = [memory]
                if isinstance(memory, MemoryView) and not self.context_budget.report(self.context_budget.estimate_request(self._digest_request([memory]))).fits:
                    lo, hi = 1, len(memory.content)
                    while lo < hi:
                        mid = (lo + hi + 1) // 2
                        if self.context_budget.report(self.context_budget.estimate_request(self._digest_request([memory.model_copy(update={"content": memory.content[:mid]})]))).fits: lo = mid
                        else: hi = mid - 1
                    width = lo
                    if width < 1:
                        raise ValueError(
                            f"single owner evidence segment exceeds context budget: {memory.id}"
                        )
                    pieces = [memory.model_copy(update={"content": memory.content[i:i + width]}) for i in range(0, len(memory.content), width)]
                for piece in pieces:
                    candidate = current + [piece]
                    candidate_ids = set().union(
                        *(self._owner_evidence_source_ids(item) for item in candidate)
                    )
                    candidate_fits = self.context_budget.report(
                        self.context_budget.estimate_request(self._digest_request(candidate))
                    ).fits and len(candidate_ids) <= 512
                    if current and not candidate_fits:
                        chunks.append(current)
                        current = []
                    current.append(piece)
                    if not self.context_budget.report(self.context_budget.estimate_request(self._digest_request(current))).fits:
                        identifier = getattr(memory, "id", "digest")
                        raise ValueError(
                            f"single owner evidence segment exceeds context budget: {identifier}"
                        )
            if current:
                chunks.append(current)
            output: list[OwnerEvidenceDigestView] = []
            for chunk in chunks:
                request = self._digest_request(chunk)
                self.context_budget.check(self.context_budget.estimate_request(request))
                _, result = await self._structured_messages(
                    request.messages, ExtractedOwnerEvidenceDigest,
                )
                allowed = set().union(
                    *(self._owner_evidence_source_ids(item) for item in chunk)
                )
                accepted = [
                    OwnerEvidenceDigestView.model_validate(item.model_dump(mode="json"))
                    for item in result.items
                    if item.source_memory_ids
                    and set(item.source_memory_ids) <= allowed
                ]
                covered = set().union(
                    *(set(item.source_memory_ids) for item in accepted), set()
                )
                if covered != allowed:
                    raise LLMOutputError(
                        "owner evidence digest omitted or invented source Memory IDs"
                    )
                output.extend(accepted)
            if not output:
                raise ValueError("owner evidence digest made no progress")
            final_prefix = owner_reasoning_prefix(target, output, list(inferred_memories), list(hypotheses), user_name=self.user_name)
            first = [ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=final_prefix), ChatMessage(role="user", content=projection_stage_prompt() + "\n" + schema_instruction(projection_type))]
            if self._owner_stage2_fits(first, actions_type):
                return output
            size = self.context_budget.estimate_text(final_prefix)
            if previous_size is not None and size >= previous_size:
                raise ValueError("owner evidence digest reduction made no progress")
            previous_size = size
            source = output
        raise ValueError("owner evidence digest reduction made no progress within 3 rounds")

    @staticmethod
    def _owner_evidence_source_ids(
        item: MemoryView | OwnerEvidenceDigestView,
    ) -> set[str]:
        if isinstance(item, MemoryView):
            return {item.id}
        return set(item.source_memory_ids)

    async def reason_continuity(
        self, continuity: ContinuityView | None, messages: Sequence[ModelMessage]
    ) -> ContinuityReasoningResult:
        source = list(messages)
        if not source:
            raise ValueError("continuity requires at least one message")
        request_for = lambda prior, chunk: self._structured_request(
            CONTINUITY_SYSTEM_PROMPT, continuity_prompt(prior, chunk), ContinuityReasoningResult
        )
        normal = request_for(continuity, source)
        if self.context_budget.report(self.context_budget.estimate_request(normal)).fits:
            _, result = await self._structured_messages(
                normal.messages, ContinuityReasoningResult,
            )
            return result
        # Reserve room for the streamed typed prior, not merely the initial
        # empty/null continuity object.
        chunk_prior = ContinuityView(
            text=(continuity.text if continuity and continuity.text else "summary"),
            related_person_ids=(continuity.related_person_ids if continuity and continuity.related_person_ids else ["person"]),
            through_message_id=(continuity.through_message_id if continuity and continuity.through_message_id else "continuity"),
        )
        chunks = self._greedy_chunks(source, lambda chunk: request_for(chunk_prior, chunk))
        # A small normal update deliberately remains one call.  Large updates
        # stream typed summaries forward; only the caller persists the final one.
        prior = continuity
        result: ContinuityReasoningResult | None = None
        for chunk in chunks:
            prior = self._fit_continuity_prior(prior, chunk, request_for)
            request = request_for(prior, chunk)
            _, result = await self._structured_messages(
                request.messages, ContinuityReasoningResult,
            )
            prior = ContinuityView(**result.model_dump())
        assert result is not None
        return result.model_copy(update={"through_message_id": source[-1].id})

    async def audit_coverage(
        self, coverage: CoverageMapView, memories: Sequence[MemoryView],
        hypotheses: Sequence[HypothesisView] = (),
    ) -> CoverageAuditPatch:
        normal = self._structured_request(COVERAGE_AUDIT_SYSTEM_PROMPT, coverage_audit_prompt(coverage, list(memories), list(hypotheses)), CoverageAuditPatch)
        if self.context_budget.report(self.context_budget.estimate_request(normal)).fits:
            _, result = await self._structured_messages(
                normal.messages, CoverageAuditPatch,
            )
            return result
        bounded_coverage, bounded_hypotheses = self._bounded_coverage_context(coverage, hypotheses)
        request_for = lambda chunk: self._structured_request(
            COVERAGE_AUDIT_SYSTEM_PROMPT,
            coverage_audit_prompt(bounded_coverage, chunk, bounded_hypotheses),
            CoverageAuditPatch,
        )
        patches: list[CoverageAuditPatch] = []
        for chunk in self._greedy_chunks(list(memories), request_for):
            request = request_for(chunk)
            _, result = await self._structured_messages(
                request.messages, CoverageAuditPatch,
            )
            patches.append(self._filter_coverage_patch(
                result,
                {memory.id for memory in chunk},
            ))
        return CoverageAuditPatch(
            criteria=[item for patch in patches for item in patch.criteria],
            boundary_upserts=[item for patch in patches for item in patch.boundary_upserts],
            boundary_transitions=[item for patch in patches for item in patch.boundary_transitions],
            life_periods=[item for patch in patches for item in patch.life_periods],
            relationship_arcs=[item for patch in patches for item in patch.relationship_arcs],
            behavioral_contexts=[item for patch in patches for item in patch.behavioral_contexts],
        )

    async def plan_learning_goals(
        self, coverage: CoverageMapView, hypotheses: Sequence[HypothesisView],
        open_goals: Sequence[LearningGoalView], recent_closed_goals: Sequence[LearningGoalView],
    ) -> GoalPlanningResult:
        normal = self._structured_request(
            GOAL_PLANNING_SYSTEM_PROMPT,
            goal_planning_prompt(coverage, list(hypotheses), list(open_goals), list(recent_closed_goals)),
            GoalPlanningResult,
        )
        if self.context_budget.report(self.context_budget.estimate_request(normal)).fits:
            return cast(GoalPlanningResult, await self._structured_call(
                GOAL_PLANNING_SYSTEM_PROMPT,
                goal_planning_prompt(coverage, list(hypotheses), list(open_goals), list(recent_closed_goals)), GoalPlanningResult,
            ))

        bounded_coverage, bounded_hypotheses, bounded_open, bounded_closed = self._bounded_goal_context(
            coverage, hypotheses, open_goals, recent_closed_goals,
        )
        candidate_request = lambda chunk: self._structured_request(
            GOAL_PLANNING_SYSTEM_PROMPT,
            goal_candidate_prompt(bounded_coverage, chunk, bounded_open, bounded_closed),
            GoalPlanningCandidates,
        )
        candidates: list[LearningGoalCandidate] = []
        # Candidate passes cover every original hypothesis; only the final
        # reconciliation uses the complete ID-preserving bounded map.
        for chunk in self._greedy_chunks(list(hypotheses) or [None], lambda items: candidate_request([item for item in items if item is not None])):
            request = candidate_request([item for item in chunk if item is not None])
            _, result = await self._structured_messages(
                request.messages, GoalPlanningCandidates,
            )
            candidates.extend(self._filter_goal_candidates(
                result.candidates,
                coverage, hypotheses,
            ))
        final_request = self._structured_request(GOAL_PLANNING_SYSTEM_PROMPT, goal_reconciliation_prompt(bounded_coverage, bounded_hypotheses, bounded_open, bounded_closed, candidates), GoalPlanningResult)
        previous_size: int | None = None
        for _ in range(3):
            if self.context_budget.report(self.context_budget.estimate_request(final_request)).fits:
                break
            chunks = self._greedy_chunks(candidates, lambda items: self._structured_request(GOAL_PLANNING_SYSTEM_PROMPT, goal_candidate_reduction_prompt(items), GoalPlanningCandidates))
            reduced: list[LearningGoalCandidate] = []
            for chunk in chunks:
                request = self._structured_request(GOAL_PLANNING_SYSTEM_PROMPT, goal_candidate_reduction_prompt(chunk), GoalPlanningCandidates)
                _, result = await self._structured_messages(
                    request.messages, GoalPlanningCandidates,
                )
                reduced.extend(self._filter_goal_candidates(
                    result.candidates,
                    coverage, hypotheses,
                ))
            size = self.context_budget.estimate_text(str([item.model_dump() for item in reduced]))
            if not reduced or previous_size is not None and size >= previous_size:
                raise ValueError("learning-goal candidate reduction made no progress")
            previous_size, candidates = size, reduced
            final_request = self._structured_request(GOAL_PLANNING_SYSTEM_PROMPT, goal_reconciliation_prompt(bounded_coverage, bounded_hypotheses, bounded_open, bounded_closed, candidates), GoalPlanningResult)
        else:
            raise ValueError("learning-goal candidate reduction made no progress within 3 rounds")
        self.context_budget.check(self.context_budget.estimate_request(final_request))
        _, result = await self._structured_messages(
            final_request.messages, GoalPlanningResult,
        )
        return result

    def _structured_request(self, system_prompt: str, user_prompt: str, result_type: type[BaseModel]) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model=self.model,
            messages=[ChatMessage(role="system", content=system_prompt + "\n\n" + schema_instruction(result_type)), ChatMessage(role="user", content=user_prompt)],
            temperature=self.temperature, response_format={"type": "json_object"}, max_tokens=self.max_tokens,
        )

    def _greedy_chunks(self, items: Sequence[Any], request_for: Any) -> list[list[Any]]:
        """Split by exact serialized request size, including CJK and schemas."""
        chunks: list[list[Any]] = []
        current: list[Any] = []
        for item in items:
            pieces = self._split_oversized_item(item, request_for)
            for piece in pieces:
                candidate = [*current, piece]
                if current and not self.context_budget.report(self.context_budget.estimate_request(request_for(candidate))).fits:
                    chunks.append(current)
                    current = []
                current.append(piece)
                self.context_budget.check(self.context_budget.estimate_request(request_for(current)))
        if current:
            chunks.append(current)
        return chunks

    def _split_oversized_item(self, item: Any, request_for: Any) -> list[Any]:
        if self.context_budget.report(self.context_budget.estimate_request(request_for([item]))).fits:
            return [item]
        content = getattr(item, "content", None)
        if not isinstance(content, str) or not content:
            raise ValueError("single context item exceeds input budget")
        lo, hi = 1, len(content)
        while lo < hi:
            middle = (lo + hi + 1) // 2
            candidate = item.model_copy(update={"content": content[:middle]})
            if self.context_budget.report(self.context_budget.estimate_request(request_for([candidate]))).fits:
                lo = middle
            else:
                hi = middle - 1
        if lo < 1:
            raise ValueError("single context item exceeds input budget")
        return [item.model_copy(update={"content": content[index:index + lo]}) for index in range(0, len(content), lo)]

    def _fit_continuity_prior(self, prior: ContinuityView | None, chunk: list[ModelMessage], request_for: Any) -> ContinuityView | None:
        if prior is None or self.context_budget.report(self.context_budget.estimate_request(request_for(prior, chunk))).fits:
            return prior
        text = prior.text
        lo, hi = 0, len(text)
        while lo < hi:
            middle = (lo + hi + 1) // 2
            candidate = prior.model_copy(update={"text": text[:middle]})
            if self.context_budget.report(self.context_budget.estimate_request(request_for(candidate, chunk))).fits:
                lo = middle
            else:
                hi = middle - 1
        candidate = prior.model_copy(update={"text": text[:lo]})
        self.context_budget.check(self.context_budget.estimate_request(request_for(candidate, chunk)))
        return candidate

    def _bounded_coverage_context(self, coverage: CoverageMapView, hypotheses: Sequence[HypothesisView]) -> tuple[CoverageMapView, list[HypothesisView]]:
        # Bound comparison-only hypotheses and boundary prose before evidence is
        # added. IDs remain present whenever scaffold space permits it.
        def fits(
            candidate_coverage: CoverageMapView,
            candidate_hypotheses: Sequence[HypothesisView],
        ) -> bool:
            request = self._structured_request(
                COVERAGE_AUDIT_SYSTEM_PROMPT,
                coverage_audit_prompt(
                    candidate_coverage, [], list(candidate_hypotheses)
                ),
                CoverageAuditPatch,
            )
            return self.context_budget.report(self.context_budget.estimate_request(request)).fits

        if fits(coverage, hypotheses):
            return coverage, list(hypotheses)
        skeleton_hypotheses = [
            item.model_copy(update={"content": "", "evidence": []})
            for item in hypotheses
        ]
        if fits(coverage, skeleton_hypotheses):
            return coverage, skeleton_hypotheses

        bounded = self._coverage_skeleton(coverage)
        if not fits(bounded, []):
            raise ValueError("coverage prompt scaffolding exceeds context budget")
        chosen: list[HypothesisView] = []
        for hypothesis in hypotheses:
            skeleton = hypothesis.model_copy(update={"content": "", "evidence": []})
            if fits(bounded, [*chosen, skeleton]):
                chosen.append(skeleton)
        return bounded, chosen

    def _bounded_goal_context(self, coverage: CoverageMapView, hypotheses: Sequence[HypothesisView], open_goals: Sequence[LearningGoalView], closed_goals: Sequence[LearningGoalView]) -> tuple[CoverageMapView, list[HypothesisView], list[LearningGoalView], list[LearningGoalView]]:
        # Goal planning has different schemas and prompt scaffolding than a
        # coverage audit, so its minimum-fit proof must use its own requests.
        def fits(
            candidate_coverage: CoverageMapView,
            candidate_hypotheses: Sequence[HypothesisView],
            candidate_open: Sequence[LearningGoalView],
            candidate_closed: Sequence[LearningGoalView],
        ) -> bool:
            candidate = self._structured_request(
                GOAL_PLANNING_SYSTEM_PROMPT,
                goal_candidate_prompt(
                    candidate_coverage, list(candidate_hypotheses),
                    list(candidate_open), list(candidate_closed),
                ),
                GoalPlanningCandidates,
            )
            final = self._structured_request(
                GOAL_PLANNING_SYSTEM_PROMPT,
                goal_reconciliation_prompt(
                    candidate_coverage, list(candidate_hypotheses),
                    list(candidate_open), list(candidate_closed), [],
                ),
                GoalPlanningResult,
            )
            return all(
                self.context_budget.report(
                    self.context_budget.estimate_request(request)
                ).fits
                for request in (candidate, final)
            )

        skeleton_hypotheses = [
            item.model_copy(update={"content": "", "evidence": []})
            for item in hypotheses
        ]
        skeleton_open = [
            goal.model_copy(update={"prompt": "", "rationale": ""})
            for goal in open_goals
        ]
        skeleton_closed = [
            goal.model_copy(update={"prompt": "", "rationale": ""})
            for goal in closed_goals
        ]
        if fits(coverage, skeleton_hypotheses, skeleton_open, skeleton_closed):
            return coverage, skeleton_hypotheses, skeleton_open, skeleton_closed

        bounded_coverage = self._coverage_skeleton(coverage)
        if not fits(bounded_coverage, [], [], []):
            raise ValueError("learning-goal minimum skeleton exceeds context budget")

        bounded_hypotheses: list[HypothesisView] = []
        bounded_open: list[LearningGoalView] = []
        bounded_closed: list[LearningGoalView] = []
        # Lifecycle IDs have priority; omitted comparison state is always no-op.
        for item in skeleton_open:
            if fits(
                bounded_coverage, bounded_hypotheses,
                [*bounded_open, item], bounded_closed,
            ):
                bounded_open.append(item)
        for item in skeleton_hypotheses:
            if fits(
                bounded_coverage, [*bounded_hypotheses, item],
                bounded_open, bounded_closed,
            ):
                bounded_hypotheses.append(item)
        for item in skeleton_closed:
            if fits(
                bounded_coverage, bounded_hypotheses,
                bounded_open, [*bounded_closed, item],
            ):
                bounded_closed.append(item)
        return bounded_coverage, bounded_hypotheses, bounded_open, bounded_closed

    @staticmethod
    def _coverage_skeleton(coverage: CoverageMapView) -> CoverageMapView:
        criteria = {
            identifier: {
                "level": value.get("level", "unknown"),
                "known_state": str(value.get("known_state", ""))[:800],
                "evidence_memory_ids": list(value.get("evidence_memory_ids", []))[:32],
            }
            for identifier, value in coverage.criteria.items()
        }
        boundaries = [
            item.model_copy(update={
                "summary": item.summary[:600],
                "evidence_memory_ids": item.evidence_memory_ids[:32],
            })
            for item in coverage.boundaries
        ]
        return coverage.model_copy(update={
            "criteria": criteria,
            "boundaries": boundaries,
            "life_periods": [str(item)[:600] for item in coverage.life_periods[:30]],
            "relationship_arcs": [str(item)[:600] for item in coverage.relationship_arcs[:30]],
            "behavioral_contexts": [str(item)[:600] for item in coverage.behavioral_contexts[:30]],
        })

    @staticmethod
    def _filter_coverage_patch(patch: CoverageAuditPatch, evidence_ids: set[str]) -> CoverageAuditPatch:
        return CoverageAuditPatch(
            criteria=[item.model_copy(update={"evidence_memory_ids": [identifier for identifier in item.evidence_memory_ids if identifier in evidence_ids]}) for item in patch.criteria],
            boundary_upserts=[item.model_copy(update={"evidence_memory_ids": [identifier for identifier in item.evidence_memory_ids if identifier in evidence_ids]}) for item in patch.boundary_upserts],
            boundary_transitions=patch.boundary_transitions,
            life_periods=patch.life_periods, relationship_arcs=patch.relationship_arcs, behavioral_contexts=patch.behavioral_contexts,
        )

    @staticmethod
    def _filter_goal_candidates(candidates: Sequence[LearningGoalCandidate], coverage: CoverageMapView, hypotheses: Sequence[HypothesisView]) -> list[LearningGoalCandidate]:
        criteria = set(coverage.criteria)
        boundaries = {item.id for item in coverage.boundaries if item.status == "open"}
        # Persistence remains the authority for focus Person/Relationship IDs.
        accepted: list[LearningGoalCandidate] = []
        for item in candidates:
            if not set(item.criteria_refs).issubset(criteria) or not set(item.boundary_ids).issubset(boundaries):
                continue
            accepted.append(item)
        return accepted

    async def synthesize(self, question: str, context: QueryContext) -> str:
        if not question.strip():
            raise ValueError("query question must not be empty")
        content = await self._chat(
            QUERY_SYNTHESIS_SYSTEM_PROMPT,
            query_synthesis_prompt(question, context),
            structured=False,
        )
        return content.strip()

    async def _structured_call(
        self,
        system_prompt: str,
        user_prompt: str,
        result_type: type[_ResultT],
    ) -> _ResultT:
        _, result = await self._structured_messages(
            [
                ChatMessage(
                    role="system",
                    content=system_prompt + "\n\n" + schema_instruction(result_type),
                ),
                ChatMessage(role="user", content=user_prompt),
            ],
            result_type,
        )
        return result

    async def _structured_messages(
        self, messages: list[ChatMessage], result_type: type[_ResultT],
    ) -> tuple[str, _ResultT]:
        """Retry transient malformed structured output with the same request."""
        attempt = 0
        while True:
            content = await self._chat_messages(messages, structured=True)
            try:
                return content, _parse_model_output(content, result_type)
            except LLMOutputError:
                if attempt >= self.max_retries:
                    raise
                delay = self._retry_delay(attempt)
                logger.warning(
                    "llm_output_retry_scheduled",
                    extra={
                        "model": self.model,
                        "attempt": attempt + 1,
                        "result_type": result_type.__name__,
                        "delay_seconds": round(delay, 3),
                    },
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def _chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        structured: bool,
    ) -> str:
        return await self._chat_messages([
            ChatMessage(role="system", content=system_prompt), ChatMessage(role="user", content=user_prompt)
        ], structured=structured)

    async def _chat_messages(self, messages: list[ChatMessage], *, structured: bool) -> str:
        request = ChatCompletionRequest(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"} if structured else None,
            max_tokens=self.max_tokens,
        )
        # Check the exact serialized request before acquiring/sending HTTP.
        # This includes system/messages and response schema JSON.
        self.context_budget.check(self.context_budget.estimate_request(request))
        client = await self._get_client()
        headers = {"Accept": "application/json", **self._headers}
        if self.api_key:
            headers.setdefault("Authorization", f"Bearer {self.api_key}")
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                response = await client.post(
                    self.endpoint,
                    headers=headers,
                    json=request.model_dump(exclude_none=True),
                )
            except httpx.TransportError as error:
                if attempt >= self.max_retries:
                    logger.exception(
                        "llm_call_transport_failed",
                        extra={
                            "model": self.model,
                            "attempt": attempt + 1,
                            "duration_ms": round(
                                (time.perf_counter() - started) * 1000, 2
                            ),
                        },
                    )
                    raise LLMRequestError(f"LLM request failed: {error}") from error
                delay = self._retry_delay(attempt)
                logger.warning(
                    "llm_call_retry_scheduled",
                    extra={
                        "model": self.model,
                        "attempt": attempt + 1,
                        "reason": type(error).__name__,
                        "delay_seconds": round(delay, 3),
                    },
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue

            logger.info(
                "llm_call_completed",
                extra={
                    "model": self.model,
                    "status": response.status_code,
                    "attempt": attempt + 1,
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000, 2
                    ),
                    "structured": structured,
                },
            )

            if not response.is_error:
                break
            if _retryable_status(response.status_code) and attempt < self.max_retries:
                delay = self._retry_delay(
                    attempt, _retry_after_seconds(response.headers.get("Retry-After"))
                )
                logger.warning(
                    "llm_call_retry_scheduled",
                    extra={
                        "model": self.model,
                        "attempt": attempt + 1,
                        "status": response.status_code,
                        "delay_seconds": round(delay, 3),
                    },
                )
                await asyncio.sleep(delay)
                attempt += 1
                continue
            detail = _response_detail(response)
            raise LLMRequestError(
                f"LLM request failed with HTTP {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
            completion = ChatCompletionResponse.model_validate(payload)
        except (ValueError, ValidationError) as error:
            raise LLMProtocolError("LLM returned an invalid chat-completion response") from error
        message = completion.choices[0].message
        return _message_content(message.content)

    def _retry_delay(self, attempt: int, retry_after: float | None = None) -> float:
        exponential = min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2**attempt),
        )
        jittered = random.uniform(exponential / 2, exponential)
        if retry_after is None:
            return jittered
        return max(jittered, min(retry_after, self.retry_max_seconds))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the owned HTTP client; injected clients remain caller-owned."""

        client, self._client = self._client, None
        if client is not None and self._owns_client:
            await client.aclose()

    async def close(self) -> None:
        await self.aclose()

    async def __aenter__(self) -> "OpenAICompatibleAdapter":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


def _message_content(content: str | list[Any] | None) -> str:
    if isinstance(content, str):
        if not content.strip():
            raise LLMProtocolError("LLM returned an empty assistant message")
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        result = "".join(text_parts)
        if result.strip():
            return result
    raise LLMProtocolError("LLM response did not contain assistant text")


_ResultT = TypeVar("_ResultT", bound=BaseModel)


def _parse_model_output(content: str, result_type: type[_ResultT]) -> _ResultT:
    """Parse JSON output, accepting the common fenced-JSON variant."""

    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json\n"):
            text = text[5:].lstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMOutputError("LLM structured output was not valid JSON") from error
    try:
        return result_type.model_validate(value)
    except ValidationError as error:
        raise LLMOutputError(
            f"LLM structured output did not match {result_type.__name__}"
        ) from error


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        detail = response.text
    else:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                detail = error.get("message") or json.dumps(error, ensure_ascii=False)
            else:
                detail = payload.get("message") or json.dumps(payload, ensure_ascii=False)
        else:
            detail = json.dumps(payload, ensure_ascii=False)
    return str(detail).strip()[:1000] or "no response body"


def _retryable_status(status_code: int) -> bool:
    return status_code in {408, 429} or 500 <= status_code <= 599


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, seconds)


def create_llm(settings: Settings) -> LlmModel:
    """Build the model Adapter from validated process settings."""

    return OpenAICompatibleAdapter.from_settings(settings)


__all__ = [
    "LLMModel",
    "LLMError",
    "LLMOutputError",
    "LLMProtocolError",
    "LLMRequestError",
    "LlmModel",
    "OpenAICompatibleAdapter",
    "create_llm",
]
