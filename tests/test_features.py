from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime

import httpx
import pytest
from pydantic import ValidationError

from gossipmemo import world as world_module
from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.context_budget import ContextBudget
from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.models import (
    ExtractedClarification,
    ExtractedClarificationResult,
    ExtractedMemory,
    ExtractedPerson,
    ExtractedRelationship,
    ExtractionResult,
    ManualMemoryRequest,
    MemoryView,
    MessageInput,
    ModelMessage,
    PersonView,
    QueryContext,
    QueryRequest,
    SourceRef,
    SupersedeRequest,
    TurnRequest,
    UserModelView,
)
from gossipmemo.query import synthesize
from gossipmemo.reasoners import ReasoningSettings
from gossipmemo.reasoners.extraction import _extract
from gossipmemo.reasoners.person import _reason_person
from gossipmemo.reasoners.user_model import _reason_user_model
from gossipmemo.store import SqliteWorldStore
from gossipmemo.transport import (
    ChatCompletionRequest,
    LLMRequestError,
    ProviderGate,
    RetryPolicy,
)
from gossipmemo.world import SocialMemoryWorld
from gossipmemo_client import AsyncGossipMemo, GossipMemo

REASONING = ReasoningSettings()


class FakeModel:
    """An `LlmTransport` double: extraction, continuity, and
    person/relationship/user_model/coverage/learning_goals reasoning all
    drive `prepare`/`complete` directly (see `reasoners/extraction.py`,
    `reasoners/continuity.py`, `reasoners/owner.py` and friends). Every
    stage is told apart by a prompt marker rather than a typed method.

    `_owner_response`'s default fallback -- an object whose fields are all
    unrelated to any reasoner's schema -- parses successfully against every
    result type used here (`ExtractionResult`, `ExtractedCoverageAudit`,
    `GoalPlanningResult`, `GoalPlanningCandidates`,
    `RelationshipProjectionResult`) because every field on each of them has
    a default and pydantic ignores unknown keys; it only needs a real
    branch for stages whose schema requires a `profile_card` or that must
    observe an actions-stage completion.
    """

    configured = True
    gate = ProviderGate()
    context_budget = ContextBudget()
    retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)

    async def aclose(self):
        return None

    def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="fake",
            messages=list(messages),
            response_format={"type": "json_object"} if structured else None,
        )

    async def complete(self, request: ChatCompletionRequest) -> str:
        return self._owner_response(request)

    @staticmethod
    def _owner_response(request: ChatCompletionRequest) -> str:
        combined = " ".join(str(message.content) for message in request.messages)
        if "Rebuild compact cross-session continuity." in combined:
            ids = re.findall(r'"id": "([^"]*)"', combined)
            return json.dumps({"through_message_id": ids[-1] if ids else ""})
        if "Review the projection above" in combined:
            return json.dumps({})
        if '"profile_card"' in combined:
            return json.dumps({"profile_card": {}})
        return json.dumps(
            {"facets": [], "closeness": None, "tone": None, "status": "unknown", "summary": ""}
        )


def _settings(database_path, *, api_key: str = "") -> Settings:
    return Settings(
        database_path=database_path,
        api_key=api_key,
        llm_base_url="http://llm.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )


def _store(tmp_path) -> SqliteWorldStore:
    store = SqliteWorldStore(tmp_path / "features.db")
    store.initialize()
    return store


def test_query_uses_fts_but_keeps_structural_fallback(tmp_path):
    store = _store(tmp_path)
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Alice plans a distinctive sabbatical in October.",
            people=["Alice"],
        ),
    )
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Alice prefers green tea.",
            people=["Alice"],
        ),
    )

    relevant = store.read(
        "personal",
        QueryRequest(question="When is Alice's sabbatical?", people=["Alice"]),
    )
    assert [memory.content for memory in relevant.memories] == [
        "Alice plans a distinctive sabbatical in October."
    ]

    fallback = store.read(
        "personal",
        QueryRequest(question="What else should I remember?", people=["Alice"]),
    )
    assert len(fallback.memories) == 2


def test_one_hop_expansion_returns_neighbor_and_relationship_memory(tmp_path):
    store = _store(tmp_path)
    receipt = store.record_messages(
        "personal",
        [
            MessageInput(
                author="user",
                content="Alice and Bob work together.",
                source=SourceRef(provider="test", item_id="relationship-1"),
            )
        ],
    )[0]
    batch_id = store.create_extraction_batch("personal", [receipt])
    assert batch_id is not None
    store.apply_extraction(
        "personal",
        batch_id,
        ExtractionResult(
            people=[
                ExtractedPerson(ref="alice", display_name="Alice"),
                ExtractedPerson(ref="bob", display_name="Bob"),
            ],
            memories=[
                ExtractedMemory(
                    content="Alice and Bob work together.",
                    basis="stated",
                    relationships=[
                        ExtractedRelationship(
                            person_a_ref="alice",
                            person_b_ref="bob",
                            facets=[{"kind": "coworker"}],
                        )
                    ],
                )
            ],
        ),
    )

    context = store.read(
        "personal",
        QueryRequest(
            question="Who works together?",
            people=["Alice"],
            expand_relationships=1,
        ),
    )
    assert {person.display_name for person in context.people} == {"Alice", "Bob"}
    assert len(context.relationships) == 1
    assert context.memories[0].content == "Alice and Bob work together."


def test_supersede_preserves_history_and_retract_reason(tmp_path):
    store = _store(tmp_path)
    original_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob prefers coffee.",
            people=["Bob"],
        ),
    )
    replacement_id = store.supersede_memory(
        "personal",
        original_id,
        SupersedeRequest(
            content="Bob now prefers tea.",
            reason="Bob corrected me.",
        ),
    )
    assert replacement_id

    with store._connect() as connection:
        original = connection.execute(
            "SELECT status, invalidation_reason FROM memories WHERE id = ?",
            (original_id,),
        ).fetchone()
        replacement = connection.execute(
            "SELECT supersedes_memory_id FROM memories WHERE id = ?",
            (replacement_id,),
        ).fetchone()
    assert dict(original) == {
        "status": "superseded",
        "invalidation_reason": "Bob corrected me.",
    }
    assert replacement["supersedes_memory_id"] == original_id
    assert store.retract_memory("personal", replacement_id, "No longer reliable")
    with store._connect() as connection:
        reason = connection.execute(
            "SELECT invalidation_reason FROM memories WHERE id = ?",
            (replacement_id,),
        ).fetchone()[0]
    assert reason == "No longer reliable"


def test_supersede_inherits_or_overrides_about_user(tmp_path):
    store = _store(tmp_path)
    original = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="I like tea.", about_user=True)
    )
    inherited = store.supersede_memory(
        "personal", original, SupersedeRequest(content="I like green tea.")
    )
    assert inherited
    overridden = store.supersede_memory(
        "personal", inherited, SupersedeRequest(content="This is about Bob.", about_user=False)
    )
    assert overridden
    with store._connect() as connection:
        rows = connection.execute(
            "SELECT about_user FROM memories WHERE id IN (?, ?) ORDER BY created_at",
            (inherited, overridden),
        ).fetchall()
    assert [row["about_user"] for row in rows] == [1, 0]


def test_message_time_requires_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        MessageInput(
            author="user",
            content="A message without a timezone.",
            occurred_at=datetime(2026, 8, 9, 12, 0),
        )


def _extraction_batch_size(combined: str) -> int:
    """Count messages in the current extraction batch (evidence plus assistant context)."""

    section = combined.split(
        "Current batch evidence (user-authored; the only messages allowed to produce memories):",
        1,
    )[1]
    section = section.split(
        "Comparison memories (deduplication/update reference only; never new evidence):", 1
    )[0]
    return len(re.findall(r'"id": "', section))


def _extraction_first_content(combined: str) -> str:
    """The first evidence message's content in an extraction request."""

    section = combined.split(
        "Current batch evidence (user-authored; the only messages allowed to produce memories):",
        1,
    )[1]
    match = re.search(r'"content": "([^"]*)"', section)
    return match.group(1) if match else ""


def test_extraction_batches_default_to_six_messages(tmp_path):
    calls: list[int] = []

    class BatchModel(FakeModel):
        async def complete(self, request: ChatCompletionRequest) -> str:
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                calls.append(_extraction_batch_size(combined))
                return json.dumps({})
            return self._owner_response(request)

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(
            store, BatchModel(), extraction_batch_timeout_seconds=0.001
        )
        await world.start()
        try:
            await world.turn(
                "personal",
                TurnRequest(
                    messages=[
                        MessageInput(author="user", content=f"message {index}")
                        for index in range(13)
                    ]
                ),
            )
            for _ in range(100):
                if len(calls) == 3:
                    break
                await asyncio.sleep(0.001)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == [6, 6, 1]


def test_partial_extraction_batch_waits_until_full(tmp_path):
    calls: list[int] = []

    class BatchModel(FakeModel):
        async def complete(self, request: ChatCompletionRequest) -> str:
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                calls.append(_extraction_batch_size(combined))
                return json.dumps({})
            return self._owner_response(request)

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(
            store, BatchModel(), extraction_batch_timeout_seconds=60
        )
        await world.start()
        try:
            await world.turn(
                "personal",
                TurnRequest(
                    messages=[
                        MessageInput(author="user", content=f"message {index}")
                        for index in range(5)
                    ]
                ),
            )
            await asyncio.sleep(0)
            assert calls == []

            await world.turn(
                "personal",
                TurnRequest(
                    messages=[MessageInput(author="assistant", content="message 5")]
                ),
            )
            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == [6]


def _make_batch(store: SqliteWorldStore, space_id: str, content: str) -> str:
    message_id = store.record_messages(
        space_id,
        [MessageInput(author="user", content=content)],
    )[0]
    batch_id = store.create_extraction_batch(space_id, [message_id])
    assert batch_id is not None
    return batch_id


def test_failing_batch_does_not_block_later_batch(tmp_path):
    from gossipmemo.reasoners.extraction import build_extraction_reasoner

    store = _store(tmp_path)
    bad_batch = _make_batch(store, "personal", "bad batch content")
    good_batch = _make_batch(store, "personal", "good batch content")

    class FlakyModel(FakeModel):
        async def complete(self, request: ChatCompletionRequest) -> str:
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                if "bad batch" in _extraction_first_content(combined):
                    raise RuntimeError("boom")
                return json.dumps({})
            return self._owner_response(request)

    reasoner = build_extraction_reasoner(store, FlakyModel(), ReasoningSettings())

    asyncio.run(reasoner.run_until_caught_up("personal"))

    with store._connect() as connection:
        bad_state = connection.execute(
            "SELECT extraction_state FROM messages WHERE extraction_batch_id = ?",
            (bad_batch,),
        ).fetchone()["extraction_state"]
        good_state = connection.execute(
            "SELECT extraction_state FROM messages WHERE extraction_batch_id = ?",
            (good_batch,),
        ).fetchone()["extraction_state"]
    assert bad_state == "failed"
    assert good_state == "completed"


def test_two_permanently_failing_batches_terminate_drain(tmp_path):
    from gossipmemo.reasoners.extraction import (
        MAX_EXTRACTION_ATTEMPTS,
        build_extraction_reasoner,
    )

    store = _store(tmp_path)
    _make_batch(store, "personal", "first batch content")
    _make_batch(store, "personal", "second batch content")

    calls: list[str] = []

    class AlwaysFailsModel(FakeModel):
        async def complete(self, request: ChatCompletionRequest) -> str:
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                calls.append(_extraction_first_content(combined))
                raise RuntimeError("boom")
            return self._owner_response(request)

    reasoner = build_extraction_reasoner(store, AlwaysFailsModel(), ReasoningSettings())

    guard = 0

    def under_guard() -> bool:
        # Fail loudly instead of hanging if the drain stops terminating.
        nonlocal guard
        guard += 1
        assert guard <= 100, "drain loop did not terminate"
        return True

    asyncio.run(reasoner.run_until_caught_up("personal", under_guard))

    assert len(calls) == 2 * MAX_EXTRACTION_ATTEMPTS


def test_capped_batch_is_skipped_but_still_reported(tmp_path):
    from gossipmemo.reasoners.extraction import (
        MAX_EXTRACTION_ATTEMPTS,
        build_extraction_reasoner,
    )

    store = _store(tmp_path)
    capped_batch = _make_batch(store, "personal", "capped batch content")
    fresh_batch = _make_batch(store, "personal", "fresh batch content")

    for _ in range(MAX_EXTRACTION_ATTEMPTS):
        store.mark_extraction_attempt("personal", capped_batch)
        store.fail_extraction("personal", capped_batch, "boom")

    calls: list[str] = []

    class RecordingModel(FakeModel):
        async def complete(self, request: ChatCompletionRequest) -> str:
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                calls.append(_extraction_first_content(combined))
                return json.dumps({})
            return self._owner_response(request)

    reasoner = build_extraction_reasoner(store, RecordingModel(), ReasoningSettings())

    asyncio.run(reasoner.run_until_caught_up("personal"))

    assert calls == ["fresh batch content"]
    reported = {
        pending.batch_id for pending in store.pending_extractions()
        if pending.space_id == "personal"
    }
    assert capped_batch in reported
    assert fresh_batch not in reported  # completed, so no longer pending


def test_batch_killed_mid_call_is_not_capped(tmp_path):
    """Attempts are counted before the call; only recorded failures retire a batch.

    A hard kill (SIGKILL past the stop grace period) leaves the count
    raised and the batch still 'pending'. Restarts must keep retrying it,
    or five unlucky kills would permanently drop healthy messages.
    """

    from gossipmemo.reasoners.extraction import (
        MAX_EXTRACTION_ATTEMPTS,
        build_extraction_reasoner,
    )

    store = _store(tmp_path)
    killed_batch = _make_batch(store, "personal", "killed batch content")

    for _ in range(MAX_EXTRACTION_ATTEMPTS + 2):
        # No fail_extraction: the process died before recording an outcome.
        store.mark_extraction_attempt("personal", killed_batch)

    killed = next(
        pending for pending in store.pending_extractions()
        if pending.batch_id == killed_batch
    )
    assert killed.attempts > MAX_EXTRACTION_ATTEMPTS and killed.state == "pending"

    calls: list[str] = []

    class RecordingModel(FakeModel):
        async def complete(self, request: ChatCompletionRequest) -> str:
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                calls.append(_extraction_first_content(combined))
                return json.dumps({})
            return self._owner_response(request)

    reasoner = build_extraction_reasoner(store, RecordingModel(), ReasoningSettings())
    asyncio.run(reasoner.run_until_caught_up("personal"))

    assert calls == ["killed batch content"]


def test_induction_waits_for_daily_scheduler(tmp_path):
    calls: list[str] = []

    class InductionModel(FakeModel):
        async def complete(self, request):
            combined = " ".join(str(message.content) for message in request.messages)
            if '"profile_card"' in combined and "Review the projection" not in combined:
                match = re.search(r'"display_name":\s*"([^"]*)"', combined)
                calls.append(match.group(1) if match else "")
            return self._owner_response(request)

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(
            store,
            InductionModel(),
            induction_interval_seconds=0.01,
        )
        await world.start()
        try:
            world.add_memory(
                "personal",
                ManualMemoryRequest(content="Bob likes tea.", people=["Bob"]),
            )
            await asyncio.sleep(0)
            assert calls == []

            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0.001)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == ["Bob"]


def test_induction_schedules_user_model_only_for_about_user_memory(tmp_path):
    calls: list[int] = []

    class UserModel(FakeModel):
        async def complete(self, request):
            combined = " ".join(str(message.content) for message in request.messages)
            if '"profile_card"' in combined and "Review the projection" not in combined:
                calls.append(combined.count("- id="))
            return self._owner_response(request)

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(store, UserModel(), induction_interval_seconds=0.01)
        await world.start()
        try:
            world.add_memory("personal", ManualMemoryRequest(content="ordinary fact"))
            await asyncio.sleep(0.03)
            assert calls == []
            world.add_memory(
                "personal", ManualMemoryRequest(content="I like tea", about_user=True)
            )
            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0.001)
        finally:
            await world.stop()
        assert calls == [1]

    asyncio.run(scenario())


def test_startup_catches_up_stale_profiles(tmp_path):
    calls: list[str] = []

    class InductionModel(FakeModel):
        async def complete(self, request):
            combined = " ".join(str(message.content) for message in request.messages)
            if '"profile_card"' in combined and "Review the projection" not in combined:
                match = re.search(r'"display_name":\s*"([^"]*)"', combined)
                calls.append(match.group(1) if match else "")
            return self._owner_response(request)

    async def scenario() -> None:
        store = _store(tmp_path)
        store.add_manual_memory(
            "personal", ManualMemoryRequest(content="Bob likes tea.", people=["Bob"])
        )
        world = SocialMemoryWorld(
            store,
            InductionModel(),
            induction_interval_seconds=60,
        )
        await world.start()
        try:
            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == ["Bob"]


def test_sync_and_async_sdk_follow_server_contract():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/turns"):
            return httpx.Response(
                202,
                json={"status": "accepted", "message_ids": ["message_1"]},
            )
        if request.url.path.endswith("/query"):
            return httpx.Response(
                200,
                json={"answer": "Tea", "people": [], "relationships": [], "memories": []},
            )
        raise AssertionError(request.url.path)

    transport = httpx.MockTransport(handler)
    with GossipMemo(
        "http://memory.test", api_key="secret", transport=transport
    ) as client:
        result = client.turn("Bob likes tea.")
        assert result == {"status": "accepted", "message_ids": ["message_1"]}
        assert client.query("What does Bob like?")["answer"] == "Tea"

    async def async_scenario() -> None:
        async with AsyncGossipMemo(
            "http://memory.test", api_key="secret", transport=transport
        ) as client:
            assert (await client.query("What does Bob like?"))["answer"] == "Tea"

    asyncio.run(async_scenario())
    assert requests
    assert all(request.headers["authorization"] == "Bearer secret" for request in requests)


def test_openai_compatible_adapter_validates_structured_output():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert ("Keep explicit durable information and a transient detail only when"
                in payload["messages"][1]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"people":[{"ref":"bob","display_name":"Bob"}],'
                                '"memories":[{"content":"Bob likes tea.",'
                                '"basis":"stated","people":[],'
                                '"relationships":[]}]}'
                            ),
                        }
                    }
                ]
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1", "key", "test-model", client=client
            )
            result = await _extract(
                adapter,
                REASONING,
                [ModelMessage(
                    id="message_1",
                    space_id="personal",
                    author="user",
                    content="Bob likes tea.",
                    occurred_at="2026-08-09T12:00:00+00:00",
                    source_provider="test",
                )]
            )
            assert result.people[0].display_name == "Bob"
            assert result.memories[0].basis == "stated"

    asyncio.run(scenario())


def _clarification_message() -> ModelMessage:
    return ModelMessage(
        id="message_1",
        space_id="personal",
        author="user",
        content="Bob likes tea.",
        occurred_at="2026-08-09T12:00:00+00:00",
        source_provider="test",
    )


class _SchemaCapturingModel(FakeModel):
    """Records the system prompt and returns a fixed structured response."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.system_content: str | None = None

    async def complete(self, request: ChatCompletionRequest) -> str:
        self.system_content = str(request.messages[0].content)
        return json.dumps(self.response)


def test_extract_requests_the_clarification_schema_only_when_probing():
    off = _SchemaCapturingModel({"people": [], "memories": []})
    result = asyncio.run(_extract(off, REASONING, [_clarification_message()]))
    assert "clarifications" not in off.system_content
    assert type(result) is ExtractionResult

    on = _SchemaCapturingModel(
        {
            "people": [],
            "memories": [],
            "clarifications": [
                {
                    "question": "Which Bob?",
                    "reason": "Two people named Bob are known.",
                    "blocked_by": "ambiguous reference",
                    "evidence_message_ids": ["message_1"],
                }
            ],
        }
    )
    result = asyncio.run(
        _extract(on, ReasoningSettings(extraction_clarification_probe=True),
                 [_clarification_message()])
    )
    assert "clarifications" in on.system_content
    assert isinstance(result, ExtractedClarificationResult)
    assert result.clarifications[0].question == "Which Bob?"


def test_clarifications_never_change_what_apply_extraction_persists(tmp_path):
    """The probe is observation-only: a result carrying clarifications must
    persist exactly the same people/memories as one without."""

    store = _store(tmp_path / "a")
    message_id = store.record_messages(
        "personal", [MessageInput(author="user", content="Bob likes tea.")],
    )[0]
    plain_batch = store.create_extraction_batch("personal", [message_id])
    assert plain_batch is not None

    base_kwargs = dict(
        people=[ExtractedPerson(ref="bob", display_name="Bob")],
        memories=[
            ExtractedMemory(content="Bob likes tea.", basis="stated", people=["bob"])
        ],
    )
    store.apply_extraction("personal", plain_batch, ExtractionResult(**base_kwargs), set())
    with store._connect() as connection:
        plain_memories = sorted(
            row["content"] for row in connection.execute(
                "SELECT content FROM memories WHERE space_id = ?", ("personal",)
            ).fetchall()
        )

    store2 = _store(tmp_path / "b")
    message_id_2 = store2.record_messages(
        "personal", [MessageInput(author="user", content="Bob likes tea.")],
    )[0]
    annotated_batch = store2.create_extraction_batch("personal", [message_id_2])
    assert annotated_batch is not None
    annotated_result = ExtractedClarificationResult(
        **base_kwargs,
        clarifications=[
            ExtractedClarification(
                question="Which Bob?",
                reason="Two people named Bob are known.",
                blocked_by="ambiguous reference",
                evidence_message_ids=[message_id_2],
            )
        ],
    )
    store2.apply_extraction("personal", annotated_batch, annotated_result, set())
    with store2._connect() as connection:
        annotated_memories = sorted(
            row["content"] for row in connection.execute(
                "SELECT content FROM memories WHERE space_id = ?", ("personal",)
            ).fetchall()
        )

    assert plain_memories == annotated_memories


def test_extraction_clarification_log_fields_survive_structured_formatter():
    from gossipmemo.logging import StructuredFormatter

    record = logging.getLogger("test").makeRecord(
        "test", logging.INFO, __file__, 1, "extraction_clarification", (), None,
        extra={
            "space_id": "personal",
            "batch_id": "batch_1",
            "question_text": "Which Bob is this?",
            "reason_text": "Two people named Bob are known.",
            "blocked_by": "ambiguous reference",
            "evidence_message_ids": ["message_1"],
        },
    )
    payload = json.loads(StructuredFormatter().format(record))

    assert payload["question_text"] == "Which Bob is this?"
    assert payload["reason_text"] == "Two people named Bob are known."
    assert payload["blocked_by"] == "ambiguous reference"
    assert payload["evidence_message_ids"] == ["message_1"]


def test_owner_reasoning_uses_an_identical_prefix_for_both_stages():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        content = '{"profile_card":{"summary":"tea"}}' if len(
            payloads) == 1 else '{"hypothesis_actions":{"upserts":[],"transitions":[]}}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1", "key", "test-model", client=client)
            result = await _reason_person(
                adapter,
                REASONING,
                PersonView(id="person_1", display_name="Bob"),
                [MemoryView(id="memory_1", content="Bob likes tea.", kind="fact",
                            basis="stated", status="active",
                            created_at="2026-08-01T00:00:00+00:00")],
            )
            assert result.profile_card == {"summary": "tea"}

    asyncio.run(scenario())
    first, second = (payload["messages"] for payload in payloads)
    assert second[:len(first)] == first
    assert second[len(first)]["role"] == "assistant"
    assert "<evidence-memories>" in first[1]["content"]
    assert "comparison-only" in first[1]["content"]
    assert "inferred_memory_actions" in second[-1]["content"]


def test_user_owner_review_schema_excludes_inferred_memory_actions():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        content = '{"profile_card":{}}' if len(
            payloads) == 1 else '{"hypothesis_actions":{"upserts":[],"transitions":[]}}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1", "key", "test-model", client=client)
            await _reason_user_model(
                adapter, REASONING, UserModelView(space_id="personal"), [])

    asyncio.run(scenario())
    assert "hypothesis_actions" in payloads[1]["messages"][-1]["content"]
    assert "inferred_memory_actions" not in payloads[1]["messages"][-1]["content"]


def test_openai_compatible_adapter_retries_transient_statuses(monkeypatch):
    statuses = iter([429, 503, 200])
    requests = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        status = next(statuses)
        if status == 200:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Recovered"}}]},
            )
        return httpx.Response(
            status,
            headers={"Retry-After": "7"} if status == 429 else {},
            json={"error": {"message": "temporarily overloaded"}},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("gossipmemo.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "gossipmemo.transport.random.uniform", lambda _lower, upper: upper
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1",
                "key",
                "test-model",
                client=client,
                max_retries=2,
                retry_base_seconds=2,
                retry_max_seconds=10,
            )
            answer = await synthesize(adapter, "Question", QueryContext(), REASONING.prompts)
            assert answer == "Recovered"

    asyncio.run(scenario())
    assert requests == 3
    assert delays == [7, 4]


def test_openai_compatible_adapter_does_not_retry_permanent_status(monkeypatch):
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    async def unexpected_sleep(delay: float) -> None:
        raise AssertionError(f"unexpected retry delay: {delay}")

    monkeypatch.setattr("gossipmemo.llm.asyncio.sleep", unexpected_sleep)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1", "key", "test-model", client=client
            )
            with pytest.raises(LLMRequestError, match="HTTP 401: bad key"):
                await synthesize(adapter, "Question", QueryContext(), REASONING.prompts)

    asyncio.run(scenario())
    assert requests == 1


def test_openai_compatible_adapter_retries_transport_errors(monkeypatch):
    requests = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx.ConnectError("model is restarting", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Recovered"}}]},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("gossipmemo.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "gossipmemo.transport.random.uniform", lambda _lower, upper: upper
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1",
                "key",
                "test-model",
                client=client,
                max_retries=1,
                retry_base_seconds=3,
                retry_max_seconds=10,
            )
            answer = await synthesize(adapter, "Question", QueryContext(), REASONING.prompts)
            assert answer == "Recovered"

    asyncio.run(scenario())
    assert requests == 2
    assert delays == [3]


def test_http_auth_and_correction_endpoints(tmp_path):
    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(store, FakeModel())
        app = create_app(
            settings=_settings(tmp_path / "features.db", api_key="secret"),
            world=world,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://server.test"
            ) as client:
                denied = await client.post(
                    "/v1/spaces/personal/memories",
                    json={"content": "Bob prefers coffee."},
                )
                assert denied.status_code == 401
                headers = {"Authorization": "Bearer secret"}
                created = await client.post(
                    "/v1/spaces/personal/memories",
                    headers=headers,
                    json={
                        "content": "Bob prefers coffee.",
                        "people": ["Bob"],
                    },
                )
                memory_id = created.json()["id"]
                corrected = await client.post(
                    f"/v1/spaces/personal/memories/{memory_id}/supersede",
                    headers=headers,
                    json={"content": "Bob now prefers tea.", "reason": "Correction"},
                )
                assert corrected.status_code == 201
                replacement_id = corrected.json()["id"]
                retracted = await client.post(
                    f"/v1/spaces/personal/memories/{replacement_id}/retract",
                    headers=headers,
                    json={"reason": "Uncertain"},
                )
                assert retracted.json()["status"] == "retracted"

    asyncio.run(scenario())


def test_http_list_people_route_coexists_with_person_dossier_route(tmp_path):
    async def scenario() -> None:
        store = _store(tmp_path)
        store.add_manual_memory(
            "personal", ManualMemoryRequest(content="x", people=["Alice Wang"])
        )
        person_id = store.list_people("personal")[0].id
        world = SocialMemoryWorld(store, FakeModel())
        app = create_app(
            settings=_settings(tmp_path / "features.db", api_key="secret"),
            world=world,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://server.test"
            ) as client:
                denied = await client.get("/v1/spaces/personal/people")
                assert denied.status_code == 401

                headers = {"Authorization": "Bearer secret"}
                listed = await client.get(
                    "/v1/spaces/personal/people", headers=headers
                )
                assert listed.status_code == 200
                assert [p["display_name"] for p in listed.json()["people"]] == ["Alice Wang"]

                searched = await client.get(
                    "/v1/spaces/personal/people",
                    headers=headers,
                    params={"q": "alice", "limit": 5},
                )
                assert [p["display_name"] for p in searched.json()["people"]] == ["Alice Wang"]

                no_match = await client.get(
                    "/v1/spaces/personal/people",
                    headers=headers,
                    params={"q": "nobody"},
                )
                assert no_match.json()["people"] == []

                dossier = await client.get(
                    f"/v1/spaces/personal/people/{person_id}", headers=headers
                )
                assert dossier.status_code == 200
                assert dossier.json()["people"][0]["id"] == person_id

    asyncio.run(scenario())


def _freeze_local_now(monkeypatch, moment: datetime) -> None:
    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(world_module, "datetime", FrozenClock)


def test_next_induction_delay_follows_configured_time_of_day(tmp_path, monkeypatch):
    world = SocialMemoryWorld(_store(tmp_path), FakeModel(), induction_time="09:30")

    _freeze_local_now(monkeypatch, datetime(2026, 8, 19, 8, 0))
    assert world._next_induction_delay() == 90 * 60

    _freeze_local_now(monkeypatch, datetime(2026, 8, 19, 10, 0))
    assert world._next_induction_delay() == 23.5 * 3600

    world.induction_interval_seconds = 42.0
    assert world._next_induction_delay() == 42.0
