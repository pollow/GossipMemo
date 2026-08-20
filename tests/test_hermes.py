from __future__ import annotations

import threading
import time

from integrations.hermes.gossipmemo import GossipMemoMemoryProvider


def test_hermes_provider_keeps_session_as_source_coordinate():
    ingested: list[list[dict]] = []
    received = threading.Event()

    class FakeClient:
        def turn(self, messages, **kwargs):
            ingested.append(messages)
            received.set()
            return {"status": "accepted"}

        def query(self, question, **kwargs):
            return {
                "answer": f"answer for {question}",
                "people": [],
                "relationships": [],
                "memories": [],
            }

        def add_memory(self, content, **kwargs):
            return {"id": "memory_1", "content": content}

        def retract(self, memory_id, **kwargs):
            return {"id": memory_id, "status": "retracted"}

        def recall(self, q, **kwargs):
            self.recall_calls.append((q, kwargs))
            return {"memories": [{"content": f"recalled for {q}"}]}

        def list_people(self, q="", **kwargs):
            self.list_people_calls.append((q, kwargs))
            return {"people": [{"id": "person_1", "display_name": f"match for {q}", "aliases": []}]}

        def guidance(self, **kwargs):
            self.guidance_calls.append(kwargs)
            return {"items": [{"id": "h1", "kind": "hypothesis", "content": "maybe true",
                              "owner_kind": "user", "status": "open"}]}

        recall_calls: list = []
        list_people_calls: list = []
        guidance_calls: list = []

        def close(self):
            return None

    provider = GossipMemoMemoryProvider(client_factory=lambda **_: FakeClient())
    provider.initialize("hermes-session-7", user_id="user-1")
    try:
        provider.sync_turn("Alice told me about Bob.", "I will remember that.")
        assert received.wait(1)
        assert len(ingested[0]) == 2
        user_message = ingested[0][0]
        assert user_message["author"] == "user"
        assert user_message["source"]["conversation_key"] == "hermes-session-7"
        assert "space" not in user_message["source"]

        result = provider.handle_tool_call(
            "gossipmemo_query", {"query": "Bob"}
        )
        assert "answer for Bob" in result

        recall_result = provider.handle_tool_call(
            "gossipmemo_recall",
            {"q": "tea", "about_user": True, "person_ids": ["p1"], "limit": 500},
        )
        assert "recalled for tea" in recall_result
        recalled_q, recalled_kwargs = provider._client.recall_calls[-1]
        assert recalled_q == "tea"
        assert recalled_kwargs["about_user"] is True
        assert recalled_kwargs["person_ids"] == ["p1"]
        assert recalled_kwargs["limit"] == 100

        empty_q_result = provider.handle_tool_call("gossipmemo_recall", {"q": "  "})
        assert "error" in empty_q_result

        people_result = provider.handle_tool_call(
            "gossipmemo_people", {"q": "alice", "limit": 500}
        )
        assert "match for alice" in people_result
        people_q, people_kwargs = provider._client.list_people_calls[-1]
        assert people_q == "alice"
        assert people_kwargs["limit"] == 200

        listing_result = provider.handle_tool_call("gossipmemo_people", {})
        assert "match for" in listing_result

        guidance_result = provider.handle_tool_call(
            "gossipmemo_guidance",
            {"person_ids": ["p1"], "kind": "hypothesis", "limit": 500},
        )
        assert "maybe true" in guidance_result
        guidance_kwargs = provider._client.guidance_calls[-1]
        assert guidance_kwargs["person_ids"] == ["p1"]
        assert guidance_kwargs["kind"] == "hypothesis"
        assert guidance_kwargs["limit"] == 200

        default_guidance_result = provider.handle_tool_call("gossipmemo_guidance", {})
        assert "maybe true" in default_guidance_result
        assert provider._client.guidance_calls[-1]["kind"] is None

        bad_kind_result = provider.handle_tool_call(
            "gossipmemo_guidance", {"kind": "nonsense"})
        assert "error" in bad_kind_result
    finally:
        provider.shutdown()


def test_hermes_tool_schemas_offer_recall_as_the_cheap_default():
    schemas = {schema["name"]: schema for schema in GossipMemoMemoryProvider().get_tool_schemas()}
    assert "gossipmemo_recall" in schemas
    assert "gossipmemo_query" in schemas["gossipmemo_recall"]["description"]
    assert "gossipmemo_recall" in schemas["gossipmemo_query"]["description"]


def test_hermes_guidance_tool_frames_hypotheses_and_goals_as_the_context_bundle_does():
    schemas = {schema["name"]: schema for schema in GossipMemoMemoryProvider().get_tool_schemas()}
    assert "gossipmemo_guidance" in schemas
    description = schemas["gossipmemo_guidance"]["description"].lower()
    assert "tentative" in description
    assert "optional" in description
    assert "not instructions" in description or "not a checklist" in description


def test_hermes_people_tool_precedes_merge_and_guardrail_survives():
    schemas = {schema["name"]: schema for schema in GossipMemoMemoryProvider().get_tool_schemas()}
    assert "gossipmemo_people" in schemas
    assert "duplicate" in schemas["gossipmemo_people"]["description"]
    merge_description = schemas["gossipmemo_merge_people"]["description"]
    assert "gossipmemo_people" in merge_description
    assert "only call this after the user" in merge_description.lower()


def test_hermes_frames_learning_goals_as_ignorable_rather_than_a_checklist():
    """Several goals at once read as an interview script without this framing.

    This also covers the item labels themselves: guidance renders as
    "Tentative hypothesis"/"Optional learning goal", never as a Memory.
    """
    formatted = GossipMemoMemoryProvider._format_context({
        "guidance": {"items": [
            {"id": "h", "kind": "hypothesis", "content": "Maybe true"},
            {"id": "g1", "kind": "learning_goal", "content": "First"},
            {"id": "g2", "kind": "learning_goal", "content": "Second"},
        ]}
    })
    lines = formatted.splitlines()
    note = next(line for line in lines if line.startswith("About the optional learning goals"))
    assert "Ignore them by default" in note
    assert "not questions to ask" in note
    assert "interview" in note
    # The note must precede the goals it frames, and must not appear when the
    # bundle carries hypotheses only.
    assert lines.index(note) < lines.index("Optional learning goal: First")
    assert lines.index("Tentative hypothesis: Maybe true") < lines.index(note)
    assert "Memory" not in formatted
    hypothesis_only = GossipMemoMemoryProvider._format_context({
        "guidance": {"items": [{"id": "h", "kind": "hypothesis", "content": "Maybe true"}]}
    })
    assert "About the optional learning goals" not in hypothesis_only


class _TurnClient:
    """A fake SDK client that records single-message turns and batch writes."""

    def __init__(self, *, turn_result=None, turn_error=None):
        self.turn_result = turn_result
        self.turn_error = turn_error
        self.turn_calls: list[tuple] = []
        self.ingested: list[list[dict]] = []
        self.turned = threading.Event()
        self.received = threading.Event()

    def turn(self, message, **kwargs):
        if isinstance(message, list):
            # The completed-turn write path posts a whole batch.
            self.ingested.append(message)
            self.received.set()
            return {"status": "accepted"}
        self.turn_calls.append((message, kwargs))
        try:
            if self.turn_error is not None:
                raise self.turn_error
            return self.turn_result
        finally:
            self.turned.set()

    def close(self):
        return None


def _wait_for(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_hermes_skips_resending_the_user_message_once_prefetch_persisted_it():
    client = _TurnClient(turn_result={"message_ids": ["message_1"], "known_people": []})
    provider = GossipMemoMemoryProvider(client_factory=lambda **_: client)
    provider.initialize("s-persisted")
    try:
        provider.queue_prefetch("Alice called.")
        assert _wait_for(
            lambda: provider._current_turn.get("s-persisted", {}).get("persisted"))
        provider.sync_turn("Alice called.", "Noted.")
        assert client.received.wait(2)
        batch = client.ingested[0]
        assert [message["author"] for message in batch] == ["assistant"]
        assert batch[0]["content"] == "Noted."
        assert "idempotency_key" not in batch[0]
        # The slot is released, so nothing accumulates across turns.
        assert provider._current_turn == {}
    finally:
        provider.shutdown()


def test_hermes_resends_the_user_message_with_its_key_when_prefetch_failed():
    client = _TurnClient(turn_error=RuntimeError("server down"))
    provider = GossipMemoMemoryProvider(client_factory=lambda **_: client)
    provider.initialize("s-failed")
    try:
        provider.queue_prefetch("Bob called.")
        assert client.turned.wait(2)
        assert _wait_for(lambda: "s-failed" in provider._current_turn)
        prefetch_key = provider._current_turn["s-failed"]["idempotency_key"]
        assert provider._current_turn["s-failed"]["persisted"] is False
        provider.sync_turn("Bob called.", "Noted.")
        assert client.received.wait(2)
        batch = client.ingested[0]
        assert [message["author"] for message in batch] == ["user", "assistant"]
        # The key is kept as a fuse: the failed call may still have committed.
        assert batch[0]["idempotency_key"] == prefetch_key
        assert batch[0]["idempotency_key"] == client.turn_calls[0][1]["idempotency_key"]
        assert provider._current_turn == {}
    finally:
        provider.shutdown()


def test_hermes_keeps_one_turn_slot_per_conversation():
    client = _TurnClient(turn_result={"message_ids": ["message_1"]})
    provider = GossipMemoMemoryProvider(client_factory=lambda **_: client)
    provider.initialize("s-slot")
    try:
        provider.queue_prefetch("First question.")
        assert _wait_for(lambda: len(client.turn_calls) == 1)
        provider.queue_prefetch("Second question.")
        assert _wait_for(lambda: len(client.turn_calls) == 2)
        assert list(provider._current_turn) == ["s-slot"]
        entry = provider._current_turn["s-slot"]
        assert entry["content"] == "Second question."
        assert isinstance(entry, dict)
        # An abandoned first turn leaves nothing behind to reclaim.
        provider.sync_turn("Second question.", "Noted.")
        assert client.received.wait(2)
        assert [message["author"] for message in client.ingested[0]] == ["assistant"]
        assert provider._current_turn == {}
    finally:
        provider.shutdown()
