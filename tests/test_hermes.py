from __future__ import annotations

import threading

from integrations.hermes.gossipmemo import GossipMemoMemoryProvider


def test_hermes_provider_keeps_session_as_source_coordinate():
    ingested: list[list[dict]] = []
    received = threading.Event()

    class FakeClient:
        def ingest(self, messages):
            ingested.append(messages)
            received.set()

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

        def close(self):
            return None

    provider = GossipMemoMemoryProvider(
        client_factory=lambda **_: FakeClient())
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
            "gossipmemo_search", {"query": "Bob"}
        )
        assert "answer for Bob" in result
    finally:
        provider.shutdown()


def test_hermes_labels_guidance_as_tentative_or_optional():
    formatted = GossipMemoMemoryProvider._format_context({
        "guidance": {"items": [
            {"id": "h", "kind": "hypothesis", "content": "Maybe true"},
            {"id": "g", "kind": "learning_goal", "content": "Learn more"},
        ]}
    })
    assert "Tentative hypothesis: Maybe true" in formatted
    assert "Optional learning goal: Learn more" in formatted
    assert "Memory" not in formatted
