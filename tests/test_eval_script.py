from datetime import timezone

from scripts.eval_real_conversations import parse_fixtures, to_messages


def test_real_conversation_fixture_parser_preserves_roles_and_order(tmp_path):
    fixture_file = tmp_path / "fixtures.txt"
    fixture_file.write_text(
        """GossipMemo test fixtures

===== FIXTURE-01: Example =====
kind: example
source_session: 20260806_134429_session
message_count: 2

--- message 1/2 | role=USER | source_id=10 ---
First message.

--- message 2/2 | role=ASSISTANT | source_id=11 ---
Second message.
""",
        encoding="utf-8",
    )

    fixtures = parse_fixtures(fixture_file)
    fixture = fixtures["01"]
    assert fixture.kind == "example"
    assert [item.role for item in fixture.messages] == ["user", "assistant"]
    assert [item.content for item in fixture.messages] == [
        "First message.",
        "Second message.",
    ]

    messages = to_messages(fixture)
    assert messages[0].occurred_at.tzinfo == timezone.utc
    assert messages[0].occurred_at < messages[1].occurred_at
    assert messages[0].source.conversation_key == "20260806_134429_session"
    assert messages[1].source.item_id == "11"
