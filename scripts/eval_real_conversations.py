#!/usr/bin/env python3
"""Run real-conversation fixtures through the full GossipMemo LLM workflow."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from gossipmemo.config import Settings
from gossipmemo.llm import create_llm
from gossipmemo.models import MessageInput, QueryRequest, SourceRef
from gossipmemo.store import SqliteWorldStore
from gossipmemo.world import SocialMemoryWorld


FIXTURE_RE = re.compile(r"^===== FIXTURE-(\d+): (.+) =====$")
MESSAGE_RE = re.compile(
    r"^--- message \d+/\d+ \| role=(USER|ASSISTANT) \| source_id=(\d+) ---$"
)


@dataclass
class FixtureMessage:
    role: str
    source_id: str
    content: str


@dataclass
class Fixture:
    number: str
    title: str
    kind: str
    source_session: str
    messages: list[FixtureMessage]


def parse_fixtures(path: Path) -> dict[str, Fixture]:
    lines = path.read_text(encoding="utf-8").splitlines()
    fixtures: dict[str, Fixture] = {}
    current: dict[str, Any] | None = None
    current_message: dict[str, str] | None = None

    def finish_message() -> None:
        nonlocal current_message
        if current is None or current_message is None:
            return
        current["messages"].append(
            FixtureMessage(
                role=current_message["role"].lower(),
                source_id=current_message["source_id"],
                content="\n".join(current_message["content"]).strip(),
            )
        )
        current_message = None

    def finish_fixture() -> None:
        nonlocal current
        finish_message()
        if current is None:
            return
        fixture = Fixture(
            number=current["number"],
            title=current["title"],
            kind=current.get("kind", ""),
            source_session=current.get("source_session", ""),
            messages=current["messages"],
        )
        fixtures[fixture.number] = fixture
        current = None

    for line in lines:
        fixture_match = FIXTURE_RE.match(line)
        if fixture_match:
            finish_fixture()
            current = {
                "number": fixture_match.group(1),
                "title": fixture_match.group(2),
                "messages": [],
            }
            continue
        message_match = MESSAGE_RE.match(line)
        if message_match and current is not None:
            finish_message()
            current_message = {
                "role": message_match.group(1),
                "source_id": message_match.group(2),
                "content": [],
            }
            continue
        if current_message is not None:
            current_message["content"].append(line)
        elif current is not None and line.startswith("kind:"):
            current["kind"] = line.split(":", 1)[1].strip()
        elif current is not None and line.startswith("source_session:"):
            current["source_session"] = line.split(":", 1)[1].strip()
    finish_fixture()
    return fixtures


def to_messages(fixture: Fixture) -> list[MessageInput]:
    session_prefix = fixture.source_session[:15]
    session_time = datetime.strptime(session_prefix, "%Y%m%d_%H%M%S").replace(
        tzinfo=timezone.utc
    )
    return [
        MessageInput(
            author=message.role,
            content=message.content,
            occurred_at=session_time + timedelta(seconds=int(message.source_id)),
            idempotency_key=f"eval:{fixture.number}:{message.source_id}",
            source=SourceRef(
                provider="gossipmemo_fixture",
                conversation_key=fixture.source_session,
                item_id=message.source_id,
                metadata={"fixture": fixture.number, "kind": fixture.kind},
            ),
        )
        for message in fixture.messages
    ]


def load_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def dump_state(store: SqliteWorldStore, space_id: str) -> dict[str, Any]:
    with store._connect() as connection:
        people_rows = connection.execute(
            "SELECT * FROM people WHERE space_id = ? ORDER BY display_name", (space_id,)
        ).fetchall()
        aliases = connection.execute(
            "SELECT person_id, value FROM person_aliases WHERE space_id = ? "
            "ORDER BY person_id, value",
            (space_id,),
        ).fetchall()
        aliases_by_person: dict[str, list[str]] = {}
        for alias in aliases:
            aliases_by_person.setdefault(alias["person_id"], []).append(alias["value"])
        people = [
            {
                "id": row["id"],
                "display_name": row["display_name"],
                "status": row["status"],
                "merged_into_person_id": row["merged_into_person_id"],
                "aliases": aliases_by_person.get(row["id"], []),
                "profile_card": load_json(row["profile_card"], {}),
                "profile_source_updated_at": row["profile_source_updated_at"],
            }
            for row in people_rows
        ]
        names = {row["id"]: row["display_name"] for row in people_rows}
        relationship_rows = connection.execute(
            "SELECT * FROM relationships WHERE space_id = ? ORDER BY created_at",
            (space_id,),
        ).fetchall()
        relationships = [
            {
                "id": row["id"],
                "people": [names.get(row["person_a_id"]), names.get(row["person_b_id"])],
                "facets": load_json(row["facets"], []),
                "closeness": row["closeness"],
                "tone": row["tone"],
                "status": row["status"],
                "summary": row["summary"],
                "profile_source_updated_at": row["profile_source_updated_at"],
            }
            for row in relationship_rows
        ]
        memory_rows = connection.execute(
            "SELECT * FROM memories WHERE space_id = ? ORDER BY created_at", (space_id,)
        ).fetchall()
        memories: list[dict[str, Any]] = []
        for row in memory_rows:
            memory_people = connection.execute(
                "SELECT p.display_name FROM memory_people mp "
                "JOIN people p ON p.id = mp.person_id WHERE mp.memory_id = ? "
                "ORDER BY p.display_name",
                (row["id"],),
            ).fetchall()
            evidence = connection.execute(
                "SELECT author, source_item_id, occurred_at FROM messages "
                "WHERE extraction_batch_id = ? ORDER BY rowid",
                (row["source_batch_id"],),
            ).fetchall()
            sources = connection.execute(
                "SELECT source_memory_id FROM memory_derivations "
                "WHERE derived_memory_id = ? ORDER BY source_memory_id",
                (row["id"],),
            ).fetchall()
            memories.append(
                {
                    "id": row["id"],
                    "content": row["content"],
                    "kind": row["kind"],
                    "basis": row["basis"],
                    "status": row["status"],
                    "about_user": bool(row["about_user"]),
                    "valid_from": row["valid_from"],
                    "valid_to": row["valid_to"],
                    "people": [item["display_name"] for item in memory_people],
                    "evidence": [dict(item) for item in evidence],
                    "source_memory_ids": [item["source_memory_id"] for item in sources],
                }
            )
        user_row = connection.execute(
            "SELECT * FROM user_models WHERE space_id = ?", (space_id,)
        ).fetchone()
        continuity_row = connection.execute(
            "SELECT * FROM continuities WHERE space_id = ?", (space_id,)
        ).fetchone()
        message_count = connection.execute(
            "SELECT COUNT(*) AS count FROM messages WHERE space_id = ?", (space_id,)
        ).fetchone()["count"]
    return {
        "message_count": message_count,
        "people": people,
        "relationships": relationships,
        "memories": memories,
        "basis_counts": {
            basis: sum(memory["basis"] == basis for memory in memories)
            for basis in sorted({memory["basis"] for memory in memories})
        },
        "user_model": load_json(user_row["profile_card"], {}) if user_row else {},
        "continuity": {
            "text": continuity_row["text"],
            "related_person_ids": load_json(
                continuity_row["related_person_ids"], []
            ),
            "through_message_id": continuity_row["through_message_id"],
        }
        if continuity_row
        else None,
    }


CASES = {
    "fixture-01": {
        "stages": [["01"]],
        "people": ["Person_A", "Person_B", "Person_C"],
        "questions": [
            "请区分事实、转述和推测：Person_A 与 Person_B、Person_C 当前分别是什么关系？哪些仍不确定？",
            "基于现有证据，Person_A 可能有哪些关系需求或行为模式？用户适合给他什么建议？",
        ],
    },
    "fixture-02": {
        "stages": [["02"]],
        "people": ["Person_A", "Person_C"],
        "questions": [
            "Person_A 和新女友是否已经分手并且仍然同居？这个说法的来源和确定性是什么？",
        ],
    },
    "fixture-03": {
        "stages": [["03"]],
        "people": ["Person_D", "Person_E"],
        "questions": [
            "目前可以确认的组织和晋升信息有哪些？哪些只是传闻或假设？对用户本人有什么已知影响？",
        ],
    },
    "fixture-02-then-01": {
        "stages": [["02"], ["01"]],
        "people": ["Person_A", "Person_B", "Person_C"],
        "questions": [
            "综合两次对话，Person_A 与 Person_B 是否已经分手或同居？哪些早期传闻被后来的信息纠正？",
        ],
    },
}


async def run_case(
    name: str,
    definition: dict[str, Any],
    fixtures: dict[str, Fixture],
    settings: Settings,
    root: Path,
) -> dict[str, Any]:
    database_path = root / f"{name}.db"
    store = SqliteWorldStore(database_path)
    world = SocialMemoryWorld(
        store,
        create_llm(settings),
        extraction_batch_size=settings.extraction_batch_size,
        extraction_batch_timeout_seconds=settings.extraction_batch_timeout_seconds,
    )
    space_id = name
    imports = []
    await world.start()
    try:
        for stage in definition["stages"]:
            messages = [
                message
                for fixture_number in stage
                for message in to_messages(fixtures[fixture_number])
            ]
            messages.sort(key=lambda item: item.occurred_at)
            imports.append(await world.import_messages(space_id, messages))
        state = dump_state(store, space_id)
        active_names = {
            person["display_name"]
            for person in state["people"]
            if person["status"] == "active"
        }
        query_people = [
            person for person in definition["people"] if person in active_names
        ]
        queries = []
        for question in definition["questions"]:
            response = await world.query(
                space_id,
                QueryRequest(
                    question=question,
                    people=query_people,
                    include_relationships=True,
                    expand_relationships=1,
                    include_evidence=True,
                    limit=100,
                ),
            )
            queries.append(
                {
                    "question": question,
                    "people_filter": query_people,
                    "answer": response.answer,
                    "memory_ids": [memory.id for memory in response.memories],
                    "relationship_ids": [
                        relationship.id for relationship in response.relationships
                    ],
                }
            )
        return {"imports": imports, "state": state, "queries": queries}
    finally:
        await world.stop()


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run text fixtures through extraction, projection induction, and "
            "query synthesis using an isolated SQLite database."
        )
    )
    parser.add_argument(
        "fixture_file",
        nargs="?",
        type=Path,
        default=Path("gossipmemo_test_data.txt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/gossipmemo-real-eval-report.json"),
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES),
        dest="cases",
        help="Run one case; repeat to run several. The default runs every case.",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        help="Keep isolated case databases here instead of a temporary directory.",
    )
    args = parser.parse_args()
    fixtures = parse_fixtures(args.fixture_file)
    settings = Settings.from_env()
    root = args.database_dir or Path(
        tempfile.mkdtemp(prefix="gossipmemo-real-eval-")
    )
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    selected_cases = args.cases or list(CASES)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": settings.llm_model,
        "extraction_policy": settings.extraction_policy,
        "fixture_counts": {
            number: len(fixture.messages) for number, fixture in fixtures.items()
        },
        "temp_directory": str(root),
        "cases": {},
    }
    for name in selected_cases:
        definition = CASES[name]
        print(f"RUN {name}", flush=True)
        try:
            report["cases"][name] = await run_case(
                name, definition, fixtures, settings, root
            )
            state = report["cases"][name]["state"]
            print(
                f"PASS {name}: {len(state['memories'])} memories, "
                f"{len(state['people'])} people, "
                f"{len(state['relationships'])} relationships",
                flush=True,
            )
        except Exception as error:
            report["cases"][name] = {
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(f"FAIL {name}: {type(error).__name__}: {error}", flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    args.report.chmod(0o600)
    for database_path in root.glob("*.db"):
        database_path.chmod(0o600)
    print(f"REPORT {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
