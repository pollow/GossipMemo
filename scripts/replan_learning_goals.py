#!/usr/bin/env python3
"""Run only the learning-goals reasoner against a database copy.

For A/B experiments on coverage-entry wording (or any other question that
only needs the planner re-run): copy the database, mutate one side, clear
`learning_goals` in both so each side plans from zero, then run this script
against each copy with `GOSSIPMEMO_LLM_TRACE_PATH` set. It never touches the
coverage audit, extraction, or owner induction -- only
`build_learning_goals_reasoner`.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from gossipmemo.config import Settings
from gossipmemo.llm import create_llm
from gossipmemo.reasoners import ReasoningSettings, build_learning_goals_reasoner
from gossipmemo.store import SqliteWorldStore


def discover_spaces(store: SqliteWorldStore) -> list[str]:
    with store._connect() as connection:
        rows = connection.execute("SELECT id FROM spaces ORDER BY id").fetchall()
    return [row["id"] for row in rows]


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run the learning-goals reasoner alone against a database."
    )
    parser.add_argument("database", type=Path, help="SQLite database to plan against.")
    parser.add_argument(
        "--space", action="append", dest="spaces",
        help="Space id to plan. Repeat for several; default is every space in the db.",
    )
    args = parser.parse_args()
    if not args.database.exists():
        raise SystemExit(f"database not found: {args.database}")

    settings = Settings.from_env()
    store = SqliteWorldStore(args.database)
    store.initialize()
    model = create_llm(settings)
    reasoner = build_learning_goals_reasoner(
        store, model, ReasoningSettings(user_name=settings.user_name)
    )
    try:
        spaces = args.spaces or discover_spaces(store)
        for space_id in spaces:
            print(f"PLAN {space_id}", flush=True)
            await reasoner.run_until_caught_up(space_id)
            with store._connect() as connection:
                count = connection.execute(
                    "SELECT COUNT(*) AS n FROM learning_goals WHERE space_id = ?",
                    (space_id,),
                ).fetchone()["n"]
            print(f"DONE {space_id}: {count} learning_goals rows", flush=True)
    finally:
        await model.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
