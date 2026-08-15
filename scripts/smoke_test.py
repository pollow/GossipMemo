#!/usr/bin/env python3
"""Exercise a running GossipMemo server through its public Python SDK."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from typing import Any

from gossipmemo_client import GossipMemo, GossipMemoError


def show(label: str, value: Any) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run an end-to-end smoke test against a GossipMemo server."
    )
    result.add_argument(
        "--base-url",
        default=os.getenv("GOSSIPMEMO_BASE_URL", "http://127.0.0.1:8765"),
    )
    result.add_argument(
        "--api-key", default=os.getenv("GOSSIPMEMO_API_KEY", "")
    )
    result.add_argument(
        "--space",
        default=os.getenv("GOSSIPMEMO_SPACE_ID", "smoke-test"),
        help="Use 'personal' to test your normal Space (default: smoke-test).",
    )
    result.add_argument("--timeout", type=float, default=180.0)
    result.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Only test manual memory and corrections, even if an LLM is configured.",
    )
    return result


def run(args: argparse.Namespace) -> None:
    run_id = uuid.uuid4().hex[:10]
    print(
        f"Testing {args.base_url} in Space {args.space!r} "
        f"(run {run_id})"
    )

    with GossipMemo(
        args.base_url,
        api_key=args.api_key,
        space_id=args.space,
        timeout=30.0,
    ) as memory:
        health = memory.health()
        show("health", health)

        if health.get("llm_configured") is not True:
            raise RuntimeError(
                "server started without a configured LLM; strict startup "
                "validation is not active"
            )

        if not args.skip_ingest:
            result = memory.ingest(
                content="Alice 跟我说，Bob 最近可能准备离职，但还没有最终决定。",
                author="user",
                source={
                    "provider": "smoke-test",
                    "conversation_key": run_id,
                    "item_id": "gossip-1",
                    "metadata": {"test_run": run_id},
                },
                idempotency_key=f"{run_id}:gossip-1",
            )
            show("ingest result", result)

            query = memory.query(
                "Bob 最近的工作状态怎么样？这个消息是谁告诉我的？",
                people=["Bob"],
                include_relationships=True,
                expand_relationships=1,
                include_evidence=True,
            )
            show("gossip query", query)
            if not query.get("memories"):
                raise RuntimeError(
                    "ingest completed, but the Bob query returned no memories; "
                    "inspect the extraction prompt/model output"
                )
        else:
            print("\nSkipping automatic ingest because --skip-ingest was used.")

        manual = memory.add_memory(
            f"Smoke Test Person prefers coffee. Test run: {run_id}",
            kind="preference",
            people=["Smoke Test Person"],
        )
        show("manual memory", manual)

        manual_query = memory.query(
            "What does Smoke Test Person prefer?",
            people=["Smoke Test Person"],
            include_evidence=True,
        )
        show("manual-memory query", manual_query)
        if not manual_query.get("memories"):
            raise RuntimeError("manual memory was not returned by query")

        corrected = memory.supersede(
            manual["id"],
            f"Smoke Test Person now prefers tea. Test run: {run_id}",
            kind="preference",
            reason="Smoke-test correction",
        )
        show("supersede", corrected)

        retracted = memory.retract(
            corrected["id"], reason="Smoke test cleanup"
        )
        show("retract", retracted)

    print("\nPASS: GossipMemo smoke test completed successfully.")


def main() -> int:
    args = parser().parse_args()
    try:
        run(args)
    except (GossipMemoError, RuntimeError, KeyError, TypeError, ValueError) as error:
        print(f"\nFAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
