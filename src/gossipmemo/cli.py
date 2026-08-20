from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .app import create_app
from .config import ConfigurationError, get_settings
from .imports import load_chat_messages
from .llm import create_llm
from .logging import configure_logging
from .store import SqliteWorldStore
from .world import SocialMemoryWorld


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="gossipmemo")
    commands = result.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the GossipMemo HTTP server")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    importing = commands.add_parser("import", help="import chat exports and USER.md")
    importing.add_argument("--space", required=True)
    importing.add_argument("--chat", action="append", type=Path, default=[])
    importing.add_argument("--user-md", type=Path)
    return result


async def _run_import(args: argparse.Namespace) -> None:
    if not args.chat and not args.user_md:
        raise SystemExit("gossipmemo import requires --chat and/or --user-md")
    try:
        messages = [
            message for path in args.chat for message in load_chat_messages(path)
        ]
        messages.sort(key=lambda message: message.occurred_at)
        markdown = None
        if args.user_md:
            try:
                markdown = args.user_md.read_text(encoding="utf-8")
            except OSError as error:
                raise ValueError(f"cannot read USER.md {args.user_md}: {error}") from error
        settings = get_settings()
        configure_logging(settings.logging_level, settings.logging_format)
        world = SocialMemoryWorld(
            SqliteWorldStore(settings.database_path),
            create_llm(settings),
            extraction_batch_size=settings.extraction_batch_size,
            extraction_batch_timeout_seconds=settings.extraction_batch_timeout_seconds,
            settings=settings,
        )
        await world.start()
        try:
            if markdown is not None:
                world.store.overwrite_user_model(args.space, {"summary": markdown})
            summary = (
                await world.import_messages(args.space, messages)
                if messages
                else {"messages": 0, "extracted": 0}
            )
            summary["user_model_overwritten"] = markdown is not None
        finally:
            await world.stop()
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    except (ConfigurationError, ValueError, RuntimeError) as error:
        raise SystemExit(f"GossipMemo import error: {error}") from error


def main() -> None:
    args = parser().parse_args()
    if args.command == "serve":
        import uvicorn

        try:
            settings = get_settings()
        except (ConfigurationError, ValueError) as error:
            raise SystemExit(f"GossipMemo configuration error: {error}") from error
        # A single process is a product invariant for SQLite + the local FIFO
        # queue. Deliberately do not expose a workers option here.
        uvicorn.run(
            create_app(settings),
            host=args.host or settings.host,
            port=args.port or settings.port,
            workers=1,
        )
    elif args.command == "import":
        asyncio.run(_run_import(args))


if __name__ == "__main__":
    main()
