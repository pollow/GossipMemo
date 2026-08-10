from __future__ import annotations

import argparse

from .app import create_app
from .config import ConfigurationError, get_settings


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="gossipmemo")
    commands = result.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="run the GossipMemo HTTP server")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    return result


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


if __name__ == "__main__":
    main()
