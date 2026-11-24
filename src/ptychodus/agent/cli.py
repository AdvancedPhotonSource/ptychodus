"""Argparse-based CLI: serve."""

from __future__ import annotations

import argparse
import sys

from ptychodus.agent.config import get_settings


def _serve(_args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    # One process: a conversation's running turn is tracked in memory.
    uvicorn.run(
        'ptychodus.agent.app:create_app',
        host=settings.host,
        port=settings.port,
        factory=True,
        log_level=settings.log_level.lower(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ptychodus-agent')
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    serve = subparsers.add_parser('serve', help='run the agent service')
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
