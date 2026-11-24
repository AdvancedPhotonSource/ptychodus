"""Argparse-based CLI: serve | rebuild-index | mint-token | revoke-token."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ptychodus.webapp.config import TokenFileSettings, get_settings


def _serve(_args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        'ptychodus.webapp.app:create_app',
        host=settings.host,
        port=settings.port,
        factory=True,
        log_level=settings.log_level.lower(),
    )


def _rebuild_index(_args: argparse.Namespace) -> None:
    from ptychodus.webapp.db.session import SessionProvider, create_engine, create_schema
    from ptychodus.webapp.ingest.reconciler import full_rescan
    from ptychodus.api.storage import StorageLayout

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    layout = StorageLayout(settings.storage_root)
    engine = create_engine(settings.database_url)
    provider = SessionProvider(engine)

    async def _run() -> None:
        await create_schema(engine)
        layout.ensure_kind_dirs()
        async with provider.session_factory() as session:
            counts = await full_rescan(session, layout)
        await provider.dispose()
        print(f'reconcile counts: {counts}')

    asyncio.run(_run())


def _tokens_file(args: argparse.Namespace) -> Path:
    if args.tokens_file is not None:
        return Path(args.tokens_file)

    tokens_file = TokenFileSettings().mcp_tokens_file

    if tokens_file is None:
        raise SystemExit(
            'Name the token file with --tokens-file or PTYCHODUS_WEBAPP_MCP_TOKENS_FILE.'
        )

    return tokens_file


def _mint_token(args: argparse.Namespace) -> None:
    from ptychodus.webapp.auth import mint_client_token

    tokens_file = _tokens_file(args)
    token = mint_client_token(tokens_file, args.client_id)
    print(token)
    print(
        f'Recorded {args.client_id!r} in {tokens_file}. The token is shown only now; the webapp '
        'accepts it after a restart.',
        file=sys.stderr,
    )


def _revoke_token(args: argparse.Namespace) -> None:
    from ptychodus.webapp.auth import revoke_client_token

    tokens_file = _tokens_file(args)

    try:
        revoke_client_token(tokens_file, args.client_id)
    except LookupError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f'Removed {args.client_id!r} from {tokens_file}; the webapp refuses its token after a '
        'restart.',
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ptychodus-webapp')
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    serve = subparsers.add_parser('serve', help='run the HTTP + MCP server')
    serve.set_defaults(func=_serve)

    rebuild = subparsers.add_parser(
        'rebuild-index', help='full reconciliation of the DB cache from disk'
    )
    rebuild.set_defaults(func=_rebuild_index)

    for name, func, help_text in (
        ('mint-token', _mint_token, 'print a new MCP bearer token for a client and record it'),
        ('revoke-token', _revoke_token, "remove a client's MCP bearer token"),
    ):
        token_parser = subparsers.add_parser(name, help=help_text)
        token_parser.add_argument('client_id', help='the client the token identifies, e.g. agent')
        token_parser.add_argument(
            '--tokens-file',
            help='the token file; defaults to PTYCHODUS_WEBAPP_MCP_TOKENS_FILE',
        )
        token_parser.set_defaults(func=func)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
