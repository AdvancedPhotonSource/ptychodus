"""Argparse-based CLI: serve | rebuild-index."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from ptychodus_store.config import get_settings


def _serve(_args: argparse.Namespace) -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        'ptychodus_store.app:create_app',
        host=settings.host,
        port=settings.port,
        factory=True,
        log_level=settings.log_level.lower(),
    )


def _rebuild_index(_args: argparse.Namespace) -> None:
    from ptychodus_store.db.session import SessionProvider, create_engine, create_schema
    from ptychodus_store.ingest.reconciler import full_rescan
    from ptychodus_store.storage.layout import StoreLayout

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    layout = StoreLayout(settings.storage_root)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='ptychodus-store')
    subparsers = parser.add_subparsers(dest='cmd', required=True)

    serve = subparsers.add_parser('serve', help='run the HTTP + MCP server')
    serve.set_defaults(func=_serve)

    rebuild = subparsers.add_parser(
        'rebuild-index', help='full reconciliation of the DB cache from disk'
    )
    rebuild.set_defaults(func=_rebuild_index)

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
