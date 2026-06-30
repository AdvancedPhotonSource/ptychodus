"""PollingObserver-based watchdog thread with per-path debounce on `manifest.json`."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from threading import Lock

from sqlalchemy.ext.asyncio import async_sessionmaker
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from ptychodus_store.ingest.pipeline import delete_manifest, ingest_manifest
from ptychodus_store.storage.layout import StoreLayout
from ptychodus_store.storage.manifest import MANIFEST_FILENAME

logger = logging.getLogger(__name__)


class _ManifestEventHandler(FileSystemEventHandler):
    """Collapses bursts of manifest events into a single debounced ingest per path."""

    def __init__(self, on_upsert, on_delete, debounce_window_s: float) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self._on_upsert = on_upsert
        self._on_delete = on_delete
        self._debounce_window_s = debounce_window_s
        self._pending_upserts: set[Path] = set()
        self._pending_deletes: set[Path] = set()
        self._lock = Lock()

    def _is_manifest(self, src_path: str) -> bool:
        return Path(src_path).name == MANIFEST_FILENAME

    def _schedule_upsert(self, path: Path) -> None:
        with self._lock:
            self._pending_upserts.add(path)
            self._pending_deletes.discard(path)
        self._on_upsert(path, self._debounce_window_s)

    def _schedule_delete(self, path: Path) -> None:
        with self._lock:
            self._pending_deletes.add(path)
            self._pending_upserts.discard(path)
        self._on_delete(path, self._debounce_window_s)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_manifest(str(event.src_path)):
            return
        self._schedule_upsert(Path(str(event.src_path)))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_manifest(str(event.src_path)):
            return
        self._schedule_upsert(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        src = str(event.src_path)
        dest = str(getattr(event, 'dest_path', '') or '')
        if self._is_manifest(src):
            self._schedule_delete(Path(src))
        if dest and self._is_manifest(dest):
            self._schedule_upsert(Path(dest))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory or not self._is_manifest(str(event.src_path)):
            return
        self._schedule_delete(Path(str(event.src_path)))


class ManifestWatcher:
    """Owns the watchdog thread and bridges file events to async ingestion."""

    def __init__(
        self,
        layout: StoreLayout,
        session_factory: async_sessionmaker,
        loop: asyncio.AbstractEventLoop,
        *,
        polling_interval_s: float = 2.0,
        debounce_window_s: float = 1.0,
    ) -> None:
        self._layout = layout
        self._session_factory = session_factory
        self._loop = loop
        self._polling_interval_s = polling_interval_s
        self._debounce_window_s = debounce_window_s
        self._observer = PollingObserver(timeout=polling_interval_s)
        self._handler = _ManifestEventHandler(
            self._schedule_upsert, self._schedule_delete, debounce_window_s
        )
        self._pending_upsert_handles: dict[Path, asyncio.TimerHandle] = {}
        self._pending_delete_handles: dict[Path, asyncio.TimerHandle] = {}
        self._started = False

    @property
    def is_alive(self) -> bool:
        return self._observer.is_alive()

    def start(self) -> None:
        if self._started:
            return
        self._layout.ensure_kind_dirs()
        self._observer.schedule(self._handler, str(self._layout.root), recursive=True)
        self._observer.start()
        self._started = True
        logger.info('manifest watcher started on %s', self._layout.root)

    def stop(self) -> None:
        if not self._started:
            return
        self._observer.stop()
        self._observer.join(timeout=self._polling_interval_s * 2 + 1)
        self._started = False
        logger.info('manifest watcher stopped')

    # --- debounce + asyncio bridge ---

    def _schedule_upsert(self, path: Path, debounce: float) -> None:
        self._loop.call_soon_threadsafe(self._debounce_upsert, path, debounce)

    def _schedule_delete(self, path: Path, debounce: float) -> None:
        self._loop.call_soon_threadsafe(self._debounce_delete, path, debounce)

    def _debounce_upsert(self, path: Path, debounce: float) -> None:
        existing = self._pending_upsert_handles.pop(path, None)
        if existing is not None:
            existing.cancel()
        handle = self._loop.call_later(
            debounce, lambda: asyncio.ensure_future(self._run_upsert(path))
        )
        self._pending_upsert_handles[path] = handle

    def _debounce_delete(self, path: Path, debounce: float) -> None:
        existing = self._pending_delete_handles.pop(path, None)
        if existing is not None:
            existing.cancel()
        handle = self._loop.call_later(
            debounce, lambda: asyncio.ensure_future(self._run_delete(path))
        )
        self._pending_delete_handles[path] = handle

    async def _run_upsert(self, path: Path) -> None:
        self._pending_upsert_handles.pop(path, None)
        try:
            async with self._session_factory() as session:
                await ingest_manifest(session, self._layout, path)
                await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception('error ingesting %s', path)

    async def _run_delete(self, path: Path) -> None:
        self._pending_delete_handles.pop(path, None)
        try:
            async with self._session_factory() as session:
                await delete_manifest(session, self._layout, path)
                await session.commit()
        except Exception:  # noqa: BLE001
            logger.exception('error deleting %s', path)
