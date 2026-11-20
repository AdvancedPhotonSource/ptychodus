from __future__ import annotations

from ptychodus.webapp.db.models import Diffraction
from ptychodus.webapp.db.session import SessionProvider
from ptychodus.api.storage import StorageLayout

import asyncio
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pytest

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.ingest import watcher as watcher_module
from ptychodus.webapp.ingest.pipeline import ingest_manifest
from ptychodus.webapp.ingest.watcher import ManifestWatcher


async def _wait_until(predicate, *, timeout: float = 8.0, interval: float = 0.2):
    """Poll `predicate` (async callable) until truthy or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        result = await predicate()
        if result:
            return result
        if asyncio.get_event_loop().time() > deadline:
            return result
        await asyncio.sleep(interval)


def _diffraction_manifest(uuid_str: str, *, label: str = 'live') -> str:
    return json.dumps(
        {
            'schema_version': 1,
            'kind': 'diffraction',
            'uuid': uuid_str,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'label': label,
        }
    )


def _publish_manifest(folder: Path, text: str) -> None:
    """Publish as the writers do: a hidden temporary file replaced onto the manifest."""
    temporary = folder / '.manifest.json.tmp'
    temporary.write_text(text)
    os.replace(temporary, folder / 'manifest.json')


def _write_diffraction_folder(root: Path, uuid_str: str, *, with_manifest: bool = True) -> Path:
    folder = root / 'diffraction' / uuid_str
    folder.mkdir(parents=True)
    with h5py.File(folder / 'diffraction.h5', 'w') as f:
        ds = f.create_dataset('patterns', data=np.zeros((2, 4, 4), dtype=np.uint16))
        ds.attrs['detector_pixel_width_m'] = 1e-5
        ds.attrs['detector_pixel_height_m'] = 1e-5
        f.create_dataset('indexes', data=np.arange(2))
        f.create_dataset('bad_pixels', data=np.zeros((4, 4), dtype=bool))
    if with_manifest:
        (folder / 'manifest.json').write_text(_diffraction_manifest(uuid_str))
    return folder


async def test_watcher_picks_up_new_manifest(
    tmp_storage_root: Path, layout: StorageLayout, session_provider: SessionProvider
):
    loop = asyncio.get_running_loop()
    watcher = ManifestWatcher(
        layout,
        session_provider.session_factory,
        loop,
        polling_interval_s=0.3,
        debounce_window_s=0.2,
    )
    watcher.start()
    try:
        uuid = uuid4()
        _write_diffraction_folder(tmp_storage_root, str(uuid))

        async def check():
            async with session_provider.session_factory() as session:
                return await repo.get_row(session, Diffraction, uuid)

        row = await _wait_until(check, timeout=8.0)
        assert row is not None, 'watcher did not ingest the new manifest in time'
    finally:
        watcher.stop()


async def test_watcher_handles_manifest_delete(
    tmp_storage_root: Path, layout: StorageLayout, session_provider: SessionProvider
):
    loop = asyncio.get_running_loop()
    watcher = ManifestWatcher(
        layout,
        session_provider.session_factory,
        loop,
        polling_interval_s=0.3,
        debounce_window_s=0.2,
    )
    watcher.start()
    try:
        uuid = uuid4()
        folder = _write_diffraction_folder(tmp_storage_root, str(uuid))

        async def appeared():
            async with session_provider.session_factory() as session:
                return await repo.get_row(session, Diffraction, uuid)

        row = await _wait_until(appeared, timeout=8.0)
        assert row is not None

        shutil.rmtree(folder)

        async def gone():
            async with session_provider.session_factory() as session:
                return (await repo.get_row(session, Diffraction, uuid)) is None

        assert await _wait_until(gone, timeout=8.0)
    finally:
        watcher.stop()


async def test_watcher_ingests_a_resource_once_its_manifest_is_published(
    tmp_storage_root: Path, layout: StorageLayout, session_provider: SessionProvider
):
    """A folder fills in place and becomes a resource when its manifest is written last."""
    loop = asyncio.get_running_loop()
    watcher = ManifestWatcher(
        layout,
        session_provider.session_factory,
        loop,
        polling_interval_s=0.3,
        debounce_window_s=0.2,
    )
    watcher.start()
    try:
        uuid = uuid4()
        folder = _write_diffraction_folder(tmp_storage_root, str(uuid), with_manifest=False)
        await asyncio.sleep(1.0)

        async def check():
            async with session_provider.session_factory() as session:
                return await repo.get_row(session, Diffraction, uuid)

        assert await check() is None, 'a folder without a manifest must not be ingested'

        _publish_manifest(folder, _diffraction_manifest(str(uuid)))
        row = await _wait_until(check, timeout=8.0)
        assert row is not None, 'watcher did not ingest the published resource'
    finally:
        watcher.stop()


async def test_a_replaced_manifest_updates_the_row_without_deleting_it(
    tmp_storage_root: Path,
    layout: StorageLayout,
    session_provider: SessionProvider,
    monkeypatch: pytest.MonkeyPatch,
):
    """A run's commit replaces its manifest, which the poller reports as delete plus create."""
    uuid = uuid4()
    folder = _write_diffraction_folder(tmp_storage_root, str(uuid))
    async with session_provider.session_factory() as session:
        await ingest_manifest(session, layout, folder / 'manifest.json')
        await session.commit()

    deleted: list[Path] = []

    async def recording_delete_manifest(session, layout, manifest_path: Path) -> None:
        deleted.append(manifest_path)

    monkeypatch.setattr(watcher_module, 'delete_manifest', recording_delete_manifest)
    loop = asyncio.get_running_loop()
    watcher = ManifestWatcher(
        layout,
        session_provider.session_factory,
        loop,
        polling_interval_s=0.3,
        debounce_window_s=0.2,
    )
    watcher.start()
    try:
        await asyncio.sleep(0.6)
        _publish_manifest(folder, _diffraction_manifest(str(uuid), label='advanced'))

        async def relabelled():
            async with session_provider.session_factory() as session:
                row = await repo.get_row(session, Diffraction, uuid)
                return row is not None and row.label == 'advanced'

        assert await _wait_until(relabelled, timeout=8.0), 'the replaced manifest was not ingested'
        await asyncio.sleep(0.6)
        assert deleted == []
    finally:
        watcher.stop()


async def test_a_stale_delete_of_an_existing_manifest_reingests(
    tmp_storage_root: Path, layout: StorageLayout, session_provider: SessionProvider
):
    uuid = uuid4()
    folder = _write_diffraction_folder(tmp_storage_root, str(uuid))
    watcher = ManifestWatcher(
        layout,
        session_provider.session_factory,
        asyncio.get_running_loop(),
        polling_interval_s=0.3,
        debounce_window_s=0.2,
    )

    await watcher._run_delete(folder / 'manifest.json')

    async with session_provider.session_factory() as session:
        assert await repo.get_row(session, Diffraction, uuid) is not None
