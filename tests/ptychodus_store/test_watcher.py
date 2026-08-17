from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import pytest

from ptychodus_store.db import repositories as repo
from ptychodus_store.ingest.watcher import ManifestWatcher

pytestmark = pytest.mark.asyncio


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


def _write_diffraction_folder(root: Path, uuid_str: str) -> Path:
    folder = root / 'diffraction' / uuid_str
    folder.mkdir(parents=True)
    with h5py.File(folder / 'diffraction.h5', 'w') as f:
        ds = f.create_dataset('patterns', data=np.zeros((2, 4, 4), dtype=np.uint16))
        ds.attrs['detector_pixel_width_m'] = 1e-5
        ds.attrs['detector_pixel_height_m'] = 1e-5
        f.create_dataset('indexes', data=np.arange(2))
        f.create_dataset('bad_pixels', data=np.zeros((4, 4), dtype=bool))
    manifest = {
        'schema_version': 1,
        'kind': 'diffraction',
        'uuid': uuid_str,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'label': 'live',
    }
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    return folder


async def test_watcher_picks_up_new_manifest(  # type: ignore[no-untyped-def]
    tmp_storage_root, layout, session_provider
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
                return await repo.get_row(session, 'diffraction', uuid)

        row = await _wait_until(check, timeout=8.0)
        assert row is not None, 'watcher did not ingest the new manifest in time'
    finally:
        watcher.stop()


async def test_watcher_handles_manifest_delete(  # type: ignore[no-untyped-def]
    tmp_storage_root, layout, session_provider
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
                return await repo.get_row(session, 'diffraction', uuid)

        row = await _wait_until(appeared, timeout=8.0)
        assert row is not None

        shutil.rmtree(folder)

        async def gone():
            async with session_provider.session_factory() as session:
                return (await repo.get_row(session, 'diffraction', uuid)) is None

        assert await _wait_until(gone, timeout=8.0)
    finally:
        watcher.stop()
