"""Shared fixtures for ptychodus_store tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import h5py
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.config import Settings
from ptychodus_store.db.session import SessionProvider, create_engine, create_schema
from ptychodus_store.storage.layout import StoreLayout


@pytest.fixture
def tmp_storage_root(tmp_path: Path) -> Path:
    root = tmp_path / 'store'
    layout = StoreLayout(root)
    layout.ensure_kind_dirs()
    return root


@pytest.fixture
def layout(tmp_storage_root: Path) -> StoreLayout:
    return StoreLayout(tmp_storage_root)


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator:
    engine = create_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def session_provider(db_engine) -> AsyncIterator[SessionProvider]:  # type: ignore[no-untyped-def]
    yield SessionProvider(db_engine)


@pytest_asyncio.fixture
async def db_session(session_provider: SessionProvider) -> AsyncIterator[AsyncSession]:
    async with session_provider.session_factory() as session:
        yield session


# --------------------------------------------------------------------------------------
# Seed helpers — write a manifest + minimal HDF5 file
# --------------------------------------------------------------------------------------


def _write_manifest(folder: Path, manifest: dict) -> None:
    (folder / 'manifest.json').write_text(json.dumps(manifest))


def _write_diffraction_h5(path: Path, *, num_patterns: int = 4, h: int = 8, w: int = 12) -> None:
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset('patterns', data=np.zeros((num_patterns, h, w), dtype=np.uint16))
        ds.attrs['detector_pixel_width_m'] = 55e-6
        ds.attrs['detector_pixel_height_m'] = 55e-6
        f.create_dataset('indexes', data=np.arange(num_patterns))
        f.create_dataset('bad_pixels', data=np.zeros((h, w), dtype=bool))


def _write_product_h5(path: Path) -> None:
    with h5py.File(path, 'w') as f:
        f.attrs['name'] = 'test-product'
        f.attrs['comments'] = ''
        f.attrs['detector_object_distance_m'] = 1.5
        f.attrs['probe_energy_eV'] = 9000.0
        f.attrs['probe_photon_count'] = 1_000_000
        f.attrs['exposure_time_s'] = 0.1
        f.attrs['mass_attenuation_m2_kg'] = 0.0
        f.attrs['tomography_angle_deg'] = 0.0
        obj = f.create_dataset('object', data=np.zeros((1, 16, 16), dtype=np.complex64))
        obj.attrs['pixel_width_m'] = 1e-9
        obj.attrs['pixel_height_m'] = 1e-9
        obj.attrs['center_x_m'] = 0.0
        obj.attrs['center_y_m'] = 0.0
        probe = f.create_dataset('probe', data=np.zeros((1, 8, 8), dtype=np.complex64))
        probe.attrs['pixel_width_m'] = 1e-9
        probe.attrs['pixel_height_m'] = 1e-9
        f.create_dataset('probe_position_indexes', data=np.arange(4))
        f.create_dataset('probe_position_x_m', data=np.zeros(4))
        f.create_dataset('probe_position_y_m', data=np.zeros(4))
        f.create_dataset('loss_epochs', data=np.array([1, 2, 3]))
        f.create_dataset('loss_values', data=np.array([0.3, 0.2, 0.1]))


def _write_fluorescence_h5(path: Path, elements: list[str], h: int = 6, w: int = 10) -> None:
    with h5py.File(path, 'w') as f:
        f.create_dataset('element_names', data=np.array(elements, dtype='S16'))
        f.create_dataset('element_maps', data=np.zeros((len(elements), h, w), dtype=np.float32))


@pytest.fixture
def seed_campaign(tmp_storage_root: Path) -> Callable[..., UUID]:
    def _seed(
        uuid: UUID | None = None,
        *,
        label: str = 'campaign-a',
        sample_name: str = 'sample-x',
        tags: list[str] | None = None,
    ) -> UUID:
        uuid = uuid or uuid4()
        folder = tmp_storage_root / 'campaign' / str(uuid)
        folder.mkdir(parents=True)
        _write_manifest(
            folder,
            {
                'schema_version': 1,
                'kind': 'campaign',
                'uuid': str(uuid),
                'created_at': datetime.now(timezone.utc).isoformat(),
                'label': label,
                'sample_name': sample_name,
                'tags': tags or [],
            },
        )
        return uuid

    return _seed


@pytest.fixture
def seed_diffraction(tmp_storage_root: Path) -> Callable[..., UUID]:
    def _seed(
        uuid: UUID | None = None,
        *,
        campaign_uuid: UUID | None = None,
        derived_from: list[dict] | None = None,
        probe_energy_eV: float | None = 8000.0,  # noqa: N803
        write_h5: bool = True,
    ) -> UUID:
        uuid = uuid or uuid4()
        folder = tmp_storage_root / 'diffraction' / str(uuid)
        folder.mkdir(parents=True)
        if write_h5:
            _write_diffraction_h5(folder / 'diffraction.h5')
        manifest = {
            'schema_version': 1,
            'kind': 'diffraction',
            'uuid': str(uuid),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'probe_energy_eV': probe_energy_eV,
        }
        if campaign_uuid is not None:
            manifest['campaign_uuid'] = str(campaign_uuid)
        if derived_from is not None:
            manifest['derived_from'] = derived_from
        _write_manifest(folder, manifest)
        return uuid

    return _seed


@pytest.fixture
def seed_product(tmp_storage_root: Path) -> Callable[..., UUID]:
    def _seed(
        uuid: UUID | None = None,
        *,
        derived_from: list[dict] | None = None,
        write_h5: bool = True,
    ) -> UUID:
        uuid = uuid or uuid4()
        folder = tmp_storage_root / 'product' / str(uuid)
        folder.mkdir(parents=True)
        if write_h5:
            _write_product_h5(folder / 'product.h5')
        manifest = {
            'schema_version': 1,
            'kind': 'product',
            'uuid': str(uuid),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        if derived_from is not None:
            manifest['derived_from'] = derived_from
        _write_manifest(folder, manifest)
        return uuid

    return _seed


@pytest.fixture
def seed_fluorescence(tmp_storage_root: Path) -> Callable[..., UUID]:
    def _seed(
        uuid: UUID | None = None,
        *,
        derived_from: list[dict] | None = None,
        elements: list[str] | None = None,
    ) -> UUID:
        uuid = uuid or uuid4()
        folder = tmp_storage_root / 'fluorescence' / str(uuid)
        folder.mkdir(parents=True)
        _write_fluorescence_h5(folder / 'fluorescence.h5', elements or ['Fe', 'Cu'])
        manifest = {
            'schema_version': 1,
            'kind': 'fluorescence',
            'uuid': str(uuid),
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        if derived_from is not None:
            manifest['derived_from'] = derived_from
        _write_manifest(folder, manifest)
        return uuid

    return _seed


# --------------------------------------------------------------------------------------
# FastAPI client fixture
# --------------------------------------------------------------------------------------


@pytest_asyncio.fixture
async def app_client(
    tmp_storage_root: Path,
    db_engine,  # type: ignore[no-untyped-def]
) -> AsyncIterator[AsyncClient]:
    from fastapi import FastAPI

    from ptychodus_store.db.session import SessionProvider
    from ptychodus_store.mcp_server import bind_session_provider
    from ptychodus_store.routers import (
        admin,
        campaign,
        diffraction,
        fluorescence,
        health,
        lineage,
    )
    from ptychodus_store.routers import product as product_router

    layout = StoreLayout(tmp_storage_root)
    provider = SessionProvider(db_engine)
    bind_session_provider(provider)
    settings = Settings(  # type: ignore[call-arg]
        storage_root=tmp_storage_root,
        database_url='sqlite+aiosqlite:///:memory:',
        auto_reconcile_on_startup=False,
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.layout = layout
    app.state.session_provider = provider
    api_prefix = '/api/v1'
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(campaign.router, prefix=api_prefix)
    app.include_router(diffraction.router, prefix=api_prefix)
    app.include_router(product_router.router, prefix=api_prefix)
    app.include_router(fluorescence.router, prefix=api_prefix)
    app.include_router(lineage.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        yield client
