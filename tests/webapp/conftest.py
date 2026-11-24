"""Shared fixtures for ptychodus.webapp tests."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

import json
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import h5py
import numpy as np
import pytest
import pytest_asyncio
from fastmcp import FastMCP
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.api.processing import RunRecord
from ptychodus.webapp.config import Settings
from ptychodus.webapp.db.session import SessionProvider, create_engine, create_schema
from ptychodus.webapp.mcp_server import create_mcp_server
from ptychodus.webapp.rendering.offload import RenderRunner
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.api.storage import StorageLayout


@pytest.fixture
def tmp_storage_root(tmp_path: Path) -> Path:
    root = tmp_path / 'storage'
    layout = StorageLayout(root)
    layout.ensure_kind_dirs()
    return root


@pytest.fixture
def layout(tmp_storage_root: Path) -> StorageLayout:
    return StorageLayout(tmp_storage_root)


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator:
    engine = create_engine('sqlite+aiosqlite:///:memory:')
    await create_schema(engine)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def session_provider(db_engine: AsyncEngine) -> AsyncIterator[SessionProvider]:
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
        f.attrs['photon_energy_eV'] = 9000.0
        f.attrs['probe_photon_count'] = 1_000_000
        f.attrs['exposure_time_s'] = 0.1
        f.attrs['mass_attenuation_m2_per_kg'] = 0.0
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


def _write_analyzable_product_h5(path: Path, position_shift_x_m: float) -> None:
    """A product every analysis report can describe, with no I0.

    A Gaussian probe over a random-phase object, at a 4 x 4 grid of probe positions 6 px
    apart, shifted in x by `position_shift_x_m`.
    """
    pixel_size_m = 1e-9
    rng = np.random.default_rng(0)

    with h5py.File(path, 'w') as f:
        f.attrs['name'] = 'analyzable-product'
        f.attrs['comments'] = ''
        f.attrs['detector_object_distance_m'] = 1.5
        f.attrs['photon_energy_eV'] = 9000.0
        f.attrs['probe_photon_count'] = 1_000_000
        f.attrs['exposure_time_s'] = 0.1
        f.attrs['mass_attenuation_m2_per_kg'] = 0.0
        f.attrs['tomography_angle_deg'] = 0.0
        phase = 0.3 * rng.standard_normal((1, 64, 64))
        obj = f.create_dataset('object', data=np.exp(1j * phase).astype(np.complex64))
        obj.attrs['pixel_width_m'] = pixel_size_m
        obj.attrs['pixel_height_m'] = pixel_size_m
        obj.attrs['center_x_m'] = 0.0
        obj.attrs['center_y_m'] = 0.0
        y_px, x_px = np.mgrid[:16, :16] - 8
        gaussian = np.exp(-(x_px**2 + y_px**2) / (2 * 3.0**2))
        probe = f.create_dataset('probe', data=gaussian[np.newaxis].astype(np.complex64))
        probe.attrs['pixel_width_m'] = pixel_size_m
        probe.attrs['pixel_height_m'] = pixel_size_m
        grid_m = (np.arange(4) - 1.5) * 6 * pixel_size_m
        x_m, y_m = np.meshgrid(grid_m, grid_m)
        f.create_dataset('probe_position_indexes', data=np.arange(16))
        f.create_dataset('probe_position_x_m', data=x_m.ravel() + position_shift_x_m)
        f.create_dataset('probe_position_y_m', data=y_m.ravel())


def _write_fluorescence_h5(path: Path, elements: list[str], h: int = 6, w: int = 10) -> None:
    """Write the XRF-Maps v10 NNLS layout expected by :func:`load_fluorescence_data`."""
    with h5py.File(path, 'w') as f:
        group = f.require_group('/MAPS/XRF_Analyzed/NNLS')
        group.create_dataset(
            'Counts_Per_Sec', data=np.zeros((len(elements), h, w), dtype=np.float32)
        )
        group.create_dataset('Channel_Names', data=np.array(elements, dtype='S16'))


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
        photon_energy_eV: float | None = 8000.0,  # noqa: N803
        polarization: str | None = None,
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
            'photon_energy_eV': photon_energy_eV,
            'polarization': polarization,
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
        data_file: str | None = 'product.h5',
        run: RunRecord | None = None,
        analyzable: bool = False,
        position_shift_x_m: float = 0.0,
    ) -> UUID:
        """`data_file` is the file the manifest names, `None` for a run with no data yet.

        The product is blank, with a probe carrying no power, unless `analyzable`.
        """
        uuid = uuid or uuid4()
        folder = tmp_storage_root / 'product' / str(uuid)
        folder.mkdir(parents=True)
        if write_h5 and data_file is not None:
            if analyzable:
                _write_analyzable_product_h5(folder / data_file, position_shift_x_m)
            else:
                _write_product_h5(folder / data_file)
        manifest: dict = {
            'schema_version': 1,
            'kind': 'product',
            'uuid': str(uuid),
            'created_at': datetime.now(timezone.utc).isoformat(),
            'files': {} if data_file is None else {'product': data_file},
        }
        if derived_from is not None:
            manifest['derived_from'] = derived_from
        if run is not None:
            manifest['run'] = run.model_dump(mode='json')
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
# MCP server and FastAPI client fixtures
# --------------------------------------------------------------------------------------


@pytest.fixture
def service(session_provider: SessionProvider, layout: StorageLayout) -> ResourceService:
    return ResourceService(session_provider, layout, RenderRunner(4))


@pytest.fixture
def bound_mcp(service: ResourceService) -> FastMCP:
    return create_mcp_server(service)


@pytest_asyncio.fixture
async def app_client(
    tmp_storage_root: Path,
    session_provider: SessionProvider,
    service: ResourceService,
) -> AsyncIterator[AsyncClient]:
    from fastapi import FastAPI

    from ptychodus.webapp.routers import (
        admin,
        campaign,
        diffraction,
        fluorescence,
        health,
        lineage,
        visualization,
    )
    from ptychodus.webapp.routers import product as product_router
    from ptychodus.webapp.routers.errors import add_service_error_handlers

    settings = Settings(
        storage_root=tmp_storage_root,
        database_url='sqlite+aiosqlite:///:memory:',
        auto_reconcile_on_startup=False,
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.layout = StorageLayout(tmp_storage_root)
    app.state.session_provider = session_provider
    app.state.service = service
    add_service_error_handlers(app)
    api_prefix = '/api/v1'
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(campaign.router, prefix=api_prefix)
    app.include_router(diffraction.router, prefix=api_prefix)
    app.include_router(product_router.router, prefix=api_prefix)
    app.include_router(fluorescence.router, prefix=api_prefix)
    app.include_router(lineage.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    app.include_router(visualization.router, prefix=api_prefix)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        yield client
