"""Parse → validate → upsert pipeline shared by the watcher and the reconciler."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.storage import h5_introspect
from ptychodus_store.storage.layout import LayoutError, StoreLayout
from ptychodus_store.storage.manifest import (
    CampaignManifest,
    DiffractionManifest,
    FluorescenceManifest,
    ManifestLoadError,
    ProductManifest,
    ResourceKind,
    load_manifest,
)

logger = logging.getLogger(__name__)


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _common_bookkeeping(
    *,
    uuid: UUID,
    folder: Path,
    manifest_path: Path,
    manifest_created_at: datetime,
    state: IngestState,
    error_message: str | None,
) -> dict[str, Any]:
    return {
        'uuid': uuid,
        'folder_path': str(folder),
        'manifest_mtime': _mtime(manifest_path),
        'created_from_manifest_at': manifest_created_at,
        'ingest_state': state,
        'error_message': error_message,
    }


def _values_for_campaign(m: CampaignManifest, **bookkeeping: Any) -> dict[str, Any]:
    return {
        **bookkeeping,
        'label': m.label,
        'comments': m.comments,
        'sample_name': m.sample_name,
        'sample_description': m.sample_description,
        'tags': list(m.tags),
    }


def _values_for_diffraction(
    m: DiffractionManifest, h5: dict[str, Any] | None, **bookkeeping: Any
) -> dict[str, Any]:
    h5 = h5 or {}
    pattern_shape = h5.get('pattern_shape')
    return {
        **bookkeeping,
        'label': m.label,
        'comments': m.comments,
        'campaign_uuid': m.campaign_uuid,
        'detector_distance_m': m.detector_distance_m,
        'probe_energy_eV': m.probe_energy_eV,
        'probe_photon_count': m.probe_photon_count,
        'exposure_time_s': m.exposure_time_s,
        'tomography_angle_deg': m.tomography_angle_deg,
        'tilt_angle_deg': m.tilt_angle_deg,
        'polarization': m.polarization.value if m.polarization is not None else None,
        'crop_center_x_px': m.crop_center_x_px,
        'crop_center_y_px': m.crop_center_y_px,
        'pattern_dtype': h5.get('pattern_dtype'),
        'pattern_height_px': pattern_shape[0] if pattern_shape else None,
        'pattern_width_px': pattern_shape[1] if pattern_shape else None,
        'num_patterns_total': h5.get('num_patterns_total'),
        'detector_pixel_width_m': h5.get('detector_pixel_width_m'),
        'detector_pixel_height_m': h5.get('detector_pixel_height_m'),
    }


def _values_for_product(
    m: ProductManifest, h5: dict[str, Any] | None, **bookkeeping: Any
) -> dict[str, Any]:
    h5 = h5 or {}
    obj_shape = h5.get('object_shape') or (None, None, None)
    probe_shape = h5.get('probe_shape') or (None, None, None)
    return {
        **bookkeeping,
        'name': h5.get('name'),
        'comments': h5.get('comments'),
        'detector_distance_m': h5.get('detector_distance_m'),
        'probe_energy_eV': h5.get('probe_energy_eV'),
        'probe_photon_count': h5.get('probe_photon_count'),
        'exposure_time_s': h5.get('exposure_time_s'),
        'mass_attenuation_m2_kg': h5.get('mass_attenuation_m2_kg'),
        'tomography_angle_deg': h5.get('tomography_angle_deg'),
        'tilt_angle_deg': h5.get('tilt_angle_deg'),
        'polarization': h5.get('polarization'),
        'object_layers': obj_shape[0],
        'object_height_px': obj_shape[1],
        'object_width_px': obj_shape[2],
        'object_pixel_width_m': h5.get('object_pixel_width_m'),
        'object_pixel_height_m': h5.get('object_pixel_height_m'),
        'probe_modes': probe_shape[0],
        'probe_height_px': probe_shape[1],
        'probe_width_px': probe_shape[2],
        'num_scan_points': h5.get('num_scan_points'),
        'num_loss_epochs': h5.get('num_loss_epochs'),
    }


def _values_for_fluorescence(
    m: FluorescenceManifest, h5: dict[str, Any] | None, **bookkeeping: Any
) -> dict[str, Any]:
    h5 = h5 or {}
    map_shape = h5.get('map_shape')
    return {
        **bookkeeping,
        'label': m.label,
        'comments': m.comments,
        'element_names': list(h5.get('element_names') or []),
        'map_height_px': map_shape[0] if map_shape else None,
        'map_width_px': map_shape[1] if map_shape else None,
    }


class DeclaredFileError(Exception):
    """Raised when a manifest declares a file name that is not a plain child of its folder."""


def _declared_file(folder: Path, files: dict[str, str], key: str, default: str) -> Path:
    """Resolve a manifest-declared file name against its resource folder.

    Manifest contents are untrusted: anyone able to write into the storage root controls this
    string, and ``folder / value`` silently accepts both absolute paths and ``..`` segments.
    Requiring a bare file name keeps introspection inside the resource folder.
    """
    name = files.get(key, default)

    if name != Path(name).name or name in ('', '.', '..'):
        raise DeclaredFileError(f'{key!r} must be a plain file name, got {name!r}')

    return folder / name


def _introspect(
    kind: str, folder: Path, files: dict[str, str]
) -> tuple[dict[str, Any] | None, IngestState, str | None]:
    """Open declared HDF5 file(s) and pull HDF5-derived metadata.

    Returns (introspected_dict, state, error_message).
    """
    try:
        if kind == ResourceKind.DIFFRACTION:
            target = _declared_file(folder, files, 'diffraction', 'diffraction.h5')
            if not target.is_file():
                return None, IngestState.MISSING_FILES, f'missing file: {target.name}'
            return h5_introspect.introspect_diffraction(target), IngestState.VALID, None

        if kind == ResourceKind.PRODUCT:
            target = _declared_file(folder, files, 'product', 'product.h5')
            if not target.is_file():
                return None, IngestState.MISSING_FILES, f'missing file: {target.name}'
            return h5_introspect.introspect_product(target), IngestState.VALID, None

        if kind == ResourceKind.FLUORESCENCE:
            target = _declared_file(folder, files, 'fluorescence', 'fluorescence.h5')
            if not target.is_file():
                return None, IngestState.MISSING_FILES, f'missing file: {target.name}'
            return h5_introspect.introspect_fluorescence(target), IngestState.VALID, None

        return None, IngestState.VALID, None
    except DeclaredFileError as exc:
        return None, IngestState.INVALID, str(exc)
    except h5_introspect.IntrospectionError as exc:
        return None, IngestState.INVALID, str(exc)


async def _reevaluate_orphan(session: AsyncSession, kind: str, uuid: UUID) -> None:
    """Flip VALID↔ORPHANED for one row based on whether its outgoing edges resolve."""
    row = await repo.get_row(session, kind, uuid)
    if row is None or row.ingest_state in (IngestState.INVALID, IngestState.MISSING_FILES):
        return
    edges = await repo.outgoing_edges(session, uuid)
    has_unresolved = False
    for edge in edges:
        if not await repo.row_exists(session, edge.target_kind, edge.target_uuid):
            has_unresolved = True
            break
    desired = IngestState.ORPHANED if has_unresolved else IngestState.VALID
    if row.ingest_state != desired:
        row.ingest_state = desired


async def _propagate_orphan_to_referrers(session: AsyncSession, uuid: UUID) -> None:
    """When a row's existence changes, re-evaluate every row that points at it."""
    for edge in await repo.edges_referencing(session, uuid):
        await _reevaluate_orphan(session, edge.source_kind, edge.source_uuid)


async def ingest_manifest(session: AsyncSession, layout: StoreLayout, manifest_path: Path) -> None:
    """Parse a manifest at `manifest_path`, validate, introspect HDF5, upsert the row."""
    try:
        location = layout.parse_manifest_path(manifest_path)
    except LayoutError as exc:
        logger.warning('skipping %s: %s', manifest_path, exc)
        return

    try:
        manifest = load_manifest(
            manifest_path, expected_kind=location.kind, expected_uuid=location.uuid
        )
    except ManifestLoadError as exc:
        logger.warning('invalid manifest %s: %s', manifest_path, exc)
        bookkeeping = _common_bookkeeping(
            uuid=location.uuid,
            folder=location.folder,
            manifest_path=manifest_path,
            manifest_created_at=datetime.now(timezone.utc),
            state=IngestState.INVALID,
            error_message=str(exc),
        )
        # Insert a minimal placeholder so the error is visible via the API.
        await _upsert_minimal(session, location.kind, bookkeeping)
        return

    files = getattr(manifest, 'files', {}) or {}
    h5, state, error_message = _introspect(location.kind, location.folder, files)

    bookkeeping = _common_bookkeeping(
        uuid=location.uuid,
        folder=location.folder,
        manifest_path=manifest_path,
        manifest_created_at=manifest.created_at,
        state=state,
        error_message=error_message,
    )

    if isinstance(manifest, CampaignManifest):
        values = _values_for_campaign(manifest, **bookkeeping)
    elif isinstance(manifest, DiffractionManifest):
        values = _values_for_diffraction(manifest, h5, **bookkeeping)
    elif isinstance(manifest, ProductManifest):
        values = _values_for_product(manifest, h5, **bookkeeping)
    elif isinstance(manifest, FluorescenceManifest):
        values = _values_for_fluorescence(manifest, h5, **bookkeeping)
    else:  # pragma: no cover — discriminated union exhausted
        raise AssertionError(f'unexpected manifest type: {type(manifest)!r}')

    await repo.upsert_row(session, location.kind, values)

    # Rewrite outgoing edges from manifest.derived_from
    derived_from = list(getattr(manifest, 'derived_from', []) or [])
    edges = [(ref.kind, ref.uuid) for ref in derived_from]
    await repo.replace_edges(session, location.kind, location.uuid, edges)

    # Re-evaluate orphan state for this row and anyone pointing at it
    if state == IngestState.VALID:
        await _reevaluate_orphan(session, location.kind, location.uuid)
    await _propagate_orphan_to_referrers(session, location.uuid)


async def _upsert_minimal(session: AsyncSession, kind: str, bookkeeping: dict[str, Any]) -> None:
    """Upsert just the bookkeeping fields for an invalid manifest, so it's visible."""
    if kind == ResourceKind.CAMPAIGN:
        values: dict[str, Any] = {
            **bookkeeping,
            'label': '',
            'comments': '',
            'sample_name': '',
            'sample_description': '',
            'tags': [],
        }
    elif kind == ResourceKind.DIFFRACTION:
        values = {**bookkeeping, 'label': '', 'comments': ''}
    elif kind == ResourceKind.PRODUCT:
        values = {**bookkeeping}
    elif kind == ResourceKind.FLUORESCENCE:
        values = {**bookkeeping, 'label': '', 'comments': '', 'element_names': []}
    else:
        return
    await repo.upsert_row(session, kind, values)


async def delete_manifest(session: AsyncSession, layout: StoreLayout, manifest_path: Path) -> None:
    """Handle a manifest delete event: drop the row, clear edges, propagate orphan state."""
    try:
        location = layout.parse_manifest_path(manifest_path)
    except LayoutError:
        return
    await repo.delete_row(session, location.kind, location.uuid)
    await repo.replace_edges(session, location.kind, location.uuid, [])
    await _propagate_orphan_to_referrers(session, location.uuid)
