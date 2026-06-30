from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ptychodus_store.storage.manifest import (
    HDF5_OWNED_KEYS,
    CampaignManifest,
    DerivedFromRef,
    DiffractionManifest,
    FluorescenceManifest,
    ManifestLoadError,
    ProductManifest,
    ResourceKind,
    load_manifest,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_diffraction_manifest_default_files() -> None:
    m = DiffractionManifest(uuid=uuid4(), created_at=datetime.now(timezone.utc))
    assert m.files == {'diffraction': 'diffraction.h5'}
    assert m.derived_from == []
    assert m.campaign_uuid is None


def test_product_manifest_minimal_fields() -> None:
    m = ProductManifest(uuid=uuid4(), created_at=datetime.now(timezone.utc))
    assert m.files == {'product': 'product.h5'}
    assert m.derived_from == []


def test_fluorescence_manifest_defaults() -> None:
    m = FluorescenceManifest(uuid=uuid4(), created_at=datetime.now(timezone.utc))
    assert m.files == {'fluorescence': 'fluorescence.h5'}


def test_derived_from_ref_rejects_campaign() -> None:
    with pytest.raises(ValidationError):
        DerivedFromRef(kind='campaign', uuid=uuid4())  # type: ignore[arg-type]


def test_manifest_extra_rejects_hdf5_owned_keys() -> None:
    # detector_pixel_width_m is HDF5-owned for diffraction; supplying it via extra must raise
    with pytest.raises(ValidationError):
        DiffractionManifest(
            uuid=uuid4(),
            created_at=datetime.now(timezone.utc),
            extra={'detector_pixel_width_m': 1.0},
        )


def test_hdf5_owned_disjoint_from_model_fields() -> None:
    """Sanity: HDF5-owned keys must NOT also appear as manifest fields."""
    for kind, model_cls in (
        (ResourceKind.CAMPAIGN, CampaignManifest),
        (ResourceKind.DIFFRACTION, DiffractionManifest),
        (ResourceKind.PRODUCT, ProductManifest),
        (ResourceKind.FLUORESCENCE, FluorescenceManifest),
    ):
        manifest_fields = set(model_cls.model_fields.keys())
        clash = HDF5_OWNED_KEYS[kind] & manifest_fields
        assert not clash, f'{kind}: manifest fields overlap HDF5-owned keys: {clash}'


def test_load_manifest_rejects_uuid_mismatch(tmp_path: Path) -> None:
    folder_uuid = uuid4()
    other_uuid = uuid4()
    path = tmp_path / 'manifest.json'
    _write(
        path,
        {
            'schema_version': 1,
            'kind': 'diffraction',
            'uuid': str(other_uuid),
            'created_at': _now(),
        },
    )
    with pytest.raises(ManifestLoadError, match='does not match folder name'):
        load_manifest(path, expected_kind='diffraction', expected_uuid=folder_uuid)


def test_load_manifest_rejects_kind_mismatch(tmp_path: Path) -> None:
    uuid = uuid4()
    path = tmp_path / 'manifest.json'
    _write(
        path,
        {
            'schema_version': 1,
            'kind': 'product',
            'uuid': str(uuid),
            'created_at': _now(),
        },
    )
    with pytest.raises(ManifestLoadError, match='does not match folder kind'):
        load_manifest(path, expected_kind='diffraction', expected_uuid=uuid)


def test_load_manifest_rejects_self_reference(tmp_path: Path) -> None:
    uuid = uuid4()
    path = tmp_path / 'manifest.json'
    _write(
        path,
        {
            'schema_version': 1,
            'kind': 'product',
            'uuid': str(uuid),
            'created_at': _now(),
            'derived_from': [{'kind': 'product', 'uuid': str(uuid)}],
        },
    )
    with pytest.raises(ManifestLoadError, match='self-references'):
        load_manifest(path, expected_kind='product', expected_uuid=uuid)


def test_load_manifest_happy_path(tmp_path: Path) -> None:
    uuid = uuid4()
    path = tmp_path / 'manifest.json'
    _write(
        path,
        {
            'schema_version': 1,
            'kind': 'diffraction',
            'uuid': str(uuid),
            'created_at': _now(),
            'label': 'foo',
            'probe_energy_eV': 8000.0,
        },
    )
    m = load_manifest(path, expected_kind='diffraction', expected_uuid=uuid)
    assert isinstance(m, DiffractionManifest)
    assert m.label == 'foo'
    assert m.probe_energy_eV == 8000.0
