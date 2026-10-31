"""The storage contract: manifests, the layout, and the chainable savers.

Everything here is ``ptychodus.api.storage``, which the web service reads and the
command-line writers write, so it is tested on a bare install.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json

import numpy
import numpy.testing
import pytest
from pydantic import ValidationError

from ptychodus.api import storage
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import load_diffraction_data, load_product
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product
from ptychodus.api.storage import (
    HDF5_OWNED_KEYS,
    MANIFEST_FILENAME,
    CampaignManifest,
    DerivedFromRef,
    DiffractionManifest,
    FluorescenceManifest,
    LayoutError,
    ManifestLoadError,
    ProductManifest,
    ResourceKind,
    StorageLayout,
    load_manifest,
    resolve_storage_inputs,
    save_campaign_resource,
    save_diffraction_resource,
    save_manifest,
    save_product_resource,
)

from parallax_scan_fixtures import METADATA, PIXEL_GEOMETRY, make_positions, make_probe

# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------


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
        DerivedFromRef(kind='campaign', uuid=uuid4())


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
            'photon_energy_eV': 8000.0,
        },
    )
    m = load_manifest(path, expected_kind='diffraction', expected_uuid=uuid)
    assert isinstance(m, DiffractionManifest)
    assert m.label == 'foo'
    assert m.photon_energy_eV == 8000.0


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_resource_folder_resolution(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    uuid = uuid4()
    folder = layout.resource_folder('product', uuid)
    assert folder == tmp_path.resolve() / 'product' / str(uuid)


def test_manifest_path_resolution(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    uuid = uuid4()
    m = layout.manifest_path('fluorescence', uuid)
    assert m.name == 'manifest.json'
    assert m.parent == tmp_path.resolve() / 'fluorescence' / str(uuid)


def test_parse_manifest_path_valid(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    layout.ensure_kind_dirs()
    uuid = uuid4()
    folder = tmp_path / 'campaign' / str(uuid)
    folder.mkdir()
    m = folder / 'manifest.json'
    m.touch()
    loc = layout.parse_manifest_path(m)
    assert loc.kind == 'campaign'
    assert loc.uuid == uuid
    assert loc.folder == folder.resolve()


def test_parse_manifest_rejects_outside_root(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path / 'inside')
    elsewhere = tmp_path / 'outside' / 'campaign' / str(uuid4()) / 'manifest.json'
    elsewhere.parent.mkdir(parents=True)
    elsewhere.touch()
    with pytest.raises(LayoutError):
        layout.parse_manifest_path(elsewhere)


def test_parse_manifest_rejects_non_uuid_folder(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    layout.ensure_kind_dirs()
    bad = tmp_path / 'product' / 'not-a-uuid' / 'manifest.json'
    bad.parent.mkdir()
    bad.touch()
    with pytest.raises(LayoutError):
        layout.parse_manifest_path(bad)


def test_parse_manifest_rejects_unknown_kind(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    layout.ensure_kind_dirs()
    bad = tmp_path / 'mystery' / str(uuid4()) / 'manifest.json'
    bad.parent.mkdir(parents=True)
    bad.touch()
    with pytest.raises(LayoutError):
        layout.parse_manifest_path(bad)


def test_iter_manifest_paths_orders_campaigns_first(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    layout.ensure_kind_dirs()

    for kind in ('product', 'diffraction', 'campaign', 'fluorescence'):
        folder = tmp_path / kind / str(uuid4())
        folder.mkdir()
        (folder / 'manifest.json').touch()

    paths = layout.iter_manifest_paths()
    kinds = [p.parent.parent.name for p in paths]
    # Campaign first; then diffraction, product, fluorescence in any order — the
    # spec orders them as campaign, diffraction, product, fluorescence.
    assert kinds[0] == 'campaign'
    assert kinds == ['campaign', 'diffraction', 'product', 'fluorescence']


# ---------------------------------------------------------------------------
# Chainable savers
# ---------------------------------------------------------------------------


def _data() -> AssembledDiffractionData:
    return AssembledDiffractionData(
        indexes=numpy.arange(3),
        patterns=numpy.arange(48, dtype=numpy.uint16).reshape(3, 4, 4),
        pixel_geometry=PixelGeometry(width_m=75e-6, height_m=75e-6),
        bad_pixels=numpy.zeros((4, 4), dtype=bool),
    )


def _product() -> Product:
    return Product(
        metadata=METADATA,
        probe_positions=ProbePositionSequence(make_positions()),
        probes=ProbeEnsemble.from_probe(make_probe(0.0)),
        object_=Object(
            array=numpy.ones((1, 8, 8), dtype=complex),
            pixel_geometry=PIXEL_GEOMETRY,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def _resource_folders(root: Path) -> list[Path]:
    return sorted(path for path in root.glob('*/*') if path.is_dir())


def test_saved_resources_chain_and_load_back(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    campaign = save_campaign_resource(layout, label='fly001 demo')
    diffraction = save_diffraction_resource(
        layout, _data(), label='scan 42', campaign=campaign, tomography_angle_deg=12.5
    )
    product = save_product_resource(layout, _product(), derived_from=[diffraction])

    assert layout.locate(campaign).kind == ResourceKind.CAMPAIGN
    assert layout.locate(diffraction).kind == ResourceKind.DIFFRACTION
    location = layout.locate(product)
    assert location == layout.parse_manifest_path(location.manifest_path)

    diffraction_manifest = load_manifest(
        layout.manifest_path(ResourceKind.DIFFRACTION, diffraction),
        expected_kind=ResourceKind.DIFFRACTION,
        expected_uuid=diffraction,
    )
    product_manifest = load_manifest(
        location.manifest_path, expected_kind=ResourceKind.PRODUCT, expected_uuid=product
    )
    assert isinstance(diffraction_manifest, DiffractionManifest)
    assert isinstance(product_manifest, ProductManifest)
    assert diffraction_manifest.campaign_uuid == campaign
    assert diffraction_manifest.label == 'scan 42'
    assert diffraction_manifest.tomography_angle_deg == 12.5
    assert product_manifest.derived_from == [DerivedFromRef(kind='diffraction', uuid=diffraction)]

    diffraction_file, product_file = resolve_storage_inputs(layout, product)
    numpy.testing.assert_array_equal(
        load_diffraction_data(diffraction_file).get_patterns(), _data().get_patterns()
    )
    numpy.testing.assert_allclose(
        load_product(product_file).probes.get_array(), _product().probes.get_array()
    )
    assert not any((tmp_path / StorageLayout.STAGING_DIRECTORY).iterdir())


def test_a_bad_reference_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    campaign = save_campaign_resource(layout, label='a')
    diffraction = save_diffraction_resource(layout, _data())
    before = _resource_folders(tmp_path)

    with pytest.raises(LayoutError, match='not a campaign'):
        save_diffraction_resource(layout, _data(), campaign=diffraction)

    with pytest.raises(LayoutError, match='derived from campaign'):
        save_product_resource(layout, _product(), derived_from=[campaign])

    with pytest.raises(LayoutError, match='No resource'):
        save_product_resource(layout, _product(), derived_from=[uuid4()])

    assert _resource_folders(tmp_path) == before
    assert not any((tmp_path / StorageLayout.STAGING_DIRECTORY).iterdir())


def test_a_failed_save_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = StorageLayout(tmp_path)
    diffraction = save_diffraction_resource(layout, _data())

    def fail(file_path: Path, product: Product) -> None:
        file_path.write_bytes(b'partial')
        raise OSError('disk full')

    monkeypatch.setattr(storage, 'save_product', fail)

    with pytest.raises(OSError, match='disk full'):
        save_product_resource(layout, _product(), derived_from=[diffraction])

    assert not (tmp_path / ResourceKind.PRODUCT).exists() or not any(
        (tmp_path / ResourceKind.PRODUCT).iterdir()
    )
    assert not any((tmp_path / StorageLayout.STAGING_DIRECTORY).iterdir())


def test_two_saves_never_share_a_folder(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    first = save_diffraction_resource(layout, _data())
    second = save_diffraction_resource(layout, _data())

    assert first != second
    assert len(_resource_folders(tmp_path)) == 2


def test_a_saved_manifest_loads_back(tmp_path: Path) -> None:
    uuid = uuid4()
    path = tmp_path / MANIFEST_FILENAME
    save_manifest(
        path, CampaignManifest(uuid=uuid, created_at=datetime.now(timezone.utc), label='a')
    )
    manifest = load_manifest(path, expected_kind=ResourceKind.CAMPAIGN, expected_uuid=uuid)

    assert isinstance(manifest, CampaignManifest) and manifest.label == 'a'


def test_storage_inputs_resolve_only_a_stored_product(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    diffraction = save_diffraction_resource(layout, _data())
    orphan = save_product_resource(layout, _product())

    with pytest.raises(LayoutError, match='No resource'):
        resolve_storage_inputs(layout, uuid4())

    with pytest.raises(LayoutError, match='not a product'):
        resolve_storage_inputs(layout, diffraction)

    with pytest.raises(LayoutError, match='0 diffraction'):
        resolve_storage_inputs(layout, orphan)
