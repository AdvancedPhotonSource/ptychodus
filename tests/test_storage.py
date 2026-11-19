"""The storage contract: manifests, the layout, and the chainable savers.

Everything here is ``ptychodus.api.storage``, which the web service reads and the
command-line writers write, so it is tested on a bare install.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4
import json

import numpy
import numpy.testing
import pytest
from pydantic import ValidationError

from ptychodus.api import storage
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.fluorescence import ElementMap, FluorescenceDataset
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import (
    load_diffraction_data,
    load_fluorescence_data,
    load_product,
    save_product,
)
from ptychodus.api.processing import RunRecord, RunState, create_run_record
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
    commit_checkpoint,
    finish_run,
    load_manifest,
    resolve_resource_file,
    resolve_storage_inputs,
    save_campaign_resource,
    save_diffraction_resource,
    save_fluorescence_resource,
    save_manifest,
    save_product_resource,
    start_run,
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

    inputs = resolve_storage_inputs(layout, product)
    assert inputs.diffraction_uuid == diffraction
    numpy.testing.assert_array_equal(
        load_diffraction_data(inputs.diffraction_file).get_patterns(), _data().get_patterns()
    )
    numpy.testing.assert_allclose(
        load_product(inputs.product_file).probes.get_array(), _product().probes.get_array()
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


# ---------------------------------------------------------------------------
# Durable writes
# ---------------------------------------------------------------------------


def _dataset() -> FluorescenceDataset:
    return FluorescenceDataset(
        _element_maps=[
            ElementMap('Fe', numpy.arange(20, dtype=numpy.float32).reshape(4, 5)),
            ElementMap('Cu', numpy.ones((4, 5), dtype=numpy.float32)),
        ],
        counts_per_second_path='/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec',
        channel_names_path='/MAPS/XRF_Analyzed/NNLS/Channel_Names',
    )


def _running(**changes: object) -> RunRecord:
    record = create_run_record(
        RunState.RUNNING, command='ptychodus-reconstruct', options='reconstruct_options.yaml'
    )
    return record.revise(**changes) if changes else record


def _load(layout: StorageLayout, kind: str, uuid: object) -> storage.Manifest:
    location = layout.locate(uuid)  # type: ignore[arg-type]
    return load_manifest(location.manifest_path, expected_kind=kind, expected_uuid=location.uuid)


def test_a_save_is_flushed_before_its_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, str]] = []
    real_fsync_file = storage._fsync_file
    real_fsync_directory = storage._fsync_directory
    real_rename = Path.rename

    def recording_fsync_file(path: Path) -> None:
        events.append(('fsync', path.name))
        real_fsync_file(path)

    def recording_fsync_directory(path: Path) -> None:
        events.append(('fsync', path.name))
        real_fsync_directory(path)

    def recording_rename(self: Path, target: Path) -> Path:
        events.append(('rename', self.name))
        return real_rename(self, target)

    monkeypatch.setattr(storage, '_fsync_file', recording_fsync_file)
    monkeypatch.setattr(storage, '_fsync_directory', recording_fsync_directory)
    monkeypatch.setattr(Path, 'rename', recording_rename)
    uuid = save_diffraction_resource(StorageLayout(tmp_path), _data())

    rename_at = events.index(('rename', str(uuid)))
    flushed_first = {name for kind, name in events[:rename_at] if kind == 'fsync'}
    flushed_after = {name for kind, name in events[rename_at:] if kind == 'fsync'}
    assert {'diffraction.h5', MANIFEST_FILENAME, str(uuid)} <= flushed_first
    assert {str(ResourceKind.DIFFRACTION), StorageLayout.STAGING_DIRECTORY} <= flushed_after


def test_a_manifest_replaced_mid_write_keeps_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uuid = uuid4()
    path = tmp_path / MANIFEST_FILENAME
    before = CampaignManifest(uuid=uuid, created_at=datetime.now(timezone.utc), label='old')
    save_manifest(path, before)

    def fail(source: object, target: object) -> None:
        raise OSError('disk full')

    monkeypatch.setattr(storage.os, 'replace', fail)

    with pytest.raises(OSError, match='disk full'):
        save_manifest(path, before.model_copy(update={'label': 'new'}))

    manifest = load_manifest(path, expected_kind=ResourceKind.CAMPAIGN, expected_uuid=uuid)
    assert isinstance(manifest, CampaignManifest) and manifest.label == 'old'
    assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST_FILENAME]


# ---------------------------------------------------------------------------
# Fluorescence and run records
# ---------------------------------------------------------------------------


def test_a_fluorescence_dataset_saves_and_loads_back(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    diffraction = save_diffraction_resource(layout, _data())
    product = save_product_resource(layout, _product(), derived_from=[diffraction])
    uuid = save_fluorescence_resource(layout, _dataset(), label='2xfm 0123', derived_from=[product])

    manifest = _load(layout, ResourceKind.FLUORESCENCE, uuid)
    assert isinstance(manifest, FluorescenceManifest)
    assert manifest.label == '2xfm 0123'
    assert manifest.derived_from == [DerivedFromRef(kind='product', uuid=product)]
    loaded = load_fluorescence_data(resolve_resource_file(layout, uuid, ResourceKind.FLUORESCENCE))
    assert [emap.name for emap in loaded] == ['Fe', 'Cu']
    numpy.testing.assert_array_equal(loaded[0].counts_per_second, _dataset()[0].counts_per_second)


def test_a_fluorescence_save_with_a_bad_reference_writes_nothing(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)

    with pytest.raises(LayoutError, match='No resource'):
        save_fluorescence_resource(layout, _dataset(), derived_from=[uuid4()])

    assert _resource_folders(tmp_path) == []


def test_savers_record_the_run_and_a_manifest_without_one_loads(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    run = create_run_record(RunState.SUCCEEDED, command='ptychodus-stream')
    diffraction = save_diffraction_resource(layout, _data(), run=run)
    imported = save_product_resource(layout, _product())

    recorded = _load(layout, ResourceKind.DIFFRACTION, diffraction).run
    assert recorded == run
    assert recorded is not None and recorded.state == RunState.SUCCEEDED
    assert _load(layout, ResourceKind.PRODUCT, imported).run is None


def test_a_resource_file_resolves_through_its_manifest(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    diffraction = save_diffraction_resource(layout, _data())

    assert resolve_resource_file(layout, diffraction, ResourceKind.DIFFRACTION) == (
        layout.resource_folder(ResourceKind.DIFFRACTION, diffraction) / 'diffraction.h5'
    )

    with pytest.raises(LayoutError, match='not a product'):
        resolve_resource_file(layout, diffraction, ResourceKind.PRODUCT)


# ---------------------------------------------------------------------------
# Run resources
# ---------------------------------------------------------------------------


def _start_product_run(layout: StorageLayout, input_file: Path) -> tuple[UUID, Path]:
    diffraction = save_diffraction_resource(layout, _data())
    return start_run(
        layout,
        ResourceKind.PRODUCT,
        initial=input_file,
        derived_from=[diffraction],
        options_file_name='reconstruct_options.yaml',
        options_document='library: PtyChi\n',
        log_file_name='reconstruct.log',
        run=_running(),
    )


def test_a_run_starts_in_place_with_a_copy_of_its_input(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    source = save_product_resource(layout, _product())
    input_file = resolve_resource_file(layout, source, ResourceKind.PRODUCT)
    before = (input_file.stat().st_ino, input_file.read_bytes())

    uuid, folder = _start_product_run(layout, input_file)

    manifest = _load(layout, ResourceKind.PRODUCT, uuid)
    assert isinstance(manifest, ProductManifest)
    assert manifest.run is not None and manifest.run.state == RunState.RUNNING
    assert manifest.files == {
        'options': 'reconstruct_options.yaml',
        'log': 'reconstruct.log',
        'product': 'product.000000.h5',
    }
    assert (folder / 'reconstruct_options.yaml').read_text() == 'library: PtyChi\n'
    checkpoint = folder / 'product.000000.h5'
    assert checkpoint.stat().st_ino != before[0]
    assert (input_file.stat().st_ino, input_file.read_bytes()) == before
    assert checkpoint.read_bytes() == before[1]
    assert not any((tmp_path / StorageLayout.STAGING_DIRECTORY).iterdir())


def test_a_run_without_initial_data_waits_for_its_first_checkpoint(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    product = save_product_resource(layout, _product())
    uuid, folder = start_run(
        layout,
        ResourceKind.FLUORESCENCE,
        initial=None,
        derived_from=[product],
        options_file_name='enhance_xrf_options.yaml',
        options_document='algorithm: vspi\n',
        log_file_name='enhance_xrf.log',
        run=_running(command='ptychodus-enhance-xrf', options='enhance_xrf_options.yaml'),
    )

    assert 'fluorescence' not in _load(layout, ResourceKind.FLUORESCENCE, uuid).files  # type: ignore[union-attr]

    with pytest.raises(LayoutError, match='no fluorescence data yet'):
        resolve_resource_file(layout, uuid, ResourceKind.FLUORESCENCE)

    checkpoint = folder / 'fluorescence.000002.h5'
    storage.save_fluorescence_data(checkpoint, _dataset())
    commit_checkpoint(layout, uuid, data_file=checkpoint, run=_running(progress=2, progress_goal=2))
    assert resolve_resource_file(layout, uuid, ResourceKind.FLUORESCENCE) == checkpoint


def test_checkpoints_advance_a_run_until_it_finishes(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    uuid, folder = _start_product_run(
        layout, resolve_resource_file(layout, save_product_resource(layout, _product()), 'product')
    )
    checkpoint = folder / 'product.000005.h5'
    save_product(checkpoint, _product())

    commit_checkpoint(
        layout, uuid, data_file=checkpoint, run=_running(progress=5, progress_goal=10)
    )
    manifest = _load(layout, ResourceKind.PRODUCT, uuid)
    assert manifest.files['product'] == 'product.000005.h5'  # type: ignore[union-attr]
    assert manifest.run is not None and manifest.run.progress == 5

    final = folder / 'product.h5'
    save_product(final, _product())
    finish_run(layout, uuid, data_file=final, run=_running().revise(state=RunState.SUCCEEDED))
    manifest = _load(layout, ResourceKind.PRODUCT, uuid)
    assert manifest.files['product'] == 'product.h5'  # type: ignore[union-attr]
    assert manifest.run is not None and manifest.run.state == RunState.SUCCEEDED

    with pytest.raises(LayoutError, match='not running'):
        commit_checkpoint(layout, uuid, data_file=checkpoint, run=_running())


def test_a_run_refuses_a_missing_or_foreign_file_and_a_wrong_state(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    uuid, folder = _start_product_run(
        layout, resolve_resource_file(layout, save_product_resource(layout, _product()), 'product')
    )

    with pytest.raises(LayoutError, match='not a file in the folder'):
        commit_checkpoint(layout, uuid, data_file=folder / 'product.000009.h5', run=_running())

    with pytest.raises(LayoutError, match='keeps the run running'):
        commit_checkpoint(
            layout,
            uuid,
            data_file=folder / 'product.000000.h5',
            run=_running().revise(state=RunState.FAILED),
        )

    with pytest.raises(LayoutError, match='terminal state'):
        finish_run(layout, uuid, data_file=None, run=_running())


def test_a_run_can_finish_before_any_data(tmp_path: Path) -> None:
    layout = StorageLayout(tmp_path)
    product = save_product_resource(layout, _product())
    uuid, _ = start_run(
        layout,
        ResourceKind.FLUORESCENCE,
        initial=None,
        derived_from=[product],
        options_file_name='enhance_xrf_options.yaml',
        options_document='algorithm: vspi\n',
        log_file_name='enhance_xrf.log',
        run=_running(),
    )

    finish_run(
        layout,
        uuid,
        data_file=None,
        run=_running().revise(state=RunState.CANCELLED),
    )
    manifest = _load(layout, ResourceKind.FLUORESCENCE, uuid)
    assert manifest.run is not None and manifest.run.state == RunState.CANCELLED
    assert 'fluorescence' not in manifest.files  # type: ignore[union-attr]
