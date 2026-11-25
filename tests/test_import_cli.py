"""``ptychodus-import``: prepared data published into a storage root as imported resources."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import h5py
import numpy
import numpy.testing
import pytest

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.diffraction import Polarization
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.fluorescence import ElementMap, FluorescenceDataset
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import (
    FluorescenceFileKeys,
    load_diffraction_data,
    load_fluorescence_data,
    save_diffraction_data,
    save_fluorescence_data,
    save_product,
)
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product
from ptychodus.api.storage import (
    STORAGE_ROOT_ENVIRONMENT_VARIABLE,
    DiffractionManifest,
    FluorescenceManifest,
    Manifest,
    ProductManifest,
    ResourceKind,
    StorageLayout,
    load_manifest,
    resolve_resource_file,
    resolve_storage_inputs,
    save_diffraction_resource,
    save_product_resource,
)
from ptychodus.cli import import_resources
from ptychodus.plugins.nsls2_pyxrf_file import PyXRFFileReader

from parallax_scan_fixtures import METADATA, PIXEL_GEOMETRY, make_positions, make_probe

_OPTIONS_DOCUMENT = 'library: PtyChi\nalgorithm: LSQML\noptions: {}\n'


@pytest.fixture(autouse=True)
def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(import_resources, 'configure_logging', lambda args: None)
    monkeypatch.delenv(STORAGE_ROOT_ENVIRONMENT_VARIABLE, raising=False)


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


def _prepare(directory: Path, *, options_document: str | None = None) -> Path:
    directory.mkdir()
    save_diffraction_data(directory / 'diffraction.h5', _data())
    save_product(directory / 'product.h5', _product())

    if options_document is not None:
        (directory / 'reconstruct_options.yaml').write_text(options_document)

    return directory


def _root(tmp_path: Path) -> Path:
    root = tmp_path / 'storage'
    root.mkdir()
    return root


def _main(monkeypatch: pytest.MonkeyPatch, *argv: object) -> ExitCode:
    monkeypatch.setattr('sys.argv', ['ptychodus-import', *map(str, argv)])
    return import_resources.main()


def _usage_error(monkeypatch: pytest.MonkeyPatch, *argv: object) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _main(monkeypatch, *argv)

    assert excinfo.value.code == 2


def _resource_folders(root: Path) -> list[Path]:
    return sorted(path for path in root.glob('*/*') if path.is_dir())


def _manifest(root: Path, kind: str, uuid: UUID) -> Manifest:
    location = StorageLayout(root).locate(uuid)
    assert location.kind == kind
    return load_manifest(location.manifest_path, expected_kind=kind, expected_uuid=uuid)


def _printed_uuids(capsys: pytest.CaptureFixture[str]) -> list[UUID]:
    return [UUID(line) for line in capsys.readouterr().out.split()]


def test_a_prepared_directory_becomes_a_product_the_reconstructor_resolves(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _root(tmp_path)
    prepared = _prepare(tmp_path / 'fly042', options_document=_OPTIONS_DOCUMENT)

    assert _main(monkeypatch, 'prepared', prepared, '--storage-root', root) is ExitCode.SUCCESS

    diffraction_uuid, product_uuid = _printed_uuids(capsys)
    diffraction = _manifest(root, ResourceKind.DIFFRACTION, diffraction_uuid)
    product = _manifest(root, ResourceKind.PRODUCT, product_uuid)
    assert isinstance(diffraction, DiffractionManifest)
    assert isinstance(product, ProductManifest)
    assert diffraction.run is None and product.run is None
    assert diffraction.label == 'fly042'
    assert diffraction.photon_energy_eV == METADATA.photon_energy_eV
    assert diffraction.detector_distance_m == METADATA.detector_distance_m
    assert [ref.uuid for ref in product.derived_from] == [diffraction_uuid]

    inputs = resolve_storage_inputs(StorageLayout(root), product_uuid)
    numpy.testing.assert_array_equal(
        load_diffraction_data(inputs.diffraction_file).get_patterns(), _data().get_patterns()
    )
    assert inputs.options_file is not None
    assert inputs.options_file.read_text() == _OPTIONS_DOCUMENT
    assert product.files['options'] == 'reconstruct_options.yaml'


def test_a_prepared_directory_without_options_imports_only_its_data(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _root(tmp_path)
    prepared = _prepare(tmp_path / 'prepared')

    _main(monkeypatch, 'prepared', prepared, '--storage-root', root, '--label', 'scan 7')

    diffraction_uuid, product_uuid = _printed_uuids(capsys)
    assert _manifest(root, ResourceKind.DIFFRACTION, diffraction_uuid).label == 'scan 7'  # type: ignore[union-attr]
    product = _manifest(root, ResourceKind.PRODUCT, product_uuid)
    assert product.files == {'product': 'product.h5'}  # type: ignore[union-attr]


def test_malformed_options_import_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = _root(tmp_path)
    prepared = _prepare(tmp_path / 'prepared', options_document='- not\n- a mapping\n')

    assert _main(monkeypatch, 'prepared', prepared, '--storage-root', root) is ExitCode.USAGE
    assert _resource_folders(root) == []


def test_a_lone_diffraction_file_records_only_the_metadata_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _root(tmp_path)
    file = tmp_path / 'scan9.h5'
    save_diffraction_data(file, _data())

    _main(
        monkeypatch,
        'diffraction',
        file,
        '--storage-root',
        root,
        '--photon-energy-eV',
        8800,
        '--polarization',
        'left_circular',
    )

    (uuid,) = _printed_uuids(capsys)
    manifest = _manifest(root, ResourceKind.DIFFRACTION, uuid)
    assert isinstance(manifest, DiffractionManifest)
    assert manifest.label == 'scan9'
    assert manifest.photon_energy_eV == 8800
    assert manifest.polarization is Polarization.LEFT_CIRCULAR
    assert manifest.detector_distance_m is None and manifest.exposure_time_s is None
    assert manifest.run is None


def test_a_product_is_imported_onto_a_stored_diffraction_resource(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _root(tmp_path)
    diffraction = save_diffraction_resource(StorageLayout(root), _data())
    file = tmp_path / 'elsewhere.h5'
    save_product(file, _product())

    _main(monkeypatch, 'product', file, '--diffraction', diffraction, '--storage-root', root)

    (uuid,) = _printed_uuids(capsys)
    assert resolve_storage_inputs(StorageLayout(root), uuid).diffraction_uuid == diffraction


@pytest.mark.parametrize('target', ['product', 'unknown'])
def test_a_product_onto_anything_but_diffraction_is_refused(
    target: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _root(tmp_path)
    layout = StorageLayout(root)
    diffraction = save_diffraction_resource(layout, _data())
    stored_product = save_product_resource(layout, _product(), derived_from=[diffraction])
    file = tmp_path / 'product.h5'
    save_product(file, _product())
    before = _resource_folders(root)

    uuid = stored_product if target == 'product' else uuid4()
    _usage_error(monkeypatch, 'product', file, '--diffraction', uuid, '--storage-root', root)
    assert _resource_folders(root) == before


def test_a_pyxrf_dataset_is_stored_where_its_readers_find_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # PyXRF keeps its fits at its own paths; stored there, the resource would not load.
    root = _root(tmp_path)
    file = tmp_path / 'scan2D_1.h5'
    maps = numpy.random.default_rng(0).uniform(0.0, 100.0, size=(2, 3, 4))

    with h5py.File(file, 'w') as h5_file:
        detsum = h5_file.require_group('xrfmap/detsum')
        detsum.create_dataset('xrf_fit', data=maps)
        detsum.create_dataset('xrf_fit_name', data=['Fe_K', 'Pt_L'], dtype='S20')

    _main(monkeypatch, 'fluorescence', file, '--file-type', 'PyXRF', '--storage-root', root)

    (uuid,) = _printed_uuids(capsys)
    manifest = _manifest(root, ResourceKind.FLUORESCENCE, uuid)
    assert isinstance(manifest, FluorescenceManifest)
    assert manifest.label == 'scan2D_1' and manifest.derived_from == [] and manifest.run is None

    stored = load_fluorescence_data(
        resolve_resource_file(StorageLayout(root), uuid, ResourceKind.FLUORESCENCE)
    )
    assert stored.counts_per_second_path == FluorescenceFileKeys.COUNTS_PER_SECOND

    for stored_map, read_map in zip(stored, PyXRFFileReader().read(file), strict=True):
        assert stored_map.name == read_map.name
        numpy.testing.assert_allclose(stored_map.counts_per_second, read_map.counts_per_second)


def test_an_xrf_maps_dataset_is_read_by_default(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _root(tmp_path)
    file = tmp_path / 'maps.h5'
    save_fluorescence_data(
        file,
        FluorescenceDataset(
            _element_maps=[ElementMap('Fe', numpy.ones((4, 5), dtype=numpy.float32))],
            counts_per_second_path=FluorescenceFileKeys.COUNTS_PER_SECOND,
            channel_names_path=FluorescenceFileKeys.CHANNEL_NAMES,
        ),
    )

    _main(monkeypatch, 'fluorescence', file, '--storage-root', root)

    (uuid,) = _printed_uuids(capsys)
    stored = load_fluorescence_data(
        resolve_resource_file(StorageLayout(root), uuid, ResourceKind.FLUORESCENCE)
    )
    assert [element_map.name for element_map in stored] == ['Fe']


def test_a_directory_that_is_not_prepared_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _root(tmp_path)
    prepared = _prepare(tmp_path / 'prepared')
    (prepared / 'product.h5').unlink()

    _usage_error(monkeypatch, 'prepared', prepared, '--storage-root', root)
    assert _resource_folders(root) == []


def test_a_storage_root_is_required_and_must_exist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = _prepare(tmp_path / 'prepared')

    _usage_error(monkeypatch, 'prepared', prepared)
    _usage_error(monkeypatch, 'prepared', prepared, '--storage-root', tmp_path / 'missing')
    assert not (tmp_path / 'missing').exists()


def test_the_storage_root_defaults_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _root(tmp_path)
    monkeypatch.setenv(STORAGE_ROOT_ENVIRONMENT_VARIABLE, str(root))

    _main(monkeypatch, 'prepared', _prepare(tmp_path / 'prepared'))

    assert len(_printed_uuids(capsys)) == 2
    assert len(_resource_folders(root)) == 2


def test_a_dry_run_reads_but_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _root(tmp_path)
    prepared = _prepare(tmp_path / 'prepared', options_document=_OPTIONS_DOCUMENT)

    assert _main(monkeypatch, 'prepared', prepared, '--storage-root', root, '--dry-run') is (
        ExitCode.SUCCESS
    )
    assert capsys.readouterr().out == ''
    assert _resource_folders(root) == []


def test_an_unreadable_file_fails_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _root(tmp_path)
    file = tmp_path / 'not_diffraction.h5'
    file.write_bytes(b'not HDF5')

    assert _main(monkeypatch, 'diffraction', file, '--storage-root', root) is ExitCode.FAILURE
    assert _resource_folders(root) == []
