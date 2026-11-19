"""The seam between a driver's ``--prepare-only`` and ``ptychodus-reconstruct``.

The prepared directory is written by one command and read by another, possibly on another
machine and another day, so the two halves are exercised together here: the directory is
written by the same function the drivers call, and read back by the command's ``main()``.
pty-chi itself is never run; the reconstruction tail is stubbed at the one call that would
start it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychodus.api.assemble import AssembledDiffractionData  # noqa: E402
from ptychodus.api.exit_codes import ExitCode  # noqa: E402
from ptychodus.api.plugins import PluginRegistry  # noqa: E402
from ptychodus.api.geometry import PixelGeometry  # noqa: E402
from ptychodus.api.object import Object, ObjectCenter  # noqa: E402
from ptychodus.api.probe import ProbeEnsemble  # noqa: E402
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence  # noqa: E402
from ptychodus.api.product import Product, ProductMetadata  # noqa: E402
from ptychodus.api.io import load_product, save_product  # noqa: E402
from ptychodus.api.processing import RunState  # noqa: E402
from ptychodus.api.reconstruct import ReconstructInput, ReconstructOutput  # noqa: E402
from ptychodus.api.storage import (  # noqa: E402
    STORAGE_ROOT_ENVIRONMENT_VARIABLE,
    DerivedFromRef,
    ProductManifest,
    ResourceKind,
    StorageLayout,
    load_manifest,
    save_diffraction_resource,
    save_product_resource,
)
from ptychodus.cli import CancellationToken  # noqa: E402
from ptychodus.cli import _reconstruct_standard, reconstruct_prepared  # noqa: E402
from ptychodus.cli._reconstruct_common import (  # noqa: E402
    ResolvedReconstruction,
    default_reconstruction,
)

PIXEL_M = 1.0e-9
NUM_PATTERNS = 4
PATTERN_PX = 8


@pytest.fixture(autouse=True)
def _neutral_launcher_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear every rank source, so the writes are not skipped under `srun` or `torchrun`."""
    for name in ('RANK', 'SLURM_PROCID', 'LOCAL_WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_process_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``main()`` from claiming the test process's signals and root logger."""
    monkeypatch.setattr(reconstruct_prepared, 'install_signal_handlers', CancellationToken)
    monkeypatch.setattr(reconstruct_prepared, 'configure_logging', lambda args: None)
    monkeypatch.setattr(reconstruct_prepared, 'open_action_log', lambda *args, **kwargs: None)


def _assembled_data() -> AssembledDiffractionData:
    rng = numpy.random.default_rng(0)
    return AssembledDiffractionData(
        indexes=numpy.arange(NUM_PATTERNS),
        patterns=rng.random((NUM_PATTERNS, PATTERN_PX, PATTERN_PX)).astype(numpy.float32),
        pixel_geometry=PixelGeometry(width_m=7.5e-5, height_m=7.5e-5),
        bad_pixels=numpy.zeros((PATTERN_PX, PATTERN_PX), dtype=numpy.bool_),
    )


def _product(*, far_field: bool = True) -> Product:
    rng = numpy.random.default_rng(1)
    pixel_geometry = PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M)
    return Product(
        metadata=ProductMetadata(
            name='prepared',
            comments='',
            detector_distance_m=2.5,
            photon_energy_eV=10_000.0,
            probe_photon_count=1.0e6,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
            far_field=far_field,
        ),
        probe_positions=ProbePositionSequence(
            [ProbePosition(index=i, x_m=i * PIXEL_M, y_m=0.0) for i in range(NUM_PATTERNS)]
        ),
        probes=ProbeEnsemble(
            array=rng.standard_normal((1, 1, PATTERN_PX, PATTERN_PX)).astype(numpy.complex64),
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.ones((1, 24, 24), dtype=numpy.complex64),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
            layer_spacing_m=[],
        ),
        losses=[],
    )


def _prepare(directory: Path, *, far_field: bool = True, write_options: bool = True) -> None:
    libraries = PluginRegistry.load_plugins().reconstructor_libraries
    resolved = default_reconstruction(libraries, far_field=far_field)
    resolved.options.task.reconstructor_options.num_epochs = 6  # type: ignore[attr-defined]
    _reconstruct_standard._write_prepared_directory(
        logging.getLogger('test'),
        directory,
        _assembled_data(),
        _product(far_field=far_field),
        resolved,
    )

    if not write_options:
        (directory / 'reconstruct_options.yaml').unlink()


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> ExitCode:
    monkeypatch.setattr('sys.argv', ['ptychodus-reconstruct', *argv])
    return reconstruct_prepared.main()


def _capture_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run(
        logger: logging.Logger,
        resolved: ResolvedReconstruction,
        reconstruct_input: ReconstructInput,
        output_directory: Path,
        **kwargs: object,
    ) -> None:
        captured.update(
            resolved=resolved,
            reconstruct_input=reconstruct_input,
            task=getattr(resolved.options, 'task', None),
            output_directory=output_directory,
            **kwargs,
        )

    monkeypatch.setattr(reconstruct_prepared, 'run_reconstruction', fake_run)
    return captured


def test_preparing_writes_exactly_the_three_inputs(tmp_path: Path) -> None:
    _prepare(tmp_path / 'prepared')

    assert sorted(path.name for path in (tmp_path / 'prepared').iterdir()) == [
        'diffraction.h5',
        'product.h5',
        'reconstruct_options.yaml',
    ]


def test_preparing_over_an_earlier_reconstruction_warns(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    (tmp_path / 'product.000100.h5').write_bytes(b'')

    with caplog.at_level('WARNING', logger='test'):
        _prepare(tmp_path)

    assert any('initial product' in record.getMessage() for record in caplog.records)


def test_a_prepared_directory_reconstructs_with_its_staged_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = tmp_path / 'prepared'
    result = tmp_path / 'result'
    _prepare(prepared)
    before = {path.name: path.stat().st_mtime_ns for path in prepared.iterdir()}
    captured = _capture_run(monkeypatch)

    code = _main(monkeypatch, '-i', str(prepared), '-o', str(result))

    assert code is ExitCode.SUCCESS
    assert captured['reconstruct_input'].diffraction_patterns.shape[0] == NUM_PATTERNS
    assert captured['task'].reconstructor_options.num_epochs == 6
    assert captured['output_directory'] == result
    assert result.is_dir()
    # The prepared directory is an input and stays one.
    assert {path.name: path.stat().st_mtime_ns for path in prepared.iterdir()} == before


def test_a_near_field_product_without_options_falls_back_to_bh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = tmp_path / 'prepared'
    _prepare(prepared, far_field=False, write_options=False)
    captured = _capture_run(monkeypatch)

    code = _main(monkeypatch, '-i', str(prepared), '-o', str(tmp_path / 'result'))

    assert code is ExitCode.SUCCESS
    assert type(captured['task']).__name__ == 'BHOptions'


def test_a_dry_run_writes_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepared = tmp_path / 'prepared'
    result = tmp_path / 'result'
    _prepare(prepared)
    captured = _capture_run(monkeypatch)

    code = _main(monkeypatch, '-i', str(prepared), '-o', str(result), '--dry-run')

    assert code is ExitCode.SUCCESS
    assert not captured
    assert not result.exists()


def test_reconstructing_in_place_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The final product.h5 would overwrite the initial one it was refined from.
    _prepare(tmp_path)

    with pytest.raises(SystemExit) as caught:
        _main(monkeypatch, '-i', str(tmp_path), '-o', str(tmp_path / '.'))

    assert caught.value.code == 2


def test_a_directory_missing_its_product_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prepared = tmp_path / 'prepared'
    _prepare(prepared)
    (prepared / 'product.h5').unlink()

    with pytest.raises(SystemExit) as caught:
        _main(monkeypatch, '-i', str(prepared), '-o', str(tmp_path / 'result'))

    assert caught.value.code == 2


def _store(root: Path) -> UUID:
    """A product in a storage root, derived from its diffraction data; return its uuid."""
    layout = StorageLayout(root)
    diffraction = save_diffraction_resource(layout, _assembled_data(), label='scan 1')
    return save_product_resource(layout, _product(), derived_from=[diffraction])


def _stored_run(monkeypatch: pytest.MonkeyPatch, mode: str) -> dict[str, Any]:
    """Stand in for the reconstruction tail, writing what it writes and ending as `mode`.

    Two outputs are yielded and checkpointed through ``on_sync``; then the run completes,
    is cancelled (after its outputs, or before any), or raises.
    """
    captured: dict[str, Any] = {'commits': []}

    def fake_run(
        logger: logging.Logger,
        resolved: ResolvedReconstruction,
        reconstruct_input: ReconstructInput,
        output_directory: Path,
        *,
        cancellation: CancellationToken,
        on_sync: Any = None,
    ) -> Product | None:
        captured.update(reconstruct_input=reconstruct_input, output_directory=output_directory)

        if mode == 'cancelled before output':
            cancellation.cancel('SIGTERM')
            return None

        product = reconstruct_input.product

        for progress in (1, 2):
            checkpoint = output_directory / f'product.{progress:06d}.h5'
            save_product(checkpoint, product)
            on_sync(ReconstructOutput(product, progress=progress, progress_goal=4), checkpoint)
            captured['commits'].append(_manifest(output_directory).files['product'])

        if mode == 'failed':
            raise FloatingPointError('loss is nan')

        if mode == 'cancelled':
            cancellation.cancel('SIGTERM')

        save_product(output_directory / 'product.h5', product)
        return product

    monkeypatch.setattr(reconstruct_prepared, 'run_reconstruction', fake_run)
    return captured


def _manifest(folder: Path) -> ProductManifest:
    manifest = load_manifest(
        folder / 'manifest.json',
        expected_kind=ResourceKind.PRODUCT,
        expected_uuid=UUID(folder.name),
    )
    assert isinstance(manifest, ProductManifest)
    return manifest


def _main_stored(monkeypatch: pytest.MonkeyPatch, root: Path, product: UUID) -> ExitCode:
    return _main(monkeypatch, '--storage-root', str(root), '--product', str(product))


@pytest.mark.parametrize(
    ('mode', 'code', 'state', 'result'),
    [
        ('succeeded', ExitCode.SUCCESS, RunState.SUCCEEDED, 'product.h5'),
        ('cancelled', ExitCode.CANCELLED, RunState.CANCELLED, 'product.h5'),
        ('cancelled before output', ExitCode.CANCELLED, RunState.CANCELLED, 'product.000000.h5'),
        ('failed', ExitCode.FAILURE, RunState.FAILED, 'product.000002.h5'),
    ],
)
def test_a_stored_product_reconstructs_into_a_new_resource(
    mode: str,
    code: ExitCode,
    state: RunState,
    result: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    root = tmp_path / 'storage'
    product = _store(root)
    layout = StorageLayout(root)
    input_file = layout.resource_folder(ResourceKind.PRODUCT, product) / 'product.h5'
    input_bytes = input_file.read_bytes()
    captured = _stored_run(monkeypatch, mode)

    assert _main_stored(monkeypatch, root, product) is code

    folder = captured['output_directory']
    assert folder.parent == layout.kind_dir(ResourceKind.PRODUCT)
    assert capsys.readouterr().out.strip() == folder.name
    assert captured['reconstruct_input'].diffraction_patterns.shape[0] == NUM_PATTERNS

    manifest = _manifest(folder)
    diffraction = _manifest_source(layout, product)
    assert manifest.derived_from == [
        DerivedFromRef(kind='diffraction', uuid=diffraction),
        DerivedFromRef(kind='product', uuid=product),
    ]
    assert manifest.files['product'] == result
    assert manifest.files['options'] == 'reconstruct_options.yaml'
    assert manifest.files['log'] == 'reconstruct.log'
    assert manifest.run is not None
    assert manifest.run.state == state
    assert manifest.run.command == 'ptychodus-reconstruct'
    assert manifest.run.options == 'reconstruct_options.yaml'
    assert manifest.run.reason == ('FloatingPointError' if mode == 'failed' else None)
    assert (folder / 'reconstruct_options.yaml').is_file()
    assert (folder / 'product.000000.h5').read_bytes() == input_bytes
    assert input_file.read_bytes() == input_bytes
    load_product(folder / result)

    if mode != 'cancelled before output':
        assert captured['commits'] == ['product.000001.h5', 'product.000002.h5']
        assert (manifest.run.progress, manifest.run.progress_goal) == (2, 4)


def _manifest_source(layout: StorageLayout, product: UUID) -> UUID:
    manifest = load_manifest(
        layout.manifest_path(ResourceKind.PRODUCT, product),
        expected_kind=ResourceKind.PRODUCT,
        expected_uuid=product,
    )
    assert isinstance(manifest, ProductManifest)
    return manifest.derived_from[0].uuid


def test_the_storage_root_defaults_to_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    product = _store(tmp_path / 'storage')
    captured = _stored_run(monkeypatch, 'succeeded')
    monkeypatch.setenv(STORAGE_ROOT_ENVIRONMENT_VARIABLE, str(tmp_path / 'storage'))

    code = _main(monkeypatch, '--product', str(product))

    assert code is ExitCode.SUCCESS
    assert captured['output_directory'].parent.parent == (tmp_path / 'storage').resolve()


def test_a_stored_dry_run_creates_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / 'storage'
    product = _store(root)
    before = sorted(path.name for path in (root / ResourceKind.PRODUCT).iterdir())

    assert _main(
        monkeypatch, '--storage-root', str(root), '--product', str(product), '--dry-run'
    ) is (ExitCode.SUCCESS)
    assert sorted(path.name for path in (root / ResourceKind.PRODUCT).iterdir()) == before


@pytest.mark.parametrize(
    'case',
    [
        'no root',
        'unknown uuid',
        'both inputs',
        'product with -o',
        'input without -o',
        'multi-process',
    ],
)
def test_a_bad_product_request_is_a_usage_error(
    case: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    product = _store(tmp_path / 'storage')
    _prepare(tmp_path / 'prepared')
    monkeypatch.delenv(STORAGE_ROOT_ENVIRONMENT_VARIABLE, raising=False)
    root = ['--storage-root', str(tmp_path / 'storage')]
    output = ['-o', str(tmp_path / 'result')]

    if case == 'multi-process':
        monkeypatch.setattr(reconstruct_prepared, 'environment_world_size', lambda: 2)

    argv = {
        'no root': ['--product', str(product)],
        'unknown uuid': [*root, '--product', str(uuid4())],
        'both inputs': ['-i', str(tmp_path / 'prepared'), '--product', str(product), *output],
        'product with -o': [*root, '--product', str(product), *output],
        'input without -o': ['-i', str(tmp_path / 'prepared')],
        'multi-process': [*root, '--product', str(product)],
    }[case]

    with pytest.raises(SystemExit) as caught:
        _main(monkeypatch, *argv)

    assert caught.value.code == 2
