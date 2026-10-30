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

import numpy
import pytest

pytest.importorskip('ptychi')

from ptychi.api import LSQMLOptions  # noqa: E402

from ptychodus.api.assemble import AssembledDiffractionData  # noqa: E402
from ptychodus.api.exit_codes import ExitCode  # noqa: E402
from ptychodus.api.geometry import PixelGeometry  # noqa: E402
from ptychodus.api.object import Object, ObjectCenter  # noqa: E402
from ptychodus.api.probe import ProbeEnsemble  # noqa: E402
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence  # noqa: E402
from ptychodus.api.product import Product, ProductMetadata  # noqa: E402
from ptychodus.api.reconstruct import ReconstructInput  # noqa: E402
from ptychodus.cli import _reconstruct_standard, reconstruct_prepared  # noqa: E402
from ptychodus.cli._reconstruct_common import CancellationToken  # noqa: E402

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
    options = LSQMLOptions()
    options.reconstructor_options.num_epochs = 6
    _reconstruct_standard._write_prepared_directory(
        logging.getLogger('test'),
        directory,
        _assembled_data(),
        _product(far_field=far_field),
        options,
    )

    if not write_options:
        (directory / 'ptychi_options.yaml').unlink()


def _main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> ExitCode:
    monkeypatch.setattr('sys.argv', ['ptychodus-reconstruct', *argv])
    return reconstruct_prepared.main()


def _capture_run(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_run(
        logger: logging.Logger,
        reconstruct_input: ReconstructInput,
        options: object,
        output_directory: Path,
        **kwargs: object,
    ) -> None:
        captured.update(
            reconstruct_input=reconstruct_input,
            options=options,
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
        'ptychi_options.yaml',
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

    code = _main(monkeypatch, '-i', str(prepared), '-o', str(result), '--num-sync-epochs', '3')

    assert code is ExitCode.SUCCESS
    assert captured['reconstruct_input'].diffraction_patterns.shape[0] == NUM_PATTERNS
    assert captured['options'].reconstructor_options.num_epochs == 6
    assert captured['output_directory'] == result
    assert captured['num_sync_epochs'] == 3
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
    assert type(captured['options']).__name__ == 'BHOptions'


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
