"""``ptychodus-xmcd``: an end-to-end run, path guards, dry run and the polarization check."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy
import pytest

from ptychodus.api.diffraction import Polarization
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import load_product, save_product
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.cli import xmcd

_CANVAS_PX = 48


@pytest.fixture(autouse=True)
def _no_root_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ``main()`` from configuring the test process's root logger."""
    monkeypatch.setattr(xmcd, 'configure_logging', lambda args: None)


def _truth() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    rng = numpy.random.default_rng(11)
    y = numpy.arange(_CANVAS_PX)[:, None]
    x = numpy.arange(_CANVAS_PX)[None, :]
    amp = numpy.exp(-(((y - 24) ** 2 + (x - 24) ** 2) / 60.0))
    o_struct = (amp * (0.7 + 0.05 * rng.standard_normal(amp.shape))).astype(numpy.complex128)
    # Radially symmetric domains: a linear magnetic-phase trend would be absorbed by the
    # phase-ramp standardization, as documented.
    domains = numpy.where((y - 24) ** 2 + (x - 24) ** 2 < 30, 1.0, -1.0)
    m_factor = (1.0 + 0.05 * amp) * numpy.exp(0.02j * domains * amp)
    return amp, o_struct, m_factor


def _product(array: numpy.ndarray, *, name: str, polarization: Polarization | None) -> Product:
    pixel_geometry = PixelGeometry(width_m=1.0e-8, height_m=1.0e-8)
    return Product(
        metadata=ProductMetadata(
            name=name,
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=700.0,
            probe_photon_count=1.0e9,
            exposure_time_s=0.1,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
            polarization=polarization,
        ),
        probe_positions=ProbePositionSequence(
            [ProbePosition(index=index, x_m=index * 1.0e-8, y_m=0.0) for index in range(4)]
        ),
        probes=ProbeEnsemble(
            array=numpy.ones((1, 1, 4, 4), dtype=numpy.complex128),
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=array,
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def _write_inputs(
    directory: Path,
    *,
    lcp_polarization: Polarization | None = Polarization.LEFT_CIRCULAR,
    rcp_polarization: Polarization | None = Polarization.RIGHT_CIRCULAR,
) -> tuple[Path, Path]:
    _, o_struct, m_factor = _truth()
    lcp_array = numpy.roll(o_struct / m_factor, shift=(2, -3), axis=(0, 1))
    lcp_file = directory / 'lcp.h5'
    rcp_file = directory / 'rcp.h5'
    save_product(lcp_file, _product(lcp_array, name='lcp', polarization=lcp_polarization))
    save_product(rcp_file, _product(o_struct * m_factor, name='rcp', polarization=rcp_polarization))
    return lcp_file, rcp_file


def _run(monkeypatch: pytest.MonkeyPatch, *argv: str | Path) -> ExitCode:
    monkeypatch.setattr('sys.argv', ['ptychodus-xmcd', *(str(arg) for arg in argv)])
    return xmcd.main()


def test_decomposes_a_product_pair_into_two_products(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lcp_file, rcp_file = _write_inputs(tmp_path)
    structural_file = tmp_path / 'out' / 'structural.h5'
    magnetic_file = tmp_path / 'out' / 'magnetic.h5'

    exit_code = _run(
        monkeypatch,
        '--lcp', lcp_file,
        '--rcp', rcp_file,
        '--structural', structural_file,
        '--magnetic', magnetic_file,
    )  # fmt: skip

    assert exit_code == ExitCode.SUCCESS
    rcp = load_product(rcp_file)
    structural = load_product(structural_file)
    magnetic = load_product(magnetic_file)

    structural_array = structural.object_.get_array()[0]
    magnetic_array = magnetic.object_.get_array()[0]
    numpy.testing.assert_allclose(
        structural_array * magnetic_array, rcp.object_.get_array()[0], atol=1e-9
    )

    # Each output matches its truth up to the global factor the standardization absorbs.
    amp, o_struct, m_factor = _truth()
    support = amp > 0.2
    for estimate, truth in ((structural_array, o_struct), (magnetic_array, m_factor)):
        factor = numpy.vdot(estimate[support], truth[support]) / numpy.vdot(
            estimate[support], estimate[support]
        )
        numpy.testing.assert_allclose(factor * estimate[support], truth[support], atol=5e-3)

    for product, component in ((structural, 'structural'), (magnetic, 'magnetic')):
        assert product.metadata.name == f'rcp-{component}'
        assert product.metadata.polarization is None
        numpy.testing.assert_array_equal(
            product.probe_positions.get_coordinates_x_m(),
            rcp.probe_positions.get_coordinates_x_m(),
        )


@pytest.mark.parametrize('collision', ['input', 'outputs'])
def test_refuses_an_output_that_collides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, collision: str
) -> None:
    lcp_file, rcp_file = _write_inputs(tmp_path)
    structural_file = rcp_file if collision == 'input' else tmp_path / 'same.h5'
    magnetic_file = tmp_path / 'same.h5' if collision == 'outputs' else tmp_path / 'magnetic.h5'

    with pytest.raises(SystemExit) as excinfo:
        _run(
            monkeypatch,
            '--lcp', lcp_file,
            '--rcp', rcp_file,
            '--structural', structural_file,
            '--magnetic', magnetic_file,
        )  # fmt: skip

    assert excinfo.value.code == ExitCode.USAGE
    assert not magnetic_file.exists()


def test_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lcp_file, rcp_file = _write_inputs(tmp_path)
    structural_file = tmp_path / 'structural.h5'
    magnetic_file = tmp_path / 'magnetic.h5'

    exit_code = _run(
        monkeypatch,
        '--lcp', lcp_file,
        '--rcp', rcp_file,
        '--structural', structural_file,
        '--magnetic', magnetic_file,
        '--dry-run',
    )  # fmt: skip

    assert exit_code == ExitCode.SUCCESS
    assert not structural_file.exists()
    assert not magnetic_file.exists()


def test_warns_when_recorded_polarizations_look_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    lcp_file, rcp_file = _write_inputs(
        tmp_path,
        lcp_polarization=Polarization.RIGHT_CIRCULAR,
        rcp_polarization=None,
    )

    with caplog.at_level(logging.WARNING, logger='xmcd'):
        _run(
            monkeypatch,
            '--lcp', lcp_file,
            '--rcp', rcp_file,
            '--structural', tmp_path / 'structural.h5',
            '--magnetic', tmp_path / 'magnetic.h5',
            '--dry-run',
        )  # fmt: skip

    swapped = [record for record in caplog.records if 'swapped' in record.getMessage()]
    assert len(swapped) == 1
    assert str(lcp_file) in swapped[0].getMessage()
