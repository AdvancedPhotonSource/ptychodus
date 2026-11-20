"""pty-chi places every probe where ptychodus does.

ptychodus puts a probe's beam axis, pixel ``N // 2``, on its probe position; pty-chi puts
the probe's midpoint there and snaps the integer part of each position against an
origin. The adapter converts between the two at the boundary. These tests compare
pty-chi's own forward model, fed through that conversion, with ptychodus's simulator on
one product, for even and odd probes at integer and fractional positions of both
parities -- the case an origin of the wrong parity jitters by half a pixel.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

pytest.importorskip('ptychi')

import numpy  # noqa: E402

from ptychodus.api.geometry import PixelGeometry  # noqa: E402
from ptychodus.api.object import Object, ObjectCenter  # noqa: E402
from ptychodus.api.probe import ProbeEnsemble  # noqa: E402
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence  # noqa: E402
from ptychodus.api.product import Product, ProductMetadata  # noqa: E402
from ptychodus.api.simulate.diffraction import generate_diffraction_data  # noqa: E402
from ptychodus.api.reconstruct import ReconstructInput  # noqa: E402
from ptychodus.plugins.ptychi.task import (  # noqa: E402
    _detector_response,
    _to_ptychi_positions_px,
    align_probe_gauge_with_ptychi,
    align_task_options_with_product,
    apply_probe_gauge_to_options,
)

_PIXEL_M = 1.0e-8
_OBJECT_PX = 48
# Far enough that the Fresnel transform ptychodus simulates with and the plain FFT
# pty-chi propagates with agree to round-off: N**2 times the pixel Fresnel number is
# about 1e-9 here.
_DETECTOR_DISTANCE_M = 1.0e3
_PHOTON_ENERGY_EV = 10000.0

# Relative to the object's center, which on the even canvas sits between two pixels:
# these put the axis on whole object pixels of both parities, where pty-chi cuts the
# patch at a corner and applies no subpixel shift at all.
_WHOLE_PIXEL_OFFSETS_PX = [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5), (-2.5, 3.5)]
_FRACTIONAL_OFFSETS_PX = [(0.0, 0.0), (1.0, 0.0), (0.25, -0.75), (1.4, 0.6)]


def _rough_probe(probe_px: int) -> numpy.ndarray:
    """Asymmetric, random in phase and bright out to the window's edge."""
    rng = numpy.random.default_rng(probe_px)
    yy, xx = numpy.mgrid[:probe_px, :probe_px] - probe_px // 2
    envelope = numpy.exp(-((xx - 0.7) ** 2 + 2 * (yy + 0.3) ** 2) / 6.0)
    return envelope * numpy.exp(1j * rng.uniform(-1.0, 1.0, envelope.shape))


def _smooth_probe(probe_px: int) -> numpy.ndarray:
    """Band-limited and dark at the window's edge, so a Fourier shift translates it."""
    yy, xx = numpy.mgrid[:probe_px, :probe_px] - probe_px // 2
    envelope = numpy.exp(-(xx**2 + yy**2) / (2 * (probe_px / 10) ** 2))
    return envelope * numpy.exp(1j * 0.05 * (xx**2 + yy**2) / probe_px)


def _product(probe: numpy.ndarray, offsets_px: list[tuple[float, float]]) -> Product:
    rng = numpy.random.default_rng(7)
    obj = numpy.exp(1j * rng.uniform(-1.0, 1.0, (_OBJECT_PX, _OBJECT_PX))) * rng.uniform(
        0.5, 1.0, (_OBJECT_PX, _OBJECT_PX)
    )
    pixel_geometry = PixelGeometry(width_m=_PIXEL_M, height_m=_PIXEL_M)
    positions = [
        ProbePosition(index=index, x_m=dx * _PIXEL_M, y_m=dy * _PIXEL_M)
        for index, (dx, dy) in enumerate(offsets_px)
    ]
    return Product(
        metadata=ProductMetadata(
            name='registration',
            comments='',
            detector_distance_m=_DETECTOR_DISTANCE_M,
            photon_energy_eV=_PHOTON_ENERGY_EV,
            probe_photon_count=1.0e6,
            exposure_time_s=1.0,
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeEnsemble(
            array=probe[numpy.newaxis, numpy.newaxis].astype(complex),
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=obj[numpy.newaxis].astype(complex),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def _ptychi_patterns(product: Product, measured: numpy.ndarray) -> numpy.ndarray:
    """Intensities pty-chi's forward model predicts, centered as ptychodus stores them."""
    import torch
    from ptychi.api import LSQMLOptions
    from ptychi.api.task import PtychographyTask

    options = align_task_options_with_product(LSQMLOptions(), product)
    object_geometry = product.object_.get_geometry()
    position_x_px, position_y_px = _to_ptychi_positions_px(
        object_geometry.map_probe_positions_to_object_x_px(product.probe_positions),
        object_geometry.map_probe_positions_to_object_y_px(product.probe_positions),
        product.probes,
    )
    task = PtychographyTask(
        options,
        diffraction_data=measured,
        object_data=product.object_.get_array(),
        probe_data=product.probes.get_array(),
        probe_position_x_px=position_x_px,
        probe_position_y_px=position_y_px,
    )

    with task, torch.no_grad():
        reconstructor = task.reconstructor
        assert reconstructor is not None
        indices = torch.arange(len(product.probe_positions))
        predicted = cast(Any, reconstructor).forward_model.forward(indices).cpu().numpy()

    return numpy.fft.fftshift(predicted, axes=(-2, -1))


def _assert_patterns_agree(product: Product, *, rtol: float) -> None:
    measured = numpy.array(generate_diffraction_data(product).get_patterns(), dtype=float)
    predicted = _ptychi_patterns(product, measured)

    # Compared per pattern up to the propagators' normalizations, which differ by a
    # constant; a probe placed half a pixel off changes the pattern's shape.
    for got, want in zip(predicted, measured):
        numpy.testing.assert_allclose(got * (want.sum() / got.sum()), want, atol=rtol * want.max())


@pytest.mark.parametrize('probe_px', [8, 9, 16])
def test_a_whole_pixel_axis_lands_where_ptychodus_puts_it(probe_px: int) -> None:
    """Exact for any probe, because neither side shifts it: a half-pixel registration
    error, or a snap that alternates with the position's parity, fails here."""
    _assert_patterns_agree(_product(_rough_probe(probe_px), _WHOLE_PIXEL_OFFSETS_PX), rtol=1e-5)


@pytest.mark.parametrize('probe_px', [16, 17, 32])
def test_fractional_positions_agree_for_a_band_limited_probe(probe_px: int) -> None:
    """ptychodus floors the corner and pty-chi rounds it, so the two Fourier-shift the
    probe by residuals a pixel apart. For a field that is band-limited and dark at the
    window's edge both are the same translation; a rough probe wraps differently."""
    _assert_patterns_agree(_product(_smooth_probe(probe_px), _FRACTIONAL_OFFSETS_PX), rtol=1e-4)


def _maps(shape: tuple[int, int]) -> tuple[numpy.ndarray, numpy.ndarray]:
    rng = numpy.random.default_rng(11)
    flat_field = rng.uniform(0.6, 1.4, shape)
    background = rng.uniform(0.0, 0.05, shape)
    return flat_field, background


# Even sizes only: pty-chi moves centered patterns and maps to its corner frame with
# fftshift, which on an odd grid lands the DC pixel at N - 1 rather than 0, so there its
# gain and offset sit a pixel from the intensity its model predicts. The data and maps
# stay consistent with each other, which a reconstruction absorbs as a probe tilt, but a
# pixel-for-pixel comparison against ptychodus cannot pass.
@pytest.mark.parametrize('probe_px', [8, 16])
def test_carried_maps_reach_ptychi_as_its_detector_response(probe_px: int) -> None:
    """pty-chi's ``intensity * gain + offset`` with the adapter's map reproduces what
    ptychodus simulates with the same flat field and background."""
    import torch
    from ptychi.api import LSQMLOptions
    from ptychi.api.task import PtychographyTask

    product = _product(_rough_probe(probe_px), _WHOLE_PIXEL_OFFSETS_PX)
    flat_field, background = _maps((probe_px, probe_px))
    measured = numpy.array(
        generate_diffraction_data(
            product, background=background, flat_field=flat_field
        ).get_patterns(),
        dtype=float,
    )
    parameters = ReconstructInput(
        diffraction_patterns=measured,
        bad_pixels=numpy.zeros((probe_px, probe_px), dtype=bool),
        product=product,
        background=background,
        flat_field=flat_field,
    )
    options = align_task_options_with_product(LSQMLOptions(), product)
    response_map, valid_pixel_mask = _detector_response(parameters, flat_field_floor=1.0e-3)
    gauge = align_probe_gauge_with_ptychi(options, parameters, valid_pixel_mask=valid_pixel_mask)
    object_geometry = product.object_.get_geometry()
    position_x_px, position_y_px = _to_ptychi_positions_px(
        object_geometry.map_probe_positions_to_object_x_px(product.probe_positions),
        object_geometry.map_probe_positions_to_object_y_px(product.probe_positions),
        product.probes,
    )
    task = PtychographyTask(
        apply_probe_gauge_to_options(options, gauge),
        diffraction_data=measured,
        object_data=product.object_.get_array(),
        probe_data=product.probes.get_array() * gauge.to_ptychi,
        probe_position_x_px=position_x_px,
        probe_position_y_px=position_y_px,
        valid_pixel_mask=valid_pixel_mask,
        detector_response_map=response_map,
    )

    with task, torch.no_grad():
        reconstructor = task.reconstructor
        assert reconstructor is not None
        indices = torch.arange(len(product.probe_positions))
        predicted = cast(Any, reconstructor).forward_model.forward(indices).cpu().numpy()

    numpy.testing.assert_allclose(
        numpy.fft.fftshift(predicted, axes=(-2, -1)), measured, rtol=1e-4, atol=1e-6
    )


class TestDetectorResponse:
    _SHAPE = (4, 5)

    def _parameters(self, **maps: numpy.ndarray) -> ReconstructInput:
        bad_pixels = numpy.zeros(self._SHAPE, dtype=bool)
        bad_pixels[0, 0] = True
        return ReconstructInput(
            diffraction_patterns=numpy.zeros((1, *self._SHAPE)),
            bad_pixels=bad_pixels,
            product=_product(_smooth_probe(16), [(0.0, 0.0)]),
            **maps,
        )

    def test_no_maps_is_no_response(self) -> None:
        response, valid = _detector_response(self._parameters(), flat_field_floor=1.0e-3)

        assert response is None
        assert not valid[0, 0] and valid.sum() == valid.size - 1

    def test_a_flat_field_alone_is_the_gain_over_a_zero_offset(self) -> None:
        flat_field = numpy.full(self._SHAPE, 1.2)

        response, _ = _detector_response(
            self._parameters(flat_field=flat_field), flat_field_floor=1.0e-3
        )

        assert response is not None
        numpy.testing.assert_array_equal(response[0], flat_field)
        numpy.testing.assert_array_equal(response[1], numpy.zeros(self._SHAPE))

    def test_a_background_alone_is_the_offset_over_unit_gain(self) -> None:
        background = numpy.full(self._SHAPE, 0.3)

        response, _ = _detector_response(
            self._parameters(background=background), flat_field_floor=1.0e-3
        )

        assert response is not None
        numpy.testing.assert_array_equal(response[0], numpy.ones(self._SHAPE))
        numpy.testing.assert_array_equal(response[1], background)

    def test_an_unresponsive_pixel_takes_unit_gain_and_leaves_the_mask(self) -> None:
        flat_field = numpy.full(self._SHAPE, 1.0)
        flat_field[2, 3] = 0.0

        response, valid = _detector_response(
            self._parameters(flat_field=flat_field), flat_field_floor=1.0e-3
        )

        assert response is not None
        assert response[0][2, 3] == 1.0
        assert not valid[2, 3]
