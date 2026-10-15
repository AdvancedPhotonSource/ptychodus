"""Characterization tests for the VSPI linear operator in
ptychodus.model.fluorescence.vspi.

The operator is the forward and adjoint model of probe-weighted fluorescence
imaging: each scan position integrates the object against that position's
normalized probe intensity. Its adjointness is what ``scipy.sparse.linalg.lsmr``
relies on, and nothing else in the suite exercises it.
"""

from __future__ import annotations

import numpy
import numpy.testing
import pytest
from scipy.sparse.linalg import lsmr

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.model.fluorescence.vspi import VSPILinearOperator

_PIXEL_SIZE_M = 1.0e-7
_CANVAS_PX = 24
_PROBE_PX = 6


def _make_product(positions_px: list[tuple[float, float]]) -> Product:
    """Build a product whose scan positions sit at the given object-pixel centers.

    Positions are given in pixels and converted to meters through the inverse of
    the centered-pixel convention, so a test can place a patch at an exact integer
    or at a deliberate sub-pixel offset without doing the arithmetic twice.
    """
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    radius_px = (_CANVAS_PX - 1) / 2

    positions = [
        ProbePosition(
            index=index,
            x_m=(x_px - radius_px) * _PIXEL_SIZE_M,
            y_m=(y_px - radius_px) * _PIXEL_SIZE_M,
        )
        for index, (x_px, y_px) in enumerate(positions_px)
    ]

    yy, xx = numpy.mgrid[:_PROBE_PX, :_PROBE_PX] - (_PROBE_PX - 1) / 2
    probe_array = numpy.exp(-(yy**2 + xx**2) / 4.0).astype(complex)

    return Product(
        metadata=ProductMetadata(
            name='vspi',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10000.0,
            probe_photon_count=1.0e9,
            exposure_time_s=0.1,
            mass_attenuation_m2_per_kg=5.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeSequence(
            array=probe_array[numpy.newaxis, :, :],
            opr_weights=None,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.ones((_CANVAS_PX, _CANVAS_PX), dtype=complex),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


# A patch carries zero sub-pixel offset when ``center - (extent - 1) / 2`` is whole,
# so for an even extent the zero-offset centers are the half-integers.
_INTEGER_POSITIONS = [(8.5, 8.5), (11.5, 11.5), (14.5, 14.5)]
_SUBPIXEL_POSITIONS = [(8.3, 9.7), (11.5, 11.5), (14.2, 13.8)]


def test_operator_shape_is_positions_by_object_pixels() -> None:
    operator = VSPILinearOperator(_make_product(_INTEGER_POSITIONS))
    assert operator.shape == (len(_INTEGER_POSITIONS), _CANVAS_PX * _CANVAS_PX)


def test_matvec_matches_hand_computed_psf_weighted_sum() -> None:
    """At a zero-offset position the patch is a plain slice, so the forward model
    reduces to a sum of the object against the normalized probe intensity."""
    product = _make_product(_INTEGER_POSITIONS)
    operator = VSPILinearOperator(product)

    rng = numpy.random.default_rng(1234)
    canvas = rng.standard_normal((_CANVAS_PX, _CANVAS_PX))

    probe_intensity = product.probes[0].get_intensity()
    psf = probe_intensity / probe_intensity.sum()
    half = (_PROBE_PX - 1) / 2

    expected = []
    for x_px, y_px in _INTEGER_POSITIONS:
        y_lower = int(y_px - half)
        x_lower = int(x_px - half)
        patch = canvas[y_lower : y_lower + _PROBE_PX, x_lower : x_lower + _PROBE_PX]
        expected.append(numpy.sum(psf * patch))

    numpy.testing.assert_allclose(operator.matvec(canvas.ravel()), expected, atol=1e-12)


@pytest.mark.parametrize('positions', [_INTEGER_POSITIONS, _SUBPIXEL_POSITIONS])
def test_operator_is_its_own_adjoint_pair(positions: list[tuple[float, float]]) -> None:
    """``<A x, y> == <x, A^T y>``, the identity the LSMR solve depends on.

    A patch gathered or scattered outside the canvas silently loses or misplaces
    weight, which breaks this identity without raising, so it is the property
    worth pinning rather than either half on its own.
    """
    operator = VSPILinearOperator(_make_product(positions))
    rng = numpy.random.default_rng(99)
    x = rng.standard_normal(_CANVAS_PX * _CANVAS_PX)
    y = rng.standard_normal(len(positions))

    assert numpy.dot(operator.matvec(x), y) == pytest.approx(
        numpy.dot(x, operator.rmatvec(y)), rel=1e-12
    )


def test_lsmr_recovers_a_scaled_object() -> None:
    """The operator is usable by the solver it exists for, not merely self-consistent."""
    positions = [(x, y) for y in (9.0, 11.0, 13.0) for x in (9.0, 11.0, 13.0)]
    operator = VSPILinearOperator(_make_product(positions))

    truth = numpy.zeros((_CANVAS_PX, _CANVAS_PX))
    truth[8:16, 8:16] = 3.0
    measured = operator.matvec(truth.ravel())

    solution = lsmr(operator, measured, damp=1.0e-3, maxiter=400)[0]

    # The operator only sees the illuminated region, so agreement is asserted on
    # the forward projection rather than pixelwise on an underdetermined canvas.
    numpy.testing.assert_allclose(operator.matvec(solution), measured, rtol=1e-3)


# Recorded from the implementation that predates the bounds-checked interpolation
# support. The sub-pixel gather is the part of the operator most sensitive to a
# change in how the interpolation window is resolved, so these values are what
# establish that such a change left in-bounds results untouched.
_MIXED_OFFSET_POSITIONS = [(x, y) for y in (9.4, 11.5, 13.7) for x in (9.1, 11.5, 13.2)]
_RECORDED_MATVEC = [
    0.05467424994413332,
    -0.2195539520463686,
    -0.027188382148486318,
    0.510526405231019,
    -0.07200159268662666,
    -0.008814545322601593,
    -0.031139224181464396,
    -0.31459668683233544,
    -0.0881442044547162,
]


def test_matvec_reproduces_recorded_values() -> None:
    operator = VSPILinearOperator(_make_product(_MIXED_OFFSET_POSITIONS))
    rng = numpy.random.default_rng(2026)
    x = rng.standard_normal(_CANVAS_PX * _CANVAS_PX)

    numpy.testing.assert_allclose(operator.matvec(x), _RECORDED_MATVEC, rtol=0.0, atol=0.0)


def _make_opr_product(opr_weights: numpy.ndarray) -> Product:
    """A product whose probe is mixed per position by ``opr_weights``."""
    pixel_geometry = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
    radius_px = (_CANVAS_PX - 1) / 2
    positions = [
        ProbePosition(
            index=index,
            x_m=(9.5 + 2 * index - radius_px) * _PIXEL_SIZE_M,
            y_m=(11.5 - radius_px) * _PIXEL_SIZE_M,
        )
        for index in range(opr_weights.shape[0])
    ]
    yy, xx = numpy.mgrid[:_PROBE_PX, :_PROBE_PX] - (_PROBE_PX - 1) / 2
    base = numpy.exp(-(yy**2 + xx**2) / 4.0).astype(complex)
    array = numpy.stack([base * (mode + 1) for mode in range(opr_weights.shape[1])])

    return Product(
        metadata=ProductMetadata(
            name='vspi-opr',
            comments='',
            detector_distance_m=1.0,
            photon_energy_eV=10000.0,
            probe_photon_count=1.0e9,
            exposure_time_s=0.1,
            mass_attenuation_m2_per_kg=5.0,
            tomography_angle_deg=0.0,
        ),
        probe_positions=ProbePositionSequence(positions),
        probes=ProbeSequence(
            array=array[:, numpy.newaxis, :, :],
            opr_weights=opr_weights,
            pixel_geometry=pixel_geometry,
        ),
        object_=Object(
            array=numpy.ones((_CANVAS_PX, _CANVAS_PX), dtype=complex),
            pixel_geometry=pixel_geometry,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )


def test_operator_is_its_own_adjoint_pair_under_opr() -> None:
    """Each position carries a different probe, so forward and adjoint must agree on
    which one -- a mismatch between them would break the solve while leaving the
    fixed-probe adjoint test passing.
    """
    weights = numpy.array([[1.0, 0.2], [0.4, 1.3], [0.9, 0.7]])
    operator = VSPILinearOperator(_make_opr_product(weights))

    rng = numpy.random.default_rng(3)
    x = rng.standard_normal(_CANVAS_PX * _CANVAS_PX)
    y = rng.standard_normal(weights.shape[0])

    assert numpy.dot(operator.matvec(x), y) == pytest.approx(
        numpy.dot(x, operator.rmatvec(y)), rel=1e-12
    )


def test_psf_varies_per_position_under_opr() -> None:
    """Guards the premise of the adjoint test above: if the probe did not actually
    vary, that test would hold trivially."""
    weights = numpy.array([[1.0, 0.2], [0.4, 1.3], [0.9, 0.7]])
    product = _make_opr_product(weights)

    powers = [float(product.probes[index].get_intensity().sum()) for index in range(3)]

    assert len(set(numpy.round(powers, 9))) == 3
