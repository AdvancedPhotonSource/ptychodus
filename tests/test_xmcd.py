"""Unit tests for ptychodus.api.xmcd.

The XMCD decomposition assumes the helicity model

    O_+ (RCP) = O_struct * M
    O_- (LCP) = O_struct / M

so the tests synthesize ground-truth ``O_struct`` and ``M``, build the RCP/LCP
pair from them, and verify that estimate_xmcd recovers the inputs.
"""

from __future__ import annotations

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, align_objects
from ptychodus.api.xmcd import XMCDResult, estimate_xmcd


def _make_object(
    array: numpy.ndarray,
    *,
    pixel_geometry: PixelGeometry | None = None,
    center: ObjectCenter | None = None,
    layer_spacing_m: list[float] | None = None,
) -> Object:
    if pixel_geometry is None:
        pixel_geometry = PixelGeometry(width_m=1.0e-9, height_m=1.0e-9)
    if center is None:
        center = ObjectCenter(x_m=0.0, y_m=0.0)
    if layer_spacing_m is None:
        layer_spacing_m = []
    return Object(
        array=array.astype(numpy.complex128),
        pixel_geometry=pixel_geometry,
        center=center,
        layer_spacing_m=layer_spacing_m,
    )


def _build_xmcd_pair(
    o_struct: numpy.ndarray,
    m_factor: numpy.ndarray,
    *,
    pixel_geometry: PixelGeometry | None = None,
    rcp_center: ObjectCenter | None = None,
    lcp_center: ObjectCenter | None = None,
) -> tuple[Object, Object]:
    """Build RCP/LCP ground-truth pair following the XMCD helicity model."""
    assert o_struct.shape == m_factor.shape
    rcp_array = (o_struct * m_factor)[numpy.newaxis]
    lcp_array = (o_struct / m_factor)[numpy.newaxis]
    rcp = _make_object(rcp_array, pixel_geometry=pixel_geometry, center=rcp_center)
    lcp = _make_object(lcp_array, pixel_geometry=pixel_geometry, center=lcp_center)
    return rcp, lcp


def test_estimate_xmcd_pure_absorption_recovers_structural_and_magnetic_amplitudes() -> None:
    """Real-positive O_struct and M → recovered amplitudes match, recovered phases ~ 0."""
    rng = numpy.random.default_rng(0)
    o_struct = rng.uniform(0.4, 0.9, size=(16, 16)).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=(16, 16)).astype(numpy.complex128)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    result = estimate_xmcd(rcp, lcp)

    structural = result.structural_object.get_array()[0]
    magnetic = result.magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(numpy.abs(structural), numpy.abs(o_struct), atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.abs(magnetic), numpy.abs(m_factor), atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(structural), 0.0, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(magnetic), 0.0, atol=1.0e-9)


def test_estimate_xmcd_pure_phase_recovers_within_principal_branch() -> None:
    """Unit-modulus O_struct=exp(i*beta) and M=exp(i*alpha) with phases in (-pi/2, pi/2)."""
    rng = numpy.random.default_rng(1)
    beta = rng.uniform(-1.0, 1.0, size=(16, 16))  # safely inside (-pi/2, pi/2)
    alpha = rng.uniform(-1.0, 1.0, size=(16, 16))
    o_struct = numpy.exp(1j * beta)
    m_factor = numpy.exp(1j * alpha)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    result = estimate_xmcd(rcp, lcp)

    structural = result.structural_object.get_array()[0]
    magnetic = result.magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(numpy.abs(structural), 1.0, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.abs(magnetic), 1.0, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(structural), beta, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(magnetic), alpha, atol=1.0e-9)


def test_estimate_xmcd_combined_complex_recovers_both_components() -> None:
    """Combined amplitude + phase: random complex O_struct and M, all inside the principal branch."""
    rng = numpy.random.default_rng(2)
    amp_struct = rng.uniform(0.4, 0.9, size=(32, 32))
    amp_mag = rng.uniform(0.8, 1.2, size=(32, 32))
    phase_struct = rng.uniform(-0.5, 0.5, size=(32, 32))
    phase_mag = rng.uniform(-0.5, 0.5, size=(32, 32))
    o_struct = amp_struct * numpy.exp(1j * phase_struct)
    m_factor = amp_mag * numpy.exp(1j * phase_mag)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    result = estimate_xmcd(rcp, lcp)

    structural = result.structural_object.get_array()[0]
    magnetic = result.magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(structural, o_struct, atol=1.0e-9)
    numpy.testing.assert_allclose(magnetic, m_factor, atol=1.0e-9)


def test_estimate_xmcd_magnetic_phase_wraps_outside_principal_branch() -> None:
    """A magnetic phase of +3pi/4 wraps to -pi/4 because of the half-angle branch cut.

    The half-angle phase extraction ``0.5 * angle(z)`` gives values in
    ``(-pi/2, pi/2]``. So a magnetic phase ``alpha`` outside that range comes
    back as ``alpha - pi`` (or ``alpha + pi``), which is the principal-branch
    representative of the same magnetic transmission factor up to sign. This
    test documents that behavior so it is not later treated as a regression.
    """
    o_struct = numpy.ones((8, 8), dtype=numpy.complex128) * 0.7
    alpha = 3 * numpy.pi / 4  # outside (-pi/2, pi/2]
    m_factor = numpy.full((8, 8), numpy.exp(1j * alpha), dtype=numpy.complex128)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    result = estimate_xmcd(rcp, lcp)
    recovered_alpha = numpy.angle(result.magnetic_object.get_array()[0])

    expected_alpha = alpha - numpy.pi  # = -pi/4
    numpy.testing.assert_allclose(recovered_alpha, expected_alpha, atol=1.0e-9)


def test_estimate_xmcd_epsilon_keeps_magnetic_finite_at_zero_lcp_pixels() -> None:
    """A column of exact zeros in the LCP array must not produce NaN/Inf."""
    rng = numpy.random.default_rng(3)
    o_struct = rng.uniform(0.4, 0.9, size=(16, 16)).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=(16, 16)).astype(numpy.complex128)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)
    # Force a dead column on the LCP side.
    lcp_array = lcp.get_array().copy()
    lcp_array[0, :, 5] = 0.0 + 0.0j
    lcp_zeroed = _make_object(lcp_array[0])

    result = estimate_xmcd(rcp, lcp_zeroed)

    assert numpy.all(numpy.isfinite(result.structural_object.get_array()))
    assert numpy.all(numpy.isfinite(result.magnetic_object.get_array()))

    # Pixels away from the dead column should still recover correctly.
    structural = result.structural_object.get_array()[0]
    keep = numpy.ones_like(o_struct, dtype=bool)
    keep[:, 5] = False
    numpy.testing.assert_allclose(numpy.abs(structural[keep]), numpy.abs(o_struct[keep]), atol=1e-6)


def test_estimate_xmcd_multi_layer_input_flattens() -> None:
    rng = numpy.random.default_rng(4)
    layer_shape = (8, 8)
    o_struct = rng.uniform(0.4, 0.9, size=layer_shape).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=layer_shape).astype(numpy.complex128)

    rcp_layers = numpy.stack([o_struct * m_factor, o_struct * m_factor], axis=0)
    lcp_layers = numpy.stack([o_struct / m_factor, o_struct / m_factor], axis=0)
    rcp = _make_object(rcp_layers, layer_spacing_m=[1.0e-6])
    lcp = _make_object(lcp_layers, layer_spacing_m=[1.0e-6])

    result = estimate_xmcd(rcp, lcp)

    assert result.structural_object.num_layers == 1
    assert result.magnetic_object.num_layers == 1


def test_estimate_xmcd_raises_on_pixel_geometry_mismatch() -> None:
    rng = numpy.random.default_rng(5)
    o_struct = rng.uniform(0.4, 0.9, size=(8, 8)).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=(8, 8)).astype(numpy.complex128)
    rcp_array = (o_struct * m_factor)[numpy.newaxis]
    lcp_array = (o_struct / m_factor)[numpy.newaxis]
    rcp = _make_object(rcp_array, pixel_geometry=PixelGeometry(width_m=1.0e-9, height_m=1.0e-9))
    lcp = _make_object(lcp_array, pixel_geometry=PixelGeometry(width_m=2.0e-9, height_m=1.0e-9))
    with pytest.raises(ValueError, match='pixel geometry mismatch'):
        estimate_xmcd(rcp, lcp)


def test_estimate_xmcd_raises_on_array_shape_mismatch() -> None:
    rng = numpy.random.default_rng(6)
    rcp = _make_object(rng.uniform(0.5, 1.0, size=(1, 8, 8)).astype(numpy.complex128))
    lcp = _make_object(rng.uniform(0.5, 1.0, size=(1, 16, 16)).astype(numpy.complex128))
    with pytest.raises(ValueError, match='array shape mismatch'):
        estimate_xmcd(rcp, lcp)


def test_estimate_xmcd_output_inherits_rcp_pixel_geometry_and_center() -> None:
    """Result objects stamp RCP's pixel geometry and center, not LCP's.

    This is intentional: after align_objects, the aligned LCP carries an
    offset center encoding the alignment shift, but the structural/magnetic
    outputs sit on RCP's coordinate frame so they line up with the RCP
    reconstruction for visualization and saving.
    """
    rng = numpy.random.default_rng(7)
    o_struct = rng.uniform(0.4, 0.9, size=(8, 8)).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=(8, 8)).astype(numpy.complex128)
    pixel_geom = PixelGeometry(width_m=2.5e-9, height_m=3.5e-9)
    rcp_center = ObjectCenter(x_m=11.0e-9, y_m=-7.0e-9)
    lcp_center = ObjectCenter(x_m=99.0e-9, y_m=99.0e-9)
    rcp, lcp = _build_xmcd_pair(
        o_struct,
        m_factor,
        pixel_geometry=pixel_geom,
        rcp_center=rcp_center,
        lcp_center=lcp_center,
    )

    result = estimate_xmcd(rcp, lcp)

    assert result.structural_object.get_pixel_geometry() == pixel_geom
    assert result.magnetic_object.get_pixel_geometry() == pixel_geom
    assert result.structural_object.get_center() == rcp_center
    assert result.magnetic_object.get_center() == rcp_center
    assert isinstance(result, XMCDResult)


def test_estimate_xmcd_end_to_end_with_align_objects_recovers_known_signal() -> None:
    """Integration: known XMCD pair + known shift on LCP → align_objects + estimate_xmcd recovers."""
    rng = numpy.random.default_rng(8)
    # Use a localized smooth feature so phase_cross_correlation has structure to lock onto.
    y = numpy.arange(48)[:, None]
    x = numpy.arange(48)[None, :]
    amp = numpy.exp(-(((y - 24) ** 2 + (x - 24) ** 2) / 60.0))
    o_struct = (amp * (0.7 + 0.05 * rng.standard_normal(amp.shape))).astype(numpy.complex128)
    m_factor = (1.0 + 0.05 * amp).astype(numpy.complex128)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    # Apply an integer-pixel shift to the LCP array so align_objects has work to do.
    shift_yx = (3, -2)
    lcp_shifted_array = numpy.roll(lcp.get_array(), shift=shift_yx, axis=(-2, -1))
    lcp_shifted = _make_object(lcp_shifted_array[0])

    cropped_rcp, aligned_lcp = align_objects(rcp, lcp_shifted)
    result = estimate_xmcd(cropped_rcp, aligned_lcp)

    structural = result.structural_object.get_array()[0]
    magnetic = result.magnetic_object.get_array()[0]

    # Compare only in the support region where amp is significant — outside, the
    # signals are weak and noise dominates.
    support = amp > 0.2
    numpy.testing.assert_allclose(
        numpy.abs(structural)[support], numpy.abs(o_struct)[support], atol=2e-3
    )
    numpy.testing.assert_allclose(
        numpy.abs(magnetic)[support], numpy.abs(m_factor)[support], atol=2e-3
    )


def test_estimate_xmcd_end_to_end_with_mismatched_shapes_recovers_known_signal() -> None:
    """Integration: RCP and LCP with different shapes → align_objects trims and aligns.

    Simulates the pty-chi failure mode this fix targets: rounding/padding leaves the
    two reconstructions with slightly different array shapes. The full
    align_objects → estimate_xmcd chain must trim to a common shape and still
    recover the known structural + magnetic signal on that shared region.
    """
    rng = numpy.random.default_rng(9)

    # Reference (RCP) at 48x48; moving (LCP) built at 50x52 with a small integer
    # shift so both pieces of the fix — trimming and sub-pixel alignment — are exercised.
    def _feature(shape: tuple[int, int], center_yx: tuple[float, float]) -> numpy.ndarray:
        h, w = shape
        y = numpy.arange(h)[:, None]
        x = numpy.arange(w)[None, :]
        cy, cx = center_yx
        return numpy.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / 60.0))

    amp_rcp = _feature((48, 48), center_yx=(24.0, 24.0))
    o_struct_rcp = (amp_rcp * (0.7 + 0.05 * rng.standard_normal(amp_rcp.shape))).astype(
        numpy.complex128
    )
    m_factor_rcp = (1.0 + 0.05 * amp_rcp).astype(numpy.complex128)
    # For the LCP side, take the corresponding physical content by center-embedding
    # the same underlying feature into the larger array.
    o_struct_lcp = numpy.zeros((50, 52), dtype=numpy.complex128)
    m_factor_lcp = numpy.ones((50, 52), dtype=numpy.complex128)
    o_struct_lcp[1:49, 2:50] = o_struct_rcp
    m_factor_lcp[1:49, 2:50] = m_factor_rcp

    rcp = _make_object((o_struct_rcp * m_factor_rcp)[numpy.newaxis])
    lcp = _make_object((o_struct_lcp / m_factor_lcp)[numpy.newaxis])

    cropped_rcp, aligned_lcp = align_objects(rcp, lcp)
    assert cropped_rcp.get_array().shape == (1, 48, 48)
    assert aligned_lcp.get_array().shape == (1, 48, 48)

    result = estimate_xmcd(cropped_rcp, aligned_lcp)
    structural = result.structural_object.get_array()[0]
    magnetic = result.magnetic_object.get_array()[0]

    support = amp_rcp > 0.2
    numpy.testing.assert_allclose(
        numpy.abs(structural)[support], numpy.abs(o_struct_rcp)[support], atol=5e-3
    )
    numpy.testing.assert_allclose(
        numpy.abs(magnetic)[support], numpy.abs(m_factor_rcp)[support], atol=5e-3
    )
