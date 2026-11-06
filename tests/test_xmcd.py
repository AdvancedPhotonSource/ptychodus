"""Unit tests for ptychodus.api.xmcd.

The XMCD decomposition assumes the helicity model

    O_+ (RCP) = O_struct * M
    O_- (LCP) = O_struct / M

so the tests synthesize ground-truth ``O_struct`` and ``M``, build the RCP/LCP
pair from them, and verify that the object-level decomposition and the
product-level ``estimate_xmcd`` recover the inputs.
"""

from __future__ import annotations
from dataclasses import replace

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, RegistrationQuantity, align_objects
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import LossValue, Product, ProductMetadata
from ptychodus.api.diffraction import Polarization
from ptychodus.api.xmcd import _decompose_objects, estimate_xmcd

_EPSILON = 1.0e-12


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


def test_decompose_objects_pure_absorption_recovers_structural_and_magnetic_amplitudes() -> None:
    """Real-positive O_struct and M → recovered amplitudes match, recovered phases ~ 0."""
    rng = numpy.random.default_rng(0)
    o_struct = rng.uniform(0.4, 0.9, size=(16, 16)).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=(16, 16)).astype(numpy.complex128)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    structural_object, magnetic_object = _decompose_objects(rcp, lcp, epsilon=_EPSILON)

    structural = structural_object.get_array()[0]
    magnetic = magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(numpy.abs(structural), numpy.abs(o_struct), atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.abs(magnetic), numpy.abs(m_factor), atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(structural), 0.0, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(magnetic), 0.0, atol=1.0e-9)


def test_decompose_objects_pure_phase_recovers_within_principal_branch() -> None:
    """Unit-modulus O_struct=exp(i*beta) and M=exp(i*alpha) with phases in (-pi/2, pi/2)."""
    rng = numpy.random.default_rng(1)
    beta = rng.uniform(-1.0, 1.0, size=(16, 16))  # safely inside (-pi/2, pi/2)
    alpha = rng.uniform(-1.0, 1.0, size=(16, 16))
    o_struct = numpy.exp(1j * beta)
    m_factor = numpy.exp(1j * alpha)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    structural_object, magnetic_object = _decompose_objects(rcp, lcp, epsilon=_EPSILON)

    structural = structural_object.get_array()[0]
    magnetic = magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(numpy.abs(structural), 1.0, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.abs(magnetic), 1.0, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(structural), beta, atol=1.0e-9)
    numpy.testing.assert_allclose(numpy.angle(magnetic), alpha, atol=1.0e-9)


def test_decompose_objects_combined_complex_recovers_both_components() -> None:
    """Combined amplitude + phase: random complex O_struct and M, all inside the principal branch."""
    rng = numpy.random.default_rng(2)
    amp_struct = rng.uniform(0.4, 0.9, size=(32, 32))
    amp_mag = rng.uniform(0.8, 1.2, size=(32, 32))
    phase_struct = rng.uniform(-0.5, 0.5, size=(32, 32))
    phase_mag = rng.uniform(-0.5, 0.5, size=(32, 32))
    o_struct = amp_struct * numpy.exp(1j * phase_struct)
    m_factor = amp_mag * numpy.exp(1j * phase_mag)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    structural_object, magnetic_object = _decompose_objects(rcp, lcp, epsilon=_EPSILON)

    structural = structural_object.get_array()[0]
    magnetic = magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(structural, o_struct, atol=1.0e-9)
    numpy.testing.assert_allclose(magnetic, m_factor, atol=1.0e-9)


def test_decompose_objects_magnetic_phase_wraps_outside_principal_branch() -> None:
    """A magnetic phase of +3pi/4 wraps to -pi/4 because of the half-angle branch cut.

    The half-angle phase extraction ``0.5 * angle(z)`` gives values in
    ``(-pi/2, pi/2]``. So a magnetic phase ``alpha`` outside that range comes
    back as ``alpha - pi`` (or ``alpha + pi``), which is the principal-branch
    representative of the same magnetic transmission factor up to sign. The
    structural object flips sign with it, so the pair still reproduces both
    measurements. This test documents that behavior so it is not later treated
    as a regression.
    """
    o_struct = numpy.ones((8, 8), dtype=numpy.complex128) * 0.7
    alpha = 3 * numpy.pi / 4  # outside (-pi/2, pi/2]
    m_factor = numpy.full((8, 8), numpy.exp(1j * alpha), dtype=numpy.complex128)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    structural_object, magnetic_object = _decompose_objects(rcp, lcp, epsilon=_EPSILON)
    recovered_alpha = numpy.angle(magnetic_object.get_array()[0])

    expected_alpha = alpha - numpy.pi  # = -pi/4
    numpy.testing.assert_allclose(recovered_alpha, expected_alpha, atol=1.0e-9)
    structural = structural_object.get_array()[0]
    magnetic = magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(structural, -o_struct, atol=1.0e-9)
    numpy.testing.assert_allclose(structural * magnetic, rcp.get_array()[0], atol=1.0e-9)


def test_decompose_objects_structural_phase_crossing_half_pi_has_no_sign_flips() -> None:
    """A structural phase ramp across pi/2 is recovered whole, consistent with M.

    The model fixes ``(O_struct, M)`` only up to a joint sign. Halving the
    structural and magnetic angles on independent branches would flip
    ``O_struct`` alone wherever its phase crosses ``pi/2``, breaking
    ``O_struct * M = O_+``.
    """
    beta = numpy.tile(numpy.linspace(0.0, 2.5, 64), (8, 1))
    alpha = numpy.where(numpy.arange(64) < 32, 0.02, -0.02)[numpy.newaxis].repeat(8, axis=0)
    o_struct = 0.8 * numpy.exp(1j * beta)
    m_factor = 1.01 * numpy.exp(1j * alpha)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)

    structural_object, magnetic_object = _decompose_objects(rcp, lcp, epsilon=_EPSILON)

    structural = structural_object.get_array()[0]
    magnetic = magnetic_object.get_array()[0]
    numpy.testing.assert_allclose(structural, o_struct, atol=1.0e-9)
    numpy.testing.assert_allclose(magnetic, m_factor, atol=1.0e-9)
    numpy.testing.assert_allclose(structural * magnetic, rcp.get_array()[0], atol=1.0e-9)
    numpy.testing.assert_allclose(structural / magnetic, lcp.get_array()[0], atol=1.0e-9)


def test_decompose_objects_epsilon_keeps_magnetic_finite_at_zero_lcp_pixels() -> None:
    """A column of exact zeros in the LCP array must not produce NaN/Inf."""
    rng = numpy.random.default_rng(3)
    o_struct = rng.uniform(0.4, 0.9, size=(16, 16)).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=(16, 16)).astype(numpy.complex128)
    rcp, lcp = _build_xmcd_pair(o_struct, m_factor)
    # Force a dead column on the LCP side.
    lcp_array = lcp.get_array().copy()
    lcp_array[0, :, 5] = 0.0 + 0.0j
    lcp_zeroed = _make_object(lcp_array[0])

    structural_object, magnetic_object = _decompose_objects(rcp, lcp_zeroed, epsilon=_EPSILON)

    assert numpy.all(numpy.isfinite(structural_object.get_array()))
    assert numpy.all(numpy.isfinite(magnetic_object.get_array()))

    # Pixels away from the dead column should still recover correctly.
    structural = structural_object.get_array()[0]
    keep = numpy.ones_like(o_struct, dtype=bool)
    keep[:, 5] = False
    numpy.testing.assert_allclose(numpy.abs(structural[keep]), numpy.abs(o_struct[keep]), atol=1e-6)


def test_decompose_objects_multi_layer_input_flattens() -> None:
    rng = numpy.random.default_rng(4)
    layer_shape = (8, 8)
    o_struct = rng.uniform(0.4, 0.9, size=layer_shape).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=layer_shape).astype(numpy.complex128)

    rcp_layers = numpy.stack([o_struct * m_factor, o_struct * m_factor], axis=0)
    lcp_layers = numpy.stack([o_struct / m_factor, o_struct / m_factor], axis=0)
    rcp = _make_object(rcp_layers, layer_spacing_m=[1.0e-6])
    lcp = _make_object(lcp_layers, layer_spacing_m=[1.0e-6])

    structural_object, magnetic_object = _decompose_objects(rcp, lcp, epsilon=_EPSILON)

    assert structural_object.num_layers == 1
    assert magnetic_object.num_layers == 1


def test_decompose_objects_raises_on_pixel_geometry_mismatch() -> None:
    rng = numpy.random.default_rng(5)
    o_struct = rng.uniform(0.4, 0.9, size=(8, 8)).astype(numpy.complex128)
    m_factor = rng.uniform(0.8, 1.2, size=(8, 8)).astype(numpy.complex128)
    rcp_array = (o_struct * m_factor)[numpy.newaxis]
    lcp_array = (o_struct / m_factor)[numpy.newaxis]
    rcp = _make_object(rcp_array, pixel_geometry=PixelGeometry(width_m=1.0e-9, height_m=1.0e-9))
    lcp = _make_object(lcp_array, pixel_geometry=PixelGeometry(width_m=2.0e-9, height_m=1.0e-9))
    with pytest.raises(ValueError, match='pixel geometry mismatch'):
        _decompose_objects(rcp, lcp, epsilon=_EPSILON)


def test_decompose_objects_raises_on_array_shape_mismatch() -> None:
    rng = numpy.random.default_rng(6)
    rcp = _make_object(rng.uniform(0.5, 1.0, size=(1, 8, 8)).astype(numpy.complex128))
    lcp = _make_object(rng.uniform(0.5, 1.0, size=(1, 16, 16)).astype(numpy.complex128))
    with pytest.raises(ValueError, match='array shape mismatch'):
        _decompose_objects(rcp, lcp, epsilon=_EPSILON)


def test_decompose_objects_output_inherits_rcp_pixel_geometry_and_center() -> None:
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

    structural_object, magnetic_object = _decompose_objects(rcp, lcp, epsilon=_EPSILON)

    assert structural_object.get_pixel_geometry() == pixel_geom
    assert magnetic_object.get_pixel_geometry() == pixel_geom
    assert structural_object.get_center() == rcp_center
    assert magnetic_object.get_center() == rcp_center


def test_decompose_objects_end_to_end_with_align_objects_recovers_known_signal() -> None:
    """Integration: known XMCD pair + known shift on LCP → align_objects + decomposition recovers."""
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
    structural_object, magnetic_object = _decompose_objects(
        cropped_rcp, aligned_lcp, epsilon=_EPSILON
    )

    structural = structural_object.get_array()[0]
    magnetic = magnetic_object.get_array()[0]

    # Compare only in the support region where amp is significant — outside, the
    # signals are weak and noise dominates.
    support = amp > 0.2
    numpy.testing.assert_allclose(
        numpy.abs(structural)[support], numpy.abs(o_struct)[support], atol=2e-3
    )
    numpy.testing.assert_allclose(
        numpy.abs(magnetic)[support], numpy.abs(m_factor)[support], atol=2e-3
    )


def test_decompose_objects_end_to_end_with_mismatched_shapes_recovers_known_signal() -> None:
    """Integration: RCP and LCP with different shapes → align_objects trims and aligns.

    Simulates the failure mode this fix targets: rounding/padding leaves the
    two reconstructions with slightly different array shapes. The full
    align_objects → decomposition chain must trim to a common shape and still
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

    structural_object, magnetic_object = _decompose_objects(
        cropped_rcp, aligned_lcp, epsilon=_EPSILON
    )
    structural = structural_object.get_array()[0]
    magnetic = magnetic_object.get_array()[0]

    support = amp_rcp > 0.2
    numpy.testing.assert_allclose(
        numpy.abs(structural)[support], numpy.abs(o_struct_rcp)[support], atol=5e-3
    )
    numpy.testing.assert_allclose(
        numpy.abs(magnetic)[support], numpy.abs(m_factor_rcp)[support], atol=5e-3
    )


def _product(
    o_struct: numpy.ndarray,
    m_factor: numpy.ndarray,
    *,
    name: str,
    polarization: Polarization,
) -> Product:
    pixel_geometry = PixelGeometry(width_m=1.0e-8, height_m=1.0e-8)
    array = (
        o_struct * m_factor if polarization is Polarization.RIGHT_CIRCULAR else o_struct / m_factor
    )
    return Product(
        metadata=ProductMetadata(
            name=name,
            comments='measured',
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
            center=ObjectCenter(x_m=1.0e-8, y_m=-2.0e-8),
        ),
        losses=[LossValue(epoch=0, value=1.0)],
    )


def test_estimate_xmcd_recovers_signal_on_the_rcp_product() -> None:
    """Products in, products out: RCP's frame, provenance named, polarization cleared."""
    rng = numpy.random.default_rng(10)
    y = numpy.arange(48)[:, None]
    x = numpy.arange(48)[None, :]
    amp = numpy.exp(-(((y - 24) ** 2 + (x - 24) ** 2) / 60.0))
    o_struct = (amp * (0.7 + 0.05 * rng.standard_normal(amp.shape))).astype(numpy.complex128)
    m_factor = (1.0 + 0.05 * amp).astype(numpy.complex128)
    rcp = _product(o_struct, m_factor, name='scan_rcp', polarization=Polarization.RIGHT_CIRCULAR)
    lcp = _product(o_struct, m_factor, name='scan_lcp', polarization=Polarization.LEFT_CIRCULAR)
    lcp_array = numpy.roll(lcp.object_.get_array(), shift=(3, -2), axis=(-2, -1))
    # A global phase on the LCP reconstruction is an ambiguity the standardization removes.
    lcp = replace(
        lcp,
        object_=Object(
            array=lcp_array * numpy.exp(0.4j),
            pixel_geometry=lcp.object_.get_pixel_geometry(),
            center=lcp.object_.get_center(),
        ),
    )

    result = estimate_xmcd(lcp, rcp)

    # The standardization absorbs the weighted mean |M|**2 into the LCP scale, so each
    # output matches its truth up to one global factor, while the RCP reference is
    # reproduced exactly by their product.
    support = amp > 0.2
    structural = result.structural_product.object_.get_array()[0]
    magnetic = result.magnetic_product.object_.get_array()[0]
    numpy.testing.assert_allclose(structural * magnetic, rcp.object_.get_array()[0], atol=1e-9)

    for estimate, truth in ((structural, o_struct), (magnetic, m_factor)):
        factor = numpy.vdot(estimate[support], truth[support]) / numpy.vdot(
            estimate[support], estimate[support]
        )
        assert abs(factor - 1.0) < 0.05
        numpy.testing.assert_allclose(factor * estimate[support], truth[support], atol=5e-3)

    for product, component in (
        (result.structural_product, 'structural'),
        (result.magnetic_product, 'magnetic'),
    ):
        assert product.metadata.name == f'scan_rcp-{component}'
        assert product.metadata.comments == (
            f"XMCD {component} component of LCP 'scan_lcp' and RCP 'scan_rcp'"
        )
        assert product.metadata.polarization is None
        assert product.metadata.photon_energy_eV == rcp.metadata.photon_energy_eV
        assert product.losses == []
        assert product.probe_positions is rcp.probe_positions
        assert product.probes is rcp.probes
        assert product.object_.get_pixel_geometry() == rcp.object_.get_pixel_geometry()
        assert product.object_.get_center() == rcp.object_.get_center()


@pytest.mark.parametrize(
    'registration_quantity',
    # VARIATION trades a small sub-pixel bias for ramp insensitivity, so a shifted LCP
    # object does not reproduce the unshifted decomposition to this tolerance.
    [
        quantity
        for quantity in RegistrationQuantity
        if quantity is not RegistrationQuantity.VARIATION
    ],
)
def test_estimate_xmcd_is_invariant_to_lcp_shift_and_phase_offset(
    registration_quantity: RegistrationQuantity,
) -> None:
    """Shifting the LCP object and offsetting its phase leaves the decomposition unchanged."""
    rng = numpy.random.default_rng(10)
    y = numpy.arange(48)[:, None]
    x = numpy.arange(48)[None, :]
    amp = numpy.exp(-(((y - 24) ** 2 + (x - 24) ** 2) / 60.0))
    o_struct = (amp * (0.7 + 0.05 * rng.standard_normal(amp.shape))).astype(numpy.complex128)
    m_factor = (1.0 + 0.05 * amp) * numpy.exp(0.05j * amp * numpy.tanh((x - 24) / 4.0))
    rcp = _product(o_struct, m_factor, name='rcp', polarization=Polarization.RIGHT_CIRCULAR)
    lcp = _product(o_struct, m_factor, name='lcp', polarization=Polarization.LEFT_CIRCULAR)
    moved_lcp = replace(
        lcp,
        object_=Object(
            array=numpy.roll(lcp.object_.get_array(), shift=(3, -2), axis=(-2, -1))
            * numpy.exp(0.3j),
            pixel_geometry=lcp.object_.get_pixel_geometry(),
            center=lcp.object_.get_center(),
        ),
    )

    expected = estimate_xmcd(lcp, rcp, registration_quantity=registration_quantity)
    actual = estimate_xmcd(moved_lcp, rcp, registration_quantity=registration_quantity)

    support = amp > 0.2
    for actual_product, expected_product in (
        (actual.structural_product, expected.structural_product),
        (actual.magnetic_product, expected.magnetic_product),
    ):
        numpy.testing.assert_allclose(
            actual_product.object_.get_array()[0][support],
            expected_product.object_.get_array()[0][support],
            atol=1e-3,
        )

    # Against the ground truth, up to the scale and phase ramp standardization absorbs.
    numpy.testing.assert_allclose(
        numpy.abs(actual.structural_product.object_.get_array()[0])[support],
        numpy.abs(o_struct)[support],
        rtol=5e-2,
    )


def test_estimate_xmcd_warns_on_multi_layer_objects(caplog: pytest.LogCaptureFixture) -> None:
    single_layer = numpy.ones((8, 8), dtype=numpy.complex128)
    product = replace(
        _product(single_layer, single_layer, name='rcp', polarization=Polarization.RIGHT_CIRCULAR),
        object_=_make_object(numpy.ones((2, 8, 8)), layer_spacing_m=[1.0e-6]),
    )

    estimate_xmcd(product, product)

    assert 'flattens multi-layer objects' in caplog.text
