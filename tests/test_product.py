"""Unit tests for Product helpers in ptychodus.api.product."""

from __future__ import annotations

from dataclasses import replace

import numpy
import numpy.testing

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import compute_magnification


def _make_metadata() -> ProductMetadata:
    return ProductMetadata(
        name='t',
        comments='',
        detector_distance_m=1.0,
        photon_energy_eV=10000.0,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=1.0,
        tomography_angle_deg=0.0,
    )


def _make_product(
    *,
    positions: list[ProbePosition],
    probe_array: numpy.ndarray,
    opr_weights: numpy.ndarray | None = None,
) -> Product:
    pixel_geometry = PixelGeometry(width_m=1.0e-7, height_m=1.0e-7)
    obj = Object(
        array=numpy.zeros((8, 8), dtype=complex),
        pixel_geometry=pixel_geometry,
        center=ObjectCenter(x_m=0.0, y_m=0.0),
    )
    probes = ProbeSequence(
        array=probe_array.astype(complex),
        opr_weights=opr_weights,
        pixel_geometry=pixel_geometry,
    )
    return Product(
        metadata=_make_metadata(),
        probe_positions=ProbePositionSequence(positions),
        probes=probes,
        object_=obj,
        losses=[],
    )


class TestIterPositionProbes:
    def test_yields_all_positions_without_opr(self) -> None:
        """With no OPR weights, every position must yield the shared probe.

        This is the bug fix: previously ``len(probes) == 1`` caused
        ``zip(positions, probes)`` to truncate after the first position.
        """
        positions = [ProbePosition(index=i, x_m=float(i), y_m=0.0) for i in range(3)]
        probe = numpy.ones((4, 4), dtype=complex)
        product = _make_product(positions=positions, probe_array=probe)

        pairs = list(product.iter_position_probes())
        assert len(pairs) == 3
        # Without OPR, every position shares the same probe content.
        for _, p in pairs:
            numpy.testing.assert_array_equal(p.get_array()[0], probe)

    def test_yields_all_positions_with_opr(self) -> None:
        """OPR-equipped products are unchanged: same length, OPR-weighted per index."""
        positions = [ProbePosition(index=i, x_m=float(i), y_m=0.0) for i in range(3)]
        # 2 coherent modes, 1 incoherent mode, 4x4 spatial. OPR mixes them per position.
        probe = numpy.stack(
            [numpy.full((1, 4, 4), 1.0), numpy.full((1, 4, 4), 2.0)],
            axis=0,
        ).astype(complex)
        opr_weights = numpy.array(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            dtype=float,
        )
        product = _make_product(positions=positions, probe_array=probe, opr_weights=opr_weights)

        pairs = list(product.iter_position_probes())
        assert len(pairs) == 3
        # The first incoherent mode of each probe is the OPR-weighted sum across
        # coherent modes: 1.0, 2.0, and 1.5 respectively.
        numpy.testing.assert_allclose(pairs[0][1].get_array()[0], 1.0)
        numpy.testing.assert_allclose(pairs[1][1].get_array()[0], 2.0)
        numpy.testing.assert_allclose(pairs[2][1].get_array()[0], 1.5)

    def test_yields_position_objects_with_correct_fields(self) -> None:
        positions = [
            ProbePosition(index=0, x_m=1.5e-7, y_m=-2.5e-7),
            ProbePosition(index=1, x_m=3.5e-7, y_m=+0.5e-7),
        ]
        # OPR weights so we genuinely walk both positions (also covered by the no-OPR test).
        opr_weights = numpy.ones((2, 1), dtype=float)
        product = _make_product(
            positions=positions,
            probe_array=numpy.ones((4, 4), dtype=complex),
            opr_weights=opr_weights,
        )

        pairs = list(product.iter_position_probes())
        assert pairs[0][0].index == 0
        assert pairs[0][0].x_m == 1.5e-7
        assert pairs[0][0].y_m == -2.5e-7
        assert pairs[1][0].index == 1
        assert pairs[1][0].x_m == 3.5e-7
        assert pairs[1][0].y_m == +0.5e-7


class TestMetadataMagnification:
    """ProductMetadata exposes the geometry it records; the formula itself is
    covered against :func:`compute_magnification` in tests/test_propagator.py.
    """

    def test_delegates_to_the_shared_formula(self) -> None:
        metadata = replace(_make_metadata(), detector_distance_m=1.0, focus_object_distance_m=5e-3)

        assert metadata.magnification == compute_magnification(1.0, 5e-3)

    def test_defaults_to_no_focusing_optic(self) -> None:
        assert _make_metadata().magnification == 1.0


class TestMetadataFarField:
    """The propagation regime is declared, not inferred from the illumination.

    Regression guard for the conflation this field removed: a nonzero focus-object
    distance used to be the only near-field signal, which made parallel-beam near
    field and far field with a focusing optic both inexpressible.
    """

    def test_defaults_to_far_field(self) -> None:
        assert _make_metadata().far_field

    def test_is_independent_of_the_magnification(self) -> None:
        with_optic = replace(
            _make_metadata(), detector_distance_m=1.0, focus_object_distance_m=5e-3
        )

        assert with_optic.magnification != 1.0
        assert with_optic.far_field

    def test_near_field_without_a_focusing_optic_is_expressible(self) -> None:
        parallel_beam = replace(_make_metadata(), focus_object_distance_m=0.0, far_field=False)

        assert parallel_beam.magnification == 1.0
        assert not parallel_beam.far_field
