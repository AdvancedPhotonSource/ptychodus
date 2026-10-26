"""The frame cdtools' FancyPtycho places probes in, and the checks before a run.

Needs no backend installed: the mapping is numpy. ``test_cdtools_reconstruct.py``
confirms against cdtools itself that the frame puts the probe where ptychodus does.
"""

from __future__ import annotations

from dataclasses import replace

import numpy
import pytest

from cdtools_fixtures import (
    METADATA,
    OBJECT_HEIGHT_PX,
    OBJECT_WIDTH_PX,
    PIXEL_HEIGHT_M,
    PIXEL_WIDTH_M,
    PROBE_HEIGHT_PX,
    PROBE_WIDTH_PX,
    make_object,
    make_positions,
    make_product,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble, ProbeGeometry
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.plugins.cdtools.geometry import (
    apply_translation_offsets,
    build_frame,
    verify_product,
)


def _corners(frame_translations: numpy.ndarray, min_translation: numpy.ndarray) -> numpy.ndarray:
    """What cdtools' translations_to_pixel minus min_translation gives for this basis."""
    rows = frame_translations[:, 1] / PIXEL_HEIGHT_M
    columns = frame_translations[:, 0] / PIXEL_WIDTH_M
    return numpy.stack([rows, columns], axis=1) - min_translation


def test_the_frame_corner_is_the_corner_ptychodus_anchors_its_patches_at() -> None:
    product = make_product()
    frame = build_frame(product)
    corners = _corners(frame.translations, frame.min_translation)
    object_geometry = product.object_.get_geometry()
    probe_geometry = product.probes.get_geometry()

    for corner, position in zip(corners, product.probe_positions):
        center = object_geometry.map_coordinates_probe_to_object(position)
        bounds = probe_geometry.resolve_patch_bounds(center.x_px, center.y_px)
        assert corner[0] == pytest.approx(bounds.y_slice.start + bounds.dy, abs=1e-9)
        assert corner[1] == pytest.approx(bounds.x_slice.start + bounds.dx, abs=1e-9)


def test_the_bases_step_rows_along_y_and_columns_along_x() -> None:
    frame = build_frame(make_product())
    numpy.testing.assert_allclose(
        frame.obj_basis, [[0.0, PIXEL_WIDTH_M], [PIXEL_HEIGHT_M, 0.0], [0.0, 0.0]]
    )
    assert frame.detector_basis[1, 0] > 0.0 and frame.detector_basis[0, 1] > 0.0


def test_translations_are_relative_to_the_object_center() -> None:
    """cdtools holds them in single precision, where an absolute stage coordinate would
    cost a visible fraction of a pixel."""
    far_away_m = 3.0e-3
    product = make_product()
    object_ = product.object_
    center = object_.get_center()
    shifted = replace(
        product,
        object_=Object(
            array=object_.get_array(),
            pixel_geometry=object_.get_pixel_geometry(),
            center=ObjectCenter(x_m=center.x_m + far_away_m, y_m=center.y_m - far_away_m),
        ),
        probe_positions=ProbePositionSequence(
            [
                replace(p, x_m=p.x_m + far_away_m, y_m=p.y_m - far_away_m)
                for p in product.probe_positions
            ]
        ),
    )
    numpy.testing.assert_allclose(
        build_frame(shifted).translations, build_frame(product).translations, atol=1.0e-15
    )
    assert numpy.max(numpy.abs(build_frame(shifted).translations)) < 1.0e-6


def test_the_detector_basis_is_the_far_field_pitch_of_the_probe_grid() -> None:
    product = make_product()
    frame = build_frame(product)
    detector_pixels = PixelGeometry(
        width_m=float(frame.detector_basis[0, 1]), height_m=float(frame.detector_basis[1, 0])
    )
    implied = ProbeGeometry.from_far_field(
        detector_pixels,
        ImageExtent(width_px=PROBE_WIDTH_PX, height_px=PROBE_HEIGHT_PX),
        wavelength_m=METADATA.photon_wavelength_m,
        distance_m=METADATA.detector_distance_m,
    )
    assert implied.pixel_width_m == pytest.approx(PIXEL_WIDTH_M)
    assert implied.pixel_height_m == pytest.approx(PIXEL_HEIGHT_M)


class TestTranslationOffsets:
    def test_an_offset_moves_rows_along_y_and_columns_along_x(self) -> None:
        positions = make_positions()
        offsets = numpy.zeros((len(positions), 2))
        offsets[0] = (0.5, -2.0)
        moved = apply_translation_offsets(
            positions, offsets, pixel_width_m=PIXEL_WIDTH_M, pixel_height_m=PIXEL_HEIGHT_M
        )
        assert moved[0].y_m - positions[0].y_m == pytest.approx(0.5 * PIXEL_HEIGHT_M)
        assert moved[0].x_m - positions[0].x_m == pytest.approx(-2.0 * PIXEL_WIDTH_M)
        assert moved[1] == positions[1]

    def test_a_moved_position_keeps_its_index_and_photon_count(self) -> None:
        positions = make_positions()
        moved = apply_translation_offsets(
            positions,
            numpy.ones((len(positions), 2)),
            pixel_width_m=PIXEL_WIDTH_M,
            pixel_height_m=PIXEL_HEIGHT_M,
        )
        assert [p.index for p in moved] == [p.index for p in positions]
        assert [p.probe_photon_count for p in moved] == [p.probe_photon_count for p in positions]

    def test_a_row_per_position_is_required(self) -> None:
        with pytest.raises(ValueError, match='translation offsets'):
            apply_translation_offsets(
                make_positions(),
                numpy.zeros((2, 2)),
                pixel_width_m=PIXEL_WIDTH_M,
                pixel_height_m=PIXEL_HEIGHT_M,
            )


class TestVerifyProduct:
    def test_the_fixture_is_accepted(self) -> None:
        verify_product(make_product())

    def test_near_field_is_refused(self) -> None:
        with pytest.raises(ValueError, match='far-field'):
            verify_product(make_product(metadata=replace(METADATA, far_field=False)))

    def test_a_multislice_object_is_refused(self) -> None:
        with pytest.raises(ValueError, match='single-layer'):
            verify_product(make_product(object_=make_object(num_layers=2)))

    def test_a_probe_on_another_grid_is_refused(self) -> None:
        product = make_product()
        probes = ProbeEnsemble(
            array=product.probes.get_array(),
            opr_weights=None,
            pixel_geometry=PixelGeometry(width_m=2 * PIXEL_WIDTH_M, height_m=PIXEL_HEIGHT_M),
        )
        with pytest.raises(ValueError, match='Resample'):
            verify_product(replace(product, probes=probes))

    @pytest.mark.parametrize(
        ('dx_px', 'dy_px'),
        [
            pytest.param(-(OBJECT_WIDTH_PX - PROBE_WIDTH_PX) / 2 - 0.4, 0.0, id='left'),
            pytest.param((OBJECT_WIDTH_PX - PROBE_WIDTH_PX) / 2 + 1.0, 0.0, id='right'),
            pytest.param(0.0, -(OBJECT_HEIGHT_PX - PROBE_HEIGHT_PX) / 2 - 0.4, id='top'),
            pytest.param(0.0, (OBJECT_HEIGHT_PX - PROBE_HEIGHT_PX) / 2 + 1.0, id='bottom'),
        ],
    )
    def test_a_window_off_the_canvas_is_refused_naming_its_scan_index(
        self, dx_px: float, dy_px: float
    ) -> None:
        product = make_product()
        center = product.object_.get_center()
        stray = replace(
            product.probe_positions[0],
            index=777,
            x_m=center.x_m + dx_px * PIXEL_WIDTH_M,
            y_m=center.y_m + dy_px * PIXEL_HEIGHT_M,
        )
        positions = ProbePositionSequence([*product.probe_positions, stray])

        with pytest.raises(ValueError, match=r'scan index 777.*ExtraPaddingX'):
            verify_product(replace(product, probe_positions=positions))

    def test_a_window_flush_with_the_edge_is_accepted(self) -> None:
        """The integer window decides, as it does for ptychodus's own patches: the
        sub-pixel remainder is a shift of the probe within it."""
        product = make_product()
        center = product.object_.get_center()
        edge = ProbePosition(
            index=5,
            x_m=center.x_m + ((OBJECT_WIDTH_PX - PROBE_WIDTH_PX) / 2 + 0.6) * PIXEL_WIDTH_M,
            y_m=center.y_m - (OBJECT_HEIGHT_PX - PROBE_HEIGHT_PX) / 2 * PIXEL_HEIGHT_M,
        )
        verify_product(replace(product, probe_positions=ProbePositionSequence([edge])))
