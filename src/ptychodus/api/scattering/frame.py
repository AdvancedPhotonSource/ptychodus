"""The exit frame of an off-axis scattering geometry, and carrying data into it.

Off the forward direction the detector records the Fourier transform of the sample projected
along the exit beam. Scan positions reach that exit frame through a linear map
(:func:`compute_exit_frame_transform`), and a probe stated across the incident beam is
resampled onto it (:func:`transform_probe_to_exit_frame`). :func:`resolve_exit_frame_transform`
is the one rule for where the map applies.

The geometry itself is :mod:`ptychodus.api.scattering.geometry`, which depends on nothing else
in ptychodus, so the metadata can hold it while this module builds on the probe and the affine
primitive.
"""

from __future__ import annotations

import logging

import numpy

from ..affine import AffineTransform
from ..interpolate import resample_complex_at
from ..probe import ProbeEnsemble
from .geometry import ScatteringGeometry, find_scattering_issues

__all__ = [
    'compute_exit_frame_transform',
    'resolve_exit_frame_transform',
    'transform_probe_to_exit_frame',
]

logger = logging.getLogger(__name__)


def compute_exit_frame_transform(
    geometry: ScatteringGeometry, *, grazing_sine_floor: float = 1.0e-6
) -> AffineTransform:
    """The map taking scan positions into the frame where the 2D Fourier relation holds.

    Scan positions are measured across the incident beam. The detector records the
    Fourier transform of the sample projected along the exit beam, onto the plane its
    own axes span. A point of a thin sample's surface reaches the first plane by
    projection along the incident beam and the second by projection along the exit
    beam, so the two planes are related through the surface by a linear map, returned
    here with zero translation. Positions carried through it make the off-axis problem
    transmission ptychography on the detector's ordinary far-field grid.

    The map is the identity for a forward exit beam whatever the incidence. Its
    determinant is negative exactly when the geometry is a reflection, so it cannot be
    decomposed into :class:`ptychodus.api.affine.AffineTransformComponents`.

    Args:
        geometry: Exit-beam and surface directions.
        grazing_sine_floor: Sine of the angle between a beam and the surface at or
            below which the projection counts as degenerate. A beam grazing the surface
            spreads its footprint without bound, so a scan step there has no finite image.
            Raising it refuses more nearly grazing geometries.

    Raises:
        ValueError: if the incident or the exit beam grazes the surface.
    """
    normal = geometry.get_surface_normal()
    incident_sine = abs(geometry.get_incident_sine())
    exit_sine = abs(geometry.get_exit_sine())

    if min(incident_sine, exit_sine) <= grazing_sine_floor:
        raise ValueError(
            'A beam grazes the sample surface (incident sine '
            f'{incident_sine:.3g}, exit sine {exit_sine:.3g}); the projection between '
            'the scan frame and the detector frame is degenerate.'
        )

    # A scan point (x, y) lies on the surface at z = slope_x * x + slope_y * y, and
    # projecting it along the exit beam keeps its components along the detector axes.
    slope_x = -normal[0] / normal[2]
    slope_y = -normal[1] / normal[2]
    column_axis, row_axis = geometry.get_detector_axes()
    return AffineTransform.from_linear_matrix(
        numpy.array(
            [
                [
                    column_axis[0] + column_axis[2] * slope_x,
                    column_axis[1] + column_axis[2] * slope_y,
                ],
                [row_axis[0] + row_axis[2] * slope_x, row_axis[1] + row_axis[2] * slope_y],
            ]
        )
    )


def resolve_exit_frame_transform(
    geometry: ScatteringGeometry, *, far_field: bool, grazing_sine_floor: float = 1.0e-6
) -> AffineTransform:
    """The map from scan positions to the exit frame, or the identity where it is undefined.

    The exit frame is defined for an off-axis geometry in far field with neither beam
    grazing the surface; there this is :func:`compute_exit_frame_transform`.
    A forward geometry is the identity. Anything else is logged as a warning, naming the
    issue, and is also the identity, so the positions stay as the scan recorded them. Every
    place lab-frame positions enter a product resolves the transform here, so all of them
    agree on where it applies.

    Args:
        geometry: Exit-beam and surface directions.
        far_field: Declared propagation regime.
        grazing_sine_floor: See :func:`ptychodus.api.scattering.geometry.find_scattering_issues`.
    """
    if geometry.is_forward:
        return AffineTransform.create_identity()

    issues = find_scattering_issues(
        geometry, far_field=far_field, grazing_sine_floor=grazing_sine_floor
    )

    if issues:
        for issue in issues:
            logger.warning('%s geometry %s: %s', geometry.regime, geometry, issue.describe())

        return AffineTransform.create_identity()

    return compute_exit_frame_transform(geometry, grazing_sine_floor=grazing_sine_floor)


def transform_probe_to_exit_frame(
    probes: ProbeEnsemble,
    geometry: ScatteringGeometry,
    *,
    far_field: bool,
    interpolation_order: int = 3,
    grazing_sine_floor: float = 1.0e-6,
) -> ProbeEnsemble:
    """Carry a probe stated across the incident beam into the exit frame.

    A model probe -- a zone plate, a mirror pair, an aperture -- describes the beam's cross
    section. Off axis, the detector sees it projected through the surface along the exit
    beam: each exit-frame pixel samples the probe where its surface point lies across the
    incident beam (:meth:`ScatteringGeometry.project_onto_surface`). Every mode is
    resampled, zero beyond the array, and the whole ensemble is rescaled so the first
    coherent mode keeps its ``sum(abs(P)**2)``, the photons per exposure. OPR weights pass
    through. Where :func:`resolve_exit_frame_transform` gives the identity, so does this,
    and *probes* is returned unchanged.

    A probe estimated from the patterns, or read from an exit-frame product, is already in
    the exit frame and must not pass through here.

    Args:
        probes: Probe across the incident beam, on the detector's far-field pitch.
        geometry: Exit-beam and surface directions.
        far_field: Declared propagation regime.
        interpolation_order: Spline order of the resampling. Lowering it is faster and
            blurs fine structure.
        grazing_sine_floor: See :func:`ptychodus.api.scattering.geometry.find_scattering_issues`.
    """
    if geometry.is_forward or find_scattering_issues(
        geometry, far_field=far_field, grazing_sine_floor=grazing_sine_floor
    ):
        return probes

    probe_geometry = probes.get_geometry()
    coordinates = probe_geometry.get_transverse_coordinates()
    points_m = geometry.project_onto_surface(coordinates.x_m, coordinates.y_m)
    column = points_m[..., 0] / probe_geometry.pixel_width_m + (probe_geometry.width_px - 1) / 2
    row = points_m[..., 1] / probe_geometry.pixel_height_m + (probe_geometry.height_px - 1) / 2

    array = probes.get_array()
    exit_array = numpy.empty_like(array)

    for index in numpy.ndindex(array.shape[:-2]):
        exit_array[index] = resample_complex_at(
            array[index], row, column, order=interpolation_order
        )

    exit_power = numpy.sum(numpy.square(numpy.abs(exit_array[0])))

    if exit_power > 0.0:
        exit_array *= numpy.sqrt(numpy.sum(numpy.square(numpy.abs(array[0]))) / exit_power)

    return ProbeEnsemble(
        array=exit_array,
        opr_weights=probes.get_opr_weights_or_none(),
        pixel_geometry=probes.get_pixel_geometry(),
    )
