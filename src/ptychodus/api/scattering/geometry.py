"""Exit-beam and sample-surface geometry for kinematic Bragg and Laue ptychography.

When the detector sits off the forward direction, the 2D Fourier relation holds for the
sample projected along the exit beam, not along the incident one. For a sample thin
enough that the probe does not change across its depth -- thickness times ``sin 2θ``
small against the probe -- the problem is then ordinary transmission ptychography in an
affine-transformed frame, the exit frame. The geometry here states the directions that
transform is built from. Phase reconstructed in that frame is ``-G · u``, the lattice
displacement projected onto the scattering vector, rather than a refractive-index phase.

See the Bragg and Laue guide in the documentation for the frames, the conventions and how
a product carries them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

import numpy

from ..typing import RealArrayType

__all__ = [
    'ScatteringGeometry',
    'ScatteringIssue',
    'ScatteringRegime',
    'compute_displacement_m',
    'find_scattering_issues',
]


def _rotate_about_x(angle_rad: float) -> RealArrayType:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return numpy.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotate_about_y(angle_rad: float) -> RealArrayType:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return numpy.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


class ScatteringRegime(StrEnum):
    """Which way the measured beam leaves the sample, derived from the angles."""

    TRANSMISSION = 'transmission'
    """The exit beam continues along the incident one."""
    BRAGG = 'bragg'
    """Off axis, with both beams on the same side of the surface: a reflection."""
    LAUE = 'laue'
    """Off axis, with the exit beam leaving through the far side of the surface."""


class ScatteringIssue(StrEnum):
    """A way an off-axis geometry falls outside what the exit-frame model supports."""

    NEAR_FIELD = 'near_field'
    MULTISLICE = 'multislice'
    GRAZING_INCIDENCE = 'grazing_incidence'
    GRAZING_EXIT = 'grazing_exit'

    def describe(self) -> str:
        """One sentence saying what is wrong and what ptychodus does about it."""
        descriptions = {
            ScatteringIssue.NEAR_FIELD: (
                'The exit-frame transform is derived for far-field geometry; near-field '
                'positions are left untransformed, so the reconstruction treats an off-axis '
                'measurement as transmission.'
            ),
            ScatteringIssue.MULTISLICE: (
                'A multislice object propagates along the incident beam, which ignores the '
                'tilt between the crystal and the exit beam; the thin-sample model holds for '
                'a single slice.'
            ),
            ScatteringIssue.GRAZING_INCIDENCE: (
                'The incident beam grazes the surface, so a scan step has no finite image in '
                'the exit frame; positions are left untransformed.'
            ),
            ScatteringIssue.GRAZING_EXIT: (
                'The exit beam grazes the surface, so the projection onto the detector frame '
                'is degenerate; positions are left untransformed.'
            ),
        }
        return descriptions[self]


@dataclass(frozen=True)
class ScatteringGeometry:
    """Directions of the exit beam and the sample surface relative to the incident beam.

    Angles are stated in the frame probe positions and oriented detector frames share in
    forward transmission: z along the incident beam, x along detector columns and y along
    detector rows.

    * `two_theta_deg` swings the detector arm about +y, taking the exit beam from +z
      toward +x.
    * `detector_gamma_deg` then elevates the exit beam out of the x-z plane toward +y.
      The detector rides the arm, its columns staying in the arm's horizontal plane, so
      its axes and the exit direction are the columns of ``R_y(2θ) R_x(-γ)``.
    * `incidence_angle_deg` is the angle between the incident beam and the surface plane.
      The surface contains y and the direction ``R_y(θ_i) z``, so 90 degrees is normal
      incidence, and the angle turns in the same sense as two-theta: a mirrored instrument
      negates both.

    The exit beam is forward when both detector angles are zero, and the transform is then
    the identity whatever the incidence. Whether the geometry is reflection (Bragg) or
    transmission (Laue) follows from the angles, so it is the :attr:`regime` property rather
    than a field that could contradict them.

    These are not the metadata's ``tomography_angle_deg`` or ``tilt_angle_deg``, which
    orient a sample within a transmission measurement and change neither the detector's
    direction nor the frame positions are in. In a Bragg or Laue geometry, the sample's
    orientation is `incidence_angle_deg`.
    """

    two_theta_deg: float = 0.0
    detector_gamma_deg: float = 0.0
    incidence_angle_deg: float = 90.0

    @property
    def is_forward(self) -> bool:
        """Whether the exit beam continues along the incident one."""
        return self.two_theta_deg == 0.0 and self.detector_gamma_deg == 0.0

    @property
    def is_reflection(self) -> bool:
        """Whether the incident and exit beams lie on the same side of the surface.

        That is Bragg geometry. It is also what makes the position transform mirror the
        scan: projected along beams on one side of the surface, its two views have
        opposite handedness. A beam lying in the surface plane is neither.
        """
        return self.get_incident_sine() * self.get_exit_sine() < 0.0

    @property
    def regime(self) -> ScatteringRegime:
        """Transmission when the exit beam is forward, else Bragg or Laue."""
        if self.is_forward:
            return ScatteringRegime.TRANSMISSION

        return ScatteringRegime.BRAGG if self.is_reflection else ScatteringRegime.LAUE

    def get_incident_sine(self) -> float:
        """Signed sine of the angle between the incident beam and the surface."""
        return float(self.get_surface_normal()[2])

    def get_exit_sine(self) -> float:
        """Signed sine of the angle between the exit beam and the surface."""
        return float(self.get_surface_normal() @ self.get_exit_direction())

    def _get_detector_rotation(self) -> RealArrayType:
        return _rotate_about_y(math.radians(self.two_theta_deg)) @ _rotate_about_x(
            -math.radians(self.detector_gamma_deg)
        )

    def get_exit_direction(self) -> RealArrayType:
        """Unit vector along the exit beam."""
        return self._get_detector_rotation()[:, 2]

    def get_detector_axes(self) -> tuple[RealArrayType, RealArrayType]:
        """Unit vectors along the detector's columns and rows, both normal to the exit beam."""
        rotation = self._get_detector_rotation()
        return rotation[:, 0], rotation[:, 1]

    def get_surface_axes(self) -> tuple[RealArrayType, RealArrayType]:
        """Orthonormal in-plane surface axes: its trace in the x-z plane, then y.

        At normal incidence they are x and y, so a surface map there reads in the same
        coordinates as the probe positions.
        """
        incidence_rad = math.radians(self.incidence_angle_deg)
        trace = numpy.array([math.sin(incidence_rad), 0.0, math.cos(incidence_rad)])
        return trace, numpy.array([0.0, 1.0, 0.0])

    def get_surface_normal(self) -> RealArrayType:
        """Unit normal to the surface, ``(cos θ_i, 0, -sin θ_i)``: toward the source at normal incidence."""
        trace, y_axis = self.get_surface_axes()
        return numpy.cross(y_axis, trace)

    def get_scattering_vector_per_m(self, photon_wavelength_m: float) -> RealArrayType:
        """``G = k (k_f - z)``, the momentum transfer that meets the Bragg condition at the detector center.

        In radians per meter, in the frame the angles are stated in. It is zero for a
        forward exit beam.
        """
        wavenumber_per_m = 2.0 * math.pi / photon_wavelength_m
        return wavenumber_per_m * (self.get_exit_direction() - numpy.array([0.0, 0.0, 1.0]))

    def project_onto_surface(self, u_m: RealArrayType, v_m: RealArrayType) -> RealArrayType:
        """The surface point each exit-frame coordinate is the projection of, as ``(..., 3)``.

        A point ``r`` on the surface projects along the exit beam to ``(e1 · r, e2 · r)``
        on the detector's axes and satisfies ``n · r = 0``; the three equations are solved
        for ``r``. Its x and y are where the point lies across the incident beam, and its
        components along :meth:`get_surface_axes` are where it lies on the surface.

        Raises:
            numpy.linalg.LinAlgError: if the exit beam lies in the surface plane, where
                the projection is degenerate.
        """
        column_axis, row_axis = self.get_detector_axes()
        system = numpy.vstack((column_axis, row_axis, self.get_surface_normal()))
        inverse = numpy.linalg.inv(system)
        return u_m[..., numpy.newaxis] * inverse[:, 0] + v_m[..., numpy.newaxis] * inverse[:, 1]


def find_scattering_issues(
    geometry: ScatteringGeometry,
    *,
    far_field: bool,
    num_object_layers: int = 1,
    grazing_sine_floor: float = 1.0e-6,
) -> tuple[ScatteringIssue, ...]:
    """The ways *geometry* falls outside what the exit-frame model supports.

    A forward geometry has none: it is ordinary transmission ptychography. Off axis, the
    model is far field and thin sample, and needs both beams to cross the surface at a
    finite angle.

    Args:
        geometry: Exit-beam and surface directions.
        far_field: Declared propagation regime.
        num_object_layers: Slices in the object; more than one is a multislice model.
        grazing_sine_floor: Sine of the angle between a beam and the surface at or below
            which the beam counts as grazing. Raising it flags more nearly grazing
            geometries.
    """
    if geometry.is_forward:
        return ()

    issues: list[ScatteringIssue] = []

    if not far_field:
        issues.append(ScatteringIssue.NEAR_FIELD)

    if num_object_layers > 1:
        issues.append(ScatteringIssue.MULTISLICE)

    if abs(geometry.get_incident_sine()) <= grazing_sine_floor:
        issues.append(ScatteringIssue.GRAZING_INCIDENCE)

    if abs(geometry.get_exit_sine()) <= grazing_sine_floor:
        issues.append(ScatteringIssue.GRAZING_EXIT)

    return tuple(issues)


def compute_displacement_m(
    phase_rad: RealArrayType, geometry: ScatteringGeometry, *, photon_wavelength_m: float
) -> RealArrayType:
    """The lattice displacement along the scattering vector that an exit-frame phase implies.

    A crystal displaced by ``u`` scatters near ``G`` with the envelope ``exp(-i G · u)``,
    so the displacement component along ``G`` is ``-phase / |G|``. Three ambiguities carry
    over from the phase:

    * a global phase is a constant displacement offset;
    * a linear phase ramp is a homogeneous strain or rotation, relative to the reference
      lattice whose Bragg condition is met at the detector center;
    * the result wraps every ``2π / |G|``, the lattice-plane spacing, as the phase wraps
      every ``2π``.

    Raises:
        ValueError: for a forward geometry, where ``G`` is zero and the phase is a
            transmission phase rather than a displacement.
    """
    magnitude_per_m = float(
        numpy.linalg.norm(geometry.get_scattering_vector_per_m(photon_wavelength_m))
    )

    if magnitude_per_m == 0.0:
        raise ValueError(
            'A forward geometry has no scattering vector to project a displacement on.'
        )

    return -phase_rad / magnitude_per_m
