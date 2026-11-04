"""Exit-beam and sample-surface geometry for kinematic Bragg and Laue ptychography.

When the detector sits off the forward direction, the 2D Fourier relation holds for the
sample projected along the exit beam, not along the incident one. For a sample thin
enough that the probe does not change across its depth -- thickness times ``sin 2θ``
small against the probe -- the problem is then ordinary transmission ptychography in an
affine-transformed frame. The geometry here states the directions that transform is built
from. Phase reconstructed in that frame is ``G · u``, the projection of the lattice
displacement onto the scattering vector, rather than a refractive-index phase.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy

from .typing import RealArrayType

__all__ = ['ScatteringGeometry']


def _rotate_about_x(angle_rad: float) -> RealArrayType:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return numpy.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]])


def _rotate_about_y(angle_rad: float) -> RealArrayType:
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return numpy.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])


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
    transmission (Laue) follows from the angles, so it is a property rather than a field
    that could contradict them.
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
        normal = self.get_surface_normal()
        incident_cosine = float(normal[2])
        exit_cosine = float(normal @ self.get_exit_direction())
        return incident_cosine * exit_cosine < 0.0

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
