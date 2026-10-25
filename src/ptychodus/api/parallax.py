"""Defocus from the parallax between bright-field images, from accumulated moments alone.

A defocused probe carries a quadratic phase, so each point of the bright-field disk is
lit by rays crossing the sample at a different place: the ray reaching spatial frequency
``q`` passes ``lambda * z * q`` from the probe's centre. The image any one bright-field
detector pixel records as the probe scans is therefore the sample shifted by
``s = -lambda * z * q``, the parallax tilt-corrected bright-field imaging corrects for.
Fitting that line to the measured shifts gives the defocus, sign included, in the frame
of the forward model: the distance a probe is propagated from its focus to reach the
sample.

The ray picture needs the defocus to exceed the depth of focus ``lambda / NA**2`` --
a Fresnel number of the probe's shadow above one. Nearer focus, each bright-field image
is a differently weighted phase-contrast image rather than a shifted copy of one image,
and registering them still gives shifts that grow with ``q`` -- a systematic slope that
the scatter of the fit cannot reveal.
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
import math

import numpy
from scipy.ndimage import binary_erosion
from skimage.registration import phase_cross_correlation

from .accumulate import DiffractionMoments
from .fourier import fourier_shift_2d
from .object import ObjectGeometry
from .probe import ProbeGeometry, estimate_probe_extent
from .probe_positions import ProbePositionSequence
from .simulate.object import interpolate_scan_values
from .typing import BooleanArrayType, RealArrayType

__all__ = [
    'ParallaxEstimate',
    'estimate_defocus_from_parallax',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallaxEstimate:
    """A defocus fitted to the parallax of `num_bins` bright-field images.

    `standard_error_m` is the defocus's standard error from the scatter of the shifts
    about the fitted line; it is zero when the shifts lie on it exactly.
    `depth_of_focus_m` is the probe's ``lambda / NA**2``, from the second moments of the
    mean pattern as :attr:`ProbeExtent.depth_of_focus_m` measures it. Within about that
    distance of focus the parallax is not a shift, so a fitted defocus no larger than
    it is not evidence of one.
    """

    defocus_distance_m: float
    standard_error_m: float
    depth_of_focus_m: float
    num_bins: int

    @property
    def significance(self) -> float:
        """The defocus in units of its standard error; zero for a defocus of zero."""
        if self.defocus_distance_m == 0.0:
            return 0.0

        if self.standard_error_m > 0.0:
            return abs(self.defocus_distance_m) / self.standard_error_m

        return math.inf

    @property
    def fresnel_number(self) -> float:
        """The defocus in depths of focus, as :attr:`ProbeExtent.fresnel_number`."""
        return abs(self.defocus_distance_m) / self.depth_of_focus_m

    def is_resolved(
        self, *, minimum_significance: float = 3.0, minimum_defocus_dof: float = 1.0
    ) -> bool:
        """Whether the defocus is both far enough from focus and statistically real.

        *minimum_defocus_dof* is the number of depths of focus the defocus must reach,
        and *minimum_significance* the number of its standard errors.
        """
        return (
            self.fresnel_number >= minimum_defocus_dof and self.significance >= minimum_significance
        )


def estimate_defocus_from_parallax(
    moments: DiffractionMoments,
    probe_positions: ProbePositionSequence,
    object_geometry: ObjectGeometry,
    probe_geometry: ProbeGeometry,
    *,
    photon_wavelength_m: float,
    bright_field_fraction: float = 0.5,
    upsample_factor: int = 20,
    minimum_bins: int = 4,
) -> ParallaxEstimate | None:
    """Estimate the probe's defocus from the shifts between its bright-field images.

    *moments* must have been accumulated with a binning step, from far-field patterns on
    *probe_geometry*'s reciprocal grid, one detector pixel being ``1 / (N * dx)``; no
    detector pitch is needed. Binned patterns are used as they are stored, so normalize
    them by a monitor first if the beam drifted.

    1. The bright field is every bin whose mean reaches *bright_field_fraction* of the
       brightest, less the bins on its edge, whose images mix the disk with the dark
       field.
    2. Each remaining bin's values are mapped onto *object_geometry* by
       :func:`interpolate_scan_values`, pairing patterns with *probe_positions* by
       index, and cropped to the scan's hull.
    3. Each image is registered against their mean to ``1 / upsample_factor`` of a
       pixel, and again against the mean of the images so aligned. The cross-correlation
       is not normalized to its phase: whitening amplifies the interpolation's noise
       until it pulls every shift toward zero.
    4. The shifts are fitted by least squares to ``s_b = a - lambda * z * q_b``. The
       offset ``a``, shared by every bin, absorbs both a common shift and wherever the
       beam centre lies, so ``q_b`` is measured from the frame midpoint.
    5. The depth of focus is :func:`estimate_probe_extent`'s, from the mean pattern.

    A featureless sample gives shifts that are noise about zero, and a probe within the
    depth of focus gives a defocus no larger than it; judge the result by
    :meth:`ParallaxEstimate.is_resolved`.

    Returns None when fewer than *minimum_bins* bright-field bins remain or fewer than
    three positions pair with a pattern.

    Raises:
        ValueError: if *moments* holds no binned patterns or its frames do not match
            *probe_geometry*.
    """
    binned_patterns = moments.binned_patterns
    binning = moments.binning

    if binned_patterns is None or binning is None:
        raise ValueError('Parallax needs moments accumulated with a binning step!')

    if moments.mean_frame.shape != (probe_geometry.height_px, probe_geometry.width_px):
        raise ValueError(
            f'Frames of shape {moments.mean_frame.shape} do not match the '
            f'{probe_geometry.height_px} x {probe_geometry.width_px} probe!'
        )

    bright_field = binary_erosion(
        _find_bright_bins(binned_patterns, bright_field_fraction),
        structure=numpy.ones((3, 3), dtype=bool),
    )
    num_bins = int(numpy.count_nonzero(bright_field))

    if num_bins < minimum_bins:
        logger.info(
            f'Only {num_bins} bright-field bins away from its edge; too few for a parallax'
            ' estimate.'
        )
        return None

    paired, positions = probe_positions.interpolate_at_indexes(moments.indexes)

    if len(positions) < 3:
        logger.info('Fewer than three positions pair with a pattern; no parallax estimate.')
        return None

    values = binned_patterns[paired][:, bright_field].astype(numpy.float64)
    grids, supported = interpolate_scan_values(object_geometry, positions, values)
    shifts_yx_px = _register_images(_crop_to_contrast(grids, supported), upsample_factor)

    rows, columns = numpy.nonzero(bright_field)
    bin_center_y_px = (rows + 0.5) * binning.bin_size_y - 0.5 - (probe_geometry.height_px - 1) / 2
    bin_center_x_px = (columns + 0.5) * binning.bin_size_x - 0.5 - (probe_geometry.width_px - 1) / 2
    dq_y_per_m = 1.0 / (probe_geometry.height_px * probe_geometry.pixel_height_m)
    dq_x_per_m = 1.0 / (probe_geometry.width_px * probe_geometry.pixel_width_m)
    q_y_per_m = bin_center_y_px * dq_y_per_m
    q_x_per_m = bin_center_x_px * dq_x_per_m
    shift_y_m = shifts_yx_px[:, 0] * object_geometry.pixel_height_m
    shift_x_m = shifts_yx_px[:, 1] * object_geometry.pixel_width_m

    # Unknowns (a_x, a_y, c), one equation per bin and axis, with c = -lambda * z.
    design = numpy.zeros((2 * num_bins, 3))
    design[:num_bins, 0] = 1.0
    design[:num_bins, 2] = q_x_per_m
    design[num_bins:, 1] = 1.0
    design[num_bins:, 2] = q_y_per_m
    observed = numpy.concatenate((shift_x_m, shift_y_m))
    solution, _, _, _ = numpy.linalg.lstsq(design, observed, rcond=None)
    residual = observed - design @ solution
    variance = float(residual @ residual) / (2 * num_bins - 3)
    slope_variance = variance * float(numpy.linalg.inv(design.T @ design)[2, 2])
    defocus_distance_m = float(-solution[2] / photon_wavelength_m)
    extent = estimate_probe_extent(
        moments.mean_frame,
        probe_geometry,
        photon_wavelength_m=photon_wavelength_m,
        defocus_distance_m=defocus_distance_m,
        bad_pixels=moments.bad_pixels,
    )

    return ParallaxEstimate(
        defocus_distance_m=defocus_distance_m,
        standard_error_m=math.sqrt(slope_variance) / photon_wavelength_m,
        depth_of_focus_m=extent.depth_of_focus_m,
        num_bins=num_bins,
    )


def _find_bright_bins(
    binned_patterns: RealArrayType, bright_field_fraction: float
) -> BooleanArrayType:
    """The bins of the mean binned pattern bright enough to be in the disk."""
    if not 0.0 < bright_field_fraction <= 1.0:
        raise ValueError('The bright-field fraction must lie in (0, 1]!')

    mean_bins = numpy.asarray(numpy.mean(binned_patterns, axis=0, dtype=numpy.float64))
    usable = numpy.isfinite(mean_bins)

    if not usable.any():
        return usable

    return usable & (mean_bins >= bright_field_fraction * numpy.max(mean_bins[usable]))


def _crop_to_contrast(grids: RealArrayType, supported: BooleanArrayType) -> RealArrayType:
    """Each image's contrast within the bounding box of the support.

    Dividing by its mean puts every bin on one contrast scale regardless of how bright it
    is. The box is not tapered: a fixed window is a feature that does not move, and pulls
    large shifts toward zero.
    """
    rows = numpy.flatnonzero(supported.any(axis=1))
    columns = numpy.flatnonzero(supported.any(axis=0))
    box = grids[:, rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    mean = numpy.mean(box, axis=(-2, -1), keepdims=True)
    return box / numpy.where(mean > 0.0, mean, 1.0) - 1.0


def _register_images(images: RealArrayType, upsample_factor: int) -> RealArrayType:
    """The shift ``(dy, dx)`` of each image from their common reference, in pixels.

    The reference is the mean image, then the mean of the images shifted onto it, which
    is sharper than a mean of images that disagree by the very shifts being measured.
    """
    shifts = _register_to(numpy.mean(images, axis=0), images, upsample_factor)
    aligned = numpy.stack(
        [
            fourier_shift_2d(image.astype(numpy.complex128), dx=-dx, dy=-dy).real
            for image, (dy, dx) in zip(images, shifts)
        ]
    )
    return _register_to(numpy.mean(aligned, axis=0), images, upsample_factor)


def _register_to(
    reference: RealArrayType, images: RealArrayType, upsample_factor: int
) -> RealArrayType:
    shifts = numpy.zeros((images.shape[0], 2))

    for index, image in enumerate(images):
        # The returned shift registers the image onto the reference, so the image lies
        # displaced from it by its negative.
        registration, _, _ = phase_cross_correlation(
            reference, image, upsample_factor=upsample_factor, normalization=None
        )
        shifts[index] = -registration

    return shifts
