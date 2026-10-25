"""A coarse probe and object from accumulated diffraction moments alone, to start from.

Everything here reads one :class:`~ptychodus.api.accumulate.DiffractionMoments` and the
probe positions, so a single pass over the patterns -- in memory or streamed -- is all a
starting product costs. The probe is the back-propagated root of the mean pattern, a
single mode carrying the defocus's quadratic phase and scaled to the incident photon
count; the object is the STXM amplitude with the DPC phase, the shadow of that probe's own phase
curvature removed. The data must be far field.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
import logging

import numpy

from .accumulate import DiffractionMoments
from .geometry import ImageExtent
from .interpolate import interpolate_at_indexes
from .object import ObjectGeometry, compute_object_geometry
from .parallax import ParallaxEstimate, estimate_defocus_from_parallax
from .probe import Probe, ProbeEnsemble, ProbeGeometry
from .probe_positions import ProbePositionSequence
from .product import Product, ProductMetadata, reconcile_defocus_distance_m
from .simulate.object import STXMDPCSignals, generate_layers, generate_stxm_dpc_object
from .simulate.probe import generate_average_pattern_probe, rescale_probe_intensity

__all__ = [
    'DefocusSource',
    'KickstartEstimate',
    'KickstartParameters',
    'estimate_probe_from_moments',
    'estimate_product_from_moments',
]

logger = logging.getLogger(__name__)


class DefocusSource(StrEnum):
    """Where the defocus a kickstarted probe carries came from."""

    STATED = 'stated'
    """The product's focus position, or failing that the defocus a caller set."""
    PARALLAX = 'parallax'
    """Measured from the parallax of the bright-field images."""
    NONE = 'none'
    """Neither: the probe is the flat-phase beam at its focus."""


@dataclass(frozen=True)
class KickstartParameters:
    """How a kickstart estimate is made; every field is optional.

    *defocus_distance_m* is how far past its focus the object sits, positive downstream;
    a product naming a focus position overrides it. With *parallax_min_significance*
    and *parallax_min_defocus_dof*, a parallax defocus is used only when it reaches that
    many standard errors and that many depths of focus; it is checked against a stated
    defocus, warning when they differ in sign or by more than *parallax_rtol* of it.
    Moments accumulated without a binning step give no parallax estimate.

    *incident_photon_count_quantile* sets the unattenuated incident photon count from the
    per-pattern totals. *bright_field_mad_threshold* keeps only the mean pattern's bright
    field before back-propagating it; None keeps the whole pattern.
    *regularization_fov_fraction* damps long-wavelength drift in the integrated DPC
    phase; see :func:`generate_stxm_dpc_object`.
    """

    defocus_distance_m: float = 0.0
    parallax_min_significance: float = 3.0
    parallax_min_defocus_dof: float = 1.0
    parallax_rtol: float = 0.5
    incident_photon_count_quantile: float = 0.99
    bright_field_mad_threshold: float | None = 4.5
    regularization_fov_fraction: float = 0.0


@dataclass(frozen=True)
class KickstartEstimate:
    """A kickstarted product and how it was arrived at.

    `incident_photon_count` is the unattenuated incident photon count the probe is scaled to and
    the object's transmission is referenced to; `monitor_normalized` says whether the
    totals were normalized by measured incident photon counts first. `parallax` is the
    parallax estimate whether or not it was used.
    """

    product: Product
    defocus_distance_m: float
    defocus_source: DefocusSource
    parallax: ParallaxEstimate | None
    incident_photon_count: float
    monitor_normalized: bool


def estimate_probe_from_moments(
    moments: DiffractionMoments,
    probe_geometry: ProbeGeometry,
    *,
    photon_wavelength_m: float,
    detector_distance_m: float,
    defocus_distance_m: float,
    incident_photon_count: float,
    bright_field_mad_threshold: float | None = 4.5,
) -> Probe:
    """The probe whose far field is the mean pattern, *defocus_distance_m* past its focus.

    The root of the mean pattern, its bad pixels inpainted and with only its bright
    field kept when *bright_field_mad_threshold* is given, is back-propagated by
    :func:`generate_average_pattern_probe` onto *probe_geometry*, needing no detector
    pitch, and carried the signed defocus further. The single mode is then scaled so
    ``sum|P|**2`` is *incident_photon_count*.
    """
    probe = generate_average_pattern_probe(
        probe_geometry,
        moments.mean_frame,
        bad_pixels=moments.bad_pixels,
        photon_wavelength_m=photon_wavelength_m,
        detector_distance_m=detector_distance_m,
        defocus_distance_m=defocus_distance_m,
        bright_field_mad_threshold=bright_field_mad_threshold,
    )
    return rescale_probe_intensity(probe, incident_photon_count)


def estimate_product_from_moments(
    moments: DiffractionMoments,
    probe_positions: ProbePositionSequence,
    *,
    metadata: ProductMetadata,
    probe_geometry: ProbeGeometry,
    object_geometry: ObjectGeometry | None = None,
    layer_spacing_m: Sequence[float] = (),
    parameters: KickstartParameters | None = None,
) -> KickstartEstimate:
    """A starting product from far-field diffraction moments and the probe positions.

    1. Totals are normalized by the positions' measured incident photon counts, when they carry
       them, paired by index; the incident photon count is a quantile of the totals.
    2. The defocus is the stated one when nonzero, else a resolved parallax estimate,
       else zero with a warning. The parallax is estimated whenever *moments* were
       binned, and logged whichever is used.
    3. The probe follows :func:`estimate_probe_from_moments`.
    4. The object is :func:`generate_stxm_dpc_object` on *object_geometry* -- by default
       the scan's bounding box plus a probe width -- with that probe setting its
       support and correcting the shadow its phase curvature casts, its transmission
       referenced to the incident photon count. *layer_spacing_m* reslices it into layers.

    Raises:
        ValueError: if *metadata* is not far field, or the moments and positions share
            no index.
    """
    if not metadata.far_field:
        raise ValueError('A kickstart estimate needs far-field diffraction data!')

    parameters = KickstartParameters() if parameters is None else parameters

    if object_geometry is None:
        object_geometry = compute_object_geometry(probe_positions, probe_geometry)

    moments, monitor_normalized = _normalize_by_monitor(moments, probe_positions)
    incident_photon_count = moments.estimate_incident_photon_count(
        quantile=parameters.incident_photon_count_quantile
    )
    parallax = (
        None
        if moments.binning is None
        else estimate_defocus_from_parallax(
            moments,
            probe_positions,
            object_geometry,
            probe_geometry,
            photon_wavelength_m=metadata.photon_wavelength_m,
        )
    )
    defocus_distance_m, defocus_source = _decide_defocus(metadata, parameters, parallax)
    probe = estimate_probe_from_moments(
        moments,
        probe_geometry,
        photon_wavelength_m=metadata.photon_wavelength_m,
        detector_distance_m=metadata.detector_distance_m,
        defocus_distance_m=defocus_distance_m,
        incident_photon_count=incident_photon_count,
        bright_field_mad_threshold=parameters.bright_field_mad_threshold,
    )
    object_ = generate_stxm_dpc_object(
        object_geometry,
        STXMDPCSignals.from_moments(moments),
        probe_positions,
        detector_extent=ImageExtent(
            width_px=probe_geometry.width_px, height_px=probe_geometry.height_px
        ),
        probe=probe,
        incident_photon_count=incident_photon_count,
        regularization_fov_fraction=parameters.regularization_fov_fraction,
    )

    if len(layer_spacing_m) > 0:
        object_ = generate_layers(object_, layer_spacing_m)

    product = Product(
        metadata=metadata,
        probe_positions=probe_positions,
        probes=ProbeEnsemble.from_probe(probe),
        object_=object_,
        losses=[],
    )
    return KickstartEstimate(
        product=product,
        defocus_distance_m=defocus_distance_m,
        defocus_source=defocus_source,
        parallax=parallax,
        incident_photon_count=incident_photon_count,
        monitor_normalized=monitor_normalized,
    )


def _normalize_by_monitor(
    moments: DiffractionMoments, probe_positions: ProbePositionSequence
) -> tuple[DiffractionMoments, bool]:
    """Normalize by the positions' incident photon counts, averaged over duplicate indexes."""
    counts = probe_positions.get_probe_photon_counts()

    if counts is None:
        return moments, False

    indexes = probe_positions.get_indexes()
    unique_indexes = numpy.unique(indexes)
    _, unique_counts = interpolate_at_indexes(indexes, counts, unique_indexes)
    return moments.normalize_by_monitor(unique_indexes, unique_counts), True


def _decide_defocus(
    metadata: ProductMetadata,
    parameters: KickstartParameters,
    parallax: ParallaxEstimate | None,
) -> tuple[float, DefocusSource]:
    resolved = parallax is not None and parallax.is_resolved(
        minimum_significance=parameters.parallax_min_significance,
        minimum_defocus_dof=parameters.parallax_min_defocus_dof,
    )

    if parallax is None:
        logger.info('No parallax estimate of the defocus.')
    else:
        logger.info(
            f'Parallax defocus {parallax.defocus_distance_m * 1e3:.4g} '
            f'+/- {parallax.standard_error_m * 1e3:.2g} mm over {parallax.num_bins} bins; '
            f'depth of focus {parallax.depth_of_focus_m * 1e3:.4g} mm, '
            f'Fresnel number {parallax.fresnel_number:.3g}; '
            f'{"resolved" if resolved else "not resolved"}.'
        )

    stated_m = reconcile_defocus_distance_m(
        metadata.focus_object_distance_m, parameters.defocus_distance_m
    )

    if stated_m != 0.0:
        if resolved and parallax is not None:
            measured_m = parallax.defocus_distance_m

            if numpy.sign(measured_m) != numpy.sign(stated_m) or abs(
                measured_m - stated_m
            ) > parameters.parallax_rtol * abs(stated_m):
                logger.warning(
                    f'The parallax measures a defocus of {measured_m * 1e3:.4g} mm against '
                    f'the {stated_m * 1e3:.4g} mm stated; using the stated defocus.'
                )

        return stated_m, DefocusSource.STATED

    if resolved and parallax is not None:
        logger.info('Using the parallax defocus.')
        return parallax.defocus_distance_m, DefocusSource.PARALLAX

    logger.warning(
        'No defocus is stated and the parallax resolves none; the probe keeps a flat '
        'phase, the beam at its focus.'
    )
    return 0.0, DefocusSource.NONE
