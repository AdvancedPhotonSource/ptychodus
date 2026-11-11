"""Objects estimated from the measured data: STXM-DPC, Paganin and spectral phase retrieval.

STXM-DPC and Paganin read the per-pattern totals the diffraction moments carry, and use
the product's probe, when it is on the object's pixels, to bound the region the estimate
is supported on. The spectral estimate reads every pixel of the assembled patterns, so it
waits for the moments as the others do and then corrects the patterns as the moments
were corrected; the probe is its sensing operator, so it needs one. All three correct for
beam decay by the I0 the probe positions record.
"""

from __future__ import annotations
import logging

import numpy
from pydantic import BaseModel, Field

from ptychodus.api.geometry import GeometryNotDefinedError, ImageExtent
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.probe import Probe
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.simulate.object import (
    ObjectGenerator,
    SpectralPreprocessing,
    STXMDPCSignals,
    generate_paganin_object,
    generate_spectral_object,
    generate_stxm_dpc_object,
)

from ._options import expect_options

logger = logging.getLogger(__name__)


def _get_i0_counts_by_index(geometry_provider: ObjectGeometryProvider) -> dict[int, float]:
    """The I0 readings the probe positions carry, keyed by scan index; empty without any."""
    positions = geometry_provider.get_probe_positions()
    return (
        positions
        if isinstance(positions, ProbePositionSequence)
        else ProbePositionSequence(positions)
    ).get_probe_photon_counts_by_index()


def compute_decay_corrected_signals(geometry_provider: ObjectGeometryProvider) -> STXMDPCSignals:
    """STXM-DPC signals from the bound data's moments, corrected for beam decay by I0.

    The I0 the probe positions carry scales each total by ``mean(I0) / I0_j``; without
    any the totals are used as they are.

    Raises:
        GeometryNotDefinedError: while the moments are not yet available, so the build
            is retried once they are.
    """
    moments = geometry_provider.get_diffraction_moments()

    if moments is None:
        raise GeometryNotDefinedError('No diffraction moments to estimate the object from yet.')

    i0_counts_by_index = _get_i0_counts_by_index(geometry_provider)

    if i0_counts_by_index:
        moments = moments.normalize_by_i0(
            numpy.fromiter(i0_counts_by_index.keys(), dtype=numpy.intp),
            numpy.fromiter(i0_counts_by_index.values(), dtype=numpy.float64),
        )

    return STXMDPCSignals.from_moments(moments)


class STXMDPCOptions(BaseModel):
    # The wavelength, detector distance and regime are the experiment's, read from the
    # provider at build time.
    regularization_fov_fraction: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        title='Regularization [FOV fraction]:',
        description=(
            'Damps phase variations longer than this fraction of the canvas, halving them '
            'at exactly that length. Zero disables it.'
        ),
    )
    shadow_correction: bool = Field(
        False,
        title='Probe Shadow Correction',
        description="Subtract the probe's own phase gradient from the centroids.",
    )


class STXMDPCObjectGenerator(ObjectGenerator):
    """STXM amplitude and DPC phase from far-field diffraction data.

    The probe, when there is one, sets the supported region to its illumination and,
    with shadow correction on, removes the probe's own phase gradient from the
    centroids. The object is not rebuilt when the probe changes, so rebuild it after
    changing the probe.
    """

    def get_default_options(self) -> BaseModel:
        return STXMDPCOptions()

    @property
    def reads_diffraction_data(self) -> bool:
        return True

    @property
    def reads_probe(self) -> bool:
        return True

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        dpc = expect_options(options, STXMDPCOptions)

        if not geometry_provider.far_field:
            raise ValueError(
                'STXM-DPC requires far-field data: the centre-of-mass theorem relates a'
                ' far-field centroid to the exit-wave phase gradient.'
            )

        moments = geometry_provider.get_diffraction_moments()

        if moments is None:
            # Not yet bound, or the moments are still being computed; the build is retried.
            raise GeometryNotDefinedError('No diffraction moments to estimate the object from.')

        if dpc.shadow_correction and probe is None:
            logger.warning('No probe to correct the STXM-DPC shadow term with; skipping it.')

        height_px, width_px = moments.mean_frame.shape
        return generate_stxm_dpc_object(
            geometry_provider.get_object_geometry(),
            compute_decay_corrected_signals(geometry_provider),
            geometry_provider.get_probe_positions(),
            detector_extent=ImageExtent(width_px=width_px, height_px=height_px),
            probe=probe,
            correct_shadow=dpc.shadow_correction,
            regularization_fov_fraction=dpc.regularization_fov_fraction,
        )


class PaganinOptions(BaseModel):
    # The wavelength and propagation distance are the experiment's, read from the
    # provider at build time; only delta/beta is the specimen's, and unknowable from
    # the product. They enter the filter as one product, so a stale duplicate of either
    # would be absorbed invisibly into a hand-tuned delta/beta.
    delta_over_beta: float = Field(100.0, ge=1e-3, title='Delta / Beta:')


class PaganinObjectGenerator(ObjectGenerator):
    """Paganin phase retrieval on the STXM intensity of the diffraction data.

    The probe, when there is one, sets the region the mean intensity I_0 is taken over
    to its illumination. The object is not rebuilt when the probe changes, so rebuild it
    after changing the probe.
    """

    def get_default_options(self) -> BaseModel:
        return PaganinOptions()

    @property
    def reads_diffraction_data(self) -> bool:
        return True

    @property
    def reads_probe(self) -> bool:
        return True

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        paganin = expect_options(options, PaganinOptions)

        if geometry_provider.get_diffraction_moments() is None:
            # Not yet bound, or the moments are still being computed; the build is retried.
            raise GeometryNotDefinedError('No diffraction moments to estimate the object from.')

        return generate_paganin_object(
            geometry_provider.get_object_geometry(),
            compute_decay_corrected_signals(geometry_provider),
            geometry_provider.get_probe_positions(),
            probe=probe,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            propagation_distance_m=geometry_provider.object_plane_propagation_distance_m,
            delta_over_beta=paganin.delta_over_beta,
        )


class SpectralOptions(BaseModel):
    # The wavelength, propagation distance and regime are the experiment's, read from the
    # provider at build time.
    preprocessing: SpectralPreprocessing = Field(
        SpectralPreprocessing.OPTIMAL,
        title='Preprocessing:',
        description=(
            'How each intensity, normalized to unit mean, is weighted: optimal is'
            ' max(-floor, 1 - 1/y), threshold keeps the brightest fraction of pixels, and'
            ' identity is the intensity itself.'
        ),
    )
    preprocessing_lower_bound: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        title='Lower Bound:',
        description=(
            'Floor of the optimal preprocessing, which every intensity below 1 / (1 +'
            ' floor) of the mean takes.'
        ),
    )
    threshold_fraction: float = Field(
        0.02,
        ge=0.01,
        le=1.0,
        title='Threshold Fraction:',
        description='Fraction of good pixels the threshold preprocessing sets to one.',
    )
    eigensolver_tolerance: float = Field(
        1.0e-2,
        ge=1.0e-8,
        le=1.0,
        title='Eigensolver Tolerance:',
        description='Relative accuracy of the leading eigenvalue the Lanczos solver seeks.',
    )
    eigensolver_max_restarts: int = Field(
        50,
        ge=1,
        le=1000,
        title='Eigensolver Max Restarts:',
        description='Restarts the Lanczos solver may take before it stops unconverged.',
    )
    reference_amplitude_correction: bool = Field(
        False,
        title='Reference Amplitude Correction',
        description=(
            'Divide by the estimate from the probe alone, evening out pixels fewer'
            ' patterns cover, at twice the cost.'
        ),
    )


class SpectralObjectGenerator(ObjectGenerator):
    """The spectral estimate: the leading eigenvector of the intensity-weighted covariance.

    Reads every pixel of the assembled patterns, corrected for the detector maps they
    still carry and for beam decay by I0, and needs the product's probe on the object's
    pixels. The object is not rebuilt when the probe changes, so rebuild it after
    changing the probe.
    """

    def get_default_options(self) -> BaseModel:
        return SpectralOptions()

    @property
    def reads_diffraction_data(self) -> bool:
        return True

    @property
    def reads_probe(self) -> bool:
        return True

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        spectral = expect_options(options, SpectralOptions)

        if geometry_provider.get_diffraction_moments() is None:
            # Not yet bound, or the moments are still being computed; the build is retried.
            raise GeometryNotDefinedError('No diffraction moments to estimate the object from.')

        assembled_data = geometry_provider.get_assembled_data()

        if assembled_data is None:
            raise GeometryNotDefinedError('No diffraction data to estimate the object from.')

        if probe is None:
            raise ValueError(
                "The spectral estimate needs the product's probe on the object's pixels."
            )

        return generate_spectral_object(
            geometry_provider.get_object_geometry(),
            assembled_data,
            geometry_provider.get_probe_positions(),
            probe=probe,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            propagation_distance_m=geometry_provider.object_plane_propagation_distance_m,
            far_field=geometry_provider.far_field,
            preprocessing=spectral.preprocessing,
            lower_bound=spectral.preprocessing_lower_bound,
            threshold_fraction=spectral.threshold_fraction,
            tolerance=spectral.eigensolver_tolerance,
            max_restarts=spectral.eigensolver_max_restarts,
            correct_amplitude=spectral.reference_amplitude_correction,
            i0_counts_by_index=_get_i0_counts_by_index(geometry_provider),
        )
