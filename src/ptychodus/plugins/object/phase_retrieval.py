"""Objects estimated from the measured data: STXM-DPC and Paganin phase retrieval.

Both read the per-pattern totals the diffraction moments carry, corrected for beam decay
by the I0 the probe positions record, and both use the product's probe, when it is on
the object's pixels, to bound the region the estimate is supported on.
"""

from __future__ import annotations
import logging

import numpy
from pydantic import BaseModel, Field

from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.probe import Probe
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.simulate.object import (
    ObjectGenerator,
    STXMDPCSignals,
    generate_paganin_object,
    generate_stxm_dpc_object,
)

from ._options import expect_options

logger = logging.getLogger(__name__)


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

    positions = geometry_provider.get_probe_positions()
    i0_counts_by_index = (
        positions
        if isinstance(positions, ProbePositionSequence)
        else ProbePositionSequence(positions)
    ).get_probe_photon_counts_by_index()

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

        assembled_data = geometry_provider.get_assembled_data()

        if assembled_data is None:
            # Not yet bound; the build is retried once the product has data.
            raise GeometryNotDefinedError('No diffraction data to estimate the object from.')

        if dpc.shadow_correction and probe is None:
            logger.warning('No probe to correct the STXM-DPC shadow term with; skipping it.')

        return generate_stxm_dpc_object(
            geometry_provider.get_object_geometry(),
            compute_decay_corrected_signals(geometry_provider),
            geometry_provider.get_probe_positions(),
            detector_extent=assembled_data.get_image_extent(),
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

        if geometry_provider.get_assembled_data() is None:
            # Not yet bound; the build is retried once the product has data.
            raise GeometryNotDefinedError('No diffraction data to estimate the object from.')

        return generate_paganin_object(
            geometry_provider.get_object_geometry(),
            compute_decay_corrected_signals(geometry_provider),
            geometry_provider.get_probe_positions(),
            probe=probe,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            propagation_distance_m=geometry_provider.object_plane_propagation_distance_m,
            delta_over_beta=paganin.delta_over_beta,
        )
