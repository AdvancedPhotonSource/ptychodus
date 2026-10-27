from __future__ import annotations
import logging

import numpy

from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.simulate.object import STXMDPCSignals, generate_stxm_dpc_object

from .builder import ObjectBuilder
from .settings import ObjectSettings

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


class STXMDPCObjectBuilder(ObjectBuilder):
    """STXM amplitude and DPC phase from the product's diffraction data.

    The product's probe, when it is on the object's pixels, sets the supported region to
    its illumination and, with shadow correction on, removes the probe's own phase
    gradient from the centroids. The object reads the probe when it is built
    and is not rebuilt when the probe changes, so rebuild it after changing the probe.
    """

    def __init__(self, settings: ObjectSettings) -> None:
        super().__init__(settings, 'stxm_dpc')
        self._settings = settings

        # The wavelength, detector distance and regime are the experiment's, read from
        # the provider at build time.
        self.regularization_fov_fraction = settings.dpc_regularization_fov_fraction.copy()
        self._add_parameter('regularization_fov_fraction', self.regularization_fov_fraction)
        self.shadow_correction = settings.dpc_shadow_correction.copy()
        self._add_parameter('shadow_correction', self.shadow_correction)

    def copy(self) -> STXMDPCObjectBuilder:
        builder = STXMDPCObjectBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def reads_diffraction_data(self) -> bool:
        return True

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        if not geometry_provider.far_field:
            raise ValueError(
                'STXM-DPC requires far-field data: the centre-of-mass theorem relates a'
                ' far-field centroid to the exit-wave phase gradient.'
            )

        assembled_data = geometry_provider.get_assembled_data()

        if assembled_data is None:
            # Not yet bound; the build is retried once the product has data.
            raise GeometryNotDefinedError('No diffraction data to estimate the object from.')

        correct_shadow = self.shadow_correction.get_value()
        probe = self._get_probe_on_object_pixels(
            geometry_provider,
            consequence='using the hull of the positions and skipping the shadow correction',
        )

        if correct_shadow and self._probe_supplier is None:
            logger.warning('No probe to correct the STXM-DPC shadow term with; skipping it.')

        object_ = generate_stxm_dpc_object(
            geometry_provider.get_object_geometry(),
            compute_decay_corrected_signals(geometry_provider),
            geometry_provider.get_probe_positions(),
            detector_extent=assembled_data.get_image_extent(),
            probe=probe,
            correct_shadow=correct_shadow,
            regularization_fov_fraction=self.regularization_fov_fraction.get_value(),
        )
        return self._rescale_to_mean_transmission(self._pad_object(object_), geometry_provider)
