from __future__ import annotations
import logging

from ptychodus.api.assemble import AssembledDiffractionData, compute_probe_photon_counts_by_index
from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.simulate.object import (
    STXMDPCSignals,
    compute_stxm_dpc_signals,
    generate_stxm_dpc_object,
)

from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


def compute_monitored_signals(
    assembled_data: AssembledDiffractionData, geometry_provider: ObjectGeometryProvider
) -> STXMDPCSignals:
    """STXM-DPC signals normalized by measured incident photon counts from either side.

    Counts recorded with the patterns take precedence, then counts recorded with the
    probe positions; a pattern with neither is left unnormalized.
    """
    positions = ProbePositionSequence(geometry_provider.get_probe_positions())
    return compute_stxm_dpc_signals(
        assembled_data,
        monitor_counts_by_index=compute_probe_photon_counts_by_index(
            assembled_data, positions, fall_back_to_totals=False
        ),
    )


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
            compute_monitored_signals(assembled_data, geometry_provider),
            geometry_provider.get_probe_positions(),
            detector_extent=assembled_data.get_image_extent(),
            probe=probe,
            correct_shadow=correct_shadow,
            regularization_fov_fraction=self.regularization_fov_fraction.get_value(),
        )
        return self._rescale_to_mean_transmission(self._pad_object(object_), geometry_provider)
