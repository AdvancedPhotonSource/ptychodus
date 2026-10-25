from __future__ import annotations
from collections.abc import Callable
import logging
import math

from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.probe import Probe, ProbeEnsemble
from ptychodus.api.simulate.object import compute_stxm_dpc_signals, generate_stxm_dpc_object

from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class STXMDPCObjectBuilder(ObjectBuilder):
    def __init__(self, settings: ObjectSettings) -> None:
        super().__init__(settings, 'stxm_dpc')
        self._settings = settings
        self._probe_supplier: Callable[[], ProbeEnsemble] | None = None

        # The wavelength, detector distance and regime are the experiment's, read from
        # the provider at build time.
        self.regularization_fov_fraction = settings.dpc_regularization_fov_fraction.copy()
        self._add_parameter('regularization_fov_fraction', self.regularization_fov_fraction)
        self.shadow_correction = settings.dpc_shadow_correction.copy()
        self._add_parameter('shadow_correction', self.shadow_correction)

    def bind_probe_supplier(self, supplier: Callable[[], ProbeEnsemble] | None) -> None:
        self._probe_supplier = supplier

    def copy(self) -> STXMDPCObjectBuilder:
        builder = STXMDPCObjectBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _get_shadow_probe(self, geometry_provider: ObjectGeometryProvider) -> Probe | None:
        """The probe to correct the shadow term with, or None to build uncorrected.

        A geometry edit can rebuild the object before the probe catches up, so a probe
        on other pixels degrades the guess rather than failing it.
        """
        if not self.shadow_correction.get_value():
            return None

        if self._probe_supplier is None:
            logger.warning('No probe to correct the STXM-DPC shadow term with; skipping it.')
            return None

        probe = self._probe_supplier().get_probe_no_opr()
        probe_pixel_geometry = probe.get_pixel_geometry()
        object_pixel_geometry = geometry_provider.get_object_geometry().get_pixel_geometry()

        if not (
            math.isclose(probe_pixel_geometry.width_m, object_pixel_geometry.width_m, rel_tol=1e-6)
            and math.isclose(
                probe_pixel_geometry.height_m, object_pixel_geometry.height_m, rel_tol=1e-6
            )
        ):
            logger.warning(
                f'Probe pixels ({probe_pixel_geometry}) differ from object pixels'
                f' ({object_pixel_geometry}); skipping the STXM-DPC shadow correction.'
            )
            return None

        return probe

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
        object_ = generate_stxm_dpc_object(
            geometry_provider.get_object_geometry(),
            compute_stxm_dpc_signals(assembled_data),
            geometry_provider.get_probe_positions(),
            detector_extent=assembled_data.get_image_extent(),
            probe=self._get_shadow_probe(geometry_provider),
            regularization_fov_fraction=self.regularization_fov_fraction.get_value(),
        )
        return self._rescale_to_mean_transmission(self._pad_object(object_), geometry_provider)
