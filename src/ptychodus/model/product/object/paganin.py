from __future__ import annotations
import logging


from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.simulate.object import generate_paganin_object

from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class PaganinObjectBuilder(ObjectBuilder):
    def __init__(self, settings: ObjectSettings) -> None:
        super().__init__(settings, 'paganin')
        self._settings = settings

        # The wavelength and propagation distance are the experiment's, read from the
        # provider at build time; only delta/beta is the specimen's, and unknowable
        # from the product. They enter the filter as one product, so a stale duplicate
        # of either would be absorbed invisibly into a hand-tuned delta/beta.
        self.delta_over_beta = settings.paganin_delta_over_beta.copy()
        self._add_parameter('delta_over_beta', self.delta_over_beta)

    def copy(self) -> PaganinObjectBuilder:
        builder = PaganinObjectBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def reads_diffraction_data(self) -> bool:
        return True

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        assembled_data = geometry_provider.get_assembled_data()

        if assembled_data is None:
            # Not yet bound; the build is retried once the product has data.
            raise GeometryNotDefinedError('No diffraction data to estimate the object from.')

        object_ = generate_paganin_object(
            geometry_provider.get_object_geometry(),
            assembled_data,
            geometry_provider.get_probe_positions(),
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            propagation_distance_m=geometry_provider.object_plane_propagation_distance_m,
            delta_over_beta=self.delta_over_beta.get_value(),
        )
        return self._rescale_to_mean_transmission(self._pad_object(object_), geometry_provider)
