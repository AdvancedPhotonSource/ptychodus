from __future__ import annotations
import logging


from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.simulate.object import generate_paganin_object

from .builder import ObjectBuilder
from .stxm_dpc import compute_decay_corrected_signals
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class PaganinObjectBuilder(ObjectBuilder):
    """Paganin phase retrieval on the STXM intensity of the product's diffraction data.

    The product's probe, when it is on the object's pixels, sets the region the mean
    intensity I_0 is taken over to its illumination. The object reads the probe when it is built and is
    not rebuilt when the probe changes, so rebuild it after changing the probe.
    """

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
        if geometry_provider.get_assembled_data() is None:
            # Not yet bound; the build is retried once the product has data.
            raise GeometryNotDefinedError('No diffraction data to estimate the object from.')

        object_ = generate_paganin_object(
            geometry_provider.get_object_geometry(),
            compute_decay_corrected_signals(geometry_provider),
            geometry_provider.get_probe_positions(),
            probe=self._get_probe_on_object_pixels(
                geometry_provider, consequence='using the hull of the positions'
            ),
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            propagation_distance_m=geometry_provider.object_plane_propagation_distance_m,
            delta_over_beta=self.delta_over_beta.get_value(),
        )
        return self._rescale_to_mean_transmission(self._pad_object(object_), geometry_provider)
