from abc import abstractmethod, ABC
import logging

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.product import ProductMetadata

from .settings import ProductSettings

logger = logging.getLogger(__name__)


class UniqueNameFactory(ABC):
    @abstractmethod
    def create_unique_name(self, candidate_name: str) -> str:
        pass


class MetadataRepositoryItem(ParameterGroup):
    def __init__(
        self,
        settings: ProductSettings,
        *,
        name: str = '',
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mass_attenuation_m2_kg: float | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._name_factory: UniqueNameFactory | None = None

        self._name = settings.name.copy()

        if name is not None:
            self.set_name(name)

        self._add_parameter('name', self._name)

        self.comments = self.create_string_parameter('comments', comments)

        self.detector_distance_m = settings.detector_distance_m.copy()

        if detector_distance_m is not None:
            self.detector_distance_m.set_value(detector_distance_m)

        self._add_parameter('detector_distance_m', self.detector_distance_m)

        self.probe_energy_eV = settings.probe_energy_eV.copy()

        if probe_energy_eV is not None:
            self.probe_energy_eV.set_value(probe_energy_eV)

        self._add_parameter('probe_energy_eV', self.probe_energy_eV)

        self.probe_photon_count = settings.probe_photon_count.copy()

        if probe_photon_count is not None:
            self.probe_photon_count.set_value(probe_photon_count)

        self._add_parameter('probe_photon_count', self.probe_photon_count)

        self.exposure_time_s = settings.exposure_time_s.copy()

        if exposure_time_s is not None:
            self.exposure_time_s.set_value(exposure_time_s)

        self._add_parameter('exposure_time_s', self.exposure_time_s)

        self.mass_attenuation_m2_kg = settings.mass_attenuation_m2_kg.copy()

        if mass_attenuation_m2_kg is not None:
            self.mass_attenuation_m2_kg.set_value(mass_attenuation_m2_kg)

        self._add_parameter('mass_attenuation_m2_kg', self.mass_attenuation_m2_kg)

        self._index = -1  # used by ProductRepository

    def assign(self, metadata: ProductMetadata) -> None:
        self.set_name(metadata.name)
        self.comments.set_value(metadata.comments)
        self.detector_distance_m.set_value(metadata.detector_distance_m)
        self.probe_energy_eV.set_value(metadata.probe_energy_eV)
        self.probe_photon_count.set_value(metadata.probe_photon_count)
        self.exposure_time_s.set_value(metadata.exposure_time_s)
        self.mass_attenuation_m2_kg.set_value(metadata.mass_attenuation_m2_kg)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    def _set_name_factory(self, name_factory: UniqueNameFactory) -> None:
        self._name_factory = name_factory
        self.set_name(self.get_name())

    def set_name(self, name: str) -> None:
        if name:
            if self._name_factory is not None:
                name = self._name_factory.create_unique_name(name)

            self._name.set_value(name)
        else:
            self._name.notify_observers()

    def get_name(self) -> str:
        return self._name.get_value()

    def get_metadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self._name.get_value(),
            comments=self.comments.get_value(),
            detector_distance_m=self.detector_distance_m.get_value(),
            probe_energy_eV=self.probe_energy_eV.get_value(),
            probe_photon_count=self.probe_photon_count.get_value(),
            exposure_time_s=self.exposure_time_s.get_value(),
            mass_attenuation_m2_kg=self.mass_attenuation_m2_kg.get_value(),
        )
