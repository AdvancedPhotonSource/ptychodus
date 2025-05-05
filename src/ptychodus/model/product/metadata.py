from __future__ import annotations
from abc import abstractmethod, ABC
import logging

from ptychodus.api.parametric import Parameter, ParameterGroup
from ptychodus.api.product import ProductMetadata

from .settings import ProductSettings

logger = logging.getLogger(__name__)


class UniqueNameFactory(ABC):
    @abstractmethod
    def create_unique_name(self, candidate_name: str) -> str:
        pass


class UniqueStringParameter(Parameter[str]):
    def __init__(
        self, value: str | None, name_factory: UniqueNameFactory, parent: Parameter[str]
    ) -> None:
        super().__init__(parent)
        self._value = name_factory.create_unique_name(value or parent.get_value())
        self._name_factory = name_factory

    def get_value(self) -> str:
        return self._value

    def set_value(self, value: str, *, notify: bool = True) -> None:
        if value:
            if self._value != value:
                self._value = self._name_factory.create_unique_name(value)

                if notify:
                    self.notify_observers()
        else:
            self.notify_observers()

    def get_value_as_string(self) -> str:
        return str(self._value)

    def set_value_from_string(self, value: str) -> None:
        self.set_value(str(value))

    def copy(self) -> UniqueStringParameter:
        return UniqueStringParameter(self.get_value(), self._name_factory, self)


class MetadataRepositoryItem(ParameterGroup):
    def __init__(
        self,
        settings: ProductSettings,
        name_factory: UniqueNameFactory,
        *,
        name: str = '',
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mass_attenuation_m2_kg: float | None = None,
        tomography_angle_deg: float | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings

        self.name = UniqueStringParameter(name, name_factory, settings.name.copy())
        self._add_parameter('name', self.name)

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

        self.tomography_angle_deg = settings.tomography_angle_deg.copy()

        if tomography_angle_deg is not None:
            self.tomography_angle_deg.set_value(tomography_angle_deg)

        self._add_parameter('tomography_angle_deg', self.tomography_angle_deg)

    def assign(self, metadata: ProductMetadata) -> None:
        self.name.set_value(metadata.name)
        self.comments.set_value(metadata.comments)
        self.detector_distance_m.set_value(metadata.detector_distance_m)
        self.probe_energy_eV.set_value(metadata.probe_energy_eV)
        self.probe_photon_count.set_value(metadata.probe_photon_count)
        self.exposure_time_s.set_value(metadata.exposure_time_s)
        self.mass_attenuation_m2_kg.set_value(metadata.mass_attenuation_m2_kg)
        self.tomography_angle_deg.set_value(metadata.tomography_angle_deg)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    def get_metadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self.name.get_value(),
            comments=self.comments.get_value(),
            detector_distance_m=self.detector_distance_m.get_value(),
            probe_energy_eV=self.probe_energy_eV.get_value(),
            probe_photon_count=self.probe_photon_count.get_value(),
            exposure_time_s=self.exposure_time_s.get_value(),
            mass_attenuation_m2_kg=self.mass_attenuation_m2_kg.get_value(),
            tomography_angle_deg=self.tomography_angle_deg.get_value(),
        )
