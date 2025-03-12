from __future__ import annotations
from abc import ABC, abstractmethod
import logging

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.product import ProductMetadata

from .settings import ProductSettings

logger = logging.getLogger(__name__)


class UniqueNameFactory(ABC):
    @abstractmethod
    def createUniqueName(self, candidateName: str) -> str:
        pass


class MetadataRepositoryItem(ParameterGroup):
    def __init__(
        self,
        settings: ProductSettings,
        nameFactory: UniqueNameFactory,
        *,
        name: str = '',
        comments: str = '',
        detectorDistanceInMeters: float | None = None,
        probeEnergyInElectronVolts: float | None = None,
        probePhotonCount: float | None = None,
        exposureTimeInSeconds: float | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._nameFactory = nameFactory

        self._name = settings.name.copy()
        self._setName(name if name else settings.name.get_value())
        self._add_parameter('name', self._name)
        self.comments = self.create_string_parameter('comments', comments)

        self.detectorDistanceInMeters = settings.detectorDistanceInMeters.copy()

        if detectorDistanceInMeters is not None:
            self.detectorDistanceInMeters.set_value(detectorDistanceInMeters)

        self._add_parameter('detector_distance_m', self.detectorDistanceInMeters)

        self.probeEnergyInElectronVolts = settings.probeEnergyInElectronVolts.copy()

        if probeEnergyInElectronVolts is not None:
            self.probeEnergyInElectronVolts.set_value(probeEnergyInElectronVolts)

        self._add_parameter('probe_energy_eV', self.probeEnergyInElectronVolts)

        self.probePhotonCount = settings.probePhotonCount.copy()

        if probePhotonCount is not None:
            self.probePhotonCount.set_value(probePhotonCount)

        self._add_parameter('probe_photon_count', self.probePhotonCount)

        self.exposureTimeInSeconds = settings.exposureTimeInSeconds.copy()

        if exposureTimeInSeconds is not None:
            self.exposureTimeInSeconds.set_value(exposureTimeInSeconds)

        self._add_parameter('exposure_time_s', self.exposureTimeInSeconds)

        self._index = -1

    def assignItem(self, item: MetadataRepositoryItem, *, notify: bool = True) -> None:
        self.setName(item.getName())
        self.comments.set_value(item.comments.get_value())
        self.detectorDistanceInMeters.set_value(item.detectorDistanceInMeters.get_value())
        self.probeEnergyInElectronVolts.set_value(item.probeEnergyInElectronVolts.get_value())
        self.probePhotonCount.set_value(item.probePhotonCount.get_value())
        self.exposureTimeInSeconds.set_value(item.exposureTimeInSeconds.get_value())

    def assign(self, metadata: ProductMetadata) -> None:
        self.setName(metadata.name)
        self.comments.set_value(metadata.comments)
        self.detectorDistanceInMeters.set_value(metadata.detector_distance_m)
        self.probeEnergyInElectronVolts.set_value(metadata.probe_energy_eV)
        self.probePhotonCount.set_value(metadata.probe_photon_count)
        self.exposureTimeInSeconds.set_value(metadata.exposure_time_s)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    def getName(self) -> str:
        return self._name.get_value()

    def _setName(self, name: str) -> None:
        uniqueName = self._nameFactory.createUniqueName(name)
        self._name.set_value(uniqueName)

    def setName(self, name: str) -> None:
        if name:
            self._setName(name)
        else:
            self._name.notify_observers()

    def getIndex(self) -> int:
        return self._index

    def getMetadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self._name.get_value(),
            comments=self.comments.get_value(),
            detector_distance_m=self.detectorDistanceInMeters.get_value(),
            probe_energy_eV=self.probeEnergyInElectronVolts.get_value(),
            probe_photon_count=self.probePhotonCount.get_value(),
            exposure_time_s=self.exposureTimeInSeconds.get_value(),
        )
