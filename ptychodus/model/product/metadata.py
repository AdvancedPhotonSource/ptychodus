from __future__ import annotations
from abc import ABC, abstractmethod
import logging

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.product import ProductMetadata

from ..patterns import ProductSettings

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
        probePhotonsPerSecond: float | None = None,
        exposureTimeInSeconds: float | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._nameFactory = nameFactory

        self._name = settings.name.copy()
        self.setName(name if name else settings.name.getValue())
        self._addParameter('name', self._name)
        self.comments = self.createStringParameter('comments', comments)

        self.detectorDistanceInMeters = settings.detectorDistanceInMeters.copy()

        if detectorDistanceInMeters is not None:
            self.detectorDistanceInMeters.setValue(detectorDistanceInMeters)

        self._addParameter('detector_distance_m', self.detectorDistanceInMeters)

        self.probeEnergyInElectronVolts = settings.probeEnergyInElectronVolts.copy()

        if probeEnergyInElectronVolts is not None:
            self.probeEnergyInElectronVolts.setValue(probeEnergyInElectronVolts)

        self._addParameter('probe_energy_eV', self.probeEnergyInElectronVolts)

        self.probePhotonsPerSecond = settings.probePhotonsPerSecond.copy()

        if probePhotonsPerSecond is not None:
            self.probePhotonsPerSecond.setValue(probePhotonsPerSecond)

        self._addParameter('probe_photons_per_second', self.probePhotonsPerSecond)

        self.exposureTimeInSeconds = settings.exposureTimeInSeconds.copy()

        if exposureTimeInSeconds is not None:
            self.exposureTimeInSeconds.setValue(exposureTimeInSeconds)

        self._addParameter('exposure_time_s', self.exposureTimeInSeconds)

        self._index = -1

    def assign(self, item: MetadataRepositoryItem) -> None:
        self.setName(item.getName())
        self.comments.setValue(item.comments.getValue())
        self.detectorDistanceInMeters.setValue(item.detectorDistanceInMeters.getValue())
        self.probeEnergyInElectronVolts.setValue(item.probeEnergyInElectronVolts.getValue())
        self.probePhotonsPerSecond.setValue(item.probePhotonsPerSecond.getValue())
        self.exposureTimeInSeconds.setValue(item.exposureTimeInSeconds.getValue())

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

    def getName(self) -> str:
        return self._name.getValue()

    def setName(self, name: str) -> None:
        uniqueName = self._nameFactory.createUniqueName(name)
        self._name.setValue(uniqueName)

    def getIndex(self) -> int:
        return self._index

    def getMetadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self._name.getValue(),
            comments=self.comments.getValue(),
            detectorDistanceInMeters=self.detectorDistanceInMeters.getValue(),
            probeEnergyInElectronVolts=self.probeEnergyInElectronVolts.getValue(),
            probePhotonsPerSecond=self.probePhotonsPerSecond.getValue(),
            exposureTimeInSeconds=self.exposureTimeInSeconds.getValue(),
        )
