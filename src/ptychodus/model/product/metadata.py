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
        probePhotonCount: float | None = None,
        exposureTimeInSeconds: float | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._nameFactory = nameFactory

        self._name = settings.name.copy()
        self._setName(name if name else settings.name.getValue())
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

        self.probePhotonCount = settings.probePhotonCount.copy()

        if probePhotonCount is not None:
            self.probePhotonCount.setValue(probePhotonCount)

        self._addParameter('probe_photon_count', self.probePhotonCount)

        self.exposureTimeInSeconds = settings.exposureTimeInSeconds.copy()

        if exposureTimeInSeconds is not None:
            self.exposureTimeInSeconds.setValue(exposureTimeInSeconds)

        self._addParameter('exposure_time_s', self.exposureTimeInSeconds)

        self._index = -1

    def assignItem(self, item: MetadataRepositoryItem, *, notify: bool = True) -> None:
        self.setName(item.getName())
        self.comments.setValue(item.comments.getValue())
        self.detectorDistanceInMeters.setValue(item.detectorDistanceInMeters.getValue())
        self.probeEnergyInElectronVolts.setValue(item.probeEnergyInElectronVolts.getValue())
        self.probePhotonCount.setValue(item.probePhotonCount.getValue())
        self.exposureTimeInSeconds.setValue(item.exposureTimeInSeconds.getValue())

    def assign(self, metadata: ProductMetadata) -> None:
        self.setName(metadata.name)
        self.comments.setValue(metadata.comments)
        self.detectorDistanceInMeters.setValue(metadata.detectorDistanceInMeters)
        self.probeEnergyInElectronVolts.setValue(metadata.probeEnergyInElectronVolts)
        self.probePhotonCount.setValue(metadata.probePhotonCount)
        self.exposureTimeInSeconds.setValue(metadata.exposureTimeInSeconds)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

    def getName(self) -> str:
        return self._name.getValue()

    def _setName(self, name: str) -> None:
        uniqueName = self._nameFactory.createUniqueName(name)
        self._name.setValue(uniqueName)

    def setName(self, name: str) -> None:
        if name:
            self._setName(name)
        else:
            self._name.notifyObservers()

    def getIndex(self) -> int:
        return self._index

    def getMetadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self._name.getValue(),
            comments=self.comments.getValue(),
            detectorDistanceInMeters=self.detectorDistanceInMeters.getValue(),
            probeEnergyInElectronVolts=self.probeEnergyInElectronVolts.getValue(),
            probePhotonCount=self.probePhotonCount.getValue(),
            exposureTimeInSeconds=self.exposureTimeInSeconds.getValue(),
        )
