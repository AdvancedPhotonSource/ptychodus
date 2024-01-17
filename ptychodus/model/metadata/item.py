from ...api.experiment import ExperimentMetadata
from ...api.observer import Observable


class MetadataRepositoryItem(Observable):

    def __init__(self, metadata: ExperimentMetadata) -> None:
        super().__init__()
        self._name = metadata.name
        self._comments = metadata.comments
        self._probeEnergyInElectronVolts = metadata.probeEnergyInElectronVolts
        self._detectorObjectDistanceInMeters = metadata.detectorObjectDistanceInMeters

    def setName(self, name: str) -> None:
        if self._name != name:
            self._name = name
            self.notifyObservers()

    def getName(self) -> str:
        return self._name

    def setComments(self, comments: str) -> None:
        if self._comments != comments:
            self._comments = comments
            self.notifyObservers()

    def getComments(self) -> str:
        return self._comments

    def setDetectorObjectDistanceInMeters(self, distanceInMeters: float) -> None:
        if self._detectorObjectDistanceInMeters != distanceInMeters:
            self._detectorObjectDistanceInMeters = distanceInMeters
            self.notifyObservers()

    def getDetectorObjectDistanceInMeters(self) -> float:
        return max(0., self._detectorObjectDistanceInMeters)

    def setProbeEnergyInElectronVolts(self, energyInElectronVolts: float) -> None:
        if self._probeEnergyInElectronVolts != energyInElectronVolts:
            self._probeEnergyInElectronVolts = energyInElectronVolts
            self.notifyObservers()

    def getProbeEnergyInElectronVolts(self) -> float:
        return max(0., self._probeEnergyInElectronVolts)

    def getProbeWavelengthInMeters(self) -> float:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planckConstant_eV_per_Hz = 4.135667696e-15
        lightSpeedInMetersPerSecond = 299792458
        hc_eVm = planckConstant_eV_per_Hz * lightSpeedInMetersPerSecond
        return hc_eVm / self.getProbeEnergyInElectronVolts()

    def getMetadata(self) -> ExperimentMetadata:
        return ExperimentMetadata(
            name=self._name,
            comments=self._comments,
            probeEnergyInElectronVolts=self._probeEnergyInElectronVolts,
            detectorObjectDistanceInMeters=self._detectorObjectDistanceInMeters,
        )
