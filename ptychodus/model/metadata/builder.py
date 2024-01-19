from ...api.experiment import ExperimentMetadata


class MetadataBuilder:

    def __init__(self) -> None:
        self._probeEnergyInElectronVolts = 10000.
        self._detectorObjectDistanceInMeters = 1.

    def setProbeEnergyInElectronVolts(self, energyInElectronVolts: float) -> None:
        self._probeEnergyInElectronVolts = energyInElectronVolts

    def setDetectorObjectDistanceInMeters(self, distanceInMeters: float) -> None:
        self._detectorObjectDistanceInMeters = distanceInMeters

    def build(self, name: str) -> ExperimentMetadata:
        return ExperimentMetadata(
            name=name,
            comments='',
            probeEnergyInElectronVolts=self._probeEnergyInElectronVolts,
            detectorObjectDistanceInMeters=self._detectorObjectDistanceInMeters,
        )
