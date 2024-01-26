from ...api.experiment import ExperimentMetadata
from ...api.parametric import ParameterRepository


class MetadataRepositoryItem(ParameterRepository):

    def __init__(self, metadata: ExperimentMetadata) -> None:
        super().__init__('Metadata')  # FIXME snake_case?
        self.name = self._registerStringParameter('Name', metadata.name)
        self.comments = self._registerStringParameter('Comments', metadata.comments)
        self.probeEnergyInElectronVolts = self._registerRealParameter(
            'ProbeEnergyInElectronVolts', metadata.probeEnergyInElectronVolts, minimum=0.)
        self.detectorObjectDistanceInMeters = self._registerRealParameter(
            'DetectorObjectDistanceInMeters', metadata.detectorObjectDistanceInMeters, minimum=0.)

    @property
    def probeWavelengthInMeters(self) -> float:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planckConstant_eV_per_Hz = 4.135667696e-15
        lightSpeedInMetersPerSecond = 299792458
        hc_eVm = planckConstant_eV_per_Hz * lightSpeedInMetersPerSecond
        return hc_eVm / self.probeEnergyInElectronVolts.getValue()

    def getMetadata(self) -> ExperimentMetadata:
        return ExperimentMetadata(
            name=self.name.getValue(),
            comments=self.comments.getValue(),
            probeEnergyInElectronVolts=self.probeEnergyInElectronVolts.getValue(),
            detectorObjectDistanceInMeters=self.detectorObjectDistanceInMeters.getValue(),
        )
