from ...api.product import ProductMetadata
from ...api.parametric import ParameterRepository


class MetadataRepositoryItem(ParameterRepository):

    def __init__(self, metadata: ProductMetadata) -> None:
        super().__init__('Metadata')  # FIXME snake_case?
        # FIXME support sys.getsizeof(self)
        self.name = self._registerStringParameter('Name', metadata.name)
        self.comments = self._registerStringParameter('Comments', metadata.comments)
        self.probeEnergyInElectronVolts = self._registerRealParameter(
            'ProbeEnergyInElectronVolts', metadata.probeEnergyInElectronVolts, minimum=0.)
        self.detectorDistanceInMeters = self._registerRealParameter(
            'DetectorDistanceInMeters', metadata.detectorDistanceInMeters, minimum=0.)

    @property
    def probeWavelengthInMeters(self) -> float:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planckConstant_eV_per_Hz = 4.135667696e-15
        lightSpeedInMetersPerSecond = 299792458
        hc_eVm = planckConstant_eV_per_Hz * lightSpeedInMetersPerSecond
        return hc_eVm / self.probeEnergyInElectronVolts.getValue()

    def getMetadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self.name.getValue(),
            comments=self.comments.getValue(),
            probeEnergyInElectronVolts=self.probeEnergyInElectronVolts.getValue(),
            detectorDistanceInMeters=self.detectorDistanceInMeters.getValue(),
        )
