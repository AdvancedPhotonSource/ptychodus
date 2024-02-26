from ...api.product import ProductMetadata
from ...api.parametric import Parameter, ParameterRepository


class MetadataRepositoryItem(ParameterRepository):

    def __init__(self, name: Parameter[str], metadata: ProductMetadata) -> None:
        super().__init__('Metadata')  # FIXME snake_case?
        # FIXME support sys.getsizeof(self)
        self._registerParameter('Name', name)
        self.name = name
        self.comments = self._registerStringParameter('Comments', metadata.comments)
        self.probeEnergyInElectronVolts = self._registerRealParameter(
            'ProbeEnergyInElectronVolts', metadata.probeEnergyInElectronVolts, minimum=0.)
        self.detectorDistanceInMeters = self._registerRealParameter(
            'DetectorDistanceInMeters', metadata.detectorDistanceInMeters, minimum=0.)

    def getMetadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self.name.getValue(),
            comments=self.comments.getValue(),
            probeEnergyInElectronVolts=self.probeEnergyInElectronVolts.getValue(),
            detectorDistanceInMeters=self.detectorDistanceInMeters.getValue(),
        )