from ptychodus.api.product import ProductMetadata

from ...patterns import DiffractionDatasetSettings


class MetadataBuilder:

    def __init__(self, settings: DiffractionDatasetSettings) -> None:
        self._settings = settings

    def createDefault(self, name: str, comments: str = '') -> ProductMetadata:
        return ProductMetadata(
            name=name,
            comments=comments,
            probeEnergyInElectronVolts=float(self._settings.probeEnergyInElectronVolts.value),
            detectorDistanceInMeters=float(self._settings.detectorDistanceInMeters.value),
        )
