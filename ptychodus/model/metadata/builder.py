from ...api.product import ProductMetadata
from .settings import MetadataSettings


class MetadataBuilder:

    def __init__(self, settings: MetadataSettings) -> None:
        self._settings = settings

    def build(self, name: str, comments: str = '') -> ProductMetadata:
        return ProductMetadata(
            name=name,
            comments=comments,
            probeEnergyInElectronVolts=float(self._settings.probeEnergyInElectronVolts.value),
            detectorDistanceInMeters=float(self._settings.detectorDistanceInMeters.value),
        )
