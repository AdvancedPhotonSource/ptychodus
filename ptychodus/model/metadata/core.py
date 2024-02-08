from ..patterns import DiffractionDatasetSettings
from .builder import MetadataBuilder


class MetadataCore:

    def __init__(self, settings: DiffractionDatasetSettings) -> None:
        self.builder = MetadataBuilder(settings)
