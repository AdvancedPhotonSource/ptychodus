import os

from ptychodus.api.settings import SettingsRegistry

from .compute import GenesisComputeClient
from .settings import GenesisSettings
from .transfer import GenesisGlobusTransferClient
from .executor import GenesisExecutor

from ..diffraction import DiffractionAPI
from ..processing import ProcessingAPI
from ..product import ProductAPI


class GenesisCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        processing_api: ProcessingAPI,
    ) -> None:
        self.settings = GenesisSettings(settings_registry)

        try:
            api_base_url = os.environ['IRI_API_BASE_URL']
        except KeyError:
            pass
        else:
            self.globus_transfer_client = GenesisGlobusTransferClient(
                api_base_url=api_base_url, access_token=os.environ['IRI_TRANSFER_API_TOKEN']
            )
            self.compute_client = GenesisComputeClient(
                api_base_url=api_base_url, access_token=os.environ['IRI_COMPUTE_API_TOKEN']
            )

        self.executor = GenesisExecutor(
            self.settings, settings_registry, diffraction_api, product_api, processing_api
        )
