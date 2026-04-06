from typing import Final
import os

from ptychodus.api.settings import SettingsRegistry

from .compute import GenesisComputeClient
from .settings import GenesisSettings
from .transfer import GenesisGlobusTransferClient

NERSC_IRI_SCOPE: Final[str] = (
    'https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api'
)


class GenesisCore:
    def __init__(self, registry: SettingsRegistry) -> None:
        self.settings = GenesisSettings(registry)

        try:
            api_base_url = os.environ['IRI_API_BASE_URL']
        except KeyError:
            pass
        else:
            self.globus_transfer_client = GenesisGlobusTransferClient(
                api_base_url=api_base_url, token=os.environ['IRI_TRANSFER_API_TOKEN']
            )
            self.compute_client = GenesisComputeClient(
                api_base_url=api_base_url, token=os.environ['IRI_COMPUTE_API_TOKEN']
            )
