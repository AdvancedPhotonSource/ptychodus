from ptychodus.api.plugins import PluginRegistry

from .lynxScanFile import LynxScanFileReader
from .neXusDiffractionFile import NeXusDiffractionFileReader
from .velociprobeScanFile import VelociprobeScanFileReader


def registerPlugins(registry: PluginRegistry) -> None:
    neXusFileReader = NeXusDiffractionFileReader()
    registry.registerPlugin(neXusFileReader)
    registry.registerPlugin(LynxScanFileReader())
    registry.registerPlugin(VelociprobeScanFileReader(neXusFileReader))
