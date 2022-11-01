from ptychodus.api.plugins import PluginRegistry

from .lynxScanFile import LynxScanFileReader
from .neXusDiffractionFile import NeXusDiffractionFileReader
from .velociprobeScanFile import VelociprobeScanFileReader


def registerPlugins(registry: PluginRegistry) -> None:
    velociprobeFileReader = VelociprobeScanFileReader()
    neXusFileReader = NeXusDiffractionFileReader(velociprobeFileReader)

    registry.registerPlugin(neXusFileReader)
    registry.registerPlugin(LynxScanFileReader())
    registry.registerPlugin(velociprobeFileReader)
