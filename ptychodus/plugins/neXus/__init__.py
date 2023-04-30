from ptychodus.api.plugins import PluginRegistry

from .neXusDiffractionFile import NeXusDiffractionFileReader
from .velociprobeScanFile import VelociprobeScanFileReader


def registerPlugins(registry: PluginRegistry) -> None:
    neXusFileReader = NeXusDiffractionFileReader()

    registry.registerPlugin(neXusFileReader)
    registry.registerPlugin(
        VelociprobeScanFileReader.createLaserInterferometerInstance(neXusFileReader))
    registry.registerPlugin(
        VelociprobeScanFileReader.createPositionEncoderInstance(neXusFileReader))
