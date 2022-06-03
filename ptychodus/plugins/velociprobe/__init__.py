from ptychodus.api.plugins import PluginRegistry

from .velociprobeDataFile import VelociprobeDataFileReader
from .velociprobeScanFile import VelociprobeScanFileReader, VelociprobeScanYPositionSource


def registerPlugins(registry: PluginRegistry) -> None:
    dataFileReader = VelociprobeDataFileReader()
    registry.registerPlugin(dataFileReader)
    registry.registerPlugin(
        VelociprobeScanFileReader(dataFileReader, VelociprobeScanYPositionSource.ENCODER))
    registry.registerPlugin(
        VelociprobeScanFileReader(dataFileReader,
                                  VelociprobeScanYPositionSource.LASER_INTERFEROMETER))
