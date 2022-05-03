from ptychodus.api.plugins import PluginRegistry

from .velociprobeReaders import VelociprobeDataFileReader, VelociprobeScanFileReader, \
        VelociprobeScanYPositionSource


def registerPlugins(registry: PluginRegistry) -> None:
    dataFileReader = VelociprobeDataFileReader()
    registry.registerPlugin(dataFileReader)
    registry.registerPlugin(
        VelociprobeScanFileReader(dataFileReader, VelociprobeScanYPositionSource.ENCODER))
    registry.registerPlugin(
        VelociprobeScanFileReader(dataFileReader,
                                  VelociprobeScanYPositionSource.LASER_INTERFEROMETER))
