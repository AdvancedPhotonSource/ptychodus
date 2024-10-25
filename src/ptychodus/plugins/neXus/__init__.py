from ptychodus.api.plugins import PluginRegistry

from .neXusDiffractionFile import NeXusDiffractionFileReader
from .velociprobeScanFile import VelociprobeScanFileReader


def registerPlugins(registry: PluginRegistry) -> None:
    neXusFileReader = NeXusDiffractionFileReader()

    registry.diffractionFileReaders.registerPlugin(
        neXusFileReader,
        simpleName='NeXus',
        displayName='NeXus Master Files (*.h5 *.hdf5)',
    )
    registry.scanFileReaders.registerPlugin(
        VelociprobeScanFileReader.createLaserInterferometerInstance(neXusFileReader),
        simpleName='VelociprobeLaserInterferometer',
        displayName='Velociprobe Scan Files - Laser Interferometer (*.txt)',
    )
    registry.scanFileReaders.registerPlugin(
        VelociprobeScanFileReader.createPositionEncoderInstance(neXusFileReader),
        simpleName='VelociprobePositionEncoder',
        displayName='Velociprobe Scan Files - Position Encoder (*.txt)',
    )
