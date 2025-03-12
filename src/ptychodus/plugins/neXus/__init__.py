from ptychodus.api.plugins import PluginRegistry

from .neXusDiffractionFile import NeXusDiffractionFileReader
from .velociprobeScanFile import VelociprobeScanFileReader


def register_plugins(registry: PluginRegistry) -> None:
    neXusFileReader = NeXusDiffractionFileReader()

    registry.diffractionFileReaders.register_plugin(
        neXusFileReader,
        simple_name='NeXus',
        display_name='NeXus Master Files (*.h5 *.hdf5)',
    )
    registry.scanFileReaders.register_plugin(
        VelociprobeScanFileReader.createLaserInterferometerInstance(neXusFileReader),
        simple_name='VelociprobeLaserInterferometer',
        display_name='Velociprobe Scan Files - Laser Interferometer (*.txt)',
    )
    registry.scanFileReaders.register_plugin(
        VelociprobeScanFileReader.createPositionEncoderInstance(neXusFileReader),
        simple_name='VelociprobePositionEncoder',
        display_name='Velociprobe Scan Files - Position Encoder (*.txt)',
    )
