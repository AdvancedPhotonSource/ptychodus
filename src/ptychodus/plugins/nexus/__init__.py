from ptychodus.api.plugins import PluginRegistry

from .nexus_diffraction_file import NeXusDiffractionFileReader
from .velociprobe_scan_file import VelociprobeScanFileReader


def register_plugins(registry: PluginRegistry) -> None:
    nexus_file_reader = NeXusDiffractionFileReader()

    registry.diffraction_file_readers.register_plugin(
        nexus_file_reader,
        simple_name='NeXus',
        display_name='NeXus Master Files (*.h5 *.hdf5)',
    )
    registry.scan_file_readers.register_plugin(
        VelociprobeScanFileReader.createLaserInterferometerInstance(nexus_file_reader),
        simple_name='VelociprobeLaserInterferometer',
        display_name='Velociprobe Scan Files - Laser Interferometer (*.txt)',
    )
    registry.scan_file_readers.register_plugin(
        VelociprobeScanFileReader.createPositionEncoderInstance(nexus_file_reader),
        simple_name='VelociprobePositionEncoder',
        display_name='Velociprobe Scan Files - Position Encoder (*.txt)',
    )
