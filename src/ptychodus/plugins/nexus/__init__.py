from ptychodus.api.plugins import PluginRegistry

from .nexus_diffraction_file import NeXusDiffractionFileReader
from .velociprobe_position_file import VelociprobePositionFileReader


def register_plugins(registry: PluginRegistry) -> None:
    nexus_file_reader = NeXusDiffractionFileReader()

    registry.diffraction_file_readers.register_plugin(
        nexus_file_reader,
        simple_name='NeXus',
        display_name='NeXus Master Files (*.h5 *.hdf5)',
    )
    registry.position_file_readers.register_plugin(
        VelociprobePositionFileReader.create_laser_interferometer_instance(nexus_file_reader),
        simple_name='APS_Velociprobe-LI',
        display_name='APS Velociprobe Files - Laser Interferometer (*.txt)',
    )
    registry.position_file_readers.register_plugin(
        VelociprobePositionFileReader.create_position_encoder_instance(nexus_file_reader),
        simple_name='APS_Velociprobe-PE',
        display_name='APS Velociprobe Files - Position Encoder (*.txt)',
    )
