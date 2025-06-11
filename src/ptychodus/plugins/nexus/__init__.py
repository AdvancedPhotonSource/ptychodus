from ptychodus.api.plugins import PluginRegistry

from .nexus_diffraction_file import NeXusDiffractionFileReader
from .velociprobe_position_file import VelociprobePositionFileReader


def register_plugins(registry: PluginRegistry) -> None:
    nexus_file_reader = NeXusDiffractionFileReader()

    registry.diffraction_file_readers.register_plugin(
        nexus_file_reader,
        simple_name='APS_Velociprobe',
        display_name='APS 33-ID Velociprobe Files (*.h5 *.hdf5)',
    )
    registry.position_file_readers.register_plugin(
        VelociprobePositionFileReader.create_laser_interferometer_instance(nexus_file_reader),
        simple_name='APS_Velociprobe_LI',
        display_name='APS 33-ID Velociprobe Files - Laser Interferometer (*.txt)',
    )
    registry.position_file_readers.register_plugin(
        VelociprobePositionFileReader.create_position_encoder_instance(nexus_file_reader),
        simple_name='APS_Velociprobe_PE',
        display_name='APS 33-ID Velociprobe Files - Position Encoder (*.txt)',
    )
