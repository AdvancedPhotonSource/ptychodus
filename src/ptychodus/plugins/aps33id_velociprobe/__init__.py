from ptychodus.api.plugins import PluginRegistry

from .aps33id_velociprobe_diffraction_file import VelociprobeDiffractionFileReader
from .aps33id_velociprobe_position_file import VelociprobePositionFileReader


def register_plugins(registry: PluginRegistry) -> None:
    diffraction_file_reader = VelociprobeDiffractionFileReader()

    registry.diffraction_file_readers.register_plugin(
        diffraction_file_reader,
        simple_name='APS_Velociprobe',
        display_name='APS 33-ID Velociprobe Files (*.h5 *.hdf5)',
    )
    registry.position_file_readers.register_plugin(
        VelociprobePositionFileReader.create_laser_interferometer_instance(diffraction_file_reader),
        simple_name='APS_Velociprobe_LI',
        display_name='APS 33-ID Velociprobe Files - Laser Interferometer (*.txt)',
    )
    registry.position_file_readers.register_plugin(
        VelociprobePositionFileReader.create_position_encoder_instance(diffraction_file_reader),
        simple_name='APS_Velociprobe_PE',
        display_name='APS 33-ID Velociprobe Files - Position Encoder (*.txt)',
    )
