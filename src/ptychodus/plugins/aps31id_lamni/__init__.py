from ptychodus.api.plugins import PluginRegistry

from .diffraction_file import LamNIDiffractionFileReader
from .position_file import LamNIPositionFileReader


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        LamNIDiffractionFileReader(),
        simple_name='APS_LamNI',
        display_name='APS 31-ID-E LamNI Files (*.h5 *.hdf5)',
    )
    registry.probe_position_file_readers.register_plugin(
        LamNIPositionFileReader(),
        simple_name=LamNIPositionFileReader.SIMPLE_NAME,
        display_name=LamNIPositionFileReader.DISPLAY_NAME,
    )
