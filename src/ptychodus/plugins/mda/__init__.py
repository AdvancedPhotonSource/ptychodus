from ptychodus.api.plugins import PluginRegistry

from .mda_position_file import MDAFlatScanPositionFileReader, MDAPositionFileReader


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='MDA',
        display_name='EPICS MDA Files (*.mda)',
    )
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-3),
        simple_name='APS_2IDD',
        display_name='APS 2-ID-D Microprobe Files (*.mda)',
    )
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-3),
        simple_name='APS_2IDE',
        display_name='APS 2-ID-E Microprobe Files (*.mda)',
    )
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='APS_BNP',
        display_name='APS 2-ID-D Bionanoprobe Files (*.h5 *.hdf5)',
    )
    registry.probe_position_file_readers.register_plugin(
        MDAFlatScanPositionFileReader(scale_to_meters=1.0e-3),
        simple_name='APS_ISN_MDA',
        display_name='APS 19-ID-E In-situ Nanoprobe Files (*.mda)',
    )
    registry.probe_position_file_readers.register_plugin(
        MDAFlatScanPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='CNM_APS_HXN',
        display_name='CNM/APS 26-ID-C Hard X-ray Nanoprobe Files (*.mda)',
    )
