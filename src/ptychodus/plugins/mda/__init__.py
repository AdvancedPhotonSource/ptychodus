from ptychodus.api.plugins import PluginRegistry

from .mda_position_file import (
    MDADetectorChannelPositionFileReader,
    MDAFlatScanPositionFileReader,
    MDAPositionFileReader,
)


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
    # MDAFile is an XDR reader, so the filter names the format it can actually open.
    # Pre-APS-U scans keep their positions in the XRF map instead; see
    # ptychodus.plugins.aps02id_bnp_position_file.
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='APS_BNP',
        display_name='APS 2-ID-D Bionanoprobe Files (*.mda)',
    )
    # ISN fly scans expose one positioner -- the X setpoint -- and record both axis
    # encoders as detector channels, so the flat-scan reader, which needs two
    # positioners, raises on every one of these files. Descriptions observed on real
    # 19idAERO files; the index fallbacks match them and are used only if the
    # descriptions ever change.
    registry.probe_position_file_readers.register_plugin(
        MDADetectorChannelPositionFileReader(
            scale_to_meters=1.0e-3,
            x_description='X Axis',
            y_description='Piezo Y',
            x_index_fallback=1,
            y_index_fallback=0,
        ),
        simple_name='APS_ISN_MDA',
        display_name='APS 19-ID-E In-situ Nanoprobe Files (*.mda)',
    )
    registry.probe_position_file_readers.register_plugin(
        MDAFlatScanPositionFileReader(scale_to_meters=1.0e-6),
        simple_name='CNM_APS_HXN',
        display_name='CNM/APS 26-ID-C Hard X-ray Nanoprobe Files (*.mda)',
    )
