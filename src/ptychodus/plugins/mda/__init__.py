from ptychodus.api.plugins import PluginRegistry

from .mda_position_file import (
    RASTER_LINE_INDEX_STRIDE,
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
    # The XFM fly scan drives the positioner over two more points per line than the Eiger
    # records, so numbering positions by running count places them two further along with
    # every line. The count is not stated anywhere in the file -- the MCS scalers stop a
    # point earlier still -- so the positions are numbered line-major and the surplus is
    # left unclaimed by whatever per-line frame count the detector turns out to have.
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(
            scale_to_meters=1.0e-3,
            line_index_stride=RASTER_LINE_INDEX_STRIDE,
        ),
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
    # Atomic scans the sample stage open loop: both positioners are piezo driver command
    # voltages ("nano stage X/Y control", V DC) and neither axis records an encoder, so
    # the scale below is the 10 um/V driver calibration rather than a unit conversion.
    # It lives here because the file states the unit as volts and says nothing about what
    # a volt moves. A stage or driver swap changes it.
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-5),
        simple_name='APS_Atomic',
        display_name='APS 34-ID Atomic Files (*.mda)',
    )
