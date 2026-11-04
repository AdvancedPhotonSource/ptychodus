from ptychodus.api.plugins import PluginRegistry

from .mda_position_file import (
    RASTER_LINE_INDEX_STRIDE,
    MDADetectorChannelPositionFileReader,
    MDAPositionFileReader,
)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-6, description='EPICS MDA Files'),
        simple_name='MDA',
    )
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-3, description='APS 2-ID-D Microprobe Files'),
        simple_name='APS_2IDD',
    )
    # The XFM fly scan drives the positioner over two more points per line than the Eiger
    # records, so numbering positions by running count places them two further along with
    # every line. The count is not stated anywhere in the file -- the MCS scalers stop a
    # point earlier still -- so the positions are numbered line-major and the surplus is
    # left unclaimed by whatever per-line frame count the detector turns out to have.
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(
            scale_to_meters=1.0e-3,
            description='APS 2-ID-E Microprobe Files',
            line_index_stride=RASTER_LINE_INDEX_STRIDE,
        ),
        simple_name='APS_2IDE',
    )
    # MDAFile is an XDR reader, so the filter names the format it can actually open.
    # Pre-APS-U scans keep their positions in the XRF map instead; see
    # ptychodus.plugins.aps02id_bnp_position_file.
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(scale_to_meters=1.0e-6, description='APS 2-ID-D Bionanoprobe Files'),
        simple_name='APS_BNP',
    )
    # ISN fly scans expose one positioner -- the X setpoint -- and record both axis
    # encoders as detector channels, so the positions are read from the channels.
    # Descriptions observed on real 19idAERO files; the index fallbacks match them and are
    # used only if the descriptions ever change.
    registry.probe_position_file_readers.register_plugin(
        MDADetectorChannelPositionFileReader(
            scale_to_meters=1.0e-3,
            description='APS 19-ID-E In-situ Nanoprobe Files',
            x_channel='X Axis',
            y_channel='Piezo Y',
            x_index_fallback=1,
            y_index_fallback=0,
        ),
        simple_name='APS_ISN_MDA',
    )
    # 26-ID-C records the zone-plate piezo encoders as detector channels in every era --
    # in the rank-1 spirals and in each row of the rank-2 rasters, whose outer axis is X in
    # some scans and Y in others -- while the positioners and channel labels changed. The
    # PV names did not, so the channels are chosen by them.
    #
    # The x encoder runs opposite to the Medipix frame's columns. Reconstructing 2019 scan
    # 29 and 2022 scans 29 and 247 under each of the eight axis flips and swaps, only this
    # one (or its twin, y reversed with the opposite defocus) resolves the sample; the
    # others smear it, and the transposed ones fit 1.4 to 3 times worse. Not established
    # for the 2023 Eiger scan, which did not reconstruct under any of them.
    registry.probe_position_file_readers.register_plugin(
        MDADetectorChannelPositionFileReader(
            scale_to_meters=1.0e-6,
            description='CNM/APS 26-ID-C Hard X-ray Nanoprobe Files',
            x_channel='26idcnpi:m34.RBV',
            y_channel='26idcnpi:m35.RBV',
            x_scale_to_meters=-1.0e-6,
        ),
        simple_name='CNM_APS_HXN',
    )
    # 34-ID-E scans the sample stage open loop: both positioners are piezo driver command
    # voltages ("nano stage X/Y control", V DC) and neither axis records an encoder, so
    # the scale below is the 10 um/V driver calibration rather than a unit conversion.
    # It lives here because the file states the unit as volts and says nothing about what
    # a volt moves. A stage or driver swap changes it.
    registry.probe_position_file_readers.register_plugin(
        MDAPositionFileReader(
            scale_to_meters=1.0e-5, description='APS 34-ID-E Microdiffraction Files'
        ),
        simple_name='APS_34IDE',
    )
