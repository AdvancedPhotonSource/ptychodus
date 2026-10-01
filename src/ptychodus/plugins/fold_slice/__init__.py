from ptychodus.api.plugins import PluginRegistry

from ..h5_diffraction_file import H5DiffractionFileReader, H5DiffractionFileWriter
from .position_file import FoldSlicePositionFileReader
from .product_file import FoldSliceProductFileReader

_DIFFRACTION_DATA_PATH = '/dp'
_DISPLAY_NAME = 'fold_slice Files (*.h5 *.hdf5)'
_SIMPLE_NAME = 'fold_slice'


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        # Written by the preprocessing step at the Eiger instruments (VelociProbe, BNP,
        # 2-ID-E, ISN, 9-ID-D), none of which records geometry in the file, so the one
        # Eiger pitch serves the whole family.
        H5DiffractionFileReader(data_path=_DIFFRACTION_DATA_PATH, detector_pixel_size_m=75e-6),
        simple_name=_SIMPLE_NAME,
        display_name=_DISPLAY_NAME,
    )
    registry.diffraction_file_writers.register_plugin(
        H5DiffractionFileWriter(data_path=_DIFFRACTION_DATA_PATH),
        simple_name=_SIMPLE_NAME,
        display_name=_DISPLAY_NAME,
    )
    registry.probe_position_file_readers.register_plugin(
        FoldSlicePositionFileReader(),
        simple_name=_SIMPLE_NAME,
        display_name=_DISPLAY_NAME,
    )
    registry.register_product_file_reader_with_adapters(
        # The same Eiger pitch, needed here to pin the detector distance against the
        # sample pixel size the file does record.
        FoldSliceProductFileReader(detector_pixel_size_m=75e-6),
        simple_name=FoldSliceProductFileReader.SIMPLE_NAME,
        display_name=FoldSliceProductFileReader.DISPLAY_NAME,
    )
