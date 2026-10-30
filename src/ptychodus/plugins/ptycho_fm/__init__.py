"""PtychoFM, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.files import FileFormat
from ptychodus.api.plugins import PluginRegistry

from ..h5_diffraction_file import H5DiffractionFileReader
from .core import PtychoFMReconstructorLibrary
from .training_file import PtychoFMProductFileReader, PtychoFMTrainingDataFileWriter

__all__ = [
    'PtychoFMProductFileReader',
    'PtychoFMReconstructorLibrary',
    'PtychoFMTrainingDataFileWriter',
]


def register_plugins(registry: PluginRegistry) -> None:
    # One name for the library and its file formats, so the format offered by default
    # is the selected backend's own.
    registry.reconstructor_libraries.register_plugin(
        PtychoFMReconstructorLibrary(),
        simple_name=PtychoFMProductFileReader.SIMPLE_NAME,
        display_name='PtychoFM',
    )
    registry.diffraction_file_readers.register_plugin(
        # The pattern half is a lone '/dp' dataset, which the generic reader already
        # covers. No pitch: a training pair outlives the instrument that produced it
        # and records nothing about the detector.
        H5DiffractionFileReader(
            data_path='/dp',
            description='PtychoFM Training Pair',
            extensions=FileFormat.HDF5.extensions,
        ),
        simple_name=PtychoFMProductFileReader.SIMPLE_NAME,
    )
    registry.register_product_file_reader_with_adapters(
        PtychoFMProductFileReader(),
        simple_name=PtychoFMProductFileReader.SIMPLE_NAME,
    )
    registry.training_data_file_writers.register_plugin(
        PtychoFMTrainingDataFileWriter(),
        simple_name=PtychoFMProductFileReader.SIMPLE_NAME,
    )
