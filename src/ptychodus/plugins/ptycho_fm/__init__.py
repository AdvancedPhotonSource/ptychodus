"""PtychoFM, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from ..h5_diffraction_file import H5DiffractionFileReader
from .core import PtychoFMReconstructorLibrary
from .training_file import PtychoFMProductFileReader

__all__ = [
    'PtychoFMProductFileReader',
    'PtychoFMReconstructorLibrary',
]


def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        PtychoFMReconstructorLibrary(),
        simple_name='PtychoFM',
        display_name='PtychoFM',
    )
    registry.diffraction_file_readers.register_plugin(
        # The pattern half is a lone '/dp' dataset, which the generic reader already
        # covers. No pitch: a training pair outlives the instrument that produced it
        # and records nothing about the detector.
        H5DiffractionFileReader(data_path='/dp'),
        simple_name=PtychoFMProductFileReader.SIMPLE_NAME,
        display_name=PtychoFMProductFileReader.DISPLAY_NAME,
    )
    registry.register_product_file_reader_with_adapters(
        PtychoFMProductFileReader(),
        simple_name=PtychoFMProductFileReader.SIMPLE_NAME,
        display_name=PtychoFMProductFileReader.DISPLAY_NAME,
    )
