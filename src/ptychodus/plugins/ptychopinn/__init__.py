"""PtychoPINN, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .core import PtychoPINNReconstructorLibrary
from .training_file import PtychoPINNTrainingDataFileWriter

__all__ = [
    'PtychoPINNReconstructorLibrary',
    'PtychoPINNTrainingDataFileWriter',
]


def register_plugins(registry: PluginRegistry) -> None:
    # One name for both, so the format offered by default is the selected backend's own.
    simple_name = 'PtychoPINN'

    registry.reconstructor_libraries.register_plugin(
        PtychoPINNReconstructorLibrary(),
        simple_name=simple_name,
        display_name='PtychoPINN',
    )
    registry.training_data_file_writers.register_plugin(
        PtychoPINNTrainingDataFileWriter(), simple_name=simple_name
    )
