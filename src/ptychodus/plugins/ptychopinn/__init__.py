"""PtychoPINN, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .core import PtychoPINNReconstructorLibrary

__all__ = [
    'PtychoPINNReconstructorLibrary',
]


def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        PtychoPINNReconstructorLibrary(),
        simple_name='PtychoPINN',
        display_name='PtychoPINN',
    )
