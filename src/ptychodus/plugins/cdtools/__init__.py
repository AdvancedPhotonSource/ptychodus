"""cdtools, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .core import CDToolsReconstructorLibrary

__all__ = [
    'CDToolsReconstructorLibrary',
]


def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        CDToolsReconstructorLibrary(),
        simple_name='cdtools',
        display_name='cdtools',
    )
