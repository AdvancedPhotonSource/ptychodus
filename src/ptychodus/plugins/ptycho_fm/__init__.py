"""PtychoFM, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .core import PtychoFMReconstructorLibrary

__all__ = [
    'PtychoFMReconstructorLibrary',
]


def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        PtychoFMReconstructorLibrary(),
        simple_name='PtychoFM',
        display_name='PtychoFM',
    )
