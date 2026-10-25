"""ptychodus's own reconstructors, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .core import PtychodusReconstructorLibrary

__all__ = [
    'PtychodusReconstructorLibrary',
]


def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        PtychodusReconstructorLibrary(),
        simple_name='Ptychodus',
        display_name='Ptychodus',
    )
