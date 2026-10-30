"""pty-chi, discovered as a plugin rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .core import PtyChiReconstructorLibrary

__all__ = [
    'PtyChiReconstructorLibrary',
]


def register_plugins(registry: PluginRegistry) -> None:
    registry.reconstructor_libraries.register_plugin(
        PtyChiReconstructorLibrary(),
        simple_name='PtyChi',
        display_name='pty-chi',
    )
