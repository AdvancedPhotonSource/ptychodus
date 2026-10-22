"""Readers for ESRF ID16A near-field ptychography scans.

A scan directory holds a SPEC-written metadata master, one or more LIMA frame files
and a dark stack. The diffraction and position readers accept the master or any frame
file; the dark readers take ``dark_0000.h5``.

These are holograms, not far-field patterns, so the diffraction reader declares the
near-field regime and the cone-beam geometry it implies.
"""

from typing import Final

from ptychodus.api.plugins import PluginRegistry

from .bad_pixels import ESRFID16ADarkBadPixelsFileReader
from .diffraction import ESRFID16ANearFieldDiffractionFileReader
from .pixel_maps import ESRFID16ADarkFileReader
from .probe_positions import ESRFID16ANearFieldPositionFileReader

__all__ = [
    'ESRFID16ADarkBadPixelsFileReader',
    'ESRFID16ADarkFileReader',
    'ESRFID16ANearFieldDiffractionFileReader',
    'ESRFID16ANearFieldPositionFileReader',
]

_SIMPLE_NAME: Final[str] = 'ESRF_ID16A_NFP'
_DARK_SIMPLE_NAME: Final[str] = 'ESRF_ID16A_Dark'
_DISPLAY_NAME: Final[str] = 'ESRF ID16A Near-Field Ptychography Files (*.h5 *.hdf5)'
_DARK_DISPLAY_NAME: Final[str] = 'ESRF ID16A Dark Field Files (*.h5 *.hdf5)'


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        ESRFID16ANearFieldDiffractionFileReader(),
        simple_name=_SIMPLE_NAME,
        display_name=_DISPLAY_NAME,
    )
    registry.probe_position_file_readers.register_plugin(
        ESRFID16ANearFieldPositionFileReader(),
        simple_name=_SIMPLE_NAME,
        display_name=_DISPLAY_NAME,
    )
    registry.background_file_readers.register_plugin(
        ESRFID16ADarkFileReader(),
        simple_name=_DARK_SIMPLE_NAME,
        display_name=_DARK_DISPLAY_NAME,
    )
    registry.bad_pixels_file_readers.register_plugin(
        ESRFID16ADarkBadPixelsFileReader(),
        simple_name=_DARK_SIMPLE_NAME,
        display_name=_DARK_DISPLAY_NAME,
    )
