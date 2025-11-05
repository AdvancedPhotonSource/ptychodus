from importlib.metadata import version, PackageNotFoundError
from typing import Final

try:
    __version__ = version('ptychodus')
except PackageNotFoundError:
    __version__ = 'unknown'

try:
    from .ptychodus_stream_processor import PtychodusAdImageProcessor
except ModuleNotFoundError:
    pass

__all__ = [
    'api',
    'cli',
    'model',
    'view',
    'controller',
    'PtychodusAdImageProcessor',
    'VERSION_STRING',
]

VERSION_STRING: Final[str] = f'Ptychodus ({__version__})'
