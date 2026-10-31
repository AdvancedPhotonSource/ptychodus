from importlib.metadata import version, PackageNotFoundError
from typing import Final

try:
    __version__ = version('ptychodus')
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'

__all__ = [
    'api',
    'cli',
    'model',
    'view',
    'controller',
    'VERSION_STRING',
]

VERSION_STRING: Final[str] = f'Ptychodus ({__version__})'
