"""ptychodus-store: filesystem-watched catalog service for ptychodus artifacts."""

from importlib.metadata import version, PackageNotFoundError

__all__ = ['__version__']

try:
    # ptychodus_store ships inside the ptychodus distribution
    __version__ = version('ptychodus')
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'
