from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ptychodus")
except PackageNotFoundError:
    # package is not installed
    pass

__name__ = 'ptychodus'
__all__ = ['api', 'model', 'view', 'controller']
