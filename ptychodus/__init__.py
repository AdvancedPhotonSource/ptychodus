from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('ptychodus')
except PackageNotFoundError:
    pass

try:
    from .ptychodusAdImageProcessor import PtychodusAdImageProcessor
except ModuleNotFoundError:
    pass

__all__ = [
    'api',
    'model',
    'view',
    'controller',
    'PtychodusAdImageProcessor',
]
