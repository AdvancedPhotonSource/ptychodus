from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('ptychodus')
except PackageNotFoundError:
    pass

try:
    from .ptychodus_stream_processor import PtychodusAdImageProcessor
except ModuleNotFoundError:
    pass

__all__ = [
    'api',
    'model',
    'view',
    'controller',
    'PtychodusAdImageProcessor',
]
