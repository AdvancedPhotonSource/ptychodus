"""The ptychozoon fluorescence enhancer, offered only where ptychozoon is installed.

ptychozoon is maintained separately, so this package shares nothing with the
enhancers ptychodus implements itself: not a declaration, not a settings key.
"""

from importlib.util import find_spec
import logging

from ptychodus.api.plugins import PluginRegistry

from .enhancer import PtychozoonFluorescenceEnhancer

__all__ = [
    'PtychozoonFluorescenceEnhancer',
]

logger = logging.getLogger(__name__)


def register_plugins(registry: PluginRegistry) -> None:
    # Located rather than imported: the package pulls CuPy, which only the child may
    # load, and whether it is present is all registration needs to know.
    if find_spec('ptychozoon') is None:
        logger.info('ptychozoon not found.')
        return

    registry.fluorescence_enhancers.register_plugin(
        PtychozoonFluorescenceEnhancer(), simple_name='ptychozoon', display_name='ptychozoon'
    )
