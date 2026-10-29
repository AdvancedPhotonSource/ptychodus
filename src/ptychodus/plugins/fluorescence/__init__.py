"""Fluorescence enhancers ptychodus implements itself, discovered as plugins."""

from ptychodus.api.plugins import PluginRegistry

from .two_step import TwoStepFluorescenceEnhancer
from .vspi import VSPIFluorescenceEnhancer

__all__ = [
    'TwoStepFluorescenceEnhancer',
    'VSPIFluorescenceEnhancer',
]


def register_plugins(registry: PluginRegistry) -> None:
    # Both names the same, and both the name a settings file, a workflow call and the
    # docs use. A friendlier display name is a separate change, visible wherever an
    # enhancer is selected.
    enhancers = {
        'two_step': TwoStepFluorescenceEnhancer(),
        'vspi': VSPIFluorescenceEnhancer(),
    }

    for name, enhancer in enhancers.items():
        registry.fluorescence_enhancers.register_plugin(
            enhancer, simple_name=name, display_name=name
        )
