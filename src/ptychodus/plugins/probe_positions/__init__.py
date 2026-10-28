"""Scan patterns, discovered as plugins rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .cartesian import CartesianProbePositionsGenerator, CartesianVariant
from .concentric import ConcentricProbePositionsGenerator
from .lissajous import LissajousProbePositionsGenerator
from .spiral import SpiralProbePositionsGenerator

__all__ = [
    'CartesianProbePositionsGenerator',
    'CartesianVariant',
    'ConcentricProbePositionsGenerator',
    'LissajousProbePositionsGenerator',
    'SpiralProbePositionsGenerator',
]


def register_plugins(registry: PluginRegistry) -> None:
    # Both names the same, and both the name already in settings files, in the
    # workflow API and in the docs. A friendlier display name is a separate
    # change, visible wherever a pattern is selected.
    for variant in CartesianVariant:
        name = variant.name.lower()
        registry.probe_position_generators.register_plugin(
            CartesianProbePositionsGenerator(variant),
            simple_name=name,
            display_name=name,
        )

    registry.probe_position_generators.register_plugin(
        ConcentricProbePositionsGenerator(),
        simple_name='concentric',
        display_name='concentric',
    )
    registry.probe_position_generators.register_plugin(
        SpiralProbePositionsGenerator(),
        simple_name='spiral',
        display_name='spiral',
    )
    registry.probe_position_generators.register_plugin(
        LissajousProbePositionsGenerator(),
        simple_name='lissajous',
        display_name='lissajous',
    )
