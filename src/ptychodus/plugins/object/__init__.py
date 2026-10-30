"""Object models, discovered as plugins rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .phase_retrieval import PaganinObjectGenerator, STXMDPCObjectGenerator
from .siemens_star import SiemensStarObjectGenerator
from .texture import (
    DeadLeavesObjectGenerator,
    FractalNoiseObjectGenerator,
    GaussianRandomFieldObjectGenerator,
    RandomObjectGenerator,
)

__all__ = [
    'DeadLeavesObjectGenerator',
    'FractalNoiseObjectGenerator',
    'GaussianRandomFieldObjectGenerator',
    'PaganinObjectGenerator',
    'RandomObjectGenerator',
    'STXMDPCObjectGenerator',
    'SiemensStarObjectGenerator',
]


def register_plugins(registry: PluginRegistry) -> None:
    # Both names the same, and both the name a settings file, a workflow call and the
    # docs use. A friendlier display name is a separate change, visible wherever an
    # object model is selected.
    generators = {
        'dead_leaves': DeadLeavesObjectGenerator(),
        'fractal_noise': FractalNoiseObjectGenerator(),
        'gaussian_random_field': GaussianRandomFieldObjectGenerator(),
        'paganin': PaganinObjectGenerator(),
        'random': RandomObjectGenerator(),
        'siemens_star': SiemensStarObjectGenerator(),
        'stxm_dpc': STXMDPCObjectGenerator(),
    }

    for name, generator in generators.items():
        registry.object_generators.register_plugin(generator, simple_name=name, display_name=name)
