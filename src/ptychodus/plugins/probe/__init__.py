"""Probe models, discovered as plugins rather than named in a composition root."""

from ptychodus.api.plugins import PluginRegistry

from .aperture import DiskProbeGenerator, RectangularProbeGenerator, SuperGaussianProbeGenerator
from .average_pattern import AveragePatternProbeGenerator
from .optics import FresnelZonePlateProbeGenerator, KBMirrorProbeGenerator
from .polynomial import HermiteGaussProbeGenerator, ZernikeProbeGenerator

__all__ = [
    'AveragePatternProbeGenerator',
    'DiskProbeGenerator',
    'FresnelZonePlateProbeGenerator',
    'HermiteGaussProbeGenerator',
    'KBMirrorProbeGenerator',
    'RectangularProbeGenerator',
    'SuperGaussianProbeGenerator',
    'ZernikeProbeGenerator',
]


def register_plugins(registry: PluginRegistry) -> None:
    # Both names the same, and both the name a settings file, a workflow call and the
    # docs use. A friendlier display name is a separate change, visible wherever a
    # probe model is selected.
    generators = {
        'average_pattern': AveragePatternProbeGenerator(),
        'disk': DiskProbeGenerator(),
        'fresnel_zone_plate': FresnelZonePlateProbeGenerator(),
        'hermite_gauss': HermiteGaussProbeGenerator(),
        'kb_mirror': KBMirrorProbeGenerator(),
        'rectangular': RectangularProbeGenerator(),
        'super_gaussian': SuperGaussianProbeGenerator(),
        'zernike': ZernikeProbeGenerator(),
    }

    for name, generator in generators.items():
        registry.probe_generators.register_plugin(generator, simple_name=name, display_name=name)
