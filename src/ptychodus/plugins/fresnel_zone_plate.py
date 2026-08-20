from ptychodus.api.simulate.probe import FresnelZonePlate
from ptychodus.api.plugins import PluginRegistry


def register_plugins(registry: PluginRegistry) -> None:
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(160e-6, 70e-9, 60e-6),
        display_name='APS 2-ID-D',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(160e-6, 30e-9, 80e-6),
        display_name='CNM/APS 26-ID-C HXN',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(114.8e-6, 60e-9, 40e-6),
        display_name='APS 31-ID-E LYNX',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(180e-6, 15e-9, 15e-6),
        display_name='APS 33-ID-C PtychoProbe',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(180e-6, 50e-9, 60e-6),
        display_name='APS 33-ID-C VelociProbe',
    )
