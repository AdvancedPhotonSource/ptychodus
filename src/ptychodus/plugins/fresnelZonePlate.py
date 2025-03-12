from ptychodus.api.probe import FresnelZonePlate
from ptychodus.api.plugins import PluginRegistry


def register_plugins(registry: PluginRegistry) -> None:
    registry.fresnelZonePlates.register_plugin(
        FresnelZonePlate(160e-6, 70e-9, 60e-6),
        display_name='2-ID-D',
    )
    registry.fresnelZonePlates.register_plugin(
        FresnelZonePlate(160e-6, 30e-9, 80e-6),
        display_name='HXN',
    )
    registry.fresnelZonePlates.register_plugin(
        FresnelZonePlate(114.8e-6, 60e-9, 40e-6),
        display_name='LYNX',
    )
    registry.fresnelZonePlates.register_plugin(
        FresnelZonePlate(180e-6, 50e-9, 60e-6),
        display_name='Velociprobe',
    )
