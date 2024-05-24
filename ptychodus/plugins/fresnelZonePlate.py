from ptychodus.api.probe import FresnelZonePlate
from ptychodus.api.plugins import PluginRegistry


def registerPlugins(registry: PluginRegistry) -> None:
    registry.fresnelZonePlates.registerPlugin(
        FresnelZonePlate(160e-6, 70e-9, 60e-6),
        displayName='2-ID-D',
    )
    registry.fresnelZonePlates.registerPlugin(
        FresnelZonePlate(160e-6, 30e-9, 80e-6),
        displayName='HXN',
    )
    registry.fresnelZonePlates.registerPlugin(
        FresnelZonePlate(114.8e-6, 60e-9, 40e-6),
        displayName='LYNX',
    )
    registry.fresnelZonePlates.registerPlugin(
        FresnelZonePlate(180e-6, 50e-9, 60e-6),
        displayName='Velociprobe',
    )
