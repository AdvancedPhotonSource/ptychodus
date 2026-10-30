from ptychodus.api.simulate.probe import FresnelZonePlate
from ptychodus.api.plugins import PluginRegistry


def register_plugins(registry: PluginRegistry) -> None:
    # Each preset goes by one name, the readable one, so the name a person picks from
    # the menu is the name a script passes.
    zone_plates = {
        'APS 2-ID-D': FresnelZonePlate(
            zone_plate_diameter_m=160e-6,
            outermost_zone_width_m=70e-9,
            central_beamstop_diameter_m=60e-6,
        ),
        'CNM/APS 26-ID-C Hard X-ray Nanoprobe': FresnelZonePlate(
            zone_plate_diameter_m=160e-6,
            outermost_zone_width_m=30e-9,
            central_beamstop_diameter_m=80e-6,
        ),
        'APS 31-ID-E LamNI': FresnelZonePlate(
            zone_plate_diameter_m=114.8e-6,
            outermost_zone_width_m=60e-9,
            central_beamstop_diameter_m=40e-6,
        ),
        'APS 33-ID-C PtychoProbe': FresnelZonePlate(
            zone_plate_diameter_m=180e-6,
            outermost_zone_width_m=15e-9,
            central_beamstop_diameter_m=15e-6,
        ),
        'APS 33-ID-C VelociProbe': FresnelZonePlate(
            zone_plate_diameter_m=180e-6,
            outermost_zone_width_m=50e-9,
            central_beamstop_diameter_m=60e-6,
        ),
    }

    for name, zone_plate in zone_plates.items():
        registry.fresnel_zone_plates.register_plugin(
            zone_plate, simple_name=name, display_name=name
        )
