from ptychodus.api.simulate.probe import FresnelZonePlate
from ptychodus.api.plugins import PluginRegistry


def register_plugins(registry: PluginRegistry) -> None:
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(
            zone_plate_diameter_m=160e-6,
            outermost_zone_width_m=70e-9,
            central_beamstop_diameter_m=60e-6,
        ),
        simple_name='APS_2IDD',
        display_name='APS 2-ID-D',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(
            zone_plate_diameter_m=160e-6,
            outermost_zone_width_m=30e-9,
            central_beamstop_diameter_m=80e-6,
        ),
        simple_name='CNM_APS_HXN',
        display_name='CNM/APS 26-ID-C Hard X-ray Nanoprobe',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(
            zone_plate_diameter_m=114.8e-6,
            outermost_zone_width_m=60e-9,
            central_beamstop_diameter_m=40e-6,
        ),
        simple_name='APS_LamNI',
        display_name='APS 31-ID-E LamNI',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(
            zone_plate_diameter_m=180e-6,
            outermost_zone_width_m=15e-9,
            central_beamstop_diameter_m=15e-6,
        ),
        simple_name='APS_PtychoProbe',
        display_name='APS 33-ID-C PtychoProbe',
    )
    registry.fresnel_zone_plates.register_plugin(
        FresnelZonePlate(
            zone_plate_diameter_m=180e-6,
            outermost_zone_width_m=50e-9,
            central_beamstop_diameter_m=60e-6,
        ),
        simple_name='APS_Velociprobe',
        display_name='APS 33-ID-C VelociProbe',
    )
