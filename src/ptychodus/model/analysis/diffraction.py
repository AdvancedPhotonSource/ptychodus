import numpy

from ptychodus.api.propagator import (
    FraunhoferPropagator,
    PropagatorParameters,
)

from ..product import ProductRepository
from .interpolators import BarycentricArrayInterpolator


class DiffractionSimulator:
    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository

    def simulate(self, product_index: int) -> None:
        product = self._repository[product_index].get_product()
        # TODO also support multislice
        interpolator = BarycentricArrayInterpolator(product.object_.get_layer(0))
        probe_geometry = product.probes.get_geometry()

        propagator_parameters = PropagatorParameters(
            wavelength_m=product.metadata.probe_wavelength_m,
            width_px=probe_geometry.width_px,
            height_px=probe_geometry.height_px,
            pixel_width_m=probe_geometry.pixel_width_m,
            pixel_height_m=probe_geometry.pixel_height_m,
            propagation_distance_m=product.metadata.detector_distance_m,
        )

        # TODO also support near-field propagation
        propagator = FraunhoferPropagator(propagator_parameters)

        for probe_position, probe in zip(product.probe_positions, product.probes):
            object_geometry = product.object_.get_geometry()
            object_position = object_geometry.map_coordinates_probe_to_object(probe_position)

            probe_geometry = product.probes.get_geometry()
            object_patch = interpolator.get_patch(
                object_position.coordinate_x_px,
                object_position.coordinate_y_px,
                probe_geometry.width_px,
                probe_geometry.height_px,
            )

            intensity = numpy.zeros((probe_geometry.height_px, probe_geometry.width_px))

            for imode in range(probe.num_incoherent_modes):
                exit_wave = probe.get_incoherent_mode(imode) * object_patch
                wavefield = propagator.propagate(exit_wave)
                intensity += numpy.square(numpy.abs(wavefield))

            print(intensity.sum())  # FIXME apply simulated diffraction patterns
