from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import (
    Product,
    ProductFileReader,
    ProductFileWriter,
    ProductMetadata,
)
from ptychodus.api.reconstructor import LossValue
from ptychodus.api.scan import PositionSequence, ScanPoint

logger = logging.getLogger(__name__)


class H5ProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'HDF5'
    DISPLAY_NAME: Final[str] = 'Ptychodus Product Files (*.h5 *.hdf5)'

    NAME: Final[str] = 'name'
    COMMENTS: Final[str] = 'comments'
    DETECTOR_OBJECT_DISTANCE: Final[str] = 'detector_object_distance_m'
    PROBE_ENERGY: Final[str] = 'probe_energy_eV'
    PROBE_PHOTON_COUNT: Final[str] = 'probe_photon_count'
    EXPOSURE_TIME: Final[str] = 'exposure_time_s'
    MASS_ATTENUATION: Final[str] = 'mass_attenuation_m2_kg'
    TOMOGRAPHY_ANGLE: Final[str] = 'tomography_angle_deg'

    PROBE_ARRAY: Final[str] = 'probe'
    OPR_WEIGHTS: Final[str] = 'opr_weights'
    PROBE_PIXEL_HEIGHT: Final[str] = 'pixel_height_m'
    PROBE_PIXEL_WIDTH: Final[str] = 'pixel_width_m'
    PROBE_POSITION_INDEXES: Final[str] = 'probe_position_indexes'
    PROBE_POSITION_X: Final[str] = 'probe_position_x_m'
    PROBE_POSITION_Y: Final[str] = 'probe_position_y_m'

    OBJECT_ARRAY: Final[str] = 'object'
    OBJECT_CENTER_X: Final[str] = 'center_x_m'
    OBJECT_CENTER_Y: Final[str] = 'center_y_m'
    OBJECT_LAYER_SPACING: Final[str] = 'object_layer_spacing_m'
    OBJECT_PIXEL_HEIGHT: Final[str] = 'pixel_height_m'
    OBJECT_PIXEL_WIDTH: Final[str] = 'pixel_width_m'

    LOSS_EPOCHS: Final[str] = 'loss_epochs'
    LOSS_VALUES: Final[str] = 'loss_values'

    def read(self, file_path: Path) -> Product:
        point_list: list[ScanPoint] = []

        with h5py.File(file_path, 'r') as h5_file:
            probe_photon_count = 0.0

            try:
                probe_photon_count = float(h5_file.attrs[self.PROBE_PHOTON_COUNT])
            except KeyError:
                logger.debug('Probe photon count not found.')

            mass_attenuation_m2_kg = 0.0

            try:
                mass_attenuation_m2_kg = float(h5_file.attrs[self.MASS_ATTENUATION])
            except KeyError:
                logger.debug('Mass attenuation not found.')

            tomography_angle_deg = 0.0

            try:
                tomography_angle_deg = float(h5_file.attrs[self.TOMOGRAPHY_ANGLE])
            except KeyError:
                logger.debug('Tomography angle not found.')

            metadata = ProductMetadata(
                name=str(h5_file.attrs[self.NAME]),
                comments=str(h5_file.attrs[self.COMMENTS]),
                detector_distance_m=float(h5_file.attrs[self.DETECTOR_OBJECT_DISTANCE]),
                probe_energy_eV=float(h5_file.attrs[self.PROBE_ENERGY]),
                probe_photon_count=probe_photon_count,
                exposure_time_s=float(h5_file.attrs[self.EXPOSURE_TIME]),
                mass_attenuation_m2_kg=mass_attenuation_m2_kg,
                tomography_angle_deg=tomography_angle_deg,
            )

            h5_scan_indexes = h5_file[self.PROBE_POSITION_INDEXES]
            h5_scan_x = h5_file[self.PROBE_POSITION_X]
            h5_scan_y = h5_file[self.PROBE_POSITION_Y]

            for idx, x_m, y_m in zip(h5_scan_indexes[()], h5_scan_x[()], h5_scan_y[()]):
                point = ScanPoint(idx, x_m, y_m)
                point_list.append(point)

            h5_probe = h5_file[self.PROBE_ARRAY]
            probe_pixel_geometry = PixelGeometry(
                width_m=float(h5_probe.attrs[self.PROBE_PIXEL_WIDTH]),
                height_m=float(h5_probe.attrs[self.PROBE_PIXEL_HEIGHT]),
            )

            try:
                opr_weights = h5_probe.attrs[self.OPR_WEIGHTS]
            except KeyError:
                logger.debug('OPR weights not found.')
                opr_weights = None

            probe = ProbeSequence(
                array=h5_probe[()],
                opr_weights=opr_weights,
                pixel_geometry=probe_pixel_geometry,
            )

            h5_object = h5_file[self.OBJECT_ARRAY]
            object_pixel_geometry = PixelGeometry(
                width_m=float(h5_object.attrs[self.OBJECT_PIXEL_WIDTH]),
                height_m=float(h5_object.attrs[self.OBJECT_PIXEL_HEIGHT]),
            )
            object_center = ObjectCenter(
                position_x_m=float(h5_object.attrs[self.OBJECT_CENTER_X]),
                position_y_m=float(h5_object.attrs[self.OBJECT_CENTER_Y]),
            )
            h5_object_layer_spacing = h5_file[self.OBJECT_LAYER_SPACING]
            object_ = Object(
                array=h5_object[()],
                pixel_geometry=object_pixel_geometry,
                center=object_center,
                layer_spacing_m=h5_object_layer_spacing[()],
            )

            try:
                h5_loss_values = h5_file[self.LOSS_VALUES]
            except KeyError:
                h5_loss_values = h5_file['costs']

            loss_values = h5_loss_values[()]

            try:
                loss_epochs = h5_file[self.LOSS_EPOCHS][()]
            except KeyError:
                loss_epochs = numpy.arange(len(loss_values))

            losses: list[LossValue] = []

            for epoch, value in zip(loss_epochs, loss_values):
                loss = LossValue(epoch, value)
                losses.append(loss)

        return Product(
            metadata=metadata,
            positions=PositionSequence(point_list),
            probes=probe,
            object_=object_,
            losses=losses,
        )

    def write(self, file_path: Path, product: Product) -> None:
        scan_indexes: list[int] = []
        scan_x_m: list[float] = []
        scan_y_m: list[float] = []

        for point in product.positions:
            scan_indexes.append(point.index)
            scan_x_m.append(point.position_x_m)
            scan_y_m.append(point.position_y_m)

        with h5py.File(file_path, 'w') as h5_file:
            metadata = product.metadata
            h5_file.attrs[self.NAME] = metadata.name
            h5_file.attrs[self.COMMENTS] = metadata.comments
            h5_file.attrs[self.DETECTOR_OBJECT_DISTANCE] = metadata.detector_distance_m
            h5_file.attrs[self.PROBE_ENERGY] = metadata.probe_energy_eV
            h5_file.attrs[self.PROBE_PHOTON_COUNT] = metadata.probe_photon_count
            h5_file.attrs[self.EXPOSURE_TIME] = metadata.exposure_time_s
            h5_file.attrs[self.MASS_ATTENUATION] = metadata.mass_attenuation_m2_kg

            h5_file.create_dataset(self.PROBE_POSITION_INDEXES, data=scan_indexes)
            h5_file.create_dataset(self.PROBE_POSITION_X, data=scan_x_m)
            h5_file.create_dataset(self.PROBE_POSITION_Y, data=scan_y_m)

            probe = product.probes
            h5_probe = h5_file.create_dataset(self.PROBE_ARRAY, data=probe.get_array())

            try:
                opr_weights = probe.get_opr_weights()
            except ValueError:
                pass
            else:
                h5_file.create_dataset(self.OPR_WEIGHTS, data=opr_weights)

            probe_pixel_geometry = probe.get_pixel_geometry()
            h5_probe.attrs[self.PROBE_PIXEL_WIDTH] = probe_pixel_geometry.width_m
            h5_probe.attrs[self.PROBE_PIXEL_HEIGHT] = probe_pixel_geometry.height_m

            object_ = product.object_
            object_geometry = object_.get_geometry()
            h5_object = h5_file.create_dataset(self.OBJECT_ARRAY, data=object_.get_array())
            h5_object.attrs[self.OBJECT_CENTER_X] = object_geometry.center_x_m
            h5_object.attrs[self.OBJECT_CENTER_Y] = object_geometry.center_y_m
            h5_object.attrs[self.OBJECT_PIXEL_WIDTH] = object_geometry.pixel_width_m
            h5_object.attrs[self.OBJECT_PIXEL_HEIGHT] = object_geometry.pixel_height_m
            h5_file.create_dataset(self.OBJECT_LAYER_SPACING, data=object_.layer_spacing_m)

            loss_epochs: list[int] = []
            loss_values: list[float] = []

            for loss in product.losses:
                loss_epochs.append(loss.epoch)
                loss_values.append(loss.value)

            h5_file.create_dataset(self.LOSS_EPOCHS, data=loss_epochs)
            h5_file.create_dataset(self.LOSS_VALUES, data=loss_values)


def register_plugins(registry: PluginRegistry) -> None:
    h5_product_file_io = H5ProductFileIO()

    registry.register_product_file_reader_with_adapters(
        h5_product_file_io,
        simple_name=H5ProductFileIO.SIMPLE_NAME,
        display_name=H5ProductFileIO.DISPLAY_NAME,
    )
    registry.product_file_writers.register_plugin(
        h5_product_file_io,
        simple_name=H5ProductFileIO.SIMPLE_NAME,
        display_name=H5ProductFileIO.DISPLAY_NAME,
    )
