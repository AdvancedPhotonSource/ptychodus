from pathlib import Path
from typing import Any, Final
import logging

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


class NPZProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'NPZ'
    DISPLAY_NAME: Final[str] = 'Ptychodus NumPy Zipped Archive (*.npz)'

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
    PROBE_PIXEL_HEIGHT: Final[str] = 'probe_pixel_height_m'
    PROBE_PIXEL_WIDTH: Final[str] = 'probe_pixel_width_m'
    PROBE_POSITION_INDEXES: Final[str] = 'probe_position_indexes'
    PROBE_POSITION_X: Final[str] = 'probe_position_x_m'
    PROBE_POSITION_Y: Final[str] = 'probe_position_y_m'

    OBJECT_ARRAY: Final[str] = 'object'
    OBJECT_CENTER_X: Final[str] = 'object_center_x_m'
    OBJECT_CENTER_Y: Final[str] = 'object_center_y_m'
    OBJECT_LAYER_SPACING: Final[str] = 'object_layer_spacing_m'
    OBJECT_PIXEL_HEIGHT: Final[str] = 'object_pixel_height_m'
    OBJECT_PIXEL_WIDTH: Final[str] = 'object_pixel_width_m'

    LOSS_EPOCHS: Final[str] = 'loss_epochs'
    LOSS_VALUES: Final[str] = 'loss_values'

    def read(self, file_path: Path) -> Product:
        with numpy.load(file_path) as npz_file:
            probe_photon_count = 0.0

            try:
                probe_photon_count = float(npz_file[self.PROBE_PHOTON_COUNT])
            except KeyError:
                logger.debug('Probe photon count not found.')

            mass_attenuation_m2_kg = 0.0

            try:
                mass_attenuation_m2_kg = float(npz_file[self.MASS_ATTENUATION])
            except KeyError:
                logger.debug('Mass attenuation not found.')

            tomography_angle_deg = 0.0

            try:
                tomography_angle_deg = float(npz_file[self.TOMOGRAPHY_ANGLE])
            except KeyError:
                logger.debug('Tomography angle not found.')

            metadata = ProductMetadata(
                name=str(npz_file[self.NAME]),
                comments=str(npz_file[self.COMMENTS]),
                detector_distance_m=float(npz_file[self.DETECTOR_OBJECT_DISTANCE]),
                probe_energy_eV=float(npz_file[self.PROBE_ENERGY]),
                probe_photon_count=probe_photon_count,
                exposure_time_s=float(npz_file[self.EXPOSURE_TIME]),
                mass_attenuation_m2_kg=mass_attenuation_m2_kg,
                tomography_angle_deg=tomography_angle_deg,
            )

            scan_indexes = npz_file[self.PROBE_POSITION_INDEXES]
            scan_x_m = npz_file[self.PROBE_POSITION_X]
            scan_y_m = npz_file[self.PROBE_POSITION_Y]

            probe_pixel_geometry = PixelGeometry(
                width_m=float(npz_file[self.PROBE_PIXEL_WIDTH]),
                height_m=float(npz_file[self.PROBE_PIXEL_HEIGHT]),
            )

            try:
                opr_weights = npz_file[self.OPR_WEIGHTS]
            except KeyError:
                logger.debug('OPR weights not found.')
                opr_weights = None

            probe = ProbeSequence(
                array=npz_file[self.PROBE_ARRAY],
                opr_weights=opr_weights,
                pixel_geometry=probe_pixel_geometry,
            )

            object_pixel_geometry = PixelGeometry(
                width_m=float(npz_file[self.OBJECT_PIXEL_WIDTH]),
                height_m=float(npz_file[self.OBJECT_PIXEL_HEIGHT]),
            )
            object_center = ObjectCenter(
                position_x_m=float(npz_file[self.OBJECT_CENTER_X]),
                position_y_m=float(npz_file[self.OBJECT_CENTER_Y]),
            )
            object_ = Object(
                array=npz_file[self.OBJECT_ARRAY],
                pixel_geometry=object_pixel_geometry,
                center=object_center,
                layer_spacing_m=npz_file[self.OBJECT_LAYER_SPACING],
            )

            try:
                loss_values = npz_file[self.LOSS_VALUES]
            except KeyError:
                loss_values = npz_file['costs']

            try:
                loss_epochs = npz_file[self.LOSS_EPOCHS]
            except KeyError:
                loss_epochs = numpy.arange(len(loss_values))

        point_list: list[ScanPoint] = []

        for idx, x_m, y_m in zip(scan_indexes, scan_x_m, scan_y_m):
            point = ScanPoint(idx, x_m, y_m)
            point_list.append(point)

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
        contents: dict[str, Any] = dict()
        scan_indexes: list[int] = []
        scan_x_m: list[float] = []
        scan_y_m: list[float] = []

        for point in product.positions:
            scan_indexes.append(point.index)
            scan_x_m.append(point.position_x_m)
            scan_y_m.append(point.position_y_m)

        metadata = product.metadata
        contents[self.NAME] = metadata.name
        contents[self.COMMENTS] = metadata.comments
        contents[self.DETECTOR_OBJECT_DISTANCE] = metadata.detector_distance_m
        contents[self.PROBE_ENERGY] = metadata.probe_energy_eV
        contents[self.PROBE_PHOTON_COUNT] = metadata.probe_photon_count
        contents[self.EXPOSURE_TIME] = metadata.exposure_time_s
        contents[self.MASS_ATTENUATION] = metadata.mass_attenuation_m2_kg

        contents[self.PROBE_POSITION_INDEXES] = scan_indexes
        contents[self.PROBE_POSITION_X] = scan_x_m
        contents[self.PROBE_POSITION_Y] = scan_y_m

        probe = product.probes
        contents[self.PROBE_ARRAY] = probe.get_array()

        try:
            opr_weights = probe.get_opr_weights()
        except ValueError:
            pass
        else:
            contents[self.OPR_WEIGHTS] = opr_weights

        probe_pixel_geometry = probe.get_pixel_geometry()
        contents[self.PROBE_PIXEL_WIDTH] = probe_pixel_geometry.width_m
        contents[self.PROBE_PIXEL_HEIGHT] = probe_pixel_geometry.height_m

        object_ = product.object_
        object_geometry = object_.get_geometry()
        contents[self.OBJECT_ARRAY] = object_.get_array()
        contents[self.OBJECT_CENTER_X] = object_geometry.center_x_m
        contents[self.OBJECT_CENTER_Y] = object_geometry.center_y_m
        contents[self.OBJECT_PIXEL_WIDTH] = object_geometry.pixel_width_m
        contents[self.OBJECT_PIXEL_HEIGHT] = object_geometry.pixel_height_m
        contents[self.OBJECT_LAYER_SPACING] = object_.layer_spacing_m

        loss_epochs: list[int] = []
        loss_values: list[float] = []

        for loss in product.losses:
            loss_epochs.append(loss.epoch)
            loss_values.append(loss.value)

        contents[self.LOSS_EPOCHS] = loss_epochs
        contents[self.LOSS_VALUES] = loss_values

        numpy.savez(file_path, **contents)


def register_plugins(registry: PluginRegistry) -> None:
    npz_product_file_io = NPZProductFileIO()

    registry.register_product_file_reader_with_adapters(
        npz_product_file_io,
        simple_name=NPZProductFileIO.SIMPLE_NAME,
        display_name=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.product_file_writers.register_plugin(
        npz_product_file_io,
        simple_name=NPZProductFileIO.SIMPLE_NAME,
        display_name=NPZProductFileIO.DISPLAY_NAME,
    )
