"""I/O utilities for assembled diffraction data and data products."""

from __future__ import annotations
from enum import StrEnum
from pathlib import Path
import logging

import h5py

from .geometry import PixelGeometry
from .object import Object, ObjectCenter
from .probe import ProbeSequence
from .probe_positions import ProbePositionSequence, ProbePosition
from .product import Product, ProductMetadata
from .reconstructor import AssembledDiffractionData, LossValue

__all__ = [
    'StandardFileLayout',
    'load_diffraction_data',
    'load_product',
    'save_diffraction_data',
    'save_product',
]

logger = logging.getLogger(__name__)


class StandardFileLayout(StrEnum):
    DIFFRACTION = 'diffraction.h5'
    FLUORESCENCE_IN = 'fluorescence-in.h5'
    FLUORESCENCE_OUT = 'fluorescence-out.h5'
    PRODUCT_IN = 'product-in.h5'
    PRODUCT_OUT = 'product-out.h5'
    SETTINGS = 'settings.ini'


class DiffractionFileKeys(StrEnum):
    PATTERNS = 'patterns'
    INDEXES = 'indexes'
    BAD_PIXELS = 'bad_pixels'


def load_diffraction_data(file: Path, *, mmap_file: Path | None = None) -> AssembledDiffractionData:
    if mmap_file is not None:
        raise NotImplementedError('Load to memory map not implemented yet!')

    with h5py.File(file, 'r') as h5_file:
        h5_indexes = h5_file[DiffractionFileKeys.INDEXES]

        if not isinstance(h5_indexes, h5py.Dataset):
            raise ValueError('Indexes are not a dataset!')

        h5_patterns = h5_file[DiffractionFileKeys.PATTERNS]

        if not isinstance(h5_patterns, h5py.Dataset):
            raise ValueError('Patterns are not a dataset!')

        h5_bad_pixels = h5_file[DiffractionFileKeys.BAD_PIXELS]

        if not isinstance(h5_bad_pixels, h5py.Dataset):
            raise ValueError('Bad pixels are not a dataset!')

        return AssembledDiffractionData(
            h5_indexes[()],
            h5_patterns[()],
            h5_bad_pixels[()],
        )


def save_diffraction_data(
    file: Path, data: AssembledDiffractionData, *, compression: str = 'lzf'
) -> None:
    with h5py.File(file, 'w') as h5_file:
        h5_file.create_dataset(
            DiffractionFileKeys.INDEXES, data=data._indexes, compression=compression
        )
        h5_file.create_dataset(
            DiffractionFileKeys.PATTERNS, data=data._patterns, compression=compression
        )
        h5_file.create_dataset(
            DiffractionFileKeys.BAD_PIXELS, data=data._bad_pixels, compression=compression
        )


class ProductFileKeys(StrEnum):
    NAME = 'name'
    COMMENTS = 'comments'
    DETECTOR_OBJECT_DISTANCE = 'detector_object_distance_m'
    PROBE_ENERGY = 'probe_energy_eV'
    PROBE_PHOTON_COUNT = 'probe_photon_count'
    EXPOSURE_TIME = 'exposure_time_s'
    MASS_ATTENUATION = 'mass_attenuation_m2_kg'
    TOMOGRAPHY_ANGLE = 'tomography_angle_deg'
    PROBE_ARRAY = 'probe'
    OPR_WEIGHTS = 'opr_weights'
    PROBE_PIXEL_HEIGHT = 'pixel_height_m'
    PROBE_PIXEL_WIDTH = 'pixel_width_m'
    PROBE_POSITION_INDEXES = 'probe_position_indexes'
    PROBE_POSITION_X = 'probe_position_x_m'
    PROBE_POSITION_Y = 'probe_position_y_m'
    OBJECT_ARRAY = 'object'
    OBJECT_CENTER_X = 'center_x_m'
    OBJECT_CENTER_Y = 'center_y_m'
    OBJECT_LAYER_SPACING = 'object_layer_spacing_m'
    OBJECT_PIXEL_HEIGHT = 'pixel_height_m'
    OBJECT_PIXEL_WIDTH = 'pixel_width_m'
    LOSS_EPOCHS = 'loss_epochs'
    LOSS_VALUES = 'loss_values'


def load_product(file: Path) -> Product:
    point_list: list[ProbePosition] = []

    with h5py.File(file, 'r') as h5_file:
        name = str(h5_file.attrs.get(ProductFileKeys.NAME, 'Unnamed'))
        comments = str(h5_file.attrs.get(ProductFileKeys.COMMENTS, ''))
        probe_photon_count = int(h5_file.attrs.get(ProductFileKeys.PROBE_PHOTON_COUNT, 0))
        exposure_time_s = float(h5_file.attrs.get(ProductFileKeys.EXPOSURE_TIME, 0.0))
        mass_attenuation_m2_kg = float(h5_file.attrs.get(ProductFileKeys.MASS_ATTENUATION, 0.0))
        tomography_angle_deg = float(h5_file.attrs.get(ProductFileKeys.TOMOGRAPHY_ANGLE, 0.0))

        metadata = ProductMetadata(
            name=name,
            comments=comments,
            detector_distance_m=float(h5_file.attrs[ProductFileKeys.DETECTOR_OBJECT_DISTANCE]),
            probe_energy_eV=float(h5_file.attrs[ProductFileKeys.PROBE_ENERGY]),
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
            tomography_angle_deg=tomography_angle_deg,
        )

        h5_object = h5_file[ProductFileKeys.OBJECT_ARRAY]

        if not isinstance(h5_object, h5py.Dataset):
            raise ValueError('Object array is not a dataset!')

        object_pixel_geometry = PixelGeometry(
            width_m=float(h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_WIDTH]),
            height_m=float(h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_HEIGHT]),
        )
        object_center = ObjectCenter(
            coordinate_x_m=float(h5_object.attrs[ProductFileKeys.OBJECT_CENTER_X]),
            coordinate_y_m=float(h5_object.attrs[ProductFileKeys.OBJECT_CENTER_Y]),
        )

        try:
            h5_object_layer_spacing = h5_file[ProductFileKeys.OBJECT_LAYER_SPACING]
        except KeyError:
            logger.debug('Object layer spacing not found.')
            layer_spacing_m = []
        else:
            if isinstance(h5_object_layer_spacing, h5py.Dataset):
                layer_spacing_m = h5_object_layer_spacing[()]
            else:
                raise ValueError('Object layer spacing is not a dataset!')

        object_ = Object(
            array=h5_object[()],
            pixel_geometry=object_pixel_geometry,
            center=object_center,
            layer_spacing_m=layer_spacing_m,
        )

        h5_probe = h5_file[ProductFileKeys.PROBE_ARRAY]

        if not isinstance(h5_probe, h5py.Dataset):
            raise ValueError('Probe array is not a dataset!')

        probe_pixel_geometry = PixelGeometry(
            width_m=float(
                h5_probe.attrs.get(ProductFileKeys.PROBE_PIXEL_WIDTH, object_pixel_geometry.width_m)
            ),
            height_m=float(
                h5_probe.attrs.get(
                    ProductFileKeys.PROBE_PIXEL_HEIGHT, object_pixel_geometry.height_m
                )
            ),
        )

        try:
            h5_opr_weights = h5_file[ProductFileKeys.OPR_WEIGHTS]
        except KeyError:
            logger.debug('OPR weights not found.')
            opr_weights = None
        else:
            if isinstance(h5_opr_weights, h5py.Dataset):
                opr_weights = h5_opr_weights[()]
            else:
                raise ValueError('OPR weights is not a dataset!')

        probe = ProbeSequence(
            array=h5_probe[()],
            opr_weights=opr_weights,
            pixel_geometry=probe_pixel_geometry,
        )

        h5_position_indexes = h5_file[ProductFileKeys.PROBE_POSITION_INDEXES]

        if not isinstance(h5_position_indexes, h5py.Dataset):
            raise ValueError('Probe position indexes is not a dataset!')

        h5_position_x = h5_file[ProductFileKeys.PROBE_POSITION_X]

        if not isinstance(h5_position_x, h5py.Dataset):
            raise ValueError('Probe position X is not a dataset!')

        h5_position_y = h5_file[ProductFileKeys.PROBE_POSITION_Y]

        if not isinstance(h5_position_y, h5py.Dataset):
            raise ValueError('Probe position Y is not a dataset!')

        for idx, x_m, y_m in zip(h5_position_indexes[()], h5_position_x[()], h5_position_y[()]):
            point = ProbePosition(idx, x_m, y_m)
            point_list.append(point)

        losses: list[LossValue] = []

        try:
            h5_loss_epochs = h5_file[ProductFileKeys.LOSS_EPOCHS]
            h5_loss_values = h5_file[ProductFileKeys.LOSS_VALUES]
        except KeyError:
            logger.debug('Losses not found!')
        else:
            if not isinstance(h5_loss_epochs, h5py.Dataset):
                raise ValueError('Loss epochs are not a dataset!')

            if not isinstance(h5_loss_values, h5py.Dataset):
                raise ValueError('Loss values are not a dataset!')

            for epoch, value in zip(h5_loss_epochs[()], h5_loss_values[()]):
                loss = LossValue(epoch, value)
                losses.append(loss)

    return Product(
        metadata=metadata,
        probe_positions=ProbePositionSequence(point_list),
        probes=probe,
        object_=object_,
        losses=losses,
    )


def save_product(file: Path, product: Product) -> None:
    scan_indexes: list[int] = []
    scan_x_m: list[float] = []
    scan_y_m: list[float] = []

    for point in product.probe_positions:
        scan_indexes.append(point.index)
        scan_x_m.append(point.coordinate_x_m)
        scan_y_m.append(point.coordinate_y_m)

    with h5py.File(file, 'w') as h5_file:
        metadata = product.metadata
        h5_file.attrs[ProductFileKeys.NAME] = metadata.name
        h5_file.attrs[ProductFileKeys.COMMENTS] = metadata.comments
        h5_file.attrs[ProductFileKeys.DETECTOR_OBJECT_DISTANCE] = metadata.detector_distance_m
        h5_file.attrs[ProductFileKeys.PROBE_ENERGY] = metadata.probe_energy_eV
        h5_file.attrs[ProductFileKeys.PROBE_PHOTON_COUNT] = metadata.probe_photon_count
        h5_file.attrs[ProductFileKeys.EXPOSURE_TIME] = metadata.exposure_time_s
        h5_file.attrs[ProductFileKeys.MASS_ATTENUATION] = metadata.mass_attenuation_m2_kg

        h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_INDEXES, data=scan_indexes)
        h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_X, data=scan_x_m)
        h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_Y, data=scan_y_m)

        probe = product.probes
        h5_probe = h5_file.create_dataset(ProductFileKeys.PROBE_ARRAY, data=probe.get_array())

        try:
            opr_weights = probe.get_opr_weights()
        except ValueError:
            pass
        else:
            h5_file.create_dataset(ProductFileKeys.OPR_WEIGHTS, data=opr_weights)

        probe_pixel_geometry = probe.get_pixel_geometry()
        h5_probe.attrs[ProductFileKeys.PROBE_PIXEL_WIDTH] = probe_pixel_geometry.width_m
        h5_probe.attrs[ProductFileKeys.PROBE_PIXEL_HEIGHT] = probe_pixel_geometry.height_m

        object_ = product.object_
        object_geometry = object_.get_geometry()
        h5_object = h5_file.create_dataset(ProductFileKeys.OBJECT_ARRAY, data=object_.get_array())
        h5_object.attrs[ProductFileKeys.OBJECT_CENTER_X] = object_geometry.center_x_m
        h5_object.attrs[ProductFileKeys.OBJECT_CENTER_Y] = object_geometry.center_y_m
        h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_WIDTH] = object_geometry.pixel_width_m
        h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_HEIGHT] = object_geometry.pixel_height_m
        h5_file.create_dataset(ProductFileKeys.OBJECT_LAYER_SPACING, data=object_.layer_spacing_m)

        loss_epochs: list[int] = []
        loss_values: list[float] = []

        for loss in product.losses:
            loss_epochs.append(loss.epoch)
            loss_values.append(loss.value)

        h5_file.create_dataset(ProductFileKeys.LOSS_EPOCHS, data=loss_epochs)
        h5_file.create_dataset(ProductFileKeys.LOSS_VALUES, data=loss_values)
