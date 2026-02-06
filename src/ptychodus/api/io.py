from __future__ import annotations
from enum import StrEnum
from pathlib import Path
import logging

import h5py
import numpy

from .common import BYTES_PER_MEGABYTE
from .diffraction import (
    BadPixels,
    DiffractionIndexes,
    DiffractionPattern,
    DiffractionPatternCounts,
    DiffractionPatternDType,
    DiffractionPatterns,
)
from .geometry import PixelGeometry
from .object import Object, ObjectCenter
from .probe import ProbeSequence
from .probe_positions import ProbePositionSequence, ProbePosition
from .product import Product, ProductMetadata
from .reconstructor import LossValue

__all__ = [
    'AssembledDiffractionData',
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


class AssembledDiffractionData:
    def __init__(
        self, indexes: DiffractionIndexes, patterns: DiffractionPatterns, bad_pixels: BadPixels
    ) -> None:
        self._indexes = indexes
        self._patterns = patterns
        self._bad_pixels = bad_pixels

        if indexes.ndim != 1:
            raise ValueError(
                f'Unexpected number of dimensions for indexes! (actual={indexes.ndim} expected=1)'
            )

        if patterns.ndim != 3:
            raise ValueError(
                f'Unexpected number of dimensions for patterns! (actual={patterns.ndim} expected=3)'
            )

        if bad_pixels.ndim != 2:
            raise ValueError(
                f'Unexpected number of dimensions for bad pixels! (actual={bad_pixels.ndim} expected=2)'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Number of indexes does not match number of patterns!')

        if patterns.shape[1:] != bad_pixels.shape:
            raise ValueError('Patterns shape does not match bad pixels shape!')

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=int),
            patterns=numpy.zeros((1, 1, 1), dtype=int),
            bad_pixels=numpy.zeros((1, 1), dtype=bool),
        )

    def get_patterns_shape(self) -> tuple[int, int, int]:
        return self._patterns.shape

    def get_patterns_dtype(self) -> DiffractionPatternDType:
        return self._patterns.dtype

    def get_pattern(self, index: int) -> DiffractionPattern:
        return self._patterns[index]

    def get_bad_pixels(self) -> BadPixels:
        return self._bad_pixels

    def assemble(self, data: AssembledDiffractionData, offset: int) -> AssembledDiffractionData:
        assembled_indexes = slice(offset, offset + len(data._indexes))

        self._indexes[assembled_indexes] = data._indexes
        indexes_view = self._indexes[assembled_indexes]
        indexes_view.flags.writeable = False

        self._patterns[assembled_indexes, :, :] = data._patterns
        patterns_view = self._patterns[assembled_indexes, :, :]
        patterns_view.flags.writeable = False

        return AssembledDiffractionData(
            indexes=indexes_view,
            patterns=patterns_view,
            bad_pixels=data._bad_pixels,
        )

    def get_assembled_indexes(self) -> DiffractionIndexes:
        return self._indexes[self._indexes >= 0]

    def get_assembled_patterns(self) -> DiffractionPatterns:
        return self._patterns[self._indexes >= 0]

    def get_assembled_pattern_counts(self) -> DiffractionPatternCounts:
        good_pixels = numpy.logical_not(self._bad_pixels)
        assembled_patterns = self.get_assembled_patterns()
        pattern_counts = numpy.sum(assembled_patterns[:, good_pixels], axis=-1)
        return pattern_counts

    def get_average_pattern(self) -> DiffractionPattern:
        assembled_patterns = self.get_assembled_patterns()
        return numpy.mean(assembled_patterns, axis=0)

    def __str__(self) -> str:
        number, height, width = self._patterns.shape
        dtype = str(self._patterns.dtype)
        size_MB = self._patterns.nbytes / BYTES_PER_MEGABYTE  # noqa: N806
        return f'{number} x {height}H x {width}W {dtype} [{size_MB:.2f}MB]'


class DiffractionFileKeys(StrEnum):
    PATTERNS = 'patterns'
    INDEXES = 'indexes'
    BAD_PIXELS = 'bad_pixels'


def load_diffraction_data(file: Path, *, mmap_file: Path | None = None) -> AssembledDiffractionData:
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
            h5_patterns[()],  # FIXME support memmap
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

        # FIXME probe pixel height/width from object

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

        h5_scan_indexes = h5_file[ProductFileKeys.PROBE_POSITION_INDEXES]
        h5_scan_x = h5_file[ProductFileKeys.PROBE_POSITION_X]
        h5_scan_y = h5_file[ProductFileKeys.PROBE_POSITION_Y]

        for idx, x_m, y_m in zip(h5_scan_indexes[()], h5_scan_x[()], h5_scan_y[()]):
            point = ProbePosition(idx, x_m, y_m)
            point_list.append(point)

        h5_probe = h5_file[ProductFileKeys.PROBE_ARRAY]
        probe_pixel_geometry = PixelGeometry(
            width_m=float(h5_probe.attrs[ProductFileKeys.PROBE_PIXEL_WIDTH]),
            height_m=float(h5_probe.attrs[ProductFileKeys.PROBE_PIXEL_HEIGHT]),
        )

        try:
            opr_weights = h5_probe.attrs[ProductFileKeys.OPR_WEIGHTS]
        except KeyError:
            logger.debug('OPR weights not found.')
            opr_weights = None

        probe = ProbeSequence(
            array=h5_probe[()],
            opr_weights=opr_weights,
            pixel_geometry=probe_pixel_geometry,
        )

        h5_object = h5_file[ProductFileKeys.OBJECT_ARRAY]
        object_pixel_geometry = PixelGeometry(
            width_m=float(h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_WIDTH]),
            height_m=float(h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_HEIGHT]),
        )
        object_center = ObjectCenter(
            coordinate_x_m=float(h5_object.attrs[ProductFileKeys.OBJECT_CENTER_X]),
            coordinate_y_m=float(h5_object.attrs[ProductFileKeys.OBJECT_CENTER_Y]),
        )

        try:
            layer_spacing_m = h5_file[ProductFileKeys.OBJECT_LAYER_SPACING][()]
        except KeyError:
            layer_spacing_m = []

        object_ = Object(
            array=h5_object[()],
            pixel_geometry=object_pixel_geometry,
            center=object_center,
            layer_spacing_m=layer_spacing_m,
        )

        losses: list[LossValue] = []

        try:
            loss_epochs = h5_file[ProductFileKeys.LOSS_EPOCHS][()]
            loss_values = h5_file[ProductFileKeys.LOSS_VALUES][()]
        except KeyError:
            logger.debug('Losses not found!')
        else:
            for epoch, value in zip(loss_epochs, loss_values):
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
