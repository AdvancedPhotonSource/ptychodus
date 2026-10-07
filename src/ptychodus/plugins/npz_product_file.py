"""Product reader/writer for the ptychodus NumPy zipped archive (``*.npz``).

The archive is a flat namespace of arrays, so every value the hierarchical HDF5 product of
:mod:`ptychodus.api.io` stores as an attribute on a dataset is stored here under a prefixed
top-level key instead -- ``probe_pixel_width_m`` rather than a ``pixel_width_m`` attribute on
the probe. Those three prefixed keys aside, the spelling matches
:class:`~ptychodus.api.io.ProductFileKeys` exactly, so the two formats read as one vocabulary.

The two readers also agree on which keys are optional: a product written by an older version,
or by something else entirely, loads with the same defaults either way rather than raising in
one format and succeeding in the other.
"""

from pathlib import Path
from typing import Any, Final
import logging

import numpy
from numpy.lib.npyio import NpzFile

from ptychodus.api.diffraction import Polarization
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
from ptychodus.api.reconstruct import LossValue
from ptychodus.api.probe_positions import ProbePositionSequence, ProbePosition

logger = logging.getLogger(__name__)


def _read_array(npz_file: NpzFile, key: str) -> Any | None:
    """Return the array stored under *key*, or None when the archive omits it."""
    try:
        return npz_file[key]
    except KeyError:
        logger.debug('%s not found.', key)
        return None


def _read_real(npz_file: NpzFile, key: str, default: float) -> float:
    """Return the scalar stored under *key*, or *default* when the archive omits it."""
    value = _read_array(npz_file, key)
    return default if value is None else float(value)


def _read_boolean(npz_file: NpzFile, key: str, default: bool) -> bool:
    """Return the flag stored under *key*, or *default* when the archive omits it."""
    value = _read_array(npz_file, key)
    return default if value is None else bool(value)


def _read_text(npz_file: NpzFile, key: str, default: str) -> str:
    """Return the string stored under *key*, or *default* when the archive omits it."""
    value = _read_array(npz_file, key)
    return default if value is None else str(value)


class NPZProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'NPZ'
    DISPLAY_NAME: Final[str] = 'Ptychodus NumPy Zipped Archive (*.npz)'

    NAME: Final[str] = 'name'
    COMMENTS: Final[str] = 'comments'
    DETECTOR_OBJECT_DISTANCE: Final[str] = 'detector_object_distance_m'
    FOCUS_OBJECT_DISTANCE: Final[str] = 'focus_object_distance_m'
    PHOTON_ENERGY: Final[str] = 'photon_energy_eV'
    PROBE_PHOTON_COUNT: Final[str] = 'probe_photon_count'
    EXPOSURE_TIME: Final[str] = 'exposure_time_s'
    MASS_ATTENUATION: Final[str] = 'mass_attenuation_m2_per_kg'
    TOMOGRAPHY_ANGLE: Final[str] = 'tomography_angle_deg'
    TILT_ANGLE: Final[str] = 'tilt_angle_deg'
    POLARIZATION: Final[str] = 'polarization'
    FAR_FIELD: Final[str] = 'far_field'

    PROBE_ARRAY: Final[str] = 'probe'
    OPR_WEIGHTS: Final[str] = 'opr_weights'
    PROBE_PIXEL_HEIGHT: Final[str] = 'probe_pixel_height_m'
    PROBE_PIXEL_WIDTH: Final[str] = 'probe_pixel_width_m'
    PROBE_POSITION_INDEXES: Final[str] = 'probe_position_indexes'
    PROBE_POSITION_X: Final[str] = 'probe_position_x_m'
    PROBE_POSITION_Y: Final[str] = 'probe_position_y_m'
    # Per position, and plural, as in ProductFileKeys; PROBE_PHOTON_COUNT above is the
    # single metadata value for the whole product.
    PROBE_PHOTON_COUNTS: Final[str] = 'probe_photon_counts'

    OBJECT_ARRAY: Final[str] = 'object'
    OBJECT_CENTER_X: Final[str] = 'object_center_x_m'
    OBJECT_CENTER_Y: Final[str] = 'object_center_y_m'
    OBJECT_LAYER_SPACING: Final[str] = 'object_layer_spacing_m'
    OBJECT_PIXEL_HEIGHT: Final[str] = 'object_pixel_height_m'
    OBJECT_PIXEL_WIDTH: Final[str] = 'object_pixel_width_m'

    LOSS_EPOCHS: Final[str] = 'loss_epochs'
    LOSS_VALUES: Final[str] = 'loss_values'

    def _read_polarization(self, npz_file: NpzFile, file_path: Path) -> Polarization | None:
        raw = _read_array(npz_file, self.POLARIZATION)

        if raw is None:
            return None

        try:
            return Polarization(str(raw))
        except ValueError:
            logger.warning(
                'Unknown polarization %r in %s; setting polarization=None.', str(raw), file_path
            )
            return None

    def read(self, file_path: Path) -> Product:
        with numpy.load(file_path) as npz_file:
            metadata = ProductMetadata(
                name=_read_text(npz_file, self.NAME, 'Unnamed'),
                comments=_read_text(npz_file, self.COMMENTS, ''),
                detector_distance_m=float(npz_file[self.DETECTOR_OBJECT_DISTANCE]),
                photon_energy_eV=float(npz_file[self.PHOTON_ENERGY]),
                probe_photon_count=_read_real(npz_file, self.PROBE_PHOTON_COUNT, 0.0),
                exposure_time_s=_read_real(npz_file, self.EXPOSURE_TIME, 0.0),
                mass_attenuation_m2_per_kg=_read_real(npz_file, self.MASS_ATTENUATION, 0.0),
                tomography_angle_deg=_read_real(npz_file, self.TOMOGRAPHY_ANGLE, 0.0),
                focus_object_distance_m=_read_real(npz_file, self.FOCUS_OBJECT_DISTANCE, 0.0),
                tilt_angle_deg=_read_real(npz_file, self.TILT_ANGLE, 0.0),
                polarization=self._read_polarization(npz_file, file_path),
                far_field=_read_boolean(npz_file, self.FAR_FIELD, True),
            )

            scan_indexes = npz_file[self.PROBE_POSITION_INDEXES]
            scan_x_m = npz_file[self.PROBE_POSITION_X]
            scan_y_m = npz_file[self.PROBE_POSITION_Y]
            # The all-or-nothing invariant on ProbePositionSequence means write() only
            # emits this when every position had a count, so a present array aligns 1:1
            # with the indexes and needs no mask.
            position_photon_counts = _read_array(npz_file, self.PROBE_PHOTON_COUNTS)

            object_pixel_geometry = PixelGeometry(
                width_m=float(npz_file[self.OBJECT_PIXEL_WIDTH]),
                height_m=float(npz_file[self.OBJECT_PIXEL_HEIGHT]),
            )
            object_center = ObjectCenter(
                x_m=float(npz_file[self.OBJECT_CENTER_X]),
                y_m=float(npz_file[self.OBJECT_CENTER_Y]),
            )
            layer_spacing_m = _read_array(npz_file, self.OBJECT_LAYER_SPACING)
            object_ = Object(
                array=npz_file[self.OBJECT_ARRAY],
                pixel_geometry=object_pixel_geometry,
                center=object_center,
                layer_spacing_m=[] if layer_spacing_m is None else layer_spacing_m,
            )

            # A probe written without its own pixel size shares the object's, which is the
            # sampling the reconstruction actually ran at.
            probe_pixel_geometry = PixelGeometry(
                width_m=_read_real(npz_file, self.PROBE_PIXEL_WIDTH, object_pixel_geometry.width_m),
                height_m=_read_real(
                    npz_file, self.PROBE_PIXEL_HEIGHT, object_pixel_geometry.height_m
                ),
            )
            probe = ProbeSequence(
                array=npz_file[self.PROBE_ARRAY],
                opr_weights=_read_array(npz_file, self.OPR_WEIGHTS),
                pixel_geometry=probe_pixel_geometry,
            )

            loss_values = _read_array(npz_file, self.LOSS_VALUES)

            if loss_values is None:
                # Archives written before the key was renamed.
                loss_values = _read_array(npz_file, 'costs')

            if loss_values is None:
                loss_values = numpy.empty(0)

            loss_epochs = _read_array(npz_file, self.LOSS_EPOCHS)

            if loss_epochs is None:
                loss_epochs = numpy.arange(len(loss_values))

        point_list: list[ProbePosition] = []

        for offset, (idx, x_m, y_m) in enumerate(zip(scan_indexes, scan_x_m, scan_y_m)):
            photon_count = (
                None if position_photon_counts is None else float(position_photon_counts[offset])
            )
            point = ProbePosition(idx, x_m, y_m, probe_photon_count=photon_count)
            point_list.append(point)

        losses: list[LossValue] = []

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

    def write(self, file_path: Path, product: Product) -> None:
        contents: dict[str, Any] = dict()
        scan_indexes = product.probe_positions.get_indexes()
        scan_x_m = product.probe_positions.get_coordinates_x_m()
        scan_y_m = product.probe_positions.get_coordinates_y_m()
        position_photon_counts = product.probe_positions.get_probe_photon_counts()

        metadata = product.metadata
        contents[self.NAME] = metadata.name
        contents[self.COMMENTS] = metadata.comments
        contents[self.DETECTOR_OBJECT_DISTANCE] = metadata.detector_distance_m
        contents[self.PHOTON_ENERGY] = metadata.photon_energy_eV
        contents[self.PROBE_PHOTON_COUNT] = metadata.probe_photon_count
        contents[self.EXPOSURE_TIME] = metadata.exposure_time_s
        contents[self.MASS_ATTENUATION] = metadata.mass_attenuation_m2_per_kg
        contents[self.TOMOGRAPHY_ANGLE] = metadata.tomography_angle_deg
        contents[self.FOCUS_OBJECT_DISTANCE] = metadata.focus_object_distance_m
        contents[self.TILT_ANGLE] = metadata.tilt_angle_deg
        contents[self.FAR_FIELD] = metadata.far_field

        # savez cannot store None, so an absent key is how both formats spell "unpolarized".
        if metadata.polarization is not None:
            contents[self.POLARIZATION] = metadata.polarization.value

        contents[self.PROBE_POSITION_INDEXES] = scan_indexes
        contents[self.PROBE_POSITION_X] = scan_x_m
        contents[self.PROBE_POSITION_Y] = scan_y_m

        if position_photon_counts is not None:
            contents[self.PROBE_PHOTON_COUNTS] = position_photon_counts

        probe = product.probes
        contents[self.PROBE_ARRAY] = probe.get_array()

        opr_weights = probe.get_opr_weights_or_none()

        if opr_weights is not None:
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
