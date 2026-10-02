"""Reader pair for the ptychodus HDF5 training data format (``*.h5``).

The file is a flat namespace of datasets written by
:func:`ptychodus.api.io.save_training_data`: diffraction patterns with the
bad-pixel mask saying which of their values were filled rather than measured, the
probe mode stack, the object, and the probe positions in object pixels. Two
readers split it the way ptychodus consumes it -- one yields the diffraction
dataset, the other the product.

Three quantities a product carries are absent from the file and are recovered
rather than defaulted. The detector pixel size follows from the far-field
relation, which is its own inverse, so the recorded object pixel size, energy and
detector distance give it back exactly. The object center is the origin, because
the positions were written as offsets about the object array's own center. Scan
indexes are consecutive, and both readers must agree on that, since patterns and
positions are paired by index rather than by array order.
"""

from pathlib import Path
from typing import Final

import h5py
import numpy

from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import TrainingDataFileKeys
from ptychodus.api.object import Object, ObjectGeometry, ObjectPosition
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.propagate import compute_far_field_pixel_geometry


def _read_array(h5_file: h5py.File, key: str) -> numpy.ndarray:
    item = h5_file[key]

    if not isinstance(item, h5py.Dataset):
        raise ValueError(f'"{key}" is not a dataset!')

    return item[()]


def _read_scalar(h5_file: h5py.File, key: str) -> float:
    return float(_read_array(h5_file, key))


def _read_object_geometry(h5_file: h5py.File, object_array: numpy.ndarray) -> ObjectGeometry:
    return ObjectGeometry(
        width_px=object_array.shape[-1],
        height_px=object_array.shape[-2],
        pixel_width_m=_read_scalar(h5_file, TrainingDataFileKeys.OBJECT_PIXEL_WIDTH),
        pixel_height_m=_read_scalar(h5_file, TrainingDataFileKeys.OBJECT_PIXEL_HEIGHT),
        # The positions are offsets about the object array's own center, so that
        # center is the origin of the frame they were written in rather than a
        # placeholder for one the file failed to record.
        center_x_m=0.0,
        center_y_m=0.0,
    )


class TrainingDataDiffractionFileReader(DiffractionFileReader):
    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            patterns = _read_array(h5_file, TrainingDataFileKeys.PATTERNS)
            bad_pixels = _read_array(h5_file, TrainingDataFileKeys.BAD_PIXELS)
            object_array = _read_array(h5_file, TrainingDataFileKeys.OBJECT_ARRAY)
            object_geometry = _read_object_geometry(h5_file, object_array)
            detector_distance_m = _read_scalar(
                h5_file, TrainingDataFileKeys.DETECTOR_OBJECT_DISTANCE
            )
            probe_energy_eV = _read_scalar(h5_file, TrainingDataFileKeys.PROBE_ENERGY)  # noqa: N806

        num_patterns, detector_height, detector_width = patterns.shape
        detector_extent = ImageExtent(width_px=detector_width, height_px=detector_height)
        detector_pixel_geometry: PixelGeometry | None = None

        try:
            # The far-field relation is its own inverse, so applying it to the
            # object pixel size returns the detector pitch the file cannot store.
            detector_pixel_geometry = compute_far_field_pixel_geometry(
                object_geometry.get_pixel_geometry(),
                detector_extent,
                wavelength_m=energy_eV_to_wavelength_m(probe_energy_eV),
                propagation_distance_m=detector_distance_m,
            )
        except ZeroDivisionError:
            # Degenerate geometry; leaving this unset falls back to the detector
            # settings, as every reader that records no pitch at all does.
            pass

        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=patterns.dtype,
            detector_extent=detector_extent,
            detector_distance_m=detector_distance_m,
            detector_pixel_geometry=detector_pixel_geometry,
            probe_energy_eV=probe_energy_eV,
            file_path=file_path,
        )

        contents_tree = DiffractionDatasetLayoutNode.create_root()
        contents_tree.add_child(
            file_path.stem, type(patterns).__name__, f'{patterns.dtype}{patterns.shape}'
        )

        array = SimpleDiffractionArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            patterns=patterns,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, [array], bad_pixels=bad_pixels)


class TrainingDataProductFileReader(ProductFileReader):
    def read(self, file_path: Path) -> Product:
        with h5py.File(file_path, 'r') as h5_file:
            object_array = _read_array(h5_file, TrainingDataFileKeys.OBJECT_ARRAY)
            probe_array = _read_array(h5_file, TrainingDataFileKeys.PROBE_ARRAY)
            position_x_px = _read_array(h5_file, TrainingDataFileKeys.PROBE_POSITION_X_PX)
            position_y_px = _read_array(h5_file, TrainingDataFileKeys.PROBE_POSITION_Y_PX)
            object_geometry = _read_object_geometry(h5_file, object_array)
            detector_distance_m = _read_scalar(
                h5_file, TrainingDataFileKeys.DETECTOR_OBJECT_DISTANCE
            )
            probe_energy_eV = _read_scalar(h5_file, TrainingDataFileKeys.PROBE_ENERGY)  # noqa: N806

        metadata = ProductMetadata(
            name=file_path.stem,
            comments='',
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=0.0,  # not included in file
            exposure_time_s=0.0,  # not included in file
            mass_attenuation_m2_kg=0.0,  # not included in file
            tomography_angle_deg=0.0,  # not included in file
        )

        point_list = [
            object_geometry.map_coordinates_object_to_probe(ObjectPosition(index, x_px, y_px))
            for index, (x_px, y_px) in enumerate(zip(position_x_px, position_y_px))
        ]

        # Far field puts the probe on the same sample-plane sampling as the object,
        # which is why only one pixel size is recorded.
        pixel_geometry = object_geometry.get_pixel_geometry()

        return Product(
            metadata=metadata,
            probe_positions=ProbePositionSequence(point_list),
            probes=ProbeSequence(
                array=probe_array,
                opr_weights=None,  # not included in file
                pixel_geometry=pixel_geometry,
            ),
            object_=Object(
                array=object_array,
                pixel_geometry=pixel_geometry,
                center=object_geometry.get_center(),
                layer_spacing_m=[],
            ),
            losses=[],  # not included in file
        )


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'Ptychodus_Training_Data'  # noqa: N806
    DISPLAY_NAME: Final[str] = 'Ptychodus Training Data Files (*.h5 *.hdf5)'  # noqa: N806

    registry.diffraction_file_readers.register_plugin(
        TrainingDataDiffractionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
    registry.register_product_file_reader_with_adapters(
        TrainingDataProductFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
