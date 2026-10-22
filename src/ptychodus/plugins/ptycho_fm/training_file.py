"""The paired HDF5 files PtychoFM trains from, written and read.

A scan is two files sharing a stem: ``<stem>_dp.hdf5`` holding the diffraction
patterns and ``<stem>_para.hdf5`` holding the reconstruction they are supervised
against. Neither refers to the other, so the suffix is the whole of the association.

The parameter file is a subset of the ptychodus product layout -- every dataset and
attribute in it is a `ProductFileKeys` member -- restricted to what the training
loader reads. Writing it from the product keys rather than from literals is what
keeps the two descriptions of the same thing from drifting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import ProductFileKeys
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.preprocess.diffraction import zero_bad_pixels
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput

logger = logging.getLogger(__name__)

__all__ = [
    'PtychoFMProductFileReader',
    'resolve_training_pair',
    'save_training_data',
]

_DIFFRACTION_SUFFIX = '_dp'
_PARAMETER_SUFFIX = '_para'
_FILE_SUFFIX = '.hdf5'

# The patterns, under the name the loader looks for. It is not a product key: the
# product layout keeps patterns in a separate file of its own.
_DIFFRACTION_DATA_PATH = 'dp'

# The pattern maximum the loader divides by. Written because PtychoFM's own producer
# writes it and its normalization tooling expects it; nothing in ptychodus reads it.
_NORMALIZATION = 'normalization'


def resolve_training_pair(file_path: Path) -> tuple[Path, Path]:
    """The pattern and parameter paths a path names, from either member or a stem.

    Either member identifies the pair, because the loader on the other side accepts
    either and a reader should too. A stem carrying neither suffix is taken as the
    shared stem, which is what lets a writer be handed one destination and produce
    both files from it.

    Returns:
        The pattern path and the parameter path, in that order. Neither is checked
        for existence.
    """
    stem = file_path.stem

    for suffix in (_DIFFRACTION_SUFFIX, _PARAMETER_SUFFIX):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    return (
        file_path.with_name(f'{stem}{_DIFFRACTION_SUFFIX}{_FILE_SUFFIX}'),
        file_path.with_name(f'{stem}{_PARAMETER_SUFFIX}{_FILE_SUFFIX}'),
    )


def _verify_spatial_size(shape: tuple[int, ...], img_size: int, what: str) -> None:
    """Raise unless a trailing ``(height, width)`` is exactly ``img_size`` square."""
    if tuple(shape) != (img_size, img_size):
        height, width = shape
        raise ValueError(
            f'{what} are {height}x{width}, but PtychoFM requires {img_size}x{img_size} '
            'to match the encoder input. Adjust the pattern crop or the model image size.'
        )


def save_training_data(file_path: Path, parameters: ReconstructInput, *, img_size: int) -> None:
    """Write a ReconstructInput as the HDF5 pair PtychoFM trains from.

    `file_path` names the shared stem rather than a file: the patterns go to
    ``<stem>_dp.hdf5`` and everything else to ``<stem>_para.hdf5``, so the path handed
    in is itself never created.

    Patterns and probe are stored at their native spatial size and the object patch is
    cut at the pattern size, so both must already be `img_size` square; nothing
    downstream resizes them, and a mismatch would otherwise surface as a shape error
    inside the model. The probe is the mode stack at OPR index zero, carrying the
    leading axis the loader requires.

    Bad pixels are zeroed rather than inpainted, which is what the model is trained
    against.

    Raises:
        ValueError: when the patterns or the probe are not `img_size` square.
    """
    diffraction_path, parameter_path = resolve_training_pair(file_path)

    patterns = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)
    object_ = parameters.product.object_
    pixel_geometry = object_.get_pixel_geometry()
    probe_array = parameters.product.probes.get_probe_no_opr().get_array()
    positions = parameters.product.probe_positions

    _verify_spatial_size(patterns.shape[-2:], img_size, 'Diffraction patterns')
    _verify_spatial_size(probe_array.shape[-2:], img_size, 'Probe')

    with h5py.File(diffraction_path, 'w') as h5_file:
        h5_file.create_dataset(_DIFFRACTION_DATA_PATH, data=patterns)

    with h5py.File(parameter_path, 'w') as h5_file:
        object_dataset = h5_file.create_dataset(
            ProductFileKeys.OBJECT_ARRAY, data=object_.get_layer(0)[numpy.newaxis, :, :]
        )
        object_dataset.attrs[ProductFileKeys.OBJECT_PIXEL_HEIGHT] = pixel_geometry.height_m
        object_dataset.attrs[ProductFileKeys.OBJECT_PIXEL_WIDTH] = pixel_geometry.width_m

        try:
            center = object_.get_center()
        except ValueError:
            # Recorded when it is known. Requiring one would refuse an export that
            # works today, and the loader does not read it.
            pass
        else:
            object_dataset.attrs[ProductFileKeys.OBJECT_CENTER_X] = center.x_m
            object_dataset.attrs[ProductFileKeys.OBJECT_CENTER_Y] = center.y_m

        h5_file.create_dataset(
            ProductFileKeys.PROBE_ARRAY, data=probe_array[numpy.newaxis, :, :, :]
        )
        h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_INDEXES, data=positions.get_indexes())
        h5_file.create_dataset(
            ProductFileKeys.PROBE_POSITION_X,
            data=numpy.asarray(positions.get_coordinates_x_m(), dtype=numpy.float64),
        )
        h5_file.create_dataset(
            ProductFileKeys.PROBE_POSITION_Y,
            data=numpy.asarray(positions.get_coordinates_y_m(), dtype=numpy.float64),
        )
        h5_file.create_dataset(_NORMALIZATION, data=numpy.float32(patterns.max()))


def _require(h5_file: h5py.File, key: str) -> Any:
    """Read a dataset the format requires, naming the file when it is absent.

    A fold_slice parameter file shares the ``_para`` suffix and nothing else, so the
    likeliest wrong file has to fail by saying which description it did not match.
    """
    try:
        return h5_file[key]
    except KeyError:
        raise ValueError(
            f'"{h5_file.filename}" is not a PtychoFM parameter file: no "{key}" dataset.'
        ) from None


def _read_positions(
    h5_file: h5py.File, object_extent_px: tuple[int, int], pixel_height_m: float
) -> ProbePositionSequence:
    """Read the scan, converting to meters when it was stored in pixels.

    PtychoFM infers the unit from the span: a scan reaching between a tenth and ten
    times the object's own extent along both axes is already in pixels, and anything
    else is in meters. Reproduced rather than reinvented, because a reader that
    disagreed with the trainer about the unit would describe a different scan. The
    trainer never consults the pixel width, so neither does this.
    """
    x_m = numpy.asarray(_require(h5_file, ProductFileKeys.PROBE_POSITION_X)[()], dtype=float)
    y_m = numpy.asarray(_require(h5_file, ProductFileKeys.PROBE_POSITION_Y)[()], dtype=float)
    height_px, width_px = object_extent_px

    in_pixels = (
        0.1 * height_px < numpy.ptp(y_m) < 10.0 * height_px
        and 0.1 * width_px < numpy.ptp(x_m) < 10.0 * width_px
    )

    if in_pixels:
        x_m = x_m * pixel_height_m
        y_m = y_m * pixel_height_m

    indexes_dataset = h5_file.get(ProductFileKeys.PROBE_POSITION_INDEXES)
    indexes = (
        numpy.arange(len(x_m)) if indexes_dataset is None else numpy.asarray(indexes_dataset[()])
    )

    return ProbePositionSequence(
        [ProbePosition(int(index), x, y) for index, x, y in zip(indexes, x_m, y_m)]
    )


class PtychoFMProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'PtychoFM'
    DISPLAY_NAME: Final[str] = 'PtychoFM Training Pair (*.hdf5)'

    def read(self, file_path: Path) -> Product:
        _, parameter_path = resolve_training_pair(file_path)

        with h5py.File(parameter_path, 'r') as h5_file:
            object_dataset = _require(h5_file, ProductFileKeys.OBJECT_ARRAY)
            object_array = object_dataset[()]
            pixel_height_m = float(object_dataset.attrs[ProductFileKeys.OBJECT_PIXEL_HEIGHT])
            pixel_width_m = float(
                object_dataset.attrs.get(ProductFileKeys.OBJECT_PIXEL_WIDTH, pixel_height_m)
            )
            pixel_geometry = PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m)

            positions = _read_positions(
                h5_file, object_array.shape[-2:], pixel_height_m=pixel_height_m
            )
            probe_array = _require(h5_file, ProductFileKeys.PROBE_ARRAY)[()]

            center = ObjectCenter(
                x_m=float(object_dataset.attrs.get(ProductFileKeys.OBJECT_CENTER_X, 0.0)),
                y_m=float(object_dataset.attrs.get(ProductFileKeys.OBJECT_CENTER_Y, 0.0)),
            )
            spacing_dataset = h5_file.get(ProductFileKeys.OBJECT_LAYER_SPACING)
            layer_spacing_m = (
                [0.0] * (object_array.shape[-3] - 1)
                if spacing_dataset is None
                else [float(spacing) for spacing in spacing_dataset[()]]
            )

        metadata = ProductMetadata(
            name=resolve_training_pair(file_path)[1].stem.removesuffix(_PARAMETER_SUFFIX),
            comments='',
            detector_distance_m=0.0,  # not included in file
            photon_energy_eV=0.0,  # not included in file
            probe_photon_count=0.0,  # not included in file
            exposure_time_s=0.0,  # not included in file
            mass_attenuation_m2_per_kg=0.0,  # not included in file
            tomography_angle_deg=0.0,  # not included in file
        )

        return Product(
            metadata=metadata,
            probe_positions=positions,
            probes=ProbeEnsemble(
                array=probe_array,
                opr_weights=None,  # the format carries the stack at OPR index zero
                # The format records no detector, so a probe on the object grid is the
                # only sampling available -- and the one the writer above assumed.
                pixel_geometry=pixel_geometry,
            ),
            object_=Object(
                array=object_array,
                pixel_geometry=pixel_geometry,
                center=center,
                layer_spacing_m=layer_spacing_m,
            ),
            losses=[],
        )
