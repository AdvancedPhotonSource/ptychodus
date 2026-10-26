"""I/O utilities for assembled diffraction data, data products and CXI files.

CXI (Coherent X-ray Imaging, version 1.6) files are read and written by the
``*_cxi_*`` functions. A CXI translation is the sample's offset from the beam in a
right-handed frame with y up and z downstream, and images follow CXI's default
detector orientation, whose rows step towards -y and columns towards -x. Under that
orientation the translation equals the probe position in ptychodus coordinates, so
positions are written and read without a sign change.
"""

from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntFlag, StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast, TypeVar
import logging
import re
import warnings

import h5py
import numpy

from .assemble import AssembledDiffractionData
from .constants import (
    ELECTRON_VOLT_J,
    energy_eV_to_J,
    energy_eV_to_wavelength_m,
    energy_J_to_eV,
    wavelength_m_to_energy_eV,
)
from .diffraction import (
    BadPixels,
    BeamCenter,
    DetectorBackground,
    DiffractionDataset,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPatterns,
    FlatField,
    PixelMap,
    Polarization,
)
from .typing import AnyArrayType, ComplexArrayType, NumberArrayType, RealArrayType
from .preprocess.diffraction import inpaint_bad_pixels, zero_bad_pixels
from .fluorescence import ElementMap, FluorescenceDataset
from .geometry import ImageExtent, PixelGeometry
from .object import Object, ObjectCenter
from .probe import ProbeEnsemble
from .probe_positions import ProbePositionSequence, ProbePosition, calculate_scan_geometry
from .product import LossValue, Product, ProductMetadata
from .propagate import compute_far_field_pixel_geometry, compute_near_field_pixel_geometry
from .reconstruct import ReconstructInput

__all__ = [
    'CXIDiffractionContents',
    'CXIFileKeys',
    'CXIPixelMask',
    'CXIVariant',
    'FluorescenceFileKeys',
    'StandardFileLayout',
    'load_cxi_background',
    'load_cxi_bad_pixels',
    'load_cxi_diffraction_contents',
    'load_cxi_flat_field',
    'load_cxi_product',
    'load_diffraction_data',
    'load_fluorescence_data',
    'load_product',
    'save_cxi_diffraction',
    'save_cxi_product',
    'save_cxi_reconstruct_input',
    'save_diffraction_data',
    'save_fluorescence_data',
    'save_product',
    'sanitize_path_component',
    'resolve_external_link_path',
]

_T = TypeVar('_T')

logger = logging.getLogger(__name__)

_UNSAFE_PATH_CHARS = re.compile(r'[^A-Za-z0-9._-]')
_MAX_PATH_COMPONENT_LENGTH = 128


def sanitize_path_component(name: str, *, fallback: str = 'unnamed') -> str:
    """Reduce an untrusted name to a single safe path component.

    Product names are read verbatim from user-supplied files, so they must never reach a
    filesystem join or a remote shell command unfiltered. Separators, shell metacharacters,
    and leading dots are replaced so the result cannot traverse directories, expand in a
    shell, or create a hidden entry.
    """
    cleaned = _UNSAFE_PATH_CHARS.sub('_', name).strip(' .')
    return cleaned[:_MAX_PATH_COMPONENT_LENGTH] or fallback


def resolve_external_link_path(base_directory: Path, filename: str) -> Path:
    """Resolve an HDF5 external-link target against the directory holding the master file.

    The target is chosen by whoever wrote the master file, so an absolute path or a parent
    traversal would let a crafted dataset pull in any HDF5 file the user can read. Symlinks
    are deliberately not resolved: beamline data directories legitimately contain them.
    """
    link_path = Path(filename)

    if link_path.is_absolute() or '..' in link_path.parts:
        raise ValueError(f'Refusing external link outside the data directory: {filename!r}')

    return base_directory / link_path


class StandardFileLayout(StrEnum):
    """Conventional file names used in the ptychodus standard HDF5 workflow directory."""

    DIFFRACTION = 'diffraction.h5'
    FLUORESCENCE = 'fluorescence.h5'
    # Console capture: whatever the run wrote to its standard output and standard
    # error, including the output of C extensions that a sys.stdout reassignment
    # would not reach. Named for the application rather than for reconstruction, so
    # that any activity writing into a workflow directory uses the one name.
    LOG = 'ptychodus.log'
    # Stem only; the per-backend extension comes from
    # TrainableReconstructor.get_model_file_extension(). The full path is
    # f'{input_directory}/{MODEL_BASENAME}{ext}'.
    MODEL_BASENAME = 'model'
    PRODUCT = 'product.h5'
    # Serialized reconstructor options: the resolved task a run was launched with,
    # after every value the product supplies has been folded in. Distinct from
    # SETTINGS, which holds the application's own configuration rather than what the
    # reconstruction backend received.
    PTYCHI_OPTIONS = 'ptychi_options.yaml'
    SETTINGS = 'settings.ini'

    def path(self, directory: Path) -> Path:
        return directory / self.value

    def checkpoint_path(self, directory: Path, epoch: int) -> Path:
        # Per-epoch snapshot beside the final artifact, for a streaming reconstruction
        # that publishes intermediate results.
        p = Path(self.value)
        return directory / f'{p.stem}.{epoch:06d}{p.suffix}'


class DiffractionFileKeys(StrEnum):
    """HDF5 dataset and attribute names for the assembled diffraction data file."""

    PATTERNS = 'patterns'
    DETECTOR_PIXEL_HEIGHT = 'detector_pixel_height_m'
    DETECTOR_PIXEL_WIDTH = 'detector_pixel_width_m'
    INDEXES = 'indexes'
    BAD_PIXELS = 'bad_pixels'
    PROBE_PHOTON_COUNTS = 'probe_photon_counts'
    BACKGROUND = 'background'
    FLAT_FIELD = 'flat_field'


def load_diffraction_data(
    file: Path,
    *,
    mmap_file: Path | None = None,
    mmap_chunk_frames: int = 1024,
) -> AssembledDiffractionData:
    """Load assembled diffraction data from an HDF5 file written by :func:`save_diffraction_data`.

    When ``mmap_file`` is given, the patterns are staged into a read-only ``numpy.memmap``
    backed by that path rather than held in RAM; indexes and bad pixels are small and load
    normally. The caller owns ``mmap_file`` — it is overwritten if present and never removed.
    ``mmap_chunk_frames`` sets how many frames are copied from HDF5 into the memory map
    per read.

    A background or flat field the patterns were not corrected for is loaded when the
    file carries one; files without them load as before.
    """
    with h5py.File(file, 'r') as h5_file:
        h5_indexes = h5_file[DiffractionFileKeys.INDEXES]

        if not isinstance(h5_indexes, h5py.Dataset):
            raise ValueError('Indexes are not a dataset!')

        h5_patterns = h5_file[DiffractionFileKeys.PATTERNS]

        if not isinstance(h5_patterns, h5py.Dataset):
            raise ValueError('Patterns are not a dataset!')

        detector_pixel_geometry = PixelGeometry(
            width_m=float(h5_patterns.attrs[DiffractionFileKeys.DETECTOR_PIXEL_WIDTH]),
            height_m=float(h5_patterns.attrs[DiffractionFileKeys.DETECTOR_PIXEL_HEIGHT]),
        )

        h5_bad_pixels = h5_file[DiffractionFileKeys.BAD_PIXELS]

        if not isinstance(h5_bad_pixels, h5py.Dataset):
            raise ValueError('Bad pixels are not a dataset!')

        if mmap_file is None:
            patterns = h5_patterns[()]
        else:
            # HDF5 cannot be memory mapped directly, so stage patterns block by block into
            # mmap_file, then reopen read-only so nothing downstream can dirty the caller's
            # scratch file.
            staging = numpy.memmap(
                mmap_file, dtype=h5_patterns.dtype, mode='w+', shape=h5_patterns.shape
            )
            for lo in range(0, h5_patterns.shape[0], mmap_chunk_frames):
                hi = lo + mmap_chunk_frames
                staging[lo:hi] = h5_patterns[lo:hi]
            staging.flush()
            del staging
            patterns = numpy.memmap(
                mmap_file, dtype=h5_patterns.dtype, mode='r', shape=h5_patterns.shape
            )

        probe_photon_counts = None
        h5_probe_photon_counts = h5_file.get(DiffractionFileKeys.PROBE_PHOTON_COUNTS)
        if isinstance(h5_probe_photon_counts, h5py.Dataset):
            probe_photon_counts = h5_probe_photon_counts[()]

        background = None
        h5_background = h5_file.get(DiffractionFileKeys.BACKGROUND)
        if isinstance(h5_background, h5py.Dataset):
            background = h5_background[()]

        flat_field = None
        h5_flat_field = h5_file.get(DiffractionFileKeys.FLAT_FIELD)
        if isinstance(h5_flat_field, h5py.Dataset):
            flat_field = h5_flat_field[()]

        return AssembledDiffractionData(
            h5_indexes[()],
            patterns,
            detector_pixel_geometry,
            h5_bad_pixels[()],
            probe_photon_counts=probe_photon_counts,
            background=background,
            flat_field=flat_field,
        )


def save_diffraction_data(
    file: Path, data: AssembledDiffractionData, *, compression: str = 'lzf'
) -> None:
    """Write assembled diffraction data to an HDF5 file."""
    with h5py.File(file, 'w') as h5_file:
        h5_file.create_dataset(
            DiffractionFileKeys.INDEXES, data=data._indexes, compression=compression
        )
        h5_patterns = h5_file.create_dataset(
            DiffractionFileKeys.PATTERNS, data=data._patterns, compression=compression
        )
        detector_pixel_geometry = data.get_pixel_geometry()
        h5_patterns.attrs[DiffractionFileKeys.DETECTOR_PIXEL_WIDTH] = (
            detector_pixel_geometry.width_m
        )
        h5_patterns.attrs[DiffractionFileKeys.DETECTOR_PIXEL_HEIGHT] = (
            detector_pixel_geometry.height_m
        )
        h5_file.create_dataset(
            DiffractionFileKeys.BAD_PIXELS, data=data._bad_pixels, compression=compression
        )
        if data.has_measured_probe_photon_counts():
            assert data._probe_photon_counts is not None
            h5_file.create_dataset(
                DiffractionFileKeys.PROBE_PHOTON_COUNTS,
                data=data._probe_photon_counts,
                compression=compression,
            )

        background = data.get_background()
        if background is not None:
            h5_file.create_dataset(
                DiffractionFileKeys.BACKGROUND, data=background, compression=compression
            )

        flat_field = data.get_flat_field()
        if flat_field is not None:
            h5_file.create_dataset(
                DiffractionFileKeys.FLAT_FIELD, data=flat_field, compression=compression
            )


class ProductFileKeys(StrEnum):
    """HDF5 dataset and attribute names for the product file."""

    NAME = 'name'
    COMMENTS = 'comments'
    DETECTOR_OBJECT_DISTANCE = 'detector_object_distance_m'
    FOCUS_OBJECT_DISTANCE = 'focus_object_distance_m'
    PHOTON_ENERGY = 'photon_energy_eV'
    PROBE_PHOTON_COUNT = 'probe_photon_count'
    EXPOSURE_TIME = 'exposure_time_s'
    MASS_ATTENUATION = 'mass_attenuation_m2_per_kg'
    TOMOGRAPHY_ANGLE = 'tomography_angle_deg'
    TILT_ANGLE = 'tilt_angle_deg'
    POLARIZATION = 'polarization'
    FAR_FIELD = 'far_field'
    PROBE_ARRAY = 'probe'
    OPR_WEIGHTS = 'opr_weights'
    PROBE_PIXEL_HEIGHT = 'pixel_height_m'
    PROBE_PIXEL_WIDTH = 'pixel_width_m'
    PROBE_POSITION_INDEXES = 'probe_position_indexes'
    PROBE_POSITION_X = 'probe_position_x_m'
    PROBE_POSITION_Y = 'probe_position_y_m'
    PROBE_PHOTON_COUNTS = 'probe_photon_counts'
    OBJECT_ARRAY = 'object'
    OBJECT_CENTER_X = 'center_x_m'
    OBJECT_CENTER_Y = 'center_y_m'
    OBJECT_LAYER_SPACING = 'object_layer_spacing_m'
    OBJECT_PIXEL_HEIGHT = 'pixel_height_m'
    OBJECT_PIXEL_WIDTH = 'pixel_width_m'
    LOSS_EPOCHS = 'loss_epochs'
    LOSS_VALUES = 'loss_values'


def _read_photon_energy_eV(h5_file: h5py.File) -> float:  # noqa: N802
    """Read the photon energy, accepting the pre-rename attribute name.

    Unlike the other product attributes this one has no sensible default -- every
    wavelength downstream follows from it -- so a missing value raises rather than
    silently becoming zero.
    """
    # A frozen historical spelling rather than part of the key vocabulary: it is never
    # written, so it has no place in ProductFileKeys.
    legacy_key = 'probe_energy_eV'

    for key in (ProductFileKeys.PHOTON_ENERGY, legacy_key):
        try:
            return float(h5_file.attrs[key])
        except KeyError:
            continue

    raise KeyError(
        f'Product is missing {ProductFileKeys.PHOTON_ENERGY!s} (and the legacy {legacy_key}).'
    )


def load_product(file: Path) -> Product:
    """Load a data product from an HDF5 file written by :func:`save_product`."""
    point_list: list[ProbePosition] = []

    with h5py.File(file, 'r') as h5_file:
        name = str(h5_file.attrs.get(ProductFileKeys.NAME, 'Unnamed'))
        comments = str(h5_file.attrs.get(ProductFileKeys.COMMENTS, ''))
        probe_photon_count = float(h5_file.attrs.get(ProductFileKeys.PROBE_PHOTON_COUNT, 0.0))
        exposure_time_s = float(h5_file.attrs.get(ProductFileKeys.EXPOSURE_TIME, 0.0))
        mass_attenuation_m2_per_kg = float(h5_file.attrs.get(ProductFileKeys.MASS_ATTENUATION, 0.0))
        tomography_angle_deg = float(h5_file.attrs.get(ProductFileKeys.TOMOGRAPHY_ANGLE, 0.0))
        tilt_angle_deg = float(h5_file.attrs.get(ProductFileKeys.TILT_ANGLE, 0.0))
        focus_object_distance_m = float(
            h5_file.attrs.get(ProductFileKeys.FOCUS_OBJECT_DISTANCE, 0.0)
        )
        # HDF5 hands back numpy.bool; cast at the boundary as the floats above do.
        far_field = bool(h5_file.attrs.get(ProductFileKeys.FAR_FIELD, True))

        polarization: Polarization | None = None
        if ProductFileKeys.POLARIZATION in h5_file.attrs:
            polarization = _parse_polarization(h5_file.attrs[ProductFileKeys.POLARIZATION], file)

        metadata = ProductMetadata(
            name=name,
            comments=comments,
            detector_distance_m=float(h5_file.attrs[ProductFileKeys.DETECTOR_OBJECT_DISTANCE]),
            photon_energy_eV=_read_photon_energy_eV(h5_file),
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_per_kg=mass_attenuation_m2_per_kg,
            tomography_angle_deg=tomography_angle_deg,
            focus_object_distance_m=focus_object_distance_m,
            tilt_angle_deg=tilt_angle_deg,
            polarization=polarization,
            far_field=far_field,
        )

        h5_object = h5_file[ProductFileKeys.OBJECT_ARRAY]

        if not isinstance(h5_object, h5py.Dataset):
            raise ValueError('Object array is not a dataset!')

        object_pixel_geometry = PixelGeometry(
            width_m=float(h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_WIDTH]),
            height_m=float(h5_object.attrs[ProductFileKeys.OBJECT_PIXEL_HEIGHT]),
        )
        object_center = ObjectCenter(
            x_m=float(h5_object.attrs[ProductFileKeys.OBJECT_CENTER_X]),
            y_m=float(h5_object.attrs[ProductFileKeys.OBJECT_CENTER_Y]),
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

        probe = ProbeEnsemble(
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

        # The all-or-nothing invariant on ProbePositionSequence means save_product only
        # writes probe_photon_counts when every position had a count; a present
        # dataset therefore aligns 1:1 with probe_position_indexes and needs no dict.
        position_photon_counts_array: RealArrayType | None = None
        h5_position_photon_counts = h5_file.get(ProductFileKeys.PROBE_PHOTON_COUNTS)
        if isinstance(h5_position_photon_counts, h5py.Dataset):
            position_photon_counts_array = h5_position_photon_counts[()]

        indexes = h5_position_indexes[()]
        xs = h5_position_x[()]
        ys = h5_position_y[()]
        for i, (idx, x_m, y_m) in enumerate(zip(indexes, xs, ys)):
            photon_count = (
                None
                if position_photon_counts_array is None
                else float(position_photon_counts_array[i])
            )
            point = ProbePosition(idx, x_m, y_m, probe_photon_count=photon_count)
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
    """Write a data product to an HDF5 file."""
    scan_indexes = product.probe_positions.get_indexes()
    scan_x_m = product.probe_positions.get_coordinates_x_m()
    scan_y_m = product.probe_positions.get_coordinates_y_m()
    position_photon_counts = product.probe_positions.get_probe_photon_counts()

    with h5py.File(file, 'w') as h5_file:
        metadata = product.metadata
        h5_file.attrs[ProductFileKeys.NAME] = metadata.name
        h5_file.attrs[ProductFileKeys.COMMENTS] = metadata.comments
        h5_file.attrs[ProductFileKeys.DETECTOR_OBJECT_DISTANCE] = metadata.detector_distance_m
        h5_file.attrs[ProductFileKeys.PHOTON_ENERGY] = metadata.photon_energy_eV
        h5_file.attrs[ProductFileKeys.PROBE_PHOTON_COUNT] = metadata.probe_photon_count
        h5_file.attrs[ProductFileKeys.EXPOSURE_TIME] = metadata.exposure_time_s
        h5_file.attrs[ProductFileKeys.MASS_ATTENUATION] = metadata.mass_attenuation_m2_per_kg
        h5_file.attrs[ProductFileKeys.TOMOGRAPHY_ANGLE] = metadata.tomography_angle_deg
        h5_file.attrs[ProductFileKeys.FOCUS_OBJECT_DISTANCE] = metadata.focus_object_distance_m
        h5_file.attrs[ProductFileKeys.TILT_ANGLE] = metadata.tilt_angle_deg
        h5_file.attrs[ProductFileKeys.FAR_FIELD] = metadata.far_field
        if metadata.polarization is not None:
            h5_file.attrs[ProductFileKeys.POLARIZATION] = metadata.polarization.value

        h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_INDEXES, data=scan_indexes)
        h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_X, data=scan_x_m)
        h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_Y, data=scan_y_m)
        if position_photon_counts is not None:
            h5_file.create_dataset(
                ProductFileKeys.PROBE_PHOTON_COUNTS,
                data=position_photon_counts,
            )

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


class FluorescenceFileKeys(StrEnum):
    """HDF5 paths for the primary XRF-Maps NNLS layout used by :func:`save_fluorescence_data`."""

    COUNTS_PER_SECOND = '/MAPS/XRF_Analyzed/NNLS/Counts_Per_Sec'
    CHANNEL_NAMES = '/MAPS/XRF_Analyzed/NNLS/Channel_Names'


_FLUORESCENCE_LOAD_PATHS: tuple[tuple[str, str], ...] = (
    (FluorescenceFileKeys.COUNTS_PER_SECOND, FluorescenceFileKeys.CHANNEL_NAMES),
    ('/MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec', '/MAPS/XRF_Analyzed/Fitted/Channel_Names'),
    ('/MAPS/XRF_fits', '/MAPS/channel_names'),
)


def _locate_fluorescence_datasets(h5_file: h5py.File) -> tuple[h5py.Dataset, h5py.Dataset]:
    for cps_path, names_path in _FLUORESCENCE_LOAD_PATHS:
        cps = h5_file.get(cps_path)
        names = h5_file.get(names_path)
        if isinstance(cps, h5py.Dataset) and isinstance(names, h5py.Dataset):
            return cps, names

    tried = ', '.join(cps for cps, _ in _FLUORESCENCE_LOAD_PATHS)
    raise KeyError(f'No known fluorescence layout in file (tried: {tried}).')


def _split_h5_path(data_path: str) -> tuple[str, str]:
    stripped = data_path.strip('/')
    parts = stripped.rsplit('/', 1)
    if len(parts) == 1:
        return '/', parts[0]
    return '/' + parts[0], parts[1]


def load_fluorescence_data(file: Path) -> FluorescenceDataset:
    """Load a fluorescence dataset from an XRF-Maps HDF5 file.

    Recognises the v10 NNLS layout (``/MAPS/XRF_Analyzed/NNLS/...``, preferred),
    the v10 iterative-matrix-fitting layout (``/MAPS/XRF_Analyzed/Fitted/...``),
    and the legacy v9 layout (``/MAPS/XRF_fits`` + ``/MAPS/channel_names``).
    """
    element_maps: list[ElementMap] = []

    with h5py.File(file, 'r') as h5_file:
        h5_counts_per_second, h5_channel_names = _locate_fluorescence_datasets(h5_file)

        counts_per_second = h5_counts_per_second[...]
        channel_names = h5_channel_names[...]

        for bname, cps in zip(channel_names, counts_per_second):
            if isinstance(bname, bytes):
                name = bname.decode('utf-8', errors='replace')
            else:
                name = str(bname)
            element_maps.append(ElementMap(name, cps))

        counts_per_second_path = h5_counts_per_second.name
        channel_names_path = h5_channel_names.name

    return FluorescenceDataset(
        _element_maps=element_maps,
        counts_per_second_path=counts_per_second_path,
        channel_names_path=channel_names_path,
    )


def save_fluorescence_data(file: Path, dataset: FluorescenceDataset) -> None:
    """Write a fluorescence dataset to an HDF5 file at the paths the dataset carries."""
    counts_group_path, counts_ds_name = _split_h5_path(dataset.counts_per_second_path)
    names_group_path, names_ds_name = _split_h5_path(dataset.channel_names_path)

    channel_names = [emap.name for emap in dataset]
    counts_per_second = [emap.counts_per_second for emap in dataset]

    with h5py.File(file, 'w') as h5_file:
        counts_group = h5_file.require_group(counts_group_path)
        counts_group.create_dataset(counts_ds_name, data=numpy.stack(counts_per_second))
        names_group = h5_file.require_group(names_group_path)
        names_group.create_dataset(names_ds_name, data=channel_names, dtype='S256')


def _repair_bad_pixels(
    parameters: ReconstructInput, *, inpaint: bool
) -> tuple[DiffractionPatterns, BadPixels]:
    """Repair the patterns, and return them with the mask saying which pixels were filled.

    An absent mask becomes an all-False one, so a writer always has a mask to store.
    """
    patterns = parameters.diffraction_patterns
    bad_pixels = parameters.bad_pixels

    if bad_pixels is None:
        bad_pixels = numpy.zeros(patterns.shape[1:], dtype=bool)

    repaired = (
        inpaint_bad_pixels(patterns, bad_pixels)
        if inpaint
        else zero_bad_pixels(patterns, bad_pixels)
    )

    return repaired, bad_pixels


class CXIVariant(StrEnum):
    """Layout written for the reconstruction and detector fields a CXI file leaves open.

    Every variant stores translations, energies and detector geometry identically, and
    every variant keeps what ptychodus needs to read its own file back exactly in the
    ``ptychodus`` group of the entry. They differ in where the object, probe and
    detector corrections go, so the file suits the program that will read it.
    """

    STANDARD = 'standard'
    """Strict CXI 1.6: object and probe in titled ``image_N`` groups."""
    SHARP = 'sharp'
    """SHARP's layout: object in ``image_1``, probe in its ``process_1/final_illumination``."""
    CDTOOLS = 'cdtools'
    """cdtools' raw-data layout: adds ``source_1/wavelength`` and ``detector_1/qe_mask``."""


class CXIFileKeys(StrEnum):
    """HDF5 paths ptychodus writes in a CXI file.

    Keys starting with ``/`` are relative to the file root; the rest are relative to
    the entry group (``/entry_1`` when written), and ``entry[key]`` resolves both.
    Names without a slash past the image keys are members of an ``image_N`` group.
    """

    CXI_VERSION = '/cxi_version'
    NUMBER_OF_ENTRIES = '/number_of_entries'

    TITLE = 'title'
    EXPERIMENT_DESCRIPTION = 'experiment_description'
    PROGRAM_NAME = 'program_name'

    SOURCE = 'instrument_1/source_1'
    SOURCE_ENERGY = 'instrument_1/source_1/energy'
    SOURCE_WAVELENGTH = 'instrument_1/source_1/wavelength'
    SOURCE_ILLUMINATION = 'instrument_1/source_1/illumination'

    DETECTOR = 'instrument_1/detector_1'
    DETECTOR_DATA = 'instrument_1/detector_1/data'
    DETECTOR_DISTANCE = 'instrument_1/detector_1/distance'
    DETECTOR_X_PIXEL_SIZE = 'instrument_1/detector_1/x_pixel_size'
    DETECTOR_Y_PIXEL_SIZE = 'instrument_1/detector_1/y_pixel_size'
    DETECTOR_BASIS_VECTORS = 'instrument_1/detector_1/basis_vectors'
    DETECTOR_CORNER_POSITION = 'instrument_1/detector_1/corner_position'
    DETECTOR_MASK = 'instrument_1/detector_1/mask'
    DETECTOR_TRANSLATION = 'instrument_1/detector_1/translation'
    DETECTOR_DATA_DARK = 'instrument_1/detector_1/data_dark'
    DETECTOR_QE_MASK = 'instrument_1/detector_1/qe_mask'

    SAMPLE_NAME = 'sample_1/name'
    SAMPLE_TRANSLATION = 'sample_1/geometry_1/translation'

    DATA_1 = 'data_1'
    DATA_1_DATA = 'data_1/data'
    DATA_1_TRANSLATION = 'data_1/translation'
    DATA_2_DATA = 'data_2/data'

    IMAGE_1 = 'image_1'
    IMAGE_2 = 'image_2'
    FINAL_ILLUMINATION = 'image_1/process_1/final_illumination'

    IMAGE_DATA = 'data'
    IMAGE_DATA_TYPE = 'data_type'
    IMAGE_DATA_SPACE = 'data_space'
    IMAGE_DIMENSIONALITY = 'dimensionality'
    IMAGE_SIZE = 'image_size'
    IMAGE_PROCESS = 'process_1'
    IMAGE_MODE = 'mode'
    IMAGE_LAYER = 'layer'
    PROCESS_PROGRAM = 'program'
    PROCESS_VERSION = 'version'
    PROCESS_DATE = 'date'

    PTYCHODUS = 'ptychodus'
    LAYOUT_VERSION = 'layout_version'
    VARIANT = 'variant'
    DIFFRACTION_INDEXES = 'ptychodus/diffraction_indexes'
    FLAT_FIELD = 'ptychodus/flat_field'
    PROBE_POSITION_INDEXES = 'ptychodus/probe_position_indexes'
    PROBE_PHOTON_COUNTS = 'ptychodus/probe_photon_counts'
    PROBE = 'ptychodus/probe'
    PROBE_PIXEL_WIDTH = 'ptychodus/probe_pixel_width_m'
    PROBE_PIXEL_HEIGHT = 'ptychodus/probe_pixel_height_m'
    PROBE_OPR_MODES = 'ptychodus/probe_opr_modes'
    OPR_WEIGHTS = 'ptychodus/opr_weights'
    OBJECT = 'ptychodus/object'
    OBJECT_PIXEL_WIDTH = 'ptychodus/object_pixel_width_m'
    OBJECT_PIXEL_HEIGHT = 'ptychodus/object_pixel_height_m'
    OBJECT_CENTER_X = 'ptychodus/object_center_x_m'
    OBJECT_CENTER_Y = 'ptychodus/object_center_y_m'
    OBJECT_LAYER_SPACING = 'ptychodus/object_layer_spacing_m'
    LOSS_EPOCHS = 'ptychodus/loss_epochs'
    LOSS_VALUES = 'ptychodus/loss_values'
    PROBE_PHOTON_COUNT = 'ptychodus/probe_photon_count'
    EXPOSURE_TIME = 'ptychodus/exposure_time_s'
    MASS_ATTENUATION = 'ptychodus/mass_attenuation_m2_per_kg'
    TOMOGRAPHY_ANGLE = 'ptychodus/tomography_angle_deg'
    TILT_ANGLE = 'ptychodus/tilt_angle_deg'
    FOCUS_OBJECT_DISTANCE = 'ptychodus/focus_object_distance_m'
    POLARIZATION = 'ptychodus/polarization'
    FAR_FIELD = 'ptychodus/far_field'
    BEAM_CENTER_X = 'ptychodus/beam_center_x_px'
    BEAM_CENTER_Y = 'ptychodus/beam_center_y_px'


class CXIPixelMask(IntFlag):
    """Bits of a CXI detector mask; a pixel with none of them set is good."""

    INVALID = 0x00000001
    """The pixel is invalid (CXI standard)."""
    HAS_SIGNAL = 0x00001000
    """The pixel's signal is above background (CXI standard); informational, not bad."""
    REPAIRED = 0x40000000
    """The value was filled in rather than measured. A high bit, as CXI asks of its users."""


@dataclass(frozen=True)
class CXIDiffractionContents:
    """Where a CXI file keeps its patterns, and everything known about them but the values.

    `data_path` is the absolute HDF5 path of the pattern dataset, so the patterns can
    be read later without holding the file open.
    """

    data_path: str
    indexes: DiffractionIndexes
    bad_pixels: BadPixels | None
    background: DetectorBackground | None
    flat_field: FlatField | None
    metadata: DiffractionMetadata


def _get_or_none(getter: Callable[[], _T]) -> _T | None:
    """Call a getter that raises ValueError for a missing value, mapping that to None."""
    try:
        return getter()
    except ValueError:
        return None


def _parse_polarization(raw: object, source: object) -> Polarization | None:
    """Decode a stored polarization, or None (with a warning) when it is unrecognized."""
    if isinstance(raw, bytes):
        raw = raw.decode('utf-8', errors='replace')

    try:
        return Polarization(str(raw))
    except ValueError:
        logger.warning('Unknown polarization %r in %s; setting polarization=None.', raw, source)
        return None


# ---------- CXI reading ----------


def _open_cxi_entry(h5_file: h5py.File, entry: str | None) -> h5py.Group:
    """Return the requested entry, else ``entry_1``, else the lowest-numbered ``entry_N``."""
    if entry is not None:
        group = h5_file.get(entry)

        if not isinstance(group, h5py.Group):
            raise KeyError(f'CXI file has no entry "{entry}".')

        return group

    numbered = sorted(
        (int(name.removeprefix('entry_')), name)
        for name in h5_file
        if re.fullmatch(r'entry_\d+', name) and isinstance(h5_file.get(name), h5py.Group)
    )

    if not numbered:
        raise ValueError('CXI file has no entry_N group.')

    return h5_file[numbered[0][1]]


def _read_cxi_array(group: h5py.Group, key: str) -> AnyArrayType | None:
    item = group.get(key)
    return numpy.asarray(item[()]) if isinstance(item, h5py.Dataset) else None


def _read_cxi_string(group: h5py.Group, key: str) -> str | None:
    item = group.get(key)

    if not isinstance(item, h5py.Dataset):
        return None

    value = item[()]

    if isinstance(value, numpy.ndarray):
        value = value.ravel()[0] if value.size else ''

    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')

    return str(value)


def _read_cxi_float(group: h5py.Group, key: str) -> float | None:
    """Read a scalar quantity in SI units, or None when absent or unusable.

    A per-frame array (CXI allows one wherever a value can change during a scan) is
    reduced to its median. A ``units`` attribute naming a common length or energy
    unit is converted; without one the value is taken to be SI, as CXI requires.
    """
    unit_scales = {
        'm': 1.0,
        'mm': 1.0e-3,
        'um': 1.0e-6,
        'µm': 1.0e-6,
        'nm': 1.0e-9,
        'J': 1.0,
        'eV': ELECTRON_VOLT_J,
        'keV': 1.0e3 * ELECTRON_VOLT_J,
    }
    item = group.get(key)

    if not isinstance(item, h5py.Dataset):
        return None

    try:
        values = numpy.asarray(item[()], dtype=numpy.float64)
    except (TypeError, ValueError):
        logger.warning('Ignoring non-numeric %s.', item.name)
        return None

    if values.size == 0 or not numpy.isfinite(values).any():
        return None

    value = numpy.nanmedian(values)
    units = item.attrs.get('units')

    if isinstance(units, bytes):
        units = units.decode('utf-8', errors='replace')

    if units is not None:
        scale = unit_scales.get(str(units).strip())

        if scale is None:
            logger.warning('Ignoring unknown units %r on %s; assuming SI.', units, item.name)
        else:
            value = value * scale

    return float(value)


def _read_cxi_photon_energy_eV(entry: h5py.Group) -> float | None:  # noqa: N802
    """Photon energy from the source, else from the beam, else from a wavelength."""
    for key in (CXIFileKeys.SOURCE_ENERGY, 'instrument_1/beam_1/incident_energy'):
        energy_J = _read_cxi_float(entry, key)  # noqa: N806

        if energy_J is not None and energy_J > 0.0:
            return energy_J_to_eV(energy_J)

    for key in ('instrument_1/beam_1/incident_wavelength', CXIFileKeys.SOURCE_WAVELENGTH):
        wavelength_m = _read_cxi_float(entry, key)

        if wavelength_m is not None and wavelength_m > 0.0:
            return wavelength_m_to_energy_eV(wavelength_m)

    return None


def _read_cxi_basis_vectors(
    entry: h5py.Group,
) -> RealArrayType | None:
    """Return the detector basis as a (2, 3) array, rows for the slow and fast axes.

    A per-frame stack contributes its first frame, and a basis stored transposed,
    as (3, 2), is turned the right way round; the two shapes cannot be confused.
    """
    basis = _read_cxi_array(entry, CXIFileKeys.DETECTOR_BASIS_VECTORS)

    if basis is None:
        return None

    vectors = basis.astype(numpy.float64)

    if vectors.ndim == 3:
        vectors = vectors[0]

    if vectors.shape == (3, 2):
        vectors = vectors.T

    if vectors.shape != (2, 3) or not numpy.isfinite(vectors).all():
        logger.warning('Ignoring basis_vectors of shape %s.', vectors.shape)
        return None

    return vectors


def _read_cxi_detector_pixel_geometry(entry: h5py.Group) -> PixelGeometry | None:
    """Detector pitch from the pixel sizes, else from the basis vectors."""
    width_m = _read_cxi_float(entry, CXIFileKeys.DETECTOR_X_PIXEL_SIZE)
    height_m = _read_cxi_float(entry, CXIFileKeys.DETECTOR_Y_PIXEL_SIZE)
    basis = _read_cxi_basis_vectors(entry)

    if basis is not None:
        slow_m = float(numpy.linalg.norm(basis[0]))
        fast_m = float(numpy.linalg.norm(basis[1]))

        if slow_m > 0.0 and fast_m > 0.0:
            # CXI's default orientation steps rows towards -y and columns towards -x.
            is_default = numpy.allclose(basis[0] / slow_m, (0, -1, 0), atol=1e-6) and (
                numpy.allclose(basis[1] / fast_m, (-1, 0, 0), atol=1e-6)
            )

            if not is_default:
                logger.warning(
                    'Detector basis_vectors are not in the default orientation; the'
                    ' patterns are read as stored, without reorienting them.'
                )

            width_m = fast_m if width_m is None else width_m
            height_m = slow_m if height_m is None else height_m

    width_m = height_m if width_m is None else width_m
    height_m = width_m if height_m is None else height_m

    if width_m is None or height_m is None or width_m <= 0.0 or height_m <= 0.0:
        return None

    return PixelGeometry(width_m=width_m, height_m=height_m)


def _read_cxi_detector_distance_m(entry: h5py.Group) -> float | None:
    """Detector distance, else the corner position's distance along the detector normal."""
    distance_m = _read_cxi_float(entry, CXIFileKeys.DETECTOR_DISTANCE)

    if distance_m is not None and distance_m > 0.0:
        return distance_m

    corner = _read_cxi_array(entry, CXIFileKeys.DETECTOR_CORNER_POSITION)

    if corner is None:
        return None

    corner = numpy.asarray(corner, dtype=numpy.float64).reshape(-1, 3)[0]
    basis = _read_cxi_basis_vectors(entry)

    if basis is not None:
        normal = numpy.cross(basis[0], basis[1])
        norm = numpy.linalg.norm(normal)

        if norm > 0.0:
            return float(abs(numpy.dot(corner, normal / norm)))

    return float(abs(corner[2]))


def _read_cxi_bad_pixels(
    entry: h5py.Group, extent: ImageExtent | None, *, ignored_mask_bits: int
) -> BadPixels | None:
    """Read the detector mask as bad pixels: any bit outside `ignored_mask_bits` is bad.

    A per-frame mask is combined so a pixel bad in any frame is bad. A mask that does
    not match `extent` is ignored with a warning.
    """
    mask = _read_cxi_array(entry, CXIFileKeys.DETECTOR_MASK)

    if mask is None:
        return None

    # Widen first: narrow signed masks cannot represent the complement of a high bit.
    bad_pixels = (mask.astype(numpy.int64) & ~int(ignored_mask_bits)) != 0

    if bad_pixels.ndim == 3:
        bad_pixels = bad_pixels.any(axis=0)

    if bad_pixels.ndim != 2 or (extent is not None and bad_pixels.shape != extent.get_shape()):
        logger.warning('Ignoring a detector mask of shape %s.', mask.shape)
        return None

    return bad_pixels


def _read_cxi_pixel_map(
    entry: h5py.Group, keys: Sequence[str], extent: ImageExtent | None
) -> PixelMap | None:
    """Read the first map found under `keys`, averaging a stack over its leading axes."""
    for key in keys:
        pixel_map = _read_cxi_array(entry, key)

        if pixel_map is None:
            continue

        values = pixel_map.astype(numpy.float64)

        if values.ndim > 2:
            with warnings.catch_warnings():
                # An all-NaN pixel stays NaN, which marks it bad downstream.
                warnings.simplefilter('ignore', RuntimeWarning)
                values = numpy.nanmean(values.reshape(-1, *values.shape[-2:]), axis=0)

        if values.ndim != 2 or (extent is not None and values.shape != extent.get_shape()):
            logger.warning('Ignoring %s of shape %s.', key, values.shape)
            return None

        return cast(PixelMap, values)

    return None


def _read_cxi_background(
    entry: h5py.Group, extent: ImageExtent | None
) -> DetectorBackground | None:
    return _read_cxi_pixel_map(entry, (CXIFileKeys.DETECTOR_DATA_DARK,), extent)


def _read_cxi_flat_field(entry: h5py.Group, extent: ImageExtent | None) -> FlatField | None:
    return _read_cxi_pixel_map(
        entry, (CXIFileKeys.FLAT_FIELD, CXIFileKeys.DETECTOR_QE_MASK), extent
    )


def _find_cxi_patterns(entry: h5py.Group) -> h5py.Dataset:
    for key in (CXIFileKeys.DATA_1_DATA, CXIFileKeys.DETECTOR_DATA):
        item = entry.get(key)

        if isinstance(item, h5py.Dataset) and item.ndim in (2, 3) and item.dtype.kind in 'uif':
            return item

    raise ValueError('CXI entry holds no diffraction patterns in data_1 or detector_1.')


def _read_cxi_diffraction_contents(
    entry: h5py.Group, file_path: Path, *, ignored_mask_bits: int
) -> CXIDiffractionContents:
    patterns = _find_cxi_patterns(entry)
    num_patterns = 1 if patterns.ndim == 2 else patterns.shape[0]
    height_px, width_px = patterns.shape[-2:]
    extent = ImageExtent(width_px=width_px, height_px=height_px)

    indexes = numpy.arange(num_patterns)
    stored_indexes = _read_cxi_array(entry, CXIFileKeys.DIFFRACTION_INDEXES)

    if stored_indexes is not None:
        if stored_indexes.shape == (num_patterns,):
            indexes = stored_indexes.astype(numpy.intp)
        else:
            logger.warning('Ignoring %d stored pattern indexes.', stored_indexes.size)

    probe_photon_count = _read_cxi_float(entry, CXIFileKeys.PROBE_PHOTON_COUNT)
    beam_center_x = _read_cxi_float(entry, CXIFileKeys.BEAM_CENTER_X)
    beam_center_y = _read_cxi_float(entry, CXIFileKeys.BEAM_CENTER_Y)
    raw_polarization = _read_cxi_string(entry, CXIFileKeys.POLARIZATION)

    metadata = DiffractionMetadata(
        num_patterns_per_array=[num_patterns],
        pattern_dtype=patterns.dtype,
        detector_extent=extent,
        detector_distance_m=_read_cxi_detector_distance_m(entry),
        detector_pixel_geometry=_read_cxi_detector_pixel_geometry(entry),
        beam_center=(
            None
            if beam_center_x is None or beam_center_y is None
            else BeamCenter(x_px=int(beam_center_x), y_px=int(beam_center_y))
        ),
        photon_energy_eV=_read_cxi_photon_energy_eV(entry),
        probe_photon_count=None if probe_photon_count is None else int(probe_photon_count),
        exposure_time_s=_read_cxi_float(entry, CXIFileKeys.EXPOSURE_TIME),
        tomography_angle_deg=_read_cxi_float(entry, CXIFileKeys.TOMOGRAPHY_ANGLE),
        tilt_angle_deg=_read_cxi_float(entry, CXIFileKeys.TILT_ANGLE),
        polarization=(
            None if raw_polarization is None else _parse_polarization(raw_polarization, file_path)
        ),
        file_path=file_path,
    )

    if 'instrument_1/detector_1/data_white' in entry:
        logger.debug('Ignoring detector_1/data_white, which ptychodus has no use for.')

    return CXIDiffractionContents(
        data_path=patterns.name,
        indexes=indexes,
        bad_pixels=_read_cxi_bad_pixels(entry, extent, ignored_mask_bits=ignored_mask_bits),
        background=_read_cxi_background(entry, extent),
        flat_field=_read_cxi_flat_field(entry, extent),
        metadata=metadata,
    )


def load_cxi_diffraction_contents(
    file_path: Path,
    *,
    entry: str | None = None,
    ignored_mask_bits: int = CXIPixelMask.HAS_SIGNAL,
) -> CXIDiffractionContents:
    """Read everything about a CXI file's diffraction patterns except the pattern values.

    Patterns come from ``data_1/data``, else ``detector_1/data``. Detector geometry
    comes from the pixel sizes or basis vectors, the distance from ``distance`` or
    the corner position, and the energy from the source or beam. The mask marks a
    pixel bad when any of its bits outside `ignored_mask_bits` is set. The background
    comes from ``data_dark`` and the flat-field response from ``ptychodus/flat_field``
    or ``qe_mask``. `entry` selects an entry; by default ``entry_1``.
    """
    with h5py.File(file_path, 'r') as h5_file:
        group = _open_cxi_entry(h5_file, entry)
        return _read_cxi_diffraction_contents(group, file_path, ignored_mask_bits=ignored_mask_bits)


def load_cxi_bad_pixels(
    file_path: Path,
    *,
    entry: str | None = None,
    ignored_mask_bits: int = CXIPixelMask.HAS_SIGNAL,
) -> BadPixels:
    """Read the detector mask of a CXI file, which may hold nothing else.

    Raises ValueError when the file has no mask.
    """
    with h5py.File(file_path, 'r') as h5_file:
        group = _open_cxi_entry(h5_file, entry)
        bad_pixels = _read_cxi_bad_pixels(group, None, ignored_mask_bits=ignored_mask_bits)

    if bad_pixels is None:
        raise ValueError(f'"{file_path}" has no usable detector mask.')

    return bad_pixels


def load_cxi_background(file_path: Path, *, entry: str | None = None) -> DetectorBackground:
    """Read the detector background (``data_dark``) of a CXI file; ValueError if absent."""
    with h5py.File(file_path, 'r') as h5_file:
        background = _read_cxi_background(_open_cxi_entry(h5_file, entry), None)

    if background is None:
        raise ValueError(f'"{file_path}" has no detector background.')

    return background


def load_cxi_flat_field(file_path: Path, *, entry: str | None = None) -> FlatField:
    """Read the flat-field response of a CXI file; ValueError if absent."""
    with h5py.File(file_path, 'r') as h5_file:
        flat_field = _read_cxi_flat_field(_open_cxi_entry(h5_file, entry), None)

    if flat_field is None:
        raise ValueError(f'"{file_path}" has no flat field.')

    return flat_field


def _read_cxi_probe_positions(entry: h5py.Group) -> ProbePositionSequence:
    translation = None

    for key in (
        CXIFileKeys.SAMPLE_TRANSLATION,
        CXIFileKeys.DATA_1_TRANSLATION,
        CXIFileKeys.DETECTOR_TRANSLATION,
    ):
        translation = _read_cxi_array(entry, key)

        if translation is not None:
            break

    if translation is None:
        logger.warning('CXI entry has no translations; returning no probe positions.')
        return ProbePositionSequence()

    translation = translation.astype(numpy.float64)

    if translation.ndim != 2 or translation.shape[1] not in (2, 3):
        raise ValueError(f'Translations must have shape (N, 2) or (N, 3); got {translation.shape}.')

    num_positions = translation.shape[0]
    indexes = numpy.arange(num_positions)
    stored_indexes = _read_cxi_array(entry, CXIFileKeys.PROBE_POSITION_INDEXES)

    if stored_indexes is not None:
        if stored_indexes.shape == (num_positions,):
            indexes = stored_indexes.astype(numpy.intp)
        else:
            logger.warning('Ignoring %d stored position indexes.', stored_indexes.size)

    photon_counts = _read_cxi_array(entry, CXIFileKeys.PROBE_PHOTON_COUNTS)

    if photon_counts is not None and photon_counts.shape != (num_positions,):
        logger.warning('Ignoring %d stored photon counts.', photon_counts.size)
        photon_counts = None

    return ProbePositionSequence(
        [
            ProbePosition(
                int(index),
                float(x_m),
                float(y_m),
                None if photon_counts is None else float(photon_counts[number]),
            )
            for number, (index, x_m, y_m) in enumerate(
                zip(indexes, translation[:, 0], translation[:, 1])
            )
        ]
    )


def _as_complex_image(array: AnyArrayType, name: str) -> ComplexArrayType:
    if numpy.iscomplexobj(array):
        return cast(ComplexArrayType, array)

    logger.warning('%s is real-valued; reading it as a complex image with no phase.', name)
    return array.astype(numpy.complex64)


def _find_titled_cxi_images(entry: h5py.Group) -> dict[str, h5py.Group]:
    """Map each ``image_N`` group's lowercase title to the group; untitled ones key ''."""
    images: dict[str, h5py.Group] = {}

    for name in sorted(entry, key=lambda text: (len(text), text)):
        group = entry.get(name)

        if re.fullmatch(r'image_\d+', name) and isinstance(group, h5py.Group):
            title = (_read_cxi_string(group, CXIFileKeys.TITLE) or '').strip().lower()
            images.setdefault(title, group)

    return images


def _image_pixel_geometry(group: h5py.Group, shape: tuple[int, ...]) -> PixelGeometry | None:
    image_size = _read_cxi_array(group, CXIFileKeys.IMAGE_SIZE)

    if image_size is None or image_size.size < 2 or len(shape) < 2:
        return None

    width_m = float(image_size.ravel()[0]) / shape[-1]
    height_m = float(image_size.ravel()[1]) / shape[-2]
    return PixelGeometry(width_m, height_m) if width_m > 0.0 and height_m > 0.0 else None


def _read_extension_pixel_geometry(
    entry: h5py.Group, width_key: str, height_key: str
) -> PixelGeometry | None:
    width_m = _read_cxi_float(entry, width_key)
    height_m = _read_cxi_float(entry, height_key)

    if width_m is None or height_m is None or width_m <= 0.0 or height_m <= 0.0:
        return None

    return PixelGeometry(width_m, height_m)


def _read_cxi_product_metadata(entry: h5py.Group, file_path: Path) -> ProductMetadata:
    name = (
        _read_cxi_string(entry, CXIFileKeys.TITLE)
        or _read_cxi_string(entry, CXIFileKeys.SAMPLE_NAME)
        or file_path.stem
    )
    detector_distance_m = _read_cxi_detector_distance_m(entry)
    photon_energy_eV = _read_cxi_photon_energy_eV(entry)  # noqa: N806

    if detector_distance_m is None:
        logger.warning('%s records no detector distance; using 0.', file_path)

    if photon_energy_eV is None:
        logger.warning('%s records no photon energy; using 0.', file_path)

    raw_polarization = _read_cxi_string(entry, CXIFileKeys.POLARIZATION)
    far_field = _read_cxi_array(entry, CXIFileKeys.FAR_FIELD)

    def scalar(key: str) -> float:
        value = _read_cxi_float(entry, key)
        return 0.0 if value is None else value

    return ProductMetadata(
        name=name,
        comments=_read_cxi_string(entry, CXIFileKeys.EXPERIMENT_DESCRIPTION) or '',
        detector_distance_m=0.0 if detector_distance_m is None else detector_distance_m,
        photon_energy_eV=0.0 if photon_energy_eV is None else photon_energy_eV,
        probe_photon_count=scalar(CXIFileKeys.PROBE_PHOTON_COUNT),
        exposure_time_s=scalar(CXIFileKeys.EXPOSURE_TIME),
        mass_attenuation_m2_per_kg=scalar(CXIFileKeys.MASS_ATTENUATION),
        tomography_angle_deg=scalar(CXIFileKeys.TOMOGRAPHY_ANGLE),
        focus_object_distance_m=scalar(CXIFileKeys.FOCUS_OBJECT_DISTANCE),
        tilt_angle_deg=scalar(CXIFileKeys.TILT_ANGLE),
        polarization=(
            None if raw_polarization is None else _parse_polarization(raw_polarization, file_path)
        ),
        far_field=True if far_field is None else bool(far_field),
    )


def _compute_sample_plane_pixel_geometry(
    entry: h5py.Group, metadata: ProductMetadata, shape: tuple[int, ...]
) -> PixelGeometry | None:
    """The object-plane pitch implied by the detector pitch, energy and distance."""
    detector_pixel_geometry = _read_cxi_detector_pixel_geometry(entry)

    if detector_pixel_geometry is None or len(shape) < 2:
        return None

    if not metadata.far_field:
        return compute_near_field_pixel_geometry(
            detector_pixel_geometry, magnification=metadata.magnification
        )

    if metadata.photon_energy_eV <= 0.0 or metadata.detector_distance_m <= 0.0:
        return None

    return compute_far_field_pixel_geometry(
        detector_pixel_geometry,
        ImageExtent(width_px=shape[-1], height_px=shape[-2]),
        wavelength_m=metadata.photon_wavelength_m,
        propagation_distance_m=metadata.detector_distance_m,
    )


def _read_cxi_probe(
    entry: h5py.Group,
    images: dict[str, h5py.Group],
    metadata: ProductMetadata,
    num_positions: int,
) -> ProbeEnsemble:
    probe_group: h5py.Group | None = None
    array = _read_cxi_array(entry, CXIFileKeys.PROBE)

    if array is None and 'probe' in images:
        probe_group = images['probe']
        array = _read_cxi_array(probe_group, CXIFileKeys.IMAGE_DATA)
        opr_modes = _read_cxi_array(entry, CXIFileKeys.PROBE_OPR_MODES)

        if array is not None and opr_modes is not None:
            modes = array if array.ndim == 3 else array[numpy.newaxis, ...]

            if opr_modes.shape[1:] == modes.shape:
                array = numpy.concatenate((modes[numpy.newaxis, ...], opr_modes))
            else:
                logger.warning('Ignoring OPR modes of shape %s.', opr_modes.shape)

    for key in (CXIFileKeys.FINAL_ILLUMINATION, CXIFileKeys.SOURCE_ILLUMINATION):
        if array is None:
            array = _read_cxi_array(entry, key)

    if array is None:
        return ProbeEnsemble(None, None, None)

    array = _as_complex_image(array, 'Probe')
    opr_weights = _read_cxi_array(entry, CXIFileKeys.OPR_WEIGHTS)
    num_coherent_modes = array.shape[0] if array.ndim == 4 else 1

    if opr_weights is not None and opr_weights.shape != (num_positions, num_coherent_modes):
        logger.warning('Ignoring OPR weights of shape %s.', opr_weights.shape)
        opr_weights = None

    pixel_geometry = _read_extension_pixel_geometry(
        entry, CXIFileKeys.PROBE_PIXEL_WIDTH, CXIFileKeys.PROBE_PIXEL_HEIGHT
    )

    if pixel_geometry is None and probe_group is not None:
        pixel_geometry = _image_pixel_geometry(probe_group, array.shape)

    if pixel_geometry is None:
        pixel_geometry = _compute_sample_plane_pixel_geometry(entry, metadata, array.shape)

    return ProbeEnsemble(
        array,
        None if opr_weights is None else opr_weights.astype(numpy.float64),
        pixel_geometry,
    )


def _read_cxi_object(
    entry: h5py.Group,
    images: dict[str, h5py.Group],
    positions: ProbePositionSequence,
    probe_pixel_geometry: PixelGeometry | None,
) -> Object:
    object_group: h5py.Group | None = None
    array = _read_cxi_array(entry, CXIFileKeys.OBJECT)

    if array is None:
        object_group = images.get('object')

        if object_group is None and set(images) == {''}:
            object_group = images['']

        if object_group is not None:
            array = _read_cxi_array(object_group, CXIFileKeys.IMAGE_DATA)

    if array is None:
        array = _read_cxi_array(entry, 'instrument_1/detector_1/initial_image')

    if array is None:
        return Object(None, None, None)

    array = _as_complex_image(array, 'Object')
    num_layers = array.shape[0] if array.ndim == 3 else 1
    layer_spacing = _read_cxi_array(entry, CXIFileKeys.OBJECT_LAYER_SPACING)

    if layer_spacing is None and object_group is not None:
        layer_offsets = _read_cxi_array(object_group, CXIFileKeys.IMAGE_LAYER)

        if layer_offsets is not None:
            layer_spacing = numpy.diff(layer_offsets.astype(numpy.float64))

    layer_spacing_m = [] if layer_spacing is None else [float(v) for v in layer_spacing.ravel()]

    if len(layer_spacing_m) != num_layers - 1:
        if layer_spacing_m or num_layers > 1:
            logger.warning(
                'Object has %d layers but %d layer spacings; using zero spacing.',
                num_layers,
                len(layer_spacing_m),
            )

        layer_spacing_m = [0.0] * (num_layers - 1)

    pixel_geometry = _read_extension_pixel_geometry(
        entry, CXIFileKeys.OBJECT_PIXEL_WIDTH, CXIFileKeys.OBJECT_PIXEL_HEIGHT
    )

    if pixel_geometry is None and object_group is not None:
        pixel_geometry = _image_pixel_geometry(object_group, array.shape)

    if pixel_geometry is None:
        pixel_geometry = probe_pixel_geometry

    center_x_m = _read_cxi_float(entry, CXIFileKeys.OBJECT_CENTER_X)
    center_y_m = _read_cxi_float(entry, CXIFileKeys.OBJECT_CENTER_Y)

    if center_x_m is None or center_y_m is None:
        # Centering the object on the scan keeps every position on it.
        scan_geometry = calculate_scan_geometry(positions)
        center_x_m = 0.0 if scan_geometry is None else scan_geometry.center_x_m
        center_y_m = 0.0 if scan_geometry is None else scan_geometry.center_y_m

    return Object(
        array,
        pixel_geometry,
        ObjectCenter(center_x_m, center_y_m),
        layer_spacing_m,
    )


def _read_cxi_losses(entry: h5py.Group) -> list[LossValue]:
    values = _read_cxi_array(entry, CXIFileKeys.LOSS_VALUES)

    if values is None:
        return []

    values = numpy.atleast_1d(values)
    epochs = _read_cxi_array(entry, CXIFileKeys.LOSS_EPOCHS)

    if epochs is None or numpy.atleast_1d(epochs).shape != values.shape:
        epochs = numpy.arange(values.size)

    return [
        LossValue(int(epoch), float(value))
        for epoch, value in zip(numpy.atleast_1d(epochs), values)
    ]


def load_cxi_product(file_path: Path, *, entry: str | None = None) -> Product:
    """Read a data product (positions, probe, object and metadata) from a CXI file.

    The object comes from the ``image_N`` titled ``object``, else a lone untitled
    image, else ``detector_1/initial_image``; the probe from the image titled
    ``probe``, else ``image_1/process_1/final_illumination``, else
    ``source_1/illumination``. Arrays ptychodus stored in full in its own group take
    precedence. A missing probe or object reads as an empty one, so a file holding
    only measured data still yields its positions. Pixel sizes missing from the file
    are computed from the detector geometry, and a missing object center centers the
    object on the scan.
    """
    with h5py.File(file_path, 'r') as h5_file:
        group = _open_cxi_entry(h5_file, entry)
        metadata = _read_cxi_product_metadata(group, file_path)
        positions = _read_cxi_probe_positions(group)
        images = _find_titled_cxi_images(group)
        probes = _read_cxi_probe(group, images, metadata, len(positions))
        object_ = _read_cxi_object(
            group, images, positions, _get_or_none(probes.get_pixel_geometry)
        )
        losses = _read_cxi_losses(group)

    return Product(
        metadata=metadata,
        probe_positions=positions,
        probes=probes,
        object_=object_,
        losses=losses,
    )


# ---------- CXI writing ----------


@dataclass(frozen=True)
class _CXIDiffractionPayload:
    """Diffraction data on its way into a CXI file, from whichever source it came."""

    metadata: DiffractionMetadata
    blocks: Iterable[tuple[DiffractionIndexes, DiffractionPatterns]]
    bad_pixels: BadPixels
    bad_pixel_flags: int
    background: DetectorBackground | None
    flat_field: FlatField | None


def _replace_cxi_dataset(group: h5py.Group, key: str, value: object) -> None:
    if key in group:
        del group[key]

    group.create_dataset(key, data=value)


def _write_cxi_energy(
    entry: h5py.Group,
    photon_energy_eV: float | None,  # noqa: N803
    variant: CXIVariant,
) -> None:
    if photon_energy_eV is None or photon_energy_eV <= 0.0:
        return

    _replace_cxi_dataset(entry, CXIFileKeys.SOURCE_ENERGY, energy_eV_to_J(photon_energy_eV))

    if variant == CXIVariant.CDTOOLS:
        _replace_cxi_dataset(
            entry, CXIFileKeys.SOURCE_WAVELENGTH, energy_eV_to_wavelength_m(photon_energy_eV)
        )


def _write_cxi_diffraction(
    entry: h5py.Group,
    payload: _CXIDiffractionPayload,
    variant: CXIVariant,
    compression: str | None,
) -> None:
    metadata = payload.metadata
    height_px, width_px = payload.bad_pixels.shape
    patterns_dataset: h5py.Dataset | None = None
    index_list: list[DiffractionIndexes] = []

    for indexes, patterns in payload.blocks:
        if patterns.ndim == 2:
            patterns = patterns[numpy.newaxis, ...]

        if patterns_dataset is None:
            patterns_dataset = entry.create_dataset(
                CXIFileKeys.DETECTOR_DATA,
                shape=(0, height_px, width_px),
                maxshape=(None, height_px, width_px),
                chunks=(1, height_px, width_px),
                dtype=patterns.dtype,
                compression=compression,
            )

        offset = patterns_dataset.shape[0]
        patterns_dataset.resize(offset + patterns.shape[0], axis=0)
        patterns_dataset[offset:] = patterns
        index_list.append(numpy.asarray(indexes))

    if patterns_dataset is None:
        patterns_dataset = entry.create_dataset(
            CXIFileKeys.DETECTOR_DATA,
            shape=(0, height_px, width_px),
            maxshape=(None, height_px, width_px),
            chunks=(1, height_px, width_px),
            dtype=metadata.pattern_dtype,
            compression=compression,
        )

    patterns_dataset.attrs['axes'] = 'translation:y:x'
    entry[CXIFileKeys.DATA_1_DATA] = h5py.SoftLink(patterns_dataset.name)
    entry.create_dataset(
        CXIFileKeys.DIFFRACTION_INDEXES,
        data=(
            numpy.concatenate(index_list) if index_list else numpy.zeros(0, dtype=numpy.intp)
        ).astype(numpy.int64),
    )

    mask = numpy.where(payload.bad_pixels, payload.bad_pixel_flags, 0).astype(numpy.uint32)
    entry.create_dataset(CXIFileKeys.DETECTOR_MASK, data=mask, compression=compression)

    if metadata.detector_distance_m is not None:
        _replace_cxi_dataset(entry, CXIFileKeys.DETECTOR_DISTANCE, metadata.detector_distance_m)

    pixel_geometry = metadata.detector_pixel_geometry

    if pixel_geometry is not None:
        entry.create_dataset(CXIFileKeys.DETECTOR_X_PIXEL_SIZE, data=pixel_geometry.width_m)
        entry.create_dataset(CXIFileKeys.DETECTOR_Y_PIXEL_SIZE, data=pixel_geometry.height_m)
        basis = numpy.array(
            [[0.0, -pixel_geometry.height_m, 0.0], [-pixel_geometry.width_m, 0.0, 0.0]]
        )
        entry.create_dataset(
            CXIFileKeys.DETECTOR_BASIS_VECTORS,
            data=basis.T if variant == CXIVariant.CDTOOLS else basis,
        )
        # The outer corner of pixel (0, 0), which the default basis puts at +x and +y of
        # the beam; the beam center is a pixel-center coordinate.
        beam_center_x = (width_px - 1) / 2
        beam_center_y = (height_px - 1) / 2

        if metadata.beam_center is not None:
            beam_center_x = metadata.beam_center.x_px
            beam_center_y = metadata.beam_center.y_px

        corner_position = (
            (beam_center_x + 0.5) * pixel_geometry.width_m,
            (beam_center_y + 0.5) * pixel_geometry.height_m,
            0.0 if metadata.detector_distance_m is None else metadata.detector_distance_m,
        )
        entry.create_dataset(CXIFileKeys.DETECTOR_CORNER_POSITION, data=corner_position)

    _write_cxi_energy(entry, metadata.photon_energy_eV, variant)

    if payload.background is not None:
        entry.create_dataset(
            CXIFileKeys.DETECTOR_DATA_DARK, data=payload.background, compression=compression
        )

    if payload.flat_field is not None:
        entry.create_dataset(
            CXIFileKeys.FLAT_FIELD, data=payload.flat_field, compression=compression
        )

        if variant == CXIVariant.CDTOOLS:
            entry.create_dataset(
                CXIFileKeys.DETECTOR_QE_MASK, data=payload.flat_field, compression=compression
            )

    optional_scalars = (
        (CXIFileKeys.PROBE_PHOTON_COUNT, metadata.probe_photon_count),
        (CXIFileKeys.EXPOSURE_TIME, metadata.exposure_time_s),
        (CXIFileKeys.TOMOGRAPHY_ANGLE, metadata.tomography_angle_deg),
        (CXIFileKeys.TILT_ANGLE, metadata.tilt_angle_deg),
    )

    for key, value in optional_scalars:
        if value is not None:
            _replace_cxi_dataset(entry, key, value)

    if metadata.polarization is not None:
        _replace_cxi_dataset(entry, CXIFileKeys.POLARIZATION, str(metadata.polarization))

    if metadata.beam_center is not None:
        entry.create_dataset(CXIFileKeys.BEAM_CENTER_X, data=metadata.beam_center.x_px)
        entry.create_dataset(CXIFileKeys.BEAM_CENTER_Y, data=metadata.beam_center.y_px)


def _write_cxi_process(group: h5py.Group) -> h5py.Group:
    try:
        ptychodus_version = version('ptychodus')
    except PackageNotFoundError:
        ptychodus_version = 'unknown'

    process = group.require_group(CXIFileKeys.IMAGE_PROCESS)
    process.create_dataset(CXIFileKeys.PROCESS_PROGRAM, data='ptychodus')
    process.create_dataset(CXIFileKeys.PROCESS_VERSION, data=ptychodus_version)
    process.create_dataset(
        CXIFileKeys.PROCESS_DATE, data=datetime.now(timezone.utc).isoformat(timespec='seconds')
    )
    return process


def _write_cxi_image(
    entry: h5py.Group,
    name: str,
    *,
    title: str,
    array: NumberArrayType,
    axis_name: str,
    axis_values: NumberArrayType,
    pixel_geometry: PixelGeometry | None,
) -> h5py.Dataset:
    """Write one image group: 2-D when `array` holds one plane, else with a named axis."""
    group = entry.create_group(name)
    group.create_dataset(CXIFileKeys.TITLE, data=title)

    if array.shape[0] == 1:
        data = group.create_dataset(CXIFileKeys.IMAGE_DATA, data=array[0])
    else:
        data = group.create_dataset(CXIFileKeys.IMAGE_DATA, data=array)
        data.attrs['axes'] = f'{axis_name}:y:x'
        group.create_dataset(axis_name, data=axis_values)

    group.create_dataset(CXIFileKeys.IMAGE_DATA_TYPE, data='amplitude')
    group.create_dataset(CXIFileKeys.IMAGE_DATA_SPACE, data='real')
    group.create_dataset(CXIFileKeys.IMAGE_DIMENSIONALITY, data=2)

    if pixel_geometry is not None:
        height_px, width_px = array.shape[-2:]
        group.create_dataset(
            CXIFileKeys.IMAGE_SIZE,
            data=(width_px * pixel_geometry.width_m, height_px * pixel_geometry.height_m, 0.0),
        )

    _write_cxi_process(group)

    for member, target in (
        ('source_1', CXIFileKeys.SOURCE),
        ('detector_1', CXIFileKeys.DETECTOR),
    ):
        if target in entry:
            group[member] = h5py.SoftLink(entry[target].name)

    return data


def _write_cxi_product(
    entry: h5py.Group, product: Product, variant: CXIVariant, *, has_patterns: bool
) -> None:
    metadata = product.metadata

    if metadata.name:
        _replace_cxi_dataset(entry, CXIFileKeys.TITLE, metadata.name)
        _replace_cxi_dataset(entry, CXIFileKeys.SAMPLE_NAME, metadata.name)

    if metadata.comments:
        _replace_cxi_dataset(entry, CXIFileKeys.EXPERIMENT_DESCRIPTION, metadata.comments)

    _replace_cxi_dataset(entry, CXIFileKeys.DETECTOR_DISTANCE, metadata.detector_distance_m)
    _write_cxi_energy(entry, metadata.photon_energy_eV, variant)

    positions = product.probe_positions
    translation = numpy.zeros((len(positions), 3))
    translation[:, 0] = positions.get_coordinates_x_m()
    translation[:, 1] = positions.get_coordinates_y_m()
    translation_dataset = entry.create_dataset(CXIFileKeys.SAMPLE_TRANSLATION, data=translation)
    entry.create_dataset(
        CXIFileKeys.PROBE_POSITION_INDEXES, data=positions.get_indexes().astype(numpy.int64)
    )
    photon_counts = positions.get_probe_photon_counts()

    if photon_counts is not None:
        entry.create_dataset(CXIFileKeys.PROBE_PHOTON_COUNTS, data=photon_counts)

    if has_patterns:
        entry[CXIFileKeys.DATA_1_TRANSLATION] = h5py.SoftLink(translation_dataset.name)
        entry[CXIFileKeys.DETECTOR_TRANSLATION] = h5py.SoftLink(translation_dataset.name)

    probes = product.probes
    object_ = product.object_
    probe_array = probes.get_array()
    object_array = object_.get_array()
    probe_pixel_geometry = _get_or_none(probes.get_pixel_geometry)
    object_pixel_geometry = _get_or_none(object_.get_pixel_geometry)
    has_probe = probes.width_px > 0 and probes.height_px > 0
    has_object = object_.width_px > 0 and object_.height_px > 0
    primary_image: h5py.Dataset | None = None

    if variant == CXIVariant.SHARP:
        dropped = (has_probe and probe_array.shape[:2] != (1, 1)) or (
            has_object and object_array.shape[0] != 1
        )

        if dropped:
            logger.warning(
                'The SHARP layout holds one probe mode and one object layer; the rest are'
                ' kept only in the ptychodus group.'
            )

        if has_object:
            image = entry.create_group(CXIFileKeys.IMAGE_1)
            primary_image = image.create_dataset(CXIFileKeys.IMAGE_DATA, data=object_array[0])
            _write_cxi_process(image)

            if has_probe:
                entry.create_dataset(CXIFileKeys.FINAL_ILLUMINATION, data=probe_array[0, 0])

        if has_probe:
            _replace_cxi_dataset(entry, CXIFileKeys.SOURCE_ILLUMINATION, probe_array[0, 0])

        if dropped and has_probe:
            entry.create_dataset(CXIFileKeys.PROBE, data=probe_array)

        if dropped and has_object:
            entry.create_dataset(CXIFileKeys.OBJECT, data=object_array)
    else:
        image_names = iter((CXIFileKeys.IMAGE_1, CXIFileKeys.IMAGE_2))

        if has_object:
            layer_offsets = numpy.concatenate(([0.0], numpy.cumsum(object_.layer_spacing_m)))
            primary_image = _write_cxi_image(
                entry,
                next(image_names),
                title='object',
                array=object_array,
                axis_name=CXIFileKeys.IMAGE_LAYER,
                axis_values=layer_offsets,
                pixel_geometry=object_pixel_geometry,
            )

        if has_probe:
            probe_image = _write_cxi_image(
                entry,
                next(image_names),
                title='probe',
                array=probe_array[0],
                axis_name=CXIFileKeys.IMAGE_MODE,
                axis_values=numpy.arange(probe_array.shape[1]),
                pixel_geometry=probe_pixel_geometry,
            )
            primary_image = probe_image if primary_image is None else primary_image

            if probe_array.shape[0] > 1:
                entry.create_dataset(CXIFileKeys.PROBE_OPR_MODES, data=probe_array[1:])

    # Every entry needs a Data class: the patterns are data_1 when present, and the
    # reconstruction follows; with neither image, the translations stand in.
    if primary_image is not None:
        data_key = CXIFileKeys.DATA_2_DATA if has_patterns else CXIFileKeys.DATA_1_DATA
        entry[data_key] = h5py.SoftLink(primary_image.name)
    elif not has_patterns:
        entry[CXIFileKeys.DATA_1_DATA] = h5py.SoftLink(translation_dataset.name)

    opr_weights = probes.get_opr_weights_or_none()

    if opr_weights is not None:
        entry.create_dataset(CXIFileKeys.OPR_WEIGHTS, data=opr_weights)

    if probe_pixel_geometry is not None:
        entry.create_dataset(CXIFileKeys.PROBE_PIXEL_WIDTH, data=probe_pixel_geometry.width_m)
        entry.create_dataset(CXIFileKeys.PROBE_PIXEL_HEIGHT, data=probe_pixel_geometry.height_m)

    if object_pixel_geometry is not None:
        entry.create_dataset(CXIFileKeys.OBJECT_PIXEL_WIDTH, data=object_pixel_geometry.width_m)
        entry.create_dataset(CXIFileKeys.OBJECT_PIXEL_HEIGHT, data=object_pixel_geometry.height_m)

    object_center = _get_or_none(object_.get_center)

    if object_center is not None:
        entry.create_dataset(CXIFileKeys.OBJECT_CENTER_X, data=object_center.x_m)
        entry.create_dataset(CXIFileKeys.OBJECT_CENTER_Y, data=object_center.y_m)

    if object_.layer_spacing_m:
        entry.create_dataset(
            CXIFileKeys.OBJECT_LAYER_SPACING,
            data=numpy.asarray(object_.layer_spacing_m, dtype=numpy.float64),
        )

    if product.losses:
        entry.create_dataset(
            CXIFileKeys.LOSS_EPOCHS,
            data=numpy.array([loss.epoch for loss in product.losses], dtype=numpy.int64),
        )
        entry.create_dataset(
            CXIFileKeys.LOSS_VALUES,
            data=numpy.array([loss.value for loss in product.losses], dtype=numpy.float64),
        )

    for key, value in (
        (CXIFileKeys.PROBE_PHOTON_COUNT, metadata.probe_photon_count),
        (CXIFileKeys.EXPOSURE_TIME, metadata.exposure_time_s),
        (CXIFileKeys.MASS_ATTENUATION, metadata.mass_attenuation_m2_per_kg),
        (CXIFileKeys.TOMOGRAPHY_ANGLE, metadata.tomography_angle_deg),
        (CXIFileKeys.TILT_ANGLE, metadata.tilt_angle_deg),
        (CXIFileKeys.FOCUS_OBJECT_DISTANCE, metadata.focus_object_distance_m),
        (CXIFileKeys.FAR_FIELD, metadata.far_field),
    ):
        _replace_cxi_dataset(entry, key, value)

    if metadata.polarization is None:
        if CXIFileKeys.POLARIZATION in entry:
            del entry[CXIFileKeys.POLARIZATION]
    else:
        _replace_cxi_dataset(entry, CXIFileKeys.POLARIZATION, str(metadata.polarization))


def _write_cxi_file(
    file_path: Path,
    *,
    diffraction: _CXIDiffractionPayload | None,
    product: Product | None,
    variant: CXIVariant,
    compression: str | None,
) -> None:
    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset(CXIFileKeys.CXI_VERSION, data=160)
        h5_file.create_dataset(CXIFileKeys.NUMBER_OF_ENTRIES, data=1)
        entry = h5_file.create_group('entry_1')
        entry.create_dataset(CXIFileKeys.PROGRAM_NAME, data='ptychodus')
        extension = entry.create_group(CXIFileKeys.PTYCHODUS)
        extension.attrs[CXIFileKeys.LAYOUT_VERSION] = 1
        extension.attrs[CXIFileKeys.VARIANT] = str(variant)

        if diffraction is not None:
            _write_cxi_diffraction(entry, diffraction, variant, compression)

        if product is not None:
            _write_cxi_product(entry, product, variant, has_patterns=diffraction is not None)


def save_cxi_diffraction(
    file_path: Path,
    dataset: DiffractionDataset,
    *,
    variant: CXIVariant = CXIVariant.STANDARD,
    compression: str | None = 'gzip',
) -> None:
    """Write a diffraction dataset to a CXI file, one array after another.

    The mask flags bad pixels as invalid; the dataset's background and flat field,
    when present, are written as ``data_dark`` and as a flat-field response. gzip is
    the default because every HDF5 build can read it.
    """
    payload = _CXIDiffractionPayload(
        metadata=dataset.get_metadata(),
        blocks=((array.get_indexes(), array.get_patterns()) for array in dataset),
        bad_pixels=dataset.get_bad_pixels(),
        bad_pixel_flags=int(CXIPixelMask.INVALID),
        background=dataset.get_background(),
        flat_field=dataset.get_flat_field(),
    )
    _write_cxi_file(
        file_path, diffraction=payload, product=None, variant=variant, compression=compression
    )


def save_cxi_product(
    file_path: Path, product: Product, *, variant: CXIVariant = CXIVariant.STANDARD
) -> None:
    """Write a data product to a CXI file.

    Translations are written unchanged as CXI sample translations; the module
    docstring explains why no sign change is needed.
    """
    _write_cxi_file(file_path, diffraction=None, product=product, variant=variant, compression=None)


def _derive_detector_pixel_geometry(product: Product, extent: ImageExtent) -> PixelGeometry | None:
    """The detector pitch the product's sample-plane pitch implies, or None if unknown."""
    metadata = product.metadata

    try:
        if metadata.far_field:
            if metadata.photon_energy_eV <= 0.0 or metadata.detector_distance_m <= 0.0:
                return None

            # The Fraunhofer relation is its own inverse.
            return compute_far_field_pixel_geometry(
                product.probes.get_pixel_geometry(),
                extent,
                wavelength_m=metadata.photon_wavelength_m,
                propagation_distance_m=metadata.detector_distance_m,
            )

        object_pixel_geometry = product.object_.get_pixel_geometry()
        return PixelGeometry(
            width_m=object_pixel_geometry.width_m * metadata.magnification,
            height_m=object_pixel_geometry.height_m * metadata.magnification,
        )
    except (ValueError, ZeroDivisionError):
        return None


def save_cxi_reconstruct_input(
    file_path: Path,
    parameters: ReconstructInput,
    *,
    variant: CXIVariant = CXIVariant.STANDARD,
    inpaint_bad_pixels: bool = False,
    compression: str | None = 'gzip',
) -> None:
    """Write everything a reconstruction starts from to one CXI file.

    Patterns, mask, detector maps and the initial product share one entry. Bad pixels
    are zeroed and flagged invalid; with `inpaint_bad_pixels` they are filled instead
    and also flagged as repaired. The detector pitch is derived from the product's
    sample-plane pitch, since the patterns no longer record their own.
    """
    patterns, bad_pixels = _repair_bad_pixels(parameters, inpaint=inpaint_bad_pixels)
    product = parameters.product
    num_patterns, height_px, width_px = patterns.shape
    extent = ImageExtent(width_px=width_px, height_px=height_px)
    indexes = product.probe_positions.get_indexes()

    if len(indexes) != num_patterns:
        indexes = numpy.arange(num_patterns)

    flags = CXIPixelMask.INVALID

    if inpaint_bad_pixels:
        flags |= CXIPixelMask.REPAIRED

    payload = _CXIDiffractionPayload(
        metadata=DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=patterns.dtype,
            detector_extent=extent,
            detector_distance_m=product.metadata.detector_distance_m,
            detector_pixel_geometry=_derive_detector_pixel_geometry(product, extent),
            photon_energy_eV=product.metadata.photon_energy_eV,
        ),
        blocks=[(indexes, patterns)],
        bad_pixels=bad_pixels,
        bad_pixel_flags=int(flags),
        background=parameters.background,
        flat_field=parameters.flat_field,
    )
    _write_cxi_file(
        file_path, diffraction=payload, product=product, variant=variant, compression=compression
    )
