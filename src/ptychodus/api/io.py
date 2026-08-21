"""I/O utilities for assembled diffraction data and data products."""

from __future__ import annotations
from enum import StrEnum
from pathlib import Path
import logging
import re

import h5py
import numpy

from .diffraction import AssembledDiffractionData, Polarization, zero_bad_pixels
from .fluorescence import ElementMap, FluorescenceDataset
from .geometry import PixelGeometry
from .object import Object, ObjectCenter
from .probe import ProbeSequence
from .probe_positions import ProbePositionSequence, ProbePosition
from .product import LossValue, Product, ProductMetadata
from .reconstruct import ReconstructInput

__all__ = [
    'FluorescenceFileKeys',
    'StandardFileLayout',
    'load_diffraction_data',
    'load_fluorescence_data',
    'load_product',
    'save_diffraction_data',
    'save_fluorescence_data',
    'save_product',
    'save_ptychopinn_training_data',
    'sanitize_path_component',
    'resolve_external_link_path',
]

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
    FLUORESCENCE_IN = 'fluorescence-in.h5'
    FLUORESCENCE_OUT = 'fluorescence-out.h5'
    # Stem only; the per-backend extension comes from
    # TrainableReconstructor.get_model_file_extension(). The full path is
    # f'{input_directory}/{MODEL_BASENAME}{ext}'.
    MODEL_BASENAME = 'model'
    PRODUCT_IN = 'product-in.h5'
    PRODUCT_OUT = 'product-out.h5'
    SETTINGS = 'settings.ini'


class DiffractionFileKeys(StrEnum):
    """HDF5 dataset and attribute names for the assembled diffraction data file."""

    PATTERNS = 'patterns'
    DETECTOR_PIXEL_HEIGHT = 'detector_pixel_height_m'
    DETECTOR_PIXEL_WIDTH = 'detector_pixel_width_m'
    INDEXES = 'indexes'
    BAD_PIXELS = 'bad_pixels'


def load_diffraction_data(file: Path, *, mmap_file: Path | None = None) -> AssembledDiffractionData:
    """Load assembled diffraction data from an HDF5 file written by :func:`save_diffraction_data`."""
    if mmap_file is not None:
        raise NotImplementedError('Load to memory map not implemented yet!')

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

        return AssembledDiffractionData(
            h5_indexes[()],
            h5_patterns[()],
            detector_pixel_geometry,
            h5_bad_pixels[()],
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


class ProductFileKeys(StrEnum):
    """HDF5 dataset and attribute names for the product file."""

    NAME = 'name'
    COMMENTS = 'comments'
    DETECTOR_OBJECT_DISTANCE = 'detector_object_distance_m'
    PROBE_ENERGY = 'probe_energy_eV'
    PROBE_PHOTON_COUNT = 'probe_photon_count'
    EXPOSURE_TIME = 'exposure_time_s'
    MASS_ATTENUATION = 'mass_attenuation_m2_kg'
    TOMOGRAPHY_ANGLE = 'tomography_angle_deg'
    TILT_ANGLE = 'tilt_angle_deg'
    POLARIZATION = 'polarization'
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
    """Load a data product from an HDF5 file written by :func:`save_product`."""
    point_list: list[ProbePosition] = []

    with h5py.File(file, 'r') as h5_file:
        name = str(h5_file.attrs.get(ProductFileKeys.NAME, 'Unnamed'))
        comments = str(h5_file.attrs.get(ProductFileKeys.COMMENTS, ''))
        probe_photon_count = int(h5_file.attrs.get(ProductFileKeys.PROBE_PHOTON_COUNT, 0))
        exposure_time_s = float(h5_file.attrs.get(ProductFileKeys.EXPOSURE_TIME, 0.0))
        mass_attenuation_m2_kg = float(h5_file.attrs.get(ProductFileKeys.MASS_ATTENUATION, 0.0))
        tomography_angle_deg = float(h5_file.attrs.get(ProductFileKeys.TOMOGRAPHY_ANGLE, 0.0))
        tilt_angle_deg = float(h5_file.attrs.get(ProductFileKeys.TILT_ANGLE, 0.0))

        polarization: Polarization | None = None
        if ProductFileKeys.POLARIZATION in h5_file.attrs:
            raw_polarization = h5_file.attrs[ProductFileKeys.POLARIZATION]
            if isinstance(raw_polarization, bytes):
                raw_polarization = raw_polarization.decode('utf-8', errors='replace')
            try:
                polarization = Polarization(str(raw_polarization))
            except ValueError:
                logger.warning(
                    'Unknown polarization %r in %s; setting polarization=None.',
                    raw_polarization,
                    file,
                )

        metadata = ProductMetadata(
            name=name,
            comments=comments,
            detector_distance_m=float(h5_file.attrs[ProductFileKeys.DETECTOR_OBJECT_DISTANCE]),
            probe_energy_eV=float(h5_file.attrs[ProductFileKeys.PROBE_ENERGY]),
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
            tomography_angle_deg=tomography_angle_deg,
            tilt_angle_deg=tilt_angle_deg,
            polarization=polarization,
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
    """Write a data product to an HDF5 file."""
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
        h5_file.attrs[ProductFileKeys.TOMOGRAPHY_ANGLE] = metadata.tomography_angle_deg
        h5_file.attrs[ProductFileKeys.TILT_ANGLE] = metadata.tilt_angle_deg
        if metadata.polarization is not None:
            h5_file.attrs[ProductFileKeys.POLARIZATION] = metadata.polarization.value

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
        element_maps=element_maps,
        counts_per_second_path=counts_per_second_path,
        channel_names_path=channel_names_path,
    )


def save_fluorescence_data(file: Path, dataset: FluorescenceDataset) -> None:
    """Write a fluorescence dataset to an HDF5 file at the paths the dataset carries."""
    counts_group_path, counts_ds_name = _split_h5_path(dataset.counts_per_second_path)
    names_group_path, names_ds_name = _split_h5_path(dataset.channel_names_path)

    channel_names = [emap.name for emap in dataset.element_maps]
    counts_per_second = [emap.counts_per_second for emap in dataset.element_maps]

    with h5py.File(file, 'w') as h5_file:
        counts_group = h5_file.require_group(counts_group_path)
        counts_group.create_dataset(counts_ds_name, data=numpy.stack(counts_per_second))
        names_group = h5_file.require_group(names_group_path)
        names_group.create_dataset(names_ds_name, data=channel_names, dtype='S256')


def save_ptychopinn_training_data(
    file_path: Path,
    parameters: ReconstructInput,
    *,
    multimodal_probe: bool,
) -> None:
    """Write a ReconstructInput to the NPZ format consumed by PtychoPINN trainers.

    Bad pixels are zeroed in `diff3d` before writing. `multimodal_probe`
    selects the `probeGuess` shape: True writes the full `(N_modes, H, W)`
    array (ptycho_torch); False writes only mode 0 as `(H, W)` (legacy
    ptycho). All scan points are assigned `scan_index=0` (single-object
    assumption).
    """
    object_geometry = parameters.product.object_.get_geometry()
    position_x_px: list[float] = list()
    position_y_px: list[float] = list()

    for scan_point in parameters.product.probe_positions:
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        position_x_px.append(object_point.coordinate_x_px)
        position_y_px.append(object_point.coordinate_y_px)

    xcoords = numpy.array(position_x_px)
    ycoords = numpy.array(position_y_px)
    diff3d = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)

    probe = parameters.product.probes.get_probe_no_opr()
    probe_array = probe.get_array() if multimodal_probe else probe.get_incoherent_mode(0)

    numpy.savez(
        file_path,
        xcoords=xcoords,
        ycoords=ycoords,
        xcoords_start=xcoords,
        ycoords_start=ycoords,
        diff3d=diff3d,
        probeGuess=probe_array,
        objectGuess=parameters.product.object_.get_layer(0),
        scan_index=numpy.zeros(len(parameters.product.probe_positions), dtype=int),
    )
