"""Readers for NSLS-II 3-ID Hard X-ray Nanoprobe (HXN) files.

The HDF5 readers sense which of two layouts a file holds from the datasets present:

- Style 2, ``scan_<sid>.h5``, written for the beamline's ptychography software:
  ``/diffamp`` (n, height, width) holds the square root of each frame's counts, every
  frame FFT-shifted so that zero frequency sits in its corners; ``/points`` (2, n) the
  positions in micrometers, row 0 x (its range and step match ``x_range`` and
  ``dr_x``); ``/ic`` the I0 per point; and ``lambda_nm``, ``z_m``, ``ccd_pixel_um`` and
  ``angle``. The patterns are squared back to counts and shifted to center.
- Style 1, ``scan_<sid>_<det>.h5``, exported from the raw data: frames at
  ``/diff_data/<det>/det_images``, or where the areaDetector HDF5 plugin wrote them,
  ``/entry/data/data``; ``/scan/scan_positions`` (2, n) in micrometers, row 0 x (its
  range and step match ``scan_start1``, ``scan_end1`` and ``num1``, with ``motors[0]``
  the x stage); ``/diff_data/<det>/Io``, the I0 as a (num2, num1) map in scan order;
  ``/scan/energy`` in keV, ``/scan/exposure_time``, and the sample rotation of each
  microscope, ``/scan/zp_theta`` and ``/scan/mll_theta``. No detector distance or pitch
  is recorded; the pitch follows from the detector's name.

The MATLAB layout (``*.mat``, version 7.3) is read separately: ``det_data`` holds
centered counts, ``energy`` the photon energy in keV, ``det_dist`` and ``det_pixel_size``
lengths in micrometers, ``mask`` the usable pixels, ``th`` the sample rotation,
``pos_xy`` (frame, x, y) the positions in micrometers, and ``obj`` and ``prb`` a
reconstruction on ``img_pixel_size_x`` by ``img_pixel_size_y`` micrometer pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.constants import EnergyUnit, LengthUnit, wavelength_m_to_energy_eV
from ptychodus.api.diffraction import (
    BadPixels,
    CropRegion,
    DiffractionArray,
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionFileReader,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPatterns,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import (
    ProbePosition,
    ProbePositionFileReader,
    ProbePositionParseError,
    ProbePositionSequence,
)
from ptychodus.api.product import LossValue, Product, ProductFileReader, ProductMetadata

from .h5_diffraction_file import (
    H5DiffractionFileTreeBuilder,
    H5DiffractionPatternArray,
    warn_on_dropped_frames,
)

logger = logging.getLogger(__name__)

_DESCRIPTION: Final = 'NSLS-II 3-ID HXN Files'
_MATLAB_DESCRIPTION: Final = 'NSLS-II 3-ID HXN MATLAB Files'
_METERS_PER_MICROMETER: Final = LengthUnit.MICROMETER.meters_per_unit


def _read_scalar(h5_file: h5py.File, path: str, *, scale: float = 1.0) -> float | None:
    """The scalar at `path` times `scale`, or None when the file does not record one."""
    item = h5_file.get(path)

    if not isinstance(item, h5py.Dataset) or item.size != 1:
        logger.debug(f'"{h5_file.filename}" records no {path}.')
        return None

    value = float(numpy.squeeze(item[()])) * scale
    logger.debug(f'{path} = {value} from "{h5_file.filename}".')
    return value


def _detector_name(h5_file: h5py.File) -> str | None:
    """The detector a Style 1 file was exported from: its group under ``/diff_data``."""
    group = h5_file.get('/diff_data')

    if isinstance(group, h5py.Group):
        for name, item in group.items():
            if isinstance(item, h5py.Group):
                return str(name)

    return None


def _detector_pixel_size_m(detector_name: str | None) -> float | None:
    """The pitch of an HXN detector, by the name the beamline gives it."""
    if detector_name is None:
        return None

    for prefix, pixel_size_m in (('merlin', 55e-6), ('eiger', 75e-6)):
        if detector_name.casefold().startswith(prefix):
            return pixel_size_m

    logger.debug(f'No pixel pitch is known for detector "{detector_name}".')
    return None


def _style1_rotation_path(h5_file: h5py.File) -> str:
    """The sample rotation of the microscope whose stages drove the scan."""
    motors = h5_file.get('/scan/motors')

    if isinstance(motors, h5py.Dataset) and motors.size > 0:
        first = numpy.ravel(motors[()])[0]
        name = first.decode() if isinstance(first, bytes) else str(first)

        if name.casefold().startswith('zp'):
            return '/scan/zp_theta'

    return '/scan/mll_theta'


class _DiffampArray(DiffractionArray):
    """A block of Style 2 frames, squared back to counts and shifted to center.

    `integral` says the squared amplitudes are whole counts, which are then returned as
    unsigned integers; otherwise they are returned as single-precision floats.
    """

    def __init__(
        self, file_path: Path, indexes: DiffractionIndexes, start: int, *, integral: bool
    ) -> None:
        super().__init__()
        self._file_path = file_path
        self._indexes = indexes
        self._start = start
        self._integral = integral

    def get_label(self) -> str:
        return f'{self._file_path.stem}[{self._start}:{self._start + self._indexes.size}]'

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_num_patterns(self) -> int:
        return self._indexes.size

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        with h5py.File(self._file_path, 'r') as h5_file:
            amplitudes = h5_file['/diffamp'][self._start : self._start + self._indexes.size]

        counts = numpy.fft.fftshift(numpy.square(amplitudes), axes=(-2, -1))
        patterns = (
            numpy.rint(counts).astype(numpy.uint32)
            if self._integral
            else counts.astype(numpy.float32)
        )

        return patterns if read_region is None else read_region.apply_to(patterns)


class NSLS2HXNDiffractionFileReader(DiffractionFileReader):
    """Patterns from a Style 1 or Style 2 HXN file, whichever the file holds.

    `diffamp_block_size` is the number of Style 2 frames read at once, which bounds the
    memory a read takes: those files store float64 amplitudes. `integral_tolerance` is
    how far a squared amplitude may sit from a whole count for the frames to be read as
    counts.
    """

    def __init__(self, *, diffamp_block_size: int = 1024, integral_tolerance: float = 1e-6) -> None:
        self._diffamp_block_size = diffamp_block_size
        self._integral_tolerance = integral_tolerance

    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = H5DiffractionFileTreeBuilder().build(h5_file)
            diffamp = h5_file.get('/diffamp')

            if isinstance(diffamp, h5py.Dataset):
                logger.info(f'Reading "{file_path}" as an HXN Style 2 file.')
                return self._read_style2(file_path, h5_file, diffamp, contents_tree)

            detector_name = _detector_name(h5_file)
            data_paths = ['/entry/data/data']

            if detector_name is not None:
                data_paths.insert(0, f'/diff_data/{detector_name}/det_images')

            for data_path in data_paths:
                data = h5_file.get(data_path)

                if isinstance(data, h5py.Dataset) and data.ndim == 3:
                    logger.info(f'Reading "{file_path}" as an HXN Style 1 file ({data_path}).')
                    warn_on_dropped_frames(h5_file, file_path)
                    return self._read_style1(file_path, h5_file, data, detector_name, contents_tree)

        tried = ', '.join(['/diffamp', '/diff_data/<detector>/det_images', '/entry/data/data'])
        raise ValueError(f'"{file_path}" holds no HXN diffraction patterns (tried {tried}).')

    def _read_style2(
        self,
        file_path: Path,
        h5_file: h5py.File,
        diffamp: h5py.Dataset,
        contents_tree: DiffractionDatasetLayoutNode,
    ) -> DiffractionDataset:
        num_patterns, detector_height, detector_width = diffamp.shape
        first_counts = numpy.square(diffamp[0])
        integral = bool(
            numpy.all(
                numpy.abs(first_counts - numpy.rint(first_counts)) <= self._integral_tolerance
            )
        )

        if not integral:
            logger.warning(
                f'"{file_path}" holds squared amplitudes that are not whole counts;'
                ' reading them as float32.'
            )

        array_list: list[DiffractionArray] = []
        num_patterns_per_array: list[int] = []

        for start in range(0, num_patterns, self._diffamp_block_size):
            stop = min(start + self._diffamp_block_size, num_patterns)
            indexes: DiffractionIndexes = numpy.arange(start, stop)
            array_list.append(_DiffampArray(file_path, indexes, start, integral=integral))
            num_patterns_per_array.append(stop - start)

        wavelength_m = _read_scalar(
            h5_file, '/lambda_nm', scale=LengthUnit.NANOMETER.meters_per_unit
        )
        pixel_size_m = _read_scalar(h5_file, '/ccd_pixel_um', scale=_METERS_PER_MICROMETER)
        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=numpy.dtype(numpy.uint32 if integral else numpy.float32),
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_distance_m=_read_scalar(h5_file, '/z_m'),
            detector_pixel_geometry=None
            if pixel_size_m is None
            else PixelGeometry(pixel_size_m, pixel_size_m),
            photon_energy_eV=None
            if wavelength_m is None
            else wavelength_m_to_energy_eV(wavelength_m),
            tomography_angle_deg=_read_scalar(h5_file, '/angle'),
            file_path=file_path,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, array_list)

    def _read_style1(
        self,
        file_path: Path,
        h5_file: h5py.File,
        data: h5py.Dataset,
        detector_name: str | None,
        contents_tree: DiffractionDatasetLayoutNode,
    ) -> DiffractionDataset:
        num_patterns, detector_height, detector_width = data.shape
        pixel_size_m = _detector_pixel_size_m(detector_name)
        array = H5DiffractionPatternArray(
            label=file_path.stem,
            indexes=numpy.arange(num_patterns),
            file_path=file_path,
            data_path=data.name,
        )
        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=data.dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_pixel_geometry=None
            if pixel_size_m is None
            else PixelGeometry(pixel_size_m, pixel_size_m),
            photon_energy_eV=_read_scalar(h5_file, '/scan/energy', scale=1000.0),
            exposure_time_s=_read_scalar(h5_file, '/scan/exposure_time'),
            tomography_angle_deg=_read_scalar(h5_file, _style1_rotation_path(h5_file)),
            file_path=file_path,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, [array])


class NSLS2HXNPositionFileReader(ProbePositionFileReader):
    """Positions and I0 from a Style 1 or Style 2 HXN file, whichever the file holds.

    `x_scale_to_meters` and `y_scale_to_meters` convert the recorded micrometers; a
    negative scale reverses an axis.
    """

    def __init__(
        self,
        *,
        x_scale_to_meters: float = _METERS_PER_MICROMETER,
        y_scale_to_meters: float = _METERS_PER_MICROMETER,
    ) -> None:
        self._x_scale_to_meters = x_scale_to_meters
        self._y_scale_to_meters = y_scale_to_meters

    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> ProbePositionSequence:
        with h5py.File(file_path, 'r') as h5_file:
            points = h5_file.get('/points')

            if isinstance(points, h5py.Dataset):
                logger.info(f'Reading "{file_path}" as an HXN Style 2 file.')
                positions_um = numpy.asarray(points[()], dtype=float)
                i0 = h5_file.get('/ic')
                i0_counts = numpy.ravel(i0[()]) if isinstance(i0, h5py.Dataset) else None
            else:
                points = h5_file.get('/scan/scan_positions')

                if not isinstance(points, h5py.Dataset):
                    raise ProbePositionParseError(
                        f'"{file_path}" holds no HXN positions'
                        ' (tried /points, /scan/scan_positions).'
                    )

                logger.info(f'Reading "{file_path}" as an HXN Style 1 file.')
                positions_um = numpy.asarray(points[()], dtype=float)
                detector_name = _detector_name(h5_file)
                i0 = (
                    None if detector_name is None else h5_file.get(f'/diff_data/{detector_name}/Io')
                )
                # The map is (num2, num1) with num1 the fast axis, so a C-order ravel
                # visits it in scan order.
                i0_counts = numpy.ravel(i0[()]) if isinstance(i0, h5py.Dataset) else None

        if positions_um.ndim != 2 or positions_um.shape[0] != 2:
            raise ProbePositionParseError(
                f'"{file_path}" positions have shape {positions_um.shape}; expected (2, N).'
            )

        num_points = positions_um.shape[1]

        if i0_counts is not None and i0_counts.size != num_points:
            logger.warning(
                f'"{file_path}" records {i0_counts.size} I0 values for {num_points}'
                ' positions; I0 is left unset.'
            )
            i0_counts = None

        return ProbePositionSequence(
            [
                ProbePosition(
                    index=index,
                    x_m=float(x_um) * self._x_scale_to_meters,
                    y_m=float(y_um) * self._y_scale_to_meters,
                    probe_photon_count=None if i0_counts is None else float(i0_counts[index]),
                )
                for index, (x_um, y_um) in enumerate(positions_um.T)
            ]
        )


class _MATLABPatternArray(H5DiffractionPatternArray):
    """MATLAB ``det_data`` frames, each transposed back to MATLAB's orientation.

    MATLAB stores arrays column-major and HDF5 row-major, so every 2-D array of a v7.3
    file reads back transposed. The read region is stated in MATLAB's orientation and is
    transposed into the stored one before the read.
    """

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        stored_region = (
            None
            if read_region is None
            else CropRegion(x_range=read_region.y_range, y_range=read_region.x_range)
        )
        return numpy.swapaxes(super().get_patterns(read_region=stored_region), -1, -2)


class NSLS2MATLABDiffractionFileReader(DiffractionFileReader):
    def get_file_description(self) -> str:
        return _MATLAB_DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.mat'}

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            data = h5_file['det_data']

            if not isinstance(data, h5py.Dataset):
                raise ValueError(f'Expected {data.name} to be a dataset; got {type(data)}.')

            # Stored transposed: the last axis is MATLAB's rows.
            num_patterns, detector_width, detector_height = data.shape
            pixel_size_m = _read_scalar(h5_file, 'det_pixel_size', scale=_METERS_PER_MICROMETER)
            photon_energy_keV = _read_scalar(h5_file, 'energy')  # noqa: N806
            bad_pixels: BadPixels | None = None
            mask = h5_file.get('mask')

            # MATLAB marks a usable pixel 1; the mask is stored transposed like the frames.
            if isinstance(mask, h5py.Dataset) and mask.shape == (detector_width, detector_height):
                bad_pixels = numpy.asarray(mask[()]).T == 0

            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=data.dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                detector_distance_m=_read_scalar(h5_file, 'det_dist', scale=_METERS_PER_MICROMETER),
                detector_pixel_geometry=None
                if pixel_size_m is None
                else PixelGeometry(pixel_size_m, pixel_size_m),
                photon_energy_eV=None
                if photon_energy_keV is None
                else EnergyUnit.KILOELECTRONVOLT.to_electronvolts(photon_energy_keV),
                tomography_angle_deg=_read_scalar(h5_file, 'th'),
                file_path=file_path,
            )
            contents_tree = H5DiffractionFileTreeBuilder().build(h5_file)
            array = _MATLABPatternArray(
                label=file_path.stem,
                indexes=numpy.arange(num_patterns),
                file_path=file_path,
                data_path=data.name,
            )

        return SimpleDiffractionDataset(metadata, contents_tree, [array], bad_pixels)


class NSLS2MATLABProductFileReader(ProductFileReader):
    def get_file_description(self) -> str:
        return _MATLAB_DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.mat'}

    def read(self, file_path: Path) -> Product:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            detector_distance_m = LengthUnit.MICROMETER.to_meters(
                float(numpy.squeeze(h5_file['det_dist'][()]))
            )
            photon_energy_eV = EnergyUnit.KILOELECTRONVOLT.to_electronvolts(  # noqa: N806
                float(numpy.squeeze(h5_file['energy'][()]))
            )
            tomography_angle_deg = _read_scalar(h5_file, 'th')

            metadata = ProductMetadata(
                name=file_path.stem,
                comments='',
                detector_distance_m=detector_distance_m,
                photon_energy_eV=photon_energy_eV,
                probe_photon_count=0.0,  # not included in file
                exposure_time_s=0.0,  # not included in file
                mass_attenuation_m2_per_kg=0.0,  # not included in file
                tomography_angle_deg=0.0 if tomography_angle_deg is None else tomography_angle_deg,
            )

            pixel_geometry = PixelGeometry(
                width_m=LengthUnit.MICROMETER.to_meters(
                    float(numpy.squeeze(h5_file['img_pixel_size_x'][()]))
                ),
                height_m=LengthUnit.MICROMETER.to_meters(
                    float(numpy.squeeze(h5_file['img_pixel_size_y'][()]))
                ),
            )
            positions_m = h5_file['pos_xy'][()].T * _METERS_PER_MICROMETER

            for index, _xy in enumerate(positions_m):
                point = ProbePosition(
                    index=index,
                    x_m=_xy[1],
                    y_m=_xy[2],
                )
                point_list.append(point)

            # Stored transposed, as every 2-D array of a v7.3 file is.
            probe_array = h5_file['prb'][()].T.astype(complex)
            probes = ProbeEnsemble(
                array=probe_array, opr_weights=None, pixel_geometry=pixel_geometry
            )

            object_array = h5_file['obj'][()].T.astype(complex)
            object_ = Object(
                array=object_array,
                pixel_geometry=pixel_geometry,
                center=None,
            )
            loss: list[LossValue] = list()

        return Product(
            metadata=metadata,
            probe_positions=ProbePositionSequence(point_list),
            probes=probes,
            object_=object_,
            losses=loss,
        )


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        NSLS2HXNDiffractionFileReader(),
        simple_name='NSLS_II_HXN',
    )
    registry.probe_position_file_readers.register_plugin(
        NSLS2HXNPositionFileReader(),
        simple_name='NSLS_II_HXN',
    )
    registry.diffraction_file_readers.register_plugin(
        NSLS2MATLABDiffractionFileReader(),
        simple_name='NSLS_II_MATLAB',
    )
    registry.register_product_file_reader_with_adapters(
        NSLS2MATLABProductFileReader(),
        simple_name='NSLS_II_MATLAB',
    )
