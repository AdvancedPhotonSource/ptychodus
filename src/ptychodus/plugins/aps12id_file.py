"""Readers for APS 12-ID-E Ptycho-SAXS scans in the layout written from 2026_2 on.

A scan directory ``ptycho/S<scan>/`` holds a NeXus master file
``<sample>_<scan>_master.h5`` beside the frame files ``<sample>_<scan>_<NNNNN>.h5``. A fly
scan writes one file holding a stack of frames; a step scan writes one file per point, each
holding a single 2-D frame. Either way the frames are at ``/entry/data/data`` and run in
scan traversal order across the files taken in numeric order.

The master carries the geometry and the nominal scan positions. Energy is recorded in keV
and every length in millimeters; both are converted here. ``/entry/sample/positions`` lists
one (horizontal, vertical) row per scan point in traversal order, so row ``i`` pairs with
frame ``i``. A sibling ``positions/S<scan>/`` directory holds the raw positioner stream,
which is not calibrated and is not read.

The Pilatus writes ``-1`` into its inter-module gaps and ``-2`` into dead pixels, so any
negative value in the first frame marks a bad pixel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import logging
import re

import h5py
import numpy

from ptychodus.api.constants import LengthUnit
from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePosition,
    ProbePositionFileReader,
    ProbePositionParseError,
    ProbePositionSequence,
)

from .h5_diffraction_file import H5DiffractionFileTreeBuilder, H5DiffractionPatternArray

logger = logging.getLogger(__name__)

_DESCRIPTION: Final = 'APS 12-ID-E Ptycho-SAXS Files'
_MASTER_SUFFIX: Final = '_master'
_METERS_PER_MILLIMETER: Final = LengthUnit.MILLIMETER.meters_per_unit


def _find_master(file_path: Path) -> Path:
    """The master file of the scan `file_path` belongs to.

    Accepts the master itself, any frame file, or a file of the scan's raw positioner
    directory, ``positions/S<scan>/``, whose master is in the sibling ``ptycho/S<scan>/``.
    """
    stem = file_path.stem

    if stem.endswith(_MASTER_SUFFIX):
        if file_path.is_file():
            return file_path
        raise ValueError(f'"{file_path}" does not exist.')

    match = re.fullmatch(r'(?P<prefix>.+)_\d+', stem)

    if match is None:
        raise ValueError(
            f'"{file_path.name}" is not named <sample>_<scan>_<number>{file_path.suffix}'
            f' or <sample>_<scan>{_MASTER_SUFFIX}{file_path.suffix}.'
        )

    master_name = f'{match["prefix"]}{_MASTER_SUFFIX}{file_path.suffix}'
    scan_directory = file_path.parent
    candidates = [
        scan_directory / master_name,
        scan_directory.parent.parent / 'ptycho' / scan_directory.name / master_name,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise ValueError(
        f'No master file for "{file_path}" at "{candidates[0]}"; this is not a 12-ID-E scan'
        ' in the layout written from 2026_2 on.'
    )


def _frame_files(master: Path) -> list[Path]:
    """Every frame file beside `master`, in numeric order."""
    prefix = master.stem.removesuffix(_MASTER_SUFFIX)
    pattern = re.compile(rf'{re.escape(prefix)}_(\d+)')
    numbered: list[tuple[int, Path]] = []

    for candidate in master.parent.glob(f'{prefix}_*{master.suffix}'):
        match = pattern.fullmatch(candidate.stem)

        if match is not None:
            numbered.append((int(match.group(1)), candidate))

    return [path for _, path in sorted(numbered)]


def _read_scalar(h5_file: h5py.File, path: str, *, scale: float = 1.0) -> float | None:
    """The scalar at `path` times `scale`, or None when the file does not record it."""
    try:
        item = h5_file[path]
    except KeyError:
        logger.debug(f'"{path}" not found in "{h5_file.filename}".')
        return None

    if not isinstance(item, h5py.Dataset):
        logger.debug(f'"{path}" in "{h5_file.filename}" is not a dataset.')
        return None

    return float(numpy.atleast_1d(item[()])[0]) * scale


def _read_string(h5_file: h5py.File, path: str) -> str | None:
    """The string at `path`, or None when the file does not record a usable one."""
    try:
        item = h5_file[path]
    except KeyError:
        logger.debug(f'"{path}" not found in "{h5_file.filename}".')
        return None

    if not isinstance(item, h5py.Dataset):
        logger.debug(f'"{path}" in "{h5_file.filename}" is not a dataset.')
        return None

    value = numpy.atleast_1d(item[()])[0]

    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')

    return str(value).strip() or None


class APS12IDDiffractionFileReader(DiffractionFileReader):
    # The master records the pitch; this is the Pilatus 2M pitch, for a master that does not.
    DETECTOR_PIXEL_SIZE_M: Final[float] = 172e-6
    DATA_PATH: Final[str] = '/entry/data/data'

    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> DiffractionDataset:
        master = _find_master(file_path)
        array_list: list[DiffractionArray] = []
        num_patterns_per_array: list[int] = []
        bad_pixels: BadPixels | None = None
        pattern_dtype = numpy.dtype(numpy.int32)
        detector_height = 0
        detector_width = 0
        offset = 0

        for frame_file in _frame_files(master):
            try:
                with h5py.File(frame_file, 'r') as h5_file:
                    dataset = h5_file[self.DATA_PATH]

                    if not isinstance(dataset, h5py.Dataset) or dataset.ndim not in (2, 3):
                        raise ValueError(f'"{self.DATA_PATH}" is not a 2-D or 3-D dataset')

                    *leading, detector_height, detector_width = dataset.shape
                    num_patterns = leading[0] if leading else 1
                    pattern_dtype = dataset.dtype

                    if bad_pixels is None:
                        first_frame = dataset[0] if leading else dataset[()]
                        bad_pixels = numpy.asarray(first_frame) < 0
            except (OSError, KeyError, ValueError, IndexError) as exc:
                # Pairing is by traversal order, so a skipped file would shift every later
                # frame onto the wrong position. Stopping keeps what precedes it correct.
                logger.warning(f'Ending the scan before unreadable "{frame_file}": {exc}')
                break

            indexes = numpy.arange(num_patterns) + offset
            offset += num_patterns
            num_patterns_per_array.append(num_patterns)
            array_list.append(
                H5DiffractionPatternArray(frame_file.stem, indexes, frame_file, self.DATA_PATH)
            )

        if not array_list:
            raise ValueError(f'No readable frame files beside "{master}".')

        with h5py.File(master, 'r') as h5_file:
            contents_tree = H5DiffractionFileTreeBuilder().build(h5_file)
            photon_energy_keV = _read_scalar(  # noqa: N806
                h5_file, '/entry/instrument/monochromator/monoE'
            )
            exposure_time_s = _read_scalar(h5_file, '/entry/instrument/detector/ExposureTime')
            detector_distance_m = _read_scalar(
                h5_file, '/entry/instrument/detector/distance', scale=_METERS_PER_MILLIMETER
            )
            pixel_width_m = _read_scalar(
                h5_file, '/entry/instrument/detector/x_pixel_size', scale=_METERS_PER_MILLIMETER
            )
            pixel_height_m = _read_scalar(
                h5_file, '/entry/instrument/detector/y_pixel_size', scale=_METERS_PER_MILLIMETER
            )
            tomography_angle_deg = _read_scalar(h5_file, '/entry/sample/theta')
            # The instrument writes the sample name here, not at the NeXus
            # '/entry/sample/name'; a master that predates the field reads as None.
            sample_name = _read_string(h5_file, '/entry/sample/sample_name')

        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=pattern_dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_distance_m=detector_distance_m,
            detector_pixel_geometry=PixelGeometry(
                width_m=self.DETECTOR_PIXEL_SIZE_M if pixel_width_m is None else pixel_width_m,
                height_m=self.DETECTOR_PIXEL_SIZE_M if pixel_height_m is None else pixel_height_m,
            ),
            photon_energy_eV=None if photon_energy_keV is None else 1000.0 * photon_energy_keV,
            exposure_time_s=exposure_time_s,
            tomography_angle_deg=tomography_angle_deg,
            sample_name=sample_name,
            file_path=master,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, array_list, bad_pixels)


class APS12IDPositionFileReader(ProbePositionFileReader):
    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> ProbePositionSequence:
        positions_path = '/entry/sample/positions'

        try:
            master = _find_master(file_path)
        except ValueError as exc:
            raise ProbePositionParseError(str(exc)) from None

        with h5py.File(master, 'r') as h5_file:
            try:
                dataset = h5_file[positions_path]
            except KeyError:
                raise ProbePositionParseError(
                    f'"{master}" records no scan positions at "{positions_path}".'
                ) from None

            positions_mm = numpy.asarray(dataset[()], dtype=float)

        if positions_mm.ndim != 2 or positions_mm.shape[1] != 2:
            raise ProbePositionParseError(
                f'"{positions_path}" in "{master}" has shape {positions_mm.shape}; expected (N, 2).'
            )

        positions_m = _METERS_PER_MILLIMETER * positions_mm
        return ProbePositionSequence(
            [
                ProbePosition(index=index, x_m=float(x_m), y_m=float(y_m))
                for index, (x_m, y_m) in enumerate(positions_m)
            ]
        )


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS12IDDiffractionFileReader(),
        simple_name='APS_PtychoSAXS',
    )
    registry.probe_position_file_readers.register_plugin(
        APS12IDPositionFileReader(),
        simple_name='APS_PtychoSAXS',
    )
