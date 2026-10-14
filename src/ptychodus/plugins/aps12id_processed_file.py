"""Readers for pre-processed APS 12-ID-E Ptycho-SAXS scans.

A processed scan is a master file plus one file per scan line, the lines named
``<sample>_<scan>_<line>.h5`` beside it. Each line file carries its own patterns at
``/dp`` and its own positions at ``/positions``; the master carries the beam center as
a ``beam_center_YX`` attribute. This differs from the raw layout handled by
:mod:`ptychodus.plugins.aps12id_diffraction_file`, which is one file per scan point.

Positions come from column 1 (y) and column 2 (x) of ``/positions``, in nanometers,
with x negated -- the same convention as the raw ``.dat`` reader in
:mod:`ptychodus.plugins.aps12id_position_file`. Unlike the raw ``.dat`` files, which
hold several oversampled rows per scan point, ``/positions`` holds one row per point,
so rows and indexes correspond one to one.
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
    BeamCenter,
    DiffractionArray,
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
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

from .h5_diffraction_file import H5DiffractionPatternArray

logger = logging.getLogger(__name__)

_SIMPLE_NAME: Final[str] = 'APS_PtychoSAXS_Processed'
_DISPLAY_NAME: Final[str] = 'APS 12-ID-E Ptycho-SAXS Processed Files (*.h5 *.hdf5)'

_DATA_PATH: Final[str] = '/dp'
_POSITIONS_PATH: Final[str] = '/positions'
_BEAM_CENTER_ATTR: Final[str] = 'beam_center_YX'

# Pilatus pitch at this instrument; these files carry no detector geometry.
_DETECTOR_PIXEL_SIZE_M: Final[float] = 172e-6


def _line_files(file_path: Path) -> list[Path]:
    """Every line file of this scan, ordered by line number.

    Accepts either the master or any line file as the entry point, so the caller does
    not have to know which one they are holding.
    """
    stem = file_path.stem
    prefix = stem[: -len('_master')] if stem.endswith('_master') else stem.rsplit('_', 1)[0]
    candidates: list[tuple[int, Path]] = []

    for candidate in file_path.parent.glob(f'{prefix}_*{file_path.suffix}'):
        if candidate.stem.endswith('_master'):
            continue

        match = re.search(r'_(\d+)$', candidate.stem)

        if match is not None:
            candidates.append((int(match.group(1)), candidate))

    if not candidates:
        raise ValueError(f'No line files found beside "{file_path}".')

    return [path for _, path in sorted(candidates)]


def _read_beam_center(file_path: Path) -> tuple[int, int] | None:
    """Beam center as (x, y) from the master file, when it names one."""
    stem = file_path.stem
    master = (
        file_path
        if stem.endswith('_master')
        else file_path.with_name(f'{stem.rsplit("_", 1)[0]}_master{file_path.suffix}')
    )

    if not master.is_file():
        return None

    with h5py.File(master, 'r') as h5_file:
        raw = h5_file.attrs.get(_BEAM_CENTER_ATTR)

    if raw is None:
        return None

    center_yx = numpy.atleast_1d(numpy.asarray(raw, dtype=float)).reshape(-1)

    if center_yx.size < 2:
        logger.warning('Ignoring malformed %s in %s', _BEAM_CENTER_ATTR, master)
        return None

    return int(round(center_yx[1])), int(round(center_yx[0]))


class APS12IDProcessedDiffractionFileReader(DiffractionFileReader):
    def read(self, file_path: Path) -> DiffractionDataset:
        line_files = _line_files(file_path)
        num_patterns_per_array: list[int] = []
        array_list: list[DiffractionArray] = []
        contents_tree = DiffractionDatasetLayoutNode.create_root()
        detector_height = 0
        detector_width = 0
        pattern_dtype = numpy.dtype(numpy.uint32)
        offset = 0

        for index, line_file in enumerate(line_files):
            with h5py.File(line_file, 'r') as h5_file:
                try:
                    dataset = h5_file[_DATA_PATH]
                except KeyError:
                    logger.warning('Skipping %s: no "%s".', line_file, _DATA_PATH)
                    continue

                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
                    logger.warning('Skipping %s: "%s" is not a 3D dataset.', line_file, _DATA_PATH)
                    continue

                num_patterns, detector_height, detector_width = dataset.shape
                pattern_dtype = dataset.dtype

            # Indexes run consecutively across the lines, matching the order the
            # positions are concatenated in by the reader below.
            indexes = numpy.arange(num_patterns) + offset
            offset += num_patterns
            num_patterns_per_array.append(num_patterns)
            array_list.append(
                H5DiffractionPatternArray(line_file.stem, indexes, line_file, _DATA_PATH)
            )
            contents_tree.add_child(line_file.stem, 'HDF5', str(index))

        if not array_list:
            raise ValueError(f'No readable line files beside "{file_path}".')

        beam_center = _read_beam_center(file_path)
        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=pattern_dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_pixel_geometry=PixelGeometry(
                width_m=_DETECTOR_PIXEL_SIZE_M, height_m=_DETECTOR_PIXEL_SIZE_M
            ),
            beam_center=None if beam_center is None else BeamCenter(*beam_center),
            file_path=file_path,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


class APS12IDProcessedPositionFileReader(ProbePositionFileReader):
    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = []
        scale = LengthUnit.NANOMETER.meters_per_unit

        for line_file in _line_files(file_path):
            with h5py.File(line_file, 'r') as h5_file:
                try:
                    dataset = h5_file[_POSITIONS_PATH]
                except KeyError:
                    raise ProbePositionParseError(
                        f'"{line_file}" has no "{_POSITIONS_PATH}".'
                    ) from None

                positions = numpy.atleast_2d(numpy.asarray(dataset[()], dtype=float))

            if positions.shape[-1] < 3:
                raise ProbePositionParseError(
                    f'"{_POSITIONS_PATH}" in "{line_file}" needs at least three columns; '
                    f'found {positions.shape[-1]}.'
                )

            for row in positions:
                point_list.append(
                    ProbePosition(
                        index=len(point_list),
                        x_m=-scale * float(row[2]),
                        y_m=+scale * float(row[1]),
                    )
                )

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS12IDProcessedDiffractionFileReader(),
        simple_name=_SIMPLE_NAME,
        display_name=_DISPLAY_NAME,
    )
    registry.probe_position_file_readers.register_plugin(
        APS12IDProcessedPositionFileReader(),
        simple_name=_SIMPLE_NAME,
        display_name=_DISPLAY_NAME,
    )
