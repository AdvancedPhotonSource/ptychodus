"""Readers for APS 12-ID-E Ptycho-SAXS scans stored as one TIFF per scan point.

A scan is a directory of ``<sample>_<scan>_<line>_<point>.tif`` frames with a matching
directory of ``<sample>_<scan>_<line>_<point>.dat`` position files. Both readers take
any one member of the series and glob the rest, ordering by (line, point) so the
pattern and position sequences agree.

Positions use the same convention as the other 12-ID readers: column 1 is y and column
2 is x, in nanometers, with x negated. Every row of a point shares that point's index,
so oversampled rows collapse into one anchor downstream rather than being averaged here.

The TIFF and position trees are siblings -- ``tifs/<scan>/`` and ``positions/<scan>/``
-- and the point number is offset by one between them, which is why the position reader
derives its own directory rather than being handed it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import logging
import re

import numpy
import tifffile

from ptychodus.api.constants import LengthUnit
from ptychodus.api.diffraction import (
    DiffractionArray,
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionPatterns,
    SimpleDiffractionArray,
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

logger = logging.getLogger(__name__)

_SIMPLE_NAME: Final[str] = 'APS_PtychoSAXS_TIFF'
_DIFFRACTION_DISPLAY_NAME: Final[str] = 'APS 12-ID-E Ptycho-SAXS Files (*.tif *.tiff)'
_POSITION_DISPLAY_NAME: Final[str] = 'APS 12-ID-E Ptycho-SAXS Files (*.dat)'

# Pilatus pitch at this instrument; TIFF frames carry no detector geometry.
_DETECTOR_PIXEL_SIZE_M: Final[float] = 172e-6

_POINT_PATTERN: Final = re.compile(r'^(?P<prefix>.+)_(?P<line>\d+)_(?P<point>\d+)$')


def _series(file_path: Path) -> list[tuple[int, int, Path]]:
    """Every (line, point, path) of the series this file belongs to, in scan order."""
    match = _POINT_PATTERN.match(file_path.stem)

    if match is None:
        raise ValueError(
            f'"{file_path.name}" is not named <sample>_<scan>_<line>_<point>{file_path.suffix}.'
        )

    prefix = match.group('prefix')
    found: list[tuple[int, int, Path]] = []

    for candidate in file_path.parent.glob(f'{prefix}_*{file_path.suffix}'):
        candidate_match = _POINT_PATTERN.match(candidate.stem)

        if candidate_match is not None and candidate_match.group('prefix') == prefix:
            found.append(
                (int(candidate_match.group('line')), int(candidate_match.group('point')), candidate)
            )

    if not found:
        raise ValueError(f'No series members found beside "{file_path}".')

    return sorted(found)


class APS12IDTIFFDiffractionFileReader(DiffractionFileReader):
    def read(self, file_path: Path) -> DiffractionDataset:
        series = _series(file_path)
        contents_tree = DiffractionDatasetLayoutNode.create_root()
        array_list: list[DiffractionArray] = []

        # One frame per file, so the whole scan is read eagerly: there is no dataset to
        # slice lazily the way the HDF5 readers do.
        for index, (line, point, path) in enumerate(series):
            frame = numpy.asarray(tifffile.imread(path))

            if frame.ndim != 2:
                raise ValueError(f'"{path}" holds a {frame.ndim}D image; expected a single frame.')

            patterns: DiffractionPatterns = frame[numpy.newaxis]
            array_list.append(SimpleDiffractionArray(path.stem, numpy.array([index]), patterns))
            contents_tree.add_child(path.stem, 'TIFF', f'{line}_{point}')

        first = numpy.asarray(tifffile.imread(series[0][2]))
        detector_height, detector_width = first.shape

        metadata = DiffractionMetadata(
            num_patterns_per_array=[1] * len(array_list),
            pattern_dtype=first.dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_pixel_geometry=PixelGeometry(
                width_m=_DETECTOR_PIXEL_SIZE_M, height_m=_DETECTOR_PIXEL_SIZE_M
            ),
            file_path=file_path,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


class APS12IDTIFFPositionFileReader(ProbePositionFileReader):
    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = []
        scale = LengthUnit.NANOMETER.meters_per_unit

        for _, _, path in _series(file_path):
            rows = numpy.atleast_2d(numpy.genfromtxt(path))

            if rows.size == 0:
                raise ProbePositionParseError(f'"{path}" holds no position rows.')

            if rows.shape[-1] < 3:
                raise ProbePositionParseError(
                    f'"{path}" needs at least three columns; found {rows.shape[-1]}.'
                )

            index = len(point_list)

            for row in rows:
                point_list.append(
                    ProbePosition(
                        index=index,
                        x_m=-scale * float(row[2]),
                        y_m=+scale * float(row[1]),
                    )
                )

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS12IDTIFFDiffractionFileReader(),
        simple_name=_SIMPLE_NAME,
        display_name=_DIFFRACTION_DISPLAY_NAME,
    )
    registry.probe_position_file_readers.register_plugin(
        APS12IDTIFFPositionFileReader(),
        simple_name=_SIMPLE_NAME,
        display_name=_POSITION_DISPLAY_NAME,
    )
