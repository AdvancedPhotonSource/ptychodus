from pathlib import Path
from typing import Final
import logging
import re

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionFileTreeBuilder, H5DiffractionPatternArray

logger = logging.getLogger(__name__)


def _read_ndattribute_scalar(h5_file: h5py.File, path: str) -> float | None:
    try:
        value = h5_file[path][()]
    except KeyError:
        logger.warning(f'NDAttribute "{path}" not found in "{h5_file.filename}".')
        return None

    array = numpy.atleast_1d(value)
    return float(array[0])


# Raw 12-ID scans are written one file per scan point, named
# <scan>_<line>_<point>.h5, so the series is indexed by TWO fields. Matching a single
# digit run picks the line and pins the point to whatever the given file happened to
# carry, collecting one point from each line and silently discarding the rest.
_POINT_PATTERN: Final = re.compile(r'(?P<prefix>.+)_(?P<line>\d+)_(?P<point>\d+)')


def _point_series(file_path: Path) -> tuple[list[tuple[int, int, Path]], str]:
    """Every (line, point, path) of the series this file belongs to, in scan order.

    Mirrors `_series` in aps12id_tiff_file.py, which reads the same layout written as
    TIFF; the two stay deliberately parallel.
    """
    match = _POINT_PATTERN.fullmatch(file_path.stem)

    if match is None:
        raise ValueError(
            f'"{file_path.name}" is not named <scan>_<line>_<point>{file_path.suffix}.'
        )

    prefix = match['prefix']
    file_pattern = f'{prefix}_(\\d+)_(\\d+){file_path.suffix}'
    found: list[tuple[int, int, Path]] = []

    for candidate in file_path.parent.glob(f'{prefix}_*{file_path.suffix}'):
        candidate_match = _POINT_PATTERN.fullmatch(candidate.stem)

        if candidate_match is not None and candidate_match['prefix'] == prefix:
            found.append((int(candidate_match['line']), int(candidate_match['point']), candidate))

    if not found:
        raise ValueError(f'No series members found beside "{file_path}".')

    return sorted(found), file_pattern


class APS12IDDiffractionFileReader(DiffractionFileReader):
    # The NDAttributes group carries no detector geometry, so the Pilatus pitch at
    # this instrument stands in for it.
    DETECTOR_PIXEL_SIZE_M: Final[float] = 172e-6
    DATA_PATH: Final[str] = '/entry/data/data'
    ENERGY_PATH: Final[str] = '/entry/instrument/NDAttributes/monoE'
    EXPOSURE_PATH: Final[str] = '/entry/instrument/NDAttributes/ExposureTime'

    def read(self, file_path: Path) -> DiffractionDataset:
        tree_builder = H5DiffractionFileTreeBuilder()

        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = tree_builder.build(h5_file)

            try:
                h5_data = h5_file[self.DATA_PATH]
            except KeyError as exc:
                raise ValueError(f'File "{file_path}" is not an APS 12-ID data file.') from exc

            if not isinstance(h5_data, h5py.Dataset):
                raise ValueError(f'Data path "{self.DATA_PATH}" in "{file_path}" is not a dataset.')

            data_shape = h5_data.shape
            data_dtype = h5_data.dtype
            photon_energy_eV = _read_ndattribute_scalar(h5_file, self.ENERGY_PATH)  # noqa: N806
            exposure_time_s = _read_ndattribute_scalar(h5_file, self.EXPOSURE_PATH)

        if len(data_shape) == 3:
            num_patterns, detector_height, detector_width = data_shape

            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=data_dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                detector_pixel_geometry=PixelGeometry(
                    width_m=self.DETECTOR_PIXEL_SIZE_M, height_m=self.DETECTOR_PIXEL_SIZE_M
                ),
                photon_energy_eV=photon_energy_eV,
                exposure_time_s=exposure_time_s,
                file_path=file_path,
            )
            array = H5DiffractionPatternArray(
                label=file_path.stem,
                indexes=numpy.arange(num_patterns),
                file_path=file_path,
                data_path=self.DATA_PATH,
            )
            return SimpleDiffractionDataset(metadata, contents_tree, [array])

        if len(data_shape) == 2:
            detector_height, detector_width = data_shape
            series, file_pattern = _point_series(file_path)
            array_list: list[DiffractionArray] = list()

            # One pattern per file, ordered by (line, point) so the array order matches
            # the scan order the position file reports.
            for idx, (_line, _point, fp) in enumerate(series):
                indexes = numpy.array([idx])
                array = H5DiffractionPatternArray(fp.stem, indexes, fp, self.DATA_PATH)
                array_list.append(array)

            metadata = DiffractionMetadata(
                num_patterns_per_array=[1] * len(array_list),
                pattern_dtype=data_dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                detector_pixel_geometry=PixelGeometry(
                    width_m=self.DETECTOR_PIXEL_SIZE_M, height_m=self.DETECTOR_PIXEL_SIZE_M
                ),
                photon_energy_eV=photon_energy_eV,
                exposure_time_s=exposure_time_s,
                file_path=file_path.parent / file_pattern,
            )
            return SimpleDiffractionDataset(metadata, contents_tree, array_list)

        raise ValueError(
            f'Data path "{self.DATA_PATH}" in "{file_path}" has unsupported shape {data_shape}.'
        )


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS12IDDiffractionFileReader(),
        simple_name='APS_PtychoSAXS',
        display_name='APS 12-ID-E Ptycho-SAXS Files (*.h5 *.hdf5)',
    )
