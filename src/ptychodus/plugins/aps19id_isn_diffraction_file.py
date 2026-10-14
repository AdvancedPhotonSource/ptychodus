from collections.abc import Mapping
from pathlib import Path
from typing import Final
import logging
import re

import h5py
import numpy

from ptychodus.api.constants import HC_EV_ANGSTROM
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    DiffractionArray,
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray

logger = logging.getLogger(__name__)

# Splits a series member's stem into its fixed prefix and its trailing frame counter;
# see the note in aps02id_diffraction_file.py.
_SERIES_STEM: Final = re.compile(r'(?P<prefix>.*?)(?P<frame>\d+)(?P<tail>\D*)')


class ISNDiffractionFileReader(DiffractionFileReader):
    """Reader for APS 19-ID-E In-situ Nanoprobe diffraction data.

    The new ISN format stores diffraction frames across many sibling
    per-file HDF5s in a single directory (e.g. ``scan_1307_00001.h5`` ...
    ``scan_1307_00400.h5``), each holding ``/entry/data/data`` of shape
    ``(nframes, height, width)``. There is no master file and no configs
    group; geometry (detector distance, pixel size) is not recorded and is
    supplied by the user via GUI settings. Frames are indexed by array order.
    """

    # These files record no detector geometry, so the pitch of the Eiger in use at this
    # instrument stands in for it, as it does for the other APS readers. A pitch found
    # in the file would still win; none is written today.
    DETECTOR_PIXEL_SIZE_M: Final[float] = 75e-6
    DATA_PATH: Final[str] = '/entry/data/data'
    WAVELENGTH_PATH: Final[str] = '/entry/instrument/NDAttributes/Wavelength'
    COUNT_TIME_PATH: Final[str] = '/entry/instrument/NDAttributes/CountTime'

    def _get_file_series(self, file_path: Path) -> tuple[Mapping[int, Path], str]:
        """Collect the frames of one scan, keyed by frame number.

        The counter is the last digit run in the stem, with every earlier digit field
        pinned to its literal value. Both ISN namings need that rule: the newer
        `scan_1307_00001.h5` would survive a longest-run heuristic, but the older
        `19ide_0179_000.h5` has a four-digit scan number ahead of a three-digit
        counter, so choosing by length would glob across scans instead of frames.
        """
        member = _SERIES_STEM.fullmatch(file_path.stem)

        if member is None:
            raise ValueError(f'File name "{file_path.name}" carries no frame number.')

        prefix = member['prefix']
        tail = member['tail'] + file_path.suffix
        width = len(member['frame'])
        file_pattern = f'{prefix}(\\d{{{width}}}){tail}'
        series_regex = re.compile(f'{re.escape(prefix)}(?P<frame>\\d{{{width}}}){re.escape(tail)}')
        file_path_dict: dict[int, Path] = dict()

        for fp in file_path.parent.iterdir():
            z = series_regex.fullmatch(fp.name)

            if z:
                file_path_dict[int(z['frame'])] = fp

        return file_path_dict, file_pattern

    def _read_scalar(self, h5_file: h5py.File, data_path: str) -> float | None:
        """Read a per-frame NDAttribute scalar, returning None if absent or NaN."""
        try:
            values = h5_file[data_path][()]
        except KeyError:
            return None

        value = float(numpy.ravel(values)[0])
        return None if numpy.isnan(value) else value

    def read(self, file_path: Path) -> DiffractionDataset:
        file_path_mapping, file_pattern = self._get_file_series(file_path)

        num_patterns_per_array: list[int] = []
        array_list: list[DiffractionArray] = []
        contents_tree = DiffractionDatasetLayoutNode.create_root()

        pattern_dtype = numpy.dtype(numpy.int32)
        detector_extent: ImageExtent | None = None
        photon_energy_eV: float | None = None  # noqa: N806
        exposure_time_s: float | None = None
        offset = 0

        for idx, fp in sorted(file_path_mapping.items()):
            with h5py.File(fp, 'r') as h5_file:
                h5_data = h5_file[self.DATA_PATH]

                if not isinstance(h5_data, h5py.Dataset):
                    raise ValueError(f'Expected dataset at "{fp}:{self.DATA_PATH}".')

                num_patterns, detector_height, detector_width = h5_data.shape

                if detector_extent is None:
                    pattern_dtype = h5_data.dtype
                    detector_extent = ImageExtent(detector_width, detector_height)

                    wavelength_angstrom = self._read_scalar(h5_file, self.WAVELENGTH_PATH)
                    if wavelength_angstrom:
                        photon_energy_eV = HC_EV_ANGSTROM / wavelength_angstrom  # noqa: N806

                    exposure_time_s = self._read_scalar(h5_file, self.COUNT_TIME_PATH)

            indexes = numpy.arange(num_patterns) + offset
            array = H5DiffractionPatternArray(fp.stem, indexes, fp, self.DATA_PATH)
            contents_tree.add_child(array.get_label(), 'HDF5', str(idx))
            array_list.append(array)
            num_patterns_per_array.append(num_patterns)
            offset += num_patterns

        if detector_extent is None:
            raise ValueError(f'No ISN diffraction files matched "{file_pattern}".')

        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=pattern_dtype,
            detector_extent=detector_extent,
            detector_pixel_geometry=PixelGeometry(
                width_m=self.DETECTOR_PIXEL_SIZE_M,
                height_m=self.DETECTOR_PIXEL_SIZE_M,
            ),
            photon_energy_eV=photon_energy_eV,
            exposure_time_s=exposure_time_s,
            file_path=file_path.parent / file_pattern,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        ISNDiffractionFileReader(),
        simple_name='APS_ISN',
        display_name='APS 19-ID-E In-situ Nanoprobe Files (*.h5 *.hdf5)',
    )
