from collections.abc import Mapping
from pathlib import Path
import logging
import re
from typing import Final

import h5py
import numpy

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

# Splits a series member's stem into its fixed prefix and its frame counter. The
# counter is the last digit run, which `tail` enforces by admitting no digits of its
# own. Identifying it by length instead would break on these instruments: 2-ID-E
# writes `fly054_data_001.h5`, whose scan and frame fields are both three digits, and
# `max(..., key=len)` resolves that tie toward the scan number -- globbing one frame
# from each of hundreds of unrelated scans. The extension is excluded from the match
# because it has digits of its own: ".h5" would otherwise supply the last digit run.
_SERIES_STEM: Final = re.compile(r'(?P<prefix>.*?)(?P<frame>\d+)(?P<tail>\D*)')


class APS2IDDiffractionFileReader(DiffractionFileReader):
    # These files carry no detector metadata, so the pitch of the Eiger detectors on
    # these instruments stands in for it. Override it from the metadata page when a
    # different detector is in use.
    DETECTOR_PIXEL_SIZE_M: Final[float] = 75e-6

    def _get_file_series(self, file_path: Path) -> tuple[Mapping[int, Path], str]:
        """Collect the frames of one scan, keyed by frame number.

        Every digit field ahead of the counter -- the scan number here -- is pinned to
        its literal value from *file_path*, so siblings from other scans cannot match.
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

    def read(self, file_path: Path) -> DiffractionDataset:
        file_path_mapping, file_pattern = self._get_file_series(file_path)
        data_path = '/entry/data/data'

        contents_tree = DiffractionDatasetLayoutNode.create_root()
        array_list: list[DiffractionArray] = list()
        num_patterns_per_array: list[int] = list()
        detector_extent: ImageExtent | None = None
        pattern_dtype = numpy.dtype(numpy.uint32)
        offset = 0

        # Each member declares its own frame count. A fly scan can cut a line short, and
        # asserting the first file's count for the whole series makes every array that
        # disagrees fail its length check and be dropped with only a warning.
        for idx, fp in sorted(file_path_mapping.items()):
            with h5py.File(fp, 'r') as h5_file:
                h5data = h5_file[data_path]

                if not isinstance(h5data, h5py.Dataset):
                    raise ValueError(f'Expected dataset at "{fp}:{data_path}".')

                num_patterns, detector_height, detector_width = h5data.shape

                if detector_extent is None:
                    pattern_dtype = h5data.dtype
                    detector_extent = ImageExtent(detector_width, detector_height)

            indexes = numpy.arange(num_patterns) + offset
            array = H5DiffractionPatternArray(fp.stem, indexes, fp, data_path)
            contents_tree.add_child(array.get_label(), 'HDF5', str(idx))
            array_list.append(array)
            num_patterns_per_array.append(num_patterns)
            offset += num_patterns

        if detector_extent is None:
            raise ValueError(f'No diffraction files matched "{file_pattern}".')

        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=pattern_dtype,
            detector_extent=detector_extent,
            detector_pixel_geometry=PixelGeometry(
                width_m=self.DETECTOR_PIXEL_SIZE_M,
                height_m=self.DETECTOR_PIXEL_SIZE_M,
            ),
            file_path=file_path.parent / file_pattern,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_2IDD',
        display_name='APS 2-ID-D Microprobe Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_2IDE',
        display_name='APS 2-ID-E Microprobe Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_BNP',
        display_name='APS 2-ID-D Bionanoprobe Files (*.h5 *.hdf5)',
    )
