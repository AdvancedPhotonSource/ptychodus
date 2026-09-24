from collections.abc import Mapping
from pathlib import Path
import logging
import re
import sys
from typing import Final

from tifffile import TiffFile
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
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
from ptychodus.api.plugins import PluginRegistry

logger = logging.getLogger(__name__)

# Splits a series member's stem into its fixed prefix and its trailing frame counter.
# See the note in aps02id_diffraction_file.py: identifying the counter by digit-run
# length misfires whenever another field is as wide, and the extension is excluded
# because ".tif" carries no digits but ".h5"-style suffixes elsewhere do.
_SERIES_STEM: Final = re.compile(r'(?P<prefix>.*?)(?P<frame>\d+)(?P<tail>\D*)')


class TiffDiffractionPatternArray(DiffractionArray):
    def __init__(self, file_path: Path, indexes: DiffractionIndexes) -> None:
        super().__init__()
        self._file_path = file_path
        # One index per page in this file. A multi-page TIFF holds a stack, so a lone
        # index would not match what get_patterns returns and the array would be
        # dropped during assembly.
        self._indexes = indexes

    def get_label(self) -> str:
        return self._file_path.stem

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        with TiffFile(self._file_path) as tiff:
            data = tiff.asarray()

        if data.ndim == 2:
            data = data[numpy.newaxis, :, :]

        if read_region is None:
            return data
        return read_region.apply_to(data)


class TiffDiffractionFileReader(DiffractionFileReader):
    def _get_file_series(self, file_path: Path) -> tuple[Mapping[int, Path], str]:
        """Collect the members of one numbered TIFF series, keyed by frame number.

        The counter is the last digit run in the stem, and every digit ahead of it is
        pinned to its literal value, so a name whose leading field also varies -- a scan
        number, say -- cannot pull in siblings from other scans. Choosing the longest
        run instead picks the wrong field whenever the two are the same width.
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
        contents_tree = DiffractionDatasetLayoutNode.create_root()
        array_list: list[DiffractionArray] = list()
        num_patterns_per_array: list[int] = list()
        detector_extent: ImageExtent | None = None
        # Annotated, not inferred: the initial value would otherwise narrow the
        # variable to uint16 and reject the dtype read from the first page.
        pattern_dtype: numpy.dtype = numpy.dtype(numpy.uint16)
        offset = 0

        # Page counts are read per file rather than assumed uniform: a series may mix
        # single-page and stacked members, and a declared count that overstates a file
        # makes assembly drop it with only a warning. `pages` reports the count without
        # decoding the image data.
        for idx, fp in sorted(file_path_mapping.items()):
            with TiffFile(fp) as tiff:
                num_patterns = len(tiff.pages)

                if detector_extent is None:
                    page = tiff.pages[0]
                    detector_height, detector_width = page.shape[-2:]
                    detector_extent = ImageExtent(detector_width, detector_height)
                    pattern_dtype = numpy.dtype(page.dtype)

            indexes = numpy.arange(num_patterns) + offset
            array = TiffDiffractionPatternArray(fp, indexes)
            contents_tree.add_child(array.get_label(), 'TIFF', str(idx))
            array_list.append(array)
            num_patterns_per_array.append(num_patterns)
            offset += num_patterns

        if detector_extent is None:
            raise ValueError(f'No diffraction files matched "{file_pattern}".')

        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=pattern_dtype,
            detector_extent=detector_extent,
            file_path=file_path.parent / file_pattern,
        )

        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def register_plugins(registry: PluginRegistry) -> None:
    file_reader = TiffDiffractionFileReader()
    registry.diffraction_file_readers.register_plugin(
        file_reader,
        simple_name='TIFF',
        display_name='Tagged Image File Format Files (*.tif *.tiff)',
    )
    registry.diffraction_file_readers.register_plugin(
        file_reader,
        simple_name='CNM_APS_HXN_TIFF',
        display_name='CNM/APS 26-ID-C Hard X-ray Nanoprobe Files (*.tif *.tiff)',
    )
    registry.diffraction_file_readers.register_plugin(
        file_reader,
        simple_name='APS_Atomic',
        display_name='APS 34-ID-F Atomic Files (*.tif *.tiff)',
    )


if __name__ == '__main__':
    file_path = Path(sys.argv[1])
    reader = TiffDiffractionFileReader()
    tiff_file = reader.read(file_path)
    print(tiff_file)
