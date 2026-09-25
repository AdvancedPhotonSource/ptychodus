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
    DiffractionPatternDType,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray
from .mda.mda_position_file import RASTER_LINE_INDEX_STRIDE

logger = logging.getLogger(__name__)

# Splits a series member's stem into its fixed prefix and its frame counter. The
# counter is the last digit run, which `tail` enforces by admitting no digits of its
# own. Identifying it by length instead would break on these instruments: 2-ID-E
# writes `fly054_data_001.h5`, whose scan and frame fields are both three digits, and
# `max(..., key=len)` resolves that tie toward the scan number -- globbing one frame
# from each of hundreds of unrelated scans. The extension is excluded from the match
# because it has digits of its own: ".h5" would otherwise supply the last digit run.
_SERIES_STEM: Final = re.compile(r'(?P<prefix>.*?)(?P<frame>\d+)(?P<tail>\D*)')

# The areaDetector driver's per-frame counter, written by the HDF5 plugin. It runs across
# the whole acquisition rather than restarting per file, so a step of one between
# consecutive frames of a member says the detector recorded that line without dropping a
# frame in the middle of it.
_UNIQUE_ID_PATH: Final = '/entry/instrument/NDAttributes/NDArrayUniqueId'


def _warn_on_dropped_frames(h5_file: h5py.File, file_path: Path) -> None:
    """Report a member whose frames are not consecutive in the detector's own counter.

    Under line-major numbering a frame's column is its position within the member, so a
    frame the detector dropped mid-line shifts every column after it -- placing patterns
    on positions the beam never visited, without any count disagreeing. The counter is
    the only record of the drop. Files that do not carry it are left alone.
    """
    h5data = h5_file.get(_UNIQUE_ID_PATH)

    if not isinstance(h5data, h5py.Dataset) or h5data.shape[0] < 2:
        return

    unique_ids = h5data[()]
    steps = numpy.unique(numpy.diff(unique_ids))

    if steps.size != 1 or steps[0] != 1:
        logger.warning(
            'Frames in "%s" are not consecutive in the detector frame counter (steps %s);'
            ' the detector dropped frames, so columns after the gap name the wrong'
            ' scan positions.',
            file_path,
            steps.tolist(),
        )


class APS2IDDiffractionFileReader(DiffractionFileReader):
    """Reader for a scan written as one Eiger HDF5 per scan line.

    `line_index_stride` selects how patterns are numbered. Left at None they are numbered
    by running count over the series. Set to RASTER_LINE_INDEX_STRIDE they are numbered
    line-major, which pairs them with positions numbered the same way even though the
    positioner is driven over more points per line than the detector records; see that
    constant in ptychodus.plugins.mda.mda_position_file. A line holding at least the
    stride is rejected rather than allowed to collide with the next line.
    """

    # These files carry no detector metadata, so the pitch of the Eiger detectors on
    # these instruments stands in for it. Override it from the metadata page when a
    # different detector is in use.
    DETECTOR_PIXEL_SIZE_M: Final[float] = 75e-6

    def __init__(self, *, line_index_stride: int | None = None) -> None:
        self._line_index_stride = line_index_stride

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
        stride = self._line_index_stride

        # Line-major numbering reads the line off the file's own counter rather than off
        # the position of the file in the series, so a member that is absent or unreadable
        # leaves a hole instead of pulling every later line one place earlier. The counter
        # is one-based, so line zero is member one; a zero would make the first line
        # negative, and no such series is known.
        if stride is not None and min(file_path_mapping, default=1) < 1:
            raise ValueError(
                f'Series "{file_pattern}" is numbered from'
                f' {min(file_path_mapping)}; line-major indexing needs a one-based counter.'
            )

        contents_tree = DiffractionDatasetLayoutNode.create_root()
        array_list: list[DiffractionArray] = list()
        num_patterns_per_array: list[int] = list()
        detector_extent: ImageExtent | None = None
        pattern_dtype: DiffractionPatternDType | None = None
        num_unreadable = 0
        offset = 0

        # Each member declares its own frame count. A fly scan can cut a line short, and
        # asserting the first file's count for the whole series makes every array that
        # disagrees fail its length check and be dropped with only a warning.
        for idx, fp in sorted(file_path_mapping.items()):
            try:
                h5_file = h5py.File(fp, 'r')
            except OSError as exc:
                # An interrupted acquisition leaves a zero-length or half-written member.
                # The rest of the series is still a scan, and the lines it does hold keep
                # their own numbers, so the hole costs only its own patterns.
                logger.warning('Skipping unreadable "%s": %s', fp, exc)
                num_unreadable += 1
                continue

            with h5_file:
                h5data = h5_file[data_path]

                if not isinstance(h5data, h5py.Dataset):
                    raise ValueError(f'Expected dataset at "{fp}:{data_path}".')

                num_patterns, detector_height, detector_width = h5data.shape
                member_extent = ImageExtent(detector_width, detector_height)

                if stride is not None:
                    _warn_on_dropped_frames(h5_file, fp)

                # Widening rather than trusting the first member: these series mix widths
                # -- a one-frame opening line can be uint32 where the rest are uint16 --
                # and a buffer sized from a narrower first member truncates on assignment.
                pattern_dtype = (
                    h5data.dtype
                    if pattern_dtype is None
                    else numpy.promote_types(pattern_dtype, h5data.dtype)
                )

            if detector_extent is None:
                detector_extent = member_extent
            elif member_extent != detector_extent:
                raise ValueError(
                    f'"{fp}" is {member_extent.width_px}x{member_extent.height_px}, but the'
                    f' series is {detector_extent.width_px}x{detector_extent.height_px}.'
                )

            if stride is None:
                indexes = numpy.arange(num_patterns) + offset
            elif num_patterns >= stride:
                raise ValueError(
                    f'"{fp}" holds {num_patterns} patterns, which reaches the line-major'
                    f' index stride of {stride}; its numbering would collide with the next'
                    ' line.'
                )
            else:
                indexes = numpy.arange(num_patterns) + (idx - 1) * stride

            array = H5DiffractionPatternArray(fp.stem, indexes, fp, data_path)
            contents_tree.add_child(array.get_label(), 'HDF5', str(idx))
            array_list.append(array)
            num_patterns_per_array.append(num_patterns)
            offset += num_patterns

        if detector_extent is None or pattern_dtype is None:
            raise ValueError(
                f'No readable diffraction files matched "{file_pattern}"'
                f' ({num_unreadable} were unreadable).'
            )

        if num_unreadable > 0:
            logger.warning(
                'Read %d of %d members of "%s"; %d were unreadable.',
                len(array_list),
                len(file_path_mapping),
                file_pattern,
                num_unreadable,
            )

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
    # The XFM fly scan drives the positioner over more points per line than the Eiger
    # records, so the patterns are numbered line-major to pair with positions numbered
    # the same way; see RASTER_LINE_INDEX_STRIDE.
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(line_index_stride=RASTER_LINE_INDEX_STRIDE),
        simple_name='APS_2IDE',
        display_name='APS 2-ID-E Microprobe Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_BNP',
        display_name='APS 2-ID-D Bionanoprobe Files (*.h5 *.hdf5)',
    )
