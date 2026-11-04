"""Readers for CNM/APS 26-ID-C Hard X-ray Nanoprobe diffraction files.

Three detector layouts have been written, each beside an EPICS MDA scan record
``26idbSOFT_<NNNN>.mda`` that holds the positions and the beamline metadata:

- 2019: Medipix3 quad frames, one gzipped TIFF per point,
  ``Images/<N>/scan_<N>_img_<FFFFFF>.tif.gz``, with the MDA in ``../../mda/``.
- 2022: the same Medipix3, one plain TIFF per point, ``scan_<N>_img_<FFFFFF>.tif``, with
  the MDA in the same directory or its parent.
- 2023: an Eiger stack in one HDF5 file ``scan_<N>_<FFFFFF>.h5`` at ``/entry/data/data``,
  beside a metadata-only member ``scan_<N>_000000.h5`` holding an ``/entry/instrument``
  snapshot of the beamline motors, with the MDA in ``../mda/``.

Patterns pair with scan points in order: the k-th frame, sorted by frame number, is the
k-th MDA point. Checked against the MDA's per-point ``QMPX3:Stats1:Total_RBV``, which
peaks at that pairing in every Medipix era; ``QMPX3:TIFF1:FileNumber_RBV`` does not
track it (it equals the frame number in some scans, runs one ahead in others, and steps
by 0 or 2 in places), so it is not read. The Eiger's ``Stats1:Total_RBV`` matches one
point later, read as the statistics PV lagging the frame, since a frame leading its
point would need one more trigger than the scan issued.

Metadata comes from the MDA, chosen by PV name because the channel labels changed
between years: the photon energy from ``26idbDCM:sm8.RBV`` (labeled "DCM Energy" in
2019 and "Channel Cut Energy" in 2023) and the detector distance from
``26idcSOFT:sm5.RBV`` ("Det R", millimeters). The 2023 snapshot records both, as
``26-ID-B/Channel Cut Energy`` and ``26-ID-C/Det R``, plus ``26-ID-C/count_time``, and
overrides the MDA. Its other "DCM Energy" is the uncalibrated conversion of the
monochromator angle and is not used. Neither layout records a pixel pitch, so it is a
registration argument. Not read: the I0 scalers, whose channels are unlabeled;
``SAMPLE THETA``, whose zero is 90 degrees at normal incidence; and the detector arm's
two-theta, which takes some scans off the forward direction.

The Medipix writes unsigned counts into a signed 16-bit type in some years, where counts
above 32767 wrap negative inside the direct beam, so those frames are read as unsigned.
Its 32-bit frames cannot wrap; a rare -1 there flags an invalid reading and is left for
the value filter's lower bound to zero. Its inter-chip seam, rows and columns 255 to 260
of the 516-pixel frame, reads zero in 2019 and about half its neighbors' counts in 2022.
It is not masked by default: the 2022 beam is centered on it, and masking that cross
through zero frequency keeps a reconstruction from converging at all, while leaving it in
reproduces the sample. The Eiger flags invalid pixels with -1: those flagged in the first frame are bad pixels, and
those flagged only in a few frames, on the chip-boundary rows inside the beam, are left
for the value filter's lower bound to zero. A handful of single readings far below their
neighbors also occur in the 2022 frames; nothing marks them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import logging
import re

import h5py
import numpy

from ptychodus.api.constants import LengthUnit
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
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import (
    H5DiffractionFileTreeBuilder,
    H5DiffractionPatternArray,
    warn_on_dropped_frames,
)
from .mda.mda_position_file import MDAFile, read_detector_channel
from .tiff_diffraction_file import TiffDiffractionPatternArray, open_tiff

logger = logging.getLogger(__name__)

_DESCRIPTION: Final = 'CNM/APS 26-ID-C Hard X-ray Nanoprobe Files'
_DATA_PATH: Final = '/entry/data/data'
_METERS_PER_MILLIMETER: Final = LengthUnit.MILLIMETER.meters_per_unit


@dataclass(frozen=True)
class _ScanMetadata:
    photon_energy_eV: float | None = None  # noqa: N815
    detector_distance_m: float | None = None
    exposure_time_s: float | None = None

    def overridden_by(self, other: _ScanMetadata) -> _ScanMetadata:
        """This metadata with every value `other` records taking precedence."""
        return _ScanMetadata(
            photon_energy_eV=self.photon_energy_eV
            if other.photon_energy_eV is None
            else other.photon_energy_eV,
            detector_distance_m=self.detector_distance_m
            if other.detector_distance_m is None
            else other.detector_distance_m,
            exposure_time_s=self.exposure_time_s
            if other.exposure_time_s is None
            else other.exposure_time_s,
        )


def _find_mda(directory: Path, scan: int) -> Path | None:
    """The scan's MDA record, searched where each layout keeps it, or None."""
    name = f'26idbSOFT_{scan:04d}.mda'

    for candidate in (
        directory / name,
        directory.parent / name,
        directory.parent / 'mda' / name,
        directory.parent.parent / 'mda' / name,
    ):
        if candidate.is_file():
            return candidate

    logger.warning(f'No {name} near "{directory}"; the scan metadata is left unset.')
    return None


def _read_mda_metadata(directory: Path, scan: int) -> _ScanMetadata:
    """The energy and detector distance the scan's MDA records, each None when absent."""
    mda_path = _find_mda(directory, scan)

    if mda_path is None:
        return _ScanMetadata()

    try:
        mda_file = MDAFile.read(mda_path)
    except (OSError, EOFError, ValueError) as exc:
        logger.warning(f'Failed to read "{mda_path}"; the scan metadata is left unset: {exc}')
        return _ScanMetadata()

    def median(name: str, scale: float = 1.0) -> float | None:
        values = read_detector_channel(mda_file, name)

        if values is None or values.size == 0:
            logger.debug(f'"{mda_path}" records no {name}.')
            return None

        value = float(numpy.median(values)) * scale
        logger.debug(f'{name} = {value} from "{mda_path}".')
        return value

    return _ScanMetadata(
        photon_energy_eV=median('26idbDCM:sm8.RBV'),
        detector_distance_m=median('26idcSOFT:sm5.RBV', _METERS_PER_MILLIMETER),
    )


def _read_snapshot_metadata(h5_file: h5py.File) -> _ScanMetadata:
    """The energy, detector distance and exposure an HDF5 instrument snapshot records."""

    def scalar(path: str, scale: float = 1.0) -> float | None:
        item = h5_file.get(path)

        if not isinstance(item, h5py.Dataset) or item.size != 1:
            logger.debug(f'"{h5_file.filename}" records no {path}.')
            return None

        value = float(numpy.squeeze(item[()])) * scale
        logger.debug(f'{path} = {value} from "{h5_file.filename}".')
        return value

    return _ScanMetadata(
        photon_energy_eV=scalar('/entry/instrument/26-ID-B/Channel Cut Energy'),
        detector_distance_m=scalar('/entry/instrument/26-ID-C/Det R', _METERS_PER_MILLIMETER),
        exposure_time_s=scalar('/entry/instrument/26-ID-C/count_time'),
    )


def _counts_dtype(dtype: numpy.dtype) -> numpy.dtype:
    """The type a Medipix frame's counts are read as: int16 holds wrapped uint16 counts."""
    return numpy.dtype(numpy.uint16) if dtype == numpy.int16 else dtype


class _MedipixTiffArray(TiffDiffractionPatternArray):
    """A Medipix TIFF, whose int16 frames hold unsigned counts and are read as such."""

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        patterns = super().get_patterns(read_region=read_region)
        return patterns.view(_counts_dtype(patterns.dtype))


class APS26IDTiffDiffractionFileReader(DiffractionFileReader):
    """The Medipix3 layouts: one TIFF, plain or gzipped, per scan point.

    `detector_pixel_size_m` is the Medipix3 pitch the files do not record. `seam_pixels`
    are rows and columns of the frame, as recorded, to report as bad pixels -- the
    inter-chip seam is ``range(255, 261)`` -- with indexes beyond the frame ignored. None
    are by default; see the module docstring.
    """

    def __init__(
        self,
        *,
        detector_pixel_size_m: float = 55e-6,
        seam_pixels: Sequence[int] = (),
    ) -> None:
        self._detector_pixel_size_m = detector_pixel_size_m
        self._seam_pixels = tuple(seam_pixels)

    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.gz', '.tif', '.tiff'}

    def read(self, file_path: Path) -> DiffractionDataset:
        name_regex = re.compile(r'scan_(?P<scan>\d+)_img_(?P<frame>\d+)\.tiff?(?:\.gz)?')
        member = name_regex.fullmatch(file_path.name)

        if member is None:
            raise ValueError(
                f'"{file_path.name}" is not named scan_<N>_img_<F>.tif, .tiff or .tif.gz.'
            )

        scan = int(member['scan'])
        extension = file_path.name[member.end('frame') :]
        numbered: list[tuple[int, Path]] = []

        for candidate in file_path.parent.iterdir():
            match = name_regex.fullmatch(candidate.name)

            if match is not None and int(match['scan']) == scan:
                numbered.append((int(match['frame']), candidate))

        frame_files = [path for _, path in sorted(numbered)]

        # The areaDetector TIFF plugin writes one frame per file, so only the first member
        # is opened here; probing every member would inflate each gzipped one twice.
        with open_tiff(frame_files[0]) as tiff:
            page = tiff.pages[0]
            detector_height, detector_width = page.shape[-2:]
            pattern_dtype = _counts_dtype(numpy.dtype(page.dtype))

        contents_tree = DiffractionDatasetLayoutNode.create_root()
        array_list: list[DiffractionArray] = []

        for index, path in enumerate(frame_files):
            indexes: DiffractionIndexes = numpy.array([index])
            array = _MedipixTiffArray(path, indexes)
            contents_tree.add_child(array.get_label(), 'TIFF', str(index))
            array_list.append(array)

        bad_pixels: BadPixels = numpy.zeros((detector_height, detector_width), dtype=numpy.bool)
        rows = [i for i in self._seam_pixels if 0 <= i < detector_height]
        columns = [i for i in self._seam_pixels if 0 <= i < detector_width]
        bad_pixels[rows, :] = True
        bad_pixels[:, columns] = True

        scan_metadata = _read_mda_metadata(file_path.parent, scan)
        metadata = DiffractionMetadata(
            num_patterns_per_array=[1] * len(array_list),
            pattern_dtype=pattern_dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_distance_m=scan_metadata.detector_distance_m,
            detector_pixel_geometry=PixelGeometry(
                self._detector_pixel_size_m, self._detector_pixel_size_m
            ),
            photon_energy_eV=scan_metadata.photon_energy_eV,
            file_path=file_path.parent / f'scan_{scan}_img_(\\d+){extension}',
        )
        return SimpleDiffractionDataset(metadata, contents_tree, array_list, bad_pixels)


class APS26IDH5DiffractionFileReader(DiffractionFileReader):
    """The Eiger layout: ``/entry/data/data`` stacks beside an instrument snapshot.

    Any member of the scan may be opened, the snapshot included. `detector_pixel_size_m`
    is the Eiger pitch the files do not record. `frame_index_offset` is added to every
    pattern's index, for a scan whose frames lead or trail its points.
    """

    def __init__(
        self,
        *,
        detector_pixel_size_m: float = 75e-6,
        frame_index_offset: int = 0,
    ) -> None:
        self._detector_pixel_size_m = detector_pixel_size_m
        self._frame_index_offset = frame_index_offset

    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> DiffractionDataset:
        member = re.fullmatch(r'scan_(?P<scan>\d+)_\d+', file_path.stem)

        if member is None:
            raise ValueError(f'"{file_path.name}" is not named scan_<N>_<F>{file_path.suffix}.')

        scan = int(member['scan'])
        member_regex = re.compile(rf'scan_{scan}_(?P<number>\d+)')
        numbered: list[tuple[int, Path]] = []

        for candidate in file_path.parent.glob(f'scan_{scan}_*{file_path.suffix}'):
            match = member_regex.fullmatch(candidate.stem)

            if match is not None:
                numbered.append((int(match['number']), candidate))

        array_list: list[DiffractionArray] = []
        num_patterns_per_array: list[int] = []
        bad_pixels: BadPixels | None = None
        pattern_dtype = numpy.dtype(numpy.int32)
        detector_height = 0
        detector_width = 0
        snapshot_metadata = _ScanMetadata()
        contents_tree: DiffractionDatasetLayoutNode | None = None
        offset = self._frame_index_offset

        for _, path in sorted(numbered):
            with h5py.File(path, 'r') as h5_file:
                data = h5_file.get(_DATA_PATH)

                if isinstance(h5_file.get('/entry/instrument/26-ID-C'), h5py.Group):
                    snapshot_metadata = _read_snapshot_metadata(h5_file)
                    contents_tree = H5DiffractionFileTreeBuilder().build(h5_file)

                if not isinstance(data, h5py.Dataset) or data.ndim != 3:
                    continue

                num_patterns, detector_height, detector_width = data.shape
                pattern_dtype = data.dtype
                warn_on_dropped_frames(h5_file, path)

                if bad_pixels is None:
                    bad_pixels = numpy.asarray(data[0]) < 0

                if contents_tree is None:
                    contents_tree = H5DiffractionFileTreeBuilder().build(h5_file)

            indexes = numpy.arange(num_patterns) + offset
            offset += num_patterns
            num_patterns_per_array.append(num_patterns)
            array_list.append(H5DiffractionPatternArray(path.stem, indexes, path, _DATA_PATH))

        if not array_list:
            raise ValueError(f'No member of scan {scan} beside "{file_path}" holds {_DATA_PATH}.')

        scan_metadata = _read_mda_metadata(file_path.parent, scan).overridden_by(snapshot_metadata)
        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=pattern_dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_distance_m=scan_metadata.detector_distance_m,
            detector_pixel_geometry=PixelGeometry(
                self._detector_pixel_size_m, self._detector_pixel_size_m
            ),
            photon_energy_eV=scan_metadata.photon_energy_eV,
            exposure_time_s=scan_metadata.exposure_time_s,
            file_path=file_path,
        )
        return SimpleDiffractionDataset(
            metadata,
            contents_tree or DiffractionDatasetLayoutNode.create_root(),
            array_list,
            bad_pixels,
        )


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS26IDTiffDiffractionFileReader(),
        simple_name='CNM_APS_HXN_TIFF',
    )
    registry.diffraction_file_readers.register_plugin(
        APS26IDH5DiffractionFileReader(),
        simple_name='CNM_APS_HXN',
    )
