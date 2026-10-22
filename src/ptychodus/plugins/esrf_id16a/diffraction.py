"""Diffraction reader for ESRF ID16A near-field ptychography scans.

The frames are Fresnel holograms rather than far-field diffraction patterns, so the
dataset declares ``far_field=False`` and the focus position the geometry implies. A
product built from it then samples the object at the detector pitch demagnified,
which is what makes the recorded object pixel size come out.

Frames are handed on as they are stored. The Ximea sensor carries a pedestal of
roughly a sixth of the signal, which a contrast measurement cannot keep, but removing
it is the detector-background machinery's job: point the background settings at the
``dark_0000.h5`` beside the scan, read by the companion
:class:`~ptychodus.plugins.esrf_id16a.pixel_maps.ESRFID16ADarkFileReader`.
"""

from __future__ import annotations

from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.diffraction import (
    DiffractionArray,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry

from ..h5_diffraction_file import H5DiffractionFileTreeBuilder, H5DiffractionPatternArray
from ._geometry import compute_geometry
from ._scan import (
    detector_data_path,
    detector_group,
    read_header,
    read_sample_x0_mm,
    series_files,
)

logger = logging.getLogger(__name__)


class ESRFID16ANearFieldDiffractionFileReader(DiffractionFileReader):
    """Read an ID16A near-field scan from its master or any of its frame files.

    `cross_check_rtol` bounds the agreement between the focus distance the detector
    header implies and the one the sample stage reports; a wider gap is logged. It is
    a tolerance on two independent measurements of the same distance, so a caller
    working with a differently calibrated stage may want to widen it.
    """

    def __init__(self, *, cross_check_rtol: float = 1.0e-3) -> None:
        self._cross_check_rtol = cross_check_rtol

    def read(self, file_path: Path) -> DiffractionDataset:
        frame_files = series_files(file_path)
        sample_x0_mm = read_sample_x0_mm(file_path)

        num_patterns_per_array: list[int] = []
        array_list: list[DiffractionArray] = []
        tree_builder = H5DiffractionFileTreeBuilder()
        contents_tree = None
        detector_height = 0
        detector_width = 0
        pattern_dtype = numpy.dtype(numpy.uint16)
        geometry = None
        exposure_time_s = None
        offset = 0

        for frame_file in frame_files:
            with h5py.File(frame_file, 'r') as h5_file:
                data_path = detector_data_path(h5_file)
                dataset = h5_file[data_path]

                if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 3:
                    logger.warning('Skipping %s: "%s" is not a 3D dataset.', frame_file, data_path)
                    continue

                num_patterns, detector_height, detector_width = dataset.shape
                pattern_dtype = dataset.dtype

                if contents_tree is None:
                    contents_tree = tree_builder.build(h5_file)

                # Every frame file of a scan shares one geometry, so the first readable
                # one settles it; a later file disagreeing would be a different scan.
                if geometry is None:
                    header = read_header(h5_file)
                    geometry = compute_geometry(
                        header,
                        sample_x0_mm=sample_x0_mm,
                        cross_check_rtol=self._cross_check_rtol,
                    )
                    exposure_time_s = _read_exposure_time_s(h5_file)

            indexes = numpy.arange(num_patterns) + offset
            offset += num_patterns
            num_patterns_per_array.append(num_patterns)
            array_list.append(
                H5DiffractionPatternArray(frame_file.stem, indexes, frame_file, data_path)
            )

        if geometry is None or contents_tree is None:
            raise ValueError(f'No readable ESRF ID16A frame files beside "{file_path}".')

        metadata = DiffractionMetadata(
            num_patterns_per_array=num_patterns_per_array,
            pattern_dtype=pattern_dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_distance_m=geometry.detector_distance_m,
            focus_object_distance_m=geometry.focus_object_distance_m,
            far_field=False,
            detector_pixel_geometry=PixelGeometry(
                width_m=geometry.detector_pixel_size_m,
                height_m=geometry.detector_pixel_size_m,
            ),
            photon_energy_eV=geometry.photon_energy_eV,
            exposure_time_s=exposure_time_s,
            file_path=file_path,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, array_list)


def _read_exposure_time_s(h5_file: h5py.File) -> float | None:
    """Per-frame exposure from the LIMA acquisition collection, when it records one."""
    acquisition = detector_group(h5_file).get('acquisition')

    if not isinstance(acquisition, h5py.Group):
        return None

    exposure_time = acquisition.get('exposure_time')

    if not isinstance(exposure_time, h5py.Dataset):
        return None

    return float(exposure_time[()])
