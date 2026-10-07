from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.constants import EnergyUnit, LengthUnit
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    BeamCenter,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.io import resolve_external_link_path
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class PolarDiffractionFileReader(DiffractionFileReader):
    # POLAR runs an Eiger; the files record no pitch, so supply the detector's own.
    DETECTOR_PIXEL_SIZE_M: Final[float] = 75e-6
    NDARRAY_UNIQUE_ID_PATH: Final[str] = '/entry/instrument/NDAttributes/NDArrayUniqueId'
    DETECTOR_DISTANCE_PATH: Final[str] = '/entry/instrument/NDAttributes/DetectorDistance'
    BEAM_CENTER_X_PATH: Final[str] = '/entry/instrument/NDAttributes/BeamCenterX'
    BEAM_CENTER_Y_PATH: Final[str] = '/entry/instrument/NDAttributes/BeamCenterY'
    MONO_ENERGY_PATH: Final[str] = (
        '/entry/instrument/bluesky/streams/baseline/mono_energy/value_start'
    )

    def __init__(self) -> None:
        self._data_path = '/entry/externals/eiger'
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def _read_pattern_indexes(self, h5_file: h5py.File, num_patterns: int) -> numpy.ndarray:
        """Return per-pattern indexes.

        Uses the Eiger detector's NDArrayUniqueId when the file exposes it (this reader
        already opens the external Eiger file where the UID lives), normalized to
        ``uid - uid[0] + 1``. The position reader documents how that 1-based convention
        lines up with each of the two position layouts.
        """
        try:
            uid = h5_file[self.NDARRAY_UNIQUE_ID_PATH][()]
        except KeyError:
            logger.warning(
                'NDArrayUniqueId not found; falling back to sequential indexes '
                '(gap-preserving alignment with dropped Eiger frames not possible).'
            )
            return numpy.arange(1, num_patterns + 1, dtype=numpy.int64)

        if uid.shape[0] != num_patterns:
            raise ValueError(
                f'NDArrayUniqueId length {uid.shape[0]} != pattern count {num_patterns}.'
            )
        return (uid - int(uid[0]) + 1).astype(numpy.int64)

    def _read_photon_energy_eV(self, h5_file: h5py.File) -> float | None:  # noqa: N802
        """Read the monochromator energy from the master's baseline stream.

        Present in both the old and the new layout, unlike the detector attributes.
        """
        try:
            energy_keV = float(h5_file[self.MONO_ENERGY_PATH][()])  # noqa: N806
        except KeyError:
            return None
        else:
            return EnergyUnit.KILOELECTRONVOLT.to_electronvolts(energy_keV)

    def _read_detector_distance_m(self, h5_file: h5py.File) -> float | None:
        """Read the sample-detector distance from the Eiger per-frame attributes.

        The attribute is a per-frame array holding a single distinct value; take the
        first. ``DistancePV`` duplicates it and is not read.

        Only the newest layout writes it. The 2025-2 and 2026-2 layouts omit the
        attribute entirely, so ``None`` is a normal result and the caller is expected to
        supply the distance itself.
        """
        try:
            distance_mm = float(h5_file[self.DETECTOR_DISTANCE_PATH][0])
        except (KeyError, IndexError):
            return None
        else:
            return LengthUnit.MILLIMETER.to_meters(distance_mm)

    def _read_beam_center(self, h5_file: h5py.File) -> BeamCenter | None:
        """Read the direct-beam center from the Eiger per-frame attributes.

        ``BeamCenterPV`` is not read: the file describes it as
        "Eiger pixel center - doesnt work", and it merely duplicates ``BeamCenterX``.

        Only the newest layout writes these. The 2025-2 and 2026-2 layouts omit them
        entirely, so ``None`` is a normal result and the caller is expected to supply or
        estimate the center itself.
        """
        try:
            center_x_px = float(h5_file[self.BEAM_CENTER_X_PATH][0])
            center_y_px = float(h5_file[self.BEAM_CENTER_Y_PATH][0])
        except (KeyError, IndexError):
            return None
        else:
            return BeamCenter(int(round(center_x_px)), int(round(center_y_px)))

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)
            data_link = h5_file.get(self._data_path, getlink=True)
            photon_energy_eV = self._read_photon_energy_eV(h5_file)  # noqa: N806

        if not isinstance(data_link, h5py.ExternalLink):
            raise ValueError(
                f'Expected "{self._data_path}" to be an external link; got {type(data_link)}.'
            )

        data_file_path = resolve_external_link_path(file_path.parent, data_link.filename)
        logger.debug(f'Opening "{data_file_path}"...')

        with h5py.File(data_file_path, 'r') as h5_file:
            data = h5_file[data_link.path]

            if isinstance(data, h5py.Group):
                # Both the old and the new master link to /entry/instrument, so this is
                # the normal case rather than an anomaly.
                logger.debug('Link points to group; falling back to "/entry/data/data"')
                data = h5_file['/entry/data/data']

            if isinstance(data, h5py.Dataset):
                num_patterns, detector_height, detector_width = data.shape

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_patterns],
                    pattern_dtype=data.dtype,
                    detector_distance_m=self._read_detector_distance_m(h5_file),
                    detector_extent=ImageExtent(detector_width, detector_height),
                    detector_pixel_geometry=PixelGeometry(
                        width_m=self.DETECTOR_PIXEL_SIZE_M,
                        height_m=self.DETECTOR_PIXEL_SIZE_M,
                    ),
                    beam_center=self._read_beam_center(h5_file),
                    photon_energy_eV=photon_energy_eV,
                    file_path=file_path,
                )

                indexes = self._read_pattern_indexes(h5_file, num_patterns)

                array = H5DiffractionPatternArray(
                    label=file_path.stem,
                    indexes=indexes,
                    file_path=data_file_path,
                    data_path=data.name,
                )
            else:
                raise ValueError(f'Expected "{data.name}" to be a dataset; got {type(data)}.')

        return SimpleDiffractionDataset(metadata, contents_tree, [array])


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        PolarDiffractionFileReader(),
        simple_name='APS_Polar',
        display_name='APS 4-ID-B,G,H POLAR Files (*.hdf)',
    )
