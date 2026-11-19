from dataclasses import dataclass
from pathlib import Path
from typing import Final
import logging
import math

import h5py
import numpy

from ptychodus.api.constants import EnergyUnit, LengthUnit
from ptychodus.api.diffraction import (
    BadPixels,
    CropRegion,
    DiffractionArray,
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPatternDType,
    DiffractionPatterns,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
)
from ptychodus.api.typing import AnyArrayType, IntegerArrayType, RealArrayType

# importing this module also registers the HDF5 compression filters before h5py reads
from ptychodus.plugins.h5_diffraction_file import H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)

_DESCRIPTION: Final[str] = 'LCLS XPP X-ray Pump Probe Files'

# piezo stage positions (x, y, z) in microns
_PIEZO_STAGE_PATHS: Final[tuple[str, str, str]] = ('/lmc/ch03', '/lmc/ch04', '/lmc/ch05')


def _read_optional(h5_file: h5py.File, path: str) -> AnyArrayType | None:
    item = h5_file.get(path)
    return numpy.asarray(item[()]) if isinstance(item, h5py.Dataset) else None


def _read_optional_scalar(h5_file: h5py.File, path: str) -> float | None:
    """Return the median of the finite values at `path`, or None without any."""
    values = _read_optional(h5_file, path)

    if values is None:
        return None

    real_values = numpy.asarray(values, dtype=numpy.float64)
    finite = real_values[numpy.isfinite(real_values)]
    return float(numpy.median(finite)) if finite.size > 0 else None


@dataclass(frozen=True)
class LCLSShotSelection:
    """Which shots of a psana smalldata file are kept.

    A shot is kept when the X-rays were on, the I0 monitor and the detector both recorded
    it, its I0 is positive and inside ``[i0_low_threshold, i0_high_threshold)``, and its
    piezo-stage readings are finite. A flag the file does not record is not checked.

    Shot-to-shot SASE fluctuations are large, and the window keeps the shots whose
    incident intensity is close to constant. Patterns are paired with positions by shot
    index, so both readers of a file must select with the same rules for every pattern
    to meet the position it was measured at.
    """

    detector_name: str = 'jungfrau1M'
    i0_name: str = 'ipm2'
    i0_low_threshold: float = 2000.0
    i0_high_threshold: float = 3000.0

    def read_i0(self, h5_file: h5py.File) -> RealArrayType:
        return numpy.asarray(h5_file[f'/{self.i0_name}/sum'][()], dtype=numpy.float64)

    def select(self, h5_file: h5py.File) -> IntegerArrayType:
        """Return the indexes of the kept shots, in increasing order.

        Raises:
            ValueError: if no shot passes every rule.
        """
        i0 = self.read_i0(h5_file)
        rules: dict[str, numpy.typing.NDArray[numpy.bool]] = {}

        xray = _read_optional(h5_file, '/lightStatus/xray')

        if xray is not None:
            rules['X-rays off'] = xray != 1

        for name in (self.i0_name, self.detector_name):
            present = _read_optional(h5_file, f'/damage/{name}')

            if present is not None:
                rules[f'{name} not recorded'] = present != 1

        rules[f'{self.i0_name} outside [{self.i0_low_threshold}, {self.i0_high_threshold})'] = ~(
            (i0 > 0.0) & (self.i0_low_threshold <= i0) & (i0 < self.i0_high_threshold)
        )

        stage_finite = numpy.ones(i0.shape, dtype=numpy.bool)

        for path in _PIEZO_STAGE_PATHS:
            stage_finite &= numpy.isfinite(numpy.asarray(h5_file[path][()], dtype=numpy.float64))

        rules['piezo stage not finite'] = ~stage_finite

        dropped = numpy.zeros(i0.shape, dtype=numpy.bool)

        for reason, fails in rules.items():
            logger.info(f'{numpy.count_nonzero(fails)} of {i0.size} shots fail: {reason}.')
            dropped |= fails

        kept = numpy.flatnonzero(~dropped)

        if kept.size == 0:
            raise ValueError(f'No shot passes every selection rule ({", ".join(rules)}).')

        logger.info(f'Kept {kept.size} of {i0.size} shots.')
        return kept


class _LCLSPatternArray(DiffractionArray):
    """The selected frames of a calibrated smalldata image, converted to photons.

    psana records the energy deposited in each pixel in keV, so dividing by the photon
    energy in keV counts photons.
    """

    def __init__(
        self,
        label: str,
        indexes: DiffractionIndexes,
        file_path: Path,
        data_path: str,
        photon_energy_keV: float,  # noqa: N803
        dtype: DiffractionPatternDType,
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._file_path = file_path
        self._data_path = data_path
        self._photon_energy_keV = photon_energy_keV
        self._dtype = dtype

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        y_slice = slice(None) if read_region is None else read_region.y_slice
        x_slice = slice(None) if read_region is None else read_region.x_slice

        with h5py.File(self._file_path, 'r') as h5_file:
            frames = h5_file[self._data_path][self._indexes, y_slice, x_slice]

        return numpy.divide(frames, self._photon_energy_keV, dtype=self._dtype)


class LCLSDiffractionFileReader(DiffractionFileReader):
    """Calibrated, assembled detector images from a psana smalldata file, in photons.

    Only the shots `selection` keeps are read. The photon energy is required, since the
    image records keV deposited; the detector distance is an experiment's own PV, and is
    left unset when the file lacks it.
    """

    def __init__(
        self,
        selection: LCLSShotSelection = LCLSShotSelection(),
        *,
        photon_energy_keV_path: str = '/epics/lom_E',  # noqa: N803
        detector_distance_mm_path: str = '/epicsUser/sample_det_dis',
    ) -> None:
        self._selection = selection
        self._photon_energy_keV_path = photon_energy_keV_path
        self._detector_distance_mm_path = detector_distance_mm_path
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> DiffractionDataset:
        detector_name = self._selection.detector_name
        data_path = f'/{detector_name}/image_img'
        config_path = f'/UserDataCfg/{detector_name}'

        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)
            data = h5_file[data_path]

            if not isinstance(data, h5py.Dataset):
                raise ValueError(f'Expected {data_path} to be a dataset; got {type(data)}.')

            _, detector_height, detector_width = data.shape
            indexes = self._selection.select(h5_file)

            photon_energy_keV = _read_optional_scalar(h5_file, self._photon_energy_keV_path)  # noqa: N806

            if photon_energy_keV is None or photon_energy_keV <= 0.0:
                raise ValueError(
                    f'"{file_path}" records no photon energy at "{self._photon_energy_keV_path}",'
                    ' which converts its keV images to photons.'
                )

            pixel_size_m = _read_optional_scalar(h5_file, f'{config_path}/pixelsize')
            detector_distance_mm = _read_optional_scalar(h5_file, self._detector_distance_mm_path)

            bad_pixels: BadPixels | None = None
            mask = _read_optional(h5_file, f'{config_path}/image__image_mask_img')

            # The image mask is nonzero at panel gaps and masked pixels.
            if mask is not None and mask.shape == (detector_height, detector_width):
                bad_pixels = mask != 0
            else:
                logger.warning(f'"{file_path}" has no image mask; no pixel is marked bad.')

            pattern_dtype = numpy.result_type(data.dtype, numpy.float32)

        metadata = DiffractionMetadata(
            num_patterns_per_array=[indexes.size],
            pattern_dtype=pattern_dtype,
            detector_extent=ImageExtent(detector_width, detector_height),
            detector_distance_m=None
            if detector_distance_mm is None
            else detector_distance_mm * LengthUnit.MILLIMETER.meters_per_unit,
            detector_pixel_geometry=None
            if pixel_size_m is None
            else PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m),
            photon_energy_eV=EnergyUnit.KILOELECTRONVOLT.to_electronvolts(photon_energy_keV),
            file_path=file_path,
        )
        array = _LCLSPatternArray(
            label=file_path.stem,
            indexes=indexes,
            file_path=file_path,
            data_path=data_path,
            photon_energy_keV=photon_energy_keV,
            dtype=pattern_dtype,
        )
        return SimpleDiffractionDataset(metadata, contents_tree, [array], bad_pixels)


class LCLSPositionFileReader(ProbePositionFileReader):
    """Piezo-stage positions of the shots `selection` keeps, each with its I0."""

    SIMPLE_NAME: Final[str] = 'LCLS_XPP'  # noqa: N806

    def __init__(
        self,
        selection: LCLSShotSelection = LCLSShotSelection(),
        *,
        tomography_angle_deg: float = 180.0,
    ) -> None:
        self._selection = selection
        self._tomography_angle_deg = tomography_angle_deg

    def get_file_description(self) -> str:
        return _DESCRIPTION

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> ProbePositionSequence:
        with h5py.File(file_path, 'r') as h5_file:
            indexes = self._selection.select(h5_file)
            i0 = self._selection.read_i0(h5_file)[indexes]
            stage_x_um, stage_y_um, stage_z_um = (
                numpy.asarray(h5_file[path][()], dtype=numpy.float64)[indexes]
                for path in _PIEZO_STAGE_PATHS
            )

        meters_per_micrometer = LengthUnit.MICROMETER.meters_per_unit

        # vertical coordinate
        y_m = -stage_z_um * meters_per_micrometer

        # horizontal coordinate
        tomography_angle_rad = math.radians(self._tomography_angle_deg)
        x_m = (
            math.cos(tomography_angle_rad) * stage_x_um
            + math.sin(tomography_angle_rad) * stage_y_um
        ) * meters_per_micrometer

        return ProbePositionSequence(
            [
                ProbePosition(int(index), float(x), float(y), probe_photon_count=float(count))
                for index, x, y, count in zip(indexes, x_m, y_m, i0)
            ]
        )


def register_plugins(registry: PluginRegistry) -> None:
    selection = LCLSShotSelection()

    registry.diffraction_file_readers.register_plugin(
        LCLSDiffractionFileReader(selection),
        simple_name=LCLSPositionFileReader.SIMPLE_NAME,
    )
    registry.probe_position_file_readers.register_plugin(
        LCLSPositionFileReader(selection),
        simple_name=LCLSPositionFileReader.SIMPLE_NAME,
    )
