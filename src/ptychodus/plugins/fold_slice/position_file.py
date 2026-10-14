from dataclasses import dataclass
from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)

_WAVELENGTH_PATH = '/lambda'
_OBJECT_PIXEL_SIZE_PATH = '/dx'
_TOMOGRAPHY_ANGLE_PATH = '/angle'


@dataclass(frozen=True)
class FoldSliceParameters:
    """Experiment geometry a fold_slice parameter file records beside the probe positions.

    Every field is optional because the preprocessing step writes them only for a
    measured scan; a simulated dataset carries the coordinates alone. The sample-plane
    pixel size stands in for the sample-to-detector distance, which the file does not
    record directly -- the two are related by the Fraunhofer expression
    ``dx_sample = lambda * z / (N * dx_detector)``, so the distance follows once the
    detector pitch and the pattern width are known.
    """

    photon_wavelength_m: float | None = None
    object_pixel_size_m: float | None = None
    tomography_angle_deg: float | None = None


def _read_scalar(h5_file: h5py.File, data_path: str) -> float | None:
    """Read a one-element dataset as a float, or None when the file omits it."""
    dataset = h5_file.get(data_path)

    if not isinstance(dataset, h5py.Dataset):
        return None

    value = numpy.squeeze(dataset[()])

    if value.ndim != 0:
        logger.warning(f'Ignoring "{data_path}": expected one value, got shape {value.shape}.')
        return None

    return float(value)


def read_fold_slice_parameters(file_path: Path) -> FoldSliceParameters:
    """Read the experiment geometry from a fold_slice parameter file.

    Returns empty fields rather than raising when the file records none, so a simulated
    dataset -- which carries only coordinates -- reads the same way as a measured one.
    """
    with h5py.File(file_path, 'r') as h5_file:
        return FoldSliceParameters(
            photon_wavelength_m=_read_scalar(h5_file, _WAVELENGTH_PATH),
            object_pixel_size_m=_read_scalar(h5_file, _OBJECT_PIXEL_SIZE_PATH),
            tomography_angle_deg=_read_scalar(h5_file, _TOMOGRAPHY_ANGLE_PATH),
        )


class FoldSlicePositionFileReader(ProbePositionFileReader):
    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                pp_x = numpy.squeeze(h5_file['/ppX'])
                pp_y = numpy.squeeze(h5_file['/ppY'])
            except KeyError:
                logger.warning('Unable to find data.')
            else:
                if pp_x.shape == pp_y.shape:
                    logger.debug(f'Coordinate arrays have shape {pp_x.shape}.')
                else:
                    raise ProbePositionParseError('Coordinate array shape mismatch!')

                for idx, (x, y) in enumerate(zip(pp_x, pp_y)):
                    point = ProbePosition(idx, x, y)
                    point_list.append(point)

        return ProbePositionSequence(point_list)
