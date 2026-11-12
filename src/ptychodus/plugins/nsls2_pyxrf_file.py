"""Fluorescence reader for the HDF5 files PyXRF fits its element maps into.

PyXRF, the NSLS-II fluorescence fitting software, appends its fit to the raw-data file of a
scan (``NSLS2-XRF-MAP``): ``xrfmap/detsum/xrf_fit`` holds one fitted map per entry of
``xrfmap/detsum/xrf_fit_name``, in counts and normalized by no scaler. The reader keeps the
emission lines and scatter peaks, drops PyXRF's fit diagnostics, and divides by the dwell
time PyXRF records in ``xrfmap/scan_metadata`` to give counts per second. Files from older
PyXRF writers carry no scan metadata; their maps are read as they are.
"""

from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.files import FileFormat
from ptychodus.api.fluorescence import ElementMap, FluorescenceDataset, FluorescenceFileReader
from ptychodus.api.plugins import PluginRegistry

logger = logging.getLogger(__name__)


class PyXRFFileReader(FluorescenceFileReader):
    SIMPLE_NAME: Final[str] = 'PyXRF'

    MAPS_PATH: Final[str] = '/xrfmap/detsum/xrf_fit'
    NAMES_PATH: Final[str] = '/xrfmap/detsum/xrf_fit_name'
    SCAN_METADATA_PATH: Final[str] = '/xrfmap/scan_metadata'
    DWELL_ATTRIBUTE: Final[str] = 'param_dwell'

    def get_file_description(self) -> str:
        return 'NSLS-II PyXRF Fluorescence Maps'

    def get_file_extensions(self) -> set[str]:
        return FileFormat.HDF5.extensions

    def read(self, file_path: Path) -> FluorescenceDataset:
        # Fit quality and background entries PyXRF stores beside the element maps.
        diagnostic_names = frozenset({'snip_bkg', 'r_factor', 'sel_cnt', 'total_cnt'})

        with h5py.File(file_path, 'r') as h5_file:
            h5_maps = h5_file.get(self.MAPS_PATH)

            if not isinstance(h5_maps, h5py.Dataset):
                raise ValueError(f'No PyXRF fitted maps at "{self.MAPS_PATH}" in "{file_path}".')

            h5_names = h5_file.get(self.NAMES_PATH)

            if not isinstance(h5_names, h5py.Dataset):
                raise ValueError(f'No PyXRF map names at "{self.NAMES_PATH}" in "{file_path}".')

            maps = h5_maps[()]
            names = [_decode_name(raw) for raw in h5_names[()]]

            if len(names) != maps.shape[0]:
                raise ValueError(
                    f'"{file_path}" names {len(names)} maps but holds {maps.shape[0]}.'
                )

            dwell_s = None
            h5_scan_metadata = h5_file.get(self.SCAN_METADATA_PATH)

            if isinstance(h5_scan_metadata, h5py.Group):
                dwell_s = h5_scan_metadata.attrs.get(self.DWELL_ATTRIBUTE)

        if dwell_s is not None and float(dwell_s) > 0.0:
            maps = maps / float(dwell_s)
        else:
            logger.warning(
                f'"{file_path}" records no dwell time; its maps are read in the units '
                'the file holds rather than converted to counts per second.'
            )

        element_maps = [
            ElementMap(name, numpy.asarray(element_map, dtype=numpy.float64))
            for name, element_map in zip(names, maps)
            if name not in diagnostic_names
        ]

        return FluorescenceDataset(
            _element_maps=element_maps,
            counts_per_second_path=self.MAPS_PATH,
            channel_names_path=self.NAMES_PATH,
        )


def _decode_name(raw: bytes | str) -> str:
    """A map name as PyXRF stores it, fixed-length bytes or str, without padding."""
    name = raw.decode('utf-8', errors='replace') if isinstance(raw, bytes) else str(raw)
    return name.rstrip('\x00').strip()


def register_plugins(registry: PluginRegistry) -> None:
    registry.fluorescence_file_readers.register_plugin(
        PyXRFFileReader(),
        simple_name=PyXRFFileReader.SIMPLE_NAME,
    )
