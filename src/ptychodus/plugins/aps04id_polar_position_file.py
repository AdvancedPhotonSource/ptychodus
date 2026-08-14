from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.constants import ONE_MICRON_M
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)


class PolarPositionFileReader(ProbePositionFileReader):
    EIGER_EXTERNAL_LINK: Final[str] = '/entry/externals/eiger'
    NDARRAY_UNIQUE_ID_PATH: Final[str] = '/entry/instrument/NDAttributes/NDArrayUniqueId'

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with h5py.File(file_path, 'r') as h5_file:
            try:
                position_x = h5_file['/entry/instrument/NDAttributes/Xpos'][()]
            except KeyError:
                position_x = h5_file[
                    '/entry/instrument/bluesky/streams/primary/huber_hp_nanox/value'
                ][()]

            try:
                position_y = h5_file['/entry/instrument/NDAttributes/Ypos'][()]
            except KeyError:
                position_y = h5_file[
                    '/entry/instrument/bluesky/streams/primary/huber_hp_nanoy/value'
                ][()]

            if position_x.shape == position_y.shape:
                logger.debug(f'Coordinate arrays have shape {position_x.shape}.')
            else:
                raise ProbePositionParseError('Coordinate array shape mismatch!')

            indexes = self._read_trigger_indexes(h5_file, file_path, position_x.shape[0])

        for idx, x, y in zip(indexes, position_x, position_y):
            point_list.append(
                ProbePosition(
                    int(idx),
                    float(x) * ONE_MICRON_M,
                    float(y) * ONE_MICRON_M,
                )
            )

        return ProbePositionSequence(point_list)

    def _read_trigger_indexes(
        self, h5_file: h5py.File, file_path: Path, n_frames: int
    ) -> numpy.ndarray:
        """Return per-frame trigger indexes.

        Uses the Eiger detector's NDArrayUniqueId when reachable — either
        directly in this file, or via the /entry/externals/eiger external
        link on the master wrapper. Normalizes to per-scan 0-based, then
        shifts by +1 to match the flyscan convention that the detector
        skips pos_stream trigger 0 (see process_flyscan.plot_data).
        """
        uid = self._try_read_uid_here(h5_file, n_frames)
        if uid is None:
            uid = self._try_read_uid_via_external_link(h5_file, file_path, n_frames)
        if uid is not None:
            return (uid - int(uid[0]) + 1).astype(int)

        logger.warning(
            'NDArrayUniqueId not found; falling back to sequential indexes '
            '(gap-preserving alignment with dropped Eiger frames not possible).'
        )
        return numpy.arange(1, n_frames + 1, dtype=int)

    def _try_read_uid_here(self, h5_file: h5py.File, n_frames: int) -> numpy.ndarray | None:
        try:
            uid = h5_file[self.NDARRAY_UNIQUE_ID_PATH][()]
        except KeyError:
            return None
        if uid.shape[0] != n_frames:
            logger.debug(
                f'NDArrayUniqueId length {uid.shape[0]} != frame count {n_frames}; ignoring.'
            )
            return None
        return uid

    def _try_read_uid_via_external_link(
        self, h5_file: h5py.File, file_path: Path, n_frames: int
    ) -> numpy.ndarray | None:
        link = h5_file.get(self.EIGER_EXTERNAL_LINK, getlink=True)
        if not isinstance(link, h5py.ExternalLink):
            return None
        target = file_path.parent / link.filename
        try:
            with h5py.File(target, 'r') as ext:
                return self._try_read_uid_here(ext, n_frames)
        except OSError as ex:
            logger.debug(f'Could not open Eiger external link at {target}: {ex}')
            return None


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        PolarPositionFileReader(),
        simple_name='APS_Polar',
        display_name='APS 4-ID Polar Files (*.hdf)',
    )
