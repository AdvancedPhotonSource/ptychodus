"""Reader for APS 4-ID-B,G,H POLAR probe positions, entered through the master file.

POLAR records positions two different ways, and the two carry *different* index
conventions against the diffraction reader. Both are correct; neither is documented
anywhere at the instrument, so they are spelled out here.

Fly scans (``plan_name == 'flyscan'``) stream positions to ``pos_stream/scan_NNNNNN.h5``,
which the master advertises as an external link. Those positions are indexed by the raw
softGlueZynq trigger counter, which starts at **0**, while the diffraction reader emits
``uid - uid[0] + 1``, which starts at **1**. Frame ``k`` therefore pairs with trigger
``k + 1``, leaving trigger 0 unmatched -- the offset ``process_flyscan.plot_data``
implements as ``x[1:]``.

Step scans (``grid_scan``, ``rel_grid_scan``, ...) record motor readbacks in the bluesky
``primary`` stream. Those positions are indexed from the same eiger ``NDArrayUniqueId``
the diffraction reader uses, by the same ``uid - uid[0] + 1`` expression, so the ``+1``
cancels on both sides and frame ``k`` pairs with position ``k``: strict 1:1.

The eiger ``NDAttributes/Xpos`` and ``Ypos`` arrays are deliberately **not** read. They
are identically zero in every POLAR file available to us, and reading them yields a
full-length scan sitting at the origin with no error raised.

Both layouts negate x and y. These files record the stage readback, and moving the stage
``+x`` moves the probe ``-x`` relative to the sample, so the flip converts to the
probe-relative frame. The fly-scan sign is verified against the beamline's own
``4idd_data_preprocessing_flyscan_v2.py`` (``ppX = -xs[1:]``, ``ppY = -ys[1:]``); the
step-scan sign is **inferred** from it, since no reference script covers that path. It is
the same physical stage recorded through bluesky rather than softGlueZynq, so the same
convention should hold -- but a step scan that comes out mirrored is corrected with
``[ProbePositions] Affine00 = Affine11 = -1`` rather than by editing this reader.
"""

from pathlib import Path
from typing import Any, Final
import logging

import h5py
import numpy
import yaml

from ptychodus.api.constants import LengthUnit
from ptychodus.api.io import resolve_external_link_path
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

from .aps04id_polar_sgz_position_file import read_position_stream

logger = logging.getLogger(__name__)


class _BlueskyMetadataLoader(yaml.SafeLoader):
    """SafeLoader that also accepts the ``!!python/tuple`` tag bluesky writes.

    Older POLAR masters serialize ``motors`` as a tagged tuple, which ``safe_load``
    rejects outright; newer ones use a plain list. Adding one sequence constructor
    covers both without ``full_load``'s arbitrary-object construction.
    """


def _construct_tuple(loader: yaml.SafeLoader, node: yaml.SequenceNode) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


_BlueskyMetadataLoader.add_constructor('tag:yaml.org,2002:python/tuple', _construct_tuple)


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class PolarPositionFileReader(ProbePositionFileReader):
    POS_STREAM_EXTERNAL_LINK: Final[str] = '/entry/externals/pos_stream'
    EIGER_EXTERNAL_LINK: Final[str] = '/entry/externals/eiger'
    NDARRAY_UNIQUE_ID_PATH: Final[str] = '/entry/instrument/NDAttributes/NDArrayUniqueId'
    PRIMARY_STREAM_PATH: Final[str] = '/entry/instrument/bluesky/streams/primary'
    MOTORS_PATH: Final[str] = '/entry/instrument/bluesky/metadata/motors'
    PLAN_NAME_PATH: Final[str] = '/entry/plan_name'
    DEFAULT_MOTORS: Final[tuple[str, str]] = ('huber_hp_nanox', 'huber_hp_nanoy')

    def read(self, file_path: Path) -> ProbePositionSequence:
        with h5py.File(file_path, 'r') as h5_file:
            plan_name = self._read_plan_name(h5_file)

            link = h5_file.get(self.POS_STREAM_EXTERNAL_LINK, getlink=True)

            if isinstance(link, h5py.ExternalLink):
                # Only the link's filename is used. Its target is /entry/instrument,
                # which is not where the stream data lives.
                stream_path = resolve_external_link_path(file_path.parent, link.filename)
                logger.debug(f'Fly scan ({plan_name}); reading positions from "{stream_path}".')
                return read_position_stream(stream_path)

            primary = h5_file.get(self.PRIMARY_STREAM_PATH)

            if isinstance(primary, h5py.Group):
                logger.debug(f'Step scan ({plan_name}); reading positions from primary stream.')
                return self._read_primary_stream(h5_file, primary, file_path, plan_name)

        raise ProbePositionParseError(
            f'"{file_path.name}" (plan_name "{plan_name}") matches no known POLAR position '
            f'layout. Fly scans carry an external link at "{self.POS_STREAM_EXTERNAL_LINK}"; '
            f'step scans carry a bluesky stream at "{self.PRIMARY_STREAM_PATH}". Neither is '
            "present. Select the scan's master file rather than an eiger or pos_stream file; "
            'if this is the master, the scan recorded no probe positions and is not a '
            'ptychography scan.'
        )

    def _read_plan_name(self, h5_file: h5py.File) -> str:
        try:
            return _decode(h5_file[self.PLAN_NAME_PATH][()])
        except KeyError:
            return 'unknown'

    def _read_motor_names(self, h5_file: h5py.File) -> tuple[str, str]:
        """Return the (x, y) motor names, in the order the scan listed them."""
        try:
            motors_raw = h5_file[self.MOTORS_PATH][()]
        except KeyError:
            logger.warning(
                f'"{self.MOTORS_PATH}" not found; assuming {self.DEFAULT_MOTORS}.',
            )
            return self.DEFAULT_MOTORS

        motors = yaml.load(_decode(motors_raw), Loader=_BlueskyMetadataLoader)

        if not isinstance(motors, (list, tuple)) or len(motors) < 2:
            raise ProbePositionParseError(
                f'Expected at least two motor names at "{self.MOTORS_PATH}"; got {motors!r}.'
            )

        return str(motors[0]), str(motors[1])

    def _read_primary_stream(
        self, h5_file: h5py.File, primary: h5py.Group, file_path: Path, plan_name: str
    ) -> ProbePositionSequence:
        motor_x, motor_y = self._read_motor_names(h5_file)
        # The first motor listed is mapped to x and the second to y. That is scan order,
        # not a physical guarantee -- a transposed scan is corrected by the [Scan] affine.
        logger.info(f'Step scan ({plan_name}) positions: x from "{motor_x}", y from "{motor_y}".')

        coordinates: list[numpy.ndarray] = []

        for motor in (motor_x, motor_y):
            value = primary.get(f'{motor}/value')

            if not isinstance(value, h5py.Dataset):
                raise ProbePositionParseError(
                    f'Motor "{motor}" named by "{self.MOTORS_PATH}" has no "value" dataset in '
                    f'"{self.PRIMARY_STREAM_PATH}"; found {sorted(primary)[:8]}...'
                )

            coordinates.append(value[()])

        position_x, position_y = coordinates

        if position_x.shape != position_y.shape:
            raise ProbePositionParseError(
                f'Coordinate array shape mismatch: "{motor_x}" is {position_x.shape} '
                f'but "{motor_y}" is {position_y.shape}.'
            )

        logger.debug(f'Coordinate arrays have shape {position_x.shape}.')
        indexes = self._read_trigger_indexes(h5_file, file_path, position_x.shape[0])

        return ProbePositionSequence(
            [
                ProbePosition(
                    int(idx),
                    -LengthUnit.MICROMETER.to_meters(float(x)),
                    -LengthUnit.MICROMETER.to_meters(float(y)),
                )
                for idx, x, y in zip(indexes, position_x, position_y)
            ]
        )

    def _read_trigger_indexes(
        self, h5_file: h5py.File, file_path: Path, n_frames: int
    ) -> numpy.ndarray:
        """Return per-frame trigger indexes.

        Uses the eiger detector's NDArrayUniqueId when reachable -- either directly in
        this file, or via the /entry/externals/eiger external link on the master wrapper.
        Normalizes to ``uid - uid[0] + 1``, which is exactly what the diffraction reader
        emits, so step-scan positions pair 1:1 with patterns.
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
            # Reachable on aborted scans, where the detector and the scan engine stop at
            # different points. The sequential fallback still matches the diffraction
            # reader whenever the UID is contiguous, but it cannot preserve gaps.
            logger.warning(
                f'NDArrayUniqueId length {uid.shape[0]} != position count {n_frames}; '
                'ignoring it and falling back to sequential indexes. Expected on an '
                'aborted scan; gap-preserving alignment is off.'
            )
            return None
        return uid

    def _try_read_uid_via_external_link(
        self, h5_file: h5py.File, file_path: Path, n_frames: int
    ) -> numpy.ndarray | None:
        link = h5_file.get(self.EIGER_EXTERNAL_LINK, getlink=True)
        if not isinstance(link, h5py.ExternalLink):
            return None
        target = resolve_external_link_path(file_path.parent, link.filename)
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
        display_name='APS 4-ID-B,G,H POLAR Files (*.hdf)',
    )
